#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 17:19:08 2021

@author: sc
"""
import torch
from torch import nn
import ssg
from .models.classifier import PointNetCls, PointNetRelClsMulti, PointNetRelCls
from codeLib.utils.util import pytorch_count_params

class SGFN(nn.Module):
    def __init__(self,cfg,num_obj_cls, num_rel_cls, device):
        super().__init__()
        self.cfg=cfg
        self._device=device
        node_feature_dim = cfg.model.node_feature_dim
        edge_feature_dim = cfg.model.edge_feature_dim
        
        if self.cfg.model.use_spatial:
            node_feature_dim -= cfg.model.edge_descriptor_dim-3 # ignore centroid
        cfg.model.node_feature_dim = node_feature_dim
        
        models = dict()
        models['obj_encoder'] = ssg.models.node_encoder_list['sgfn'](cfg,device)
        models['rel_encoder'] = ssg.models.edge_encoder_list['sgfn'](cfg,device)
        
        if self.cfg.model.use_spatial:
            node_feature_dim += cfg.model.edge_descriptor_dim-3 # ignore centroid
        cfg.model.node_feature_dim = node_feature_dim
        
        if cfg.model.gnn.method != 'none': 
            models['gnn'] = ssg.models.gnn_list[cfg.model.gnn.method](
                dim_node=cfg.model.node_feature_dim,
                dim_edge=cfg.model.edge_feature_dim,
                dim_atten=cfg.model.gnn.hidden_dim,
                num_layers=cfg.model.gnn.num_layers,
                num_heads=cfg.model.gnn.num_heads,
                aggr='max',
                DROP_OUT_ATTEN=cfg.model.gnn.drop_out
                )
        
        with_bn =cfg.model.node_classifier.with_bn
        models['obj_predictor'] = PointNetCls(num_obj_cls, in_size=node_feature_dim,
                                 batch_norm=with_bn,drop_out=cfg.model.node_classifier.dropout)
        
        if cfg.model.multi_rel:
            models['rel_predictor'] = PointNetRelClsMulti(
                num_rel_cls, 
                in_size=edge_feature_dim, 
                batch_norm=with_bn,drop_out=True)
        else:
            models['rel_predictor'] = PointNetRelCls(
                num_rel_cls, 
                in_size=edge_feature_dim, 
                batch_norm=with_bn,drop_out=True)
            
        params = list()
        print('==trainable parameters==')
        for name, model in models.items():
            if model is None: 
                self.name = model
                continue
            # if len(cfg.GPU) > 1:
            #     model = torch.nn.DataParallel(model, config.GPU)
            model = model.to(device)
            self.add_module(name, model)
            params += list(model.parameters())
            print(name,pytorch_count_params(model))
        print('')
    def forward(self, obj_points, descriptor, node_edges, **args):
        nodes_feature = self.obj_encoder(obj_points)
        
        if self.cfg.model.use_spatial:
            tmp = descriptor[:,3:].clone()
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            nodes_feature = torch.cat([nodes_feature, tmp],dim=1)

        ''' Create edge feature '''
        # with torch.no_grad():
        #     edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor,node_edges)
        edges_feature = self.rel_encoder(descriptor,node_edges)
        
                    
        ''' GNN '''
        probs=None
        if hasattr(self, 'gnn') and self.gnn is not None:
            gnn_nodes_feature, gnn_edges_feature, probs = self.gnn(nodes_feature, edges_feature, node_edges)
            nodes_feature = gnn_nodes_feature
            edges_feature = gnn_edges_feature
        
        '''1. Node '''
        node_cls = self.obj_predictor(nodes_feature)
        '''2.Edge'''
        # edge_cls=None
        edge_cls = self.rel_predictor(edges_feature)
            
        return node_cls, edge_cls
    
    def calculate_metrics(self, **args):
        if 'node_cls_pred' in args and 'node_cls_gt' in args:
            node_cls_pred = args['node_cls_pred'].detach()
            node_cls_pred = torch.softmax(node_cls_pred, dim=1)
            node_cls_gt   = args['node_cls_gt']
            node_cls_pred = torch.max(node_cls_pred,1)[1]
            acc_node_cls = (node_cls_gt == node_cls_pred).sum().item() / node_cls_gt.nelement()
        
        if 'edge_cls_pred' in args and 'edge_cls_gt' in args:
            edge_cls_pred = args['edge_cls_pred'].detach()
            edge_cls_gt   = args['edge_cls_gt']
            if self.cfg.model.multi_rel:
                edge_cls_pred = torch.sigmoid(edge_cls_pred)
                edge_cls_pred = edge_cls_pred > 0.5
            else:
                edge_cls_pred = torch.softmax(edge_cls_pred, dim=1)
                edge_cls_pred = torch.max(edge_cls_pred,1)[1]
            acc_edgee_cls = (edge_cls_gt==edge_cls_pred).sum().item() / edge_cls_gt.nelement()
        return {
            'acc_node_cls': acc_node_cls,
            'acc_edgee_cls': acc_edgee_cls,
        }
