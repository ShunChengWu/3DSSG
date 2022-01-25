#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 17:19:08 2021

@author: sc
"""
import torch
from torch import nn
import ssg2d
from .models.classifier import PointNetCls, PointNetRelClsMulti, PointNetRelCls
from codeLib.utils.util import pytorch_count_params
import codeLib
class DestCmp(nn.Module):
    def __init__(self,cfg,num_rel_cls, device, **args):
        super().__init__()
        self.cfg=cfg
        self._device=device
        edge_feature_dim = cfg.model.edge_feature_dim
        
        node_feature_dim= cfg.model.node_feature_dim
        if self.cfg.model.use_spatial:
            node_feature_dim -= cfg.model.edge_descriptor_dim-3 # ignore centroid
        cfg.model.node_feature_dim = node_feature_dim
        
        models = dict()
        models['rel_encoder_sgfn'] = ssg2d.models.edge_encoder_list['sgfn'](cfg,device)
        
        cfg_tmp = codeLib.Config(cfg)
        cfg_tmp.model.edge_descriptor_dim=8
        models['rel_encoder_ssg2d'] = ssg2d.models.edge_encoder_list['basic'](cfg_tmp,device)
        
        
        models['rel_predictor_sgfn'] = PointNetRelClsMulti(
                num_rel_cls, 
                in_size=edge_feature_dim, 
                batch_norm=False,drop_out=True)
        models['rel_predictor_ssg2d'] = PointNetRelClsMulti(
                num_rel_cls, 
                in_size=edge_feature_dim, 
                batch_norm=False,drop_out=True)
        
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
    def forward(self, edges, descriptor):
        ''' Create edge feature '''
        edges_feature_sgfn = self.rel_encoder_sgfn(descriptor,edges)
        
        
        descriptor_8 = torch.zeros([descriptor.shape[0], 8], device=self._device)
        descriptor_8[:,0:3] = descriptor[:,0:3]
        descriptor_8[:,3:] = descriptor[:,6:]
        
        edges_feature_ssg2d = self.rel_encoder_ssg2d(descriptor_8,edges)
        
        edge_cls_sgfn = self.rel_predictor_sgfn(edges_feature_sgfn)
        edge_cls_ssg2d= self.rel_predictor_ssg2d(edges_feature_ssg2d)
            
        return edge_cls_sgfn, edge_cls_ssg2d
    
    def calculate_metrics(self, **args):
        if 'edge_cls_pred_sgfn' in args and 'edge_cls_gt' in args:
            edge_cls_pred = args['edge_cls_pred_sgfn'].detach()
            edge_cls_gt   = args['edge_cls_gt']
            if self.cfg.model.multi_rel:
                edge_cls_pred = torch.sigmoid(edge_cls_pred)
                edge_cls_pred = edge_cls_pred > 0.5
            else:
                edge_cls_pred = torch.softmax(edge_cls_pred, dim=1)
                edge_cls_pred = torch.max(edge_cls_pred,1)[1]
            acc_edgee_cls_sgfn = (edge_cls_gt==edge_cls_pred).sum().item() / edge_cls_gt.nelement()
            
        if 'edge_cls_pred_ssg2d' in args and 'edge_cls_gt' in args:
            edge_cls_pred = args['edge_cls_pred_ssg2d'].detach()
            edge_cls_gt   = args['edge_cls_gt']
            if self.cfg.model.multi_rel:
                edge_cls_pred = torch.sigmoid(edge_cls_pred)
                edge_cls_pred = edge_cls_pred > 0.5
            else:
                edge_cls_pred = torch.softmax(edge_cls_pred, dim=1)
                edge_cls_pred = torch.max(edge_cls_pred,1)[1]
            acc_edgee_cls_ssg2d = (edge_cls_gt==edge_cls_pred).sum().item() / edge_cls_gt.nelement()
        return {
            'acc_edge_cls_sgfn': acc_edgee_cls_sgfn,
            'acc_edge_cls_ssg2d': acc_edgee_cls_ssg2d,
        }