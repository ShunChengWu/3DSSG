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
import logging
logger_py = logging.getLogger(__name__)

class IMP(nn.Module):
    def __init__(self,cfg,num_obj_cls, num_rel_cls, device):
        '''
        Iterative messag passing

        Parameters
        ----------
        cfg : TYPE
            DESCRIPTION.
        num_obj_cls : TYPE
            DESCRIPTION.
        num_rel_cls : TYPE
            DESCRIPTION.
        device : TYPE
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        super().__init__()
        self.cfg=cfg
        self._device=device
        self.dim = 512
        self.update_step = 2
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        '''build models'''
        models = dict()
        
        cfg.data.use_precompute_img_feature = False
        models['roi_extrator'] = ssg.models.node_encoder_list['roi_extractor'](cfg,cfg.model.image_encoder.backend,device)
        
        node_feature_dim = models['roi_extrator'].node_feature_dim
        
        models['obj_embedding'] = nn.Sequential(
            nn.Linear(node_feature_dim, self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim),
        )
        models['pred_embedding'] = nn.Sequential(
            nn.Linear(node_feature_dim, self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim),
        )
        
        models['gnn'] = ssg.models.gnn_list[cfg.model.gnn.method](
            dim_node=cfg.model.gnn.hidden_dim,
            num_layers=cfg.model.gnn.num_layers,
            aggr=cfg.model.gnn.aggr,
            )
        
        models['obj_predictor'] = ssg.models.classifider_list['imp'](cfg.model.gnn.hidden_dim,num_obj_cls)
        models['rel_predictor'] = ssg.models.classifider_list['imp'](cfg.model.gnn.hidden_dim,num_rel_cls)
        
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
        
    def eval(self):
        # nn.Module.eval(self)
        super().train(mode=False)
        self.roi_extrator.with_precompute=True
    def train(self):
        # nn.Module.train(self)
        super().train(mode=True)
        self.roi_extrator.with_precompute=False
        
    def forward(self, images, image_boxes,node_edges, **args):
        '''compute image feature'''
        node_features, edge_features = self.roi_extrator(images, image_boxes,node_edges)
        
        node_features = self.obj_embedding(node_features)
        if len(edge_features)>0:
            edge_features = self.pred_embedding(edge_features)
        
        if hasattr(self, 'gnn') and self.gnn is not None and len(edge_features)>0 and len(node_features)>0:
            if self.cfg.model.gnn.method == 'vgfm':
                node_features,edge_features = self.gnn(node_features,edge_features,node_edges,geo_feature=args['node_descriptor_8'],**args)
            else:
                node_features,edge_features = self.gnn(node_features,edge_features,node_edges,**args)
        
        obj_class_logits = self.obj_predictor(node_features)
        if len(edge_features)>0:
            rel_class_logits = self.rel_predictor(edge_features)
        else:
            rel_class_logits = edge_features
        return obj_class_logits, rel_class_logits
    
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
            if len(edge_cls_pred)>0:
                if self.cfg.model.multi_rel:
                    edge_cls_pred = torch.sigmoid(edge_cls_pred)
                    edge_cls_pred = edge_cls_pred > 0.5
                else:
                    edge_cls_pred = torch.softmax(edge_cls_pred, dim=1)
                    edge_cls_pred = torch.max(edge_cls_pred,1)[1]
                acc_edgee_cls = (edge_cls_gt==edge_cls_pred).sum().item() / edge_cls_gt.nelement()
            else:
                acc_edgee_cls=0
        return {
            'acc_node_cls': acc_node_cls,
            'acc_edgee_cls': acc_edgee_cls,
        }

if __name__ == '__main__':
    import ssg
    import codeLib
    from ssg.data.collate import graph_collate
    import ssg.config as config
    path='./experiments/config_IMP_full_l20_0.yaml'
    
    cfg= codeLib.Config(path)
    cfg.DEVICE='cpu'
    codeLib.utils.util.set_random_seed(cfg.SEED)
    
    