#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 08:59:51 2021

@author: sc
"""

import torch
import torch.nn as nn
from ssg.models.node_encoder.base import NodeEncoderBase
from ssg.models.classifier import  classifider_list
# from torchvision import models
# from torchvision.ops import roi_align
import logging
logger_py = logging.getLogger(__name__)

class MeanMV(NodeEncoderBase):
    def __init__(self,cfg,num_obj_cls,backbone:str,device):
        super().__init__(cfg,backbone,device)
        self.global_pooling_method = cfg.model.node_encoder.aggr
        node_feature_dim = self.node_feature_dim
        self.num_obj_cls = num_obj_cls
        self.classifier = classifider_list[backbone](in_channels=node_feature_dim, out_channels=num_obj_cls)
        
    def reset_parameters(self):
        pass
    def forward(self, images, bboxes, **args):
        '''get image features'''
        images = self.preprocess(images)
        
        '''compute node feature base on the given edges'''
        n_nodes = len(images) if self.input_is_roi else len(bboxes)
        nodes_feature = torch.zeros([n_nodes, self.num_obj_cls],device=self._device)
        for node_idx in range(n_nodes):
            if not self.input_is_roi:
                kf2box = bboxes[node_idx]
                roi_features = self.postprocess(images,kf2box)
            else:
                roi_features = self.postprocess(images[node_idx],None)
                
            y = self.classifier(roi_features.view(roi_features.shape[0],-1))
            # mean the softmax output
            y = torch.nn.functional.softmax(y,dim=1).mean(0)
            # project to log space to panelize the softmax used in the trainer later
            y = y.log()
            nodes_feature[node_idx] = y
        outputs=dict()
        outputs['nodes_feature']=nodes_feature
        return outputs
 
if __name__ == '__main__':
    import codeLib
    config = codeLib.Config('../../../configs/default_mv.yaml')
    config.path = '/home/sc/research/PersistentSLAM/python/2DTSG/data/test/'
    config.data.use_precompute_img_feature=False
    
    
    model = MeanMV(config,num_obj_cls=40, backbone='vgg16',device='cpu')
    if config.data.use_precompute_img_feature:
        images = torch.rand([3,512,32,32])
    else:
        images = torch.rand([3,3,512,512])
    bboxes = [
        {0: torch.FloatTensor([0,0,1,1]), 1: torch.FloatTensor([0,0,0.5,0.5])},
        {1: torch.FloatTensor([0,0,1,1])},
        ]
    output = model(images,bboxes)