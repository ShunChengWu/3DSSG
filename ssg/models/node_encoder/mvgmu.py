#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:30:16 2021

@author: sc

https://github.com/AntixK/PyTorch-VAE/blob/master/models/twostage_vae.py
"""

import torch
import torch.nn as nn
import logging
import numpy as np
from ssg.models.node_encoder.base import NodeEncoderBase
from ssg.models.node_encoder.gmu import MemoryUnit, Association, TaskProjector, PositionwiseFeedForward
import matplotlib.pyplot as plt
from ssg.models.classifier import  classifider_list
import logging
logger_py = logging.getLogger(__name__)

class MVGMU(NodeEncoderBase):
    def __init__(self,cfg, num_obj_cls, backbone:str, device):
        super().__init__(cfg,backbone,device)
        logger_py.setLevel(cfg.log_level)
        cfg_gmu = cfg.model.gmu
        self.num_obj_cls=num_obj_cls
        self.mean_cls = cfg.model.mean_cls
        self.with_pff = cfg_gmu.with_pff
        self.memory_units = MemoryUnit(cfg_gmu.num_units, cfg_gmu.memory_dim)
        self.projector = TaskProjector(self.node_feature_dim, cfg_gmu.dot_dim, cfg_gmu.memory_dim, cfg_gmu.dot_dim)
        self.associate = Association(cfg_gmu.memory_dim, cfg_gmu.num_heads, self.memory_units, self.projector,norm=cfg_gmu.norm)
        # logger_py.info('hello?')
        if self.with_pff:
            logger_py.info('with pointwise feed forward')
            self.pff = PositionwiseFeedForward(cfg_gmu.memory_dim, cfg_gmu.pff_hidden)
        
        # self.proj = nn.Conv1d(cfg_gmu.memory_dim, self.node_feature_dim, kernel_size=1)
        # self.proj = nn.Sequential(
        #     nn.Linear(self.node_feature_dim, cfg_gmu.memory_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(cfg_gmu.memory_dim, cfg_gmu.memory_dim),
        #     )
        # self.node_feature_dim = cfg_gmu.memory_dim
        if self.mean_cls:
            self.classifier = classifider_list[backbone](in_channels=cfg_gmu.memory_dim, out_channels=num_obj_cls)
        else:
            self.classifier = None
        
        self.cls_anchor = nn.Conv1d(cfg_gmu.memory_dim, cfg_gmu.num_units, kernel_size=1)
        self.cfg_gmu=cfg_gmu

    def forward(self, images,bboxes, **args):
        images = self.preprocess(images)
        n_nodes = len(images) if self.input_is_roi else len(bboxes)
        
        # nodes_feature = torch.zeros([n_nodes, self.node_feature_dim],device=self._device)
        # logvars = torch.zeros([n_nodes, self.node_feature_dim],device=self._device)
        # mus = torch.zeros([n_nodes, self.node_feature_dim],device=self._device)
        
        '''check if all nodes have the same number of images'''
        all_same=True
        shape_0 = bboxes[0].keys() if not self.input_is_roi else images[0].shape[0]
        for node_idx in range(n_nodes):
            shape_i = bboxes[node_idx].keys() if not self.input_is_roi else images[node_idx].shape[0]
            if shape_0 != shape_i: 
                all_same=False
                break
        # all_same=False
        if all_same:
            num_img = shape_0
            shape = images[0].shape
            if not self.input_is_roi:
                roi_features = torch.stack([self.postprocess(images,bboxes[node_idx]) for node_idx in range(n_nodes)])
                roi_features = roi_features.view(n_nodes, num_img, shape[1],shape[2],shape[3]) # seperate node and images
            else:
                batch_imgs = torch.stack(images).view(n_nodes * num_img, shape[1],shape[2],shape[3]) # combine node and images
                roi_features = self.postprocess(batch_imgs,None).view(n_nodes, num_img, -1) # seperate node and images
            mu_feature, prob  = self.associate(roi_features.permute(0,2,1)) #[n_node, feature, num_img]
            
            if self.mean_cls:
                mu_feature = mu_feature.permute(0,2,1)#[n_node, num_img, feature]
                mu_feature = mu_feature.view(n_nodes*num_img, -1)  #[n_node*num_img, -1]
                if self.with_pff: mu_feature = self.pff(mu_feature)
                mu_feature = self.classifier(mu_feature)
                
                mu_feature = torch.nn.functional.softmax(mu_feature,dim=1) # [n_node*n_img, -1]
                mu_feature = mu_feature.view(n_nodes, num_img, -1)# [n_node, n_img, -1]
                mu_feature = mu_feature.mean(1)
                nodes_feature = mu_feature.log() # project to log space to panelize the softmax used in the trainer later
                
                # mu_feature = mu_feature.view(n_nodes, num_img, -1)
                # nodes_feature = mu_feature.mean(1)
                # passs
            else:
                mu_feature = mu_feature.permute(0,2,1)#[n_node, num_img, feature]
                mu_feature = mu_feature.view(n_nodes*num_img, -1)  #[n_node*num_img, -1]
                if self.with_pff:  mu_feature = self.pff(mu_feature) # []
                mu_feature = mu_feature.view(n_nodes, num_img, -1) #[]
                nodes_feature = mu_feature.mean(1) # [num_node, feature_dim]
                # if self.with_pff:
                #      nodes_feature = self.pff(nodes_feature)
        else:
            if self.mean_cls:
                nodes_feature = torch.zeros([n_nodes, self.num_obj_cls],device=self._device)
            else:
                nodes_feature = torch.zeros([n_nodes, self.cfg_gmu.memory_dim],device=self._device)
                
            for node_idx in range(n_nodes):
                if not self.input_is_roi:
                    kf2box = bboxes[node_idx]
                    roi_features = self.postprocess(images,kf2box)
                else:
                    roi_features = self.postprocess(images[node_idx],None)
                    
                roi_features = roi_features.view(roi_features.shape[0], -1)
                mu_feature, prob  = self.associate(roi_features.permute(1,0).unsqueeze(0)) #[1, feature_dim, num_image]
                
                if self.mean_cls:
                    mu_feature = mu_feature.permute(2,1,0).squeeze(-1)#[num_image, feature_dim]
                    if self.with_pff: mu_feature = self.pff(mu_feature)#[n_img, feature]
                    mu_feature = self.classifier(mu_feature)#.mean(0).flatten()
                    mu_feature = torch.nn.functional.softmax(mu_feature,dim=1) # [n_node*n_img, -1]
                    mu_feature = mu_feature.mean(0).log()
                else:
                    mu_feature = mu_feature.permute(2,1,0).squeeze(-1)#[num_image, feature_dim]
                    if self.with_pff:  mu_feature = self.pff(mu_feature)
                    mu_feature = mu_feature.mean(0) # [num_node, feature_dim]
                    mu_feature = mu_feature.flatten()
                    # mu_feature = mu_feature.mean(2).flatten()
                '''draw img'''
                # if True:
                #     cmap = plt.get_cmap('jet')
                #     fig = plt.figure()
                #     plt.imshow(prob[0].detach().cpu().permute(1,0,2).reshape(8,32,1), cmap=cmap) 
                #     plt.colorbar()
                #     x_ticks = np.arange(self.cfg_gmu.num_units*self.cfg_gmu.num_heads)
                #     x_labels = ['h'+str(h)+'m'+str(m) for h in range(self.cfg_gmu.num_heads) for m in range(self.cfg_gmu.num_units)]
                #     plt.xticks(x_ticks, x_labels, rotation=90)
                #     plt.show()
                
                '''VAE'''
                # roi_features = self.proj(roi_features) # [batch,dim,num]
                # logvar = roi_features.mean(0).flatten() # [num, dim] -> [dim]
                # mu = mu_feature.mean(2).flatten() #[batch, dim, num] -> [dim]
                # z = self.reparameterize(mu, logvar)
                # logvars[node_idx] = logvar
                # mus[node_idx] = mu
                # nodes_feature[node_idx] = z.flatten()
                
                nodes_feature[node_idx] = mu_feature
            # if not self.mean_cls and self.with_pff: #TODO: maybe pff should be applied on the associated feature before mean?
            #     nodes_feature = self.pff(nodes_feature)
        
        
        outputs=dict()
        # if self.training:
        '''estimate anchor location '''
        mu_pos = self.cls_anchor(self.memory_units()).squeeze(0).permute(1,0) # from [batch, dim, num] -> [num, dim]
        outputs['mu_pos'] = mu_pos
        # outputs['logvar'] = logvars
        # outputs['mu'] = mus
        # outputs['prob'] = prob
        outputs['nodes_feature']=nodes_feature
        return outputs
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size(),device=self._device)
        z = mu + std * esp
        return z
        
if __name__ == '__main__':
    import codeLib
    config = codeLib.Config('../../../configs/default_mv.yaml')
    # config.path = '/home/sc/research/PersistentSLAM/python/2DTSG/data/test/'
    config.model.node_encoder.method = 'gmu'
    
    
    model = MVGMU(config,backbone='vgg16',device='cpu')
    
    images = torch.rand([3,3,256,256])
    bboxes = [
        {0: torch.FloatTensor([0,0,1,1]), 1: torch.FloatTensor([0,0,0.5,0.5])},
        {1: torch.FloatTensor([0,0,1,1])},
        ]
    output = model(images,bboxes)