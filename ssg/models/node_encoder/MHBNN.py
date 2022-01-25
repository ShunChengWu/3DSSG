#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 18:28:44 2021

@author: sc

https://github.com/lliyuan/MHBNN-PyTorch/blob/master/mhbn.py

"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import roi_align
import math
import logging
logger_py = logging.getLogger(__name__)

def sign_sqrt(x):
    x = torch.mul(torch.sign(x),torch.sqrt(torch.abs(x)+1e-12))
    return x


def bilinear_pooling(x):
    '''
    x = [x1, x2, ..., xn], size d*N, d is number of features of each patch, N is number of patches of all views 
    return: d*d
    '''
    return torch.mm(x, x.t())

class MHBNN(nn.Module):
    def __init__(self,cfg,backbone:str,device):
        super().__init__()
        self.img_batch_size = cfg.model.node_encoder.img_batch_size
        self._device = device
        self.backbone = backbone.lower()
        self.local_feature_dim = cfg.model.node_encoder.local_feature_dim # in the original paper they set 128 with ModelNet40, 64 with ModelNet10
        self.node_feature_dim = cfg.model.node_feature_dim
        self.roi_region = cfg.model.node_encoder.roi_region
        
        # trainable param
        self.lambdas = nn.Parameter(torch.ones(self.node_feature_dim)*0.5, requires_grad=True)
        
        
        # set backend
        assert self.backbone in ['vgg16','res18']
        if self.backbone == 'vgg16':
            self.roi_region = [7,7]
            
            vgg16=models.vgg16(pretrained=True)        
            aggr = cfg.model.node_encoder.aggr
            self.global_pooling_method =aggr
            assert self.global_pooling_method in ['max','mean', 'sum']        
            # self.encoder = vgg16.features[:-1] # skip the last maxpool2d
            # self.fc = vgg16.classifier[:-1] # skip the last layer
            
            self.with_precompute = cfg.data.use_precompute_img_feature
            if not self.with_precompute:
                # build image encoder
                self.encoder = vgg16.features.eval()
                for param in self.encoder.parameters(): param.requires_grad = False
                
            feature_size=512
        elif self.backbone=='res18':
            # self.fc = nn.Sequential()
            self.with_precompute = cfg.data.use_precompute_img_feature
            if not self.with_precompute:
                resnet18=models.resnet18(pretrained=True)
                self.encoder = nn.Sequential(
                    resnet18.conv1,
                    resnet18.bn1,
                    resnet18.relu,
                    resnet18.maxpool,
                    resnet18.layer1,
                    resnet18.layer2,
                    resnet18.layer3,
                    resnet18.layer4
                )
                # build image encoder
                self.encoder.eval()
                for param in self.encoder.parameters(): param.requires_grad = False
            feature_size=512
        else:
            raise RuntimeError('unknown')
            
        self.proj = nn.Conv2d(feature_size, self.local_feature_dim , kernel_size=1)
        # self.proj = nn.Linear(feature_size, self.local_feature_dim ) 
        self.proj2 = nn.Linear(self.local_feature_dim*self.local_feature_dim, self.node_feature_dim ) 
            
    def reset_parameters(self):
        pass
    
    def harmonize(self, s):
        '''
        s: (d, ) tensor, sigular values
        return: (d, ) tensor after harmonized by box-cox transform
        '''
        harmonized_s = torch.zeros_like(s)
        n = s.size(0)
        for i in range(n):
            if torch.abs(self.lambdas[i]) > 1e-12:
                harmonized_s[i] = (s[i].pow(self.lambdas[i]) - 1) / self.lambdas[i]
            else:
                harmonized_s[i] = torch.log(s[i])
        return harmonized_s
    
    def forward(self, images, bboxes, **args):
        '''get image features'''
        if self.with_precompute:
            img_features = images # use precomputed image feautre to save time        
        else:
            self.encoder.eval()
            with torch.no_grad():
                # images=torch.rot90(images,3,[-1,-2])
                img_features = torch.cat([ self.encoder(p_split)  for p_split in torch.split(images,int(self.img_batch_size), dim=0) ], dim=0)
                # img_features=torch.rot90(img_features,1,[-1,-2])
                
        '''compute node feature base on the given edges'''
        n_nodes = len(bboxes)
        width,height = img_features.shape[-1], img_features.shape[-2]        
        nodes_feature = torch.zeros([n_nodes, self.node_feature_dim],device=self._device)
        
        for node_idx in range(n_nodes):
            kf2box = bboxes[node_idx]
            kf_indices = list(kf2box.keys())
            boxes = [v.view(1,4) for v in kf2box.values()]
            ''' scale bounding box with the current feature map size '''
            for box in boxes:
                box[:,0] *= width
                box[:,1] *= height
                box[:,2] *= width
                box[:,3] *= height
            selected_img_features = img_features[kf_indices]
            
            # [view, dim, h ,w]
            roi_features = roi_align(selected_img_features, boxes, self.roi_region)

            '''per view'''
            view_pool = []
            # [view, batch, dim, h, w]
            roi_features = roi_features.unsqueeze(1)
            for v in roi_features:
                v = sign_sqrt(v)
                v = self.proj(v)
                view_pool.append(v)
            # [view, dim, h, w]
            y = torch.stack(view_pool).squeeze(1)
            
            '''per batch (batch is always 1)'''
            # [feature_dim, n_views, h, w]
            b = y.transpose(0,1).contiguous()
            # [feature_dim, -1]
            b = b.view(b.shape[0],-1)
            b = bilinear_pooling(b)
            u, s, v = torch.svd(b)
            
            #harmonize singular values
            harmonized_s = self.harmonize(s)
            b = torch.mm(torch.mm(u,torch.diag(harmonized_s)), v.t())
            b = b.view(-1) # vectorized
            b = sign_sqrt(b) # late sqrt layer
            b = b / (torch.norm(b,2)+(1e-8)) # l2 norm sub-layer
            
            # roi_features = 
            x = self.proj2(b)
            nodes_feature[node_idx] = x.flatten()
        outputs=dict()
        outputs['nodes_feature']=nodes_feature
        return outputs
    
    
if __name__ == '__main__':
    import codeLib
    config = codeLib.Config('../../../configs/default_mv.yaml')
    config.path = '/home/sc/research/PersistentSLAM/python/2DTSG/data/test/'
    config.model.local_feature_dim = 64
    
    
    model = MHBNN(config,backbone='res18',device='cpu')
    
    images = torch.rand([3,3,256,256])
    bboxes = [
        {0: torch.FloatTensor([0,0,1,1]), 1: torch.FloatTensor([0,0,0.5,0.5])},
        {1: torch.FloatTensor([0,0,1,1])},
        ]
    output = model(images,bboxes)