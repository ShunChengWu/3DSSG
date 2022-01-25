#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 18:28:44 2021

@author: sc

https://github.com/weixmath/CVR/

https://openaccess.thecvf.com/content/ICCV2021/papers/Wei_Learning_Canonical_View_Representation_for_3D_Shape_Recognition_With_Arbitrary_ICCV_2021_paper.pdf


"""
import torch
import torch.nn as nn
# from torchvision import models
# from torchvision.ops import roi_align
import logging
from ssg.models.node_encoder.base import NodeEncoderBase
from models.transformer.encoders import EncoderLayer,EncoderLayer_BN
from models.transformer.utils import PositionWiseFeedForward,PositionWiseFeedForward_BN
from models.otk.layers import OTKernel
from models.otk.utils import normalize

logger_py = logging.getLogger(__name__)

def my_pad_sequence(sequences,view_num,N,max_length, padding_value=0):
    C = sequences.size(-1)
    out_tensor = torch.empty(N,max_length,C).fill_(padding_value).cuda()
    count = 0
    for i in range(0,N):
        out_tensor[i,:view_num[i],:] = sequences[count:count+view_num[i],:]
        count = count + view_num[i]
    return out_tensor

class CVR(NodeEncoderBase):
    def __init__(self,cfg,backbone:str,device):
        super().__init__(cfg,backbone,device)
        node_feature_dim = self.node_feature_dim
        self.zdim = 8
        self.encoder_meshed1 = EncoderLayer(d_model=node_feature_dim, d_k=node_feature_dim, d_v=node_feature_dim, h=8, d_ff=2048, dropout=0)
        self.ff1 = PositionWiseFeedForward(d_model=node_feature_dim,d_ff=2048,dropout=0)
        # self.encoder2 = TransformerEncoderLayer_woff(d_model=self.dim, nhead=8)
        self.encoder_meshed2 = EncoderLayer_BN(d_model=node_feature_dim, d_k=node_feature_dim, d_v=node_feature_dim, h=8, d_ff=2048, dropout=0)
        self.ff2 = PositionWiseFeedForward_BN(d_model=node_feature_dim,d_ff=2048,dropout=0)
        self.ff3 = PositionWiseFeedForward_BN(d_model=node_feature_dim, d_ff=2048,dropout=0)
        
        self.otk_layer = OTKernel(in_dim=node_feature_dim, out_size=self.zdim, heads=1, max_iter=100, eps=0.05)
        self.coord_encoder = nn.Sequential(
            nn.Linear(node_feature_dim,64),
            nn.ReLU(),
            nn.Linear(64,3)
        )
        self.coord_decoder = nn.Sequential(
            nn.Linear(3,64),
            nn.ReLU(),
            nn.Linear(64,node_feature_dim)
        )
        
        
        vert = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1],
                             [-1, -1, -1]]
        self.vert = torch.Tensor(vert).to(self._device)
        
    def reset_parameters(self):
        pass
    
    def forward(self, images, bboxes, return_meta=False, **args):
        '''get image features'''
        images = self.preprocess(images)
        
        # if self.input_is_roi:
        n_nodes = len(images) if self.input_is_roi else len(bboxes)
        # if self.with_precompute:
        #     img_features = images # use precomputed image feautre to save time        
        # else:
        #     self.encoder.eval()
        #     with torch.no_grad():
        #         # images=torch.rot90(images,3,[-1,-2])
        #         img_features = torch.cat([ self.encoder(p_split)  for p_split in torch.split(images,int(self.img_batch_size), dim=0) ], dim=0)
        #         # img_features=torch.rot90(img_features,1,[-1,-2])
                
        '''compute node feature base on the given edges'''
        # n_nodes = len(bboxes)
        # width,height = img_features.shape[-1], img_features.shape[-2]        
        # nodes_feature = torch.zeros([n_nodes, self.node_feature_dim],device=self._device)
        
        # if return_meta:
            # cos_sims=torch.zeros([n_nodes, self.zdim,self.zdim],device=self._device)
            # cos_sim2s=torch.zeros([n_nodes, self.zdim,self.zdim],device=self._device)
            # pos0s=torch.zeros([n_nodes, self.zdim,3],device=self._device)
            
        cvf = torch.zeros([n_nodes, self.zdim,self.node_feature_dim],device=self._device)
        for node_idx in range(n_nodes):
            if not self.input_is_roi:
                kf2box = bboxes[node_idx]
                y = self.postprocess(images,kf2box)
            else:
                y = self.postprocess(images[node_idx],None)
                
            
            # kf2box = bboxes[node_idx]
            # y = self.postprocess(images,kf2box)
            # kf_indices = list(kf2box.keys())
            # boxes = [v.view(1,4) for v in kf2box.values()]
            
            ''' scale bounding box with the current feature map size '''
            # for box in boxes:
            #     box[:,0] *= width
            #     box[:,1] *= height
            #     box[:,2] *= width
            #     box[:,3] *= height
            # selected_img_features = img_features[kf_indices]
            
            '''get local view features'''
            # y = roi_align(selected_img_features, boxes, self.roi_region)
            # y = y.view(y.shape[0], -1) # [views, cdim]
            # y = self.fc(y)
            y = y.view(1, y.shape[0], self.node_feature_dim)
            # y = y.unsqueeze(0)# [b, v, c]
            
            '''image-level feature encoder'''
            y0 = self.encoder_meshed1(y, y, y, attention_mask=None)
            y0 = self.ff1(y0)
            '''Canonical View Representation'''
            y1 = self.otk_layer(y0,None) 
            cvf[node_idx] = y1 
        # canonical view features (the figure they drew in their paper is wrong. The MLP in Ensuring Feature Separability should be after Optimal Transport)
        y1 = self.ff2(cvf)
        
        '''Ensuring Feature Seperability'''
        pos0 = normalize(self.coord_encoder(y1))  # Spatial representation
        pos = self.coord_decoder(pos0) # Spatial embedding 
        ''' Canonical View Aggregation '''
        y2 = self.encoder_meshed2(y1+pos, y1+pos, y1, attention_mask=None) 
        y2 = self.ff3(y2) 
        
        ''' view pooling '''
        pooled_view = y2.mean(1)
        
        nodes_feature = pooled_view.view(pooled_view.shape[0],-1)
        # nodes_feature[node_idx] = pooled_view.flatten()
        
        '''extra computation for training'''
        if return_meta:
            weight = self.otk_layer.weight
            cos_sim = torch.matmul(normalize(weight), normalize(weight).transpose(1, 2)) - torch.eye(self.zdim,self.zdim).to(self._device)
            cos_sim2 = torch.matmul(normalize(y1), normalize(y1).transpose(1, 2)) - torch.eye(self.zdim, self.zdim).to(self._device)
        
        outputs=dict()
        outputs['nodes_feature']=nodes_feature
        if return_meta:
            outputs['cos_sim']=cos_sim
            outputs['cos_sim2']=cos_sim2
            outputs['pos']=pos0
        return outputs
    
if __name__ == '__main__':
    import codeLib
    config = codeLib.Config('../../../configs/default_mv.yaml')
    config.path = '/home/sc/research/PersistentSLAM/python/2DTSG/data/test/'
    config.model.local_feature_dim = 64
    
    
    model = CVR(config,backbone='res18',device='cpu')
    
    images = torch.rand([3,3,256,256])
    bboxes = [
        {0: torch.FloatTensor([0,0,1,1]), 1: torch.FloatTensor([0,0,0.5,0.5])},
        {1: torch.FloatTensor([0,0,1,1])},
        ]
    output = model(images,bboxes)