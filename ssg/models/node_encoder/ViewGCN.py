#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:51:44 2021

@author: sc

source: https://github.com/weixmath/view-GCN/blob/master/model/view_gcn.py

Abort! this method doesn't support dynamic number of views. 
In LocalGCN, the final linear layer must have a fix size output.

"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import roi_align
import logging
import torch.nn.functional as Functional

logger_py = logging.getLogger(__name__)

class ViewGCN(nn.Module):
    def __init__(self,cfg,backbone:str,device):
        super().__init__()
        self.img_batch_size = cfg.model.node_encoder.img_batch_size
        self._device = device
        self.backbone = backbone.lower()
        self.local_feature_dim = cfg.model.local_feature_dim # in the original paper they set 128 with ModelNet40, 64 with ModelNet10
        self.node_feature_dim = cfg.model.node_feature_dim
        self.roi_region = cfg.model.node_encoder.roi_region
        
        # set backend
        assert self.backbone in ['vgg16','res18']
        if self.backbone == 'vgg16':
            self.roi_region = [7,7]
            
            vgg16=models.vgg16(pretrained=True)        
            aggr = cfg.model.node_encoder.aggr
            self.global_pooling_method =aggr
            assert self.global_pooling_method in ['max','mean', 'sum']        
            # self.encoder = vgg16.features[:-1] # skip the last maxpool2d
            self.fc = vgg16.classifier[:-1] # skip the last layer
            
            self.with_precompute = cfg.data.use_precompute_img_feature
            if not self.with_precompute:
                # build image encoder
                self.encoder = vgg16.features.eval()
                for param in self.encoder.parameters(): param.requires_grad = False
                
            feature_size=4096
        elif self.backbone=='res18':
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
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
            feature_size=512*self.roi_region[0]*self.roi_region[1]
        else:
            raise RuntimeError('unknown')
            
        self.LocalGCN1 = LocalGCN(k=4,n_views=self.num_views)
        self.NonLocalMP1 = NonLocalMP(n_view=self.num_views)
        self.LocalGCN2 = LocalGCN(k=4, n_views=self.num_views//2)
        self.NonLocalMP2 = NonLocalMP(n_view=self.num_views//2)
        self.LocalGCN3 = LocalGCN(k=4, n_views=self.num_views//4)
        self.View_selector1 = View_selector(n_views=self.num_views, sampled_view=self.num_views//2)
        self.View_selector2 = View_selector(n_views=self.num_views//2, sampled_view=self.num_views//4)
            
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
    
    def forward(self, images, bboxes, poses, **args):
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
            selected_poses = poses[kf_indices]
            
            roi_features = roi_align(selected_img_features, boxes, self.roi_region)
            if self.backbone=='res18':
                roi_features = self.pool(roi_features)
                roi_features = roi_features.view(roi_features.shape[0],-1)
            pass
            
            # v = sign_sqrt(roi_features)
            # v = self.proj(v)
            
            # b = bilinear_pooling(v)
            # u, s, v = torch.svd(b)
            
            # #harmonize singular values
            # harmonized_s = self.harmonize(s)
            # b = torch.mm(torch.mm(u,torch.diag(harmonized_s)), v.t())
            # b = b.view(-1) # vectorized
            # b = sign_sqrt(b) # late sqrt layer
            # b = b / (torch.norm(b,2)+(1e-8)) # l2 norm sub-layer
            
            # # roi_features = 
            # x = self.proj2(b)
            # nodes_feature[node_idx] = x.flatten()
        return nodes_feature
    

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn(nsample, xyz, new_xyz):
    dist = square_distance(xyz, new_xyz)
    id = torch.topk(dist,k=nsample,dim=1,largest=False)[1]
    id = torch.transpose(id, 1, 2)
    return id

class KNN_dist(nn.Module):
    def __init__(self,k):
        super(KNN_dist, self).__init__()
        self.R = nn.Sequential(
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,10),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(10,1),
        )
        self.k=k
    def forward(self,F,vertices):
        id = knn(self.k, vertices, vertices)
        F = index_points(F,id)
        v = index_points(vertices,id)
        v_0 = v[:,:,0,:].unsqueeze(-2).repeat(1,1,self.k,1)
        v_F = torch.cat((v_0, v, v_0-v,torch.norm(v_0-v,dim=-1,p=2).unsqueeze(-1)),-1)
        v_F = self.R(v_F)
        F = torch.mul(v_F, F)
        F = torch.sum(F,-2)
        return F

class View_selector(nn.Module):
    def __init__(self, n_views, sampled_view):
        super(View_selector, self).__init__()
        self.n_views = n_views
        self.s_views = sampled_view
        self.cls = nn.Sequential(
            nn.Linear(512*self.s_views, 256*self.s_views),
            nn.LeakyReLU(0.2),
            nn.Linear(256*self.s_views, 40*self.s_views))
    def forward(self,F,vertices,k):
        id = farthest_point_sample(vertices,self.s_views)
        vertices1 = index_points(vertices,id)
        id_knn = knn(k,vertices,vertices1)
        F = index_points(F,id_knn)
        vertices = index_points(vertices,id_knn)
        F1 = F.transpose(1,2).reshape(F.shape[0],k,self.s_views*F.shape[-1])
        F_score = self.cls(F1).reshape(F.shape[0],k,self.s_views,40).transpose(1,2)
        F1_ = Functional.softmax(F_score,-3)
        F1_ = torch.max(F1_,-1)[0]
        F1_id = torch.argmax(F1_,-1)
        F1_id = Functional.one_hot(F1_id,4).float()
        F1_id_v = F1_id.unsqueeze(-1).repeat(1,1,1,3)
        F1_id_F = F1_id.unsqueeze(-1).repeat(1, 1, 1, 512)
        F_new = torch.mul(F1_id_F,F).sum(-2)
        vertices_new = torch.mul(F1_id_v,vertices).sum(-2)
        return F_new,F_score,vertices_new

class LocalGCN(nn.Module):
    def __init__(self,k,n_views):
        super(LocalGCN,self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.k = k
        self.n_views = n_views
        self.KNN = KNN_dist(k=self.k)
    def forward(self,F,V):
        F = self.KNN(F, V)
        F = F.view(-1, 512)
        F = self.conv(F)
        F = F.view(-1, self.n_views, 512)
        return F

class NonLocalMP(nn.Module):
    def __init__(self,n_view):
        super(NonLocalMP,self).__init__()
        self.n_view=n_view
        self.Relation = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.Fusion = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, F):
        F_i = torch.unsqueeze(F, 2)
        F_j = torch.unsqueeze(F, 1)
        F_i = F_i.repeat(1, 1, self.n_view, 1)
        F_j = F_j.repeat(1, self.n_view, 1, 1)
        M = torch.cat((F_i, F_j), 3)
        M = self.Relation(M)
        M = torch.sum(M,-2)
        F = torch.cat((F, M), 2)
        F = F.view(-1, 512 * 2)
        F = self.Fusion(F)
        F = F.view(-1, self.n_view, 512)
        return F
    
if __name__ == '__main__':
    import codeLib
    config = codeLib.Config('../../../configs/default_mv.yaml')
    config.path = '/home/sc/research/PersistentSLAM/python/2DTSG/data/test/'
    config.model.local_feature_dim = 64
    
    
    model = ViewGCN(config,backbone='res18',device='cpu')
    
    images = torch.rand([3,3,256,256])
    bboxes = [
        {0: torch.FloatTensor([0,0,1,1]), 1: torch.FloatTensor([0,0,0.5,0.5])},
        {1: torch.FloatTensor([0,0,1,1])},
        ]
    output = model(images,bboxes)