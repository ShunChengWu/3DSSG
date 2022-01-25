#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 11:14:28 2021

@author: sc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

class TaskProjector(nn.Module):
    '''
    A learnable projector to project.
    Takes input key, query, and value and return the same.
    The idea here is to share memory units across tasks by projecting them with 
    a task specific project.
    '''
    def __init__(self, in_dim_query:int, out_dim_query:int, in_dim_key:int, out_dim_key:int):
        super().__init__()
        self.proj_q = torch.nn.Conv1d(in_dim_query, out_dim_query, kernel_size=1)
        self.proj_k = torch.nn.Conv1d(in_dim_key, out_dim_key, kernel_size=1)
        self.proj_v = torch.nn.Identity()
        self.reset_parameter()
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.proj_q.weight)
        nn.init.xavier_uniform_(self.proj_k.weight)
        nn.init.constant_(self.proj_q.bias, 0)
        nn.init.constant_(self.proj_k.bias, 0)
    def forward(self, query, key, value):
        return self.proj_q(query), self.proj_k(key), self.proj_v(value)
    
class MemoryUnit(nn.Module):
    def __init__(self,n_mu:int, dim:int):
        super().__init__()
        self.mus = nn.Parameter( torch.Tensor(1, dim, n_mu) )
        self.reset_parameter()
    def reset_parameter(self):
        torch.nn.init.xavier_uniform_(self.mus)
    def forward(self):
        '''
        return [1, dim, n_mu]
        '''
        return self.mus
    
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class Association(nn.Module):
    '''
    Find the associated memoery units.
    Given an input feature, this function return a feature by extracting information 
    from the memory units base on the dot simularity between the input feature 
    and all memory units.
    '''
    def __init__(self, output_channels:int, num_heads:int, memories:MemoryUnit, projector:TaskProjector,
                 attn_dropout:float= 0.5, norm:bool=True):
        super().__init__()
        self.num_heads = num_heads
        self.mus = memories
        self.projector = projector
        self.dropout = torch.nn.Dropout(attn_dropout)
        self.layer_norm = torch.nn.LayerNorm(output_channels, eps=1e-6) if norm else None
    def forward(self, x):
        '''
        Parameters
        ----------
        x : torch.Tensor
            Dim: [batch, dim, n_query]

        Returns
        -------
        x : TYPE
            DESCRIPTION.
        prob : TYPE
            DESCRIPTION.

        '''
        batch_dim = x.shape[0]
        
        (q,k,v) = self.projector(x, self.mus(), self.mus())
        
        q = q.view(batch_dim, int(q.shape[1]/self.num_heads), self.num_heads, -1) #Batch, dim, head, n_q
        k = k.view(1, int(k.shape[1]/self.num_heads), self.num_heads, -1)
        v = v.view(1, int(v.shape[1]/self.num_heads), self.num_heads, -1) 
        
        # (q, k, v) = [d.view(d.shape[0], int(d.shape[1]/self.num_heads), self.num_heads, -1) for d in data]
        d_q = q.shape[1]
        d_v = v.shape[1]
        scores = torch.einsum('bdhn,bkhm->bhnm', q, k) / d_q**.5 # [batch, n_head, n_query, n_key ]
        prob = torch.nn.functional.softmax(scores, dim=-1)
        prob = self.dropout(prob)
        x = torch.einsum('bhnm,bdhm->bdhn', prob, v).contiguous().view(batch_dim, d_v*self.num_heads, -1) 
        
        if self.layer_norm is not None:
            # Always normalize a feature. usually this gives better performance (or learning curve).
            x = self.layer_norm(x.permute(0,2,1)) # [batch, n_query, features]
            x = x.permute(0,2,1) # [batch, features, n_query]
        return x,prob
    
    
    
# def to_var(x):
#     if torch.cuda.is_available():
#         x = x.cuda()
#     return Variable(x)
# class VAE(nn.Module):
#     def __init__(self, image_size=784, h_dim=400, z_dim=20):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(image_size, h_dim),
#             nn.LeakyReLU(0.2),
#             nn.Linear(h_dim, z_dim*2)
#         )
        
#         self.decoder = nn.Sequential(
#             nn.Linear(z_dim, h_dim),
#             nn.ReLU(),
#             nn.Linear(h_dim, image_size),
#             nn.Sigmoid()
#         )
    
#     def reparameterize(self, mu, logvar):
#         std = logvar.mul(0.5).exp_()
#         esp = to_var(torch.randn(*mu.size()))
#         z = mu + std * esp
#         return z
    
#     def forward(self, x):
#         h = self.encoder(x)
#         mu, logvar = torch.chunk(h, 2, dim=1)
#         z = self.reparameterize(mu, logvar)
#         return self.decoder(z), mu, logvar
    
    
    
if __name__ == '__main__':
    batch = 4
    in_q = 256
    out_q = 64
    in_k = 256
    out_k = 64
    out_v = in_k
    n_query = 4
    n_mus = 8
    n_heads = 4
    
    mus = MemoryUnit(n_mus, in_k)
    projector = TaskProjector(in_q,out_q,in_k,out_k)
    association = Association(out_v, n_heads,mus,projector)
    
    x = torch.rand([batch,in_q,n_query])
    
    association(x)
    
    pass