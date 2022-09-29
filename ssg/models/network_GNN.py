#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Some codes here are modified from SuperGluePretrainedNetwork https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/superglue.py
#
import math
import torch
import torch_geometric
from .network_util import build_mlp, Gen_Index, Aggre_Index, MLP
from .networks_base import BaseNetwork
import inspect
from collections import OrderedDict
import os
from codeLib.utils import onnx
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
import torch.nn as nn
from typing import Optional
from copy import deepcopy
from torch_scatter import scatter
from codeLib.common import filter_args_create, reset_parameters_with_activation
import ssg#.models import edge_encoder_list

class TripletEdgeNet(torch.nn.Module):
    def __init__(self,dim_node,dim_edge,use_bn=False):
        super().__init__()
        self.name = 'TripletEdgeNet'
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.nn = build_mlp([dim_node*2+dim_edge,2*(dim_node+dim_edge),dim_edge],
                          do_bn= use_bn, on_last=False)
    def forward(self, x_i, edge_feature,x_j):
        x_ = torch.cat([x_i,edge_feature,x_j],dim=1)#.view(b, -1, 1)
        return self.nn(x_)
    def trace(self, pth = './tmp',name_prefix=''):
        params = inspect.signature(self.forward).parameters
        params = OrderedDict(params)
        names_i = [name for name in params.keys()]
        names_o = ['y']
        x_1 = torch.rand(1, self.dim_node)
        e = torch.rand(1, self.dim_edge)
        x_2 = torch.rand(1, self.dim_node)
        self(x_1,e,x_2)
        name = name_prefix+'_'+self.name
        onnx.export(self, (x_1,e,x_2), os.path.join(pth, name), 
                        input_names=names_i, output_names=names_o, 
                        dynamic_axes = {names_i[0]:{0:'n_edge'},
                                        names_i[1]:{0:'n_edge'},
                                        names_i[2]:{0:'n_edge'}})
        names = dict()
        names['model_'+name] = dict()
        names['model_'+name]['path'] = name
        names['model_'+name]['input']=names_i
        names['model_'+name]['output']=names_o
        return names

def scaled_dot_product_attention(query, key, value, weight = None):
    dim = query.shape[1]
    if weight is not None:
        scores = torch.einsum('bdhn,bdd,bdhm->bhnm', query, weight,key) / dim**.5
    else:
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=0)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedEdgeAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_node: int, dim_edge: int, dim_atten: int, use_bn=False,
                 attention = 'fat', use_edge:bool = True, attn_dropout:float = 0.5, **kwargs):
        super().__init__()
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.name = 'MultiHeadedEdgeAttention'
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.d_n = d_n = dim_node // num_heads
        self.d_e = d_e = dim_edge // num_heads
        self.d_o = d_o = dim_atten // num_heads
        self.num_heads = num_heads
        self.use_edge = use_edge
        self.nn_edge = build_mlp([dim_node*2+dim_edge,(dim_node+dim_edge),dim_edge],
                          do_bn= use_bn, on_last=False)
        
        self.dropout = None
        DROP_OUT_ATTEN = None
        if 'DROP_OUT_ATTEN' in kwargs:
            DROP_OUT_ATTEN = kwargs['DROP_OUT_ATTEN']
            self.dropout = torch.nn.Dropout(attn_dropout)
            # print('drop out in',self.name,'with value',DROP_OUT_ATTEN)
        
        self.attention = attention
        assert self.attention in ['scaled_dot_product_attention', 'fat']
        
        if self.attention == 'fat':
            DROP_OUT_ATTEN = None # don't do dropout on MLP
            if use_edge:
                self.nn = MLP([d_n+d_e, d_n+d_e, d_o],do_bn=use_bn,drop_out = DROP_OUT_ATTEN)
            else:
                self.nn = MLP([d_n, d_n*2, d_o],do_bn=use_bn,drop_out = DROP_OUT_ATTEN)
                
            self.proj_edge  = build_mlp([dim_edge,dim_edge])
            self.proj_query = build_mlp([dim_node,dim_node])
            self.proj_value = build_mlp([dim_node,dim_atten])
        
        elif self.attention == 'scaled_dot_product_attention':
            self.merge = build_mlp([dim_node, dim_node])
            self.proj  = torch.nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
            # self.merge = torch.nn.Conv1d(dim_node, dim_node, kernel_size=1)
            self.merge = MLP(dim_node, dim_node, kernel_size=1)
        
    def forward(self, query, edge, value):
        batch_dim = query.size(0)
        edge_feature = self.nn_edge( torch.cat([query,edge,value],dim=1) )#.view(b, -1, 1)
        
        
        if self.attention == 'fat':
            value = self.proj_value(value)
            query = self.proj_query(query).view(batch_dim, self.d_n, self.num_heads)
            edge = self.proj_edge(edge).view(batch_dim, self.d_e, self.num_heads)
            if self.use_edge:
                prob = self.nn(torch.cat([query,edge],dim=1)) # b, dim, head    
            else:
                prob = self.nn(query) # b, dim, head    
            prob = prob.softmax(1)
            if self.dropout is not None:
                prob = self.dropout(prob)
            x = torch.einsum('bm,bm->bm', prob.reshape_as(value), value)
            
        elif self.attention == 'scaled_dot_product_attention':
            query, key, value = [l(x).view(batch_dim, self.d_n, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, value, value))]
            x, prob = scaled_dot_product_attention(query, value, value)
            x = self.merge(x.contiguous().view(batch_dim, self.d_n*self.num_heads, -1)).view(batch_dim, -1)

        return x, edge_feature, prob
    def trace(self, pth = './tmp',name_prefix=''):
        params = inspect.signature(self.forward).parameters
        params = OrderedDict(params)
        names_i = [name for name in params.keys()]
        names_o = ['w_'+names_i[0], 'prob']
        x1 = torch.rand(1, self.dim_node)
        e = torch.rand(1, self.dim_edge)
        x2 = torch.rand(1, self.dim_node)
        self(x1,e,x2)
        name = name_prefix+'_'+self.name
        onnx.export(self, (x1,e,x2), os.path.join(pth, name), 
                        input_names=names_i, output_names=names_o, 
                        dynamic_axes = {names_i[0]:{0:'n_edge'},
                                        names_i[1]:{0:'n_edge'},
                                        names_i[2]:{0:'n_edge'}})
        
        names = dict()
        names['model_'+name] = dict()
        names['model_'+name]['path'] = name
        names['model_'+name]['input']=names_i
        names['model_'+name]['output']=names_o
        return names
     
class GraphEdgeAttenNetwork(BaseNetwork):
    def __init__(self, num_heads, dim_node, dim_edge, dim_atten, aggr= 'max', use_bn=False,
                 flow='target_to_source',attention = 'fat',use_edge:bool=True, **kwargs):
        super().__init__() #  "Max" aggregation.
        self.name = 'edgeatten'
        self.dim_node=dim_node
        self.dim_edge=dim_edge
        self.index_get = Gen_Index(flow=flow)
        self.index_aggr = Aggre_Index(aggr=aggr,flow=flow)
        
        self.attention = attention
        assert self.attention in ['scaled_dot_product_attention', 'fat']
        if self.attention == 'fat' or self.attention == 'scaled_dot_product_attention':
            self.edgeatten = MultiHeadedEdgeAttention(
                dim_node=dim_node,dim_edge=dim_edge,dim_atten=dim_atten,
                num_heads=num_heads,use_bn=use_bn,attention=attention,use_edge=use_edge, **kwargs)
        if self.attention == 'fat':
            self.prop = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                             do_bn= use_bn, on_last=False)
        elif self.attention == 'scaled_dot_product_attention':
            self.prop = build_mlp([dim_node+dim_node, dim_node+dim_node, dim_node],
                             do_bn= use_bn, on_last=False)
            torch.nn.init.constant_(self.prop[-1].bias, 0.0)
        else:
            raise NotImplementedError('')

    def forward(self, x, edge_feature, edge_index):
        assert x.ndim == 2
        assert edge_feature.ndim == 2
        x_i, x_j = self.index_get(x, edge_index)
        xx, gcn_edge_feature, prob = self.edgeatten(x_i,edge_feature,x_j)
        xx = self.index_aggr(xx,edge_index, dim_size = x.shape[0])
        xx = self.prop(torch.cat([x,xx],dim=1))
        return xx, gcn_edge_feature,prob
    
    def trace(self, pth = './tmp', name_prefix=''):
        n_node=2
        n_edge=4
        x = torch.rand(n_node, self.dim_node)
        edge_feature = torch.rand(n_edge, self.dim_edge)
        edge_index = torch.randint(0, n_node-1, [2,n_edge])
        edge_index[0] = torch.zeros([n_edge])
        edge_index[1] = torch.ones([n_edge])
        
        self.eval()
        self(x,edge_feature,edge_index)
        
        x_i, x_j = self.index_get(x, edge_index)
        xx, edge_feature, prob = self.edgeatten(x_i,edge_feature,x_j)
        xx = self.index_aggr(xx, edge_index, dim_size = x.shape[0])
        # y = self.prop(torch.cat([x,xx],dim=1))
        
        names_i = ['x_in']
        names_o = ['x_out']
        name_nn = name_prefix+'_'+self.name+'_prop'
        cated=torch.cat([x, xx], dim=1)
        onnx.export(self.prop, (cated), os.path.join(pth, name_nn), 
                        input_names=names_i, output_names=names_o, 
                        dynamic_axes = {names_i[0]:{0:'n_node'}})
        names_nn = dict()
        names_nn['model_'+name_nn] = dict()
        names_nn['model_'+name_nn]['path'] = name_nn
        names_nn['model_'+name_nn]['input']=names_i
        names_nn['model_'+name_nn]['output']=names_o
        
        name = name_prefix+'_'+self.name
        names_atten = self.edgeatten.trace(pth, name)
        
        names = dict()
        names[name] = dict()
        names[name]['atten'] = names_atten
        names[name]['prop'] = names_nn
        return names
    
class TripletGCN(MessagePassing):
    def __init__(self, dim_node, dim_edge, dim_hidden, aggr= 'mean', with_bn=True):
        super().__init__(aggr=aggr)
        # print('============================')
        # print('aggr:',aggr)
        # print('============================')
        self.dim_node = dim_node
        self.dim_edge = dim_edge
        self.dim_hidden = dim_hidden
        self.nn1 = build_mlp([dim_node*2+dim_edge, dim_hidden, dim_hidden*2+dim_edge],
                      do_bn= with_bn, on_last=True)
        self.nn2 = build_mlp([dim_hidden,dim_hidden,dim_node],do_bn= with_bn)
        
        self.reset_parameter()
        
    def reset_parameter(self):
        pass
        # reset_parameters_with_activation(self.nn1[0], 'relu')
        # reset_parameters_with_activation(self.nn1[3], 'relu')
        # reset_parameters_with_activation(self.nn2[0], 'relu')
        
    def forward(self, x, edge_feature, edge_index):
        gcn_x, gcn_e = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        gcn_x = x + self.nn2(gcn_x)
        return gcn_x, gcn_e

    def message(self, x_i, x_j,edge_feature):
        x = torch.cat([x_i,edge_feature,x_j],dim=1)
        x = self.nn1(x)#.view(b,-1)
        new_x_i = x[:,:self.dim_hidden]
        new_e   = x[:,self.dim_hidden:(self.dim_hidden+self.dim_edge)]
        new_x_j = x[:,(self.dim_hidden+self.dim_edge):]
        x = new_x_i+new_x_j
        return [x, new_e]
    
    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x
    
class TripletGCNModel(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """
    def __init__(self, num_layers, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.gconvs = torch.nn.ModuleList()
        
        for _ in range(self.num_layers):
            self.gconvs.append(  TripletGCN(**kwargs))

    def forward(self, node_feature, edge_feature, edges_indices):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature = gconv(node_feature, edge_feature, edges_indices)
            
            if i < (self.num_layers-1):
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)
        return node_feature, edge_feature

class MessagePassing_IMP(MessagePassing):
    def __init__(self, dim_node, aggr= 'mean', **kwargs):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        # Attention layer
        self.subj_node_gate = nn.Sequential(nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_node_gate = nn.Sequential(nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

        self.subj_edge_gate = nn.Sequential(nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_edge_gate = nn.Sequential(nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        
        
    def forward(self,x,edge_feature,edge_index):
        node_msg, edge_msg = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        return node_msg, edge_msg

    def message(self, x_i, x_j,edge_feature):
        '''Node'''
        message_pred_to_subj = self.subj_node_gate(torch.cat([x_i,edge_feature],dim=1)) * edge_feature #n_rel x d
        message_pred_to_obj  = self.obj_node_gate(torch.cat([x_j,edge_feature],dim=1)) * edge_feature#n_rel x d
        node_message = (message_pred_to_subj+message_pred_to_obj)
        
        '''Edge'''
        message_subj_to_pred = self.subj_edge_gate(torch.cat([x_i, edge_feature], 1)) * x_i  # nrel x d
        message_obj_to_pred  = self.obj_edge_gate(torch.cat([x_j, edge_feature], 1)) * x_j# nrel x d
        edge_message = (message_subj_to_pred+message_obj_to_pred)

        return [node_message, edge_message]
    
    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x   
    
class MessagePassing_VGfM(MessagePassing):
    def __init__(self, dim_node, aggr= 'mean', **kwargs):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.subj_node_gate = nn.Sequential(nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_node_gate = nn.Sequential(nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())

        self.subj_edge_gate = nn.Sequential(nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        self.obj_edge_gate = nn.Sequential(nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
        
        self.geo_edge_gate = nn.Sequential(nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
    def forward(self,x,edge_feature, geo_feature,edge_index):
        node_msg, edge_msg = self.propagate(edge_index, x=x, edge_feature=edge_feature,geo_feature=geo_feature)
        return node_msg, edge_msg

    def message(self, x_i, x_j,edge_feature, geo_feature):
        message_pred_to_subj = self.subj_node_gate(torch.cat([x_i,edge_feature],dim=1)) * edge_feature #n_rel x d
        message_pred_to_obj  = self.obj_node_gate(torch.cat([x_j,edge_feature],dim=1)) * edge_feature#n_rel x d
        node_message = (message_pred_to_subj+message_pred_to_obj)
        
        message_subj_to_pred = self.subj_edge_gate(torch.cat([x_i, edge_feature], 1)) * x_i  # nrel x d
        message_obj_to_pred  = self.obj_edge_gate(torch.cat([x_j, edge_feature], 1)) * x_j# nrel x d
        message_geo = self.geo_edge_gate(torch.cat([geo_feature,edge_feature], 1)) * geo_feature
        edge_message = (message_subj_to_pred+message_obj_to_pred+message_geo)
        
        # x = torch.cat([x_i,edge_feature,x_j],dim=1)
        # x = self.nn1(x)#.view(b,-1)
        # new_x_i = x[:,:self.dim_hidden]
        # new_e   = x[:,self.dim_hidden:(self.dim_hidden+self.dim_edge)]
        # new_x_j = x[:,(self.dim_hidden+self.dim_edge):]
        # x = new_x_i+new_x_j
        return [node_message, edge_message]
    
    def aggregate(self, x: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        x[0] = scatter(x[0], index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
        return x   

class MessagePassing_Gate(MessagePassing):
    def __init__(self,dim_node,aggr='mean',**kwargs):
        super().__init__(aggr=aggr)
        self.dim_node = dim_node
        self.temporal_gate = nn.Sequential(nn.Linear(self.dim_node * 2, 1), nn.Sigmoid())
    def forward(self,x,edge_index):
        return self.propagate(edge_index,x=x)
    def message(self,x_i,x_j):
        x_i = self.temporal_gate(torch.cat([x_i,x_j],dim=1)) * x_i
        return x_i
        
class TripletIMP(torch.nn.Module):
    def __init__(self, dim_node, num_layers, aggr= 'mean', **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.dim_node = dim_node        
        self.edge_gru = nn.GRUCell(input_size=self.dim_node, hidden_size=self.dim_node)
        self.node_gru = nn.GRUCell(input_size=self.dim_node, hidden_size=self.dim_node)
        self.msp_IMP = MessagePassing_IMP(dim_node=dim_node,aggr=aggr)        
        self.reset_parameter()
        
    def reset_parameter(self):
        pass
    
    def forward(self, x, edge_feature, edge_index, **kwargs):
        x = self.node_gru(x)
        edge_feature = self.node_gru(edge_feature)
        for i in range(self.num_layers):
            node_msg, edge_msg = self.msp_IMP(x=x, edge_feature=edge_feature,edge_index=edge_index)
            x = self.node_gru(node_msg,x)
            edge_feature = self.edge_gru(edge_msg,edge_feature)
        return x, edge_feature
    
class TripletVGfM(torch.nn.Module):
    def __init__(self, dim_node, num_layers, aggr= 'mean', **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.dim_node = dim_node        
        self.edge_gru = nn.GRUCell(input_size=self.dim_node, hidden_size=self.dim_node)
        self.node_gru = nn.GRUCell(input_size=self.dim_node, hidden_size=self.dim_node)
        
        self.msg_vgfm = MessagePassing_VGfM(dim_node=dim_node,aggr=aggr)
        self.msg_t_node = MessagePassing_Gate(dim_node=dim_node, aggr=aggr)
        self.msg_t_edge= MessagePassing_Gate(dim_node=dim_node, aggr=aggr)
        
        
        self.edge_encoder = ssg.models.edge_encoder.EdgeEncoder_VGfM()
        
        self.reset_parameter()
        
    def reset_parameter(self):
        pass
        # reset_parameters_with_activation(self.nn1[0], 'relu')
        # reset_parameters_with_activation(self.nn1[3], 'relu')
        # reset_parameters_with_activation(self.nn2[0], 'relu')
        
    def forward(self, x, edge_feature, edge_index, geo_feature, temporal_node_graph, temporal_edge_graph, **args):
        x = self.node_gru(x)
        edge_feature = self.node_gru(edge_feature)
        extended_geo_feature = self.edge_encoder(geo_feature,edge_index)
        for i in range(self.num_layers):
            node_msg, edge_msg = self.msg_vgfm(x=x, edge_feature=edge_feature,geo_feature=extended_geo_feature,edge_index=edge_index)
            if temporal_node_graph.shape[0] == 2:
                temporal_node_msg = self.msg_t_node(x=x,edge_index=temporal_node_graph)
                node_msg += temporal_node_msg
            if temporal_edge_graph.shape[0] == 2:
                temporal_edge_msg = self.msg_t_edge(x=edge_feature,edge_index=temporal_edge_graph)            
                edge_msg += temporal_edge_msg
            x = self.node_gru(node_msg,x)
            edge_feature = self.edge_gru(edge_msg,edge_feature)
        return x, edge_feature
    
class MSG_MV(MessagePassing):
    def __init__(self, dim_node:int,dim_image:int, num_heads:int):
        super().__init__(aggr='add',flow='source_to_target')
        assert dim_node % num_heads == 0 
        self.num_heads = num_heads
        self.d_k = dim_node // num_heads
        self.sqrt_dk = math.sqrt(dim_node)
        
        self.proj_q = build_mlp([dim_node,dim_node])
        self.proj_k = build_mlp([dim_image,dim_node])
        self.proj_v = build_mlp([dim_image,dim_node])
        pass
    def forward(self,node,image,edge_index):
        # source_to_target (j,i) # target_to_source
        x = (image, node)
        return self.propagate(edge_index, x=x)
        
    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        """
        Given image and node features, 
        this layer estimates 

        Args:
            x_i (Tensor): object nodes
            x_j (Tensor): image nodes
        """
        #TODO: may need to deal with multimodality
        N = x_i.shape[0]
        # n_obj = x_i.shape[0]
        # n_img = x_j.shape[0]
        
        # proj
        q_mat = self.proj_q(x_i).view(N, self.num_heads, self.d_k)
        k_mat = self.proj_k(x_j).view(N, self.num_heads, self.d_k)
        v_mat = self.proj_v(x_j).view(N, self.num_heads, self.d_k)
        # dot product
        att = torch.einsum('nhk,mhk->nhm',q_mat,k_mat)  / self.sqrt_dk
        att = torch.nn.functional.softmax(att,dim=-1)
        
        y = torch.einsum('nhm,mhk->nhk',att,v_mat)
        y = y.reshape(N,-1)
        return y
    
class MSG_FAN(MessagePassing):
    def __init__(self,
                 dim_node:int,dim_edge:int,dim_atten:int,
                 num_heads:int,
                 use_bn:bool,
                 aggr='sum',
                 attn_dropout:float = 0.5,
                 flow:str='target_to_source'):
        super().__init__(aggr=aggr,flow=flow)
        assert dim_node % num_heads == 0
        assert dim_edge % num_heads == 0
        assert dim_atten % num_heads == 0
        self.dim_node_proj = dim_node // num_heads
        self.dim_edge_proj = dim_edge // num_heads
        self.dim_value_proj = dim_atten // num_heads
        self.num_head = num_heads
        self.temperature = math.sqrt(self.dim_edge_proj)
        
        self.nn_att = MLP([self.dim_node_proj+self.dim_edge_proj,self.dim_node_proj+self.dim_edge_proj,
                           self.dim_edge_proj])
        
        self.proj_q = build_mlp([dim_node,dim_node])
        self.proj_k = build_mlp([dim_edge,dim_edge])
        self.proj_v = build_mlp([dim_node,dim_atten])
        
        self.nn_edge = build_mlp([dim_node*2+dim_edge,(dim_node+dim_edge),dim_edge],
                          do_bn= use_bn, on_last=False)
        
        self.dropout = torch.nn.Dropout(attn_dropout) if attn_dropout > 0 else torch.nn.Identity()
        
        '''update'''
        self.update_node = build_mlp([dim_node+dim_atten, dim_node+dim_atten, dim_node],
                             do_bn= use_bn, on_last=False)
    def forward(self, x, edge_feature, edge_index):
        return self.propagate(edge_index,x=x,edge_feature=edge_feature, x_ori=x)
    def message(self, x_i: Tensor, x_j:Tensor, edge_feature:Tensor) -> Tensor:
        '''
        x_i [N, D_N]
        x_j [N, D_N]
        '''
        num_node = x_i.size(0)

        '''triplet'''
        triplet_feature = torch.cat([x_i, edge_feature,x_j],dim=1)
        triplet_feature = self.nn_edge(triplet_feature)
        
        '''FAN'''
        # proj
        x_i = self.proj_q(x_i).view(num_node,self.dim_node_proj,self.num_head) # [N,D,H]
        edge = self.proj_k(edge_feature).view(num_node,self.dim_edge_proj,self.num_head) #[M,D,H]
        x_j = self.proj_v(x_j)
        # est attention
        att = self.nn_att(torch.cat([x_i,edge],dim=1)) # N, D, H
        prob = torch.nn.functional.softmax(att/self.temperature, dim=1)
        prob = self.dropout(prob)
        value = prob.reshape_as(x_j)*x_j
        
        return [value,triplet_feature,prob]
    
    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Optional[Tensor] = None, 
                  dim_size: Optional[int] = None) -> Tensor:
        inputs[0] = scatter(inputs[0],index,dim=self.node_dim,dim_size=dim_size,reduce=self.aggr)
        return inputs
    
    
    def update(self,x, x_ori):
        x[0] = self.update_node(torch.cat([x_ori,x[0]],dim=1))
        return x
    
class JoingGNN_(torch.nn.Module):
    def __init__(self, **args):
        super().__init__()
        self.msg_fan = filter_args_create(MSG_FAN,args)
        self.msg_img = filter_args_create(MSG_MV,args)
        
        dim_node  = args['dim_node'] 
        # dim_atten = args['dim_atten']
        # use_bn = args['use_bn']
        
        # self.node_gru = nn.GRUCell(input_size=dim_node, hidden_size=dim_node)
        self.nn = build_mlp([dim_node*2,dim_node])
        
    def forward(self, node,image,edge,edge_index_node_2_node,edge_index_image_2_ndoe):
        # node_gru = self.node_gru (node)
        '''message passing between image and nodes'''
        node_to_node_msg, edge_feature_msg, prob = self.msg_fan(x=node,edge_feature=edge,edge_index=edge_index_node_2_node)
        
        '''message passing between nodes and ndoes'''
        image_msg = self.msg_img(node,image,edge_index_image_2_ndoe)
        
        '''merge'''
        node_update = self.nn(torch.cat([node_to_node_msg,image_msg],dim=1))
        
        return node_update, edge_feature_msg, prob
    
class JointGNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']
        self.num_heads = kwargs['num_heads']
        self.drop_out = kwargs['drop_out']
        self.gconvs = torch.nn.ModuleList()
        
        self.drop_out = None 
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
        
        for _ in range(self.num_layers):
            self.gconvs.append(JoingGNN_(**kwargs))
            # self.gconvs.append(GraphEdgeAttenNetwork(num_heads,dim_node,dim_edge,dim_atten,aggr, **kwargs))

    def forward(self, data):
        probs = list()
        node = data['node'].x
        image = data['roi'].x
        edge = data['edge'].x
        edge_index_node_2_node = data['node','to','node'].edge_index
        edge_index_image_2_ndoe = data['roi','sees','node'].edge_index
        
        #TODO: use GRU?
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node, edge, prob = gconv(node,image,edge,edge_index_node_2_node,edge_index_image_2_ndoe)
            
            if i < (self.num_layers-1) or self.num_layers==1:
                node = torch.nn.functional.relu(node)
                edge = torch.nn.functional.relu(edge)
                
                if self.drop_out:
                    node = self.drop_out(node)
                    edge = self.drop_out(edge)
                
            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        return node, edge, probs
    
class GraphEdgeAttenNetworkLayers(torch.nn.Module):
    """ A sequence of scene graph convolution layers  """
    def __init__(self, **kwargs):
        super().__init__()
        self.num_layers = kwargs['num_layers']
        
        self.gconvs = torch.nn.ModuleList()
        self.drop_out = None 
        if 'DROP_OUT_ATTEN' in kwargs:
            self.drop_out = torch.nn.Dropout(kwargs['DROP_OUT_ATTEN'])
        
        for _ in range(self.num_layers):
            self.gconvs.append(filter_args_create(MSG_FAN,kwargs))

    def forward(self, data):
        probs = list()
        node_feature = data['node'].x
        edge_feature = data['edge'].x
        edges_indices = data['node','to','node'].edge_index
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            node_feature, edge_feature, prob = gconv(node_feature, edge_feature, edges_indices)
            
            if i < (self.num_layers-1) or self.num_layers==1:
                node_feature = torch.nn.functional.relu(node_feature)
                edge_feature = torch.nn.functional.relu(edge_feature)
                
                if self.drop_out:
                    node_feature = self.drop_out(node_feature)
                    edge_feature = self.drop_out(edge_feature)
                
            if prob is not None:
                probs.append(prob.cpu().detach())
            else:
                probs.append(None)
        return node_feature, edge_feature, probs

if __name__ == '__main__':
    TEST_FORWARD=True
    TEST_TRACE=False
    attention='scaled_dot_product_attention'
    if TEST_FORWARD:
        n_node = 8
        dim_node = 256
        dim_edge = 256
        dim_atten = 256
        num_head = 1
        
        query = torch.rand(n_node,dim_node,1)
        edge  = torch.rand(n_node,dim_edge,1)
        # model = MultiHeadedEdgeAttention(num_head, dim_node, dim_edge,dim_atten)
        # model(query,edge,value)
        
        num_edge = 8
        query = torch.rand(n_node,dim_node)
        edge  = torch.rand(num_edge,dim_edge)
        edge_index = torch.randint(0, n_node-1, [2,num_edge])
        
        # model = EdgeAtten(num_head,dim_node,dim_edge,dim_atten)
        # model(query,edge,edge_index)
        num_layers=2
        model = GraphEdgeAttenNetworkLayers(dim_node, dim_edge, dim_edge, num_layers,num_heads=num_head,attention=attention)
        model(query,edge,edge_index)
    
    if TEST_TRACE:
        pth = './tmp'
        # op_utils.create_dir(pth)
        num_heads=1
        dim_node=128
        dim_edge=128
        dim_atten=128
        use_bn=False
        MultiHeadedEdgeAttention(num_heads, dim_node, dim_edge, dim_atten).trace()
        TripletEdgeNet(dim_node, dim_edge).trace()