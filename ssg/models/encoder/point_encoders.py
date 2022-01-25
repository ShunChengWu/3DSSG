import torch
import torch.nn as nn
# from ssg2d.models.networks_base import BaseNetwork
from codeLib.common import reset_parameters_with_activation
from codeLib.utils import onnx
class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, 
                 point_size=3, out_size=1024, batch_norm = True,
                 init_weights=True, pointnet_str:str=None):
        super(PointNetfeat, self).__init__()
        self.name = 'pnetenc'
        self.use_batch_norm = batch_norm
        self.relu = nn.ReLU()
        self.point_size = point_size
        self.out_size = out_size
        
        self.conv1 = torch.nn.Conv1d(point_size, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, out_size, 1)
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(out_size)
        self.global_feat = global_feat

        self.reset_parameters()
        
    def reset_parameters(self):
        reset_parameters_with_activation(self.conv1,'relu')
        reset_parameters_with_activation(self.conv2,'relu')
        reset_parameters_with_activation(self.conv3,'relu')
        
    def forward(self, x, return_meta=False):
        assert x.ndim >2
        n_pts = x.size()[2]
      
        trans = torch.zeros([1])
        
        x = self.conv1(x)
        if self.use_batch_norm:
            self.bn1(x)
        x = self.relu(x)
        
        trans_feat = torch.zeros([1]) # cannot be None in tracing. change to 0
        pointfeat = x
        x = self.conv2(x)
        if self.use_batch_norm:
            self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        if self.use_batch_norm:
            self.bn3(x)
        x = self.relu(x)
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_size)
        
        if self.global_feat:
            if return_meta:
                return x, trans, trans_feat
            else:
                return x
            
        else:
            x = x.view(-1, self.out_size, 1).repeat(1, 1, n_pts)
            if not return_meta:
                return torch.cat([x, pointfeat], 1)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
    def trace(self, pth = './tmp',name_prefix=''):
        import os
        x = torch.rand(1, self.point_size,512)
        names_i = ['x']
        names_o = ['y']
        name = name_prefix+'_'+self.name
        input_ = (x)
        dynamic_axes = {names_i[0]:{0:'n_node', 2:'n_pts'}}
        onnx.export(self, input_, os.path.join(pth, name), 
                        input_names=names_i, output_names=names_o, 
                        dynamic_axes = dynamic_axes)
        
        names = dict()
        names['model_'+name] = dict()
        names['model_'+name]['path'] = name
        names['model_'+name]['input']=names_i
        names['model_'+name]['output']=names_o
        return names