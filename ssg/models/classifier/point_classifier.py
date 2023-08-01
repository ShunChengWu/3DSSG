import torch.nn as nn
import torch.nn.functional as F
import torch
from codeLib.utils import onnx
from codeLib.common import reset_parameters_with_activation
import os
# from .networks_base import BaseNetwork


class PointNetCls(nn.Module):
    def __init__(self, k=2, in_size=1024, batch_norm=True, drop_out: float = 0.3, init_weights=True):
        super(PointNetCls, self).__init__()
        self.name = 'pnetcls'
        self.in_size = in_size
        self.k = k
        self.use_batch_norm = batch_norm
        self.use_drop_out = drop_out > 0
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        if self.use_drop_out:
            self.dropout = nn.Dropout(p=drop_out)
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_with_activation(self.fc1, 'relu')
        reset_parameters_with_activation(self.fc2, 'relu')

    def forward(self, x):
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x
        # return F.log_softmax(x, dim=1)

    def trace(self, pth='./tmp', name_prefix=''):
        import os
        x = torch.rand(1, self.in_size)
        names_i = ['x']
        names_o = ['y']
        name = name_prefix+'_'+self.name
        input_ = (x)
        onnx.export(self, input_, os.path.join(pth, name),
                    input_names=names_i, output_names=names_o,
                    dynamic_axes={names_i[0]: {0: 'n_node', 2: 'n_pts'}})
        names = dict()
        names['model_'+name] = dict()
        names['model_'+name]['path'] = name
        names['model_'+name]['input'] = names_i
        names['model_'+name]['output'] = names_o
        return names


class PointNetRelClsMulti(nn.Module):

    def __init__(self, k=2, in_size=1024, batch_norm=True, drop_out=True,
                 init_weights=True):
        super(PointNetRelClsMulti, self).__init__()
        self.name = 'pnetcls'
        self.in_size = in_size
        self.use_bn = batch_norm
        self.use_drop_out = drop_out
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        if self.use_drop_out:
            self.dropout = nn.Dropout(p=0.3)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_with_activation(self.fc1, 'relu')
        reset_parameters_with_activation(self.fc2, 'relu')

    def forward(self, x):
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # x = torch.sigmoid(x)
        return x

    def trace(self, pth='./tmp', name_prefix=''):
        import os
        x = torch.rand(1, self.in_size)
        names_i = ['x']
        names_o = ['y']
        name = name_prefix+'_'+self.name
        onnx.export(self, (x), os.path.join(pth, name),
                    input_names=names_i, output_names=names_o,
                    dynamic_axes={names_i[0]: {0: 'n_node', 2: 'n_pts'}})

        names = dict()
        names['model_'+name] = dict()
        names['model_'+name]['path'] = name
        names['model_'+name]['input'] = names_i
        names['model_'+name]['output'] = names_o
        return names


class PointNetRelCls(nn.Module):

    def __init__(self, k=2, in_size=1024, batch_norm=True, drop_out=True,
                 init_weights=True):
        super(PointNetRelCls, self).__init__()
        self.name = 'pnetcls'
        self.in_size = in_size
        self.use_bn = batch_norm
        self.use_drop_out = drop_out
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        if self.use_drop_out:
            self.dropout = nn.Dropout(p=0.3)
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters_with_activation(self.fc1, 'relu')
        reset_parameters_with_activation(self.fc2, 'relu')

    def forward(self, x):
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
        # return F.log_softmax(x, dim=1) #, trans, trans_feat

    def trace(self, pth='./tmp', name_prefix=''):
        x = torch.rand(1, self.in_size)
        names_i = ['x']
        names_o = ['y']
        name = name_prefix+'_'+self.name
        input_ = (x)
        onnx.export(self, input_, os.path.join(pth, name),
                    input_names=names_i, output_names=names_o,
                    dynamic_axes={names_i[0]: {0: 'n_node', 1: 'n_pts'}})
        names = dict()
        names['model_'+name] = dict()
        names['model_'+name]['path'] = name
        names['model_'+name]['input'] = names_i
        names['model_'+name]['output'] = names_o
        return names
