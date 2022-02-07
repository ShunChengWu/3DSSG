import torch
import torch.nn as nn
from ssg.models import encoder
from codeLib.common import check_valid
from torchvision.ops import roi_align
from ssg.models.encoder import point_encoder_dict

from torch_geometric.nn.conv import MessagePassing

class EdgeDescriptor_8(MessagePassing):
    """ A sequence of scene graph convolution layers  """
    def __init__(self, flow="target_to_source"):
        '''
        about target_to_source or source_to_target. check https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        '''
        super().__init__(flow=flow)
    def forward(self, descriptor, edges_indices):
        size = self.__check_input__(edges_indices, None)
        coll_dict = self.__collect__(self.__user_args__,edges_indices,size, {"x":descriptor})
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        edge_feature = self.message(**msg_kwargs)
        return edge_feature
    
    def message(self, x_i, x_j):
        # source_to_target
        # (j, i)
        # 0-2: centroid, 3-5:dims, 6:volume, 7:length
        # to
        # 0-2: offset centroid, 3-5: dim log ratio, 96 volume log ratio, 7: length log ratio
        edge_feature = torch.zeros_like(x_i)
        # centroid offset
        edge_feature[:,0:3] = x_i[:,0:3]-x_j[:,0:3]
        # dim log ratio
        edge_feature[:,3:6] = torch.log(x_i[:,3:6] / x_j[:,3:6])
        # volume log ratio
        edge_feature[:,6] = torch.log( x_i[:,6] / x_j[:,6])
        # length log ratio
        edge_feature[:,7] = torch.log( x_i[:,7] / x_j[:,7])
        # edge_feature, *_ = self.ef(edge_feature.unsqueeze(-1))
        return edge_feature.unsqueeze(-1)

       

class EdgeEncoder_2DSSG(nn.Module):
    def __init__(self,cfg,device):
        super().__init__()
        self.edge_descriptor = EdgeDescriptor_8()
        self.encoder = point_encoder_dict['pointnet'](point_size=8,
                                                      out_size=cfg.model.edge_feature_dim,
                                                      batch_norm=cfg.model.edge_encoder.with_bn)
        
    def forward(self, descriptors, edges, **args):
        edges_descriptor = self.edge_descriptor(descriptors, edges)
        edges_feature = self.encoder(edges_descriptor)
        return edges_feature
        
    
class EdgeDescriptor_SGFN(MessagePassing):#TODO: move to model
    """ A sequence of scene graph convolution layers  """
    def __init__(self, flow="source_to_target"):
        super().__init__(flow=flow)
    def forward(self, descriptor, edges_indices):
        size = self.__check_input__(edges_indices, None)
        coll_dict = self.__collect__(self.__user_args__,edges_indices,size, {"x":descriptor})
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        edge_feature = self.message(**msg_kwargs)
        return edge_feature
    
    def message(self, x_i, x_j):
        # source_to_target
        # (j, i)
        # 0-2: centroid, 3-5: std, 6-8:dims, 9:volume, 10:length
        # to
        # 0-2: offset centroid, 3-5: offset std, 6-8: dim log ratio, 9: volume log ratio, 10: length log ratio
        edge_feature = torch.zeros_like(x_i)
        # centroid offset
        edge_feature[:,0:3] = x_i[:,0:3]-x_j[:,0:3]
        # std  offset
        edge_feature[:,3:6] = x_i[:,3:6]-x_j[:,3:6]
        # dim log ratio
        edge_feature[:,6:9] = torch.log(x_i[:,6:9] / x_j[:,6:9])
        # volume log ratio
        edge_feature[:,9] = torch.log( x_i[:,9] / x_j[:,9])
        # length log ratio
        edge_feature[:,10] = torch.log( x_i[:,10] / x_j[:,10])
        # edge_feature, *_ = self.ef(edge_feature.unsqueeze(-1))
        return edge_feature.unsqueeze(-1)
    
class EdgeEncoder_SGFN(nn.Module):
    def __init__(self,cfg,device):
        super().__init__()
        self.edge_descriptor = EdgeDescriptor_SGFN()
        self.encoder = point_encoder_dict['pointnet'](point_size=11,
                                                      out_size=cfg.model.edge_feature_dim,
                                                      batch_norm=cfg.model.edge_encoder.with_bn)
        
    def forward(self, descriptors, edges, **args):
        edges_descriptor = self.edge_descriptor(descriptors, edges)
        edges_feature = self.encoder(edges_descriptor)
        return edges_feature
    
class EdgeEncoder_SGPN(nn.Module):
    def __init__(self, cfg,device):
        super().__init__()
        self._device = device
        
        dim_pts = 3
        if cfg.model.use_rgb:
            dim_pts += 3
        if cfg.model.use_normal:
            dim_pts += 3
        self.dim_pts=dim_pts+1 #mask
        
        self.model = point_encoder_dict['pointnet'](point_size=self.dim_pts,
                                                      out_size=cfg.model.edge_feature_dim,
                                                      batch_norm=cfg.model.node_encoder.with_bn)
                
    def forward(self, x):
        return self.model(x)