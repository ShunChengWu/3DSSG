import torch
import torch.nn as nn
from codeLib.common import filter_args_create
from ssg.models.node_encoder.base import NodeEncoderBase
import torch_geometric
from torch_geometric.nn.conv import MessagePassing
import logging
logger_py = logging.getLogger(__name__)


class MSG_MV(MessagePassing):
    def __init__(self, aggr:str):
        super().__init__(aggr=aggr, flow='source_to_target')
    def forward(self,x,edge_index):
        node = torch.zeros([edge_index[1].max()+1,1]).to(x)
        dummpy = (x, node)
        return self.propagate(edge_index,x=dummpy)
    def message(self, x_j):
        """

        Args:
            x_j (_type_): image_feature
        """
        return x_j
            

class MVCNN(NodeEncoderBase):
    def __init__(self,cfg,backbone:str,device):
        super().__init__(cfg,backbone,device)
        self.global_pooling_method = cfg.model.image_encoder.aggr
        self.mv_msg = MSG_MV(aggr=cfg.model.image_encoder.aggr)
    def reset_parameters(self):
        pass
    def forward(self, images, **args):
        if not self.input_is_roi:
            bboxes = args['bboxes']
        edge_index = args['edge_index']
            
        '''get image features'''
        images = self.preprocess(images)
        
        '''agg'''
        msg = self.mv_msg(images,edge_index)
        
        
        # '''compute node feature base on the given edges'''
        # n_nodes = len(images) if self.input_is_roi else len(bboxes)
        # nodes_feature = torch.zeros([n_nodes, self.node_feature_dim],device=self._device)
        
        # for node_idx in range(n_nodes):
        #     if not self.input_is_roi:
        #         kf2box = bboxes[node_idx]
        #         roi_features = self.postprocess(images,kf2box)
        #     else:
        #         roi_features = self.postprocess(images[node_idx],None)
                
        #     '''do pooling on each feature channel before fc'''
        #     if self.global_pooling_method == 'max':
        #         roi_features = torch.max(roi_features,0,keepdim=True)[0]
        #     elif self.global_pooling_method == 'mean':
        #         roi_features = torch.mean(roi_features,0)
        #     elif self.global_pooling_method == 'sum':
        #         roi_features = torch.sum(roi_features,0)
        #     else:
        #         raise RuntimeError('unknown global pooling method')
        #     nodes_feature[node_idx] = roi_features.flatten()
        outputs=dict()
        outputs['nodes_feature']=msg
        return outputs
    
if __name__ == '__main__':
    import codeLib
    config = codeLib.Config('../../../configs/default_mv.yaml')
    config.path = '/home/sc/research/PersistentSLAM/python/2DTSG/data/test/'
    
    
    
    model = MVCNN(config,backbone='vgg16',device='cpu')
    if config.data.use_precompute_img_feature:
        images = torch.rand([3,512,32,32])
    else:
        images = torch.rand([3,3,512,512])
    bboxes = [
        {0: torch.FloatTensor([0,0,1,1]), 1: torch.FloatTensor([0,0,0.5,0.5])},
        {1: torch.FloatTensor([0,0,1,1])},
        ]
    output = model(images,bboxes)