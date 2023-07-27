import torch
from ssg.models.node_encoder.base import NodeEncoderBase
from ssg.models.network_GNN import MSG_MV_DIRECT
import logging
logger_py = logging.getLogger(__name__)

class MVCNN(NodeEncoderBase):
    def __init__(self,cfg,backbone:str,device):
        super().__init__(cfg,backbone,device)
        self.global_pooling_method = cfg.model.image_encoder.aggr
        self.mv_msg = MSG_MV_DIRECT(aggr=cfg.model.image_encoder.aggr)
        
    def reset_parameters(self):
        pass
    def forward(self, images, **args):
        edge_index = args['edge_index']
            
        '''get image features'''
        images = self.preprocess(images)
        '''aggr'''
        node = torch.zeros([edge_index[1].max()+1,1]).to(images)
        nodes_feature = self.mv_msg(node,images,edge_index)
        
        
        outputs=dict()
        outputs['nodes_feature']=nodes_feature
        return outputs