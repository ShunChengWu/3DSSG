import torch
import torch.nn as nn
from ssg.models.node_encoder.base import NodeEncoderBase
# from torchvision import models
# from torchvision.ops import roi_align
import logging
logger_py = logging.getLogger(__name__)

class MVCNN(NodeEncoderBase):
    def __init__(self,cfg,backbone:str,device):
        super().__init__(cfg,backbone,device)
        self.global_pooling_method = cfg.model.image_encoder.aggr
        
    def reset_parameters(self):
        pass
    def forward(self, images, **args):
        if not self.input_is_roi:
            bboxes = args['bboxes']
        '''get image features'''
        images = self.preprocess(images)
        
        '''compute node feature base on the given edges'''
        n_nodes = len(images) if self.input_is_roi else len(bboxes)
        nodes_feature = torch.zeros([n_nodes, self.node_feature_dim],device=self._device)
        
        for node_idx in range(n_nodes):
            if not self.input_is_roi:
                kf2box = bboxes[node_idx]
                roi_features = self.postprocess(images,kf2box)
            else:
                roi_features = self.postprocess(images[node_idx],None)
                
            '''do pooling on each feature channel before fc'''
            if self.global_pooling_method == 'max':
                roi_features = torch.max(roi_features,0,keepdim=True)[0]
            elif self.global_pooling_method == 'mean':
                roi_features = torch.mean(roi_features,0)
            elif self.global_pooling_method == 'sum':
                roi_features = torch.sum(roi_features,0)
            else:
                raise RuntimeError('unknown global pooling method')
            nodes_feature[node_idx] = roi_features.flatten()
        outputs=dict()
        outputs['nodes_feature']=nodes_feature
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