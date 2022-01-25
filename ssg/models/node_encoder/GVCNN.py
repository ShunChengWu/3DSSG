'''
modified from https://github.com/waxnkw/gvcnn-pytorch

but instead of using google net, use vgg16.
'''
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torchvision.ops import roi_align
from ssg.models.node_encoder.base import NodeEncoderBase
import logging
import ssg.models.encoder.inceptionV4 as Icpv4
from codeLib.common import filter_args_create
logger_py = logging.getLogger(__name__)

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class OneConvFc(nn.Module):
    """
    1*1 conv + fc to obtain the grouping schema
    """
    def __init__(self):
        super(OneConvFc, self).__init__()
        self.conv = nn.Conv2d(in_channels=192, out_channels=1, kernel_size=1, stride=1)
        self.fc = nn.Linear(in_features=60*60, out_features=1)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class GroupSchema(nn.Module):
    """
    differences from paper:
    1. Considering the amount of params, we use 1*1 conv instead of  fc
    2. if the scores are all very small, it will cause a big problem in params' update,
    so we add a softmax layer to normalize the scores after the convolution layer
    """
    def __init__(self):
        super(GroupSchema, self).__init__()
        self.score_layer = OneConvFc()
        self.sft = nn.Softmax(dim=1)

    def forward(self, raw_view):
        """
        :param raw_view: [V N C H W]
        :return:
        """
        scores = []
        for batch_view in raw_view:
            # batch_view: [N C H W]
            # y: [N]
            y = self.score_layer(batch_view)
            y = torch.sigmoid(torch.log(torch.abs(y)))
            scores.append(y)
        # view_scores: [N V]
        view_scores = torch.stack(scores, dim=0).transpose(0, 1)
        view_scores = view_scores.squeeze(dim=-1)
        return self.sft(view_scores)


def view_pool(ungrp_views, view_scores, num_grps=7):
    """
    :param ungrp_views: [V C H W]
    :param view_scores: [V]
    :param num_grps the num of groups. used to calc the interval of each group.
    :return: grp descriptors [(grp_descriptor, weight)]
    """

    def calc_scores(scores):
        """
        :param scores: [score1, score2 ....]
        :return:
        """
        n = len(scores)
        s = torch.ceil(scores[0]*n)
        for idx, score in enumerate(scores):
            if idx == 0:
                continue
            s += torch.ceil(score*n)
        s /= n
        return s

    interval = 1 / (num_grps + 1)
    # begin = 0
    view_grps = [[] for i in range(num_grps)]
    score_grps = [[] for i in range(num_grps)]

    for idx, (view, view_score) in enumerate(zip(ungrp_views, view_scores)):
        begin = 0
        for j in range(num_grps):
            right = begin + interval
            if j == num_grps-1:
                right = 1.1
            if begin <= view_score < right:
                view_grps[j].append(view)
                score_grps[j].append(view_score)
            begin += interval
    # print(score_grps)
    view_grps = [sum(views)/len(views) for views in view_grps if len(views) > 0]
    score_grps = [calc_scores(scores) for scores in score_grps if len(scores) > 0]

    shape_des = map(lambda a, b: a*b, view_grps, score_grps)
    shape_des = sum(shape_des)/sum(score_grps)

    # !!! if all scores are very small, it will cause some problems in params' update
    if sum(score_grps) < 0.1:
        # shape_des = sum(view_grps)/len(score_grps)
        print(sum(score_grps), score_grps)
    # print('score total', score_grps)
    return shape_des


def group_pool(final_view, scores):
    """
    view pooling + group fusion
    :param final_view: # [N V C H W]
    :param scores: [N V] scores
    :return: shape descriptor
    """
    shape_descriptors = []

    for idx, (ungrp_views, view_scores) in enumerate(zip(final_view, scores)):
        # ungrp_views: [V C H W]
        # view_scores: [V]

        # view pooling
        shape_descriptors.append(view_pool(ungrp_views, view_scores))
    # [N C H W]
    y = torch.stack(shape_descriptors, 0)
    # print('[2 C H W]', y.size())
    return y

class SVCNN(nn.Module):

    def __init__(self, nclasses=21, pretraining=True):
        super().__init__()
        self.nclasses = nclasses
        self.pretraining = pretraining
        
        self.net = Icpv4.inceptionv4()
        self.net.last_linear = nn.Linear(1536, nclasses)
        self.net.last_linear.apply(init_weights)

        # If not pre-trained, network should be initialized
        if self.pretraining is False:
           self.apply(init_weights)

    def get_first_n_layer(self, n):
        return self.net.features[0:n]

    def extract_feature(self, x):
        y = self.net.features(x)
        y = self.net.avg_pool(y)
        return y

    def forward(self, x):
        output = dict()
        output['nodes_feature'] = self.net(x)
        return output

    
def load_model(model, path, skip_prefix:str='encoder.'):
    ckpt = torch.load(path)
    # modify keys
    state_dicts = {key[len(skip_prefix):]: value   for key,value in ckpt['model'].items() if key.find(skip_prefix) >=0}
    model.load_state_dict(state_dicts)
    # ckpt['model'].keys()
    pass

class GVCNN(NodeEncoderBase):
    def __init__(self, cfg, num_class:int, device):
        super().__init__(cfg,backbone='others',device=device)
        # self.img_batch_size = 4
        # self._device = device
        self.node_feature_dim = 1536
        self.roi_region = cfg.model.node_encoder.roi_region
        svcnn = SVCNN(num_class)
        # filter_args_create()
        load_model(svcnn, cfg.model.node_encoder.backend_ckpt, cfg.model.node_encoder.backend_ckpt_prefix)
        
        # fcn_1
        self.nn_enc = nn.Sequential(*list(svcnn.net.features[0:5]))
        self.nn_post = nn.Sequential()
        
        # self.fcn_1 = nn.Sequential(*list(self.svcnn.net.features[0:5]))
        self.group_schema = GroupSchema()
        init_weights(self.group_schema)
        self.fcn_2 = nn.Sequential(*list(svcnn.net.features[5:]))
        self.avg_pool_2 = svcnn.net.avg_pool
        self.fc_2 = svcnn.net.last_linear
        
    def forward(self, images, bboxes, **args):
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
                
            raw_view = roi_features.unsqueeze(1)
            view_scores = self.group_schema(raw_view)
            
            
            final_view = self.fcn_2(roi_features)
            
            shape_decriptors = group_pool(final_view.unsqueeze(0), view_scores)
            
            z = self.avg_pool_2(shape_decriptors)
            
            z = z.flatten()
            
            nodes_feature[node_idx] = z
                
            # '''do pooling on each feature channel before fc'''
            # if self.global_pooling_method == 'max':
            #     roi_features = torch.max(roi_features,0,keepdim=True)[0]
            # elif self.global_pooling_method == 'mean':
            #     roi_features = torch.mean(roi_features,0)
            # elif self.global_pooling_method == 'sum':
            #     roi_features = torch.sum(roi_features,0)
            # else:
            #     raise RuntimeError('unknown global pooling method')
            # nodes_feature[node_idx] = roi_features.flatten()
        nodes_feature = self.fc_2(nodes_feature)
            
        outputs=dict()
        outputs['nodes_feature']=nodes_feature
        return outputs
        
        """
        :param x: N V C H W
        :return:
        """
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
            
            roi_features = roi_align(selected_img_features, boxes, self.roi_region)
            
            # [V N 192 _ _]
            raw_view = roi_features.unsqueeze(1)
            view_scores = self.group_schema(raw_view)
            
            
            final_view = self.fc(roi_features.view(roi_features.shape[0],-1))
            # [N V C H W] -> [N C H W]
            shape_decriptors = group_pool(final_view.unsqueeze(0), view_scores)
            
            z = self.avg_pool_2(shape_decriptors)
            
            roi_features = self.proj(shape_decriptors)
            
            nodes_feature[node_idx] = roi_features.flatten()
            
        outputs=dict()
        outputs['nodes_feature']=nodes_feature
        return outputs
    
if __name__ == '__main__':
    import codeLib
    config = codeLib.Config('../../../configs/default.yaml')
    config.path = '/home/sc/research/PersistentSLAM/python/2DTSG/data/test/'
    
    
    svcnn = SVCNN(40,True)
    config.data.use_precompute_img_feature=False
    model = GVCNN(config,backbone='vgg16',device='cpu')
    
    images = torch.rand([3,3,512,512])
    bboxes = [
        {0: torch.FloatTensor([0,0,1,1]), 1: torch.FloatTensor([0,0,0.5,0.5])},
        {1: torch.FloatTensor([0,0,1,1])},
        ]
    output = model(images,bboxes)