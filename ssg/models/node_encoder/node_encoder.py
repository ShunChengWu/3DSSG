
from torchvision import models
import torch
import torch.nn as nn
# from ssg.models import encoder
# from ssg.models.networks_base import BaseNetwork
from codeLib.common import reset_parameters_with_activation
from torchvision.ops import roi_align
from ssg.models.encoder import point_encoder_dict
import logging

logger_py = logging.getLogger(__name__)


class NodeEncoder(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        backend = cfg.model.node_encoder.backend
        self._device = device
        self.img_feature_dim = cfg.model.img_feature_dim
        self.node_feature_dim = cfg.model.node_feature_dim
        self.roi_region = cfg.model.node_encoder.roi_region
        self.with_bn = cfg.model.node_encoder.with_bn
        aggr = cfg.model.node_encoder.aggr
        size = 1
        for s in self.roi_region:
            size *= s

        # self.local_pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        self.global_pooling_method = aggr
        assert self.global_pooling_method in ['max', 'mean', 'sum']

        self.activation = nn.ReLU(inplace=True)
        # TODO: we need an encoder that transform image feature to a domain
        if False:
            pass
        #     self.fc = nn.Linear(cfg.model.img_feature_dim*size, self.node_feature_dim)
        else:
            hidden_size = cfg.model.node_encoder.hidden
            drop_out = cfg.model.node_encoder.drop_out
            self.conv1 = nn.Conv2d(
                cfg.model.img_feature_dim, hidden_size, kernel_size=self.roi_region, stride=1)
            self.conv2 = nn.Conv2d(
                hidden_size, self.node_feature_dim, kernel_size=1, stride=1)

            self.use_drop_out = drop_out > 0
            if self.use_drop_out:
                self.dropout = nn.Dropout(p=drop_out)
            if self.with_bn:
                self.bn1 = nn.BatchNorm2d(
                    hidden_size, eps=0.001, momentum=0.01)
                self.bn2 = nn.BatchNorm2d(
                    self.node_feature_dim, eps=0.001, momentum=0.01)
            # self.fc = nn.Linear(1024, self.node_feature_dim)

        self.reset_parameters()

    def reset_parameters(self):
        # reset_parameters_with_activation(self.fc, 'relu')
        reset_parameters_with_activation(self.conv1, 'relu')
        reset_parameters_with_activation(self.conv2, 'relu')

    def forward(self, images, bboxes, edges, **args):
        ''' 1.Encode image '''
        # img_features = self.img_encoder(images)
        img_features = images  # use precomputed image feautre to save time

        n_nodes = len(bboxes)
        # batch = img_features.shape[0]
        width, height = img_features.shape[-1], img_features.shape[-2]

        nodes_feature = torch.zeros(
            [n_nodes, self.node_feature_dim], device=self._device)
        for node_idx in range(n_nodes):
            kf2box = bboxes[node_idx]
            kf_indices = list(kf2box.keys())
            boxes = [v.view(1, 4) for v in kf2box.values()]
            ''' scale bounding box with the current feature map size '''
            for box in boxes:
                box[:, 0] *= width
                box[:, 1] *= height
                box[:, 2] *= width
                box[:, 3] *= height
            selected_img_features = img_features[kf_indices]
            # print('len(selected_img_features):',len(selected_img_features))

            roi_features = roi_align(
                selected_img_features, boxes, self.roi_region)

            # feature encodign
            roi_features = self.conv1(roi_features)
            if self.use_drop_out:
                roi_features = self.dropout(roi_features)
            if self.with_bn:
                roi_features = self.bn1(roi_features)
            roi_features = self.activation(roi_features)

            roi_features = self.conv2(roi_features)
            if self.use_drop_out:
                roi_features = self.dropout(roi_features)
            if self.with_bn:
                roi_features = self.bn2(roi_features)
            roi_features = self.activation(roi_features)

            if self.global_pooling_method == 'max':
                pooled_feature = torch.max(roi_features, 0, keepdim=True)[0]
            elif self.global_pooling_method == 'mean':
                pooled_feature = torch.mean(roi_features, 0)
            elif self.global_pooling_method == 'sum':
                pooled_feature = torch.sum(roi_features, 0)
            else:
                raise RuntimeError('unknown global pooling method')

            # pooled_feature = self.fc(pooled_feature.flatten())
            # pooled_feature = self.activation(pooled_feature)
            nodes_feature[node_idx] = pooled_feature.flatten()

            # '''get local features '''
            # imgs_feature = torch.zeros([len(kf2box),self.img_feature_dim],device=self._device)
            # counter=0
            # for kf_idx, box in kf2box.items():
            #     w_min,w_max = torch.floor(box[0]*width).long(), torch.ceil(box[2]*width).long()
            #     h_min,h_max = torch.floor(box[1]*height).long(), torch.ceil(box[3]*height).long()

            #     img_feature = img_features[kf_idx] # dim [channel, height, width]
            #     local_features = img_feature[:,h_min:h_max,w_min:w_max]
            #     if check_valid(local_features):
            #         print('1')
            #     imgs_feature[counter] = self.local_pooling(local_features).squeeze() # dim [channel]
            #     if  check_valid(imgs_feature[counter]):
            #         print('counter:',counter)
            #     counter+=1
            # '''pool features '''
            # if self.global_pooling_method == 'max':
            #     nodes_feature[node_idx] = torch.max(imgs_feature,0,keepdim=True)[0]
            # else:
            #     raise RuntimeError('unknown global pooling method')
        return nodes_feature


class NodeEncoderVGG16(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self._device = device
        self.node_feature_dim = cfg.model.node_feature_dim
        self.roi_region = cfg.model.node_encoder.roi_region

        if self.node_feature_dim != 4096:
            logger_py.warning(
                "self.node_feature_dim must have size 4096. add a projection layer to project final feature")
            self.proj = nn.Linear(4096, self.node_feature_dim)
        else:
            self.proj = nn.Sequential()

        if self.roi_region[0] != 7:
            logger_py.warning(
                "roi region must be 7 when using NodeEncoderVGG16. change it from {} to 7".format(self.roi_region[0]))
            self.roi_region = [7 for x in self.roi_region]

        vgg16 = models.vgg16(pretrained=True)
        aggr = cfg.model.node_encoder.aggr
        self.global_pooling_method = aggr
        assert self.global_pooling_method in ['max', 'mean', 'sum']
        # self.encoder = vgg16.features[:-1] # skip the last maxpool2d
        self.fc = vgg16.classifier[:-1]  # skip the last layer

        self.with_precompute = cfg.data.use_precompute_img_feature
        if not self.with_precompute:
            # build image encoder
            self.encoder = vgg16.features.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

    def reset_parameters(self):
        pass

    def forward(self, images, bboxes, edges, **args):
        ''' 1.Encode image '''
        if self.with_precompute:
            img_features = images  # use precomputed image feautre to save time
        else:
            self.encoder.eval()
            with torch.no_grad():
                images = torch.rot90(images, 3, [-1, -2])
                img_features = torch.cat(
                    [self.encoder(p_split) for p_split in torch.split(images, int(4), dim=0)], dim=0)
                # img_features = self.encoder(images)
                img_features = torch.rot90(img_features, 1, [-1, -2])

        n_nodes = len(bboxes)
        width, height = img_features.shape[-1], img_features.shape[-2]
        nodes_feature = torch.zeros(
            [n_nodes, self.node_feature_dim], device=self._device)

        for node_idx in range(n_nodes):
            kf2box = bboxes[node_idx]
            kf_indices = list(kf2box.keys())
            boxes = [v.view(1, 4) for v in kf2box.values()]
            ''' scale bounding box with the current feature map size '''
            for box in boxes:
                box[:, 0] *= width
                box[:, 1] *= height
                box[:, 2] *= width
                box[:, 3] *= height
            selected_img_features = img_features[kf_indices]

            roi_features = roi_align(
                selected_img_features, boxes, self.roi_region)
            roi_features = self.fc(
                roi_features.view(roi_features.shape[0], -1))
            roi_features = self.proj(roi_features)

            if self.global_pooling_method == 'max':
                pooled_feature = torch.max(roi_features, 0, keepdim=True)[0]
            elif self.global_pooling_method == 'mean':
                pooled_feature = torch.mean(roi_features, 0)
            elif self.global_pooling_method == 'sum':
                pooled_feature = torch.sum(roi_features, 0)
            else:
                raise RuntimeError('unknown global pooling method')
            nodes_feature[node_idx] = pooled_feature.flatten()
        return nodes_feature


class NodeEncoderRes18(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self._device = device
        self.node_feature_dim = cfg.model.node_feature_dim
        self.roi_region = cfg.model.node_encoder.roi_region

        if self.node_feature_dim != 512:
            logger_py.warning(
                "self.node_feature_dim must have size 512. add a projection layer to project final feature")
            self.proj = nn.Linear(512, self.node_feature_dim)
        else:
            self.proj = nn.Sequential()

        if self.roi_region[0] != 1:
            logger_py.warning(
                "roi region must be 1 when using NodeEncoderRes18. change it from {} to 1".format(self.roi_region[0]))
            self.roi_region = [1 for x in self.roi_region]

        model = models.resnet18(pretrained=True)
        aggr = cfg.model.node_encoder.aggr
        self.global_pooling_method = aggr
        assert self.global_pooling_method in ['max', 'mean', 'sum']
        # self.encoder = vgg16.features[:-1] # skip the last maxpool2d

        self.with_precompute = cfg.data.use_precompute_img_feature
        if not self.with_precompute:
            # build image encoder
            self.encoder = model.features.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

    def reset_parameters(self):
        pass

    def forward(self, images, bboxes, edges, **args):
        ''' 1.Encode image '''
        if self.with_precompute:
            img_features = images  # use precomputed image feautre to save time
        else:
            self.encoder.eval()
            with torch.no_grad():
                images = torch.rot90(images, 3, [-1, -2])
                img_features = torch.cat(
                    [self.encoder(p_split) for p_split in torch.split(images, int(4), dim=0)], dim=0)
                # img_features = self.encoder(images)
                img_features = torch.rot90(img_features, 1, [-1, -2])

        n_nodes = len(bboxes)
        width, height = img_features.shape[-1], img_features.shape[-2]
        nodes_feature = torch.zeros(
            [n_nodes, self.node_feature_dim], device=self._device)

        for node_idx in range(n_nodes):
            kf2box = bboxes[node_idx]
            kf_indices = list(kf2box.keys())
            boxes = [v.view(1, 4) for v in kf2box.values()]
            ''' scale bounding box with the current feature map size '''
            for box in boxes:
                box[:, 0] *= width
                box[:, 1] *= height
                box[:, 2] *= width
                box[:, 3] *= height
            selected_img_features = img_features[kf_indices]

            roi_features = roi_align(
                selected_img_features, boxes, self.roi_region)
            if self.global_pooling_method == 'max':
                pooled_feature = torch.max(roi_features, 0, keepdim=True)[0]
            elif self.global_pooling_method == 'mean':
                pooled_feature = torch.mean(roi_features, 0)
            elif self.global_pooling_method == 'sum':
                pooled_feature = torch.sum(roi_features, 0)
            else:
                raise RuntimeError('unknown global pooling method')
            nodes_feature[node_idx] = pooled_feature.flatten()
        return nodes_feature


class NodeEncoder_SGFN(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        # backend = cfg.model.node_encoder.backend
        self._device = device

        dim_pts = 3
        if cfg.model.use_rgb:
            dim_pts += 3
        if cfg.model.use_normal:
            dim_pts += 3
        self.dim_pts = dim_pts

        self.model = point_encoder_dict['pointnet'](point_size=dim_pts,
                                                    out_size=cfg.model.node_feature_dim,
                                                    batch_norm=cfg.model.node_encoder.with_bn)

    def forward(self, x):
        return self.model(x)
