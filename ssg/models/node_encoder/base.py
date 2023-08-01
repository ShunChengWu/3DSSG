#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 21:06:37 2021

@author: sc
"""
import torch
import torch.nn as nn
from torchvision.ops import roi_align
from torchvision import models
import logging
import os
from codeLib.utils import onnx
logger_py = logging.getLogger(__name__)


class NodeEncoderBase(nn.Module):
    def __init__(self, cfg, backbone: str, device):
        '''
        This is the base class for node encoding with given N images with given bounding boxes
        It supports different backbone methods, get obj bounding box feature with different approaches.

        if use_global is true, it compute image feature with the entire image, 
            then crop local image for the given bounding box region with roi_align.
        If use_global is false, it crop the image first with roi_align, then 
            compute image feature locally.

        Parameters
        ----------
        cfg : TYPE
            DESCRIPTION.
        backbone : str
            DESCRIPTION.
        device : TYPE
            DESCRIPTION.

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        super().__init__()
        logger_py.setLevel(cfg.log_level)
        self._device = device
        self.img_batch_size = cfg.model.image_encoder.img_batch_size
        self.use_global = cfg.model.image_encoder.use_global
        self.backbone = backbone.lower()
        self.with_precompute = cfg.data.use_precompute_img_feature
        # in the original paper they set 128 with ModelNet40, 64 with ModelNet10
        self.local_feature_dim = cfg.model.image_encoder.local_feature_dim
        self.node_feature_dim = cfg.model.node_feature_dim
        self.roi_region = cfg.model.image_encoder.roi_region
        self.input_is_roi = cfg.data.is_roi_img
        self.fine_tune = cfg.model.image_encoder.backend_finetune
        # set backend
        # assert self.backbone in ['vgg16','res18']

        if self.backbone == 'vgg16':
            if self.node_feature_dim != 25088:
                logger_py.warning('overwrite node_feature_dim from {} to {}'.format(
                    self.node_feature_dim, 25088))
                self.node_feature_dim = 25088

            if self.use_global:
                self.roi_region = [7, 7]
                self.nn_post = nn.Sequential()
            else:
                self.nn_post = nn.AdaptiveAvgPool2d(output_size=(7, 7))

            if not self.with_precompute:
                vgg16 = models.vgg16(pretrained=True)
                self.nn_enc = vgg16.features.eval()
                if not self.fine_tune:
                    logger_py.warning('freeze backend')
                    self.nn_enc.eval()
                    for param in self.nn_enc.parameters():
                        param.requires_grad = False

        elif self.backbone == 'res18':
            if self.node_feature_dim != 512:
                logger_py.warning('overwrite node_feature_dim from {} to {}'.format(
                    self.node_feature_dim, 512))
                self.node_feature_dim = 512
            if self.use_global:
                self.roi_region = [1, 1]
                self.nn_post = nn.Sequential()
            else:
                self.nn_post = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if not self.with_precompute:
                resnet18 = models.resnet18(pretrained=True)
                self.nn_enc = nn.Sequential(
                    resnet18.conv1,
                    resnet18.bn1,
                    resnet18.relu,
                    resnet18.maxpool,
                    resnet18.layer1,
                    resnet18.layer2,
                    resnet18.layer3,
                    resnet18.layer4
                )
                # build image nn_enc
                if not cfg.model.image_encoder.backend_finetune:
                    logger_py.warning('freeze backend')
                    self.nn_enc.eval()
                    for param in self.nn_enc.parameters():
                        param.requires_grad = False
        elif self.backbone == 'others':
            logger_py.info(
                'use self defined backbone. please define self.nn_enc and self.nn_post in your init.')
        else:
            raise RuntimeError('unknown')

    def preprocess(self, images):
        if not self.with_precompute and not self.fine_tune:
            self.nn_enc.eval()
        if self.input_is_roi:
            x = torch.cat([self.nn_enc(p_split) for p_split in torch.split(
                images, int(self.img_batch_size), dim=0)], dim=0)
            return self.nn_post(x).flatten(1)
            return images

        if self.with_precompute:
            return images  # use precomputed image feautre to save time
        elif self.use_global:
            # if use global feature, images feature are computed using the full image, then use ROI align to get local feature
            self.nn_enc.eval()
            with torch.no_grad():
                # images=torch.rot90(images,3,[-1,-2])
                return torch.cat([self.nn_enc(p_split) for p_split in torch.split(images, int(self.img_batch_size), dim=0)], dim=0)
        else:
            return images

    def postprocess(self, images, kf2box):
        if not self.with_precompute and not self.fine_tune:
            self.nn_enc.eval()
        width, height = images.shape[-1], images.shape[-2]
        if not self.input_is_roi:
            kf_indices = list(kf2box.keys())
            # print("kf2box",kf2box)
            boxes = [v.view(1, 4) for v in kf2box.values()]
            if self.use_global:
                ''' scale bounding box with the current feature map size '''
                for box in boxes:
                    box[:, 0] *= width
                    box[:, 1] *= height
                    box[:, 2] *= width
                    box[:, 3] *= height
                selected_img_features = images[kf_indices]

                '''get local view features'''
                y = roi_align(selected_img_features, boxes, self.roi_region)
                y = y.view(y.shape[0], -1)  # [views, cdim]
                y = self.nn_post(y)
                return y
            else:
                img_features = torch.zeros(
                    [len(boxes), self.node_feature_dim], device=self._device)
                for i in range(len(boxes)):
                    image = images[kf_indices[i]].unsqueeze(0)
                    box = boxes[i]
                    box[:, 0] *= width
                    box[:, 1] *= height
                    box[:, 2] *= width
                    box[:, 3] *= height
                    w = box[:, 2] - box[:, 0]
                    h = box[:, 3] - box[:, 1]
                    x = roi_align(image, [box], [h, w])
                    x = self.nn_enc(x)
                    x = self.nn_post(x).flatten()
                    img_features[i] = x
                return img_features
        else:
            return images
            x = torch.cat([self.nn_enc(p_split) for p_split in torch.split(
                images, int(self.img_batch_size), dim=0)], dim=0)
            x = self.nn_post(x)
            return x

    def trace(self, pth='./tmp', name_prefix=''):
        # params = inspect.signature(self.forward).parameters
        # params = OrderedDict(params)
        names_i = ['x']
        names_o = ['y']

        onnx.export(self.prop, (cated), os.path.join(pth, name_nn),
                    input_names=names_i, output_names=names_o,
                    dynamic_axes={names_i[0]: {0: 'n_node'}})
        names = dict()
        names['model_'+name] = dict()
        names['model_'+name]['path'] = name
        names['model_'+name]['input'] = names_i
        names['model_'+name]['output'] = names_o
        return names
