#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 17:19:08 2021

@author: sc
"""
import torch
from torch import nn
import ssg
from codeLib.utils.util import pytorch_count_params
import logging
from pytictoc import TicToc
logger_py = logging.getLogger(__name__)


class IMP(nn.Module):
    def __init__(self, cfg, num_obj_cls, num_rel_cls, device):
        '''
        Iterative messag passing

        Parameters
        ----------
        cfg : TYPE
            DESCRIPTION.
        num_obj_cls : TYPE
            DESCRIPTION.
        num_rel_cls : TYPE
            DESCRIPTION.
        device : TYPE
            DESCRIPTION.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        super().__init__()
        self.cfg = cfg
        self._device = device
        self.dim = 512
        self.update_step = 2
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        logger_py.info('use offical setup')
        self.cfg.model.image_encoder.update({
            'backend': "vgg16",
            'backend_finetune': False,
            'use_global': True,
        })
        self.cfg.model.gnn.update({
            "hidden_dim": 512,
            "num_layers": 2,
            "aggr": "mean"
        })

        self.times = dict()

        '''build models'''
        models = dict()

        models['roi_extractor'] = ssg.models.node_encoder_list['roi_extractor'](
            cfg, cfg.model.image_encoder.backend, device)

        models['obj_embedding'] = ssg.models.classifier.classifier_list['vgg16'](
            models['roi_extractor'].node_feature_dim,
            self.dim, replace=True,
            pretrained=True)

        models['pred_embedding'] = ssg.models.classifier.classifier_list['vgg16'](
            models['roi_extractor'].node_feature_dim,
            self.dim, replace=True,
            pretrained=True)

        models['gnn'] = ssg.models.gnn_list[cfg.model.gnn.method](
            dim_node=cfg.model.gnn.hidden_dim,
            num_layers=cfg.model.gnn.num_layers,
            aggr=cfg.model.gnn.aggr,
        )

        models['obj_predictor'] = ssg.models.classifier_list['imp'](
            cfg.model.gnn.hidden_dim, num_obj_cls)
        models['rel_predictor'] = ssg.models.classifier_list['imp'](
            cfg.model.gnn.hidden_dim, num_rel_cls)

        params = list()
        print('==trainable parameters==')
        for name, model in models.items():
            if model is None:
                self.name = model
                continue
            # if len(cfg.GPU) > 1:
            #     model = torch.nn.DataParallel(model, config.GPU)
            model = model.to(device)
            self.add_module(name, model)
            params += list(model.parameters())
            print(name, pytorch_count_params(model))
        print('')

    def eval(self):
        # nn.Module.eval(self)
        super().train(mode=False)
        # self.roi_extractor.with_precompute=True
        # self.roi_extractor.with_precompute=False

    def train(self):
        # nn.Module.train(self)
        super().train(mode=True)
        # self.roi_extractor.with_precompute=False
        # self.roi_extractor.with_precompute=True

    def forward(self, data):
        timer = TicToc()
        images = data['roi'].img
        image_boxes = data['roi'].box
        image_node_edges = data['roi', 'to', 'roi'].edge_index

        has_edge = image_node_edges.nelement() > 0
        '''compute image feature'''
        timer.tic()
        roi, edge_roi = self.roi_extractor(
            images, image_boxes, image_node_edges)
        self.times['1.roi_extractor'] = timer.tocvalue()

        timer.tic()
        data['roi'].x = self.obj_embedding(roi)
        self.times['2.obj_embedding'] = timer.tocvalue()

        '''compute edge feature'''
        timer.tic()
        if has_edge:
            data['edge2D'].x = self.pred_embedding(edge_roi)
        self.times['3.pred_embedding'] = timer.tocvalue()

        '''free memory'''
        # print(torch.cuda.memory_allocated())
        # print(torch.cuda.memory_reserved())
        del roi
        del edge_roi
        del images
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated())
        # print(torch.cuda.memory_reserved())

        '''GNN'''
        timer.tic()
        if hasattr(self, 'gnn') and self.gnn is not None and has_edge:
            data['roi'].x, data['edge2D'].x = self.gnn(data)
        self.times['4.gnn'] = timer.tocvalue()

        '''predict'''
        # object
        timer.tic()
        obj_class_logits = self.obj_predictor(data['roi'].x)
        self.times['5.obj_predictor'] = timer.tocvalue()

        # edge
        timer.tic()
        if has_edge:
            rel_class_logits = self.rel_predictor(data['edge2D'].x)
        else:
            rel_class_logits = None  # data['edge2D'].x
        self.times['6.rel_predictor'] = timer.tocvalue()

        return obj_class_logits, rel_class_logits

    def calculate_metrics(self, **args):
        outputs = {}
        if 'node_cls_pred' in args and 'node_cls_gt' in args:
            node_cls_pred = args['node_cls_pred'].detach()
            node_cls_pred = torch.softmax(node_cls_pred, dim=1)
            node_cls_gt = args['node_cls_gt']
            node_cls_pred = torch.max(node_cls_pred, 1)[1]
            acc_node_cls = (node_cls_gt == node_cls_pred).sum(
            ).item() / node_cls_gt.nelement()
            outputs['acc_node_cls'] = acc_node_cls

        if 'edge_cls_pred' in args and 'edge_cls_gt' in args and args['edge_cls_pred'] is not None and args['edge_cls_pred'].nelement() > 0:
            edge_cls_pred = args['edge_cls_pred'].detach()
            edge_cls_gt = args['edge_cls_gt']
            if self.cfg.model.multi_rel:
                edge_cls_pred = torch.sigmoid(edge_cls_pred)
                edge_cls_pred = edge_cls_pred > 0.5
            else:
                edge_cls_pred = torch.softmax(edge_cls_pred, dim=1)
                edge_cls_pred = torch.max(edge_cls_pred, 1)[1]
            acc_edgee_cls = (edge_cls_gt == edge_cls_pred).sum(
            ).item() / edge_cls_gt.nelement()

            outputs['acc_edgee_cls'] = acc_edgee_cls
        return outputs