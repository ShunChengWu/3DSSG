#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 17:19:08 2021

@author: sc
"""
import os
from codeLib.utils import onnx
import torch
from torch import nn
import ssg
from .models.classifier import PointNetCls, PointNetRelClsMulti, PointNetRelCls
from codeLib.utils.util import pytorch_count_params
import logging
logger_py = logging.getLogger(__name__)


class SGFN(nn.Module):
    def __init__(self, cfg, num_obj_cls, num_rel_cls, device):
        super().__init__()
        self.cfg = cfg
        self._device = device
        self.with_img_encoder = self.cfg.model.image_encoder.method != 'none'
        self.with_pts_encoder = self.cfg.model.node_encoder.method != 'none'
        node_feature_dim = cfg.model.node_feature_dim
        edge_feature_dim = cfg.model.edge_feature_dim

        if self.with_img_encoder and self.with_pts_encoder:
            raise NotImplementedError("")

        self.use_spatial = use_spatial = self.cfg.model.spatial_encoder.method != 'none'
        sptial_feature_dim = 0
        if use_spatial:
            if self.with_pts_encoder:
                # # ignore centroid (11-3=8)
                sptial_feature_dim = 8
                node_feature_dim -= sptial_feature_dim
                cfg.model.node_feature_dim = node_feature_dim
            if self.with_img_encoder:
                sptial_feature_dim = 6
                # sptial_feature_dim = 0
                # node_feature_dim -= sptial_feature_dim
                # cfg.model.node_feature_dim = node_feature_dim

        models = dict()
        '''point encoder'''
        if self.with_pts_encoder:
            if self.cfg.model.node_encoder.method == 'basic':
                models['obj_encoder'] = ssg.models.node_encoder_list['sgfn'](
                    cfg, device)
            else:
                models['obj_encoder'] = ssg.models.node_encoder_list[self.cfg.model.node_encoder.method](
                    cfg, device)

        '''image encoder'''
        if self.with_img_encoder:
            img_encoder_method = self.cfg.model.image_encoder.method
            if img_encoder_method == 'cvr':
                logger_py.info('use CVR original implementation')
                cfg.model.image_encoder.backend = 'res18'
                cfg.model.node_feature_dim = 512
                encoder = ssg.models.node_encoder_list[img_encoder_method](
                    cfg, cfg.model.image_encoder.backend, device)
                node_feature_dim = encoder.node_feature_dim
            elif img_encoder_method == 'mvcnn':
                logger_py.info('use MVCNN original implementation')
                cfg.model.image_encoder.backend = 'vgg16'
                cfg.model.node_feature_dim = 25088
                cfg.model.image_encoder.aggr = 'max'
                encoder = ssg.models.node_encoder_list[img_encoder_method](
                    cfg, cfg.model.image_encoder.backend, device)
                node_feature_dim = encoder.node_feature_dim
            elif img_encoder_method == 'mvcnn_res18':
                logger_py.info('use MVCNN original implementation')
                cfg.model.image_encoder.backend = 'res18'
                cfg.model.node_feature_dim = 512
                encoder = ssg.models.node_encoder_list['mvcnn'](
                    cfg, cfg.model.image_encoder.backend, device)
                node_feature_dim = encoder.node_feature_dim
            elif img_encoder_method == 'gvcnn':
                logger_py.info('use GVCNN original implementation')
                encoder = ssg.models.node_encoder_list[img_encoder_method](
                    cfg, num_obj_cls, device)
                # classifier = torch.nn.Identity()
            # use RNN with the assumption of Markov Random Fields(MRFs) and CRF  https://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf
            elif img_encoder_method == 'rnn':
                raise NotImplementedError()
            elif img_encoder_method == 'mean':  # prediction by running mean.
                logger_py.info('use mean cls feature')
                encoder = ssg.models.node_encoder_list[img_encoder_method](
                    cfg, num_obj_cls, cfg.model.image_encoder.backend, device)
                # classifier = torch.nn.Identity()
                # raise NotImplementedError()
            elif img_encoder_method == 'gmu':  # graph memory unit
                logger_py.info('use GMU')
                encoder = ssg.models.node_encoder_list[img_encoder_method](
                    cfg, num_obj_cls, cfg.model.image_encoder.backend, device)
                # if cfg.model.mean_cls:
                #     classifier = torch.nn.Identity()
                # else:
                #     node_feature_dim = cfg.model.gmu.memory_dim
                #     node_clsifier = "res18" if cfg.model.node_classifier.method == 'basic' else cfg.model.node_classifier.method #default is res18
                #     classifier = ssg.models.classifider_list[node_clsifier](in_channels=node_feature_dim, out_channels=num_obj_cls)
            else:
                raise NotImplementedError()
            node_feature_dim = encoder.node_feature_dim
            models['img_encoder'] = encoder

        '''edge encoder'''
        if self.cfg.model.edge_encoder.method == 'basic':
            models['rel_encoder'] = ssg.models.edge_encoder_list['sgfn'](
                cfg, device)
        else:
            models['rel_encoder'] = ssg.models.edge_encoder_list[self.cfg.model.edge_encoder.method](
                cfg, device)

        if use_spatial:
            if self.cfg.model.spatial_encoder.method == 'fc':
                models['spatial_encoder'] = torch.nn.Linear(
                    sptial_feature_dim, cfg.model.spatial_encoder.dim)
                node_feature_dim += cfg.model.spatial_encoder.dim
            elif self.cfg.model.spatial_encoder.method == 'identity':
                models['spatial_encoder'] = torch.nn.Identity()
                node_feature_dim += sptial_feature_dim
            else:
                raise NotImplementedError()
        else:
            models['spatial_encoder'] = torch.nn.Identity()
            node_feature_dim += sptial_feature_dim

        cfg.model.node_feature_dim = node_feature_dim

        if cfg.model.gnn.method != 'none':
            models['gnn'] = ssg.models.gnn_list[cfg.model.gnn.method](
                dim_node=cfg.model.node_feature_dim,
                dim_edge=cfg.model.edge_feature_dim,
                dim_atten=cfg.model.gnn.hidden_dim,
                num_layers=cfg.model.gnn.num_layers,
                num_heads=cfg.model.gnn.num_heads,
                aggr='max',
                DROP_OUT_ATTEN=cfg.model.gnn.drop_out,
                use_bn=False
            )

        '''build classifier'''
        with_bn = cfg.model.node_classifier.with_bn
        if self.with_img_encoder:
            if img_encoder_method == 'cvr':
                classifier = ssg.models.classifider_list['cvr'](
                    in_channels=node_feature_dim, out_channels=num_obj_cls)
            elif img_encoder_method == 'mvcnn':
                classifier = ssg.models.classifider_list['vgg16'](
                    in_channels=node_feature_dim, out_channels=num_obj_cls)
            elif img_encoder_method == 'mvcnn_res18':
                classifier = ssg.models.classifider_list['res18'](
                    in_channels=node_feature_dim, out_channels=num_obj_cls)
            elif img_encoder_method == 'gvcnn':
                classifier = torch.nn.Identity()
            elif img_encoder_method == 'mean':  # prediction by running mean.
                classifier = torch.nn.Identity()
            elif img_encoder_method == 'gmu':  # graph memory unit
                if cfg.model.mean_cls:
                    classifier = torch.nn.Identity()
                else:
                    node_feature_dim = cfg.model.gmu.memory_dim
                    node_clsifier = "res18" if cfg.model.node_classifier.method == 'basic' else cfg.model.node_classifier.method  # default is res18
                    classifier = ssg.models.classifider_list[node_clsifier](
                        in_channels=node_feature_dim, out_channels=num_obj_cls)
            else:
                raise NotImplementedError()

            # classifier = PointNetCls(num_obj_cls, in_size=node_feature_dim,
            #                          batch_norm=with_bn,drop_out=cfg.model.node_classifier.dropout)

            models['obj_predictor'] = classifier
        else:
            models['obj_predictor'] = PointNetCls(num_obj_cls, in_size=node_feature_dim,
                                                  batch_norm=with_bn, drop_out=cfg.model.node_classifier.dropout)

        if cfg.model.multi_rel:
            models['rel_predictor'] = PointNetRelClsMulti(
                num_rel_cls,
                in_size=edge_feature_dim,
                batch_norm=with_bn, drop_out=True)
        else:
            models['rel_predictor'] = PointNetRelCls(
                num_rel_cls,
                in_size=edge_feature_dim,
                batch_norm=with_bn, drop_out=True)

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

    def forward(self, data):
        '''shortcut'''
        descriptor = data['node'].desp
        edge_indices_node_to_node = data['node', 'to', 'node'].edge_index

        has_edge = edge_indices_node_to_node.nelement() > 0
        """reshape node edges if needed"""
        if has_edge and edge_indices_node_to_node.shape[0] != 2:
            edge_indices_node_to_node = edge_indices_node_to_node.t().contiguous()

        '''compute node feature'''
        # from points
        if self.with_pts_encoder:
            data['node'].x = self.obj_encoder(data['node'].pts)
        # from imgae
        if self.with_img_encoder:
            img_dict = self.img_encoder(
                data['roi'].img, edge_index=data['roi', 'sees', 'node'].edge_index)
            data['node'].x = img_dict['nodes_feature']
        # froms spatial descriptor
        if self.use_spatial:
            if self.with_pts_encoder:
                tmp = descriptor[:, 3:].clone()
                tmp[:, 6:] = tmp[:, 6:].log()  # only log on volume and length
                tmp = self.spatial_encoder(tmp)
            else:
                # in R5
                tmp = descriptor[:, 3:8].clone()
                # log on volume and length
                tmp[:, 3:] = tmp[:, 3:].log()
                # x,y ratio in R1
                # in log space for stability
                xy_ratio = tmp[:, 0].log() - tmp[:, 1].log()
                xy_ratio = xy_ratio.view(-1, 1)
                # [:, 6] -> [:, N]
                tmp = self.spatial_encoder(torch.cat([tmp, xy_ratio], dim=1))

            data['node'].x = torch.cat([data['node'].x, tmp], dim=1)

        '''compute edge feature'''
        if has_edge:
            data['node', 'to', 'node'].x = self.rel_encoder(
                descriptor, edge_indices_node_to_node)

        '''Messsage Passing'''
        if has_edge:
            ''' GNN '''
            probs = None
            node_feature_ori = None
            if not self.cfg.model.gnn.node_from_gnn:
                node_feature_ori = data['node'].x
            if hasattr(self, 'gnn') and self.gnn is not None:
                gnn_nodes_feature, gnn_edges_feature, probs = \
                    self.gnn(data)

                data['node'].x = gnn_nodes_feature
                data['node', 'to', 'node'].x = gnn_edges_feature
            if not self.cfg.model.gnn.node_from_gnn:
                data['node'].x = node_feature_ori
        '''Classification'''
        # Node
        node_cls = self.obj_predictor(data['node'].x)
        # Edge
        if has_edge:
            edge_cls = self.rel_predictor(data['node', 'to', 'node'].x)
        else:
            edge_cls = None
        return node_cls, edge_cls

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

    def trace(self, path):
        path = os.path.join(path, 'traced')
        if not os.path.exists(path):
            os.makedirs(path)
        self.eval()
        print('the traced model will be saved at', path)

        params = dict()
        if self.with_pts_encoder:
            params['enc_o'] = self.obj_encoder.trace(path, 'obj')
        if self.with_img_encoder:
            params['enc_img'] = self.img_encoder.trace(path, 'img')

        # params['enc_r'] = self.rel_encoder.trace(path,'rel')
        if self.cfg.model.gnn.method != 'none':
            params['n_layers'] = self.gnn.num_layers
            if self.cfg.model.gnn.method == 'fan':
                for i in range(self.cfg.model.gnn.num_layers):
                    params['gcn_' +
                           str(i)] = self.gnn.gconvs[i].trace(path, 'gcn_'+str(i))
            else:
                raise NotImplementedError()

        if hasattr(self.obj_predictor, 'trace'):
            params['cls_o'] = self.obj_predictor.trace(path, 'obj')
        else:
            params['cls_o'] = onnx.Linear_layer_wrapper(
                self.obj_predictor, 'obj_cls', path, 'obj')

        params['cls_r'] = self.rel_predictor.trace(path, 'rel')

        pass
