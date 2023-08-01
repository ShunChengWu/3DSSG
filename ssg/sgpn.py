#!/usr/bin/env python3
# -*- coding: utf-8 -*-
if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import torch
from torch import nn
from .models.classifier import PointNetCls, PointNetRelClsMulti, PointNetRelCls
import ssg
from codeLib.utils.util import pytorch_count_params


class SGPN(nn.Module):
    def __init__(self, cfg, num_obj_cls, num_rel_cls, device):
        '''
        Scene graph prediction network from https://arxiv.org/pdf/2004.03967.pdf
        '''
        super().__init__()
        self.cfg = cfg
        self._device = device
        node_feature_dim = cfg.model.node_feature_dim
        edge_feature_dim = cfg.model.edge_feature_dim

        '''create model'''
        models = dict()
        models['obj_encoder'] = ssg.models.node_encoder_list['sgfn'](
            cfg, device)
        models['rel_encoder'] = ssg.models.edge_encoder_list['sgpn'](
            cfg, device)

        # if cfg.model.gnn.method != 'none':
        models['gnn'] = ssg.models.gnn_list['triplet'](
            num_layers=cfg.model.gnn.num_layers,
            dim_node=cfg.model.node_feature_dim,
            dim_edge=cfg.model.edge_feature_dim,
            dim_hidden=cfg.model.gnn.hidden_dim,
            with_bn=cfg.model.gnn.with_bn
        )

        with_bn = cfg.model.node_classifier.with_bn
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
        obj_points = data['node'].pts
        rel_points = data['edge'].pts
        node_edges = data['node', 'to', 'node'].edge_index

        has_edge = node_edges.nelement() > 0
        if has_edge and node_edges.shape[0] != 2:
            node_edges = node_edges.t().contiguous()

        obj_feature = self.obj_encoder(obj_points)
        if has_edge:
            rel_feature = self.rel_encoder(rel_points)

        probs = None
        if has_edge and self.cfg.model.gnn.num_layers > 0:
            gcn_obj_feature, gcn_rel_feature = self.gnn(
                obj_feature, rel_feature, node_edges)

            if self.cfg.model.gnn.node_from_gnn:
                obj_cls = self.obj_predictor(gcn_obj_feature)
            else:
                obj_cls = self.obj_predictor(obj_feature)
            rel_cls = self.rel_predictor(gcn_rel_feature)
        else:
            obj_cls = self.obj_predictor(obj_feature)
            if has_edge:
                rel_cls = self.rel_predictor(rel_feature)
            else:
                rel_cls = None

        # if return_meta_data:
        #     return obj_cls, rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs
        # else:
        return obj_cls, rel_cls

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


if __name__ == '__main__':
    pass
