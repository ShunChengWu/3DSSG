#!/usr/bin/env python3
# -*- coding: utf-8 -*-
if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import torch
import torch.optim as optim
import torch.nn.functional as F
from model_base import BaseModel
from network_PointNet import PointNetfeat,PointNetCls,PointNetRelCls,PointNetRelClsMulti
from network_TripletGCN import TripletGCNModel
from network_GNN import GraphEdgeAttenNetworkLayers
from config import Config
import op_utils

class SGPNModel(BaseModel):
    def __init__(self,config:Config,name:str, num_class, num_rel):
        '''
        Scene graph prediction network from https://arxiv.org/pdf/2004.03967.pdf
        '''
        super().__init__(name,config)
        models = dict()
        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN
        self.flow = 'target_to_source' # we want the mess
        
        dim_point = 3
        dim_point_rel = 3
        if mconfig.USE_RGB:
            dim_point +=3
            dim_point_rel+=3
        if mconfig.USE_NORMAL:
            dim_point +=3
            dim_point_rel+=3
            
        if mconfig.USE_CONTEXT:
            dim_point_rel += 1
        
        # Object Encoder
        models['obj_encoder'] = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.point_feature_size)      
        
        # Relationship Encoder
        models['rel_encoder'] = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.edge_feature_size)
        
        # GCN        
        # models['gcn'] = GraphTripleConvNet(input_dim_obj=mconfig.point_feature_size, 
        #                         num_layers=mconfig.N_LAYERS, 
        #                         residual=mconfig.RESIDUAL, 
        #                         pooling=mconfig.POOLING,            
        #                         input_dim_pred=mconfig.point_feature_size,
        #                         hidden_dim=mconfig.gcn_hidden_feature_size) 
        # if mconfig.GCN_TYPE != 'TRIP':
        #     raise RuntimeError('PointNetGCNModel only support TRIP GCN type')
        # models['gcn'] = TripletGCNModel(num_layers=mconfig.N_LAYERS,
        #                                 dim_node = mconfig.point_feature_size,
        #                                 dim_edge = mconfig.point_feature_size,
        #                                 dim_hidden = mconfig.gcn_hidden_feature_size)
        
        if mconfig.GCN_TYPE == "TRIP":
            models['gcn'] = TripletGCNModel(num_layers=mconfig.N_LAYERS,
                                            dim_node = mconfig.point_feature_size,
                                            dim_edge = mconfig.edge_feature_size,
                                            dim_hidden = mconfig.gcn_hidden_feature_size)
        elif mconfig.GCN_TYPE == 'EAN':
            models['gcn'] = GraphEdgeAttenNetworkLayers(self.mconfig.point_feature_size,
                                self.mconfig.edge_feature_size,
                                self.mconfig.DIM_ATTEN,
                                self.mconfig.N_LAYERS, 
                                self.mconfig.NUM_HEADS,
                                self.mconfig.GCN_AGGR,
                                flow=self.flow)
        
        # node feature classifier        
        models['obj_predictor'] = PointNetCls(num_class, in_size=mconfig.point_feature_size,
                                 batch_norm=with_bn,drop_out=True)
        
        if mconfig.multi_rel_outputs:
            models['rel_predictor'] = PointNetRelClsMulti(
                num_rel, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
        else:
            models['rel_predictor'] = PointNetRelCls(
                num_rel, 
                in_size=mconfig.edge_feature_size, 
                batch_norm=with_bn,drop_out=True)
            
        params = list()
        print('==trainable parameters==')
        for name, model in models.items():
            if len(config.GPU) > 1:
                model = torch.nn.DataParallel(model, config.GPU)
            self.add_module(name, model)
            params += list(model.parameters())
            print(name,op_utils.pytorch_count_params(model))
        print('')
        
        self.optimizer = optim.Adam(
            params = params,
            lr = float(config.LR),
        )
        self.optimizer.zero_grad()
        
    def forward(self, obj_points, rel_points, edges, return_meta_data=False):
        obj_feature = self.obj_encoder(obj_points)
        rel_feature = self.rel_encoder(rel_points)

        probs=None        
        if self.mconfig.USE_GCN:
            if self.mconfig.GCN_TYPE == 'TRIP':
                gcn_obj_feature, gcn_rel_feature = self.gcn(obj_feature, rel_feature, edges)
            elif self.mconfig.GCN_TYPE == 'EAN':
                gcn_obj_feature, gcn_rel_feature, probs = self.gcn(obj_feature, rel_feature, edges)
            
            if self.mconfig.OBJ_PRED_FROM_GCN:
                obj_cls = self.obj_predictor(gcn_obj_feature)
            else:
                obj_cls = self.obj_predictor(obj_feature)
            rel_cls = self.rel_predictor(gcn_rel_feature)

        else:
            gcn_obj_feature=gcn_rel_feature=None
            obj_cls = self.obj_predictor(obj_feature)
            rel_cls = self.rel_predictor(rel_feature)
        
        if return_meta_data:
            return obj_cls, rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs
        else:
            return obj_cls, rel_cls
        
    
    def process(self, obj_points, rel_points, edges, gt_obj_cls, gt_rel_cls, weights_obj=None, weights_rel=None):
        self.iteration +=1     

        obj_pred, rel_pred, _, _, _, _, probs = self(obj_points, rel_points, edges,return_meta_data=True)
        
        # if self.mconfig.multi_rel_outputs:
        #     if self.mconfig.w_bg != 0:
        #         weight_FG_BG = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
        #     else:
        #         weight_FG_BG = None

        #     loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls, weight=weight_FG_BG)
        # else:
        #     raise NotImplementedError('')

        loss_obj = F.nll_loss(obj_pred, gt_obj_cls, weight = weights_obj)
                    
        if self.mconfig.multi_rel_outputs:
            weight_FG_BG = None
            loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls, weight=weight_FG_BG)
        else:
            loss_rel = F.nll_loss(rel_pred, gt_rel_cls, weight=None)

        loss = self.mconfig.lambda_o * loss_obj + loss_rel
        
        self.backward(loss)
        
        logs = [("Loss/cls_loss",loss_obj.detach().item()),
                ("Loss/rel_loss",loss_rel.detach().item()),
                ("Loss/loss", loss.detach().item())]
        return logs, obj_pred.detach(), rel_pred.detach(), probs
    
    def backward(self, loss):
        loss.backward()        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def norm_tensor(self, points, dim):
        # points.shape = [n, 3, npts]
        centroid = torch.mean(points, dim=-1).unsqueeze(-1) # N, 3
        points -= centroid # n, 3, npts
        furthest_distance = points.pow(2).sum(1).sqrt().max(1)[0] # find maximum distance for each n -> [n]
        points /= furthest_distance[0]
        return points
    
    def calculate_metrics(self, preds, gts):
        assert(len(preds)==2)
        assert(len(gts)==2)
        obj_pred = preds[0].detach()
        rel_pred = preds[1].detach()
        obj_gt   = gts[0]
        rel_gt   = gts[1]
        
        pred_cls = torch.max(obj_pred.detach(),1)[1]
        acc_obj = (obj_gt == pred_cls).sum().item() / obj_gt.nelement()
        
        if self.mconfig.multi_rel_outputs:
            pred_rel= rel_pred.detach() > 0.5
        else:
            pred_rel = torch.max(rel_pred.detach(),1)[1]
        acc_rel = (rel_gt==pred_rel).sum().item() / rel_gt.nelement()
        
        logs = [("Accuracy/obj_cls",acc_obj), 
                ("Accuracy/rel_cls",acc_rel)]
        return logs
    

        
if __name__ == '__main__':
    use_dataset = True
    
    config = Config('../config_example.json')
    config.MODEL.USE_RGB=False
    config.MODEL.USE_NORMAL=False
    
    if not use_dataset:
        num_obj_cls=40
        num_rel_cls=26
    else:
        from src.dataset_builder import build_dataset
        config.dataset.dataset_type = 'rio_graph'
        dataset =build_dataset(config, 'validation_scans', True, multi_rel_outputs=True, use_rgb=False, use_normal=False)
        num_obj_cls = len(dataset.classNames)
        num_rel_cls = len(dataset.relationNames)

    # build model
    mconfig = config.MODEL
    network = SGPNModel(config,'SGPNModel',num_obj_cls,num_rel_cls)

    if not use_dataset:
        max_rels = 80    
        n_pts = 10
        n_rels = n_pts*n_pts-n_pts
        n_rels = max_rels if n_rels > max_rels else n_rels
        obj_points = torch.rand([n_pts,3,128])
        rel_points = torch.rand([n_rels, 4, 256])
        edges = torch.zeros(n_rels, 2,dtype=torch.long)
        counter=0
        for i in range(n_pts):
            if counter >= edges.shape[0]: break
            for j in range(n_pts):
                if i==j:continue
                if counter >= edges.shape[0]: break
                edges[counter,0]=i
                edges[counter,1]=i
                counter +=1
    
    
        obj_gt = torch.randint(0, num_obj_cls-1, (n_pts,))
        rel_gt = torch.randint(0, num_rel_cls-1, (n_rels,))
    
        # rel_gt
        adj_rel_gt = torch.rand([n_pts, n_pts, num_rel_cls])
        rel_gt = torch.zeros(n_rels, num_rel_cls, dtype=torch.float)
        
        
        for e in range(edges.shape[0]):
            i,j = edges[e]
            for c in range(num_rel_cls):
                if adj_rel_gt[i,j,c] < 0.5: continue
                rel_gt[e,c] = 1
            
        network.process(obj_points,rel_points,edges,obj_gt,rel_gt)
        
    for i in range(100):
        if use_dataset:
            scan_id, instance2mask, obj_points, rel_points, obj_gt, rel_gt, edges = dataset.__getitem__(i)
            obj_points = obj_points.permute(0,2,1)
            rel_points = rel_points.permute(0,2,1)
        logs, obj_pred, rel_pred = network.process(obj_points,rel_points,edges,obj_gt,rel_gt)
        logs += network.calculate_metrics([obj_pred,rel_pred], [obj_gt,rel_gt])
        # pred_cls = torch.max(obj_pred.detach(),1)[1]
        # acc_obj = (obj_gt == pred_cls).sum().item() / obj_gt.nelement()
        
        # rel_pred = rel_pred.detach() > 0.5
        # acc_rel = (rel_gt==(rel_pred>0)).sum().item() / rel_pred.nelement()
        
        # print('{0:>3d} acc_obj: {1:>1.4f} acc_rel: {2:>1.4f} loss: {3:>2.3f}'.format(i,acc_obj,acc_rel,logs[0][1]))
        print('{:>3d} '.format(i),end='')
        for log in logs:
            print('{0:} {1:>2.3f} '.format(log[0],log[1]),end='')
        print('')
            
