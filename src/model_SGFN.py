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
import optimizer
import math

class SGFNModel(BaseModel):
    def __init__(self,config:Config,name:str, num_class, num_rel, dim_descriptor=11):
        super().__init__(name,config)
        models = dict()
        self.mconfig = mconfig = config.MODEL
        with_bn = mconfig.WITH_BN
        dim_f_spatial = dim_descriptor
        dim_point_rel = dim_f_spatial
        dim_point = 3
        if mconfig.USE_RGB:
            dim_point +=3
        if mconfig.USE_NORMAL:
            dim_point +=3
        self.dim_point=dim_point
        self.dim_edge=dim_point_rel
        self.num_class=num_class
        self.num_rel=num_rel
        
        self.flow = 'target_to_source' # we want the msg
        
        dim_point_feature = self.mconfig.point_feature_size
        if self.mconfig.USE_SPATIAL:
            dim_point_feature -= dim_f_spatial-3 # ignore centroid
        
        # Object Encoder
        models['obj_encoder'] = PointNetfeat(
            global_feat=True, 
            batch_norm=with_bn,
            point_size=dim_point, 
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=dim_point_feature)      
        
        ## Relationship Encoder
        models['rel_encoder'] = PointNetfeat(
            global_feat=True,
            batch_norm=with_bn,
            point_size=dim_point_rel,
            input_transform=False,
            feature_transform=mconfig.feature_transform,
            out_size=mconfig.edge_feature_size)
        
        ''' Message passing between segments and segments '''
        if self.mconfig.USE_GCN:
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
                                    flow=self.flow,
                                    attention=self.mconfig.ATTENTION,
                                    use_edge=self.mconfig.USE_GCN_EDGE,
                                    DROP_OUT_ATTEN=self.mconfig.DROP_OUT_ATTEN)
            else:
                raise NotImplementedError('')
        
        ''' node feature classifier '''
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
        
        self.optimizer = optim.AdamW(
            params = params,
            lr = float(config.LR),
            weight_decay=self.config.W_DECAY,
            amsgrad=self.config.AMSGRAD
        )
        self.optimizer.zero_grad()
        
        self.scheduler = None
        if self.config.LR_SCHEDULE == 'BatchSize':
            def _scheduler(epoch, batchsize):
                return 1/math.log(batchsize) if batchsize>1 else 1
            self.scheduler = optimizer.BatchMultiplicativeLR(self.optimizer, _scheduler)
        
    def forward(self, segments_points, edges, descriptor, imgs = None, covis_graph = None, return_meta_data=False):
        obj_feature = self.obj_encoder(segments_points)
        
        if self.mconfig.USE_SPATIAL:
            tmp = descriptor[:,3:].clone()
            tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            obj_feature = torch.cat([obj_feature, tmp],dim=1)

        ''' Create edge feature '''
        with torch.no_grad():
            edge_feature = op_utils.Gen_edge_descriptor(flow=self.flow)(descriptor,edges)
        
        rel_feature = self.rel_encoder(edge_feature)
        
                    
        ''' GNN '''
        probs=None
        if self.mconfig.USE_GCN:
            if self.mconfig.GCN_TYPE == 'TRIP':
                gcn_obj_feature, gcn_rel_feature = self.gcn(obj_feature, rel_feature, edges)
            elif self.mconfig.GCN_TYPE == 'EAN':
                gcn_obj_feature, gcn_rel_feature, probs = self.gcn(obj_feature, rel_feature, edges)
        else:
            gcn_obj_feature=gcn_rel_feature=probs=None          
            
        ''' Predict '''
        if self.mconfig.USE_GCN:
            if self.mconfig.OBJ_PRED_FROM_GCN:
                obj_cls = self.obj_predictor(gcn_obj_feature)
            else:
                obj_cls = self.obj_predictor(obj_feature)
            rel_cls = self.rel_predictor(gcn_rel_feature)
        else:
            obj_cls = self.obj_predictor(obj_feature)
            rel_cls = self.rel_predictor(rel_feature)
            
        if return_meta_data:
            return obj_cls, rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs
        else:
            return obj_cls, rel_cls
    
    def process(self, obj_points, edges, descriptor, gt_obj_cls, gt_rel_cls, weights_obj=None, weights_rel=None, ignore_none_rel=False,
                imgs = None, covis_graph = None):
        self.iteration +=1     

        obj_pred, rel_pred, _, _, _, _, probs= self(obj_points, edges, descriptor,return_meta_data=True, imgs=imgs, covis_graph=covis_graph)
        
        if self.mconfig.multi_rel_outputs:
            if self.mconfig.WEIGHT_EDGE == 'BG':
                if self.mconfig.w_bg != 0:
                    weight = self.mconfig.w_bg * (1 - gt_rel_cls) + (1 - self.mconfig.w_bg) * gt_rel_cls
                else:
                    weight = None
            elif self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                batch_mean = torch.sum(gt_rel_cls, dim=(0))
                zeros = (gt_rel_cls ==0).sum().unsqueeze(0)
                batch_mean = torch.cat([zeros,batch_mean],dim=0)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
                if ignore_none_rel:
                    weight[0] = 0
                    weight *= 1e-2 # reduce the weight from ScanNet
                    # print('set weight of none to 0')
                if 'NONE_RATIO' in self.mconfig:
                    weight[0] *= self.mconfig.NONE_RATIO
                    
                weight[torch.where(weight==0)] = weight[0].clone() if not ignore_none_rel else 0# * 1e-3
                weight = weight[1:]
                
            else:
                raise NotImplementedError("unknown weight_edge type")

            loss_rel = F.binary_cross_entropy(rel_pred, gt_rel_cls, weight=weight)
        else:
            if self.mconfig.WEIGHT_EDGE == 'DYNAMIC':
                one_hot_gt_rel = torch.nn.functional.one_hot(gt_rel_cls,num_classes = self.num_rel)
                batch_mean = torch.sum(one_hot_gt_rel, dim=(0), dtype=torch.float)
                weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf
                if ignore_none_rel: 
                    weight[-1] = 0 # assume none is the last relationship
                    weight *= 1e-2 # reduce the weight from ScanNet
            elif self.mconfig.WEIGHT_EDGE == 'OCCU':
                weight = weights_rel
            else:
                raise NotImplementedError("unknown weight_edge type")

            if 'ignore_entirely' in self.mconfig and (self.mconfig.ignore_entirely and ignore_none_rel):
                loss_rel = torch.zeros(1,device=rel_pred.device, requires_grad=False)
            else:
                loss_rel = F.nll_loss(rel_pred, gt_rel_cls, weight = weight)

        loss_obj = F.nll_loss(obj_pred, gt_obj_cls, weight = weights_obj)

        lambda_r = 1.0
        lambda_o = self.mconfig.lambda_o
        lambda_max = max(lambda_r,lambda_o)
        lambda_r /= lambda_max
        lambda_o /= lambda_max

        if 'USE_REL_LOSS' in self.mconfig and not self.mconfig.USE_REL_LOSS:
            loss = loss_obj
        elif 'ignore_entirely' in self.mconfig and (self.mconfig.ignore_entirely and ignore_none_rel):
            loss = loss_obj
        else:
            loss = lambda_o * loss_obj + lambda_r * loss_rel
            
        if self.scheduler is not None:
            self.scheduler.step(batchsize=edges.shape[1])
        self.backward(loss)
        
        logs = [("Loss/cls_loss",loss_obj.detach().item()),
                ("Loss/rel_loss",loss_rel.detach().item()),
                ("Loss/loss", loss.detach().item())]
        return logs, obj_pred.detach(), rel_pred.detach(), probs
    
    def backward(self, loss):
        loss.backward()        
        self.optimizer.step()
        self.optimizer.zero_grad()
    
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
            acc_rel = (rel_gt==pred_rel).sum().item() / rel_gt.nelement()
        else:
            pred_rel = torch.max(rel_pred.detach(),1)[1]
            acc_rel = (rel_gt==pred_rel).sum().item() / rel_gt.nelement()
            
        
        logs = [("Accuracy/obj_cls",acc_obj), 
                ("Accuracy/rel_cls",acc_rel)]
        return logs
    
    def trace(self,path):
        op_utils.create_dir(path)
        params = dict()
        params['USE_GCN']=self.mconfig.USE_GCN
        params['USE_RGB']=self.mconfig.USE_RGB
        params['USE_NORMAL']=self.mconfig.USE_NORMAL
        params['dim_point']=self.dim_point
        params['dim_edge'] =self.dim_edge
        params["DIM_ATTEN"]=self.mconfig.DIM_ATTEN
        params['obj_pred_from_gcn']=self.mconfig.OBJ_PRED_FROM_GCN
        params['dim_o_f']=self.mconfig.point_feature_size
        params['dim_r_f']=self.mconfig.edge_feature_size
        params['dim_hidden_feature']=self.mconfig.gcn_hidden_feature_size
        params['num_classes']=self.num_class
        params['num_relationships']=self.num_rel
        params['multi_rel_outputs']=self.mconfig.multi_rel_outputs
        params['flow'] = self.flow
        
        self.eval()
        params['enc_o'] = self.obj_encoder.trace(path,'obj')
        params['enc_r'] = self.rel_encoder.trace(path,'rel')
        if self.mconfig.USE_GCN:
            params['n_layers']=self.gcn.num_layers
            if self.mconfig.GCN_TYPE == 'EAN':
                for i in range(self.gcn.num_layers):
                    params['gcn_'+str(i)] = self.gcn.gconvs[i].trace(path,'gcn_'+str(i))
            else:
                raise NotImplementedError()
        params['cls_o'] = self.obj_predictor.trace(path,'obj')
        params['cls_r'] = self.rel_predictor.trace(path,'rel')
        return params
        
if __name__ == '__main__':
    use_dataset = False
    
    config = Config('../config_example.json')
    
    if not use_dataset:
        num_obj_cls=40
        num_rel_cls=26
    else:
        from src.dataset_builder import build_dataset
        config.dataset.dataset_type = 'point_graph'
        dataset =build_dataset(config, 'validation_scans', True, multi_rel_outputs=True, use_rgb=False, use_normal=False)
        num_obj_cls = len(dataset.classNames)
        num_rel_cls = len(dataset.relationNames)

    # build model
    mconfig = config.MODEL
    network = SGFNModel(config,'SceneGraphFusionNetwork',num_obj_cls,num_rel_cls)
    
    network.trace('./tmp')
    import sys
    sys.exit()

    if not use_dataset:
        max_rels = 80    
        n_pts = 10
        n_rels = n_pts*n_pts-n_pts
        n_rels = max_rels if n_rels > max_rels else n_rels
        obj_points = torch.rand([n_pts,3,128])
        rel_points = torch.rand([n_rels, 4, 256])
        edge_indices = torch.zeros(n_rels, 2,dtype=torch.long)
        counter=0
        for i in range(n_pts):
            if counter >= edge_indices.shape[0]: break
            for j in range(n_pts):
                if i==j:continue
                if counter >= edge_indices.shape[0]: break
                edge_indices[counter,0]=i
                edge_indices[counter,1]=i
                counter +=1
    
    
        obj_gt = torch.randint(0, num_obj_cls-1, (n_pts,))
        rel_gt = torch.randint(0, num_rel_cls-1, (n_rels,))
    
        # rel_gt
        adj_rel_gt = torch.rand([n_pts, n_pts, num_rel_cls])
        rel_gt = torch.zeros(n_rels, num_rel_cls, dtype=torch.float)
        
        
        for e in range(edge_indices.shape[0]):
            i,j = edge_indices[e]
            for c in range(num_rel_cls):
                if adj_rel_gt[i,j,c] < 0.5: continue
                rel_gt[e,c] = 1
            
        network.process(obj_points,edge_indices.t().contiguous(),obj_gt,rel_gt)
        
    for i in range(100):
        if use_dataset:
            scan_id, instance2mask, obj_points, edge_indices, obj_gt, rel_gt = dataset.__getitem__(i)
            
        logs, obj_pred, rel_pred = network.process(obj_points,edge_indices.t().contiguous(),obj_gt,rel_gt)
        logs += network.calculate_metrics([obj_pred,rel_pred], [obj_gt,rel_gt])
        print('{:>3d} '.format(i),end='')
        for log in logs:
            print('{0:} {1:>2.3f} '.format(log[0],log[1]),end='')
        print('')
            
