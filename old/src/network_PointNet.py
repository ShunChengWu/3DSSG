'''
The code here is modified from https://github.com/charlesq34/pointnet under MIT License
'''
if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from pointnet.graph import GraphTripleConvNet
from networks_base import BaseNetwork
import op_utils


class STN3d(nn.Module):
    def __init__(self, point_size=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(point_size, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = Variable(torch.from_numpy(
            np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))
        ).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        iden = Variable(torch.eye(self.k).view(1,self.k*self.k).repeat(batchsize,1))
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(BaseNetwork):
    def __init__(self, global_feat = True, input_transform = True, feature_transform = False, 
                 point_size=3, out_size=1024, batch_norm = True,
                 init_weights=True, pointnet_str:str=None):
        super(PointNetfeat, self).__init__()
        self.name = 'pnetenc'
        self.use_batch_norm = batch_norm
        self.relu = nn.ReLU()
        self.point_size = point_size
        self.out_size = out_size
        
        self.conv1 = torch.nn.Conv1d(point_size, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, out_size, 1)
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(out_size)
        self.global_feat = global_feat
        self.input_transform = input_transform
        self.feature_transform = feature_transform
        
        if input_transform:
            assert pointnet_str is not None
            self.pointnet_str=pointnet_str
            self.stn = STN3d(point_size=point_size)
        if self.feature_transform:
            self.fstn = STNkd(k=64)
            
        if init_weights:
            self.init_weights('constant', 1, target_op = 'BatchNorm')
            self.init_weights('xavier_normal', 1)

    def forward(self, x, return_meta=False):
        assert x.ndim >2
        n_pts = x.size()[2]
        if self.input_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            if self.pointnet_str is None and self.point_size ==3:
                x[:,:,:3] = torch.bmm(x[:,:,:3], trans)
            elif self.point_size > 3:
                assert self.pointnet_str is not None 
                for i in len(self.pointnet_str):
                    p = self.pointnet_str[i]
                    offset = i*3
                    offset_ = (i+1)*3
                    if p == 'p' or p == 'n': # point and normal
                        x[:,:,offset:offset_] = torch.bmm(x[:,:,offset:offset_], trans)
            x = x.transpose(2, 1)
        else:
            trans = torch.zeros([1])
        
        x = self.conv1(x)
        if self.use_batch_norm:
            self.bn1(x)
        x = self.relu(x)
        
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = torch.zeros([1]) # cannot be None in tracing. change to 0
        pointfeat = x
        x = self.conv2(x)
        if self.use_batch_norm:
            self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        if self.use_batch_norm:
            self.bn3(x)
        x = self.relu(x)
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.out_size)
        
        if self.global_feat:
            if return_meta:
                return x, trans, trans_feat
            else:
                return x
            
        else:
            x = x.view(-1, self.out_size, 1).repeat(1, 1, n_pts)
            if not return_meta:
                return torch.cat([x, pointfeat], 1)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
    def trace(self, pth = './tmp',name_prefix=''):
        import os
        x = torch.rand(1, self.point_size,512)
        names_i = ['x']
        names_o = ['y']
        name = name_prefix+'_'+self.name
        input_ = (x)
        dynamic_axes = {names_i[0]:{0:'n_node', 2:'n_pts'}}
        op_utils.export(self, input_, os.path.join(pth, name), 
                        input_names=names_i, output_names=names_o, 
                        dynamic_axes = dynamic_axes)
        
        names = dict()
        names['model_'+name] = dict()
        names['model_'+name]['path'] = name
        names['model_'+name]['input']=names_i
        names['model_'+name]['output']=names_o
        return names
        

class PointNetCls(BaseNetwork):
    def __init__(self, k=2, in_size=1024, batch_norm = True, drop_out = True,init_weights=True):
        super(PointNetCls, self).__init__()
        self.name = 'pnetcls'
        self.in_size=in_size
        self.k = k
        self.use_batch_norm = batch_norm
        self.use_drop_out   = drop_out
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        if drop_out:
            self.dropout = nn.Dropout(p=0.3)
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        if init_weights:
            self.init_weights('constant', 1, target_op = 'BatchNorm')
            self.init_weights('xavier_normal', 1)
    def forward(self, x):
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
    def trace(self, pth = './tmp',name_prefix=''):
        import os
        x = torch.rand(1, self.in_size)
        names_i = ['x']
        names_o = ['y']
        name = name_prefix+'_'+self.name
        input_ = (x)
        op_utils.export(self, input_, os.path.join(pth, name), 
                        input_names=names_i, output_names=names_o, 
                        dynamic_axes = {names_i[0]:{0:'n_node', 1:'n_pts'}})
        names = dict()
        names['model_'+name] = dict()
        names['model_'+name]['path'] = name
        names['model_'+name]['input']=names_i
        names['model_'+name]['output']=names_o
        return names


class PointNetRelCls(BaseNetwork):

    def __init__(self, k=2, in_size=1024, batch_norm = True, drop_out = True,
                 init_weights=True):
        super(PointNetRelCls, self).__init__()
        self.name = 'pnetcls'
        self.in_size=in_size
        self.use_bn = batch_norm
        self.use_drop_out = drop_out
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        if self.use_drop_out:
            self.dropout = nn.Dropout(p=0.3)
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        if init_weights:
            self.init_weights('constant', 1, target_op = 'BatchNorm')
            self.init_weights('xavier_normal', 1)
    def forward(self, x):
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1) #, trans, trans_feat
    def trace(self, pth = './tmp',name_prefix=''):
        import os
        x = torch.rand(1, self.in_size)
        names_i = ['x']
        names_o = ['y']
        name = name_prefix+'_'+self.name
        input_ = (x)
        op_utils.export(self, input_, os.path.join(pth, name), 
                        input_names=names_i, output_names=names_o, 
                        dynamic_axes = {names_i[0]:{0:'n_node', 1:'n_pts'}})
        names = dict()
        names['model_'+name] = dict()
        names['model_'+name]['path'] = name
        names['model_'+name]['input']=names_i
        names['model_'+name]['output']=names_o
        return names

class PointNetRelClsMulti(BaseNetwork):

    def __init__(self, k=2, in_size=1024, batch_norm = True, drop_out = True,
                 init_weights=True):
        super(PointNetRelClsMulti, self).__init__()
        self.name = 'pnetcls'
        self.in_size=in_size
        self.use_bn = batch_norm
        self.use_drop_out = drop_out
        self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)

        if self.use_drop_out:
            self.dropout = nn.Dropout(p=0.3)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

        if init_weights:
            self.init_weights('constant', 1, target_op = 'BatchNorm')
            self.init_weights('xavier_normal', 1)
    def forward(self, x):
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.use_drop_out:
            x = self.dropout(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
    def trace(self, pth = './tmp',name_prefix=''):
        import os
        x = torch.rand(1, self.in_size)
        names_i = ['x']
        names_o = ['y']
        name = name_prefix+'_'+self.name
        op_utils.export(self, (x), os.path.join(pth, name), 
                        input_names=names_i, output_names=names_o, 
                        dynamic_axes = {names_i[0]:{0:'n_node', 2:'n_pts'}})
        
        names = dict()
        names['model_'+name] = dict()
        names['model_'+name]['path'] = name
        names['model_'+name]['input']=names_i
        names['model_'+name]['output']=names_o
        return names


# class PointNetRelAndObjCls(nn.Module):

#     def __init__(self, k=2, rel_k=2, point_size=4, point_size_rel=4, feature_transform=False,
#                  with_obj_feats_in_rel_pred=False, with_bbox=True, out_size=256, with_relpn=False,
#                  evaluating=False):
#         super(PointNetRelAndObjCls, self).__init__()

#         self. with_obj_feats = with_obj_feats_in_rel_pred

#         self.feature_transform = feature_transform
#         self.feat_object = PointNetfeat(global_feat=True, point_size=point_size, feature_transform=feature_transform,
#                                         out_size=out_size)
#         self.feat_relationship = PointNetfeat(global_feat=True, point_size=point_size_rel, out_size=out_size,
#                                               feature_transform=feature_transform)

#         self.with_relpn = with_relpn
#         self.evaluating = evaluating

#         if self.with_relpn:
#             self.relpn = _RelPN(feat_dim=out_size, with_bbox=with_bbox)

#         self.rel_predictor = PointNetRelCls(rel_k, feature_transform, with_obj_feats=with_obj_feats_in_rel_pred,
#                                             in_size=out_size)
#         self.obj_predictor = PointNetCls(k, feature_transform, cls_only=False, in_size=out_size)

#     def forward(self, x_s, x_o, x_r, edges=None, bboxes=None, target_rels=None):

#         #print(self.feature_transform)
#         #print(x_s.shape, x_o.shape, x_r.shape)

#         relpn_loss = 0

#         if edges is not None:

#             x_all = x_s

#             x_all, trans, trans_feat = self.feat_object(x_all)

#             if self.with_relpn:
#                 #def forward(self, rois, roi_feat, im_info, gt_boxes, num_boxes, use_gt_boxes=False):
#                 #return roi_pairs, roi_proposals, roi_pairs_scores, self.relpn_loss_cls
#                 keep_idx, relpn_loss = self.relpn(x_all, bboxes, target_rels, edges)

#                 print(keep_idx.shape, target_rels.shape)

#                 edges = edges[keep_idx]
#                 #target_rels = target_rels[keep_idx]
#                 x_r = x_r[keep_idx]

#             # Break apart indices for subjects and objects; these have shape (num_triples,)
#             s_idx = edges[:, 0].contiguous()
#             o_idx = edges[:, 1].contiguous()
#             # Get current vectors for subjects and objects; these have shape (num_triples, Din)
#             x_s = x_all[s_idx] # feats
#             x_o = x_all[o_idx] # feats

#             trans_s = trans[s_idx]
#             trans_o = trans[o_idx]
#             if trans_feat is not None:
#                 trans_feat_s = trans_feat[s_idx]
#                 trans_feat_o = trans_feat[o_idx]
#             else:
#                 trans_feat_s = None
#                 trans_feat_o = None

#         else:

#             x_s, trans_s, trans_feat_s = self.feat_object(x_s)
#             x_o, trans_o, trans_feat_o = self.feat_object(x_o)

#         x_r, trans_r, trans_feat_r = self.feat_relationship(x_r)

#         if self.with_obj_feats:
#             x = torch.cat([x_s, x_o, x_r], 1)
#             trans = torch.cat([trans_s, trans_o, trans_r], 1)
#             #trans_feat = torch.cat([trans_feat_s, trans_feat_o, trans_feat_r], 1)
#         else:
#             x = x_r
#             trans = trans_r
#             trans_feat = trans_feat_r

#         if edges is not None:
#             x_all = self.obj_predictor(x_all)
#             x = self.rel_predictor(x)

#             if self.with_relpn:
#                 if self.evaluating:
#                     full_rels_cls = torch.zeros([target_rels.size(0), x.size(1)],
#                                                 device=x.device, dtype=x.dtype)
#                     full_rels_cls[:,0] = 1
#                     full_rels_cls[keep_idx] = x
#                 else:
#                     target_rels = target_rels[keep_idx]
#                     full_rels_cls = x
#             else:
#                 full_rels_cls = x

#             #if self.with_relpn:
#             return x_all, None, full_rels_cls, target_rels, relpn_loss,\
#                        trans_s, trans_o, trans_r, trans_feat_s, trans_feat_o, trans_feat_r
#             #else:
#             #    return x_all, None, x, \
#             #           trans_s, trans_o, trans_r, trans_feat_s, trans_feat_o, trans_feat_r

#         else:
#             x_s = self.obj_predictor(x_s)
#             x_o = self.obj_predictor(x_o)
#             x = self.rel_predictor(x)

#             return x_s, x_o, x, \
#                    trans_s, trans_o, trans_r, trans_feat_s, trans_feat_o, trans_feat_r
#             # return objs_cls, rels_cls, target_rels, relpn_loss


# class PointNetGCN(nn.Module):

#     def __init__(self, k=2, rel_k=2, point_size=4, point_size_rel=4,
#                  n_layers=2, residual=True, pooling='avg',
#                  feature_transform=False, multi_rel_outputs=False, out_size=256,
#                  obj_pred_from_gcn=True, with_relpn=False, with_bbox=True, evaluating=False,
#                  use_rgb=False):

#         super().__init__()
#         self.num_classes = k
#         self.num_relationsihps = rel_k
#         self.dim_feature = out_size
#         self.dim_hidden_feature = 512
#         self.pooling = pooling
#         self.n_layers = n_layers
        
#         self.obj_pred_from_gcn = obj_pred_from_gcn
#         self.feature_transform = feature_transform

#         self.with_bbox = with_bbox
#         self.with_relpn = with_relpn
#         self.evaluating = evaluating

#         self.multi_rel_outputs = multi_rel_outputs

#         self.feat_object = PointNetfeat(global_feat=True, point_size=point_size, feature_transform=feature_transform,
#                                         out_size=out_size)
#         self.feat_relationship = PointNetfeat(global_feat=True, point_size=point_size_rel,
#                                               feature_transform=feature_transform, out_size=out_size)

#         if self.with_relpn:
#             self.relpn = _RelPN(feat_dim=out_size, with_bbox=with_bbox)

#         self.gcn = GraphTripleConvNet(input_dim_obj=out_size, 
#                                       num_layers=n_layers, residual=residual, pooling=pooling,            
#                                       input_dim_pred=out_size,
#                                       hidden_dim=self.dim_hidden_feature) # TODO: what should the other params be?

#         self.obj_predictor = PointNetCls(k, feature_transform, in_size=out_size, cls_only=False)

#         if self.multi_rel_outputs:
#             self.rel_predictor = PointNetRelClsMulti(rel_k, feature_transform, in_size=out_size, with_obj_feats=False)
#         else:
#             self.rel_predictor = PointNetRelCls(rel_k, feature_transform, in_size=out_size, with_obj_feats=False)

#     def forward(self, objs_points, rels_points, edges, target_rels=None, batch_size=16, bboxes=None):
#         """
#         :param objs: tensor of shape (num_objects, num_channels, num_points)
#         :param rels: tensor of shape (num_relationships, num_channels, num_points_union)
#         :param edges: tensor of shape (num_relationships, 2) - indicates source and target object for the rels
#         :return: predicted classes for objects and relationships
#         """

#         #split2batches = False

#         objs_feats = []
#         rels_feats = []

#         '''
#         if split2batches:
#             for i in range((objs_points.size(0)-1) // batch_size + 1):
#                 current_bsize = min(batch_size, objs_points.size(0) - i * batch_size)
#                 print(current_bsize)
#                 objs, _, _ = self.feat_object(objs_points[i*batch_size:i*batch_size+current_bsize])

#                 objs_feats.append(objs)

#             for i in range((rels_points.size(0)-1) // batch_size + 1):
#                 current_bsize = min(batch_size, rels_points.size(0) - i * batch_size)
#                 print(current_bsize, rels_points[i*batch_size:i*batch_size+current_bsize].shape)
#                 rels, _, _ = self.feat_relationship(rels_points[i*batch_size:i*batch_size+current_bsize])

#                 rels_feats.append(rels)
                
#             print('end rel loop')

#             objs_feats = torch.cat(objs_feats, 0)
#             rels_feats = torch.cat(rels_feats, 0)
        
#         else:
#         '''
#         objs_feats, _, _ = self.feat_object(objs_points)

#         relpn_loss = 0


#         if self.with_relpn:
#             assert target_rels != None
#             #def forward(self, rois, roi_feat, im_info, gt_boxes, num_boxes, use_gt_boxes=False):
#             #return roi_pairs, roi_proposals, roi_pairs_scores, self.relpn_loss_cls
#             keep_idx, relpn_loss = self.relpn(objs_feats, bboxes, target_rels, edges)

#             #print(keep_idx.shape, target_rels.shape)

#             edges = edges[keep_idx]
#             rels_points = rels_points[keep_idx]

#         rels_feats, _, _ = self.feat_relationship(rels_points)

#         objs, rels = self.gcn(objs_feats, rels_feats, edges)

#         if self.obj_pred_from_gcn:
#             objs_cls = self.obj_predictor(objs)
#         else:
#             objs_cls = self.obj_predictor(objs_feats)

#         rels_cls = self.rel_predictor(rels)

#         if self.with_relpn:
#             if self.evaluating:
#                 full_rels_cls = torch.zeros([target_rels.size(0), rels_cls.size(1)],
#                                             device=rels_cls.device, dtype=rels_cls.dtype)
#                 if not self.multi_rel_outputs:
#                     full_rels_cls[:,0] = 1
#                 full_rels_cls[keep_idx] = rels_cls
#             else:
#                 target_rels = target_rels[keep_idx]
#                 full_rels_cls = rels_cls
#         else:
#             full_rels_cls = rels_cls

#         return objs_cls, full_rels_cls, target_rels, relpn_loss

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    model = PointNetCls()
    model.trace('./tmp')