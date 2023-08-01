import importlib
import os
from codeLib.common import filter_args_create
from codeLib.utils import onnx
import torch
from torch import nn
import ssg
import torchvision
from .models.classifier import PointNetRelClsMulti, PointNetRelCls
from codeLib.utils.util import pytorch_count_params
import logging
logger_py = logging.getLogger(__name__)

class JointSG(nn.Module):
    def __init__(self,cfg,num_obj_cls, num_rel_cls, device):
        super().__init__()
        self.cfg=cfg
        self._device=device
        self.with_img_encoder = self.cfg.model.image_encoder.method != 'none'
        self.with_pts_encoder = self.cfg.model.node_encoder.method != 'none'
        node_feature_dim = cfg.model.node_feature_dim
        edge_feature_dim = cfg.model.edge_feature_dim
        img_feature_dim  = cfg.model.img_feature_dim
        
        models = dict()
        '''point encoder'''
        if self.with_pts_encoder:
            if self.cfg.model.node_encoder.method == 'basic':
                models['obj_encoder'] = ssg.models.node_encoder_list['sgfn'](cfg,device)
            else:
                models['obj_encoder'] = ssg.models.node_encoder_list[self.cfg.model.node_encoder.method](cfg,device)
                
        '''image encoder'''
        if self.with_img_encoder:
            if cfg.model.image_encoder.backend == 'res18':
                if img_feature_dim != 512:
                    logger_py.warning('overwrite img_feature_dim from {} to {}'.format(img_feature_dim,512))
                    img_feature_dim = 512
                resnet18=torchvision.models.resnet18(pretrained=True)
                img_enc = nn.Sequential(
                    resnet18.conv1,
                    resnet18.bn1,
                    resnet18.relu,
                    resnet18.maxpool,
                    resnet18.layer1,
                    resnet18.layer2,
                    resnet18.layer3,
                    resnet18.layer4,
                    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                    torch.nn.Flatten(start_dim=1)
                )
            elif cfg.model.image_encoder.backend == 'vgg16':
                raise NotImplementedError
                # if img_feature_dim != 25088:
                #     logger_py.warning('overwrite img_feature_dim from {} to {}'.format(img_feature_dim,25088))
                #     img_feature_dim = 25088
                    
                # if self.use_global:
                #     self.roi_region = [7,7]
                #     self.nn_post = nn.Sequential()
                # else:
                #     self.nn_post = nn.AdaptiveAvgPool2d(output_size=(7, 7))
                    
                # if not self.with_precompute:
                #     vgg16=torchvision.models.vgg16(pretrained=True)
                #     img_enc = vgg16.features.eval()
            else:
                raise NotImplementedError
            # Freeze
            if not cfg.model.image_encoder.backend_finetune:
                logger_py.warning('freeze backend')
                img_enc.eval()
                for param in img_enc.parameters(): param.requires_grad = False
                
            node_feature_dim = img_feature_dim
            models['img_encoder'] = img_enc
            cfg.model.img_feature_dim = img_feature_dim
                
        '''edge encoder'''
        if self.cfg.model.edge_encoder.method == 'basic':
            models['rel_encoder'] = ssg.models.edge_encoder_list['sgfn'](cfg,device)
        else:
            models['rel_encoder'] = ssg.models.edge_encoder_list[self.cfg.model.edge_encoder.method](cfg,device)
                            
        cfg.model.node_feature_dim = node_feature_dim
        
        '''initial node feature'''
        args_jointgnn = cfg.model.gnn['jointgnn']
        args_img_msg = cfg.model.gnn[args_jointgnn['img_msg_method']]#TODO: make the config here eaisre to udnrestand
        gnn_modules = importlib.import_module('ssg.models.network_GNN').__dict__
        img_model = gnn_modules[args_jointgnn['img_msg_method']]
        models['msg_img'] = filter_args_create(img_model,
                                               {**cfg.model.gnn,**args_img_msg,
                                                'dim_node'  : 512,
                                                'dim_edge'  : cfg.model.edge_feature_dim,
                                                'dim_image' : cfg.model.img_feature_dim,}
                                               )
        
        '''GNN'''
        if cfg.model.gnn.method != 'none': 
            gnn_method = cfg.model.gnn.method.lower()
            models['gnn'] = ssg.models.gnn_list[gnn_method](
                with_geo = self.with_pts_encoder,
                dim_node  = cfg.model.node_feature_dim,
                dim_edge  = cfg.model.edge_feature_dim,
                dim_image = cfg.model.img_feature_dim,
                **cfg.model.gnn
                # dim_atten = cfg.model.gnn.hidden_dim,
                # num_layers= cfg.model.gnn.num_layers,
                # num_heads = cfg.model.gnn.num_heads,
                # aggr      = 'max',
                # drop_out  = cfg.model.gnn.drop_out,
                
                # **cfg.model.gnn[gnn_method]
                )
            
        '''spatial feature'''
        self.use_spatial = use_spatial = self.cfg.model.spatial_encoder.method != 'none'
        sptial_feature_dim=0
        if use_spatial:
            # if self.with_pts_encoder:
            #     # # ignore centroid (11-3=8)
            #     sptial_feature_dim = 8
            # elif self.with_img_encoder:
            sptial_feature_dim = 6
            # node_feature_dim -= sptial_feature_dim
            # cfg.model.node_feature_dim = node_feature_dimZ
        if use_spatial:
            if self.cfg.model.spatial_encoder.method == 'fc':
                models['spatial_encoder'] = torch.nn.Linear(sptial_feature_dim, cfg.model.spatial_encoder.dim)
                node_feature_dim+=cfg.model.spatial_encoder.dim
            elif self.cfg.model.spatial_encoder.method == 'identity':
                models['spatial_encoder'] = torch.nn.Identity()
                node_feature_dim+=sptial_feature_dim
            else:
                raise NotImplementedError()
        else:
            models['spatial_encoder'] = torch.nn.Identity()
            node_feature_dim += sptial_feature_dim
        
        '''classifier'''
        with_bn =cfg.model.node_classifier.with_bn
        
        # models['obj_predictor'] = PointNetCls(num_obj_cls, in_size=node_feature_dim,
        #                              batch_norm=with_bn,drop_out=cfg.model.node_classifier.dropout)
        models['obj_predictor'] = ssg.models.classifier_list['res18'](
            in_channels=node_feature_dim, 
            out_channels=num_obj_cls)
        
        if cfg.model.multi_rel:
            models['rel_predictor'] = PointNetRelClsMulti(
                num_rel_cls, 
                in_size=edge_feature_dim, 
                batch_norm=with_bn,drop_out=True)
        else:
            models['rel_predictor'] = PointNetRelCls(
                num_rel_cls, 
                in_size=edge_feature_dim, 
                batch_norm=with_bn,drop_out=True)
            
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
            print(name,pytorch_count_params(model))
        print('')
    def forward(self, data):
        '''shortcut'''
        descriptor =  data['node'].desp
        edge_indices_node_to_node = data['node','to','node'].edge_index
        edge_index_image_2_ndoe = data['roi','sees','node'].edge_index
        
        has_edge = edge_indices_node_to_node.nelement()>0
        """reshape node edges if needed"""
        if has_edge and edge_indices_node_to_node.shape[0] != 2:
            edge_indices_node_to_node = edge_indices_node_to_node.t().contiguous()
            
        '''Get 2D Features'''
        # Compute image feature
        if self.with_img_encoder:
            self.img_encoder.eval()
            img_batch_size = self.cfg.model.image_encoder.img_batch_size
            img_feature = torch.cat([ self.img_encoder(p_split)  for p_split in torch.split(data['roi'].img,int(img_batch_size), dim=0) ], dim=0)
            # img_feature = self.img_encoder(data['roi'].img)
        
        '''Get 3D features'''
        # from pts
        if self.with_pts_encoder:
            geo_feature = self.obj_encoder(data['node'].pts)
            data['geo_feature'].x = geo_feature
            # data['geo_feature'].x = torch.sigmoid(geo_feature)
        else:
            geo_feature = torch.zeros([edge_index_image_2_ndoe[1].max()+1,1]).to(img_feature)
            # geo_feature = torch.zeros_like(img_feature)
            
            
        '''compute initial node feature'''
        data['roi'].x = img_feature
        data['node'].x = self.msg_img(geo_feature,img_feature,edge_index_image_2_ndoe)
            
        '''compute edge feature'''
        if has_edge:
            if self.cfg.model.edge_encoder.method == 'sgpn':
                data['node','to','node'].x = self.rel_encoder(data['node','to','node'].pts)
            else:
                data['node','to','node'].x = self.rel_encoder(descriptor,edge_indices_node_to_node)
            
        '''Message Passing''' 
        if has_edge:
            ''' GNN '''
            probs=None
            if hasattr(self, 'gnn') and self.gnn is not None:
                gnn_nodes_feature, gnn_edges_feature, probs = self.gnn(data)
                data['node'].x = gnn_nodes_feature
                data['node','to','node'].x = gnn_edges_feature
        
        # froms spatial descriptor
        if self.use_spatial:
            # if self.with_pts_encoder:
            #     tmp = descriptor[:,3:].clone()
            #     tmp[:,6:] = tmp[:,6:].log() # only log on volume and length
            #     tmp = self.spatial_encoder(tmp)
            # else:
            # in R5            
            tmp = descriptor[:,3:8].clone()
            # log on volume and length
            tmp[:,3:]=tmp[:,3:].log()
            # x,y ratio in R1
            xy_ratio = tmp[:,0].log() - tmp[:,1].log() # in log space for stability
            xy_ratio = xy_ratio.view(-1,1)
            # [:, 6] -> [:, N]
            tmp = self.spatial_encoder(torch.cat([tmp,xy_ratio],dim=1))
                
            # data['node'].spatial = tmp
            data['node'].x = torch.cat([data['node'].x, tmp],dim=1)
        
        
        '''Classification'''
        # Node
        node_cls = self.obj_predictor(data['node'].x)
        # Edge
        if has_edge:
            edge_cls = self.rel_predictor(data['node','to','node'].x)
        else:
            edge_cls = None
        return node_cls, edge_cls
    
    def calculate_metrics(self, **args):
        outputs={}
        if 'node_cls_pred' in args and 'node_cls_gt' in args:
            node_cls_pred = args['node_cls_pred'].detach()
            node_cls_pred = torch.softmax(node_cls_pred, dim=1)
            node_cls_gt   = args['node_cls_gt']
            node_cls_pred = torch.max(node_cls_pred,1)[1]
            acc_node_cls = (node_cls_gt == node_cls_pred).sum().item() / node_cls_gt.nelement()
            outputs['acc_node_cls'] = acc_node_cls
        
        if 'edge_cls_pred' in args and 'edge_cls_gt' in args and args['edge_cls_pred'] is not None and args['edge_cls_pred'].nelement()>0:
            edge_cls_pred = args['edge_cls_pred'].detach()
            edge_cls_gt   = args['edge_cls_gt']
            if self.cfg.model.multi_rel:
                edge_cls_pred = torch.sigmoid(edge_cls_pred)
                edge_cls_pred = edge_cls_pred > 0.5
            else:
                edge_cls_pred = torch.softmax(edge_cls_pred, dim=1)
                edge_cls_pred = torch.max(edge_cls_pred,1)[1]
            acc_edgee_cls = (edge_cls_gt==edge_cls_pred).sum().item() / edge_cls_gt.nelement()
            
            outputs['acc_edgee_cls'] = acc_edgee_cls
        return outputs
    
    def trace(self, path):
        path = os.path.join(path,'traced')
        if not os.path.exists(path): os.makedirs(path)
        self.eval()
        print('the traced model will be saved at',path)
        
        
        params = dict()
        if self.with_pts_encoder:
            params['enc_o'] = self.obj_encoder.trace(path,'obj')
        if self.with_img_encoder:
            params['enc_img'] = self.img_encoder.trace(path,'img')

        # params['enc_r'] = self.rel_encoder.trace(path,'rel')
        if self.cfg.model.gnn.method != 'none': 
            params['n_layers']=self.gnn.num_layers
            if self.cfg.model.gnn.method == 'fan':
                for i in range(self.cfg.model.gnn.num_layers):
                    params['gcn_'+str(i)] = self.gnn.gconvs[i].trace(path,'gcn_'+str(i))
            else:
                raise NotImplementedError()
                
        if hasattr(self.obj_predictor, 'trace'):
            params['cls_o'] = self.obj_predictor.trace(path,'obj')
        else:
            params['cls_o'] = onnx.Linear_layer_wrapper(self.obj_predictor, 'obj_cls',path,'obj')
            
        params['cls_r'] = self.rel_predictor.trace(path,'rel')
        
        pass
