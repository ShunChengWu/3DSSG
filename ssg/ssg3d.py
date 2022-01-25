import torch
from torch import nn
from .models.classifier import PointNetCls, PointNetRelClsMulti, PointNetRelCls
from codeLib.utils.util import pytorch_count_params

class SSG3D(nn.Module):
    def __init__ (self, cfg, num_obj_cls, num_rel_cls, 
                  node_encoder, edge_encoder, gnn, device):
        super().__init__()
        self.cfg = cfg
        self._device=device
        
        node_feature_dim = cfg.model.node_feature_dim
        edge_feature_dim = cfg.model.edge_feature_dim
        
        
        models = dict()
        
        ''' Node encoder '''
        models['node_encoder'] = node_encoder
        # node_encoder = config.get_node_encoder(cfg)
        ''' Edge encoder '''
        models['edge_encoder'] = edge_encoder
        
        ''' GNN '''
        models['gnn'] = gnn
        
        ''' node feature classifier '''
        with_bn =cfg.model.node_classifier.with_bn
        models['obj_predictor'] = PointNetCls(num_obj_cls, in_size=node_feature_dim,
                                 batch_norm=with_bn,drop_out=cfg.model.node_classifier.dropout)
        
        # ''' edge feature classifier '''
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
        
    def forward(self, **args):
        images = args['images']
        image_bboxes = args['image_bboxes']
        image_edges = args['image_edges']
        descriptor = args['descriptor']
        node_edges = args['node_edges']
        
        ''' compute node feature '''
        # with torch.no_grad():
        nodes_feature = self.node_encoder(images=images, 
                          bboxes=image_bboxes, 
                          edges=image_edges)
        
        ''' compute edge feature '''
        edges_feature =self.edge_encoder(descriptors = descriptor,
                                         edges = node_edges)
        
        ''' update node & edge with GNN '''
        if hasattr(self, 'gnn') and self.gnn is not None:
            gnn_nodes_feature, gnn_edges_feature, probs = self.gnn(nodes_feature, edges_feature, node_edges)
            nodes_feature = gnn_nodes_feature
            edges_feature = gnn_edges_feature
        
        ''' classification '''
        '''1. Node '''
        node_cls = self.obj_predictor(nodes_feature)
        '''2.Edge'''
        # edge_cls=None
        edge_cls = self.rel_predictor(edges_feature)
    
        return node_cls, edge_cls
    
    def calculate_metrics(self, **args):
        if 'node_cls_pred' in args and 'node_cls_gt' in args:
            node_cls_pred = args['node_cls_pred'].detach()
            node_cls_pred = torch.softmax(node_cls_pred, dim=1)
            node_cls_gt   = args['node_cls_gt']
            node_cls_pred = torch.max(node_cls_pred,1)[1]
            acc_node_cls = (node_cls_gt == node_cls_pred).sum().item() / node_cls_gt.nelement()
        
        if 'edge_cls_pred' in args and 'edge_cls_gt' in args:
            edge_cls_pred = args['edge_cls_pred'].detach()
            edge_cls_gt   = args['edge_cls_gt']
            if self.cfg.model.multi_rel:
                edge_cls_pred = torch.sigmoid(edge_cls_pred)
                edge_cls_pred = edge_cls_pred > 0.5
            else:
                edge_cls_pred = torch.softmax(edge_cls_pred, dim=1)
                edge_cls_pred = torch.max(edge_cls_pred,1)[1]
            acc_edgee_cls = (edge_cls_gt==edge_cls_pred).sum().item() / edge_cls_gt.nelement()
        return {
            'acc_node_cls': acc_node_cls,
            'acc_edgee_cls': acc_edgee_cls,
        }
        