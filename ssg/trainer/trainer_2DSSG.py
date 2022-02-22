import os
import numpy as np
from tqdm import tqdm, tnrange
from collections import defaultdict
from codeLib.models import BaseTrainer
from codeLib.common import check_weights, check_valid, convert_torch_to_scalar
import torch
import torch.nn.functional as F
from ssg.utils.util_eva import EvalSceneGraph, plot_confusion_matrix
import time
import logging
from ssg.checkpoints import CheckpointIO
import torch.optim as optim
import codeLib.utils.moving_average as moving_average 
# from net_utils.libs import get_rotation_matrix_gt, get_mask_status
# from configs.data_config import NYU37_TO_PIX3D_CLS_MAPPING
logger_py = logging.getLogger(__name__)
        
class Trainer_2DSSG(BaseTrainer):
    def __init__(self, cfg, model, node_cls_names:list, edge_cls_names:list,
                 optimizer=None, device=None, 
                 **kwargs):
        super().__init__(device)
        # self._device=device if device is not None else 'cpu'
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.w_node_cls = kwargs.get('w_node_cls',None)
        self.w_edge_cls = kwargs.get('w_edge_cls',None)
        self.node_cls_names = node_cls_names#kwargs['node_cls_names']
        self.edge_cls_names = edge_cls_names#kwargs['edge_cls_names']
        self.eva_tool = EvalSceneGraph(self.node_cls_names, self.edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=0) # do not calculate topK in training mode        
        
        # print('self._device',self._device)
        if self.w_node_cls is not None: 
            print('train with weighted node class.')
            self.w_node_cls = self.w_node_cls.to(self._device)
        if self.w_edge_cls is not None: 
            print('train with weighted edge class.')
            self.w_edge_cls = self.w_edge_cls.to(self._device)
        
        self.loss_node_cls = torch.nn.CrossEntropyLoss(weight=self.w_node_cls)
        if self.cfg.model.multi_rel:
            self.loss_rel_cls = torch.nn.BCEWithLogitsLoss(pos_weight=self.w_edge_cls)
        else:
            self.loss_rel_cls = torch.nn.CrossEntropyLoss(weight=self.w_edge_cls)
            
    def zero_metrics(self):
        self.eva_tool.reset()
    
    def evaluate(self, val_loader, topk):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        it_dataset = val_loader.__iter__()
        eval_tool = EvalSceneGraph(self.node_cls_names, self.edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=topk,save_prediction=True) 
        eval_list = defaultdict(moving_average.CMA)

        time.sleep(2)# Prevent possible deadlock during epoch transition
        for data in tqdm(it_dataset,leave=False):
            eval_step_dict = self.eval_step(data,eval_tool=eval_tool)

            for k, v in eval_step_dict.items():
                eval_list[k].update(v)

        eval_dict = {k: v.avg for k, v in eval_list.items()}
        vis = self.visualize(eval_tool=eval_tool)
        eval_dict['visualization'] = vis
        del it_dataset
        return eval_dict, eval_tool
    
    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        self.model.train()
        self.optimizer.zero_grad()
        logs = self.compute_loss(data, it=it, eval_tool = self.eva_tool)
        logs['loss'].backward()
        check_weights(self.model.state_dict())
        self.optimizer.step()
        return logs
    
    
    
    def eval_step(self,data, eval_tool=None):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        eval_dict={}
        with torch.no_grad():
            eval_dict = self.compute_loss(
                data, eval_mode=True, eval_tool=eval_tool)
        eval_dict = convert_torch_to_scalar(eval_dict)
        # for (k, v) in eval_dict.items():
        #     eval_dict[k] = v.item()
        return eval_dict
    
    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        scan_id = data.get('scan_id')
        gt_rel = data.get('gt_rel')
        gt_cls = data.get('gt_cls')
        node_edges = data.get('node_edges')
        image_edges = data.get('image_edges')
        images = data.get('images')
        image_bboxes = data.get('image_boxes')
        instance2mask = data.get('instance2mask')
        descriptor = data.get('descriptor')
        
        flatten = (scan_id, gt_cls, gt_rel, images, image_bboxes, node_edges, image_edges, instance2mask, descriptor)
        flatten = self.toDevice(*flatten)
        # check_valid(*flatten)
        return flatten
    
    def compute_loss(self,data,eval_mode=False,it=None, eval_tool=None):
        ''' Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        # Initialize loss dictionary and other values
        logs = {}
        
        # Process data dictionary
        (scan_id, gt_cls, gt_rel, images, image_bboxes, node_edges, image_edges, instance2mask, descriptor) = self.process_data_dict(data)
        
        # Shortcuts
        # device = self._device
        
        ''' make forward pass through the network '''
        node_cls, edge_cls = self.model(images = images, 
                   image_bboxes = image_bboxes, 
                   image_edges = image_edges, 
                   descriptor = descriptor, 
                   node_edges = node_edges.t().contiguous())
        
        
        ''' calculate loss '''
        logs['loss'] = 0
        
        
        if self.cfg.training.lambda_mode == 'dynamic':
            # calculate loss ratio base on the number of node and edge
            batch_node = node_cls.shape[0]
            batch_edge = edge_cls.shape[0]
            self.cfg.training.lambda_node = 1
            self.cfg.training.lambda_edge = batch_edge / batch_node
            # print('lambda_edge:',self.cfg.training.lambda_edge)
            
        
        ''' 1. node class loss'''
        self.calc_node_loss(logs, node_cls, gt_cls, self.w_node_cls)
        
        ''' 2. edge class loss '''
        self.calc_edge_loss(logs, edge_cls, gt_rel, self.w_edge_cls)
        
        '''3. get metrics'''
        metrics = self.model.calculate_metrics(
            node_cls_pred=node_cls,
            node_cls_gt=gt_cls,
            edge_cls_pred=edge_cls,
            edge_cls_gt=gt_rel
        )
        for k,v in metrics.items():
            logs[k]=v
        # logs['acc_node_cls'] = metrics['acc_node_cls']
        # logs['acc_node_edge'] = metrics['acc_node_edge']
        # self.calc_node_metric(logs, node_cls, gt_cls)
        
        ''' eval tool '''
        if eval_tool is not None:
            node_cls = torch.softmax(node_cls.detach(),dim=1)
            edge_cls = torch.sigmoid(edge_cls.detach())
            eval_tool.add(scan_id, 
                          node_cls,gt_cls, 
                          edge_cls,gt_rel,
                          instance2mask,node_edges)
        return logs
        # return loss if eval_mode else loss['loss']

    def calc_node_loss(self, logs, node_cls_pred, node_cls_gt, weights=None):
        '''
        calculate node loss.
        can include
        classification loss
        attribute loss
        affordance loss
        '''
        # loss_obj = F.nll_loss(node_cls_pred, node_cls_gt, weight = weights)
        loss_obj = self.loss_node_cls(node_cls_pred, node_cls_gt)
        logs['loss'] += self.cfg.training.lambda_node * loss_obj
        logs['loss_obj'] = loss_obj
        
    def calc_edge_loss(self, logs, edge_cls_pred, edge_cls_gt, weights=None):
        if self.cfg.model.multi_rel:
            '''calculate'''
            loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
            # loss_rel = F.binary_cross_entropy(edge_cls_pred, edge_cls_gt, weight=weights)
        else:
            loss_rel = self.loss_rel_cls(edge_cls_pred,edge_cls_gt)
            # loss_rel = F.nll_loss(edge_cls_pred, edge_cls_gt, weight = weights)
        logs['loss'] += self.cfg.training.lambda_edge * loss_rel
        logs['loss_rel'] = loss_rel
    # def convert_metrics_to_log(self, metrics, eval_mode=False):
    #     tmp = dict()
    #     mode = 'train_' if eval_mode == False else 'valid_'
    #     for metric_name, dic in metrics.items():
    #         for sub, value in dic.items():
    #             tmp[metric_name+'/'+mode+sub] = value
    #     return tmp
    # def calc_node_metric(self, logs, node_cls_pred, node_cls_gt):
    #     cls_pred = node_cls_pred.detach()
    #     pred_cls = torch.max(cls_pred,1)[1]
    #     acc_cls = (node_cls_gt == pred_cls).sum().item() / node_cls_gt.nelement()
    #     logs['acc_node_cls'] = acc_cls
        
    
        
    def visualize(self,eval_tool=None):
        if eval_tool is None: eval_tool = self.eva_tool
        node_confusion_matrix, edge_confusion_matrix = eval_tool.draw(
            plot_text=False,
            grid=False,
            normalize='log',
            plot = False
            )
        return {
            'node_confusion_matrix': node_confusion_matrix,
            'edge_confusion_matrix': edge_confusion_matrix
        }