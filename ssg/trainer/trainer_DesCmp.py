import os
import numpy as np
from tqdm import tqdm, tnrange
from collections import defaultdict
from codeLib.models import BaseTrainer
from codeLib.common import check_weights, check_valid, convert_torch_to_scalar
import torch
import torch.nn.functional as F
from ssg2d.utils.util_eva import EvalSceneGraph, plot_confusion_matrix
import time
import logging
from ssg2d.checkpoints import CheckpointIO
import torch.optim as optim
import codeLib.utils.moving_average as moving_average 
from codeLib.models import BaseTrainer
from codeLib.common import check_weights, check_valid, convert_torch_to_scalar
import torch

class Trainer_DCMP(BaseTrainer):
    def __init__(self, cfg, model, node_cls_names, edge_cls_names,
                 optimizer=None, device=None, vis_dir=None, 
                 **kwargs):
        super().__init__(device)
        # self._device=device if device is not None else 'cpu'
        self.cfg = cfg
        self.model = model.to(self._device)
        self.optimizer = optimizer
        self.w_edge_cls = kwargs.get('w_edge_cls',None)
        self.edge_cls_names = edge_cls_names#kwargs['edge_cls_names']
        # self.eva_tool = EvalSceneGraph(self.node_cls_names, self.edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=0) # do not calculate topK in training mode        
        if self.cfg.model.multi_rel:
            self.loss_rel_cls = torch.nn.BCEWithLogitsLoss(pos_weight=self.w_edge_cls)
        else:
            self.loss_rel_cls = torch.nn.CrossEntropyLoss(weight=self.w_edge_cls)
            
    def zero_metrics(self):
        self.eva_tool.reset()

    def evaluate(self, val_loader, topk):
        it_dataset = val_loader.__iter__()
        # eval_tool = EvalSceneGraph(self.node_cls_names, self.edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=topk,save_prediction=True) 
        eval_list = defaultdict(moving_average.CMA)

        time.sleep(2)# Prevent possible deadlock during epoch transition
        for data in tqdm(it_dataset,leave=False):
            eval_step_dict = self.eval_step(data,eval_tool=None)

            for k, v in eval_step_dict.items():
                eval_list[k].update(v)

        eval_dict = {k: v.avg for k, v in eval_list.items()}
        vis = self.visualize(eval_tool=None)
        eval_dict['visualization'] = vis
        del it_dataset
        return eval_dict, None
        
    def train_step(self, data, it=None):
        self.model.train()
        self.optimizer.zero_grad()
        logs = self.compute_loss(data, it=it, eval_tool = None)
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
        descriptor = data.get('descriptor')
        node_edges = data.get('node_edges')
        instance2mask = data.get('instance2mask')        
        flatten = (scan_id, gt_rel, descriptor, node_edges, instance2mask)
        flatten = self.toDevice(*flatten)
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
        (scan_id, gt_rel, descriptor, node_edges, instance2mask) = self.process_data_dict(data)
        
        # Shortcuts
        # device = self._device
        
        ''' make forward pass through the network '''
        edge_cls_sgfn, edge_cls_ssg2d = self.model(
            edges=node_edges.t().contiguous(),
            descriptor=descriptor,
        )
        
        
        ''' calculate loss '''
        logs['loss'] = 0
        
        ''' 2. edge class loss '''
        self.calc_edge_loss(logs, edge_cls_sgfn, edge_cls_ssg2d, gt_rel, self.w_edge_cls)
        
        '''3. get metrics'''
        metrics = self.model.calculate_metrics(
            edge_cls_pred_sgfn=edge_cls_sgfn,
            edge_cls_pred_ssg2d=edge_cls_ssg2d,
            edge_cls_gt=gt_rel
        )
        for k,v in metrics.items():
            logs[k]=v
            
        ''' eval tool '''
        # if eval_tool is not None:
        #     node_cls = torch.softmax(node_cls.detach(),dim=1)
        #     edge_cls = torch.sigmoid(edge_cls.detach())
        #     eval_tool.add(scan_id, 
        #                   node_cls,gt_cls, 
        #                   edge_cls,gt_rel,
        #                   instance2mask,node_edges)
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
        
    def calc_edge_loss(self, logs, edge_cls_sgfn, edge_cls_ssg2d, edge_cls_gt, weights=None):
        loss_rel_sgfn = self.loss_rel_cls(edge_cls_sgfn, edge_cls_gt)
        loss_rel_ssg2d = self.loss_rel_cls(edge_cls_ssg2d, edge_cls_gt)
        # if self.cfg.model.multi_rel:
            
        #     # loss_rel = F.binary_cross_entropy(edge_cls_pred, edge_cls_gt, weight=weights)
        # else:
        #     loss_rel = self.loss_rel_cls(edge_cls_pred,edge_cls_gt)
        #     # loss_rel = F.nll_loss(edge_cls_pred, edge_cls_gt, weight = weights)
        logs['loss'] += self.cfg.training.lambda_edge * loss_rel_sgfn
        logs['loss'] += self.cfg.training.lambda_edge * loss_rel_ssg2d
        logs['loss_rel_sgfn'] = loss_rel_sgfn
        logs['loss_rel_ssg2d'] = loss_rel_ssg2d
        
    def visualize(self,eval_tool=None):
        return {}
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
    
    def toDevice(self, *args):
        output = list()
        for item in args:
            if isinstance(item,  torch.Tensor):
                output.append(item.to(self._device))
            elif isinstance(item,  dict):
                ks = item.keys()
                vs = self.toDevice(*item.values())
                item = dict(zip(ks, vs))
                output.append(item)
            elif isinstance(item, list):
                output.append(self.toDevice(*item))
            else:
                output.append(item)
        return output