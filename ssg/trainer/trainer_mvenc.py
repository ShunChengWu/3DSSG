#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 08:57:11 2021

@author: sc
"""
import torch, os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from codeLib.models import BaseTrainer
import ssg
from ssg.utils.util_eva import EvaClassificationSimple
import codeLib.utils.moving_average as moving_average 
import time
from codeLib.common import check_weights, convert_torch_to_scalar
from models.otk.utils import normalize
from codeLib.torch.visualization import show_tensor_images, save_tensor_images
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from codeLib.common import denormalize_imagenet, create_folder
from codeLib.torch.visualization import  show_tensor_images, save_tensor_images
import logging
logger_py = logging.getLogger(__name__)

class Trainer_MVENC(BaseTrainer):
    def __init__(self,cfg, model, node_cls_names:list, device=None, **kwards):
        super().__init__(device)
        logger_py.setLevel(cfg.log_level)
        self.cfg=cfg
        self.model = model
        self.node_cls_names = node_cls_names
        self.w_node_cls = kwards.get('w_node_cls',None)
        self.input_is_roi = cfg.data.input_type == 'mv_roi'
        
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = ssg.config.get_optimizer(cfg, trainable_params)
        
        
        if self.w_node_cls is not None: 
            print('train with weighted node class.')
            self.w_node_cls = self.w_node_cls.to(self._device)
        self.loss_node_cls = torch.nn.CrossEntropyLoss(weight=self.w_node_cls)
        
        self.eva_tool = EvaClassificationSimple(self.node_cls_names)
        
        if cfg.model.image_encoder.method == 'gmu':
            self.loss_mu = torch.nn.CrossEntropyLoss()
        
    def get_log_metrics(self):
        output = dict()
        for k,v in self.eva_tool.get_mean_metrics().items():
            output[k+'_node_cls'] = v
        return output
        
    def zero_metrics(self):
        self.eva_tool.reset()
        
    def evaluate(self, val_loader, topk):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        it_dataset = val_loader.__iter__()
        eval_tool = EvaClassificationSimple(self.node_cls_names) 
        eval_list = defaultdict(moving_average.CMA)

        time.sleep(2)# Prevent possible deadlock during epoch transition
        for data in tqdm(it_dataset,leave=False):
            eval_step_dict = self.eval_step(data,eval_tool=eval_tool)

            for k, v in eval_step_dict.items():
                eval_list[k].update(v)
            # break
        
        eval_dict = dict()
        for k,v in eval_tool.get_mean_metrics().items():
            eval_dict[k+'_node_cls'] = v
        for k, v in eval_list.items():
            eval_dict[k] = v.avg
        # eval_dict = {k: v.avg for k, v in eval_list.items()}
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
        
        logs = convert_torch_to_scalar(logs)
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
        return eval_dict
    
    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        return dict(zip(data.keys(), self.toDevice(*data.values()) ))
        # return self.toDevice(*data.values())
        
        scan_id = data.get('scan_id')
        gt_cls = data.get('gt_cls')
        if not self.input_is_roi:
            images = data.get('images')
            image_boxes = data.get('image_boxes')
            flatten = (scan_id, gt_cls, images, image_boxes)
        else:
            images = data.get('roi_imgs')
            flatten = (scan_id, gt_cls, images)
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
        data = self.process_data_dict(data)
        gt_cls = data['gt_cls']
        
        # print('process scan',scan_id)
        
        # Shortcuts
        # device = self._device
        
        ''' make forward pass through the network '''
        if self.input_is_roi:
            outputs = self.model(images = data['roi_imgs'], 
                   bboxes = None,
                   return_meta=True)
        else: 
            outputs = self.model(images = data['images'], 
                   bboxes = data['image_boxes'],
                   return_meta=True)  
        node_cls = outputs['node_cls']
        
        ''' calculate loss '''
        logs['loss'] = 0
        
        ''' 1. node class loss'''
        self.calc_node_loss(logs, node_cls, gt_cls, self.w_node_cls)
        
        if self.cfg.model.image_encoder.method == 'cvr':
            self.calc_cvr_loss(logs,outputs)
        elif self.cfg.model.image_encoder.method == 'gmu':
            self.calc_mu_loss(logs,outputs)
            # self.cal_KLD_loss(logs, outputs)
        
        '''3. get metrics'''
        metrics = self.model.calculate_metrics(
            node_cls_pred=node_cls.detach(),
            node_cls_gt=gt_cls
        )
        for k,v in metrics.items():
            logs[k]=v
        
        ''' eval tool '''
        if eval_tool is not None:
            node_cls_ = torch.softmax(node_cls.detach(),dim=1)
            node_cls_ = torch.max(node_cls_,1)[1]
            assert node_cls_.shape == gt_cls.shape
            eval_tool(node_cls_,gt_cls)
        return logs
        # return loss if eval_mode else loss['loss']
        
    def calc_cvr_loss(self,logs, outputs):
        '''ensure feature separability'''
        vert = self.model.encoder.vert
        cos_sim = outputs['cos_sim']
        cos_sim2 = outputs['cos_sim2']
        pos = outputs['pos']
        cos_loss = cos_sim[torch.where(cos_sim>-1)].mean()
        cos_loss2 = cos_sim2[torch.where(cos_sim2>-1)].mean()
        pos_loss = torch.norm(normalize(vert)-normalize(pos),p=2,dim=-1).mean()
        logs['loss'] += 0.1*pos_loss
        logs['pos_loss'] = pos_loss
        logs['cos_loss'] = cos_loss
        logs['cos_loss2'] = cos_loss2
        
    def calc_mu_loss(self,logs, outputs):
        anchors = torch.arange(0,self.cfg.model.gmu.num_units).to(self._device)
        pos_loss = self.loss_mu(outputs['mu_pos'], anchors)
        
        logs['loss'] += 0.001*pos_loss
        logs['pos_loss'] = pos_loss
        
    def cal_KLD_loss(self,logs,outputs):
        '''KLD'''
        KLD = torch.mean( -0.5 * torch.sum(1 + outputs['logvar'] - outputs['mu']**2 -  outputs['logvar'].exp(),dim=1), dim=0)
        logs['loss'] += 0.1*KLD
        logs['kld_loss'] = -KLD

    def calc_node_loss(self, logs, node_cls_pred, node_cls_gt, weights=None):
        '''
        calculate node loss.
        can include
        - classification loss
        - attribute loss
        - affordance loss
        '''
        # loss_obj = F.nll_loss(node_cls_pred, node_cls_gt, weight = weights)
        loss_obj = self.loss_node_cls(node_cls_pred, node_cls_gt)
        logs['loss'] += self.cfg.training.lambda_node * loss_obj
        logs['loss_obj'] = loss_obj
        
    def visualize(self,eval_tool=None):
        if eval_tool is None: eval_tool = self.eva_tool
        node_confusion_matrix = eval_tool.draw(
            plot_text=False,
            grid=True,
            normalize='log',
            plot = False
            )
        return {
            'node_confusion_matrix': node_confusion_matrix,
        }
    
    def sample(self,val_loader):
        # show_tensor_images
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        it_dataset = val_loader.__iter__()
        # eval_tool = EvaClassificationSimple(self.node_cls_names) 
        # eval_list = defaultdict(moving_average.CMA)

        # time.sleep(2)# Prevent possible deadlock during epoch transition
        self.model.eval()
        
        # right_images=defaultdict(list)
        # wrong_images=defaultdict(list)
        # wrong_names=defaultdict(list)
        wrong_idx = defaultdict(int)
        for data in tqdm(it_dataset,leave=False):
            data = self.process_data_dict(data)
            
            
            if self.input_is_roi:
                outputs = self.model(images = data['roi_imgs'], 
                       bboxes = None,
                       return_meta=True)
                images = data['roi_imgs']
            else: 
                outputs = self.model(images = data['images'], 
                       bboxes = data['image_boxes'],
                       return_meta=True)  
                images = data['images']
            pd_cls = outputs['node_cls'].cpu()
            gt_cls = data.get('gt_cls',None).cpu()
            
            node_cls_pred = torch.max(pd_cls,1)[1]
            
            if gt_cls is not None:
                is_same = (gt_cls == node_cls_pred)
                
            if self.node_cls_names is not None:
                pd_cls_name = [self.node_cls_names[idx] for idx in node_cls_pred]
                gt_cls_name = [self.node_cls_names[idx] for idx in gt_cls]
            else:
                pd_cls_name = gt_cls_name = None
                
                
            for oid in range(len(gt_cls)):
                pd_name = pd_cls_name[oid]
                gt_name = gt_cls_name[oid]
                img = images[oid]
                img= denormalize_imagenet(img)
                img= torch.clamp((img*255).byte(),0,255).byte().cpu()
                
                
                if is_same[oid]:
                    pass
                    # right_images[gt_name].append(images[oid].cpu())
                else:
                    output_folder_wrong = os.path.join(self.cfg.training.out_dir,self.cfg.name,'samples','wrong',gt_name)
                    create_folder(output_folder_wrong)
                    save_tensor_images(img,output_folder_wrong+'/'+pd_name+str(wrong_idx[gt_name])+'.png')
                    wrong_idx[gt_name]+=1
                    # wrong_images[gt_name].append(images[oid].cpu())
                    # wrong_names[gt_name].append(pd_name)
            # if len(wrong_names) > 3:break
        
        # for classname in self.node_cls_names:
        #     if classname in right_images:
        #         img_list = right_images[classname]
                
        #     if classname in wrong_images:
        #         img_list = wrong_images[classname]
        #         name_list = wrong_names[classname]
                
        #         for idx in range(len(img_list)):
        #             img = img_list[idx]
        #             name = name_list[idx]
                
        #             img= denormalize_imagenet(img)
        #             img= torch.clamp((img*255).byte(),0,255).byte()
        #             # grid_images = torchvision.utils.make_grid(img)
                    
        #             output_folder_wrong = os.path.join(self.cfg.training.out_dir,self.cfg.name,'samples','wrong',classname)
        #             create_folder(output_folder_wrong)
                    
        #             save_tensor_images(img,output_folder_wrong+'/'+name+str(idx)+'.png')