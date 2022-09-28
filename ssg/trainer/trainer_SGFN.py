import os,copy
import numpy as np
from tqdm import tqdm, tnrange
from collections import defaultdict
from codeLib.models import BaseTrainer
from codeLib.common import check_weights, check_valid, convert_torch_to_scalar
import torch
import torch.nn.functional as F
from ssg.utils.util_eva import EvalSceneGraph, EvalUpperBound, plot_confusion_matrix
import time
import logging
from ssg.checkpoints import CheckpointIO
import torch.optim as optim
import codeLib.utils.moving_average as moving_average 
from codeLib.models import BaseTrainer
from codeLib.common import check_weights, check_valid, convert_torch_to_scalar
import torch
import torchvision
import torch, os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from codeLib.models import BaseTrainer
import ssg
# from ssg.utils.util_eva import EvaClassificationSimple
import codeLib.utils.moving_average as moving_average 
import time
from codeLib.common import check_weights, convert_torch_to_scalar
# from models.otk.utils import normalize
from codeLib.torch.visualization import show_tensor_images, save_tensor_images
# import torchvision
import matplotlib.pyplot as plt
import numpy as np
from codeLib.common import denormalize_imagenet, create_folder
from codeLib.torch.visualization import  show_tensor_images, save_tensor_images
import codeLib.utils.string_numpy as snp
import logging
from ssg.utils.util_data import match_class_info_from_two, merge_batch_mask2inst
from ssg import define
from ssg.data.collate import graph_collate#, batch_graph_collate

import graphviz
import operator
from ssg.utils import util_label, util_merge_same_part
from ssg.utils.util_data import raw_to_data, cvt_all_to_dict_from_h5
from codeLib.common import rgb_2_hex
from codeLib.common import rand_24_bit, color_rgb
from ssg.utils.graph_vis import DrawSceneGraph
from ssg.trainer.eval_inst import EvalInst

logger_py = logging.getLogger(__name__)

class Trainer_SGFN(BaseTrainer, EvalInst):
    def __init__(self, cfg, model, node_cls_names:list, edge_cls_names:list,
                 device=None,  **kwargs):
        super().__init__(device)
        logger_py.setLevel(cfg.log_level)
        self.cfg = cfg
        self.model = model#.to(self._device)
        # self.optimizer = optimizer
        self.w_node_cls = kwargs.get('w_node_cls',None)
        self.w_edge_cls = kwargs.get('w_edge_cls',None)
        self.node_cls_names = node_cls_names#kwargs['node_cls_names']
        self.edge_cls_names = edge_cls_names#kwargs['edge_cls_names']
        
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = ssg.config.get_optimizer(cfg, trainable_params)
        
        if self.w_node_cls is not None: 
            logger_py.info('train with weighted node class.')
            self.w_node_cls = self.w_node_cls.to(self._device)
        if self.w_edge_cls is not None: 
            logger_py.info('train with weighted node class.')
            self.w_edge_cls= self.w_edge_cls.to(self._device)
        
        
        self.eva_tool = EvalSceneGraph(self.node_cls_names, self.edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=0,none_name=define.NAME_NONE) # do not calculate topK in training mode        
        self.loss_node_cls = torch.nn.CrossEntropyLoss(weight=self.w_node_cls)
        if self.cfg.model.multi_rel:
            self.loss_rel_cls = torch.nn.BCEWithLogitsLoss(pos_weight=self.w_edge_cls)
        else:
            self.loss_rel_cls = torch.nn.CrossEntropyLoss(weight=self.w_edge_cls)

    def zero_metrics(self): 
        self.eva_tool.reset()
        
    def evaluate(self, val_loader, topk):
        it_dataset = val_loader.__iter__()
        eval_tool = EvalSceneGraph(self.node_cls_names, self.edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=topk,save_prediction=True,
                                   none_name=define.NAME_NONE) 
        eval_list = defaultdict(moving_average.MA)

        time.sleep(2)# Prevent possible deadlock during epoch transition
        for data in tqdm(it_dataset,leave=False):
            eval_step_dict = self.eval_step(data,eval_tool=eval_tool)
            
            for k, v in eval_step_dict.items():
                eval_list[k].update(v)
            # break
        eval_dict = dict()
        obj_, edge_ = eval_tool.get_mean_metrics()
        for k,v in obj_.items():
            # print(k)
            eval_dict[k+'_node_cls'] = v
        for k,v in edge_.items():
            # print(k)
            eval_dict[k+'_edge_cls'] = v
        
        for k, v in eval_list.items():
            eval_dict[k] = v.avg
        
        vis = self.visualize(eval_tool=eval_tool)
        eval_dict['visualization'] = vis
        del it_dataset
        return eval_dict, eval_tool
    
    def sample(self, dataloader):
        pass
        
    def train_step(self, data, it=None):
        self.model.train()
        self.optimizer.zero_grad()
        logs = self.compute_loss(data, it=it, eval_tool = self.eva_tool)
        if 'loss' not in logs: return logs
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
    
    # def process_data_dict(self, data):
    #     ''' Processes the data dictionary and returns respective tensors

    #     Args:
    #         data (dictionary): data dictionary
    #     '''
    #     data =  dict(zip(data.keys(), self.toDevice(*data.values()) ))
    #     return data
        # scan_id = data.get('scan_id')
        # gt_rel = data.get('gt_rel')
        # gt_cls = data.get('gt_cls')
        # obj_points = data.get('obj_points')
        # descriptor = data.get('descriptor')
        # node_edges = data.get('node_edges')
        # instance2mask = data.get('instance2mask')
        
        # # print(gt_rel.shape)
        # # self.toDevice(*(gt_rel))
        
        # flatten = (scan_id, gt_cls, gt_rel, obj_points, descriptor, node_edges, instance2mask)
        # flatten = self.toDevice(*flatten)
        # # check_valid(*flatten)
        # return flatten
    
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
        data = data.to(self._device)
        # data = self.process_data_dict(data)
        
        # Shortcuts
        scan_id = data['scan_id']
        # gt_cls = data['gt_cls']
        # gt_rel = data['gt_rel']
        # mask2instance = data['mask2instance']
        # node_edges_ori = data['node_edges']
        # data['node_edges'] = data['node_edges'].t().contiguous()
        
        # check input valid
        # if node_edges_ori.ndim==1:
        #     return {}
        
        # print('')
        # print('gt_rel.sum():',gt_rel.sum())
        
        ''' make forward pass through the network '''
        node_cls, edge_cls = self.model(data)
        
        
        ''' calculate loss '''
        logs['loss'] = 0
        
        
        if self.cfg.training.lambda_mode == 'dynamic':
            # calculate loss ratio base on the number of node and edge
            batch_node = node_cls.shape[0]
            batch_edge = edge_cls.shape[0]
            self.cfg.training.lambda_node = 1
            self.cfg.training.lambda_edge = batch_edge / batch_node
            
        
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
            
        ''' eval tool '''
        if eval_tool is not None:
            node_cls = torch.softmax(node_cls.detach(),dim=1)
            edge_cls = torch.sigmoid(edge_cls.detach())
            eval_tool.add(scan_id, 
                          node_cls,gt_cls, 
                          edge_cls,gt_rel,
                          mask2instance,node_edges_ori)
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
            # batch_mean = torch.sum(edge_cls_gt, dim=(0))
            # zeros = (edge_cls_gt ==0).sum().unsqueeze(0)
            # batch_mean = torch.cat([zeros,batch_mean],dim=0)
            # weight = torch.abs(1.0 / (torch.log(batch_mean+1)+1)) # +1 to prevent 1 /log(1) = inf                
            # weight[torch.where(weight==0)] = weight[0].clone()
            # weight = weight[1:]
            # pass
            loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
        else:
            loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
            # loss_rel = F.binary_cross_entropy(edge_cls_pred, edge_cls_gt, weight=weights)
        # else:
            # loss_rel = self.loss_rel_cls(edge_cls_pred,edge_cls_gt)
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
    
    def get_log_metrics(self):
        output = dict()
        obj_, edge_ = self.eva_tool.get_mean_metrics()
        
        for k,v in obj_.items():
            output[k+'_node_cls'] = v
        for k,v in edge_.items():
            output[k+'_edge_cls'] = v
        return output
    
    def evaluate_inst_incre(self,dataset_seg,topk):
        ignore_missing=self.cfg.eval.ignore_missing
        '''add a none class for missing instances'''
        node_cls_names = copy.copy(self.node_cls_names)
        edge_cls_names = copy.copy(self.edge_cls_names)
        if define.NAME_NONE not in self.node_cls_names:
            node_cls_names.append(define.NAME_NONE)
        if define.NAME_NONE not in self.edge_cls_names:
            edge_cls_names.append(define.NAME_NONE)
        # remove same part
        # if define.NAME_SAME_PART in edge_cls_names: edge_cls_names.remove(define.NAME_SAME_PART)
        
        noneidx_node_cls = node_cls_names.index(define.NAME_NONE)
        noneidx_edge_cls = edge_cls_names.index(define.NAME_NONE)
        
        '''
        Find index mapping. Ignore NONE for nodes since it is used for mapping missing instance.
        Ignore SAME_PART for edges.
        '''
        seg_valid_node_cls_indices = []
        inst_valid_node_cls_indices = []
        for idx in range(len(self.node_cls_names)):
            name = self.node_cls_names[idx]
            if name == define.NAME_NONE: continue
            seg_valid_node_cls_indices.append(idx)
        for idx in range(len(node_cls_names)):
            name = node_cls_names[idx]
            if name == define.NAME_NONE: continue
            inst_valid_node_cls_indices.append(idx)
        
        seg_valid_edge_cls_indices = []
        inst_valid_edge_cls_indices = []
        for idx in range(len(self.edge_cls_names)):
            name = self.edge_cls_names[idx]
            # if name == define.NAME_SAME_PART: continue
            seg_valid_edge_cls_indices.append(idx)
        for idx in range(len(edge_cls_names)):
            name = edge_cls_names[idx]
            # if name == define.NAME_SAME_PART: continue
            inst_valid_edge_cls_indices.append(idx)
        
        
        eval_tool = EvalSceneGraph(node_cls_names, edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=topk,save_prediction=True,
                                   none_name=define.NAME_NONE) 
        eval_list = defaultdict(moving_average.MA)
        
        '''check'''
        #length
        # print('len(dataset_seg), len(dataset_inst):',len(dataset_seg), len(dataset_inst))
        print('ignore missing',ignore_missing)
        #classes
        
        
        ''' get scan_idx mapping '''
        scanid2idx_seg = dict()
        for index in range(len(dataset_seg)):
            scan_id = snp.unpack(dataset_seg.scans,index)# self.scans[idx]
            scanid2idx_seg[scan_id] = index
            
        # scanid2idx_inst = dict()
        # for index in range(len(dataset_inst)):
        #     scan_id = snp.unpack(dataset_inst.scans,index)# self.scans[idx]
        #     scanid2idx_inst[scan_id] = index
            
        
        # '''build main seg loader'''
        # seg_dataloader = torch.utils.data.DataLoader(
        #     dataset_seg, batch_size=1, num_workers=0,
        #     shuffle=False, drop_last=False,
        #     pin_memory=True,
        #     collate_fn=graph_collate,
        # )
        
        '''start eval'''
        acc_time = 0
        timer_counter=0
        self.model.eval()
        for index in tqdm(range(len(dataset_seg))):
        # for data_inst in seg_dataloader:
            # data = dataset_seg.__getitem__(index)                
            # scan_id_inst = data['scan_id'][0]
            # if scan_id_inst not in scanid2idx_seg: continue
            scan_id = '4acaebcc-6c10-2a2a-858b-29c7e4fb410d'
            index = scanid2idx_seg[scan_id]
            data_seg  = dataset_seg.__getitem__(index)
            
            # data_seg = graph_collate(data_seg)
            
            # if isinstance(data_seg,list):
            #     assert data_seg[0]['scan_id'] == data_inst['scan_id']
            # else:
            #     if isinstance(data_seg['scan_id'],list):
            #         for scan_id in data_seg['scan_id']:
            #             assert scan_id == data_inst['scan_id']
            #         # data_seg['scan_id'] = data_seg['scan_id'][0]
            #     else:
            #         assert data_seg['scan_id'] == data_inst['scan_id']
            
            
            '''process seg'''
            eval_dict={}
            with torch.no_grad():
                # logs = {}
                # data_inst = self.process_data_dict(data_inst)
                # Process data dictionary
                batch_data = data_seg
                if len(batch_data)==0:continue
                
                nodes_w=defaultdict(int)
                edges_w=defaultdict(int)
                nodes_pds_all = dict()
                edges_pds_all = dict()
                
                def fuse(old:dict,w_old:dict,new:dict):
                    for k,v in new.items():
                        if k in old:
                            old[k] = (old[k]*w_old[k]+new[k]) / (w_old[k]+1)
                            w_old[k]+=1
                        else:
                            old[k] = new[k]
                            w_old[k] = 1
                    return old,w_old
                def process(data):
                    data = self.process_data_dict(data)
                    # Shortcuts
                    scan_id = data['scan_id']
                    # gt_cls = data['gt_cls']
                    gt_rel = data['gt_rel']
                    mask2seg = data['mask2instance']
                    node_edges_ori = data['node_edges']
                    data['node_edges'] = data['node_edges'].t().contiguous()
                    seg2inst = data['seg2inst']
                    
                    # check input valid
                    if node_edges_ori.ndim==1: return {},{}, -1
                    
                    ''' make forward pass through the network '''
                    tick = time.time()
                    node_cls, edge_cls = self.model(**data)
                    tock = time.time()
                    
                    '''collect predictions on inst and edge pair'''
                    node_pds = dict()
                    edge_pds = dict()
                    
                    '''merge prediction from seg to instance (in case of "same part")'''
                    inst2masks = defaultdict(list)          
                    mask2seg= merge_batch_mask2inst(mask2seg) 
                    tmp_dict = defaultdict(list)
                    for mask, seg in mask2seg.items():
                        inst = seg2inst[seg]
                        inst2masks[inst].append(mask)
                        
                        tmp_dict[seg].append(node_cls[mask])
                    for seg, l in tmp_dict.items():
                        if seg in node_pds: 
                            raise RuntimeError()
                        pd =  torch.stack(l,dim=0)
                        pd = torch.softmax(pd,dim=1).mean(dim=0)
                        node_pds[seg] = pd
                        
                    tmp_dict = defaultdict(list)
                    for idx in range(len(node_edges_ori)):
                        src_idx,tgt_idx = data['node_edges'][0,idx].item(),data['node_edges'][1,idx].item()
                        seg_src,seg_tgt = mask2seg[src_idx],mask2seg[tgt_idx]
                        # inst_src,inst_tgt = seg2inst[seg_src],seg2inst[seg_tgt]
                        key = (seg_src,seg_tgt)
                        
                        
                        tmp_dict[key].append(edge_cls[idx])
                        
                    for key, l in tmp_dict.items():
                        if key in edge_pds: 
                            raise RuntimeError()
                        pd =  torch.stack(l,dim=0)
                        pd = torch.softmax(pd,dim=1).mean(0)
                        edge_pds[key] = pd
                        # src_inst_idx, tgt_inst_idx = inst_mask2inst[src_idx], inst_mask2inst[tgt_idx]
                        # inst_gt_pairs.add((src_inst_idx, tgt_inst_idx))
                    
                    return node_pds, edge_pds, tock-tick
                
                if isinstance(batch_data,list):
                    for idx,data in enumerate(batch_data):
                        print(idx)
                        node_pds, edge_pds, pt = process(data)
                        if pt>0:
                            acc_time += pt
                            timer_counter+=1
                        
                        fuse(nodes_pds_all,nodes_w,node_pds)
                        fuse(edges_pds_all,edges_w,edge_pds)
                        pass
                else:
                    data = batch_data
                    node_pds, edge_pds, pt = process(data)
                    if pt>0:
                        acc_time += pt
                        timer_counter+=1
                    
                    fuse(nodes_pds_all,nodes_w,node_pds)
                    fuse(edges_pds_all,edges_w,edge_pds)
                
                '''generate gt'''
                if isinstance(batch_data,list):
                    data_inst = self.process_data_dict(batch_data[-1]) # the last one is the complete one
                else:
                    data_inst = self.process_data_dict(batch_data)
                    
                inst_mask2instance = data_inst['mask2instance']
                inst_gt_cls = data_inst['gt_cls']
                inst_gt_rel = data_inst['gt_rel']
                data_inst['node_edges'] = data_inst['node_edges'].t().contiguous()
                
                '''merge nodes'''
                merged_mask2instance = dict()
                merged_instance2idx = dict()
                merged_node_cls = torch.zeros(len(inst_mask2instance),len(node_cls_names)).to(self.cfg.DEVICE)
                merged_node_cls_gt = (torch.ones(len(inst_mask2instance),dtype=torch.long) * noneidx_node_cls).to(self.cfg.DEVICE)
                counter=0
                for mask_new, (mask_old, inst)in enumerate(inst_mask2instance.items()):                    
                    # merge predictions
                    if not ignore_missing:
                        merged_instance2idx[inst]=mask_new
                        merged_mask2instance[mask_new] = inst
                        merged_node_cls_gt[mask_new] = inst_gt_cls[mask_old] # use GT class
                        if inst in nodes_pds_all:
                            '''merge nodes'''
                            node_cls_pred = nodes_pds_all[inst]#torch.softmax(nodes_pds_all[inst].view(1,-1),dim=1).mean(dim=0)
                            # node_cls_pred = torch.softmax(nodes_pds_all[inst])
                            # predictions = node_cls[inst2masks[inst]]# get all predictions on that instance
                            # node_cls_pred = torch.softmax(predictions, dim=1).mean(dim=0)# averaging the probability
                            merged_node_cls[mask_new][inst_valid_node_cls_indices] = node_cls_pred[seg_valid_node_cls_indices] # assign and ignor
                            
                        else:
                            #TODO: handle missing inst
                            merged_node_cls[mask_new][noneidx_node_cls] = 1.0
                            # merged_node_cls_gt[mask_new]=noneidx_node_cls
                            
                            
                            pass
                    else:
                        if inst not in nodes_pds_all:
                            continue
                        merged_mask2instance[counter] = inst
                        merged_instance2idx[inst]=counter
                        # predictions = node_cls[inst2masks[inst]]
                        # node_cls_pred = torch.softmax(predictions, dim=1).mean(dim=0)                        
                        node_cls_pred = torch.softmax(nodes_pds_all[inst].view(1,-1),dim=1).mean(dim=0)
                        merged_node_cls[counter][inst_valid_node_cls_indices] = node_cls_pred[seg_valid_node_cls_indices]
                        merged_node_cls_gt[counter] = inst_gt_cls[mask_old]
                        counter+=1
                if ignore_missing:
                    merged_node_cls = merged_node_cls[:counter]
                    merged_node_cls_gt = merged_node_cls_gt[:counter]
                '''merge '''
                # mask2seg= merge_batch_mask2inst(mask2seg) 
                inst_mask2inst=merge_batch_mask2inst(inst_mask2instance)
                
                # build search list for GT edge pairs
                inst_gt_pairs = set()
                inst_gt_rel_dict = dict() # gt instance pair -> predicate label
                for idx in range(len(inst_gt_rel)):
                    src_idx, tgt_idx = data_inst['node_edges'][0,idx].item(),data_inst['node_edges'][1,idx].item()
                    src_inst_idx, tgt_inst_idx = inst_mask2inst[src_idx], inst_mask2inst[tgt_idx]
                    inst_gt_pairs.add((src_inst_idx, tgt_inst_idx))
                    inst_gt_rel_dict[(src_inst_idx, tgt_inst_idx)] = inst_gt_rel[idx]
               
                # merged_edge_cls_dict = defaultdict(list) # map edge predictions on the same pair of instances.
                # for idx in range(len(gt_rel)):
                #     src_idx, tgt_idx = data['node_edges'][0,idx].item(), data['node_edges'][1,idx].item()
                #     relname = self.edge_cls_names[gt_rel[idx].item()]
                    
                #     src_seg_idx = mask2seg[src_idx]
                #     src_inst_idx = seg2inst[src_seg_idx]
                    
                #     tgt_seg_idx = mask2seg[tgt_idx]
                #     tgt_inst_idx = seg2inst[tgt_seg_idx]
                #     pair = (src_inst_idx,tgt_inst_idx)
                #     if  pair in inst_gt_pairs:
                #         merged_edge_cls_dict[pair].append( edge_cls[idx] )
                #     else:
                #         # print('cannot find seg:{}(inst:{}) to seg:{}(inst:{}) with relationship:{}.'.format(src_seg_idx,src_inst_idx,tgt_seg_idx,tgt_inst_idx,relname))
                #         pass
                # check missing rels
                # for pair in inst_gt_rel_dict:
                #     if pair not in merged_edge_cls_dict:
                #         relname = self.edge_cls_names[inst_gt_rel_dict[pair]]
                #         src_name = self.node_cls_names[inst_gt_cls[inst_instance2mask[pair[0]]]]
                #         tgt_name = self.node_cls_names[inst_gt_cls[inst_instance2mask[pair[1]]]]
                #         print('missing inst:{}({}) to inst:{}({}) with relationship:{}'.format(pair[0],src_name,pair[1],tgt_name,relname))
                        
                '''merge predictions'''
                merged_edge_cls = torch.zeros(len(inst_gt_rel),len(edge_cls_names)).to(self.cfg.DEVICE)
                merged_edge_cls_gt = (torch.ones(len(inst_gt_rel),dtype=torch.long) * noneidx_edge_cls).to(self.cfg.DEVICE)
                merged_node_edges = list() # new edge_indices
                counter=0
                for idx, (pair,inst_edge_cls) in enumerate(inst_gt_rel_dict.items()):
                    if ignore_missing:
                        if pair[0] not in merged_instance2idx: continue
                        if pair[1] not in merged_instance2idx: continue
                    # merge edge index to the new mask ids
                    mask_src = merged_instance2idx[pair[0]]
                    mask_tgt = merged_instance2idx[pair[1]]
                    merged_node_edges.append([mask_src,mask_tgt])
                    
                    if pair in edges_pds_all:
                        edge_pds = edges_pds_all[pair]#edges_pds_all[pair].view(1,-1)
                        #edge_pds = edge_pds[:,seg_valid_edge_cls_indices]
                        # seg_valid_edge_cls_indices
                        
                        #ignore same part
                        
                        # edge_pds = torch.softmax(edge_pds,dim=1).mean(0)
                        merged_edge_cls[counter][inst_valid_edge_cls_indices] = edge_pds[seg_valid_edge_cls_indices]
                    else:
                        continue
                        merged_edge_cls[counter][noneidx_edge_cls] = 1.0
                        
                    merged_edge_cls_gt[counter] = inst_edge_cls
                    counter+=1
                if ignore_missing:
                    merged_edge_cls=merged_edge_cls[:counter]
                    merged_edge_cls_gt = merged_edge_cls_gt[:counter]
                merged_node_edges = torch.tensor(merged_node_edges,dtype=torch.long)

            eval_tool.add([scan_id], 
                          merged_node_cls,
                          merged_node_cls_gt, 
                          merged_edge_cls,
                          merged_edge_cls_gt,
                          [merged_mask2instance],
                          merged_node_edges)
            if index > 10: break
            # break
        print('time:',acc_time,timer_counter,acc_time/timer_counter)
        
        eval_dict = dict()
        obj_, edge_ = eval_tool.get_mean_metrics()
        for k,v in obj_.items():
            # print(k)
            eval_dict[k+'_node_cls'] = v
        for k,v in edge_.items():
            # print(k)
            eval_dict[k+'_edge_cls'] = v
        
        for k, v in eval_list.items():
            eval_dict[k] = v.avg
        
        vis = self.visualize(eval_tool=eval_tool)
        eval_dict['visualization'] = vis
        return eval_dict, eval_tool
    
    def visualize_inst_incre(self,dataset_seg,topk):
        ignore_missing=self.cfg.eval.ignore_missing
        '''add a none class for missing instances'''
        node_cls_names = copy.copy(self.node_cls_names)
        edge_cls_names = copy.copy(self.edge_cls_names)
        if define.NAME_NONE not in self.node_cls_names:
            node_cls_names.append(define.NAME_NONE)
        if define.NAME_NONE not in self.edge_cls_names:
            edge_cls_names.append(define.NAME_NONE)
        # remove same part
        # if define.NAME_SAME_PART in edge_cls_names: edge_cls_names.remove(define.NAME_SAME_PART)
        
        noneidx_node_cls = node_cls_names.index(define.NAME_NONE)
        noneidx_edge_cls = edge_cls_names.index(define.NAME_NONE)
        
        '''
        Find index mapping. Ignore NONE for nodes since it is used for mapping missing instance.
        Ignore SAME_PART for edges.
        '''
        seg_valid_node_cls_indices = []
        inst_valid_node_cls_indices = []
        for idx in range(len(self.node_cls_names)):
            name = self.node_cls_names[idx]
            if name == define.NAME_NONE: continue
            seg_valid_node_cls_indices.append(idx)
        for idx in range(len(node_cls_names)):
            name = node_cls_names[idx]
            if name == define.NAME_NONE: continue
            inst_valid_node_cls_indices.append(idx)
        
        seg_valid_edge_cls_indices = []
        inst_valid_edge_cls_indices = []
        for idx in range(len(self.edge_cls_names)):
            name = self.edge_cls_names[idx]
            # if name == define.NAME_SAME_PART: continue
            seg_valid_edge_cls_indices.append(idx)
        for idx in range(len(edge_cls_names)):
            name = edge_cls_names[idx]
            # if name == define.NAME_SAME_PART: continue
            inst_valid_edge_cls_indices.append(idx)
        
        
        eval_tool = EvalSceneGraph(node_cls_names, edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=topk,save_prediction=True,
                                   none_name=define.NAME_NONE) 
        eval_list = defaultdict(moving_average.MA)
        
        '''check'''
        #length
        # print('len(dataset_seg), len(dataset_inst):',len(dataset_seg), len(dataset_inst))
        print('ignore missing',ignore_missing)
        #classes
        
        
        ''' get scan_idx mapping '''
        scanid2idx_seg = dict()
        for index in range(len(dataset_seg)):
            scan_id = snp.unpack(dataset_seg.scans,index)# self.scans[idx]
            scanid2idx_seg[scan_id] = index

        '''start eval'''
        acc_time = 0
        timer_counter=0
        self.model.eval()
        for index in tqdm(range(len(dataset_seg))):
        # for data_inst in seg_dataloader:
            # data = dataset_seg.__getitem__(index)                
            # scan_id_inst = data['scan_id'][0]
            # if scan_id_inst not in scanid2idx_seg: continue
            scan_id = '4acaebcc-6c10-2a2a-858b-29c7e4fb410d'
            index = scanid2idx_seg[scan_id]
            data_seg  = dataset_seg.__getitem__(index)

            '''process seg'''
            eval_dict={}
            with torch.no_grad():
                # logs = {}
                # data_inst = self.process_data_dict(data_inst)
                # Process data dictionary
                batch_data = data_seg
                if len(batch_data)==0:continue
                
                '''generate gt'''
                if isinstance(batch_data,list):
                    data_inst = self.process_data_dict(batch_data[-1]) # the last one is the complete one
                else:
                    data_inst = self.process_data_dict(batch_data)
                
                scan_id = data_inst['scan_id']
                graphDrawer = DrawSceneGraph(scan_id,self.node_cls_names, self.edge_cls_names,debug=True)
                nodes_w=defaultdict(int)
                edges_w=defaultdict(int)
                nodes_pds_all = dict()
                edges_pds_all = dict()
                
                def fuse(old:dict,w_old:dict,new:dict):
                    for k,v in new.items():
                        if k in old:
                            old[k] = (old[k]*w_old[k]+new[k]) / (w_old[k]+1)
                            w_old[k]+=1
                        else:
                            old[k] = new[k]
                            w_old[k] = 1
                    return old,w_old
                def process(data):
                    data = self.process_data_dict(data)
                    # Shortcuts
                    scan_id = data['scan_id']
                    # gt_cls = data['gt_cls']
                    # gt_rel = data['gt_rel']
                    mask2seg = data['mask2instance']
                    node_edges_ori = data['node_edges']
                    data['node_edges'] = data['node_edges'].t().contiguous()
                    # seg2inst = data['seg2inst']
                    
                    # check input valid
                    if node_edges_ori.ndim==1: return {},{}, -1
                    
                    ''' make forward pass through the network '''
                    tick = time.time()
                    node_cls, edge_cls = self.model(**data)
                    tock = time.time()
                    
                    '''collect predictions on inst and edge pair'''
                    node_pds = dict()
                    edge_pds = dict()
                    
                    '''merge prediction from seg to instance (in case of "same part")'''
                    # inst2masks = defaultdict(list)          
                    mask2seg= merge_batch_mask2inst(mask2seg) 
                    tmp_dict = defaultdict(list)
                    for mask, seg in mask2seg.items():
                        # inst = seg2inst[seg]
                        # inst2masks[inst].append(mask)
                        
                        tmp_dict[seg].append(node_cls[mask])
                    for seg, l in tmp_dict.items():
                        if seg in node_pds: 
                            raise RuntimeError()
                        pd =  torch.stack(l,dim=0)
                        pd = torch.softmax(pd,dim=1).mean(dim=0)
                        node_pds[seg] = pd
                        
                    tmp_dict = defaultdict(list)
                    for idx in range(len(node_edges_ori)):
                        src_idx,tgt_idx = data['node_edges'][0,idx].item(),data['node_edges'][1,idx].item()
                        seg_src,seg_tgt = mask2seg[src_idx],mask2seg[tgt_idx]
                        # inst_src,inst_tgt = seg2inst[seg_src],seg2inst[seg_tgt]
                        key = (seg_src,seg_tgt)
                        
                        
                        tmp_dict[key].append(edge_cls[idx])
                        
                    for key, l in tmp_dict.items():
                        if key in edge_pds: 
                            raise RuntimeError()
                        pd =  torch.stack(l,dim=0)
                        pd = torch.softmax(pd,dim=1).mean(0)
                        edge_pds[key] = pd
                        # src_inst_idx, tgt_inst_idx = inst_mask2inst[src_idx], inst_mask2inst[tgt_idx]
                        # inst_gt_pairs.add((src_inst_idx, tgt_inst_idx))
                    
                    return node_pds, edge_pds, tock-tick
                
                if isinstance(batch_data,list):
                    for idx,data in enumerate(batch_data):
                        fid = data['fid']
                        print(idx)
                        node_pds, edge_pds, pt = process(data)
                        if pt>0:
                            acc_time += pt
                            timer_counter+=1
                        
                            fuse(nodes_pds_all,nodes_w,node_pds)
                            fuse(edges_pds_all,edges_w,edge_pds)
                            
                            
                            inst_mask2instance = data_inst['mask2instance']
                            # data_inst['node_edges'] = data_inst['node_edges'].t().contiguous()
                            gts=None
                            # gts = dict()
                            # gt_nodes = gts['nodes'] = dict()
                            # gt_edges = gts['edges'] = dict()
                            # for mid in range(data_inst['gt_cls'].shape[0]):
                            #     idx = inst_mask2instance[mid]
                            #     gt_nodes[idx] = data_inst['gt_cls'][mid]
                            # for idx in range(len(data_inst['gt_rel'])):
                            #     src_idx, tgt_idx = data_inst['node_edges'][idx,0].item(),data_inst['node_edges'][idx,1].item()
                            #     src_inst_idx, tgt_inst_idx = inst_mask2instance[src_idx], inst_mask2instance[tgt_idx]
                            #     gt_edges[(src_inst_idx, tgt_inst_idx)] = data_inst['gt_rel'][idx]
                            
                            g = graphDrawer.draw({'nodes':nodes_pds_all, 'edges':edges_pds_all},
                                                 gts)
                            
                            # merged_node_cls, merged_node_cls_gt,  merged_edge_cls, \
                            #     merged_edge_cls_gt, merged_mask2instance, merged_node_edges  = \
                            #            merge_pred_with_gt(data_inst,
                            #            node_cls_names,edge_cls_names,
                            #            nodes_pds_all, edges_pds_all,
                            #            inst_valid_node_cls_indices,seg_valid_node_cls_indices,
                            #            inst_valid_edge_cls_indices,seg_valid_edge_cls_indices,
                            #            noneidx_node_cls,noneidx_edge_cls,
                            #            ignore_missing,self.cfg.DEVICE)
                                       
                            # eval_tool.add([scan_id], 
                            #                       merged_node_cls,
                            #                       merged_node_cls_gt, 
                            #                       merged_edge_cls,
                            #                       merged_edge_cls_gt,
                            #                       [merged_mask2instance],
                            #                       merged_node_edges)
                            
                            # pds = process_pd(**eval_tool.predictions[scan_id]['pd'])
                            # gts = process_gt(**eval_tool.predictions[scan_id]['gt'])
                            
                            # g =     draw_evaluation(scan_id, pds[0], pds[1], gts[0], gts[1], none_name = 'UN', 
                            #         pd_only=False, gt_only=False)
                            
                            
                            g.render(os.path.join(self.cfg['training']['out_dir'], self.cfg.name, str(fid)+'_graph'),view=True)
                # else:
                #     data = batch_data
                #     fid = data['fid']
                #     node_pds, edge_pds, pt = process(data)
                #     if pt>0:
                #         acc_time += pt
                #         timer_counter+=1
                    
                #         fuse(nodes_pds_all,nodes_w,node_pds)
                #         fuse(edges_pds_all,edges_w,edge_pds)
                        
                #         eval_tool.add([scan_id], 
                #                                       merged_node_cls,
                #                                       merged_node_cls_gt, 
                #                                       merged_edge_cls,
                #                                       merged_edge_cls_gt,
                #                                       [merged_mask2instance],
                #                                       merged_node_edges)
                                        
                #         pds = process_pd(**eval_tool.predictions[scan_id]['pd'])
                #         gts = process_gt(**eval_tool.predictions[scan_id]['gt'])
                        
                #         g =     draw_evaluation(scan_id, pds[0], pds[1], gts[0], gts[1], none_name = 'UN', 
                #                 pd_only=False, gt_only=False)
                        
                        
                #         g.render(os.path.join(self.cfg['training']['out_dir'], self.cfg.name, str(fid)+'_graph'),view=True)
            # if index > 10: break
            break
        print('time:',acc_time,timer_counter,acc_time/timer_counter)
        
        eval_dict = dict()
        obj_, edge_ = eval_tool.get_mean_metrics()
        for k,v in obj_.items():
            # print(k)
            eval_dict[k+'_node_cls'] = v
        for k,v in edge_.items():
            # print(k)
            eval_dict[k+'_edge_cls'] = v
        
        for k, v in eval_list.items():
            eval_dict[k] = v.avg
        
        vis = self.visualize(eval_tool=eval_tool)
        eval_dict['visualization'] = vis
        return eval_dict, eval_tool