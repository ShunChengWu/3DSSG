#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:57:35 2021

@author: sc
"""
import argparse,os
import codeLib
import ssg2d
import torch
import ssg2d.config as config
import torch.optim as optim
import logging
import time
import os
from ssg2d.checkpoints import CheckpointIO
from torch.utils.tensorboard import SummaryWriter
from codeLib.common import convert_torch_to_scalar
import numpy as np
import torch.multiprocessing
import cProfile, pstats
from codeLib.utils.util import pytorch_count_params
from collections import defaultdict
logging.basicConfig()
logger_py = logging.getLogger(__name__)
class TO_DEVICE():
    def __init__(self,device):
        self._device = device

    def __call__(self,*args):
        output = list()
        for item in args:
            if isinstance(item,  torch.Tensor):
                output.append(item.to(self._device ))
            elif isinstance(item,  dict):
                ks = item.keys()
                vs = self(*item.values())
                item = dict(zip(ks, vs))
                output.append(item)
            elif isinstance(item, list):
                output.append(self(*item))
            else:
                output.append(item)
                # print('hello')
                # for key,value in item.items():
                #     item[key] =
                #     output.append(self.toDevice(*item.values()))
        return output
    
    
def compare(dataloader, model1, model2, device):
    to_device = TO_DEVICE(device)
    model1=model1.eval()
    model2=model2.eval()
    
    list_out=list()
    # list_out2=list()
    list_names=list()
    # list_names2=list()
    with torch.no_grad():
        for data in iter(dataloader):
            images, image_bboxes, image_edges, gt_cls = to_device(*(data['images'], data['image_boxes'], data['image_edges'], data['gt_cls']))
            out1 = model1(images=images, 
                          bboxes=image_bboxes, 
                          edges=image_edges)
            out2 = model2(images=images, 
                          bboxes=image_bboxes, 
                          edges=image_edges)
            list_out.append(out1.detach().cpu())
            list_out.append(out2.detach().cpu())
            list_names += ['0' for c in gt_cls]
            list_names += ['1' for c in gt_cls]
            
    list_features = torch.cat(list_out,dim=0)
    return list_features, list_names

def entire_model(dataloader, model,device):
    model=model.eval()
    to_device = TO_DEVICE(device)
    output=defaultdict(list)
    with torch.no_grad():
        for data in iter(dataloader):
            images, image_bboxes, image_edges, descriptor, node_edges, gt_node, gt_rel = to_device(
                *(data['images'], data['image_boxes'], data['image_edges'], data.get('descriptor'), data.get('node_edges'), data['gt_cls'],data.get('gt_rel'))
            )
            node_cls, edge_cls = model(images = images, 
                   image_bboxes = image_bboxes, 
                   image_edges = image_edges, 
                   descriptor = descriptor, 
                   node_edges = node_edges.t().contiguous())
            
            output['node_cls'].append(node_cls.detach().cpu())
            output['edge_cls'].append(edge_cls.detach().cpu())
            output['gt_node_idx'] += [c for c in gt_node]
            # output['gt_edge_idx'] += [c for c in gt_rel]
    for v in ['edge_cls','node_cls']:
        output[v] = torch.cat(output[v],dim=0)
    return output
def main():
    cfg = parse()
    cfg.VERBOSE=False
    # Shorthands
    n_workers = cfg['training']['data_workers']
    out_dir = os.path.join(cfg['training']['out_dir'], cfg.name)
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']
    backup_every = cfg['training']['backup_every']
    lr = float(cfg.training.lr)
    
    # set random seed
    codeLib.utils.util.set_random_seed(cfg.SEED)
    
    # Output directory
    # if not os.path.exists(out_dir): os.makedirs(out_dir)
    # Get logger
    cfg.logging.method = 'tensorboard' # use tensorboard function to plot embeddings
    logger = config.get_logger(cfg)
    if logger is not None:
        logger, _ = logger
    
    # create dataset and loaders
    dataset_train = config.get_dataset(cfg,'train')
    relationNames = dataset_train.relationNames
    classNames = dataset_train.classNames
    num_obj_cls = len(dataset_train.classNames)
    num_rel_cls = len(dataset_train.relationNames)
    
    model = config.get_model(cfg,num_obj_cls=num_obj_cls, num_rel_cls=num_rel_cls)
    
    ckpt_io = CheckpointIO(out_dir, model=model)
    ckpt_io.load('model_best.pt', device=cfg.DEVICE)
    
    node_encoder = model.node_encoder# config.get_node_encoder(cfg, cfg.DEVICE)
    # print('node_encoder: ',pytorch_count_params(node_encoder))
    node_encoder =node_encoder.eval()
    
    
    model2 = config.get_model(cfg,num_obj_cls=num_obj_cls, num_rel_cls=num_rel_cls)
    node_encoder2 = model2.node_encoder.eval()
    
    to_device = TO_DEVICE(cfg.DEVICE)
    
    # list_features, list_names = compare(dataset_train, node_encoder, node_encoder2, cfg.DEVICE)
    output = entire_model(dataset_train, model, cfg.DEVICE)
    
    if 'gt_node_idx'in output:
        output['gt_node_name'] = [classNames[c] for c in output['gt_node_idx']]
        logger.add_embedding(output['node_cls'],metadata=output['gt_node_name'],tag='node_feature',
                                       global_step=0)
    # if 'gt_edge_idx' in output:
    #     output['gt_edge_name'] = [relationNames[c] for c in output['gt_edge_idx']]
    #     logger.add_embedding(output['edge_cls'],metadata=output['gt_edge_name'],tag='edge_feature',
    #                                    global_step=0)
    
    # list_features=list()
    # list_names=list()
    # with torch.no_grad():
    #     for data in iter(dataset_train):
    #         images, image_bboxes, image_edges, gt_cls = to_device(*(data['images'], data['image_boxes'], data['image_edges'], data['gt_cls']))
    #         imgs_feature = node_encoder(images=images, 
    #                                     bboxes=image_bboxes, 
    #                                     edges=image_edges)
    #         list_features.append(imgs_feature.detach().cpu())
    #         list_names += [classNames[c] for c in gt_cls]
    #         # break
    # list_features = torch.cat(list_features,dim=0)
    # print(len(list_features), len(list_names))
    # logger.add_embedding(list_features,metadata=list_names,tag='node_feature',
    #                                    global_step=1)
    
def parse():
    r"""loads model config

    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='./configs/default.yaml', help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('--mode', type=str, choices=['train','trace','eval'], default='train', help='mode. can be [train,trace,eval]',required=False)
    parser.add_argument('--loadbest', type=int, default=0,choices=[0,1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')
    parser.add_argument('--log', type=str, default='DEBUG',choices=['DEBUG','INFO','WARNING','CRITICAL'], help='')
    args = parser.parse_args()
    
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise RuntimeError('Targer config file does not exist. {}' & config_path)
        
    # load config file
    config = codeLib.Config(config_path)
    # return config
    config.LOADBEST = args.loadbest
    config.MODE = args.mode
    
    # check if name exist
    if 'name' not in config:
        config_name = os.path.basename(args.config)
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['name'] = name 
    
    # init device
    if torch.cuda.is_available() and len(config.GPU) > 0:
        config.DEVICE = torch.device("cuda")
    else:
        config.DEVICE = torch.device("cpu")

    if config.VERBOSE:
        print(config)        
        
    config.log_level = args.log
    logger_py.setLevel(config.log_level)
    return config

if __name__ == '__main__':
    main()