#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 09:25:32 2021

@author: sc
vis imgs in tensor: https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py

"""
# import os
# from ssg2d.utils import util_data
import sys
from codeLib.utils.util import read_txt_to_list
from codeLib.torch.visualization import show_tensor_images
import codeLib.utils.string_numpy as snp
from codeLib.common import normalize_imagenet, denormalize_imagenet, random_drop#, load_obj
import h5py
import torch.utils.data as data
from torchvision import transforms
# import ssg2d.utils.compute_weight as compute_weight
import torch
# import json
import numpy as np
# import pandas
from PIL import Image
# from torchvision.io import read_image
from ssg2d.utils.compute_weight import compute_weights
import ssg2d.data.transforms as T
import imageio
import torchvision
from torchvision.ops import roi_align
# from torchvision.io import read_image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle

DRAW_BBOX_IMAGE=True
# DRAW_BBOX_IMAGE=False

def compute_weight(data,classNames:list, scan_list:list, normalize:bool, verbose:bool):
    '''go thorugh all sequences and check occurance'''
    o_obj_cls = np.zeros((len(classNames)))
    for scan_id in scan_list:
        nodes = data[scan_id]['nodes']
        for node in nodes.values():
            label = node.attrs['label']
            if label not in classNames:
                continue
                raise RuntimeError('unknown label')
            idx = classNames.index(label)
            o_obj_cls[idx] += 1
        
    wobjs = compute_weights(classNames, o_obj_cls,normalize,verbose)
    return wobjs, o_obj_cls

def get_invalid_scan_ids(x):
    invalid = list()
    for scan_id in x.keys():
        if len(x[scan_id]) == 0:
            invalid.append(scan_id)
        else:
            for oid in x[scan_id].keys():
                if len(x[scan_id][oid]) == 0:
                    invalid.append(scan_id)
                    break
    return invalid


class TrivialAugmentWide(transforms.TrivialAugmentWide):
     # def _augmentation_space(self, num_bins: int):
     #    return {
     #        # op_name: (magnitudes, signed)
     #        "Identity": (torch.tensor(0.0), False),
     #        "ShearX": (torch.linspace(0.0, 0.5, num_bins), True),
     #        "ShearY": (torch.linspace(0.0, 0.5, num_bins), True),
     #        "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
     #        "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
     #        "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
     #        "Brightness": (torch.linspace(0.0, 0.1, num_bins), True),
     #        "Color": (torch.linspace(0.0, 0.1, num_bins), True),
     #        "Contrast": (torch.linspace(0.0, 0.1, num_bins), True),
     #        "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
     #        "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
     #        # "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
     #        "AutoContrast": (torch.tensor(0.0), False),
     #        # "Equalize": (torch.tensor(0.0), False),
     #    }
    def _augmentation_space(self, num_bins: int):
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.2, 0.8, num_bins), True),
            "Color": (torch.linspace(0.2, 0.8, num_bins), True),
            "Contrast": (torch.linspace(0.2, 0.8, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.8, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            # "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            # "Equalize": (torch.tensor(0.0), False),
        }
class RandAugment(transforms.RandAugment):
      def _augmentation_space(self, num_bins, image_size):
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            # "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            # "Equalize": (torch.tensor(0.0), False),
        }

class MultiViewROIImageLoader(data.Dataset):
    def __init__(self,config,mode):
        super().__init__()
        assert mode in ['train','validation','test']
        
        '''copy config values'''
        self.mode = mode
        self._device = config.DEVICE
        self.path = config.data.path
        self.h5_path = config.data.proposal_path
        self.img_path = config.data.roi_path
        self.sample_num_nn = config.data.sample_num_nn
        self.drop_img_edge = config.data.drop_img_edge
        self.relationNames=None
        self.w_node_cls=None
        self.w_edge_cls=None
        self.img_size = config.data.img_size
        
        '''set transform'''
        if self.mode == 'train':
            self.transform  = transforms.Compose([
                transforms.Resize(config.data.roi_img_size),
                TrivialAugmentWide(),
                # RandAugment(),
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(config.data.roi_img_size),
                ])
            
        '''read info'''
        # classes
        self.classNames = read_txt_to_list(config.data.path+'classes.txt')
        # scan ids
        scan_ids = read_txt_to_list(config.data.path+mode+'_scans.txt')
        
        '''load data from graph'''
        self.open_hdf5(self.h5_path)
        id_from_prop = set(self.data.keys())
        '''check roi images are valid'''
        self.open_img_hdf5(self.img_path)
        # invalid_scan_ids = get_invalid_scan_ids(self.imgs)
        # id_from_prop -= set(invalid_scan_ids)
        
        #check scan id matches
        diff = set(scan_ids).difference(id_from_prop)
        if len(diff )> 0:
            if config.VERBOSE:
                print('remove {} invalid/unexist scans.'.format(len(diff)))
            scan_ids = set(scan_ids).intersection(id_from_prop)
            if len(scan_ids) == 0:
                raise RuntimeError('no valid scans!')
        scan_ids = random_drop(list(scan_ids), config.data.fraction , replace=False)
        scan_ids = sorted(scan_ids) # use sorted here to prevent randomness
            
        if config.VERBOSE:
            print('total scans:',len(scan_ids))
        
        self.scans = snp.pack(scan_ids)#[s for s in data.keys()]
        self.size = len(scan_ids)
        ''' compute weight '''
        if mode == 'train':
            weights = compute_weight(self.data,self.classNames,scan_ids,config.data.normalize_weight, config.VERBOSE)
            self.w_node_cls = torch.from_numpy(weights[0]).float()
            
        # delete data (should be opened in getitem due to multithreading issue)
        del self.data
        del self.imgs
        
    def __len__(self):
        return self.size
    
    def open_hdf5(self, path):
        if not hasattr(self, 'data'):
            self.data = h5py.File(path,'r')
            
    def open_img_hdf5(self, path):
        if not hasattr(self, 'imgs'):
            self.imgs = h5py.File(path,'r')
            
    def reset(self):
        if hasattr(self, 'imgs'):
            self.imgs.close()
            del self.imgs
        if hasattr(self, 'data'):
            self.data.close()
            del self.data
        
    def __getitem__(self,idx):
        scan_id = snp.unpack(self.scans,idx)# self.scans[idx]
        self.open_hdf5(self.h5_path)
        self.open_img_hdf5(self.img_path)
        scan_data = self.data[scan_id]
        imgs = self.imgs[scan_id]        
        kfs = scan_data['kfs']
        nodes = scan_data['nodes']
                
        '''sample nodes'''
        node_ids = list(nodes.keys())
        if self.mode == 'train':
            node_ids = random_drop(node_ids, self.sample_num_nn)
        # print('node_ids',node_ids)
        
        '''select nodes'''
        kfs_indices=list()
        cat=list()
        images_indices = set()
        oid2idx = dict()
        idx2oid = dict()
        counter=0
        for n_id in node_ids:
            node = nodes[n_id]
            class_id = -1
            cls_label = node.attrs['label']
            
            if cls_label in self.classNames:
                class_id = self.classNames.index(cls_label)
                
            if class_id >=0:
                oid2idx[int(n_id)] = counter
                idx2oid[counter] = int(n_id)
                counter+=1
                cat.append(class_id)
                
                
        '''load images'''
        bounding_boxes = list()
        for idx in range(len(cat)):
            oid = str(idx2oid[idx])
            node = nodes[oid]
            cls_label = node.attrs['label']
            img = np.asarray(imgs[oid])
            
            if self.mode == 'train':
                kf_indices = random_drop(range(img.shape[0]), self.drop_img_edge, replace=True)
                img = img[kf_indices]
            # else:
            #     kf_indices = [idx for idx in range(img.shape[0])]
            
            img = torch.as_tensor(img)
            img = torch.clamp((img*255).byte(),0,255).byte()
            t_img = torch.stack([self.transform(x) for x in img],dim=0)
            if DRAW_BBOX_IMAGE:
                show_tensor_images(t_img.float()/255, cls_label)
            t_img= normalize_imagenet(t_img.float()/255.0)
            bounding_boxes.append( t_img)
            
        '''to tensor'''
        # Object 
        gt_class = torch.as_tensor(cat)
        # Images
        # images = torch.stack(images)
        # Poses
        # poses = torch.stack(poses)
        
                
        output = dict()
        output['scan_id'] = scan_id # str
        output['gt_cls'] = gt_class # tensor
        # output['images'] = images# tensor
        # output['poses'] = poses# tensor
        output['roi_imgs'] = bounding_boxes #list
        
        self.reset()
        return output
    
if __name__ == '__main__':
    from ssg2d import config
    import codeLib
    from ssg2d.data.collate import graph_collate
    
    
    path = './configs/default_mv.yaml'
    # path = './configs/mv/exp6_base.yaml'
    # path = './configs/exp2_all_test_overfit.yaml'
    # path = './configs/exp2_all_gnn.yaml'
    # path= './configs/exp2_all_basic_bf200.yaml'
    cfg= codeLib.Config(path)
    cfg.data.input_type='mv_roi'
    cfg.DEVICE='cpu'
    # config.model.node_encoder.backend='vgg16'
    # config.data.input_type = 'mv'
    # config.data.path= '/home/sc/research/PersistentSLAM/python/2DTSG/data/mv_scannet/'
    # config.data.drop_img_edge = 3
    # codeLib.utils.util.set_random_seed(config.SEED)
    
    
    
    dataset = config.get_dataset(cfg,'train')
    
    # dataset.__getitem__(0)
    from tqdm import tqdm
    # for data in tqdm(iter(dataset)):
    #     continue
    
    # train_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=config['training']['batch'], num_workers=config['training']['data_workers'], shuffle=False,
    #     pin_memory=False,
    #     collate_fn=graph_collate,
    # )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, num_workers=0, shuffle=True,
        pin_memory=True,
        collate_fn=graph_collate,
    )
    for epoch in tqdm(range(cfg.training.max_epoch)):
        for data in tqdm(train_loader):
            # continue
            break
        dataset.reset()
        break