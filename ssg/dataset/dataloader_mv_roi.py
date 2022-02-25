#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 09:25:32 2021

@author: sc
vis imgs in tensor: https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html#sphx-glr-auto-examples-plot-visualization-utils-py

"""
import torch.utils.data as data
import os, random, torch, json, trimesh, h5py, copy
import numpy as np
import multiprocessing as mp

# from utils import util_ply, util_data, util, define
from codeLib.common import random_drop, random_drop
from codeLib import transformation
from ssg.utils import util_ply, util_data
from codeLib.utils.util import read_txt_to_list, check_file_exist
from ssg import define
from codeLib.torch.visualization import show_tensor_images
from codeLib.common import normalize_imagenet
from torchvision import transforms
import codeLib.torchvision.transforms as cltransform
import ssg.utils.compute_weight as compute_weight
from ssg.utils.util_data import raw_to_data, cvt_all_to_dict_from_h5
import codeLib.utils.string_numpy as snp
import logging
logger_py = logging.getLogger(__name__)
DRAW_BBOX_IMAGE=True
DRAW_BBOX_IMAGE=False

class MultiViewROIImageLoader(data.Dataset):
    def __init__(self,config,mode, **args):
        super().__init__()
        assert mode in ['train','validation','test']
        self._device = config.DEVICE
        path = config.data['path']
        self.config = config
        self.mconfig = config.data
        self.path = config.data.path
        self.label_file = config.data.label_file
        self.use_data_augmentation=self.mconfig.data_augmentation
        self.root_3rscan = define.DATA_PATH
        self.path_h5 = os.path.join(self.path,'relationships_%s.h5' % (mode))
        self.path_mv = os.path.join(self.path,'proposals.h5')
        self.path_roi_img = self.mconfig.roi_img_path
        try:
            self.root_scannet = define.SCANNET_DATA_PATH
        except:
            self.root_scannet = None
        
        selected_scans = set()
        self.w_cls_obj=self.w_cls_rel=None
        # self.multi_rel_outputs = multi_rel_outputs = config.model.multi_rel
        self.shuffle_objs = False
        # self.use_rgb = config.model.use_rgb
        # self.use_normal = config.model.use_normal
        self.sample_in_runtime= config.data.sample_in_runtime
        # self.load_cache = False
        self.for_eval = mode != 'train'
        # self.max_edges=config.data.max_num_edge
        # self.full_edge = self.config.data.full_edge
        
        self.output_node = args.get('output_node', True)
        # self.output_edge = args.get('output_edge', True)    

        ''' read classes '''
        pth_classes = os.path.join(path,'classes.txt')
        pth_relationships = os.path.join(path,'relationships.txt')      
        selected_scans = read_txt_to_list(os.path.join(path,'%s_scans.txt' % (mode)))
        
        names_classes = read_txt_to_list(pth_classes)
        # names_relationships = read_txt_to_list(pth_relationships)
        
        self.classNames = sorted(names_classes)
        self.relationNames=None
        '''set transform'''
        if self.mconfig.load_images:
            if not self.for_eval:
                self.transform  = transforms.Compose([
                    transforms.Resize(config.data.roi_img_size),
                    cltransform.TrivialAugmentWide(),
                    # RandAugment(),
                    ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(config.data.roi_img_size),
                    ])
        
        ''' load data '''
        if self.mconfig.load_images:
            self.open_mv_graph()
            self.open_img()
        
        self.open_data()
        c_sg_data = cvt_all_to_dict_from_h5(self.sg_data)
        
        '''check scan_ids'''
        # filter input scans with relationship data
        tmp   = set(c_sg_data.keys())
        inter = sorted(list(tmp.intersection(selected_scans)))
        if self.mconfig.load_images:
            # filter input scans with image data
            tmp   = set(self.mv_data.keys())
            inter = sorted(list(tmp.intersection(inter)))
            
            # filter input scans with roi images
            tmp   = set(self.roi_imgs.keys())
            inter = sorted(list(tmp.intersection(inter)))
            
            #TODO: also filter out nodes when only with points input. this gives fair comparison on points and images methods.
            filtered_sg_data = dict()
            for scan_id in inter:
                mv_node_ids = [int(x) for x in self.mv_data[scan_id]['nodes'].keys()]
                sg_node_ids = c_sg_data[scan_id]['nodes'].keys()                
                inter_node_ids = set(sg_node_ids).intersection(mv_node_ids)
                
                filtered_sg_data[scan_id] = dict()
                filtered_sg_data[scan_id]['nodes'] = {nid: c_sg_data[scan_id]['nodes'][nid] for nid in inter_node_ids}
            
            c_sg_data = filtered_sg_data
        
        '''pack with snp'''
        self.size = len(inter)
        self.scans = snp.pack(inter)#[s for s in data.keys()]

        '''compute weight  ''' #TODO: rewrite this. in runtime sampling the weight might need to be calculated in each epoch.
        if not self.for_eval:
            if config.data.full_edge:
                edge_mode='fully_connected'
            else:
                edge_mode='nn'
            wobjs, wrels, o_obj_cls, o_rel_cls = compute_weight.compute_sgfn(self.classNames, None, c_sg_data, selected_scans,
                                                                        normalize=config.data.normalize_weight,
                                                                        for_BCE=False,
                                                                        edge_mode=edge_mode,
                                                                        verbose=config.VERBOSE)
            self.w_node_cls = torch.from_numpy(np.array(wobjs)).float()
            self.w_edge_cls=None
            # self.w_edge_cls = torch.from_numpy(np.array(wrels)).float()
            
        del self.sg_data    
        del self.roi_imgs
        del self.mv_data
        
    def __len__(self):
        return self.size
    
    def open_mv_graph(self):
        if not hasattr(self, 'mv_data'):
            self.mv_data = h5py.File(self.path_mv,'r')
                
    def open_data(self):
        if not hasattr(self,'sg_data'):
            self.sg_data = h5py.File(self.path_h5,'r')
            
    def open_img(self):
        if not hasattr(self, 'roi_imgs'):
            self.roi_imgs = h5py.File(self.path_roi_img,'r')
            
    def __getitem__(self,index):
        scan_id = snp.unpack(self.scans,index)# self.scans[idx]
        
        self.open_data()
        scan_data_raw = self.sg_data[scan_id]
        scan_data = raw_to_data(scan_data_raw)
        
        object_data = scan_data['nodes']
        # relationships_data = scan_data['relationships']        
        
        
        self.open_mv_graph()
        self.open_img()
        mv_data = self.mv_data[scan_id]
        mv_nodes = mv_data['nodes']
        roi_imgs = self.roi_imgs[scan_id]
            
        '''filter'''
        mv_node_ids = [int(x) for x in mv_data['nodes'].keys()]
        
        sg_node_ids = object_data.keys()                
        inter_node_ids = set(sg_node_ids).intersection(mv_node_ids)
        
        object_data = {nid: object_data[nid] for nid in inter_node_ids}
            
        ''' build mapping '''
        instance2labelName  = { int(key): node['label'] for key,node in object_data.items()  }
        
        '''build instance dict'''
        seg2inst = dict()
        for oid, odata in object_data.items():
            if 'instance_id' in odata:
                seg2inst[oid] = odata['instance_id']
            
        '''sample training set'''  
        instances_ids = list(instance2labelName.keys())
        if 0 in instances_ids: instances_ids.remove(0)
        if self.sample_in_runtime and not self.for_eval:
            selected_nodes = list(object_data.keys())
            
            mv_node_ids = [int(x) for x in mv_nodes.keys()]
            selected_nodes = list( set(selected_nodes).intersection(mv_node_ids) )
            
            use_all=False
            sample_num_nn=self.mconfig.sample_num_nn# 1 if "sample_num_nn" not in self.config else self.config.sample_num_nn
            sample_num_seed=self.mconfig.sample_num_seed#1 if "sample_num_seed" not in self.config else self.config.sample_num_seed
            if sample_num_nn<=0 or sample_num_seed <=0:
                use_all=True
                
            if not use_all:
                filtered_nodes = random_drop(selected_nodes, self.mconfig.drop_img_edge, replace=True)
            else:
                filtered_nodes = selected_nodes # use all nodes
                
            instances_ids = list(filtered_nodes)
            if 0 in instances_ids: instances_ids.remove(0)
            
        if 'max_num_node' in self.mconfig and self.mconfig.max_num_node>0 and len(instances_ids)>self.mconfig.max_num_node:
            instances_ids = random_drop(instances_ids, self.mconfig.max_num_node )
        
        if self.shuffle_objs:
            random.shuffle(instances_ids)

        ''' 
        Find instances we care abot. Build oid2idx and cat list
        oid2idx maps instances to a mask id. to randomize the order of instance in training.
        '''
        oid2idx = {}
        idx2oid = {}
        cat = []
        counter = 0
        filtered_instances = list()
        for instance_id in instances_ids:    
            class_id = -1
            instance_labelName = instance2labelName[instance_id]
            if instance_labelName in self.classNames:
                class_id = self.classNames.index(instance_labelName)

            # mask to cat:
            if (class_id >= 0) and (instance_id > 0): # insstance 0 is unlabeled.
                oid2idx[int(instance_id)] = counter
                idx2oid[counter] = int(instance_id)
                counter += 1
                filtered_instances.append(instance_id)
                cat.append(class_id)
        if len(cat) == 0:
            # logger_py.debug('filtered_nodes: {}'.format(filtered_nodes))
            logger_py.debug('cat: {}'.format(cat))
            logger_py.debug('self.classNames: {}'.format(self.classNames))
            logger_py.debug('list(object_data.keys()): {}'.format(list(object_data.keys())))
            assert len(cat) > 0

        '''load images'''
        bounding_boxes = list()
        for idx in range(len(cat)):
            oid = str(idx2oid[idx])
            node = mv_nodes[oid]
            cls_label = node.attrs['label']
            if cls_label == 'unknown':
                cls_label = self.classNames[cat[idx]]
            img = np.asarray(roi_imgs[oid])
            
            if not self.for_eval:
                kf_indices = random_drop(range(img.shape[0]), self.mconfig.drop_img_edge, replace=True)
                img = img[kf_indices]
            # else:
            #     kf_indices = [idx for idx in range(img.shape[0])]
            
            img = torch.as_tensor(img).clone()
            img = torch.clamp((img*255).byte(),0,255).byte()
            t_img = torch.stack([self.transform(x) for x in img],dim=0)
            if DRAW_BBOX_IMAGE:
                show_tensor_images(t_img.float()/255, cls_label)
            t_img= normalize_imagenet(t_img.float()/255.0)
            bounding_boxes.append( t_img)
        
        ''' to tensor '''
        gt_class = torch.from_numpy(np.array(cat))
        
        del self.sg_data
        del self.roi_imgs
        del self.mv_data
        
        output = dict()
        output['scan_id'] = scan_id # str
        output['gt_cls'] = gt_class # tensor
        output['roi_imgs'] = bounding_boxes #list
        output['instance2mask'] = oid2idx #dict
        output['seg2inst'] = seg2inst
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