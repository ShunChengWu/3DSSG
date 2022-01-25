#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 09:25:32 2021

@author: sc
"""
# import os
# from ssg2d.utils import util_data
import sys
from codeLib.utils.util import read_txt_to_list
import codeLib.utils.string_numpy as snp
from codeLib.common import normalize_imagenet, random_drop#, load_obj
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

from torchvision.ops import roi_align
# from torchvision.io import read_image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle

DRAW_BBOX_IMAGE=False

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

class MultiViewImageLoader(data.Dataset):
    def __init__(self,config,mode):
        super().__init__()
        assert mode in ['train','validation','test']
        
        '''copy config values'''
        self._device = config.DEVICE
        self.path = config.data.path
        self.h5_path = config.data.proposal_path     
        self.img_path = config.data.img_path
        self.sample_num_nn = config.data.sample_num_nn
        self.drop_img_edge = config.data.drop_img_edge
        self.relationNames=None
        self.w_node_cls=None
        self.w_edge_cls=None
        self.img_size = config.data.img_size
        
        '''set transform'''
        if config.data.img_size[0] > 0:
            # self.transform = transforms.Compose([
            #     transforms.Resize(config.data.img_size),
            #     transforms.ToTensor(),
            #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # ])
            #Note: These two can be merged. but seperated here for visualization purpose
            # self.T = T.Compose([T.Resize(config.data.img_size)])
            self.T = T.Compose([
                T.Resize(config.data.img_size),
                T.RandomZoomOut(side_range=(1.0,2.0)),
                T.Resize(config.data.img_size),
                T.RandomHorizontalFlip(0.5),
                T.RandomPhotometricDistort(),
                # transforms.ToTensor()
                ])
            self.transform = transforms.ToTensor()
        else:
            self.T = T.Compose([
                T.Resize(config.data.img_size),
                # transforms.ToTensor()
                ])
            self.transform = transforms.ToTensor()
            # self.transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # ])
            
            
        '''read info'''
        # classes
        self.classNames = read_txt_to_list(config.data.path+'classes.txt')
        # scan ids
        scan_ids = read_txt_to_list(config.data.path+mode+'_scans.txt')
        
        '''load data from graph'''
        self.open_hdf5(self.h5_path)
        id_from_prop = set(self.data.keys())
        
        #check scan id matches
        diff = set(scan_ids).difference(id_from_prop)
        if len(diff )> 0:
            if config.VERBOSE:
                print('remove {} invalid/unexist scans.'.format(len(diff)))
            scan_ids = set(scan_ids).intersection(id_from_prop)
            if len(scan_ids) == 0:
                raise RuntimeError('no valid scans!')
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
        # del self.imgs
        
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
        node_ids = random_drop(node_ids, self.sample_num_nn)
        
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
                
                kf_indices = np.asarray(node)
                kf_indices = random_drop(kf_indices, self.drop_img_edge)
                kfs_indices.append(kf_indices)
                images_indices = images_indices.union(kf_indices)
                
        '''load images'''
        # images = list()
        # poses = list()
        images_info = list()
        images = torch.zeros([len(images_indices), 3, self.img_size[0], self.img_size[1]], requires_grad=False, dtype=torch.float)
        poses  = torch.zeros([len(images_indices), 4, 4], requires_grad=False, dtype=torch.float)
        fidx2bufferidx = dict()
        counter=0
        for fid in images_indices:
            # prepare
            bfid = imgs['indices'][fid] # convert frame idx to the buffer idx 
            kf = kfs[str(fid)]
            
            #  bounding box
            img_info = dict()
            img_info['boxes'] = torch.as_tensor(np.asarray(kf), dtype=torch.float32)
            img_info['seg2idx'] = {v[0]:v[1] for v in kf.attrs['seg2idx']}
            
            # load image
            img_data = imgs['rgb'][bfid]
            img = imageio.imread(img_data)
            img = Image.fromarray(img)
            
            # load pose
            pose_data = imgs['poses'][bfid]
            pose = torch.from_numpy(pose_data)
            
            # Image transform
            img, img_info = self.T(img,img_info)
            
            # if DRAW_BBOX_IMAGE:
            #     for i in range(img_info['boxes'].shape[0]):
            #         box = img_info['boxes'][i]
            #         anchor = (box[0],box[1])
            #         box_w = box[2]-box[0]
            #         box_h = box[3]-box[1]
            #         # plt.imshow(im)
            #         # Get the current reference
            #         ax = plt.gca()
            #         # # Create a Rectangle patch
            #         rect = Rectangle(anchor,box_w,box_h,linewidth=1,edgecolor='r',facecolor='none')
            #         ax.add_patch(rect)
                    
            #         oid = str(kf.attrs['seg2idx'][i][0])
            #         label = nodes[oid].attrs['label']
            #         matplotlib.pyplot.text(anchor[0],anchor[1]+0.5*box_h,label,fontsize=12)
            #     ax.imshow(img)
            #     plt.show()
                
            # transform
            if self.transform: img = self.transform(img)
            img = normalize_imagenet(img)
            
            # append to buffer
            images_info.append(img_info)
            # poses.append(pose)
            # images.append(img)
            poses[counter] = pose
            images[counter] = img
            
            
            fidx2bufferidx[fid]=counter
            counter+=1
            width,height = img.shape[-1],img.shape[-2] # this may cause problem when input image have non-consistent size
            
        bounding_boxes = list() # bounding_boxes[node_id]{kf_id: [boxes]}
        # if DRAW_BBOX_IMAGE:
        #     img_list=list()
        #     tensor_boxes=list()
        for idx in range(len(kfs_indices)):
            kf_indices=kfs_indices[idx]
            oid = idx2oid[idx]
            
            box_dict = dict()
            for fid in kf_indices:
                bfid = fidx2bufferidx[fid]
                img_info = images_info[bfid]
                
                kf_seg2idx = img_info['seg2idx']
                bboxes = img_info['boxes']
                bid = kf_seg2idx[oid]
                
                bbox = bboxes[bid,:4].clone()
                bbox[0] /= width
                bbox[1] /= height
                bbox[2] /= width
                bbox[3] /= height
                box_dict[bfid] = bbox.view(1,-1)
                
                # # get frame
                # kf = kfs[str(fid)]
                # # get boxes
                # bboxes = np.asarray(kf)
                # # get object mapping to box index
                # kf_seg2idx = {v[0]:v[1] for v in kf.attrs['seg2idx']}
                # # get box indexP
                # bid = kf_seg2idx[oid]
                # # get box
                # bbox = bboxes[bid]
                # normalize
                # bbox = torch.FloatTensor( [bbox[0]/width,bbox[1]/height,bbox[2]/width,bbox[3]/height] )
                # # convert frame index to frame buffer index
                # bfidx = fidx2bufferidx[fid]
                # box_dict[bfidx] = bbox
                
                # for bid in range(len(bboxes)):
                #     bbox = bboxes[bid]
                #     oid = kf_seg2idx[bid][0]
                #     # scale bbox 
                #     bbox = torch.FloatTensor( [bbox[0]/width,bbox[1]/height,bbox[2]/width,bbox[3]/height] )
                #     box_dict[ oid2idx[ oid ] ] = bbox
                
                # if DRAW_BBOX_IMAGE:
                #     '''debug: plot bbox and the roi cropping from roi_align'''
                #     # from PIL import Image
                #     # print('img_path',img_path)
                #     # img_data = imgs['rgb'][fid]
                #     # img = imageio.imread(img_data)
                #     tmp = (images[bfid]*255).byte().permute(1,2,0).numpy()
                #     im = Image.fromarray(tmp)
                #     box = bboxes[bid]
                #     # box[0]*=im.size[0]
                #     # box[1]*=im.size[1]
                #     # box[2]*=im.size[0]
                #     # box[3]*=im.size[1]
                    
                #     anchor = (box[0],box[1])
                #     box_w = box[2]-box[0]
                #     box_h = box[3]-box[1]
                    
                #     # plt.imshow(im)
                #     # Get the current reference
                #     ax = plt.gca()
                #     # # Create a Rectangle patch
                #     rect = Rectangle(anchor,box_w,box_h,linewidth=1,edgecolor='r',facecolor='none')
                #     # Add the patch to the Axes
                #     ax.add_patch(rect)
                    
                #     label = nodes[str(oid)].attrs['label']
                #     matplotlib.pyplot.text(anchor[0],anchor[1]+0.5*box_h,label,fontsize=12)
                    
                #     ax.imshow(im)
                #     plt.imshow(im)
                #     plt.show()
                #     plt.close()
                    
                #     box_tensor= torch.tensor(box).view(1,4)
                #     image2 = transforms.ToTensor()(im).unsqueeze(0)*255
                #     # image2 = read_image(img_path).float().unsqueeze(0)
                #     roi_features = roi_align(image2, [box_tensor], (int(box_h),int(box_w)))
                #     roi_features = roi_features.squeeze(0).long().permute(1, 2, 0)
                #     # plt.imshow(  roi_features )
                #     # plt.close()
                #     tensor_boxes.append(box_tensor)
                #     img_list.append(transforms.ToTensor()(im)*255)
            bounding_boxes.append(box_dict)
            
            # if DRAW_BBOX_IMAGE:
            #     '''debug: plot roi image and the original image'''
            #     # print('class:',self.classNames[cat[i]])
            #     img_list = torch.stack(img_list,dim=0)
            #     roi_features = roi_align(img_list, tensor_boxes, (int(250),int(250)))
            #     for j in range(len(img_list)):
            #         img = img_list[j]
            #         roi_img = roi_features[j]
            #         img= img.long().permute(1, 2, 0)
            #         roi_img = roi_img.long().permute(1, 2, 0)
            #         plt.imshow(  img )
            #         plt.show()
            #         plt.imshow(  roi_img )
            #         plt.show()
                    
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
        output['images'] = images# tensor
        output['poses'] = poses# tensor
        output['image_boxes'] = bounding_boxes #list
        
        self.reset()
        return output
    
if __name__ == '__main__':
    import ssg2d
    import codeLib
    from ssg2d.data.collate import graph_collate
    
    
    path = './configs/default_mv.yaml'
    # path = './configs/mv/exp5_base.yaml'
    # path = './configs/exp2_all_test_overfit.yaml'
    # path = './configs/exp2_all_gnn.yaml'
    # path= './configs/exp2_all_basic_bf200.yaml'
    config = codeLib.Config(path)
    config.DEVICE='cpu'
    # config.model.node_encoder.backend='vgg16'
    # config.data.input_type = 'mv'
    # config.data.path= '/home/sc/research/PersistentSLAM/python/2DTSG/data/mv_scannet/'
    # config.data.drop_img_edge = 3
    codeLib.utils.util.set_random_seed(config.SEED)
    
    
    
    dataset = ssg2d.config.get_dataset(config,'train')
    
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
        dataset, batch_size=2, num_workers=0, shuffle=False,
        pin_memory=True,
        collate_fn=graph_collate,
    )
    for epoch in tqdm(range(config.training.max_epoch)):
        for data in tqdm(train_loader):
            continue
            # break
        dataset.reset()
        # break