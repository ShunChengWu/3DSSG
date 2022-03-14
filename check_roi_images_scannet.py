#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 15:14:13 2022

@author: sc
"""
import os
import h5py
import torch
import numpy as np
from ssg.utils.util_data import raw_to_data
from codeLib.torch.visualization import show_tensor_images
from codeLib.utils.util import read_txt_to_list

path = '/media/sc/SSD1TB2/dataset/scannet/roi.h5'
file = h5py.File(path,'r')

path = '/home/sc/research/PersistentSLAM/python/2DTSG/data/mv_scannet_scannet20/proposals.h5'
# mode = 'train'
# path = os.path.join(path,'relationships_%s.h5' % (mode))
sg_data = h5py.File(path,'r')

scan_list = read_txt_to_list('/home/sc/research/PersistentSLAM/python/2DTSG/data/mv_scannet_scannet20/train_scans.txt')
scan_list = sorted(scan_list)


for idx,scan_id in enumerate(scan_list):
    if idx<50:continue
    if scan_id not in sg_data: continue
    scan_data = sg_data[scan_id]
    kf_data   = file[scan_id]
    nodes = scan_data['nodes']
    kfs = scan_data['kfs']
    for nid in nodes:
        label = nodes[nid].attrs['label']
        # kf_ids = [str(id) for id in nodes[nid][:].tolist()]
        # kf_rois = [kfs[id] for id in kf_ids]
        
        if nid not in kf_data: continue
        
        
        img = np.asarray(kf_data[nid])
        img = torch.as_tensor(img).clone()
        img = torch.clamp((img*255).byte(),0,255).byte()
        
        if img.shape[0]>4: 
            img=img[:4]

    # scan_data = file[scan_id]
    # for nid in scan_data:
    #     if int(nid) not in object_data: continue
    #     label = object_data[int(nid)]['label']
        
    #     node_data = scan_data[nid]
        
    #     img = np.asarray(node_data)
    #     img = torch.as_tensor(img).clone()
    #     img = torch.clamp((img*255).byte(),0,255).byte()
        
        
        
        # t_img = torch.stack([self.transform(x) for x in img],dim=0)
        show_tensor_images(img.float()/255, scan_id+'_'+label)
        # break
    # break

