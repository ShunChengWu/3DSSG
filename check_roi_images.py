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
path = '/media/sc/SSD4TB/roi_2dssg_orbslam3/roi_images.h5'
file = h5py.File(path,'r')

path = '/home/sc/research/PersistentSLAM/python/3DSSG/data/3RScan_ScanNet20_2DSSG_ORBSLAM3/'
mode = 'train'
path = os.path.join(path,'relationships_%s.h5' % (mode))
sg_data = h5py.File(path,'r')
# sg_data = h5py.File(self.path_h5,'r')

# instance2labelName  = { int(key): node['label'] for key,node in object_data.items()  }

for scan_id in file:
    if scan_id not in sg_data: continue
    scan_data_raw = sg_data[scan_id]
    scan_data = raw_to_data(scan_data_raw)
    object_data = scan_data['nodes']

    scan_data = file[scan_id]
    for nid in scan_data:
        if int(nid) not in object_data: continue
        label = object_data[int(nid)]['label']
        
        node_data = scan_data[nid]
        
        img = np.asarray(node_data)
        img = torch.as_tensor(img).clone()
        img = torch.clamp((img*255).byte(),0,255).byte()
        
        
        
        # t_img = torch.stack([self.transform(x) for x in img],dim=0)
        show_tensor_images(img.float()/255, scan_id+'_'+label)
        # break
    # break

