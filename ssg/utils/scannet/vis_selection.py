#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:55:34 2021

@author: sc

see selected objects from graph
"""
import os
import numpy as np
import h5py
from util_scannet import load_scannet

datadir='/media/sc/space1/dataset/scannet/scans/'    
datafolder = '/media/sc/space1/dataset/scannet_detections/2dgt'
scan2shape2stan='/home/sc/research/lfd_lfdc_plfd/make_sequences/scan2stan2shap.txt'
path_graph = '/media/sc/space1/dataset/scannet_detections/graph.h5'
data = h5py.File(path_graph,'r')
for scene in data:
    segs_f=datadir+'/'+scene+'/'+scene+'_vh_clean_2.0.010000.segs.json'
    if not os.path.isfile(segs_f):
      print('no segs.json found for scene',segs_f)
      continue
    ag_f=datadir+'/'+scene+'/'+scene+'_vh_clean.aggregation.json'
    if not os.path.isfile(ag_f):
      print('no aggregation.json found for scene',ag_f)
      continue
    label_f=datadir+'/'+scene+'/'+scene+'_vh_clean_2.labels.ply'
    if not os.path.isfile(label_f):
      print('no label file found for scene',label_f)
      continue
    
    plydata, points, labels, instances = load_scannet(label_f, ag_f,segs_f,verbose=True)
    
    nodes = data[scene]['nodes']
    obj_ids = nodes.attrs['seg2idx'][:,0]
    
    instance_ids = np.unique(instances)
    for gt_id in instance_ids:
        if gt_id in obj_ids: continue
        segment_indices = np.where(instances == gt_id)[0]
        for index in segment_indices:
            plydata.visual.vertex_colors[index][:3] = [0,0,0]
    plydata.show()
data.close()