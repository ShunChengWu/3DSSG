#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 17:01:57 2021

@author: sc
"""
import h5py

def read(path):
    data = h5py.File(path, 'r')
    return data
    
    
def list2dict(x:list):
    return {data[0]:data[1] for data in x}
def inversedict(x:dict):
    return {v:k for k,v in x.items()}
def get_name(x:str):
    return x.decode('utf8')
    

if __name__ =='__main__':
    data = read('/media/sc/space1/dataset/scannet_detections/graph.h5')
    
    for scene, value in data.items():
        kfs = value['kfs']
        nodes = value['nodes']
        n_seg2idx=list2dict(nodes.attrs['seg2idx'])
        
        '''check obj label in image bbox'''
        for f_id, dd in kfs.items():
            seg2idx = list2dict(dd.attrs['seg2idx'])
            idx2seg = inversedict(seg2idx)
            for box_id in range(dd.shape[0]):
                obj_seg_id = idx2seg[box_id]
                obj_id = n_seg2idx[obj_seg_id]
                name=get_name(nodes[obj_id])
                print(f_id,box_id,obj_seg_id,obj_id,name)
    data.close()