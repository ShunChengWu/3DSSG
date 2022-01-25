#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 16:44:42 2021

@author: sc
"""
import numpy as np
import trimesh
import json
def rand_24_bit():
    import random
    """Returns a random 24-bit integer"""
    return random.randrange(0, 16**6)
def color_dec():
    """Alias of rand_24 bit()"""
    return rand_24_bit()
def color_hex(num=rand_24_bit()):
    """Returns a 24-bit int in hex"""
    return "%06x" % num
def color_rgb(num=rand_24_bit()):
    """Returns three 8-bit numbers, one for each channel in RGB"""
    hx = color_hex(num)
    barr = bytearray.fromhex(hx)
    return (barr[0], barr[1], barr[2])

def scannet_get_instance_ply(plydata, segs, aggre, random_color=False):
    ''' map idx to segments '''
    seg_map = dict()
    for idx in range(len(segs['segIndices'])):
        seg = segs['segIndices'][idx]
        if seg in seg_map:
            seg_map[seg].append(idx)
        else:
            seg_map[seg] = [idx]
   
    ''' Group segments '''
    aggre_seg_map = dict()
    for segGroup in aggre['segGroups']:
        aggre_seg_map[segGroup['id']] = list()
        for seg in segGroup['segments']:
            aggre_seg_map[segGroup['id']].extend(seg_map[seg])
    assert(len(aggre_seg_map) == len(aggre['segGroups']))
    # print('num of aggre_seg_map:',len(aggre_seg_map))
    
    ''' Generate random colors '''
    if random_color:
        colormap = dict()
        for seg in aggre_seg_map.keys():
            colormap[seg] = color_rgb(rand_24_bit())
            
    ''' Over write label to segments'''
    # vertices = plydata.vertices
    try:
        labels = plydata.metadata['ply_raw']['vertex']['data']['label']
    except: labels = plydata.elements[0]['label']
    
    instances = np.zeros_like(labels)
    colors = plydata.visual.vertex_colors
    used_vts = set()
    for seg, indices in aggre_seg_map.items():
        s = set(indices)
        if len(used_vts.intersection(s)) > 0:
            raise RuntimeError('duplicate vertex')
        used_vts.union(s)
        for idx in indices:
            instances[idx] = seg
            if random_color:
                colors[idx][0] = colormap[seg][0]
                colors[idx][1] = colormap[seg][1]
                colors[idx][2] = colormap[seg][2]
    return plydata, instances

def load_scannet(pth_ply, pth_agg, pth_seg, verbose=False, random_color = False):
    ''' Load GT '''
    plydata = trimesh.load(pth_ply, process=False)        
    num_verts = plydata.vertices.shape[0]
    if verbose:print('num of verts:',num_verts)
    
    ''' Load segment file'''
    with open(pth_seg) as f:
        segs = json.load(f)
    if verbose:print('len(aggre[\'segIndices\']):', len(segs['segIndices']))
    segment_ids = list(np.unique(np.array(segs['segIndices']))) # get unique segment ids
    if verbose:print('num of unique ids:', len(segment_ids))
    
    ''' Load aggregation file'''
    with open(pth_agg) as f:
        aggre = json.load(f)
    # assert(aggre['sceneId'].split('scannet.')[1]==scan_id)
    # assert(aggre['segmentsFile'].split('scannet.')[1] == scan_id+args.segs)

    plydata,instances = scannet_get_instance_ply(plydata, segs, aggre,random_color=random_color )
    
    labels = plydata.metadata['ply_raw']['vertex']['data']['label'].flatten()
    points = plydata.vertices
    
    # the label is in the range of 1 to 40. 0 is unlabeled
    # instance 0 is unlabeled.
    return plydata, points, labels, instances