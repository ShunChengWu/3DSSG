#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import open3d as o3d
from codeLib.utils import util
from enum import Enum

class SAMPLE_METHODS(Enum):
    BBOX=1
    RADIUS=2


def find_neighbors(points, segments, search_method:SAMPLE_METHODS, receptive_field:float=0.50,
                   debug=False, selected_keys:list=None):
    '''
    Return dict {seg_idx : set(neighbor indices)}

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.
    segments : TYPE
        DESCRIPTION.
    search_method : SAMPLE_METHODS
        DESCRIPTION.
    receptive_field : float, optional
        DESCRIPTION. The default is 0.50.
    debug : TYPE, optional
        DESCRIPTION. The default is False.
    selected_keys : list, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    # points = np.array(cloud.vertices.tolist())
    # segments = cloud.metadata['ply_raw']['vertex']['data']['objectId'].flatten()
    seg_ids = np.unique(segments)
    
    if selected_keys is not None:
        seg_ids = seg_ids.tolist()
        difference = set(selected_keys).difference(seg_ids)
        assert len(difference) == 0
        seg_ids = np.array(list(set(seg_ids).intersection(selected_keys)))
    
    ''' Get all segments '''
    trees = dict()
    segs_pts  = dict()
    bboxes = dict()
    # tmp_bboxes=dict()
    for idx in seg_ids:
        segs_pts[idx] = points[np.where(segments==idx)]
        trees[idx] = o3d.geometry.KDTreeFlann(segs_pts[idx][:,:3].transpose())
        bboxes[idx] = [segs_pts[idx][:,:3].min(0)-receptive_field,segs_pts[idx][:,:3].max(0)+receptive_field]
        # tmp_bboxes[idx] = [segs_pts[idx].min(0),segs_pts[idx].max(0)]

    
    if debug:
        seg_colors = dict()
        for index in seg_ids:
            seg_colors[index] = util.color_rgb(util.rand_24_bit())

    
    ''' Building Box Method '''      
    segs_neighbors = dict()
    if search_method == SAMPLE_METHODS.BBOX:
        ''' Search neighbor of each segment '''
        for seg_idx in seg_ids:
            bbox_q = bboxes[seg_idx]
            seg_n = segs_neighbors[int(seg_idx)]=list()
            for seg_tar_idx in seg_ids:            
                if seg_idx == seg_tar_idx: continue
                bbox_t = bboxes[seg_tar_idx]
                if (bbox_q[0] > bbox_t[1]).sum() + (bbox_t[0] > bbox_q[1]).sum() > 0: 
                    continue                
                seg_n.append(int(seg_tar_idx))
    elif search_method == SAMPLE_METHODS.RADIUS:
        # search neighbor for each segments
        for seg_id in seg_ids:
            def f_nn(seg_id:int, trees:dict, bboxes:dict, segs:dict, radknn:float):
                pts = segs[seg_id]
                bbox_q = bboxes[seg_id]
                neighbors = set()
                for tree_idx, tree in trees.items():
                    if tree_idx == seg_id:continue
                    if tree_idx in neighbors: continue
                    bbox_t = bboxes[tree_idx]
                    if (bbox_q[0] > bbox_t[1]).sum() + (bbox_t[0] > bbox_q[1]).sum() > 0: 
                        continue
                    
                    for i in range(len(pts)):
                        point = pts[i]
                        k, _, _ = tree.search_radius_vector_3d(point,radknn)
                        if k > 0: 
                            neighbors.add(int(tree_idx))
                            break
                return neighbors
            neighbors = list(f_nn(seg_id, trees, bboxes, segs_pts, receptive_field))
            neighbors = [int(n) for n in neighbors]
            segs_neighbors[int(seg_idx)] = neighbors
    return segs_neighbors