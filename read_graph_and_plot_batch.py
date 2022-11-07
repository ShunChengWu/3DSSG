#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 18:06:26 2021

@author: sc
"""
import trimesh
import os 
import json
import numpy as np
from codeLib.utils.util import read_txt_to_list
from ssg.utils import util_label
from ssg.utils import util_merge_same_part
import h5py
import graphviz
import operator
from ssg.utils.util_data import raw_to_data, cvt_all_to_dict_from_h5
from codeLib.common import rgb_2_hex
from codeLib.common import rand_24_bit, color_rgb

color_correct = '#0CF369' # green
color_wrong = '#FF0000' # red
color_missing_pd = '#A9A2A2' # gray
color_missing_gt = '#0077FF' #blue

def draw_prediction(scan_id, nodes,edges, colors):
    g = graphviz.Digraph(comment=scan_id,format='png',
                         node_attr={'shape': 'circle',
                                    'style': 'filled',
                                    'fontname':'helvetica',
                                    'color': 'lightblue2'},
                         edge_attr={'splines':'spline'},
                         rankdir = 'TB')
    for idx, name in nodes.items():
        g.node(idx, idx + '_' + name, color=colors[util_label.nyu40_name_to_id(name)])
    for edge, name in edges.items():
        f = edge.split('_')[0]
        t = edge.split('_')[1]
        if f not in nodes or t not in nodes: continue
        g.edge(f,t, label=name)
    return g


def draw_cluster(scan_id, inst_groups:dict, nodes,edges, colors):
    g = graphviz.Digraph(comment=scan_id,format='pdf')

    
    # g = graphviz.Digraph(comment=scan_id,format='pdf',
    #                      node_attr={'shape': 'circle',
    #                                 'style': 'filled',
    #                                 'fontname':'helvetica',
    #                                 'color': 'lightblue2'},
    #                      edge_attr={'splines':'spline'},
    #                      graph_attr={'rankdir': 'TB'})
    
    for k,v in inst_groups.items():
        with g.subgraph(name = 'cluster_'+k) as c:
            name = nodes[k]
            if name == 'none':continue
            color = '#%02x%02x%02x' % colors[util_label.nyu40_name_to_id(name)+1]

            c.attr(label=name)            
            c.attr(style='filled', color=color)
            
            
            # c.node_attr.update(style='filled', color=color)
            # c.edges([('a0', 'a1'), ('a1', 'a2'), ('a2', 'a3')])
            
            pre = []
            for idx in v:
                # name = nodes[idx]
                # color = '#%02x%02x%02x' % colors[labels_utils.nyu40_name_to_id(name)+1]
                
                c.node(idx, idx,{'shape':'circle'})
                # c.node(idx, idx, color=color)
                if len(pre)>0:
                    g.edge(pre[-1],idx)
                # pre.append(idx)
            
    return g
    
def draw_evaluation(scan_id, node_pds, edge_pds, node_gts, edge_gts, none_name = 'UN', 
                    pd_only=False, gt_only=False, ignore_none:bool=True):
    g = graphviz.Digraph(comment=scan_id,format='png',
                         node_attr={'shape': 'circle',
                                    'style': 'filled',
                                    'fontname':'helvetica',
                                    'color': 'lightblue2'},
                         edge_attr={'splines':'spline'},
                         graph_attr={'rankdir': 'TB'})
    nodes = set(node_pds.keys()).union(node_gts.keys())
    for idx in nodes:
        name_gt=none_name
        name_pd=none_name
        if idx not in node_pds: 
            color = color_missing_pd
        else:
            name_pd = node_pds[idx]
        if idx not in node_gts:
            color = color_missing_gt
        else:
            name_gt = node_gts[idx]
        if idx in node_pds and idx in node_gts:
            color = color_correct if node_gts[idx] == node_pds[idx] else color_wrong
        g.node(idx,str(idx) + '_' + name_pd+'('+name_gt+')',color=color)
        
        
    edges = set(edge_pds.keys()).union(edge_gts.keys())
    for edge in edges:
        '''
        For each edge there may have multiple labels. 
        If non ground truth labels are given, set to missing_gt
        If gt labels are given, find the union and difference.
        '''
        f = edge.split('_')[0]
        t = edge.split('_')[1]    
        names_gt=list()
        names_pd=list()
        if edge in edge_pds:
            if ignore_none: 
                if edge_pds[edge] == 'none': continue
            names_pd.append( edge_pds[edge] )
            
        if edge in edge_gts:
            if ignore_none: 
                if edge_gts[edge] == 'none': continue
            names_gt.append( edge_gts[edge] )
            
        names_pd = set(names_pd).difference([none_name])
        names_gt = set(names_gt).difference([none_name])
        # if len(names_gt) == 0: # missing gt
        #     for name in names_pd:
        #         g.edge(f,t,label=name+'('+none_name+')',color=color_missing_gt)            
        # else:
        #     corrects = set(names_gt).intersection(names_pd)
        #     wrongs = set(names_gt).difference(names_pd)
        #     for name in corrects:
        #         g.edge(f,t,label=name_pd+'('+name_gt+')',color=color_correct)
        #     for name in wrongs:
        #         name_gt = name if name in names_gt else none_name
        #         name_pd = name if name in names_pd else none_name
        #         g.edge(f,t,label=name_pd+'('+name_gt+')',color=color_wrong)    
                
        if len(names_gt) > 0:
            intersection = set(names_gt).intersection(names_pd) # match prediction
            diff_gt = set(names_gt).difference(intersection) # unmatched gt
            diff_pd = set(names_pd).difference(intersection) # unmatched pd
            for name in intersection:
                g.edge(f,t,label=name+'('+name+')',color=color_correct)
            if not gt_only:
                for name_pd in diff_pd: # in pd but not in gt
                    g.edge(f,t,label=name_pd+'('+none_name+')',color=color_wrong)
            if not pd_only:
                for name_gt in diff_gt: # in gt but not in pd
                    g.edge(f,t,label=none_name+'('+name_gt+')',color=color_wrong) # color_missing_pd
        elif len(names_gt) == 0: # missing gt
            if not gt_only:
                for name_pd in names_pd:
                    g.edge(f,t,label=name_pd+'('+none_name+')',color=color_missing_gt) # color_missing_pd
    return g
    
        
# def evaluate_prediction(pd:map, gt:map, names:list, title:str, gt_only = False):
#     c_mat = np.zeros([len(names)+1,len(names)+1])
#     c_UNKNOWN = len(names)
#     '''
#     For every predicted entry, check their corresponding gt. 
#     if predicted idx it not in gt and gt_only is one, continue. Otherwise 
#     count it as wrong prediction.
#     '''
#     for idx, name in pd.items():
#         pd_idx = names.index(name)
#         if idx not in gt:
#             if gt_only: continue
#             else: gt_idx = c_UNKNOWN
#         else:
#             gt_idx = names.index(gt[idx])
#         c_mat[gt_idx][pd_idx] += 1
        
#     '''
#     For every gt, if gt not in pd consider it as a wrong prediction. 
#     The condition of gt present in pd is covered in the previous stage.
#     '''
#     for idx, name in gt.items():
#         gt_idx = names.index(name)
#         if idx not in pd:
#             pd_idx = c_UNKNOWN
#         c_mat[gt_idx][pd_idx] += 1    
        
#     names_ = names + ['missed']
#     plot_confusion_matrix(c_mat, 
#                           target_names=names_, 
#                           title=title,
#                           plot_text=False,)
#     return c_mat

def process_gt(nodes,edges):
    nodes_gts = dict()
    for idx, pd in nodes.items():
        nodes_gts[str(idx)] = pd
        pass
    edges_gts = dict()
    for edge in edges:
        name = str(edge[0])+'_'+str(edge[1])
        if name not in edges_gts:
            edges_gts[name] = list()
        edges_gts[name].append(edge[3])
    return nodes_gts, edges_gts

def process_pd(nodes,edges):
    ns = dict()
    es = dict()
    for k, v in nodes.items():
        if isinstance(v,dict):# full output
            vv = max(v.items(), key=operator.itemgetter(1))[0]
            ns[k] = vv
        else:
            ns[k] = v
    
    for k, v in edges.items():
        if isinstance(v, dict):
            vv = max(v.items(), key=operator.itemgetter(1))[0]
            es[k] = vv
        else:
            es[k] = v
            
    return ns, es


def read_classes(read_file):  # load list of relationships
    relationships = [] 
    with open(read_file, 'r') as f: 
        for line in f: 
            relationship = line.rstrip().lower() 
            relationships.append(relationship) 
    return relationships 


# def show_semseg(pth):
#     with open(pth) as f:
#         data = json.load(f)
#     scan_id = data['scan_id']
#     for group in data['segGroups']:
#         group['objectId']
#         group['id']
#         group['partId']
#         group['index']
#         group['obb']
#         group['label']

def create_box(dimensions, width = 0.01):
    lines = list()
    lines.append( trimesh.creation.box( (dimensions[0]+width,width,width) ).apply_translation(((0,-0.5*dimensions[1],-0.5*dimensions[2]))) )
    lines.append( trimesh.creation.box( (dimensions[0]+width,width,width) ).apply_translation(((0,-0.5*dimensions[1],0.5*dimensions[2]))) )
    lines.append( trimesh.creation.box( (dimensions[0]+width,width,width) ).apply_translation(((0,0.5*dimensions[1],-0.5*dimensions[2]))) )
    lines.append( trimesh.creation.box( (dimensions[0]+width,width,width) ).apply_translation(((0,0.5*dimensions[1],0.5*dimensions[2]))) )
    
    lines.append( trimesh.creation.box( (width,dimensions[1],width) ).apply_translation((-0.5*dimensions[0],0,-0.5*dimensions[2])) )
    lines.append( trimesh.creation.box( (width,dimensions[1],width) ).apply_translation((-0.5*dimensions[0],0,0.5*dimensions[2])) )
    lines.append( trimesh.creation.box( (width,dimensions[1],width) ).apply_translation((0.5*dimensions[0],0,-0.5*dimensions[2])) )
    lines.append( trimesh.creation.box( (width,dimensions[1],width) ).apply_translation((0.5*dimensions[0],0,0.5*dimensions[2])) )
    
    lines.append( trimesh.creation.box( (width,width,dimensions[2]) ).apply_translation((-0.5*dimensions[0],-0.5*dimensions[1],0)) )
    lines.append( trimesh.creation.box( (width,width,dimensions[2]) ).apply_translation((-0.5*dimensions[0],0.5*dimensions[1],0)) )
    lines.append( trimesh.creation.box( (width,width,dimensions[2]) ).apply_translation((0.5*dimensions[0],-0.5*dimensions[1],0)) )
    lines.append( trimesh.creation.box( (width,width,dimensions[2]) ).apply_translation((0.5*dimensions[0],0.5*dimensions[1],0)) )
    
    box = trimesh.util.concatenate(lines )
    # box.apply_translation(centroid)
    return box

def draw(scan_id,scan_data, T,node_pds,node_gts,seg2iid):
    clr_plat = util_label.get_NYU40_color_palette()
    nodes = scan_data['nodes']
    # kfs = scan_data['kfs']
    meshes = list()
    for key,node in nodes.items():
        node_id = int(key)
        if node_id == 0:continue
        if seg2iid is not None:
            if node_id not in seg2iid: continue
            seg_id = str(seg2iid[node_id])
            if seg_id not in node_pds:continue
            if node_pds[seg_id] == 'none': continue
            clr = clr_plat[ util_label.nyu40_name_to_id(node_pds[seg_id])+1]
            gt_clr = clr_plat[ util_label.nyu40_name_to_id(node_gts[seg_id])+1]
            
            print(clr,rgb_2_hex(clr),node_pds[seg_id],node_gts[seg_id], rgb_2_hex(gt_clr))
        else:
            clr = color_rgb(rand_24_bit())
        
        Rinv = np.array(node['rotation']).reshape(3,3)
        R = np.transpose(Rinv)
        center = np.array(node['center']) 
        # center = Rinv @ center # transform back to object coordinate to generate box
        dims = np.array(node['dimension'])
        box = create_box(dims, 0.05)
        
        # print('center',node['center'])
        # print('R',R)
        # print('center',node['center'])
        # print('dims',node['dimension'])
        
        mat44 = np.eye(4)
        mat44[:3,:3] = R
        mat44[:3,3] = center
        box.apply_transform(mat44)
        
        '''color'''
        box.visual.vertex_colors[:,:3] = clr
        
        # if center[2]>1:continue
        # if center[2]+dims[2]>1:continue
        # break
        meshes.append(box)
    OBJ_NAME = 'mesh.refined.obj'
    # OBJ_NAME = 'labels.instances.align.annotated.v2.ply'
    pth_folder = '/media/sc/SSD1TB/dataset/3RScan/data/3RScan/'+scan_id+'/'+ OBJ_NAME
    
    mesh = trimesh.load(pth_folder, process=False)
    mesh = mesh.apply_transform(T)
    meshes.append(mesh)
    
    pth_ply ='/media/sc/SSD1TB/dataset/3RScan/data/3RScan/'+scan_id+'/2dssg_orbslam3.ply'
    # pth_ply = '/home/sc/research/ORB_SLAM3/bin/15/map.ply'
    mesh = trimesh.load(pth_ply, process=False)
    
    
    # unique_colors = np.unique(mesh.visual.vertex_colors,axis=0)
    meshes.append(mesh)
    # x = trimesh.util.concatenate( meshes )
    # x.export('bbox.ply')
    # trimesh.Scene(meshes).show()
    return meshes


# Read predictions
path_pd = '/home/sc/research/PersistentSLAM/python/3DSSG/experiments/JointSSG_orbslam_l20_11_4/predictions.json'
with open(path_pd,'r') as f:
    data_pds = json.load(f)

# load transformation
scan_dict = dict()
with open('/media/sc/SSD1TB/dataset/3RScan/data/3RScan/3RScan.json','r') as f:
    scan3r_data = json.load(f)
    for scan_data in scan3r_data:
        scan_dict[scan_data['reference']] = scan_data
        for sscan in scan_data['scans']:
            scan_dict[sscan['reference']] = sscan

ttype = 'train'
test_scans = read_txt_to_list('./data/3RScan_ScanNet20_2DSSG_ORBSLAM3/{}_scans.txt'.format(ttype))
relationship_data= h5py.File('./data/3RScan_ScanNet20_2DSSG_ORBSLAM3/relationships_{}.h5'.format(ttype),'r')

pd_only=False
gt_only=False
pth_out='./'

for scan_id in test_scans:
    scan_id = '6e67e550-1209-2cd0-8294-7cc2564cf82c'
    scan_id = '4acaebcc-6c10-2a2a-858b-29c7e4fb410d'
    print(scan_id)
    
    if 'transform' in scan_dict[scan_id]:
        T = np.asarray(scan_dict[scan_id]['transform']).reshape(4,4).transpose()
    else:
        T = np.eye(4,4)
    pth_json='/media/sc/SSD1TB/dataset/3RScan/data/3RScan/'+scan_id+'/graph_2dssg_orbslam3.json'
    # pth_json = '/home/sc/research/ORB_SLAM3/bin/15/graph.json'
    
    
    '''Load bounding box data'''
    with open(pth_json ) as f:  
        data = json.load(f)
        
    rel_data = relationship_data[scan_id]
    rel_data = raw_to_data(rel_data)
    seg2iid = {nid: node['instance_id']  for nid,node in rel_data['nodes'].items()}
        
    '''load predictions'''
    if scan_id in data_pds:
        prediction = data_pds[scan_id]
    
        node_pds, edge_pds = prediction['pd']['nodes'],prediction['pd']['edges']
        node_gts, edge_gts = prediction['gt']['nodes'],prediction['gt']['edges']
        
        node_pds,edge_pds = process_pd(node_pds,edge_pds)
        inst_groups = util_merge_same_part.collect(node_pds, edge_pds)
        
        # g = draw_cluster(scan_id,inst_groups, node_pds,edge_pds,util_label.get_NYU40_color_palette())
        g = draw_evaluation(scan_id, node_pds, edge_pds, node_gts, edge_gts, pd_only=pd_only,gt_only=gt_only)
        g.render(os.path.join(pth_out,scan_id+'_graph'),view=False)
    else:
        node_pds=node_gts=seg2iid=None
    
    
    '''draw'''
    if 'nodes' not in data:
        for scan_id, scan_data in data.items():
            # scan_data = data
            meshes = draw(scan_id,scan_data, T, node_pds,node_gts,seg2iid)
            trimesh.Scene(meshes).show()
    else:
        scan_data = data
        meshes = draw(scan_id,scan_data, T, node_pds,node_gts,seg2iid)
        trimesh.Scene(meshes).show()
        
    break