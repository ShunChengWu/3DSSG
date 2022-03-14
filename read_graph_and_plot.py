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
from codeLib.common import rand_24_bit, color_rgb
from codeLib.geoemetry.common import create_box

def show_semseg(pth):
    with open(pth) as f:
        data = json.load(f)
    scan_id = data['scan_id']
    for group in data['segGroups']:
        group['objectId']
        group['id']
        group['partId']
        group['index']
        group['obb']
        group['label']

# scan_id = '8eabc453-5af7-2f32-859d-405eb6a2e0d0'
# scan_id = '02b33e01-be2b-2d54-93fb-4145a709cec5'
# scan_id = '2e369549-e133-204c-91af-a19767a23bf2'
# scan_id = '8eabc453-5af7-2f32-859d-405eb6a2e0d0'
# scan_id = '8eabc447-5af7-2f32-8712-301083e291b3'
# scan_id = 'fcf66d88-622d-291c-871f-699b2d063630'
# scan_id = 'f62fd5fd-9a3f-2f44-883a-1e5cf819608e'
# # scan_i d= 'a644cb93-0ee5-2f66-9efb-b16adfb14eff'
# pth_json = '/media/sc/SSD1TB/dataset/3RScan/data/3RScan/'+scan_id+'/graph.json'

scan_id = '0cac757e-8d6f-2d13-8f1d-ef6123d53655'
scan_id = '0cac757e-8d6f-2d13-8f1d-ef6123d53655'
scan_id = '4acaebcc-6c10-2a2a-858b-29c7e4fb410d'
# scan_id = '095821f7-e2c2-2de1-9568-b9ce59920e29'
pth_json='/media/sc/SSD1TB/dataset/3RScan/data/3RScan/'+scan_id+'/graph_2dssg_orbslam3.json'

pth_json = '/home/sc/research/ORB_SLAM3/bin/test/graph_2dssg.json'

def render(nodes):
    meshes=list()
    for key,node in nodes.items():
        node_id = int(key)
        if node_id == 0:continue
        Rinv = np.array(node['rotation']).reshape(3,3)
        R = np.transpose(Rinv)
        center = np.array(node['center']) 
        # center = Rinv @ center # transform back to object coordinate to generate box
        dims = np.array(node['dimension'])
        box = create_box(dims,0.05)
        
        # print('center',node['center'])
        # print('R',R)
        # print('center',node['center'])
        # print('dims',node['dimension'])
        
        mat44 = np.eye(4)
        mat44[:3,:3] = R
        mat44[:3,3] = center
        box.apply_transform(mat44)
        
        if 'color' in node:
            box.visual.vertex_colors[:,:3] = [node['color'][2],node['color'][1],node['color'][0] ]
        else:
            box.visual.vertex_colors[:,:3] = color_rgb(rand_24_bit())
        if center[2]>1:continue
        if center[2]+dims[2]>1:continue
        # break
        meshes.append(box)
    return meshes

with open(pth_json ) as f: 
    data = json.load(f)
if 'nodes' not in data:
    for scan_id, scan_data in data.items():
        nodes = scan_data['nodes']
        # kfs = scan_data['kfs']
        meshes = render(nodes)
        
else:
    scan_data = data
    nodes = scan_data['nodes']
    # kfs = scan_data['kfs']
    meshes = render(nodes)
    # meshes = list()
    # for key,node in nodes.items():
        
    #     node_id = int(key)
    #     if node_id == 0:continue
    #     Rinv = np.array(node['rotation']).reshape(3,3)
    #     R = np.transpose(Rinv)
    #     center = np.array(node['center']) 
    #     # center = Rinv @ center # transform back to object coordinate to generate box
    #     dims = np.array(node['dimension'])
    #     box = create_box(dims)
        
    #     # print('center',node['center'])
    #     # print('R',R)
    #     # print('center',node['center'])
    #     # print('dims',node['dimension'])
        
    #     mat44 = np.eye(4)
    #     mat44[:3,:3] = R
    #     mat44[:3,3] = center
    #     box.apply_transform(mat44)
        
    #     if center[2]>1:continue
    #     if center[2]+dims[2]>1:continue
        
    #     box.visual.vertex_colors[:,:3] = clr
        
    #     # break
    #     meshes.append(box)
    
# trimesh.Scene(meshes).show()

# meshes = list()
# for group in data['segGroups']:
#     group['objectId']
#     group['id']
#     group['partId']
#     group['index']
#     obb = group['obb']
#     group['label']
#     # print(obb)
    
#     box = create_box(obb['centroid'],obb['axesLengths'])
#     mat44 = np.eye(4)
#     mat44[:3,:3] = np.array(obb['normalizedAxes']).reshape(3,3).transpose()
#     mat44[:3,3] = obb['centroid']
#     box.apply_transform(mat44)
#     meshes.append(box)
    
# # Load PLY
OBJ_NAME = 'mesh.refined.obj'
OBJ_NAME = 'labels.instances.align.annotated.v2.ply'
pth_folder = '/media/sc/SSD1TB/dataset/3RScan/data/3RScan/'+scan_id+'/'+ OBJ_NAME

x = trimesh.util.concatenate( meshes )
x.export('bbox.ply')

mesh = trimesh.load(pth_folder, process=False)
meshes.append(mesh)

# trimesh.util.concatenate( [ box,box] )
trimesh.Scene(meshes).show()



# Draw bbox