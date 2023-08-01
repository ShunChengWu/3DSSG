#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 11:44:41 2021

@author: sc
"""
import trimesh
import os 
import json
import numpy as np

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

def create_box(centroid,dimensions, width = 0.01):
    lines = list()
    lines.append( trimesh.creation.box( (dimensions[0],width,width) ).apply_translation(((0,-0.5*dimensions[1],-0.5*dimensions[2]))) )
    lines.append( trimesh.creation.box( (dimensions[0],width,width) ).apply_translation(((0,-0.5*dimensions[1],0.5*dimensions[2]))) )
    lines.append( trimesh.creation.box( (dimensions[0],width,width) ).apply_translation(((0,0.5*dimensions[1],-0.5*dimensions[2]))) )
    lines.append( trimesh.creation.box( (dimensions[0],width,width) ).apply_translation(((0,0.5*dimensions[1],0.5*dimensions[2]))) )
    
    lines.append( trimesh.creation.box( (width,dimensions[1],width) ).apply_translation((-0.5*dimensions[0],0,-0.5*dimensions[2])) )
    lines.append( trimesh.creation.box( (width,dimensions[1],width) ).apply_translation((-0.5*dimensions[0],0,0.5*dimensions[2])) )
    lines.append( trimesh.creation.box( (width,dimensions[1],width) ).apply_translation((0.5*dimensions[0],0,-0.5*dimensions[2])) )
    lines.append( trimesh.creation.box( (width,dimensions[1],width) ).apply_translation((0.5*dimensions[0],0,0.5*dimensions[2])) )
    
    lines.append( trimesh.creation.box( (width,width,dimensions[2]) ).apply_translation((-0.5*dimensions[0],-0.5*dimensions[1],0)) )
    lines.append( trimesh.creation.box( (width,width,dimensions[2]) ).apply_translation((-0.5*dimensions[0],0.5*dimensions[1],0)) )
    lines.append( trimesh.creation.box( (width,width,dimensions[2]) ).apply_translation((0.5*dimensions[0],-0.5*dimensions[1],0)) )
    lines.append( trimesh.creation.box( (width,width,dimensions[2]) ).apply_translation((0.5*dimensions[0],0.5*dimensions[1],0)) )
    
    box = trimesh.util.concatenate(lines )
    return box
pth_semseg='/media/sc/SSD1TB/dataset/3RScan/data/3RScan/4acaebcc-6c10-2a2a-858b-29c7e4fb410d/semseg.v2.json'
# Load obb information
with open(pth_semseg) as f: 
    data = json.load(f)
scan_id = data['scan_id']


meshes = list()
for group in data['segGroups']:
    group['objectId']
    group['id']
    group['partId']
    group['index']
    obb = group['obb']
    group['label']
    # print(obb)
    
    box = create_box(obb['centroid'],obb['axesLengths'])
    mat44 = np.eye(4)
    mat44[:3,:3] = np.array(obb['normalizedAxes']).reshape(3,3).transpose()
    mat44[:3,3] = obb['centroid']
    box.apply_transform(mat44)
    meshes.append(box)
    
# Load PLY
OBJ_NAME = 'mesh.refined.obj'
pth_folder = '/media/sc/SSD1TB/dataset/3RScan/data/3RScan/'+scan_id+'/'+ OBJ_NAME

mesh = trimesh.load(pth_folder, process=False)

meshes.append(mesh)
# trimesh.util.concatenate( [ box,box] )
trimesh.Scene(meshes).show()



# Draw bbox