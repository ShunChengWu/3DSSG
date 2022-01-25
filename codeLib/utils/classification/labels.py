#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 09:41:09 2021

@author: sc
"""

import pandas

NYU40_Label_Names = [
    'wall',
'floor',
'cabinet',
'bed',
'chair',
'sofa',
'table',
'door',
'window',
'bookshelf',
'picture',
'counter',
'blinds',
'desk',
'shelves',
'curtain',
'dresser',
'pillow',
'mirror',
'floor mat',
'clothes',
'ceiling',
'books',
'refridgerator',
'television',
'paper',
'towel',
'shower curtain',
'box',
'whiteboard',
'person',
'night stand',
'toilet',
'sink',
'lamp',
'bathtub',
'bag',
'otherstructure',
'otherfurniture',
'otherprop',
]

SCANNET20_Label_Names = [
'wall',
'floor',
'cabinet',
'bed',
'chair',
'sofa',
'table',
'door',
'window',
'bookshelf',
'picture',
'counter',
'desk',
'curtain',
'refridgerator',
'shower curtain',
'toilet',
'sink',
"bathtub",
'otherfurniture',
]

def get_ScanNet_label_mapping(label_file:str, key:str='raw_category', value:str='id')->dict:
    '''
    label_file: the path to "scannetv2-labels.combined.tsv" from ScanNet dataset
    key and values can be:
        id	raw_category	category	count	nyu40id	eigen13id	nyuClass	
        nyu40class	eigen13class	ModelNet40	ModelNet10	ShapeNetCore55	synsetoffset	
        wnsynsetid	wnsynsetkey	mpcat40	mpcat40index
    '''
    data = pandas.read_csv(label_file, delimiter= '\t')
    keys = data[key]
    values = data[value]
    mapping = {k:v for k,v in zip(keys,values)}
    return mapping

def get_NYU40_color_palette():
    return [
        (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),  
       (247, 182, 210),		# desk
       (66, 188, 102), 
       (219, 219, 141),		# curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209), 
       (227, 119, 194),		# bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]