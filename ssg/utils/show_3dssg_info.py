#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 16:02:36 2021

@author: sc

check attributes and affordance
"""
import json

# path = '/media/sc/SSD1TB/dataset/3RScan/data/3RScan/3RScan.json'
# with open(path) as f:
#     data_3r = json.load(f)
    

# with open(path_obj) as f:
#     data_obj = json.load(f)
    
# path_rel = '/media/sc/SSD1TB/dataset/3RScan/data/3RScan/relationships.json'
# with open(path_rel) as f:
#     data_rel = json.load(f)
    
# relationship_count = dict()
# for scan in data_rel['scans']:
#     scan_id = scan['scan']
    
#     #relationships
#     for rel in scan['relationships']:
#         pass

def show_obj_properties(pth):
    with open(pth) as f:
        data_obj = json.load(f)
    affordances_count = dict()
    attributs_count = dict()
    labels_count = dict()
    
    num_obj_has_affordance = 0
    num_obj_has_attribute = dict()
    num_of_object=0
    for scan in data_obj['scans']:
        for obj in scan['objects']:
            num_of_object+=1
            if 'affordances' in obj:
                num_obj_has_affordance += 1
                for name in obj['affordances']:
                    if name not in affordances_count: affordances_count[name] = 0
                    affordances_count[name] += 1
            if 'attributes' in obj:
                for key, value in obj['attributes'].items():
                    if key not in attributs_count: 
                        attributs_count[key] = dict()
                        num_obj_has_attribute[key]=0
                    num_obj_has_attribute[key]+=1
                    for attri in value:
                        if attri not in attributs_count[key]: attributs_count[key][attri] = 0
                        attributs_count[key][attri] += 1
            if 'label' in obj:
                if obj['label'] not in labels_count: labels_count[obj['label']] = 0
                labels_count[obj['label']] += 1
    
    print('number of object has afforance:', num_obj_has_affordance/num_of_object)
    print('number of afforances:',len(affordances_count))
    for key, value in affordances_count.items():
        print('\t',key,'\t',value)
    
    print('number of attributes appears in objects:')
    for k,v in num_obj_has_attribute.items():
        print('\t',k,'\t',v/num_of_object)
        
    print('number of attributes:',len(attributs_count))
    for key, value in attributs_count.items():
        print('\t',key,'(',len(value),')')
        for kk, vv in value.items():
            print('\t\t',kk,vv)
    print('number of labels:', len(labels_count))
    # for key ,value in labels_count.items():
    #     print('\t',key,'\t',value)

path_obj = '/media/sc/SSD1TB/dataset/3RScan/data/3RScan/objects.json'
show_obj_properties(path_obj)

if False:
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
    pth_semseg='/media/sc/SSD1TB/dataset/3RScan/data/3RScan/ffa41874-6f78-2040-85a8-8056ac60c764/semseg.v2.json'
    show_semseg(pth_semseg)

def show_rel_properties(pth):
    with open(pth) as f:
        data_rel = json.load(f)
    scans = data_rel['scans']
    for scan in scans:
        scan_id = scan['scan']
        print(scan_id)

path_rel = '/media/sc/SSD1TB/dataset/3RScan/data/3RScan/relationships.json'
# show_rel_properties(path_rel)

