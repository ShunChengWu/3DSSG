#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 19:30:13 2021

@author: sc

This script read all h5 file in a given folder with the given pattern, merging
them into a single h5 file with external link.
"""
import h5py
import glob
import argparse
import os
from collections import defaultdict
parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f','--folder',default='/media/sc/Backup/scannet/',help='directory')
parser.add_argument('-n','--name',default='images_*.h5',help='name pattern')
parser.add_argument('-o','--output_name',default='images.h5', help='output name')
args = parser.parse_args()

entries = defaultdict(list)
for path in glob.glob(os.path.join(args.folder,args.name)):
    name = path.split('/')[-1]
    print(path,name)
    # try:
    with h5py.File(path, 'r') as h5f:
        entries[name]=list(h5f.keys())
    # except:
    #     pass

'''create and link'''
with h5py.File(os.path.join(args.folder,args.output_name), 'w') as h5f:
    for k, v in entries.items():
        for vv in v:
            link = h5py.ExternalLink(k, vv)
            h5f[vv] = link