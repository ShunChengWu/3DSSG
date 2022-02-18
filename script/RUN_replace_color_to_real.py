#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:54:16 2020

@author: sc
"""
# if __name__ == '__main__' and __package__ is None:
#     from os import sys
#     sys.path.append('../')
import os
import numpy as np

from ssg.utils import util_ply
from ssg import define
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--scans', type=str, 
#                         default='/media/sc/SSD1TB/dataset/3RScan/data/3RScan/', help='to the directory contains scans',required=False)
parser.add_argument('--txt', type=str, 
                    default='/home/sc/research/PersistentSLAM/python/3DSSG/data/ScanNet20_InSeg_Full/validation_scans.txt', 
                    help='a txt file contains a set of scene id.',required=False)
#/home/sc/research/PersistentSLAM/python/3DSSG/data/ScanNet20_InSeg_Full/validation_scans.txt
#/media/sc/space1/dataset/scannet/Tasks/Benchmark/scannetv2_train.txt
#/media/sc/space1/dataset/scannet/Tasks/Benchmark/scannetv2_val.txt
parser.add_argument('--in_name', type=str, default='labels.instances.align.annotated.v2.ply', help='name of the output file.',required=False)
parser.add_argument('--save_name', type=str, default='color.align.ply', help='if empty set to out_name.',required=False)
parser.add_argument('--binary', type=int, default=0, help='output binary ply.',required=False)
parser.add_argument('--debug', type=int, default=0, help='debug.',required=False)
parser.add_argument('--overwrite', action='store_false', help='overwrite',required=False)
args = parser.parse_args()


# path = '/media/sc/SSD1TB/dataset/3RScan/data/3RScan/ab835faa-54c6-29a1-9b55-1a5217fcba19'
# path = '/media/sc/space1/dataset/scannet/scans/scene0000_00'
# name = 'cvvseg.ply'
# plydata = util_ply.load_rgb(path,name)
# plydata.export('colored.ply')
# ply_label_align_mesh.write('colored.ply') 



def main():
    if args.save_name == "":
        args.save_name = args.out_name
    ''' Read split '''
    print('read split file..')
    ids = open(args.txt).read().splitlines()
    print('there are',len(ids),'sequences')
    for scan_id in tqdm(sorted(ids)):
        if args.debug>0: print(scan_id)
        path = os.path.join(define.DATA_PATH,scan_id)
        pth_out = os.path.join(path,args.save_name)
        
        if args.overwrite == 0:
            if os.path.isfile(pth_out):
                print('skip scan',scan_id)
                continue
        
        colored = util_ply.load_rgb(path,args.in_name,with_worker=False)
        util_ply.save_trimesh_to_ply(colored,os.path.join(path,args.save_name),binary = args.binary>0)
        if args.debug>0: break
    print('done')
    
if __name__ == '__main__':
    main()
