#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 11:40:34 2020

@author: sc
"""

if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
    # sys.path.append('../python/GCN')
from utils.util import read_txt_to_list
from utils import define
import subprocess, os, sys, time, json, math, argparse
import multiprocessing as mp
import numpy as np

exe_path='/home/sc/research/SceneGraphFusion/bin/exe_GraphSLAM'

parser = argparse.ArgumentParser(description='Generate InSeg segmentation on 3RScan or ScanNet dataset.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--min_pyr_level', type=int, default=2, help='', required=False)
parser.add_argument('--depth_edge_threshold', type=float, default=0.98, help='', required=False)
parser.add_argument('--save_name', type=str, default='inseg.ply', help='', required=False)
parser.add_argument('--dataset',type=str,choices=['3RScan','ScanNet'],default='3RScan', help='type of dataset', required=True)
parser.add_argument('--type', type=str, default='train', choices=['train', 'validation'], help="which split of scans to use",required=True)

parser.add_argument('--thread', type=int, default=0, help='', required=False)
parser.add_argument('--overwrite', type=int, default=0, help='overwrite', required=False)
parser.add_argument('--rendered', type=int, default=1, help='use rendered depth (for ScanNet and 3RScan)', required=False)
parser.add_argument('--debug', type=int, default=0, help='', required=False)

args = parser.parse_args()

def process(pth_in, pth_out):
    startTime = time.time()
    try:
        output = subprocess.check_output([exe_path, 
                                          '--pth_in',pth_in, 
                                          '--pth_out',pth_out,
                                            '--save_name',str(args.save_name),
                                          '--rendered',str(args.rendered),
                                          '--min_pyr_level',str(args.min_pyr_level), # was 2
                                          '--depth_edge_threshold',str(args.depth_edge_threshold),
                                          '--save',str(1),
                                          '--save_graph','1',
                                          '--save_graph_ply','0',
                                          '--save_surfel_ply','1',
                                          '--save_time',str(0),
                                          '--binary', '1',
                     ],
            stderr=subprocess.STDOUT)
        sys.stdout.write(output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print('[Catched Error]', "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output.decode('utf-8'))) # omit errors
    endTime = time.time()
    return endTime-startTime



def gen_splits(pth_3rscan_json):
    with open(pth_3rscan_json,'r') as f:
        scan3r = json.load(f)
    
    train_list = list()
    val_list  = list()
    for scan in scan3r:
        ref_id = scan['reference']

        if scan['type'] == 'train':
            l = train_list
        elif scan['type'] == 'validation':
            l = val_list
        else:
            continue
        l.append(ref_id)
        for sscan in scan['scans']:
            l.append(sscan['reference'])
    
    return train_list ,val_list

def gen_splits_scannet(pth_train_txt,pth_test_txt):
    train_list = read_txt_to_list(pth_train_txt)
    val_list  = read_txt_to_list(pth_test_txt)
    return train_list, val_list

if __name__ == '__main__':
    print(args)
    
    if args.thread > 1:
        pool = mp.Pool(args.thread)
        pool.daemon = True
        
        
    ''' Read split '''
    if args.dataset == '3RScan':
        # train_scans = read_txt_to_list(os.path.join(define.ROOT_PATH,'files','train_scans.txt'))
        # val_scans = read_txt_to_list(os.path.join(define.ROOT_PATH,'files','validation_scans.txt'))
        train_scans,val_scans  = gen_splits(define.Scan3RJson_PATH)
    elif args.dataset == 'ScanNet':
        train_scans,val_scans = gen_splits_scannet(
            define.SCANNET_SPLIT_TRAIN,
            define.SCANNET_SPLIT_VAL
            )
    else:
        raise RuntimeError('unknown dataset type')
        
    # print('0cac7536-8d6f-2d13-8dc2-2f9d7aa62dc4' in train_scans)
    # print('0cac7536-8d6f-2d13-8dc2-2f9d7aa62dc4' in val_scans)
    # import sys
    # sys.exit()
      
    if args.type == 'train':
        scan_ids = train_scans
    elif args.type == 'validation':
        scan_ids = val_scans
    else:
        RuntimeError('unknown type')

    results=[]
    for scan_id in sorted(scan_ids):
        # scan_id = '4acaebcc-6c10-2a2a-858b-29c7e4fb410d'
        # --pth /media/sc/SSD1TB/dataset/3RScan/data/3RScan/0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca/sequence/
        if args.dataset == '3RScan':
            pth_in = os.path.join(define.DATA_PATH,scan_id,'sequence')
            pth_out = os.path.join(define.DATA_PATH,scan_id)
        elif args.dataset == 'ScanNet':
            pth_in = os.path.join(define.SCANNET_DATA_PATH,scan_id,scan_id+'.sens')
            pth_out = os.path.join(define.SCANNET_DATA_PATH,scan_id)
        else:
            raise RuntimeError("Unknown dataset type ", args.dataset)
            
        # check exist
        if not args.overwrite:
            if args.dataset == '3RScan':
                if os.path.isfile(os.path.join(define.DATA_PATH,scan_id,'graph.json')):
                    # print('skip',scan_id)
                    continue
            elif args.dataset == 'ScanNet':
                if os.path.isfile(os.path.join(define.SCANNET_DATA_PATH,scan_id,'graph.json')):
                    # print('skip',scan_id)
                    continue
        
        # continue
        
        # if scan_id != '2e369567-e133-204c-909a-c5da44bb58df':continue
        print('pth_in:',pth_in)
        print('pth_out:',pth_out+'/'+args.save_name)
        
        if args.thread > 1:
            results.append(
                pool.apply_async(process,(pth_in,pth_out)))
        else:
            results.append(process(pth_in,pth_out))
        if args.debug > 0:  break
    
    if args.thread > 1:
        pool.close()
        pool.join()
    if args.thread > 1:
        results = [r.get() for r in results]
    for r in results:
        print(r)
