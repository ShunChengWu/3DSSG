#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:21:59 2020
@author: sc
This problem build up the scan and rescan list.
"""
if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
from pathlib import Path
import os 
import json
import argparse
import math
import numpy as np
from utils import define

def Parser():
    parser = argparse.ArgumentParser(description='Process some integers.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--p',type=float,default=0.9, help='split percentage.')
    parser.add_argument('--pth_out',type=str,default='./',help='output path')
    parser.add_argument('--type',type=str,choices=['3RScan','ScanNet'],default='3RScan', help='type of dataset')
    return parser

def gen_splits(pth_3rscan_json, train_valid_percent = 0.8):
    with open(pth_3rscan_json,'r') as f:
        scan3r = json.load(f)
    
    train_list = list()
    test_list  = list()
    for scan in scan3r:
        ref_id = scan['reference']

        if scan['type'] == 'train':
            l = train_list
        elif scan['type'] == 'validation':
            l = test_list
        else:
            continue
        l.append(ref_id)
        for sscan in scan['scans']:
            l.append(sscan['reference'])
    
    n_train = int(math.ceil(train_valid_percent*len(train_list)))
    
    sample_train = np.random.choice(range(len(train_list)),n_train,replace=False).tolist()
    sample_valid = set(range(len(train_list))).difference(sample_train)
    assert  len(sample_train) + len(sample_valid) == len(train_list)
    
    sample_train = [train_list[i] for i in sample_train]
    sample_valid = [train_list[i] for i in sample_valid]
    
    print('train:',len(sample_train),'validation:',len(sample_valid),'test',len(test_list))
            
    return sample_train, sample_valid, test_list

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def gen_splits_scannet(pth_train_txt,pth_test_txt, train_valid_percent=0.8):
    train_list = read_txt_to_list(pth_train_txt)
    test_list  = read_txt_to_list(pth_test_txt)
    n_train = int(math.ceil(train_valid_percent*len(train_list)))
    
    sample_train = np.random.choice(range(len(train_list)),n_train,replace=False).tolist()
    sample_valid = set(range(len(train_list))).difference(sample_train)
    assert  len(sample_train) + len(sample_valid) == len(train_list)
    
    sample_train = [train_list[i] for i in sample_train]
    sample_valid = [train_list[i] for i in sample_valid]
    
    print('train:',len(sample_train),'validation:',len(sample_valid),'test',len(test_list))
            
    return sample_train, sample_valid, test_list

def save(path, scans):
    with open(path,'w') as f:
        for scan in scans:
            f.write(scan+"\n")    

if __name__ == '__main__':
    args = Parser().parse_args()
    if args.type == '3RScan':
        train_scans,validation_scans,test_scans = gen_splits(define.Scan3RJson_PATH,args.p)
    elif args.type == 'ScanNet':
        train_scans,validation_scans,test_scans = gen_splits_scannet(
            define.SCANNET_SPLIT_TRAIN,
            define.SCANNET_SPLIT_VAL,
            args.p
            )
    else:
        raise RuntimeError('')
    
    Path(args.pth_out).mkdir(parents=True, exist_ok=True)
    save(os.path.join(args.pth_out,'train_scans.txt'),train_scans)
    save(os.path.join(args.pth_out,'validation_scans.txt'),validation_scans)
    save(os.path.join(args.pth_out,'test_scans.txt'),test_scans)