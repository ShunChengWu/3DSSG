#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:54:16 2020

@author: sc
"""
import os,sys
import numpy as np
from ssg.utils import util_ply
from ssg import define
from tqdm import tqdm
import argparse
import codeLib
from ssg.utils.util_data import read_all_scan_ids 
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger_py = logging.getLogger(__name__)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c','--config',default='./configs/config_default.yaml',required=False)
parser.add_argument('--overwrite', action='store_true', help='overwrite or not.')
parser.add_argument('--in_name', type=str, default='labels.instances.align.annotated.v2.ply', help='name of the output file.',required=False)
parser.add_argument('--save_name', type=str, default='color.align.ply', help='if empty set to out_name.',required=False)
parser.add_argument('--binary', action='store_true', help='output binary ply.')
parser.add_argument('--debug', action='store_true', help='debug.')
args = parser.parse_args()

def main():
    cfg = codeLib.Config(args.config)
    ''' Read split '''
    scan_ids = sorted(read_all_scan_ids(cfg.data.path_split))

    logger_py.info('there are {} sequences'.format(len(scan_ids)))
    for scan_id in tqdm(sorted(scan_ids)):
        if args.debug: print(scan_id)
        path = os.path.join(cfg.data.path_3rscan_data,scan_id)
        pth_out = os.path.join(path,args.save_name)
        
        if not args.overwrite:
            if os.path.isfile(pth_out):
                logger_py.debug('skip scan',scan_id)
                continue
        
        colored = util_ply.load_rgb(path,args.in_name,with_worker=False)
        util_ply.save_trimesh_to_ply(colored,os.path.join(path,args.save_name),binary = args.binary)
        if args.debug: break
    logger_py.info('done')
    
if __name__ == '__main__':
    main()
