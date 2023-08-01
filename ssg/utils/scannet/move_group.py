#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 10:13:23 2021

@author: sc
"""
import h5py
from SensorData import SensorData
import argparse
import os, sys
import logging
from tqdm import tqdm

logging.basicConfig()
logger_py = logging.getLogger(__name__)
logger_py.setLevel('INFO')

flist='/media/sc/space1/dataset/scannet/Tasks/Benchmark/scannetv2_val.txt'
def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

h5f = h5py.File('/media/sc/space1/dataset/scannet/scans/images.h5', 'a')
h5f2 = h5py.File('/media/sc/SSD1TB/dataset/images_train.h5', 'a')


val_list = read_txt_to_list(flist)

logger_py.info('extracting gruops...')
fscenes = list()
for scene in h5f.keys():
    if scene not in val_list:
        '''move'''
        fscenes.append(scene)
logger_py.info('there are {} gruops to be moved.'.format(len(fscenes)))

logger_py.info('moving groups...')
for scene in tqdm(fscenes):
    h5f[scene].copy(h5f[scene], h5f2)
    del h5f[scene]
    # break
logger_py.info('done')
h5f.close()
h5f2.close()