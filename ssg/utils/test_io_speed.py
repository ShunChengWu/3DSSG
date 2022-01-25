#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 09:33:11 2021

@author: sc
"""
import h5py
import cProfile
import glob
import pickle
path_h5 = '/media/sc/SSD1TB/storage/kf_feature_tmp/bcb0fe1b-4f39-2c70-9f8c-2256ea9752ab.pkl'
path_pkl = '/media/sc/SSD1TB/storage/kf_feature_tmp/bcb0fe1b-4f39-2c70-9f8c-2256ea9752ab/*'
path_pkl_files = glob.glob(path_pkl)
def load_h5():
    def read():
        f = h5py.File(path_h5,'r')
        return f['vgg16']
    def get_all_data(f):
        size = f.shape[0]
        # print('h5size:',size)
        for i in range(size):
            return f[i,:]
    f = read()
    get_all_data(f)
    
def load_pkl():
    def read():
        # print('pklsize:',len(path_pkl_files))
        for path in path_pkl_files:
            with open(path, 'rb') as f:
                return pickle.load(f)
    read()

if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    load_h5()
    # load_pkl()
    pr.disable()
    pr.dump_stats('io_speed_cmp.dmp')