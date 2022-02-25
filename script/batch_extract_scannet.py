import os
import time
import argparse
import subprocess
import multiprocessing as mp
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map 
# import ssg2d
# from ssg2d.models import encoder
# from torchvision import transforms
# import os,json, argparse
# import open3d as o3d
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# import codeLib
# from codeLib.common import normalize_imagenet, save_obj, load_obj
# from ssg2d.utils import util_data
# import ssg2d.define as define
# import subprocess, os, sys, time
# import time

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def Parser(add_help=True):
    parser = argparse.ArgumentParser(description='Process some integers.', formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                      add_help=add_help)
    parser.add_argument('--reader_path', type=str, default='/media/sc/space1/dataset/scannet/SensReader/python/reader.py', help='the path to the scannet python reader python script')
    parser.add_argument('--pth_in', type=str, default='/media/sc/space1/dataset/scannet/scans/', help='')
    parser.add_argument('--pth_out', type=str, default='/media/sc/space1/dataset/scannet/scans/', help='')
    parser.add_argument('--txt', type=str, default='/media/sc/space1/dataset/scannet/Tasks/Benchmark/scannetv2_val.txt', help='list of scans')
    parser.add_argument('--thread', type=int, default=4, help='The number of threads to be used.')
    return parser

def run_process(arg_list):
    startTime = time.time()
    try:
        process = subprocess.Popen(
            arg_list, stdout=subprocess.PIPE)
        while True:
            output = process.stdout.readline()
            if process.poll() is not None:
                break
            if output:
                print(output.strip())
        output = process.poll()
        good = True
    except subprocess.CalledProcessError as e:
        print('[Catched Error]', "command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output.decode('utf-8'))) # omit errors
        good = False
    endTime = time.time()
    print('generate data finished', endTime-startTime) #os.path.splitext(os.path.basename(pth_in))[0]


if __name__ == '__main__':
    args = Parser().parse_args()
    n_workers = args.thread
    # pool = mp.Pool(args.thread)
    # pool.daemon = True
    
    scan_ids = read_txt_to_list(args.txt)
    
    results=[]
    
    arguments_list=list()
    for scan_id in scan_ids:
        pth_in = os.path.join(args.pth_in,scan_id,scan_id+".sens")
        pth_out = os.path.join(args.pth_out,scan_id)
        arguments=['python', args.reader_path,
                     '--filename',pth_in,
                     '--output_path', pth_out,
                     '--export_poses',
                     # '--export_intrinsics'
                ]
        print(arguments)
        arguments_list.append(arguments)
        # break
        # results.append( pool.apply_async(run_process,args=[(arguments)] ) )
        
    if n_workers>0:
        process_map(run_process, arguments_list, max_workers=n_workers, chunksize=1 )
    else:
        for args in tqdm(arguments_list):
            run_process(args)
            
            
        # break
    # print('waiting...')
    # if args.thread > 1:
    #     pool.close()
    #     results = [r.get() for r in results]
    #     pool.join()
    #     print('done')
        