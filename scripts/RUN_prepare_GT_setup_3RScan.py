import subprocess, os
import codeLib
from codeLib.utils.util import read_txt_to_list
from codeLib.subprocess import run, run_python
import argparse
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ssg import define 


helpmsg = 'Prepare all dataset'
parser = argparse.ArgumentParser(description=helpmsg,formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c','--config',default='./configs/config_default.yaml',required=False)
parser.add_argument('--thread', type=int, default=0, help='The number of threads to be used.')
parser.add_argument('--overwrite', action='store_true', help='overwrite or not.')

args = parser.parse_args()

if __name__ == '__main__':
    cfg = codeLib.Config(args.config)
    path_3rscan = cfg.data.path_3rscan
    path_3rscan_data = cfg.data.path_3rscan_data
    
    '''calculate per entity occlution'''
    py_exe = os.path.join('data_processing','calculate_entity_occlution_ratio.py')
    cmd = [py_exe,'-c',args.config,'--thread',str(args.thread)]
    if args.overwrite: cmd += ['--overwrite']
    run_python(cmd)
    
    '''build visibility graph'''
    py_exe = os.path.join('data_processing','make_visibility_graph_3rscan.py')
    cmd = [py_exe,'-c',args.config]
    if args.overwrite: cmd += ['--overwrite']
    # For label type: ScanNet20
    path_3RScan_ScanNet20 = os.path.join('data','3RScan_ScanNet20')
    run_python(cmd+['-l','scannet20','-o',path_3RScan_ScanNet20])
    # For label type 3RScan160
    path_3RScan_3RScan160 = os.path.join('data','3RScan_3RScan160')
    run_python(cmd+['-l','3rscan160','-o',path_3RScan_3RScan160])
    
    '''extract multi-view image bounding box'''
    py_exe = os.path.join('data_processing','extract_mv_box_image_3rscan.py')
    cmd = [py_exe,'-c',args.config, # don't use thread for this one. 
           '-o',os.path.join(cfg.data.path_3rscan,'data'),
           '-f',os.path.join(path_3RScan_3RScan160,define.NAME_OBJ_GRAPH)]
    if args.overwrite: cmd += ['--overwrite']
    run_python(cmd)
    
    '''generate scene graph data for GT'''
    py_exe = os.path.join('data_processing','gen_data_gt.py')
    cmd = [py_exe,
            '-o',path_3RScan_ScanNet20,
            '-l','ScanNet20',
            '--only_support_type'
            ]
    if args.overwrite: cmd += ['--overwrite']
    print('running cmd',cmd)
    run_python(cmd)
    
    # 3RScan160
    cmd = [py_exe,
            '-o',path_3RScan_3RScan160,
            '-l','3RScan160',
            ]
    if args.overwrite: cmd += ['--overwrite']
    print('running cmd',cmd)
    run_python(cmd)