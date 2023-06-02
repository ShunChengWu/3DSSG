import subprocess, os
import sys
import codeLib
from codeLib.utils.util import read_txt_to_list
from codeLib.subprocess import run, run_python
import argparse
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from ssg import define 
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger_py = logging.getLogger(__name__)


helpmsg = 'Prepare all dataset'
parser = argparse.ArgumentParser(description=helpmsg,formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c','--config',default='./configs/config_default.yaml',required=False)
parser.add_argument('--thread', type=int, default=0, help='The number of threads to be used.')
parser.add_argument('--overwrite', action='store_true', help='overwrite or not.')

args = parser.parse_args()

def download_unzip(url:str,overwrite:bool):
    filename = os.path.basename(url)
    if not os.path.isfile(os.path.join(path_3rscan,filename)):
        logger_py.info('download {}'.format(filename))
        cmd = [
            "wget",url
        ]
        run(cmd,path_3rscan)
    # Unzip
    logger_py.info('unzip {}'.format(filename))
    cmd = [
        "unzip",filename
    ]
    sp = subprocess.Popen(cmd,cwd=path_3rscan, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    if overwrite:
        sp.communicate('A'.encode())[0].rstrip()# send something to skip input()
    else:
        sp.communicate('N'.encode())[0].rstrip()# send something to skip input()
    sp.stdin.close() # close so that it will proceed

if __name__ == '__main__':
    cfg = codeLib.Config(args.config)
    path_3rscan = cfg.data.path_3rscan
    path_3rscan_data = cfg.data.path_3rscan_data
    
    '''Download color_align.zip'''
    if False:
        # Generate color_align.ply yourself (in case the link doesn't work anymore.)
        # map color
        logger_py.info('mapping color')
        cmd = [
            py_align_color,
            "-c",args.config,
        ]
        if args.overwrite: cmd += ['--overwrite']
        run_python(cmd)
        logger_py.info('done')
    else:
        # Download it from the server
        ## check file exist
        download_unzip(
            "https://www.campar.in.tum.de/public_datasets/2023_cvpr_wusc/color_align.zip",
            args.overwrite)
        
    # Download processed scans (inseg & orbslam3)
    #TODO: add url
    
    '''calculate per entity occlution'''
    try:
        '''Download from server'''
        download_unzip(
            "https://www.campar.in.tum.de/public_datasets/2023_cvpr_wusc/2dgt.zip",
            args.overwrite)
    except:
        pass
    logger_py.info('calculate per entity occlution')
    py_exe = os.path.join('data_processing','calculate_entity_occlution_ratio.py')
    cmd = [py_exe,'-c',args.config,'--thread',str(args.thread)]
    if args.overwrite: cmd += ['--overwrite']
    run_python(cmd)
    logger_py.info('done')
    
    '''build visibility graph'''
    logger_py.info('build visibility graph')
    py_exe = os.path.join('data_processing','make_visibility_graph_3rscan.py')
    cmd = [py_exe,'-c',args.config]
    if args.overwrite: cmd += ['--overwrite']
    # For label type: ScanNet20
    path_3RScan_ScanNet20 = os.path.join('data','3RScan_ScanNet20')
    run_python(cmd+['-l','scannet20','-o',path_3RScan_ScanNet20])
    # For label type 3RScan160
    path_3RScan_3RScan160 = os.path.join('data','3RScan_3RScan160')
    run_python(cmd+['-l','3rscan160','-o',path_3RScan_3RScan160])
    logger_py.info('done')
    
    '''extract multi-view image bounding box'''
    logger_py.info('extract multi-view image bounding box')
    py_exe = os.path.join('data_processing','extract_mv_box_image_3rscan.py')
    cmd = [py_exe,'-c',args.config, 
           '--thread',str(args.thread//4),# use fewer thread for this one
           '-o',os.path.join(cfg.data.path_3rscan,'data'),
           '-f',os.path.join(path_3RScan_3RScan160,define.NAME_OBJ_GRAPH)]
    if args.overwrite: cmd += ['--overwrite']
    run_python(cmd)
    logger_py.info('done')
    
    '''generate scene graph data for GT'''
    logger_py.info('generate scene graph data for GT')
    py_exe = os.path.join('data_processing','gen_data_gt.py')
    cmd = [py_exe,
            '-o',path_3RScan_ScanNet20,
            '-l','ScanNet20',
            '--only_support_type'
            ]
    if args.overwrite: cmd += ['--overwrite']
    logger_py.info('running cmd {}'.format(cmd))
    run_python(cmd)
    
    # 3RScan160
    cmd = [py_exe,
            '-o',path_3RScan_3RScan160,
            '-l','3RScan160',
            ]
    if args.overwrite: cmd += ['--overwrite']
    logger_py.info('running cmd {}'.format(cmd))
    run_python(cmd)
    logger_py.info('done')