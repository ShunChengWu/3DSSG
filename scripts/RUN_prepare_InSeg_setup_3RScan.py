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
    
    '''Download inseg.zip'''
    logger_py.info('Download Inseg.ply for all scans')
    download_unzip("https://www.campar.in.tum.de/public_datasets/2023_cvpr_wusc/reconstruction_inseg.zip",
                   args.overwrite)
    
    
    '''generate scene graph data for InSeg'''
    print('generate scene graph data for GT')
    py_exe = os.path.join('data_processing','gen_data.py')
    path_3RScan_ScanNet20 = os.path.join('data','3RScan_ScanNet20_InSeg')
    cmd = [py_exe,
            '-o',path_3RScan_ScanNet20,
            '-l','ScanNet20',
            '--only_support_type'
            ]
    if args.overwrite: cmd += ['--overwrite']
    print('running cmd',cmd)
    run_python(cmd)
    
    import sys
    sys.exit()
    
    # 3RScan160
    cmd = [py_exe,
            '-o',path_3RScan_3RScan160,
            '-l','3RScan160',
            ]
    if args.overwrite: cmd += ['--overwrite']
    print('running cmd',cmd)
    run_python(cmd)
    print('done')