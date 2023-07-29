import subprocess
import os
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


helpmsg = 'Generate data for the Sparse setup.'
parser = argparse.ArgumentParser(
    description=helpmsg, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-c', '--config', default='./configs/config_default.yaml', required=False)
parser.add_argument('--thread', type=int, default=0,
                    help='The number of threads to be used.')
parser.add_argument('--overwrite', action='store_true',
                    help='overwrite or not.')

args = parser.parse_args()
    

def download_unzip(url: str, overwrite: bool):
    filename = os.path.basename(url)
    if not os.path.isfile(os.path.join(path_3rscan, filename)):
        logger_py.info('download {}'.format(filename))
        cmd = [
            "wget", url
        ]
        run(cmd, path_3rscan)
    # Unzip
    logger_py.info('unzip {}'.format(filename))
    cmd = [
        "unzip", filename
    ]
    sp = subprocess.Popen(cmd, cwd=path_3rscan, stdout=subprocess.PIPE,
                          stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    if overwrite:
        # send something to skip input()
        sp.communicate('A'.encode())[0].rstrip()
    else:
        # send something to skip input()
        sp.communicate('N'.encode())[0].rstrip()
    sp.stdin.close()  # close so that it will proceed


if __name__ == '__main__':
    cfg = codeLib.Config(args.config)
    path_3rscan = cfg.data.path_3rscan
    path_3rscan_data = cfg.data.path_3rscan_data
    path_3RScan_ScanNet20 = cfg.data.path# os.path.join('data', '3RScan_ScanNet20_ORBSLAM3')

    '''Download orbslam.zip'''
    logger_py.info('Download orbslam.ply for all scans')
    download_unzip("https://www.campar.in.tum.de/public_datasets/2023_cvpr_wusc/reconstruction_orbslam3.zip",
                   args.overwrite)
    logger_py.info('done')

    '''generate scene graph data for orbslam'''
    logger_py.info('generate scene graph data for ORBSlam3')
    py_exe = os.path.join('data_processing', 'gen_data.py')
    cmd = [py_exe,
           '-c', args.config,
           '-o', path_3RScan_ScanNet20,
           '-l', 'ScanNet20',
           '--only_support_type',
           '--segment_type', 'ORBSLAM',
           ]
    if args.overwrite:
        cmd += ['--overwrite']
    print('running cmd', cmd)
    run_python(cmd)
    logger_py.info('done')

    ''' Generate visibility graph '''
    logger_py.info('Generate visibility graph')
    py_exe = os.path.join(
        'data_processing', 'make_visibility_graph_incremental.py')
    cmd = [py_exe,
           '-c', args.config,
           '-o', path_3RScan_ScanNet20,
           ]
    if args.overwrite:
        cmd += ['--overwrite']
    print('running cmd', cmd)
    run_python(cmd)
    logger_py.info('done')

    # '''extract multi-view image bounding box'''
    logger_py.info('extract multi-view image bounding box')
    py_exe = os.path.join('data_processing', 'extract_mv_box_image_3rscan.py')
    cmd = [py_exe, '-c', args.config,
           '--thread', str(args.thread//4),  # use fewer thread for this one
           '-f', os.path.join(path_3RScan_ScanNet20, define.NAME_VIS_GRAPH)]
    if args.overwrite:
        cmd += ['--overwrite']
    run_python(cmd)
    logger_py.info('done')