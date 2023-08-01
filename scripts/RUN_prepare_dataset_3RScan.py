'''
Generate aligned poses & labels.instances.align.annotated.v2.ply & rendered views
'''
import subprocess
import os
import sys
import codeLib
from codeLib.utils.util import read_txt_to_list
from codeLib.subprocess import run, run_python
# from utils import util
import argparse
# from utils import util_parser
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ssg.utils.util_data import read_all_scan_ids
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger_py = logging.getLogger(__name__)

helpmsg = 'Prepare all dataset'
parser = argparse.ArgumentParser(
    description=helpmsg, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-c', '--config', default='./configs/config_default.yaml', required=False)
parser.add_argument('--download', action='store_true',
                    help='download 3rscan data.')
parser.add_argument('--thread', type=int, default=8,
                    help='The number of threads to be used.')
parser.add_argument('--overwrite', action='store_true',
                    help='overwrite or not.')
args = parser.parse_args()


py_transform_ply = os.path.join('data_processing', 'transform_ply.py')
exe_rio_renderer = os.path.join('rio_renderer', 'bin', 'rio_renderer')
py_align_pose = os.path.join('data_processing', 'generate_align_pose.py')
py_align_color = os.path.join('data_processing', 'generate_align_color.py')


def download_data_3rscan(path_3rscan: str, path_3rscan_data: str):
    def run_download(cmd, cwd):
        sp = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                              stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        # send something to skip input()
        sp.communicate('1'.encode())[0].rstrip()
        sp.stdin.close()  # close so that it will proceed

    # path_3rscan = os.path.abspath(path_3rscan)
    # path_3rscan_data = os.path.abspath(path_3rscan_data)
    # check if download.py exist
    path_download = os.path.join(path_3rscan, 'download.py')
    assert os.path.isfile(path_download), "download.py should be placed at the main 3RScan directory. \
        Please fill the term of use in this page: https://waldjohannau.github.io/RIO/ to get the download script."

    types = ["semseg.v2.json", "sequence.zip", "labels.instances.annotated.v2.ply",
             "mesh.refined.v2.obj", "mesh.refined.mtl", "mesh.refined_0.png"]
    pbar = tqdm(types)
    for type in pbar:
        pbar.set_description(f'downloadin type {type}...')
        cmd = [
            'python', path_download,
            "-o", path_3rscan_data,
            "--type", type
        ]
        run_download(cmd, './')


if __name__ == '__main__':
    cfg = codeLib.Config(args.config)
    path_3rscan = cfg.data.path_3rscan
    path_3rscan_data = cfg.data.path_3rscan_data

    '''read all scan ids'''
    scan_ids = sorted(read_all_scan_ids(cfg.data.path_split))

    # download all required files
    if args.download:
        logger_py.info('download dataset')
        download_data_3rscan(path_3rscan, path_3rscan_data)

        # unzip all sequences
        cmd = r"""find . -name '*.zip' -exec sh -c 'base={};filename="${base%.*}"; unzip -o -d $filename {};' ';'   """
        run(cmd, path_3rscan_data)
        logger_py.info('done')

    # Generate aligned instance ply
    logger_py.info('generate aligned instance ply')
    cmd = [
        py_transform_ply,
        "-c", args.config,
        "--thread", str(args.thread)
    ]
    if args.overwrite:
        cmd += ['--overwrite']
    run_python(cmd)
    logger_py.info('done')

    if False:
        # Generate rendered views
        def generate_rendered_images(scan_id: str):
            # check if file exist
            if not args.overwrite:
                first_name = os.path.join(
                    path_3rscan_data, scan_id, "sequence", "frame-000000.rendered.color.jpg")
                if os.path.isfile(first_name):
                    return
            run([
                exe_rio_renderer,
                "--pth_in", os.path.join(path_3rscan_data,
                                         scan_id, "sequence"),
            ], verbose=False)
        pbar = tqdm(scan_ids)
        logger_py.info('generate rendered views')
        if args.thread > 0:
            process_map(generate_rendered_images,
                        scan_ids, max_workers=args.thread)
        else:
            for scan_id in pbar:
                pbar.set_description(
                    f"generate rendered view for scan {scan_id}")
                generate_rendered_images(scan_id)
    else:
        # Download from the server
        if os.path.isfile('rendered.zip') and not args.overwrite:
            logger_py.info('rendered.zip already downloaded')
            pass
        else:
            logger_py.info('download rendered.zip')
            cmd = [
                "wget", "https://www.campar.in.tum.de/public_datasets/2023_cvpr_wusc/rendered.zip"
            ]
            run(cmd, path_3rscan)
        # Unzip
        logger_py.info('unzip rendered.zip')
        cmd = [
            "unzip", "rendered.zip"
        ]
        sp = subprocess.Popen(cmd, cwd=path_3rscan, stdout=subprocess.PIPE,
                              stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        if args.overwrite:
            # send something to skip input()
            sp.communicate('A'.encode())[0].rstrip()
        else:
            # send something to skip input()
            sp.communicate('N'.encode())[0].rstrip()
        sp.stdin.close()  # close so that it will proceed
    logger_py.info('done')

    # Align Pose
    # TODO: update the script
    # logger_py.info('generate aligned pose')
    # cmd = [
    #     py_align_pose
    # ]
    # logger_py.info('done')

    # map color
    # logger_py.info('mapping color')
    # cmd = [
    #     py_align_color,
    #     "-c",args.config,
    # ]
    # if args.overwrite: cmd += ['--overwrite']
    # run_python(cmd)
    # logger_py.info('done')
