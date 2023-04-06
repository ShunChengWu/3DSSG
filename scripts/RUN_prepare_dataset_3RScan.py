'''
Generate aligned poses & labels.instances.align.annotated.v2.ply & rendered views
'''
import subprocess, os
import codeLib
from codeLib.utils.util import read_txt_to_list
from codeLib.subprocess import run, run_python
# from utils import util
import argparse
# from utils import util_parser
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ssg.utils.util_data import read_all_scan_ids 

helpmsg = 'Prepare all dataset'
parser = argparse.ArgumentParser(description=helpmsg,formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c','--config',default='./configs/config_default.yaml',required=False)
parser.add_argument('--download', action='store_true', help='download 3rscan data.')
parser.add_argument('--thread', type=int, default=0, help='The number of threads to be used.')
parser.add_argument('--overwrite', action='store_true', help='overwrite or not.')
args = parser.parse_args()

py_transform_ply = os.path.join('data_processing','transform_ply.py')
exe_rio_renderer = "/media/sc/SSD4TB/research/rio_renderer/bin/rio_renderer"#TODO: add script to build this ? as a submodule?



def download_data_3rscan(path_3rscan:str):
    def run_download(cmd,cwd):
        sp = subprocess.Popen(cmd,cwd=cwd, stdin=subprocess.PIPE)
        sp.stdin.write("\r\n") # send the CR/LF for pause
        sp.stdin.close() # close so that it will proceed
        
    # check if download.py exist
    path_download = os.path.join(path_3rscan,'download.py')
    assert os.path.isfile(path_download), "download.py should be placed at the main 3RScan directory."
    
    types = ["semseg.v2.json","sequence.zip","labels.instances.annotated.v2.ply","mesh.refined.v2.obj","mesh.refined.mtl","mesh.refined_0.png"]
    cwd = path_3rscan
    for type in types:
        cmd = [
            path_download,
            "-o","./data/3RScan/",
            "--type",type
        ]
        run_download(cmd,cwd)
        
if __name__ == '__main__':
    cfg = codeLib.Config(args.config)
    path_3rscan = cfg.data.path_3rscan
    path_3rscan_data = cfg.data.path_3rscan_data
    
    '''read all scan ids'''
    scan_ids = sorted(read_all_scan_ids(cfg.data.path_split))
    
    # download all required files
    if args.download:
        download_data_3rscan(path_3rscan)
        
        # unzip all sequences
        cmd = r"""find . -name '*.zip' -exec sh -c 'base={};filename="${base%.*}"; unzip -o -d $filename {};' ';'"""
        run(cmd,path_3rscan)
    
    # Generate aligned instance ply
    cmd = [
        py_transform_ply,
        "-c",args.config,
        "--thread",str(args.thread)
    ]
    if args.overwrite: cmd += ['--overwrite']
    run_python(cmd)
    
    # Generate rendered views
    def generate_rendered_images(scan_id:str):
        # check if file exist
        if not args.overwrite:
            first_name = os.path.join(path_3rscan_data,scan_id,"sequence","frame-000000.rendered.color.jpg")
            if os.path.isfile(first_name):
                return
        run([
            exe_rio_renderer,
            "--pth_in",os.path.join(path_3rscan_data,scan_id,"sequence"),
        ],verbose=False)
    pbar = tqdm(scan_ids)
    if args.thread > 0:
        process_map(generate_rendered_images,scan_ids,max_workers=args.thread)
    else:
        for scan_id in pbar:
            pbar.set_description(f"generate rendered view for scan {scan_id}")
            generate_rendered_images(scan_id)