import subprocess, os, sys, time
import codeLib
from codeLib.utils.util import read_txt_to_list
import argparse
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map 

helpmsg = 'Generate aligned pose'
parser = argparse.ArgumentParser(description=helpmsg,formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c','--config',default='./configs/config_default.yaml',required=False)
parser.add_argument('--thread', type=int, default=0, help='The number of threads to be used.')
args = parser.parse_args()
cfg = codeLib.Config(args.config)

exe=os.path.join(cfg.data.path_3rscan,'c++','rio_lib','build','bin','align_poses')# '/home/sc/research/PersistentSLAM/c++/3RScan/bin/exe_rio_renderer'

def process(scan_id:str):
    # pth_out = os.path.join(cfg.data.path_3rscan_data,sequence_name,'sequence')
    # Render images
    output = subprocess.check_output([exe,
                                      cfg.data.path_3rscan_data,
                                      scan_id,
                                      'sequence'
                     ],
            stderr=subprocess.STDOUT)
    sys.stdout.write(output.decode('utf-8'))

if __name__ == '__main__':
    '''read all scan ids'''
    train_ids = read_txt_to_list(os.path.join(cfg.data.path_file,'train_scans.txt'))
    val_ids = read_txt_to_list(os.path.join(cfg.data.path_file,'validation_scans.txt'))
    test_ids = read_txt_to_list(os.path.join(cfg.data.path_file,'test_scans.txt'))
    scan_ids  = sorted( train_ids + val_ids + test_ids)
    
    if args.thread > 0:
        process_map(process, scan_ids, max_workers=args.thread, chunksize=1 )
    else:
        for scan_id in tqdm(scan_ids):
            print(f'process scan {scan_id}')
            process(scan_id)