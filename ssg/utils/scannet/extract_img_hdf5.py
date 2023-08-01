import h5py
from SensorData import SensorData
import argparse
import os, sys
import logging
from tqdm import tqdm

logging.basicConfig()
logger_py = logging.getLogger(__name__)
# params
logger_py.setLevel('INFO')
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('-f','--scenelist',default='/media/sc/space1/dataset/scannet/Tasks/Benchmark/scannetv2_val.txt')
parser.add_argument('--scannet_dir',default='/media/sc/space1/dataset/scannet/scans/')
parser.add_argument('-o','--output_path', required=True, help='output path')
parser.add_argument('--overwrite', type=int, default=0, help='overwrite existing file.')
parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

args = parser.parse_args()
print(args)

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def main():    
    # if not os.path.exists(args.output_path):
    #   os.makedirs(args.output_path)
    h5f = h5py.File(args.output_path, 'a')
    scan_ids = read_txt_to_list(args.scenelist)

    pbar = tqdm(sorted(scan_ids))
    for scan_id in pbar:
        # logger_py.info('process scan {}'.format(scan_id))
        
        pbar.set_description('Processing {}'.format(scan_id))
        if scan_id in h5f:
            if args.overwrite==0: 
                # logger_py.info('exist. skip')
                continue
            else:
                del h5f[scan_id]

        # load the data
        fsens = os.path.join(args.scannet_dir,scan_id,scan_id+'.sens')
        # logger_py.info('loading %s...' % fsens)
        pbar.set_description('Processing {} loading...'.format(scan_id))
        sd = SensorData(fsens)
        pbar.set_description('Processing {} loaded'.format(scan_id))
        # logger_py.info('loaded!')
        
        # Save RGB
        h5g = h5f.create_group(scan_id)
        # logger_py.info('save to hdf5...')
        pbar.set_description('Processing {} saving...'.format(scan_id))
        sd.export_h5(h5g,write_rgb=True, write_depth=False, write_pose=True, frame_skip=1)
        # break
    h5f.close()
            
if __name__ == '__main__':
    main()