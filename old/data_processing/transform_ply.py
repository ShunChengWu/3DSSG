if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import os,argparse,json,time
import numpy as np
from shutil import copyfile
from plyfile import PlyData
import multiprocessing as mp
from utils import define

try:
    from sets import Set
except ImportError:
    Set = set

parser = argparse.ArgumentParser()
parser.add_argument('--scan_id', type=str, default='', help="target scan_id. leave empty to process all scans")
parser.add_argument('--thread', type=int, default=1, help="how many threads")

opt = parser.parse_args()

def resave_ply(filename_in, filename_out, matrix):
    """ Reads a PLY file from disk.
    Args:
    filename: string
    
    Returns: np.array, np.array, np.array
    """
    file = open(filename_in, 'rb')
    plydata = PlyData.read(file)
    points = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).transpose()
    points4f = np.insert(points, 3, values=1, axis=1)
    points = points4f * matrix
    
    plydata['vertex']['x'] = np.asarray(points[:,0]).flatten()
    plydata['vertex']['y'] = np.asarray(points[:,1]).flatten()
    plydata['vertex']['z'] = np.asarray(points[:,2]).flatten()
    
    plydata.write(filename_out)
    
def read_transform_matrix():
    rescan2ref = {}
    with open(define.Scan3RJson_PATH , "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            for scans in scene["scans"]:
                if "transform" in scans:
                    rescan2ref[scans["reference"]] = np.matrix(scans["transform"]).reshape(4,4)
    return rescan2ref

def main():
    pool = mp.Pool(opt.thread)
    pool.daemon = True
        
    rescan2ref = read_transform_matrix()
    result = None
    rescan_file_name = define.FILE_PATH+'/rescans.txt'
    if os.path.exists(rescan_file_name):
        counter = 0
        process_text = list()
        print('processing rescans...')
        with open(rescan_file_name, 'r') as f:
            for line in f:
                text = '{}: {}/ 1004 rescans'.format(line.rstrip(), counter)
                scan_id = line.rstrip()
                if (opt.scan_id != "") and (scan_id != opt.scan_id):
                    continue
                file_in = os.path.join(define.DATA_PATH, scan_id, define.LABEL_FILE_NAME_RAW)
                file_out = os.path.join(define.DATA_PATH, scan_id, define.LABEL_FILE_NAME)                
                
                if os.path.exists(file_out) is True: continue
                if scan_id in rescan2ref: # if not we have a hidden test scan
                    if opt.thread > 1:
                        process_text.append(text)
                        result = pool.apply_async(resave_ply,(file_in, file_out, rescan2ref[scan_id]))
                    else:
                        print(text)
                        resave_ply(file_in, file_out, rescan2ref[scan_id])
                counter += 1
        while pool._cache:
            print('\r{} {:2.2%}'.format(process_text[1-len(pool._cache)], 1-len(pool._cache)/len(process_text)),flush=True,end='')
            time.sleep(0.5)
        if opt.thread > 1:
            result = result.get()
        print('\ndone!')
    else:
        Warning('cannot find rescan file at',rescan_file_name)
    
    result = None
    reference_file_name = define.FILE_PATH+'/references.txt'
    if os.path.exists(reference_file_name):
        counter = 0
        process_text = list()
        print('processing references...')
        with open(reference_file_name, 'r') as f:
            for line in f:
                text = '{}: {}/ 432 rescans'.format(line.rstrip(), counter)
                scan_id = line.rstrip()
                if (opt.scan_id != "") and (scan_id != opt.scan_id):
                    continue
                file_in = os.path.join(define.DATA_PATH, scan_id, define.LABEL_FILE_NAME_RAW)
                file_out = os.path.join(define.DATA_PATH, scan_id, define.LABEL_FILE_NAME)
                if os.path.exists(file_out): continue
                if opt.thread > 1:
                    process_text.append(text)
                    result = pool.apply_async(copyfile,(file_in, file_out))
                else:
                    print(text)
                    copyfile(file_in, file_out)
                counter += 1
        while pool._cache:
            print('\r{} {:2.2%}'.format(process_text[1-len(pool._cache)], 1-len(pool._cache)/len(process_text)),flush=True,end='')
            time.sleep(0.5)
        if result is not None:
            result = result.get()
        pool.close()
        pool.join()
        print('\ndone!')
    else:
        Warning('cannot find reference file at',reference_file_name)

if __name__ == "__main__": main()
