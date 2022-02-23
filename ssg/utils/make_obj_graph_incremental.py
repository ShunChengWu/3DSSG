import argparse, os, pandas, h5py, logging,json
# from codeLib.utils.classification.labels import NYU40_Label_Names, SCANNET20_Label_Names
import numpy as np
from tqdm import tqdm
# import os,io
# import zipfile
# import imageio
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
from PIL import Image
from torchvision.utils import draw_bounding_boxes

import matplotlib.pyplot as plt
import codeLib
# from codeLib.torch.visualization import show_tv_grid
from codeLib.torch.visualization import show_tv_grid
from codeLib.common import color_rgb, rand_24_bit
from codeLib.utils.util import read_txt_to_list
# from codeLib.object import BoundingBox
# from codeLib.utils.classification.labels import get_ScanNet_label_mapping#get_NYU40_color_palette, NYU40_Label_Names,get_ScanNet_label_mapping
import torch
# import torchvision
# from torchvision.utils import draw_bounding_boxes
from collections import defaultdict
# import json, glob, csv, sys,os, argparse
# from tqdm import tqdm
# from tqdm.contrib.concurrent import process_map 
from ssg import define
from ssg.utils import util_label
from ssg.utils.util_3rscan import load_semseg
from ssg.objects.node import Node
from ssg.utils import util_data

# from collections import defaultdict
from codeLib.torch.visualization import show_tensor_images

structure_labels = ['wall','floor','ceiling']

width=540
height=960

DEBUG=True
DEBUG=False

random_clr_i = [color_rgb(rand_24_bit()) for _ in range(1500)]
random_clr_i[0] = (0,0,0)
ffont = '/home/sc/research/PersistentSLAM/python/2DTSG/files/Raleway-Medium.ttf'
# random_clr_l = {v:color_rgb(rand_24_bit()) for k,v in Scan3R528.items()}
# random_clr_l['none'] = (0,0,0)

def Parse():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-f','--scenelist',default='/home/sc/research/PersistentSLAM/python/2DTSG/files/scannetv2_trainval.txt',help='scene list (txt)')
    #
    # parser.add_argument('-d','--dir',default='/media/sc/SSD1TB/dataset/3RScan/data/3RScan/', 
    #                     help='3rscan dir')
    parser.add_argument('-o','--outdir',default='/home/sc/research/PersistentSLAM/python/3DSSG/data/3RScan_ScanNet20/', help='output dir',required=True)
    # parser.add_argument('-m','--min_occ',default=0.2,help='The threshold for the visibility of an object. If below this value, discard (higher, more occurance)')
    parser.add_argument('--min_object', help='if less thant min_obj objects, ignore image', default=1)
    # parser.add_argument('-l','--label_type',default='3rscan160', choices=['nyu40','eigen13','rio27', 'rio7','3rscan','3rscan160'], 
    #                     help='target label type.')
    parser.add_argument('--min_size', default=60, help='min length on bbox')
    parser.add_argument('--target_name','-n', default='graph.json', help='target graph json file name')
    # parser.add_argument('-lf','--label_file',default='/media/sc/space1/dataset/scannet/scannetv2-labels.combined.tsv', 
    #                     help='file path to scannetv2-labels.combined.tsv')
    # parser.add_argument('--skip_structure',default=0,help='should ignore sturcture labels or not')
    # parser.add_argument('--skip_edge',default=0,type=int,help='should bbox close to image boundary')
    # parser.add_argument('--skip_size',default=1,type=int,help='should filter out too small objs')
    parser.add_argument('--overwrite', type=int, default=0, help='overwrite existing file.')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    return parser

if __name__ == '__main__':
    args = Parse().parse_args()
    DEBUG = args.debug
    print(args)
    outdir=args.outdir
    # min_oc=float(args.min_occ) # maximum occlusion rate authorised
    min_obj=float(args.min_object)
    # gt2d_dir = args.gt2d_dir
    
    '''create output file'''
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    '''logging'''
    logging.basicConfig(filename=os.path.join(outdir,'objgraph_incremental.log'), level=logging.DEBUG)
    logger_py = logging.getLogger(__name__)
    if DEBUG:
        logger_py.setLevel('DEBUG')
    else:
        logger_py.setLevel('INFO')
    logger_py.debug('args')
    logger_py.debug(args)
   
    #save configs
    with open(os.path.join(outdir,'args_objgraph.txt'), 'w') as f:
        for k,v in args.__dict__.items():
            f.write('{}:{}\n'.format(k,v))
        pass
    try:
        h5f = h5py.File(os.path.join(outdir,'proposals.h5'), 'a')
    except:
        h5f = h5py.File(os.path.join(outdir,'proposals.h5'), 'w')
    # h5f.attrs['label_type'] = args.label_type
    
    '''read scenes'''
    fdata = os.path.join(define.DATA_PATH)
    train_ids = read_txt_to_list(os.path.join(define.ROOT_PATH,'files','train_scans.txt'))
    val_ids = read_txt_to_list(os.path.join(define.ROOT_PATH,'files','validation_scans.txt'))
    test_ids = read_txt_to_list(os.path.join(define.ROOT_PATH,'files','test_scans.txt'))
    
    print(len(train_ids))
    print(len(val_ids))
    print(len(test_ids))
    scan_ids  = sorted( train_ids + val_ids + test_ids)
    print(len(scan_ids))
    
    pbar = tqdm(scan_ids)
    
    '''process'''
    invalid_scans=0
    valid_scans=0
    for scan_id in pbar: #['scene0000_00']: #glob.glob('scene*'):
        if DEBUG: scan_id = '095821f7-e2c2-2de1-9568-b9ce59920e29'
        logger_py.info(scan_id)
        pbar.set_description('processing {}'.format(scan_id))
        
        pth_graph = os.path.join(fdata,scan_id,args.target_name)
        with open(pth_graph, "r") as read_file:
            data = json.load(read_file)[scan_id]
        
        
        
        # graph = util_data.load_graph(data,box_filter_size=[int(args.min_size)])
        
        '''check if the scene has been created'''
        if scan_id in h5f: 
            if args.overwrite == 0: 
                logger_py.info('exist. skip')
                continue
            else:
                del h5f[scan_id]
                
        '''calculate '''
        kfs = dict()
        objects = dict()
        node2kfs = dict()
        
        for kf_ in data['kfs']:
            bboxes = kf_['bboxes']
            if len(bboxes) < min_obj: continue
            width = kf_['rgb_dims'][0]
            height = kf_['rgb_dims'][1]
            path = kf_['path']
            fname = os.path.basename(path)
            fid = int(''.join([x for x in fname if x.isdigit()]))
            
            if str(fid) not in kfs: kfs[str(fid)] = dict()
            kf = kfs[str(fid)]
            kf['idx'] = fid
            kf['bboxes'] = dict()
            
            
            img = np.array(Image.open(path))
            img = np.rot90(img,3).copy()# Rotate image
            
            boxes=list()
            clrs =list()
            labelNames=list()
            
            # print('kfid',kf_['id'])
            
            scale = [kf_['rgb_dims'][0]/kf_['mask_dims'][0],kf_['rgb_dims'][1]/kf_['mask_dims'][1] ]
            for oid in bboxes:
                if int(oid) == 0: continue
                # print('oid',oid)
                
                '''scale bounding box back'''
                box = bboxes[oid] # xmin,ymin,xmax,ymax
                box[0] *= scale[0]
                box[2] *= scale[0]
                box[1] *= scale[1]
                box[3] *= scale[1]
                
                '''Check width and height'''
                w_ori = box[2]-box[0]
                h_ori = box[3]-box[1]
                if w_ori  < args.min_size or h_ori < args.min_size: continue
            
                '''check format is correct'''
                assert 0 <= box[0] < box[2]
                assert 0 <= box[1] < box[3]
                
                
                # check boundary
                box[0] = max(0,box[0])
                box[1] = max(0,box[1])
                box[2] = min(width, box[2])
                box[3] = min(height, box[3])
                
                '''
                The coordinate system is top-left corner (openCV). clockwise 90 degree with OpenGL(Pytorch) bottom left is equalivelent to counter clockwise 90.
                ------->x
                |
                | 
                v
                y
                
                to
                
                x
                ^
                |
                |
                ------>y
                
                '''
                # c.clockwise 90
                box_r = [0,0,0,0]
                box_r[0] = height-box[1]
                box_r[1] = box[0]
                box_r[2] = height-box[3]
                box_r[3] = box[2]
                
                box[0] = min(box_r[0],box_r[2])
                box[2] = max(box_r[0],box_r[2])
                box[1] = min(box_r[1],box_r[3])
                box[3] = max(box_r[1],box_r[3])
                
                boxes.append(box)
                labelNames.append('unknown')
                clrs.append((255,255,255))
                
                kf['bboxes'][oid] = box
                
                if str(oid) not in objects: objects[str(oid)] = dict()
                obj = objects[str(oid)]
                obj['label'] = 'unknown'
                
                if int(oid) not in node2kfs:
                    node2kfs[int(oid)]=list()
                node2kfs[int(oid)].append(fid)
                
                # break
            if DEBUG:
                torch_img = torch.from_numpy(img).permute(2,0,1)
                boxes = torch.tensor(boxes, dtype=torch.float)
                result = draw_bounding_boxes(torch_img, boxes, 
                                              labels=labelNames,
                                              colors=clrs, 
                                              width=5,
                                              font=ffont,
                                              font_size=50)
                show_tv_grid(result)
                plt.show()
                # print('')
            
        '''check if each node has at least one kf'''
        to_deletes = []
        for k,v in objects.items():
            if int(k) not in node2kfs or len(node2kfs[int(k)])==0:
                to_deletes.append(k)
        for idx in to_deletes:
            objects.pop(idx)
            
        '''check if empty'''
        if len(objects) == 0:
            # print('skip',scene)
            invalid_scans+=1
            continue
        valid_scans+=1
        
        '''save'''
        # Save objects.
        h5g = h5f.create_group(scan_id)
        seg2idx = dict()
        h5node = h5g.create_group('nodes')
        for idx, data in enumerate(objects.items()):
            oid, obj = data
            # Save the indices of KFs
            dset = h5node.create_dataset(oid,data=node2kfs[int(oid)])
            dset.attrs['label'] = str(obj['label'])
            # dset.attrs['occlution'] = str(obj['occlution'])
        
        # kfs_=list()
        if 'kfs' in h5g: del h5g['kfs']
        dkfs = h5g.create_group('kfs')
        for idx, data in enumerate(kfs.items()):
            k,v = data
            boxes = v['bboxes']
            # occlu = v['occlution']
            boxes_=list()
            seg2idx=dict()
            for ii, kk in enumerate(boxes):
                boxes_.append(boxes[kk]+[0])
                seg2idx[int(kk)] = ii
            dset = dkfs.create_dataset(k,data=boxes_)
            dset.attrs['seg2idx'] = [(k,v) for k,v in seg2idx.items()]  
            # dset = dkfs.create_dataset(k, data=occlu_)
            # dset.attrs['seg2idx'] = [(k,v) for k,v in seg2idx.items()]  
        if DEBUG: break
        # break
    print('')
    if invalid_scans+valid_scans>0:
        print('percentage of invalid scans:',invalid_scans/(invalid_scans+valid_scans), '(',invalid_scans,',',(invalid_scans+valid_scans),')')
        h5f.attrs['classes'] = util_label.NYU40_Label_Names
        # write args
        tmp = vars(args)
        if 'args' in h5f: del h5f['args']
        h5f.create_dataset('args',data=())
        for k,v in tmp.items():
            h5f['args'].attrs[k] = v
        # with open(os.path.join(outdir,'classes.txt'), 'w') as f:
        #     for cls in util_label.NYU40_Label_Names:
        #         f.write('{}\n'.format(cls))
    else:
        print('no scan processed')
    h5f.close()
    