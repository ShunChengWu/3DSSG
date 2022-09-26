import argparse, os, pandas, h5py, logging
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
# import codeLib
# from codeLib.torch.visualization import show_tv_grid
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
from ssg.utils import util_3rscan
from ssg.utils.util_3rscan import load_semseg
# from collections import defaultdict
from codeLib.torch.visualization import show_tensor_images

structure_labels = ['wall','floor','ceiling']

width=540
height=960

DEBUG=True
DEBUG=False

random_clr_i = [color_rgb(rand_24_bit()) for _ in range(1500)]
random_clr_i[0] = (0,0,0)
# random_clr_l = {v:color_rgb(rand_24_bit()) for k,v in Scan3R528.items()}
# random_clr_l['none'] = (0,0,0)

def Parse():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-f','--scenelist',default='/home/sc/research/PersistentSLAM/python/2DTSG/files/scannetv2_trainval.txt',help='scene list (txt)')
    #
    parser.add_argument('-d','--gt2d_dir',default='/media/sc/SSD1TB/dataset/3RScan/2dgt', 
                        help='directory containing the .2dgt i.e. the gt 2d detections, generated with the script')
    parser.add_argument('-o','--outdir',default='/home/sc/research/PersistentSLAM/python/3DSSG/data/3RScan_3RScan160/', help='output dir',required=True)
    parser.add_argument('-m','--min_occ',default=0.2,help='The threshold for the visibility of an object. If below this value, discard (higher, more occurance)')
    parser.add_argument('--min_object', help='if less thant min_obj objects, ignore image', default=1)
    parser.add_argument('-l','--label_type',default='3rscan160', choices=['nyu40','eigen13','rio27', 'rio7','3rscan','3rscan160','scannet20'], 
                        help='target label type.')
    parser.add_argument('--min_size', default=60, help='min length on bbox')
    # parser.add_argument('-lf','--label_file',default='/media/sc/space1/dataset/scannet/scannetv2-labels.combined.tsv', 
    #                     help='file path to scannetv2-labels.combined.tsv')
    parser.add_argument('--skip_structure',default=0,help='should ignore sturcture labels or not')
    parser.add_argument('--skip_edge',default=0,type=int,help='should bbox close to image boundary')
    parser.add_argument('--skip_size',default=1,type=int,help='should filter out too small objs')
    parser.add_argument('--overwrite', type=int, default=0, help='overwrite existing file.')
    return parser

def get_bbx_wo_flatcheck(scan_id, fn,min_oc, mapping, min_size:list=[240,240]):
    obj_by_img={}  
    data = pandas.read_csv(fn, delimiter= ' ')      
    # imgs = f[scan_id]
    obj_set=set()
    obj_set_f = set()
    # filter_struc=0
    # filter_label=0
    # filter_edge=0
    # filter_occ=0
    # filter_size=0
    filter_counter = defaultdict(int)
    filter_label = defaultdict(set)
    
    msg = 'skip fid:{}, iid:{}, lid:{}'
    
    for i in data.index:
        fname = data['frame_id'][i]
        oid = data['object_id'][i]
        olabel = data['label'][i]
        oc = data['occlution_ratio'][i]
        x1 = data['x_min'][i]
        y1 = data['y_min'][i]
        x2 = data['x_max'][i]
        y2 = data['y_max'][i]
        
        obj_set.add(oid)
        
        olabel=olabel[1:].replace("\'","") # remove utf8 charesters
        olabel=olabel.replace('_',' ')
        
        '''check conditions'''
        # Label
        if olabel not in mapping:
             if DEBUG: print(msg.format(fname,oid,olabel),'not in ', mapping.keys())
             filter_counter['label']+=1
             filter_label['label'].add(olabel)
             continue
             raise RuntimeError('all labels should be included in the mapping dict',mapping.keys(), 'query',olabel)
        olabel = mapping[olabel]
        # structure
        if args.skip_structure>0:
          if olabel in structure_labels:
              if DEBUG: print(msg.format(fname,oid,olabel),' structure label')
              filter_counter['struc']+=1
              filter_label['struc'].add(olabel)
              continue
        # On boarder
        if args.skip_edge>0:
            if float(x1)<1 or float(y1)<1 or width < float(x2) or height < float(y2):
                if DEBUG: print(msg.format(fname,oid,olabel),': on edge')
                filter_counter['edge']+=1
                filter_label['edge'].add(olabel)
                continue
        # Occurence too low
        oc=float(oc)
        oc = round(oc, 3)
        if oc<min_oc: # if occlusion rate is over the maximum authorised, then skip
            if DEBUG: print(msg.format(fname,oid,olabel),'occluded',oc,'<',min_oc)
            filter_counter['occ']+=1
            filter_label['occ'].add(olabel)
            continue
      
          # too smal
        if args.skip_size>0:
            size = [x2-x1,y2-y1]
            if size[0] < min_size[0] or size[1] < min_size[1]:
                if DEBUG: print(msg.format(fname,oid,olabel),'too small', size)
                filter_counter['size']+=1
                filter_label['size'].add(olabel)
                continue
        
        # if args.label_type == 'scannet20':
        # if olabel not in util_label.NYU40_Label_Names:
        #     if DEBUG: print('skip',fname,olabel,'not in label names')
        #     continue
      
        if fname not in obj_by_img:
            obj_by_img[fname]=[fname,[]]
        obj_by_img[fname][1].append([oid,olabel,oc,float(x1),float(y1),float(x2),float(y2)])      
        obj_set_f.add(oid)
    
    # print('totalsize:', len(data.index))
    logger_py.debug('filtered type and classes')
    for k,v in filter_counter.items():
         # print(k,v, filter_label[k])
         logger_py.debug('{}: {}. {}'.format(k,v,filter_label[k]))
    logger_py.debug('the obj filter ratio: {} ({}/{})'.format(len(obj_set_f)/len(obj_set),len(obj_set_f),len(obj_set)))
    
    '''debug vis'''
    if DEBUG:
        vis('/media/sc/SSD1TB/dataset/3RScan/data/3RScan/',scan_id,obj_by_img)
        
    return obj_by_img

def vis(datapath, scan_id,obj_by_img:dict):
    
    insta_filepattern = 'frame-{0:06d}.rendered.instances.png'
    
    for fname,v in obj_by_img.items():
        data_list = v[1]
        pth_inst = os.path.join(datapath,scan_id,'sequence',insta_filepattern.format(fname))
        iimg_data = np.array(Image.open(pth_inst), dtype=np.uint8)
        
        ori_list = set(np.unique(iimg_data).tolist()).difference([0])
        f_list = [x[0] for x in data_list]
        diff = ori_list.difference(f_list)
        if len(diff)==0: continue
        # print('f_list',f_list)
        # print(diff)
        
        clr_img_ori = np.zeros([height,width,3],dtype=np.uint8)
        clr_img = np.zeros([height,width,3],dtype=np.uint8)
        clr_img_diff = np.zeros([height,width,3],dtype=np.uint8)
        
        for oid in ori_list:
            clr_img_ori[iimg_data==oid] = random_clr_i[oid]
        
        for data in data_list:
            oid, *_ = data
            clr_img[iimg_data==oid] = random_clr_i[oid]
            
        
        for oid in diff:
            clr_img_diff[iimg_data==oid] = random_clr_i[oid]
            
        # fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        torch_img_ori = torch.as_tensor(clr_img_ori).permute(2,0,1)
        torch_img = torch.as_tensor(clr_img).permute(2,0,1)
        torch_img_diff = torch.as_tensor(clr_img_diff).permute(2,0,1)
        show_tensor_images([torch_img_ori,torch_img_diff, torch_img])
        pass

if __name__ == '__main__':
    args = Parse().parse_args()
    print(args)
    outdir=args.outdir
    min_oc=float(args.min_occ) # maximum occlusion rate authorised
    min_obj=float(args.min_object)
    gt2d_dir = args.gt2d_dir
    
    '''create output file'''
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    logging.basicConfig(filename=os.path.join(outdir,'objgraph.log'), level=logging.DEBUG)
    logger_py = logging.getLogger(__name__)
    if DEBUG:
        logger_py.setLevel('DEBUG')
    else:
        logger_py.setLevel('INFO')
    logger_py.debug('args')
    logger_py.debug(args)
    logger_py.debug('structure_labels')
    logger_py.debug(structure_labels)
    
    # scannet_w = 1296
    # scannet_h= 968

    '''create mapping'''
    # create name2idx mapping
    label_names, label_name_mapping, label_id_mapping = util_label.getLabelMapping(args.label_type,define.LABEL_MAPPING_FILE)
        
    
    #save configs
    with open(os.path.join(outdir,'args_objgraph.txt'), 'w') as f:
        for k,v in args.__dict__.items():
            f.write('{}:{}\n'.format(k,v))
        pass
    h5f = h5py.File(os.path.join(outdir,'proposals.h5'), 'a')
    h5f.attrs['label_type'] = args.label_type
    
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
        logger_py.info(scan_id)
        pbar.set_description('processing {}'.format(scan_id))
        
        # load semseg
        pth_semseg = os.path.join(fdata,scan_id,define.SEMSEG_FILE_NAME)
        mapping = load_semseg(pth_semseg,label_name_mapping)
        mapping[0] = 'none'
        
        # load image info
        info_3rscan = util_3rscan.read_3rscan_info(os.path.join(fdata,scan_id,define.IMG_FOLDER_NAME,define.INFO_NAME))
        img_w,img_h = int(info_3rscan['m_colorWidth']), int(info_3rscan['m_colorHeight'])
        
        '''load 2dgt'''
        gt2d_file = gt2d_dir+'/'+scan_id+'.2dgt'
        if not os.path.isfile(gt2d_file):
            print('file does not exists, skipping',scan_id+'.2dgt')
            continue
        '''check if the scene has been created'''
        if scan_id in h5f: 
            if args.overwrite == 0: 
                logger_py.info('exist. skip')
                continue
            else:
                del h5f[scan_id]
        
        # read data and organize by frames
        obj_by_img=get_bbx_wo_flatcheck(scan_id, gt2d_file, min_oc,label_name_mapping, [args.min_size,args.min_size])
        
        # cluster the frames, each cluster correspond to a set of objects, all the elements of a cluster are images where these objects appear
        oidss={}
        img_by_objnum={}
        kfs = dict()
        objects = dict()
        node2kfs = dict()
        
        for (fnum,(fname,seq)) in list(obj_by_img.items()):
            if len(seq)<min_obj:#if less thant min_obj objects, I don't keep the image
                continue
            if str(fnum) not in kfs: kfs[str(fnum)] = dict()
            
            kf = kfs[str(fnum)]
            kf['idx'] = int(fnum)
            if 'bboxes' not in kf: kf['bboxes'] = dict()
            if 'occlution' not in kf: kf['occlution'] = dict()
            for oid,olabel,oc,x1,y1,x2,y2 in seq:
                if str(oid) in kf['bboxes']: raise RuntimeError('exist')
                kf['bboxes'][str(oid)] = [x1/width,y1/height,x2/width,y2/height]
                kf['occlution'][str(oid)] = oc
                
                if str(oid) not in objects:
                    objects[str(oid)] = dict()
                else:
                    if objects[str(oid)]['label'] != olabel:
                        raise RuntimeError('detect different label: {} {}'.format(objects[str(oid)], olabel))
                obj = objects[str(oid)]
                obj['label'] = olabel
                
                if oid not in node2kfs:
                    node2kfs[oid] = list()
                node2kfs[oid].append(fnum)
                
        #TODO:debug
        # print_selection(scene, node2kfs,objects,kfs)
        
        '''check filtered instances'''
        int_filtered_insts = [int(x) for x in objects]
        diffs = set(mapping.keys()).difference(set(int_filtered_insts))
        if DEBUG: print('missing instances', diffs)
        logger_py.debug('missing instances: {}'.format(diffs))
        
                
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
        
        h5g = h5f.create_group(scan_id)
        # obj_ = list()
        seg2idx = dict()
        h5node = h5g.create_group('nodes')
        for idx, data in enumerate(objects.items()):
            oid, obj = data
            dset = h5node.create_dataset(oid,data=node2kfs[int(oid)])
            dset.attrs['label'] = str(obj['label'])
            # dset.attrs['occlution'] = str(obj['occlution'])
        
        # kfs_=list()
        if 'kfs' in h5g: del h5g['kfs']
        dkfs = h5g.create_group('kfs')
        for idx, data in enumerate(kfs.items()):
            k,v = data
            boxes = v['bboxes']
            occlu = v['occlution']
            boxes_=list()
            # occlu_ = list()
            seg2idx=dict()
            for ii, kk in enumerate(boxes):
                # kk,vv = dd
                boxes_.append(boxes[kk]+[occlu[kk]])
                # occlu_.append()
                seg2idx[int(kk)] = ii
            dset = dkfs.create_dataset(k,data=boxes_)
            dset.attrs['seg2idx'] = [(k,v) for k,v in seg2idx.items()]  
            # dset = dkfs.create_dataset(k, data=occlu_)
            # dset.attrs['seg2idx'] = [(k,v) for k,v in seg2idx.items()]  
        
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
    