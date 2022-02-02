#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:33:42 2021

@author: sc
"""
import os,io
import zipfile
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import codeLib
from codeLib.torch.visualization import show_tv_grid
from codeLib.common import color_rgb, rand_24_bit,create_folder
from codeLib.utils.util import read_txt_to_list
from codeLib.object import BoundingBox
from codeLib.utils.classification.labels import get_ScanNet_label_mapping#get_NYU40_color_palette, NYU40_Label_Names,get_ScanNet_label_mapping
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
from collections import defaultdict
import json, glob, csv, sys,os, argparse
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map 
from ssg import define
from ssg.utils import util_label
from ssg.utils.util_3rscan import read_3rscan_info, load_semseg

helpmsg = 'Generate object bounding box and occlusion on 3RScan dataset using the rendered label and instance'
parser = argparse.ArgumentParser(description=helpmsg,
                                 formatter_class = argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('-m', '--mode',default='train',choices=['train','val'])
# parser.add_argument('-l','--scene_list',default="/media/sc/space1/dataset/scannet/Tasks/Benchmark/scannetv2_val.txt", help="file containing on each line the id of a scene, it should also correspond to the name of its directory")

parser.add_argument('-o','--outdir',default='/media/sc/SSD1TB2/dataset/scannet/2dgt_new/',required=True)
parser.add_argument('--thread', type=int, default=0, help='The number of threads to be used.')
parser.add_argument('--downsacle','-d', type=int, default=8, help='The factor of downscaling while calculating the occurance. The larger the less accurate but faster speed.')
parser.add_argument('--overwrite', type=int, default=0, help='overwrite')
args = parser.parse_args()



fdata = os.path.join(define.DATA_PATH)
insta_filepattern = 'frame-{0:06d}.rendered.instances.png'
label_filepattern = 'frame-{0:06d}.rendered.labels.png'
ffont = '/home/sc/research/PersistentSLAM/python/2DTSG/files/Raleway-Medium.ttf'



# create name2idx mapping


# clr_pal = get_NYU40_color_palette()
# random_clr_list = [color_rgb(rand_24_bit()) for _ in range(1500)]
# random_clr_list[0] = (0,0,0)

# SOURCE_LABEL='id'
# TARGET_LABEL='id'
# TARGET_LABEL_NAME='raw_category'
# label_mapping = get_ScanNet_label_mapping(pth_scannet_label, 'id', TARGET_LABEL)
# replace None to 0
# for k in label_mapping:
#     if label_mapping[k] is None:
#         label_mapping[k]=0
# label_mapping[0]=0        

# labelname_mapping = get_ScanNet_label_mapping(pth_scannet_label, TARGET_LABEL, TARGET_LABEL_NAME)
# labelname_mapping[0]='none'
# nyu40_label_names = ['none'] + NYU40_Label_Names

class Detection(object):
    label=str()
    box=BoundingBox([0,0,0,0])
    max_iou=0
    clr = (255,255,255)
    def __init__(self,label:str,box:list,clr=(255,255,255),max_iou:float=0.0):
        self.label=label
        self.box = BoundingBox(box)
        self.clr=clr
        self.max_iou=max_iou
    def __repr__(self):
        return 'label: {}, max_iou: {}, box: {} clr: {}'.format(self.label, self.max_iou, self.box, self.clr)
    
def mask_occupancy(x:torch.tensor, down_scale:int=2):
    width,height = x.shape[-1],x.shape[-2]
    n_w, n_h = width//down_scale, height//down_scale
    
    if down_scale==1 or n_w==0 or n_h ==0:
        sampled = x
    else:
        sampled = np.zeros([n_h,n_w],dtype=np.byte)
        for w in range(n_w-1):
            for h in range(n_h-1):
                sampled[h,w] = x[h*down_scale:(h+1)*down_scale, w*down_scale: (w+1)*down_scale].any()
            # for w_ in range(down_scale):
            #     for h_ in range(down_scale):
            #         x[h*down_scale+h_, w*down_scale+w_]
    # plt.imshow(sampled)
    # plt.show()
    return sampled.sum()/sampled.size
    # torchvision.transforms.functional.resize((height,width), interpolation=)
    
class LabelImage(object):
    fid=int()
    l_img=np.array([])
    i_img=np.array([])
    detections = defaultdict(Detection)
    def __init__(self, frame_idx:int, iimg_data, limg_data):
        self.fid = frame_idx
        self.l_img = limg_data# imageio.imread(limg_data) # label
        self.i_img = iimg_data#imageio.imread(iimg_data) # instance
        # map label from raw id to NYU40
        # self.l_img = np.vectorize(label_mapping.__getitem__)(self.l_img)
        self.detections = self.process()
        
    def __repr__(self):
        
        return 'frame idx: {}, label image: {}. instance image: {}, detections: {}'.\
            format(self.fid,self.l_img.shape,self.i_img.shape,len(self.detections))
        
    @property
    def ignore_label(self):
        return ['none']
    
    def to_label_color(self):
        clr_img = np.zeros([self.l_img.shape[0],self.l_img.shape[1],3],dtype=np.uint8)
        for label, clr in random_clr_l.items():
            indices = np.where(self.l_img==label)
            clr_img[indices] = clr
        return clr_img
    def to_inst_color(self):
        clr_img = np.zeros([self.i_img.shape[0],self.i_img.shape[1],3],dtype=np.uint8)
        for lid, clr in enumerate(random_clr_i):
            indices = np.where(self.i_img==lid)
            clr_img[indices] = clr
        return clr_img
    
    def get_bbox_image(self):
        torch_img = torch.as_tensor(self.to_label_color()).permute(2,0,1)
        labelNames = ['{0:0.2f}\n'.format(v.max_iou)+v.label for b,v in self.detections.items()]
        boxes = [v.box.tolist() for b,v in self.detections.items()]
        clrs  = [(255,255,255) for b,v in self.detections.items()]
        # ocs   = [v.max_iou for b,v in self.detections.items()]
        boxes = torch.tensor(boxes, dtype=torch.float)
        result = draw_bounding_boxes(torch_img, boxes, 
                                     labels=labelNames,
                                     colors=clrs, 
                                     width=5,
                                     font=ffont,
                                     font_size=50)
        return result
    def show(self):
        show_tv_grid(self.get_bbox_image())
        plt.show()
        # show(result)

    def process(self)->defaultdict(Detection):
        instances = np.unique(self.i_img)
        # labels = np.unique(self.l_img)
        boxes = []
        labelNames=[]
        box_clrs = []
        
        boxdict=defaultdict(Detection)
        for inst in instances:
            indices = np.where(self.i_img == inst)
            label = self.l_img[indices][0]
            # label_name = self.ln_img[indices][0]
            # label_name = NYU40_Idx2Name_map[label]
            if label in self.ignore_label: continue
            box = BoundingBox( [indices[1].min(), indices[0].min(), indices[1].max(), indices[0].max()] )
            if not box.is_valid(): continue
        
            oc = mask_occupancy(self.i_img[box.y_min():box.y_max(),box.x_min():box.x_max()]==inst,down_scale=args.downsacle)
            
            box_clrs.append(random_clr_i[inst])
            boxes.append(box)
            labelNames.append(label)
            
            
            boxdict[inst] =  Detection(
                label = label,
                box = box,
                clr=random_clr_l[label],
                max_iou=oc
                )
        return boxdict

def process(scan_id):
    # check if file exist
    foutput = os.path.join(args.outdir,scan_id+'.2dgt')
    if os.path.isfile(foutput):
        if args.overwrite>0:
            os.remove(foutput)
        else:
            return
        
    # load semseg
    pth_semseg = os.path.join(fdata,scan_id,define.SEMSEG_FILE_NAME)
    mapping = load_semseg(pth_semseg)
    mapping[0] = 'none'
        
    # get number of scans
    scan_info = read_3rscan_info(os.path.join(fdata,scan_id,'sequence','_info.txt'))
    n_images = int(scan_info['m_frames.size'])
    
    # check all images exist
    for frame_id in range(n_images):
        # pth_label = os.path.join(fdata,scan_id,'sequence', label_filepattern.format(frame_id))
        pth_inst  = os.path.join(fdata,scan_id,'sequence', insta_filepattern.format(frame_id))
        
        # if not os.path.isfile(pth_label): raise RuntimeError('file not exist.',pth_label)
        if not os.path.isfile(pth_inst): raise RuntimeError('file not exist.',pth_inst)
        pass
    
    
    # create file
    fp = open(foutput,'w')
    # header
    fp.write('frame_id object_id label occlution_ratio x_min y_min x_max y_max\n')
    
    # process
    for frame_id in range(n_images):
        # pth_label = os.path.join(fdata,scan_id,'sequence', label_filepattern.format(frame_id))
        pth_inst  = os.path.join(fdata,scan_id,'sequence', insta_filepattern.format(frame_id))
        
        # limg_data = np.array(Image.open(pth_label), dtype=np.uint8)
        iimg_data = np.array(Image.open(pth_inst), dtype=np.uint8)
        
        # check keys
        diff_ids = set( np.unique(iimg_data) ).difference(set(mapping.keys()))
        for id in diff_ids: mapping[id]='none'
        
        #
        limg_data = np.vectorize(mapping.__getitem__)(iimg_data)
        limg = LabelImage(frame_id, iimg_data, limg_data)
        
        # limg.show()
        # print(limg)
        for inst, detection in limg.detections.items():
            fp.write('{} {} {} {} {} {} {} {}\n'.\
                     format(frame_id, inst, detection.label.replace(' ','_').encode('utf8'), detection.max_iou, detection.box[0], detection.box[1], detection.box[2], detection.box[3]  ))

    fp.close()
if __name__ =='__main__':
    print(args)
    # pass
    n_workers=args.thread

    train_ids = read_txt_to_list(os.path.join(define.ROOT_PATH,'files','train_scans.txt'))
    val_ids = read_txt_to_list(os.path.join(define.ROOT_PATH,'files','validation_scans.txt'))
    test_ids = read_txt_to_list(os.path.join(define.ROOT_PATH,'files','test_scans.txt'))
    
    print(len(train_ids))
    print(len(val_ids))
    print(len(test_ids))
    scan_ids  = sorted( train_ids + val_ids + test_ids)
    print(len(scan_ids))
    
    # NYU40_Label_Names = ['none'] + util_label.NYU40_Label_Names
    Scan3R528, NYU40,Eigen13,RIO27,RIO7 = util_label.getLabelNames(define.LABEL_MAPPING_FILE)
    Scan3R528[0] = 'none'
    
    # clr_pal = get_NYU40_color_palette()
    random_clr_i = [color_rgb(rand_24_bit()) for _ in range(1500)]
    random_clr_l = {v:color_rgb(rand_24_bit()) for k,v in Scan3R528.items()}
    random_clr_l['none'] = (0,0,0)
    
    # label_names, label_name_mapping, label_id_mapping = util_label.getLabelMapping('nyu40',define.LABEL_MAPPING_FILE)
    # NYU40_Name2Idx_map = {name:idx for idx, name in enumerate(NYU40_Label_Names)}
    # NYU40_Idx2Name_map = {idx:name for idx, name in enumerate(NYU40_Label_Names)}
    
    create_folder(args.outdir)
    print('num of scans:',len(scan_ids), 'processing with',n_workers,'threads')
    
    # process('eee5b056-ee2d-28f4-9bbe-f7ddf37895a6')
    
    if n_workers > 0:
        process_map(process, scan_ids, max_workers=n_workers, chunksize=1 )
    else:
        for scan_id in tqdm(scan_ids):
            process(scan_id)
            # break