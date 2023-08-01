#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:33:42 2021

@author: sc
"""
import logging,os,argparse
import codeLib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from codeLib.torch.visualization import show_tv_grid
from codeLib.common import color_rgb, rand_24_bit
from codeLib.utils.util import read_txt_to_list
from codeLib.object import BoundingBox
import torch
from torchvision.utils import draw_bounding_boxes
from collections import defaultdict
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map 
from ssg import define
from ssg.utils import util_label
from ssg.utils.util_3rscan import read_3rscan_info, load_semseg
from ssg.utils.util_data import read_all_scan_ids
import pathlib

helpmsg = 'Generate object bounding box and occlusion on 3RScan dataset using the rendered label and instance'
parser = argparse.ArgumentParser(description=helpmsg,formatter_class = argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c','--config',default='./configs/config_default.yaml',required=False)
parser.add_argument('--thread', type=int, default=0, help='The number of threads to be used.')
parser.add_argument('--overwrite', action='store_true', help='overwrite or not.')
args = parser.parse_args()
cfg = codeLib.Config(args.config)

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
    '''
    x: tensor image with shape [height, width]
    '''
    width,height = x.shape[-1],x.shape[-2]
    n_w, n_h = width//down_scale, height//down_scale
    
    if down_scale==1 or n_w==0 or n_h ==0:
        sampled = x
    else:
        sampled = np.zeros([n_h,n_w],dtype=np.byte)
        for w in range(n_w-1):
            for h in range(n_h-1):
                sampled[h,w] = x[h*down_scale:(h+1)*down_scale, w*down_scale: (w+1)*down_scale].any()
    # plt.imshow(sampled)
    # plt.show()
    return sampled.sum()/sampled.size
    
class LabelImage(object):
    fid=int()
    l_img=np.array([])
    i_img=np.array([])
    detections = defaultdict(Detection)
    def __init__(self, frame_idx:int, iimg_data, limg_data, downscale:int):
        self.fid = frame_idx
        self.l_img = limg_data
        self.i_img = iimg_data
        self.downscale = downscale
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

    def process(self)->defaultdict(Detection):
        instances = np.unique(self.i_img)
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
        
            oc = mask_occupancy(self.i_img[box.y_min():box.y_max(),box.x_min():box.x_max()]==inst,down_scale=self.downscale)
            
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
    outdir = lcfg.path_2dgt
    foutput = os.path.join(outdir,scan_id+define.TYPE_2DGT)
    if os.path.isfile(foutput):
        if os.stat(foutput).st_size == 0:
            os.remove(foutput)
        else:
            if args.overwrite:
                os.remove(foutput)
            else:
                return    
    
    # load semseg
    pth_semseg = os.path.join(cfg.data.path_3rscan_data,scan_id,define.SEMSEG_FILE_NAME)
    mapping = load_semseg(pth_semseg)
    mapping[0] = 'none'
        
    # get number of scans
    scan_info = read_3rscan_info(os.path.join(cfg.data.path_3rscan_data,scan_id,define.IMG_FOLDER_NAME,define.INFO_NAME))
    n_images = int(scan_info['m_frames.size'])
    
    # check all images exist
    for frame_id in range(n_images):
        # pth_label = os.path.join(fdata,scan_id,'sequence', label_filepattern.format(frame_id))
        pth_inst  = os.path.join(cfg.data.path_3rscan_data,scan_id,'sequence', define.NAME_PATTERN_INSTANCE_IMG.format(frame_id))
        
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
        pth_inst  = os.path.join(cfg.data.path_3rscan_data,scan_id,'sequence', define.NAME_PATTERN_INSTANCE_IMG.format(frame_id))
        
        # limg_data = np.array(Image.open(pth_label), dtype=np.uint8)
        iimg_data = np.array(Image.open(pth_inst), dtype=np.uint8)
        
        # check keys
        diff_ids = set( np.unique(iimg_data) ).difference(set(mapping.keys()))
        for id in diff_ids: mapping[id]='none'
        
        #
        limg_data = np.vectorize(mapping.__getitem__)(iimg_data)
        limg = LabelImage(frame_id, iimg_data, limg_data, cfg.data.image_graph_generation.occupancy_downscale)
        
        # limg.show()
        # print(limg)
        for inst, detection in limg.detections.items():
            fp.write('{} {} {} {} {} {} {} {}\n'.\
                     format(frame_id, inst, detection.label.replace(' ','_').encode('utf8'), detection.max_iou, detection.box[0], detection.box[1], detection.box[2], detection.box[3]  ))
    fp.close()
    
if __name__ =='__main__':    
    '''alias'''
    lcfg = cfg.data.image_graph_generation
    outdir = lcfg.path_2dgt
    path_3rscan = cfg.data.path_3rscan
    
    '''create log'''
    pathlib.Path(outdir).mkdir(exist_ok=True,parents=True)
    name_log = os.path.split(__file__)[-1].replace('.py','.log')
    path_log = os.path.join(outdir,name_log)
    logging.basicConfig(filename=path_log, level=logging.INFO)
    logger_py = logging.getLogger(name_log)
    logger_py.info(f'create log file at {path_log}')

    '''read all scan ids'''
    scan_ids  = sorted( read_all_scan_ids(cfg.data.path_split))
    logger_py.info(f'There are {len(scan_ids)} scans to be processed')
    
    '''get label mapping'''
    Scan3R528, NYU40,Eigen13,RIO27,RIO7 = util_label.getLabelNames(define.PATH_LABEL_MAPPING)
    Scan3R528[0] = 'none'
    
    '''generate random color for visualization'''
    random_clr_i = [color_rgb(rand_24_bit()) for _ in range(1500)]
    random_clr_l = {v:color_rgb(rand_24_bit()) for k,v in Scan3R528.items()}
    random_clr_l['none'] = (0,0,0)
    ffont = os.path.join(cfg.data.path_file,'Raleway-Medium.ttf')
    
    logger_py.info(f'start process {len(scan_ids)} scans with {args.thread} threads.')
    
    if args.thread > 0:
        print('process with process_map')
        print("number of scans: {}. number of threads: {}".format(len(scan_ids),args.thread))
        print('chunk size {}'.format(len(scan_ids)//args.thread))
        process_map(process, scan_ids, max_workers=args.thread, chunksize=len(scan_ids)//args.thread )
    else:
        for scan_id in tqdm(scan_ids):
            process(scan_id)