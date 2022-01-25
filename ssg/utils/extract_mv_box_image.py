#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 22:50:59 2021

@author: sc
"""
import os,glob
import h5py
import numpy as np
import imageio
from PIL import Image
from torchvision.ops import roi_align
from torchvision import transforms
import torchvision
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map 
from collections import defaultdict
import codeLib
from codeLib.torch.visualization import show_tv_grid, show_tensor_images

overwrite=False
toTensor = transforms.ToTensor()
resize = transforms.Resize([256,256])

pth_proposal = '/home/sc/research/PersistentSLAM/python/2DTSG/data/mv_scannet_test/proposals.h5'
pth_img = '/media/sc/SSD1TB2/dataset/scannet/images.h5'
pth_out = '/media/sc/SSD1TB2/dataset/scannet/roi_images/'
fbase = '/media/sc/SSD1TB2/dataset/scannet/'
pth_link = '/media/sc/SSD1TB2/dataset/scannet/roi.h5'
pattern = './roi_images/*.h5'

def safe_acos(x, epsilon=1e-7): 
    # 1e-7 for float is a good value
    return torch.acos(torch.clamp(x, -1 + epsilon, 1 - epsilon))

def getAngle(P,Q):
    # R = P @ Q.T
    theta = (torch.trace(P) -1)/2
    return safe_acos(theta) * (180/np.pi)

def is_close(pose, poses:list, t_a: float = 5, t_t: float = 0.3):
    # if len(poses)==0: return False
    for p_t in poses:
        diff_t = np.linalg.norm(pose[:3,3]-p_t[:3,3])
        
        if diff_t < t_t: 
            # print('  t',diff_t)
            return True
        diff_angle = getAngle(pose[:3,:3], p_t[:3,:3])
        
        if diff_angle < t_a: 
            # print('a  ',diff_angle)
            return True
    return False

# def process(path, scan_data, imgs):
def process(scan_id):
    path = os.path.join(pth_out, scan_id+'.h5')
    if os.path.isfile(path):
        if not overwrite: return
        else: os.remove(path)
    f = h5py.File(pth_img, 'r')
    if scan_id not in f: return
    fp = h5py.File(pth_proposal, 'r')
    scan_data = fp[scan_id]
    imgs = f[scan_id]
    
    with h5py.File(path,'w') as h5f:
        kfs = scan_data['kfs']
        nodes = scan_data['nodes']
        node_ids = list(nodes.keys())
        
        # pbar2 = tqdm(node_ids,leave=False)
        for oid in node_ids:
            node = nodes[oid]
            cls_label = node.attrs['label']
            # pbar2.set_description('process node {} {}'.format(oid, cls_label))
            
            kf_indices = np.asarray(node)
            
            img_boxes = list()
            fidx2idx=list()
            poses=list()
            counter=0
            for fid in kf_indices:
                bfid = imgs['indices'][fid] # convert frame idx to the buffer idx 
                pose = torch.from_numpy(imgs['poses'][bfid])
                
                if pose.isnan().any() or pose.isinf().any(): continue
                
                if is_close(pose, poses):
                    continue
                poses.append(pose)
                
                
                #  bounding box
                kf = kfs[str(fid)]                
                kf_seg2idx = {v[0]:v[1] for v in kf.attrs['seg2idx']}
                bid = kf_seg2idx[int(oid)]
                kfdata = np.asarray(kf)
                box = kfdata[bid,:4]
                oc  = kfdata[bid,4]
                box = torch.from_numpy(box).float().view(1,-1)
                
                
                # load image
                img_data = imgs['rgb'][bfid]
                img = imageio.imread(img_data)
                img = Image.fromarray(img)
                timg = toTensor(img).unsqueeze(0)
                w = box[:,2] - box[:,0]
                h = box[:,3] - box[:,1]
                region = roi_align(timg,[box], [h,w])
                region = resize(region).squeeze(0)
                img_boxes.append( region )
                fidx2idx.append( (fid, counter) )
                counter+=1
                # plt.imshow(region.permute(1,2,0))
                # plt.show()
            if len(img_boxes)==0: continue
            img_boxes = torch.stack(img_boxes)
            # show_tensor_images(img_boxes, title=cls_label)
            
            h5d = h5f.create_dataset(oid,data=img_boxes.numpy(), compression="gzip", compression_opts=9)
            h5d.attrs['seg2idx'] = fidx2idx
    f.close()
    fp.close()
if __name__ == '__main__':
    if not os.path.exists(pth_out):
        os.makedirs(pth_out)
    with h5py.File(pth_proposal, 'r') as fp:
        scan_ids = list(fp.keys())
    
    # for scan_id in scan_ids: process(scan_id)
    n_workers=4
    process_map(process, scan_ids, max_workers=n_workers, chunksize=1 )

    with h5py.File(pth_link, 'w') as h5f:
        for path in glob.glob(os.path.join(fbase,pattern)):
            name = path.split('/')[-1]
            name = path[len(fbase):]
            scan_id = path.split('/')[-1].split('.')[0]
            h5f[scan_id] = h5py.ExternalLink(name, './')