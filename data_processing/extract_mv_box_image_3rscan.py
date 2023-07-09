#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 22:50:59 2021

@author: sc


This script load proposals.h5, generated from make_obj_graph_3rscan.py. 
Extract bounding boxes from the input data.
It reads images and poses according to the frame idx, first compare whether
a close frame exist, if not then extract bounding boxes to the given size.

Save each bbox images to a indivisual file. Then create a link file to connect 
them. 
```
roi_images/
 scan_id2.h5
 scan_id1.h5
 ...
roi_images.h5
```

In roi_images.h5 as data:
data = {scan_id1: np.array(roi_img_data)
 scan_id2: ...}    
data.attrs['seg2idx']: map frame idx to the buffer index.

"""
import logging
import os
import glob
import pathlib
import h5py
import numpy as np
from PIL import Image
from torchvision.ops import roi_align
from torchvision import transforms
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import codeLib
from codeLib.math import getAngle
from codeLib.torch.visualization import show_tv_grid, show_tensor_images
import argparse
from ssg import define


def Parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-c', '--config', default='./configs/config_default.yaml', required=False)
    parser.add_argument('--thread', type=int, default=0, help='thread.')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite or not.')
    parser.add_argument('-f', '--filename',
                        help='path to proposals.h5', required=True)
    # parser.add_argument('-o','--outdir',default='/media/sc/SSD1TB/dataset/3RScan/', help='roi images',required=True)
    parser.add_argument('--debug', action='store_true', help='debug mode')
    return parser


args = Parse().parse_args()
cfg = codeLib.Config(args.config)

# '/home/sc/research/PersistentSLAM/python/3DSSG/data/3RScan_3RScan160/proposals.h5'
pth_proposal = args.filename
overwrite = args.overwrite
n_workers = args.thread
outdir = cfg.data.roi_img_path
# fbase = args.outdir
# '/media/sc/SSD1TB/dataset/3RScan/roi_images/'
pth_out = os.path.join(outdir, define.NAME_ROI_IMAGE.split('.')[0])
pth_link = os.path.join(outdir, define.NAME_ROI_IMAGE)

toTensor = transforms.ToTensor()
resize = transforms.Resize([256, 256])
# './data/3RScan/data/3RScan'#define.DATA_PATH# '/media/sc/SSD1TB/dataset/3RScan/data/3RScan'
fdata = cfg.data.path_3rscan_data
pattern = os.path.join(define.NAME_ROI_IMAGE.split('.')[0], '*.h5')
rgb_filepattern = define.RGB_NAME_FORMAT  # 'frame-{0:06d}.color.jpg'
pose_filepattern = define.POSE_NAME_FORMAT  # 'frame-{0:06d}.pose.txt'


def is_close(pose, poses: list, t_a: float = 5, t_t: float = 0.3):
    # if len(poses)==0: return False
    for p_t in poses:
        diff_t = np.linalg.norm(pose[:3, 3]-p_t[:3, 3])

        if diff_t < t_t:
            # print('  t',diff_t)
            return True
        diff_angle = getAngle(pose[:3, :3], p_t[:3, :3])

        if diff_angle < t_a:
            # print('a  ',diff_angle)
            return True
    return False

# def process(path, scan_data, imgs):


def process(scan_id):
    path = os.path.join(pth_out, scan_id+'.h5')
    if os.path.isfile(path):
        if not overwrite:
            return
        else:
            os.remove(path)

    fp = h5py.File(pth_proposal, 'r')
    scan_data = fp[scan_id]
    kfs = scan_data['kfs']
    nodes = scan_data['nodes']
    node_ids = list(nodes.keys())
    if len(node_ids) == 0:
        return
    if n_workers == 0:
        logger_py.info(scan_id)
    with h5py.File(path, 'w') as h5f:
        for oid in node_ids:
            node = nodes[oid]
            cls_label = node.attrs['label']
            # pbar2.set_description('process node {} {}'.format(oid, cls_label))

            kf_indices = np.asarray(node)

            img_boxes = list()
            fidx2idx = list()
            poses = list()
            counter = 0
            for fid in kf_indices:
                pth_rgb = os.path.join(
                    fdata, scan_id, define.IMG_FOLDER_NAME, rgb_filepattern.format(fid))
                pth_pos = os.path.join(
                    fdata, scan_id, define.IMG_FOLDER_NAME, pose_filepattern.format(fid))
                '''load data'''
                img_data = Image.open(pth_rgb)
                img_data = np.rot90(img_data, 3)  # Rotate image
                pos_data = np.loadtxt(pth_pos)

                height, width = img_data.shape[0], img_data.shape[1]
                # bfid = imgs['indices'][fid] # convert frame idx to the buffer idx
                pose = torch.from_numpy(pos_data)

                if pose.isnan().any() or pose.isinf().any():
                    continue

                if is_close(pose, poses):
                    continue
                poses.append(pose)

                #  bounding box
                kf = kfs[str(fid)]
                kf_seg2idx = {v[0]: v[1] for v in kf.attrs['seg2idx']}
                bid = kf_seg2idx[int(oid)]
                kfdata = np.asarray(kf)
                box = kfdata[bid, :4]
                oc = kfdata[bid, 4]

                assert box[0] <= 1
                assert box[0] >= 0
                assert box[1] <= 1
                assert box[1] >= 0
                assert box[2] <= 1
                assert box[2] >= 0
                assert box[3] <= 1
                assert box[3] >= 0

                # Denormalize
                box[0] *= width
                box[2] *= width
                box[1] *= height
                box[3] *= height

                box = torch.from_numpy(box).float().view(1, -1)
                timg = toTensor(img_data.copy()).unsqueeze(0)
                w = box[:, 2] - box[:, 0]
                h = box[:, 3] - box[:, 1]
                # if n_workers==0: logger_py.info('box: {}, dim: {}'.format(box,[h,w]))
                region = roi_align(timg, [box], [h, w])
                region = resize(region).squeeze(0)
                img_boxes.append(region)
                fidx2idx.append((fid, counter))
                counter += 1
                # plt.imshow(region.permute(1,2,0))
                # plt.show()
            if len(img_boxes) == 0:
                raise RuntimeError(
                    "scan:{}.node_id:{} has 0 img boxes".format(scan_id, oid))
            img_boxes = torch.stack(img_boxes)
            if args.debug:
                show_tensor_images(img_boxes, title=cls_label)

            h5d = h5f.create_dataset(
                oid, data=img_boxes.numpy(), compression="gzip", compression_opts=9)
            # TODO: was seg2idx. renamed. may create bug in some other files.
            h5d.attrs['fid2idx'] = fidx2idx
    fp.close()


if __name__ == '__main__':
    '''create log'''
    pathlib.Path(outdir).mkdir(exist_ok=True, parents=True)
    name_log = os.path.split(__file__)[-1].replace('.py', '.log')
    path_log = os.path.join(outdir, name_log)
    logging.basicConfig(filename=path_log, level=logging.INFO)
    logger_py = logging.getLogger(name_log)
    logger_py.info(f'create log file at {path_log}')

    pathlib.Path(pth_out).mkdir(exist_ok=True, parents=True)

    '''get all scans'''
    with h5py.File(pth_proposal, 'r') as fp:
        scan_ids = [s for s in list(fp.keys()) if isinstance(
            fp[s], h5py._hl.group.Group)]

    # for scan_id in scan_ids: process(scan_id)
    if n_workers > 0:
        process_map(process, scan_ids, max_workers=n_workers, chunksize=1)
    else:
        pbar = tqdm(scan_ids)
        for scan_id in pbar:
            # scan_id = '4acaebcc-6c10-2a2a-858b-29c7e4fb410d'
            pbar.set_description(scan_id)
            process(scan_id)

    '''create a link h5 db'''
    if outdir[-1] != '/':
        outdir += '/'
    with h5py.File(pth_link, 'w') as h5f:
        for path in glob.glob(os.path.join(outdir, pattern)):
            # name = path.split('/')[-1]
            name = path[len(outdir):]
            scan_id = path.split('/')[-1].split('.')[0]
            h5f[scan_id] = h5py.ExternalLink(name, './')
