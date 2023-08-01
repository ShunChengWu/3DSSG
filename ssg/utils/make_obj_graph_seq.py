import argparse
import os
import pandas
import h5py
import logging
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import codeLib
from codeLib.torch.visualization import show_tv_grid
from codeLib.common import color_rgb, rand_24_bit
from codeLib.utils.util import read_txt_to_list
from collections import defaultdict
from ssg import define
from ssg.utils import util_label
from ssg.objects.node import Node
from ssg.utils import util_data

structure_labels = ['wall', 'floor', 'ceiling']

width = 540
height = 960

DEBUG = True
DEBUG = False

random_clr_i = [color_rgb(rand_24_bit()) for _ in range(1500)]
random_clr_i[0] = (0, 0, 0)
ffont = '/home/sc/research/PersistentSLAM/python/2DTSG/files/Raleway-Medium.ttf'


def Parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-o', '--outdir', default='/home/sc/research/PersistentSLAM/python/3DSSG/data/3RScan_ScanNet20/', help='output dir', required=True)
    parser.add_argument(
        '--min_object', help='if less thant min_obj objects, ignore image', default=1)
    parser.add_argument('--min_size', default=0.1, help='min length on bbox')
    parser.add_argument('--target_name', '-n',
                        default='2dssg_seq.json', help='target graph json file name')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='overwrite existing file.')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    return parser


def process(data, min_obj):
    # key str(frame_id), values: {'idx': fid, 'bboxes': {object_id: [xmin,ymin,xmax,ymax]} }
    kfs = dict()
    objects = dict()
    node2kfs = dict()
    for kf_ in data['kfs']:
        bboxes = kf_['bboxes']
        if len(bboxes) < min_obj:
            continue
        width = kf_['rgb_dims'][0]
        height = kf_['rgb_dims'][1]
        path = kf_['path']
        fname = os.path.basename(path)
        fid = int(''.join([x for x in fname if x.isdigit()]))

        if str(fid) not in kfs:
            kfs[str(fid)] = dict()
        kf = kfs[str(fid)]
        kf['idx'] = fid
        kf['bboxes'] = dict()

        img = np.array(Image.open(path))
        img = np.rot90(img, 3).copy()  # Rotate image

        boxes = list()
        clrs = list()
        labelNames = list()

        # print('kfid',kf_['id'])

        # scale = [kf_['rgb_dims'][0]/kf_['mask_dims'][0],kf_['rgb_dims'][1]/kf_['mask_dims'][1] ]
        # NOTE: normalize.
        scale = [1/kf_['mask_dims'][0], 1/kf_['mask_dims'][1]]
        for oid in bboxes:
            if int(oid) == 0:
                continue
            # print('oid',oid)

            '''scale bounding box back'''
            box = bboxes[oid]  # xmin,ymin,xmax,ymax
            box[0] *= scale[0]
            box[1] *= scale[1]
            box[2] *= scale[0]
            box[3] *= scale[1]

            assert box[0] <= 1
            assert box[0] >= 0
            assert box[1] <= 1
            assert box[1] >= 0
            assert box[2] <= 1
            assert box[2] >= 0
            assert box[3] <= 1
            assert box[3] >= 0

            '''Check width and height'''
            w_ori = box[2]-box[0]
            h_ori = box[3]-box[1]
            if w_ori < args.min_size or h_ori < args.min_size:
                continue

            '''check format is correct'''
            assert 0 <= box[0] < box[2]
            assert 0 <= box[1] < box[3]

            # check boundary
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
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
            box_r = [0, 0, 0, 0]
            box_r[0] = 1-box[1]
            box_r[1] = box[0]
            box_r[2] = 1-box[3]
            box_r[3] = box[2]

            box[0] = min(box_r[0], box_r[2])
            box[2] = max(box_r[0], box_r[2])
            box[1] = min(box_r[1], box_r[3])
            box[3] = max(box_r[1], box_r[3])

            boxes.append(box)
            labelNames.append('unknown')
            clrs.append((255, 255, 255))

            kf['bboxes'][oid] = box

            if str(oid) not in objects:
                objects[str(oid)] = dict()
            obj = objects[str(oid)]
            obj['label'] = 'unknown'

            if int(oid) not in node2kfs:
                node2kfs[int(oid)] = list()
            node2kfs[int(oid)].append(fid)

            # break
        # if DEBUG:
        #     torch_img = torch.from_numpy(img).permute(2,0,1)
        #     boxes = torch.tensor(boxes, dtype=torch.float)
        #     result = draw_bounding_boxes(torch_img, boxes,
        #                                     labels=labelNames,
        #                                     colors=clrs,
        #                                     width=5,
        #                                     font=ffont,
        #                                     font_size=50)
        #     show_tv_grid(result)
        #     plt.show()
            # print('')
    return kfs, objects, node2kfs


def save_one_scene(data: dict, h5data):
    objects = data['objects']
    kfs = data['kfs']
    node2kfs = data['node2kfs']

    '''save'''
    seg2idx = dict()
    h5node = h5data.create_group('nodes')
    for oid, obj in objects.items():
        # Save the indices of KFs
        dset = h5node.create_dataset(oid, data=node2kfs[int(oid)])
        dset.attrs['label'] = str(obj['label'])

    if 'kfs' in h5data:
        del h5data['kfs']
    dkfs = h5data.create_group('kfs')
    for k, v in kfs.items():
        boxes = v['bboxes']
        # occlu = v['occlution']
        boxes_ = list()
        seg2idx = dict()
        for ii, kk in enumerate(boxes):
            boxes_.append(boxes[kk]+[0])
            seg2idx[int(kk)] = ii
        dset = dkfs.create_dataset(k, data=boxes_)
        dset.attrs['seg2idx'] = [(k, v) for k, v in seg2idx.items()]


if __name__ == '__main__':
    args = Parse().parse_args()
    DEBUG = args.debug
    print(args)
    outdir = args.outdir
    # min_oc=float(args.min_occ) # maximum occlusion rate authorised
    min_obj = float(args.min_object)
    # gt2d_dir = args.gt2d_dir

    '''create output file'''
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    '''logging'''
    filename = os.path.basename(__file__).split('.')[0]
    logging.basicConfig(filename=os.path.join(
        outdir, filename+'.log'), level=logging.DEBUG)
    logger_py = logging.getLogger(__name__)
    if DEBUG:
        logger_py.setLevel('DEBUG')
    else:
        logger_py.setLevel('INFO')
    logger_py.debug('args')
    logger_py.debug(args)

    # save configs
    with open(os.path.join(outdir, filename+'.txt'), 'w') as f:
        for k, v in args.__dict__.items():
            f.write('{}:{}\n'.format(k, v))
        pass
    try:
        h5f = h5py.File(os.path.join(outdir, 'proposals.h5'), 'a')
    except:
        h5f = h5py.File(os.path.join(outdir, 'proposals.h5'), 'w')
    # h5f.attrs['label_type'] = args.label_type

    '''read scenes'''
    fdata = os.path.join('data', '3RScan', "data",
                         "3RScan")  # os.path.join(define.DATA_PATH)
    # train_ids = read_txt_to_list(os.path.join('files','train_scans.txt'))
    # val_ids = read_txt_to_list(os.path.join('files','validation_scans.txt'))
    test_ids = read_txt_to_list(os.path.join('files', 'test_scans.txt'))

    # print(len(train_ids))
    # print(len(val_ids))
    print(len(test_ids))
    # scan_ids  = sorted( train_ids + val_ids + test_ids)
    scan_ids = sorted(test_ids)
    print(len(scan_ids))

    pbar = tqdm(scan_ids)

    '''process'''
    invalid_scans = list()
    valid_scans = list()
    for scan_id in pbar:  # ['scene0000_00']: #glob.glob('scene*'):
        if DEBUG:
            scan_id = '095821f7-e2c2-2de1-9568-b9ce59920e29'
        logger_py.info(scan_id)
        pbar.set_description('processing {}'.format(scan_id))

        pth_graph = os.path.join(fdata, scan_id, args.target_name)
        if os.path.isfile(pth_graph):
            with open(pth_graph, "r") as read_file:
                data = json.load(read_file)[scan_id]
        else:
            invalid_scans.append(scan_id)
            continue

        '''check if the scene has been created'''
        if scan_id in h5f:
            if args.overwrite == 0:
                logger_py.info('exist. skip')
                continue
            else:
                del h5f[scan_id]

        outputs = dict()
        for timestamp in data:
            '''process'''
            kfs, objects, node2kfs = process(
                data[timestamp],
                min_obj
            )

            '''check if each node has at least one kf'''
            to_deletes = []
            for k, v in objects.items():
                if int(k) not in node2kfs or len(node2kfs[int(k)]) == 0:
                    to_deletes.append(k)
            for idx in to_deletes:
                objects.pop(idx)

            '''check if empty'''
            if len(objects) == 0:
                invalid_scans.append(scan_id)
                continue
            valid_scans.append(scan_id)

            outputs[timestamp] = dict()
            outputs[timestamp]['objects'] = objects
            outputs[timestamp]['kfs'] = kfs
            outputs[timestamp]['node2kfs'] = node2kfs

        '''save per timestamp'''
        h5_scan = h5f.create_group(scan_id)
        for timestamp, data in outputs.items():
            h5_timestamp = h5_scan.create_group(timestamp)
            save_one_scene(data, h5_timestamp)

            # '''save'''
            # # Save objects.
            # h5g = h5f.create_group(scan_id)
            # seg2idx = dict()
            # h5node = h5g.create_group('nodes')
            # for idx, data in enumerate(objects.items()):
            #     oid, obj = data
            #     # Save the indices of KFs
            #     dset = h5node.create_dataset(oid,data=node2kfs[int(oid)])
            #     dset.attrs['label'] = str(obj['label'])
            #     # dset.attrs['occlution'] = str(obj['occlution'])

            # # kfs_=list()
            # if 'kfs' in h5g: del h5g['kfs']
            # dkfs = h5g.create_group('kfs')
            # for idx, data in enumerate(kfs.items()):
            #     k,v = data
            #     boxes = v['bboxes']
            #     # occlu = v['occlution']
            #     boxes_=list()
            #     seg2idx=dict()
            #     for ii, kk in enumerate(boxes):
            #         boxes_.append(boxes[kk]+[0])
            #         seg2idx[int(kk)] = ii
            #     dset = dkfs.create_dataset(k,data=boxes_)
            #     dset.attrs['seg2idx'] = [(k,v) for k,v in seg2idx.items()]
        if DEBUG:
            break
        # break
    print('')
    if len(invalid_scans)+len(valid_scans) > 0:
        print('percentage of invalid scans:', len(invalid_scans)/(len(invalid_scans)+len(valid_scans)),
              '(', len(invalid_scans), ',', (len(invalid_scans)+len(valid_scans)), ')')
        h5f.attrs['classes'] = util_label.NYU40_Label_Names
        # write args
        tmp = vars(args)
        if 'args' in h5f:
            del h5f['args']
        h5f.create_dataset('args', data=())
        for k, v in tmp.items():
            h5f['args'].attrs[k] = v
        # with open(os.path.join(outdir,'classes.txt'), 'w') as f:
        #     for cls in util_label.NYU40_Label_Names:
        #         f.write('{}\n'.format(cls))
    else:
        print('no scan processed')
    print('invalid scans:', invalid_scans)
    h5f.close()
