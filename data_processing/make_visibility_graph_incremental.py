import argparse, os, pandas, h5py, logging,json,torch
import pathlib
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

# structure_labels = ['wall','floor','ceiling']

# width=540
# height=960

DEBUG=True
DEBUG=False

random_clr_i = [color_rgb(rand_24_bit()) for _ in range(1500)]
random_clr_i[0] = (0,0,0)
# ffont = '/home/sc/research/PersistentSLAM/python/2DTSG/files/Raleway-Medium.ttf'

def Parse():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c','--config',default='./configs/config_default.yaml',required=False)
    parser.add_argument('-o','--outdir', help='output dir',required=True)
    # parser.add_argument('--target_name','-n', default='graph.json', help='target graph json file name')
    parser.add_argument('--overwrite', type=int, default=0, help='overwrite existing file.')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    return parser

if __name__ == '__main__':
    args = Parse().parse_args()
    cfg = codeLib.Config(args.config)
    lcfg = cfg.data.image_graph_generation
    
    outdir=args.outdir
    min_size=lcfg.min_box_ratio
    min_obj=lcfg.min_obj# float(args.min_object)
    
    '''create log'''
    pathlib.Path(outdir).mkdir(exist_ok=True,parents=True)
    name_log = os.path.split(__file__)[-1].replace('.py','.log')
    path_log = os.path.join(outdir,name_log)
    logging.basicConfig(filename=path_log, level=logging.DEBUG)
    logger_py = logging.getLogger(name_log)
    logger_py.info(f'create log file at {path_log}')
    if DEBUG:
        logger_py.setLevel('DEBUG')
    else:
        logger_py.setLevel('INFO')
        
    '''save config'''
    name_cfg = os.path.split(__file__)[-1].replace('.py','.json')
    pth_cfg = os.path.join(outdir,name_cfg)
    all_args = {**vars(args),**lcfg}
    with open(pth_cfg, 'w') as f:
            json.dump(all_args, f, indent=2)
    
    '''create output file'''
    try:
        h5f = h5py.File(os.path.join(outdir,define.NAME_VIS_GRAPH), 'a')
    except:
        h5f = h5py.File(os.path.join(outdir,define.NAME_VIS_GRAPH), 'w')
    
    '''read scenes'''
    fdata = cfg.data.path_3rscan_data
    '''read all scan ids'''
    scan_ids  = sorted(util_data.read_all_scan_ids(cfg.data.path_split))
    logger_py.info(f'There are {len(scan_ids)} scans to be processed')
    
    '''process'''
    invalid_scans=list()
    valid_scans=list()
    pbar = tqdm(scan_ids)
    for scan_id in pbar: #['scene0000_00']: #glob.glob('scene*'):
        if DEBUG: scan_id = '095821f7-e2c2-2de1-9568-b9ce59920e29'
        logger_py.info(scan_id)
        pbar.set_description('processing {}'.format(scan_id))
        
        pth_graph = os.path.join(fdata,scan_id, lcfg.graph_name)
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
                
        '''calculate '''
        kfs = dict() # key str(frame_id), values: {'idx': fid, 'bboxes': {object_id: [xmin,ymin,xmax,ymax]} }
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
            
            path = os.path.join(fdata,scan_id,'sequence',os.path.basename(path))
            img = np.array(Image.open(path))
            img = np.rot90(img,3).copy()# Rotate image
            
            boxes=list()
            clrs =list()
            labelNames=list()
            
            # print('kfid',kf_['id'])
            
            # scale = [kf_['rgb_dims'][0]/kf_['mask_dims'][0],kf_['rgb_dims'][1]/kf_['mask_dims'][1] ]
            scale = [1/kf_['mask_dims'][0],1/kf_['mask_dims'][1] ] #NOTE: normalize. 
            for oid in bboxes:
                if int(oid) == 0: continue
                # print('oid',oid)
                
                '''scale bounding box back'''
                box = bboxes[oid] # xmin,ymin,xmax,ymax
                box[0] *= scale[0]
                box[1] *= scale[1]
                box[2] *= scale[0]
                box[3] *= scale[1]
                
                assert box[0]<=1 
                assert box[0]>=0
                assert box[1]<=1 
                assert box[1]>=0
                assert box[2]<=1 
                assert box[2]>=0
                assert box[3]<=1 
                assert box[3]>=0
                
                '''Check width and height'''
                w_ori = box[2]-box[0]
                h_ori = box[3]-box[1]
                if w_ori  < min_size or h_ori < min_size: 
                    logger_py.debug("fid({})oid({}) dims: {} {} < min_size {}".format(fid,oid,w_ori,h_ori,min_size))
                    continue
            
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
                box_r[0] = 1-box[1]
                box_r[1] = box[0]
                box_r[2] = 1-box[3]
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
            invalid_scans.append(scan_id)
            continue
        valid_scans.append(scan_id)
        # valid_scans+=1
        
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
        if DEBUG: break
    
    if len(invalid_scans)+len(valid_scans)>0:
        logger_py.info('percentage of invalid scans: {}({}/{})'.format(len(invalid_scans)/(len(invalid_scans)+len(valid_scans)),len(invalid_scans),(len(invalid_scans)+len(valid_scans))))
        h5f.attrs['classes'] = util_label.NYU40_Label_Names
        # write args
        tmp = vars(args)
        if 'args' in h5f: del h5f['args']
        h5f.create_dataset('args',data=())
        for k,v in tmp.items():
            h5f['args'].attrs[k] = v
    else:
        logger_py.debug('no scan processed!')
        
    logger_py.info('invalid scans: {}'.format(invalid_scans))
    h5f.close()
    