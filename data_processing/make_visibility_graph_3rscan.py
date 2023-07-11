import argparse, os, pandas, h5py, logging
import pathlib
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse, os, pandas, h5py, logging,json,torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.utils import draw_bounding_boxes
import codeLib
from codeLib.torch.visualization import show_tensor_images, show_tv_grid
from codeLib.common import color_rgb, rand_24_bit
from codeLib.utils.util import read_txt_to_list
from collections import defaultdict
from ssg import define
from ssg.utils import util_label
from ssg.utils import util_3rscan
from ssg.utils.util_3rscan import load_semseg
from codeLib.torch.visualization import show_tensor_images
from ssg.utils.util_data import read_all_scan_ids

DEBUG=True
DEBUG=False

random_clr_i = [color_rgb(rand_24_bit()) for _ in range(1500)]
random_clr_i[0] = (0,0,0)

def Parse():
    helpmsg = 'Generate entity visibility graph for 3D scans'
    parser = argparse.ArgumentParser(description=helpmsg,formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c','--config',default='./configs/config_default.yaml',required=False)
    parser.add_argument('-o','--outdir', help='output dir',required=True)
    parser.add_argument('-l','--label_type',default='3rscan160', choices=['nyu40','eigen13','rio27', 'rio7','3rscan','3rscan160','scannet20'], 
                        help='target label type.',required=True)
    parser.add_argument('--overwrite', action='store_true', help='overwrite or not.')
    return parser

def get_bbx_wo_flatcheck(lcfg, fn,min_oc, mapping, min_size:list=[240,240], image_size:tuple=(960,540)):
    obj_by_img={}  
    data = pandas.read_csv(fn, delimiter= ' ')      
    # imgs = f[scan_id]
    obj_set=set()
    obj_set_f = set()
    filter_counter = defaultdict(int)
    filter_label = defaultdict(set)
    width = image_size[-1]
    height = image_size[0]
    
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
             if DEBUG: logger_py.debug(msg.format(fname,oid,olabel)+' not in '+mapping.keys())
             filter_counter['label']+=1
             filter_label['label'].add(olabel)
             continue
             raise RuntimeError('all labels should be included in the mapping dict',mapping.keys(), 'query',olabel)
        olabel = mapping[olabel]
        # structure
        if lcfg.skip_structure>0:
          if olabel in structure_labels:
              if DEBUG: logger_py.debug(msg.format(fname,oid,olabel)+' structure label')
              filter_counter['struc']+=1
              filter_label['struc'].add(olabel)
              continue
        # On boarder
        if lcfg.skip_edge>0:
            if float(x1)<1 or float(y1)<1 or width < float(x2) or height < float(y2):
                if DEBUG: logger_py.debug(msg.format(fname,oid,olabel)+': on edge')
                filter_counter['edge']+=1
                filter_label['edge'].add(olabel)
                continue
        # Occurence too low
        oc=float(oc)
        oc = round(oc, 3)
        if oc<min_oc: # if occlusion rate is over the maximum authorised, then skip
            if DEBUG: logger_py.debug(msg.format(fname,oid,olabel)+' occluded '+oc+'<'+min_oc)
            filter_counter['occ']+=1
            filter_label['occ'].add(olabel)
            continue
      
          # too smal
        if lcfg.skip_size>0:
            size = [x2-x1,y2-y1]
            if size[0] < min_size[0] or size[1] < min_size[1]:
                if DEBUG: logger_py.debug(msg.format(fname,oid,olabel)+' too small '+ size)
                filter_counter['size']+=1
                filter_label['size'].add(olabel)
                continue
      
        if fname not in obj_by_img:
            obj_by_img[fname]=[fname,[]]
        obj_by_img[fname][1].append([oid,olabel,oc,float(x1),float(y1),float(x2),float(y2)])      
        obj_set_f.add(oid)
    
    logger_py.debug('filtered type and classes')
    for k,v in filter_counter.items():
         logger_py.debug('{}: {}. {}'.format(k,v,filter_label[k]))
    logger_py.debug('the obj filter ratio: {} ({}/{})'.format(len(obj_set_f)/len(obj_set),len(obj_set_f),len(obj_set)))
        
    return obj_by_img

def vis(datapath, scan_id,obj_by_img:dict,image_size:tuple=(960,540)):
    width = image_size[-1]
    height = image_size[0]
    
    for fname,v in obj_by_img.items():
        data_list = v[1]
        pth_inst = os.path.join(datapath,scan_id,define.IMG_FOLDER_NAME,define.NAME_PATTERN_INSTANCE_IMG.format(fname))
        iimg_data = np.array(Image.open(pth_inst), dtype=np.uint8)
        
        ori_list = set(np.unique(iimg_data).tolist()).difference([0])
        f_list = [x[0] for x in data_list]
        diff = ori_list.difference(f_list)
        if len(diff)==0: continue
        
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
    cfg = codeLib.Config(args.config)
    lcfg = cfg.data.image_graph_generation
    
    outdir=args.outdir
    min_oc=lcfg.min_occ#  float(args.min_occ) # maximum occlusion rate authorised
    min_obj=lcfg.min_obj# float(args.min_object)
    gt2d_dir = lcfg.path_2dgt#args.gt2d_dir
    structure_labels = define.STRUCTURE_LABELS
    
    '''create log'''
    pathlib.Path(outdir).mkdir(exist_ok=True,parents=True)
    name_log = os.path.split(__file__)[-1].replace('.py','.log')
    path_log = os.path.join(outdir,name_log)
    logging.basicConfig(filename=path_log, level=logging.INFO)
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
    
    '''create mapping'''
    label_names, label_name_mapping, label_id_mapping = util_label.getLabelMapping(args.label_type,define.PATH_LABEL_MAPPING)
    
    '''create output file'''
    try:
        h5f = h5py.File(os.path.join(outdir,define.NAME_VIS_GRAPH), 'a')
    except:
        os.remove(os.path.join(outdir,define.NAME_VIS_GRAPH))
        h5f = h5py.File(os.path.join(outdir,define.NAME_VIS_GRAPH), 'a')
    h5f.attrs['label_type'] = args.label_type
    
    '''read scenes'''
    fdata = cfg.data.path_3rscan_data
    '''read all scan ids'''
    scan_ids  = sorted( read_all_scan_ids(cfg.data.path_split))
    logger_py.info(f'There are {len(scan_ids)} scans to be processed')
    
    '''process'''
    invalid_scans=0
    valid_scans=0
    pbar = tqdm(scan_ids)
    for scan_id in pbar: #['scene0000_00']: #glob.glob('scene*'):
        logger_py.info(scan_id)
        pbar.set_description('processing {}'.format(scan_id))
        
        # load semseg
        pth_semseg = os.path.join(fdata,scan_id,define.SEMSEG_FILE_NAME)
        mapping = load_semseg(pth_semseg,label_name_mapping)
        mapping[0] = 'none'
        
        # load image info
        info_3rscan = util_3rscan.read_3rscan_info(os.path.join(fdata,scan_id,define.IMG_FOLDER_NAME,define.INFO_NAME))
        img_h,img_w = int(info_3rscan['m_colorWidth']), int(info_3rscan['m_colorHeight'])# we already rotated the input view when generating the rendered views. so swap h and w
        
        
        '''load 2dgt'''
        gt2d_file = os.path.join(gt2d_dir,scan_id+define.TYPE_2DGT)
        if not os.path.isfile(gt2d_file):
            logger_py.debug('file does not exists, skipping',scan_id+define.TYPE_2DGT)
            continue
        
        '''check if the scene has been created'''
        if scan_id in h5f: 
            if not args.overwrite: 
                logger_py.info('exist. skip')
                continue
            else:
                del h5f[scan_id]
        
        # read data and organize by frames
        obj_by_img=get_bbx_wo_flatcheck(lcfg, gt2d_file, min_oc,label_name_mapping, lcfg.min_box_size, (img_h,img_w))
            
        '''debug vis'''
        if DEBUG:
            vis(lcfg.path_3rscan_data,scan_id,obj_by_img,(img_h,img_w))
        
        # cluster the frames, each cluster correspond to a set of objects, all the elements of a cluster are images where these objects appear
        oidss={}
        img_by_objnum={}
        kfs = dict()
        objects = dict()
        node2kfs = dict()
        
        for (kId,(fname,seq)) in list(obj_by_img.items()):
            if len(seq)<min_obj:#if less thant min_obj objects, I don't keep the image
                continue
            if str(kId) not in kfs: kfs[str(kId)] = dict()
            
            kf = kfs[str(kId)]
            kf['idx'] = int(kId)
            if 'bboxes' not in kf: kf['bboxes'] = dict()
            if 'occlution' not in kf: kf['occlution'] = dict()
            for oid,olabel,oc,x1,y1,x2,y2 in seq:
                if str(oid) in kf['bboxes']: raise RuntimeError('exist')
                box = [x1/img_w,y1/img_h,x2/img_w,y2/img_h]
                assert box[0]<=1 
                assert box[0]>=0
                assert box[1]<=1 
                assert box[1]>=0
                assert box[2]<=1 
                assert box[2]>=0
                assert box[3]<=1 
                assert box[3]>=0
                kf['bboxes'][str(oid)] = box
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
                node2kfs[oid].append(kId)
        
        '''check filtered instances'''
        int_filtered_insts = [int(x) for x in objects]
        diffs = set(mapping.keys()).difference(set(int_filtered_insts))
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
            invalid_scans+=1
            continue
        valid_scans+=1
        
        h5g = h5f.create_group(scan_id)
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
            kId,v = data
            boxes = v['bboxes']
            occlu = v['occlution']
            boxes_=list()
            seg2idx=dict()
            for ii, kk in enumerate(boxes):
                boxes_.append(boxes[kk]+[occlu[kk]])
                seg2idx[int(kk)] = ii
            dset = dkfs.create_dataset(kId,data=boxes_)
            dset.attrs['seg2idx'] = [(k,v) for k,v in seg2idx.items()]  
            
    
    
    if invalid_scans+valid_scans>0:
        logger_py.info('percentage of invalid scans: {}({}/{})'.format(invalid_scans/(invalid_scans+valid_scans),invalid_scans,(invalid_scans+valid_scans)))
        # h5f.attrs['classes'] = util_label.NYU40_Label_Names
        # write args
        if 'args' in h5f: del h5f['args']
        h5f.create_dataset('args',data=())
        for k,v in all_args.items():
            h5f['args'].attrs[k] = v
    else:
        logger_py.debug('no scan processed!')
    h5f.close()
    