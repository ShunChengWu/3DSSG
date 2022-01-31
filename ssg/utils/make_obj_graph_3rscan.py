import json, glob, csv, sys, argparse, os, sys, codeLib, pandas, h5py, logging
from codeLib.utils.classification.labels import NYU40_Label_Names, SCANNET20_Label_Names
import numpy as np
from tqdm import tqdm
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

logging.basicConfig()
logger_py = logging.getLogger(__name__)
logger_py.setLevel('INFO')

structure_labels = ['wall','floor','ceiling']

class_map = {
    'scannet20': 'nyu40class',
    'nyu': 'nyuClass',
    'nyu40': 'nyu40class',
    'eigen13': 'eigen13class',
    'modelnet40': 'ModelNet40',
    'modelnet10': 'ModelNet10',
    'shapenet55': 'ShapeNetCore55',
    'mpnyu40': 'mpcat40'
}
width=540
height=960

DEBUG=True
DEBUG=False

def Parse():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-f','--scenelist',default='/home/sc/research/PersistentSLAM/python/2DTSG/files/scannetv2_trainval.txt',help='scene list (txt)')
    #
    parser.add_argument('-d','--gt2d_dir',default='/media/sc/SSD1TB/dataset/3RScan/2dgt', 
                        help='directory containing the .2dgt i.e. the gt 2d detections, generated with the script')
    parser.add_argument('-o','--outdir',default='/home/sc/research/PersistentSLAM/python/2DTSG/data/mv_scannet_test/', help='output dir',required=True)
    parser.add_argument('-m','--max_occ',default=0.5,help='The threshold for the visibility of an object. If below this value, discard')
    parser.add_argument('--min_object', help='if less thant min_obj objects, ignore image', default=1)
    parser.add_argument('-l','--label_type',default='nyu40', choices=['nyu40','eigen13','rio27', 'rio7','3rscan','3rscan160'], 
                        help='target label type.')
    # parser.add_argument('-lf','--label_file',default='/media/sc/space1/dataset/scannet/scannetv2-labels.combined.tsv', 
    #                     help='file path to scannetv2-labels.combined.tsv')
    parser.add_argument('--skip_structure',default=0,help='should ignore sturcture labels or not')
    parser.add_argument('--overwrite', type=int, default=0, help='overwrite existing file.')
    return parser

def get_bbx_wo_flatcheck(scan_id, fn,max_oc, mapping, min_size:list=[240,240]):
  obj_by_img={}  
  data = pandas.read_csv(fn, delimiter= ' ')      
  # imgs = f[scan_id]
  for i in data.index:
      fname = data['frame_id'][i]
      oid = data['object_id'][i]
      olabel = data['label'][i]
      oc = data['occlution_ratio'][i]
      x1 = data['x_min'][i]
      y1 = data['y_min'][i]
      x2 = data['x_max'][i]
      y2 = data['y_max'][i]
      
      olabel=olabel[1:].replace("\'","") # remove utf8 charesters
      olabel=olabel.replace('_',' ')
      
      if float(x1)<1 or float(y1)<1 or width < float(x2) or height < float(y2):
          if DEBUG: print('skip',fname,olabel,': on edge')
          continue
      
      oc=float(oc)
      oc = round(oc, 3)
      if oc<max_oc: # if occlusion rate is over the maximum authorised, then skip
          if DEBUG: print('skip',fname,olabel,'occluded',oc)
          continue
      
      size = [x2-x1,y2-y1]
      
      if size[0] < min_size[0] or size[1] < min_size[1]:
          if DEBUG: print('skip',fname,olabel,'too small', size)
          continue
      # print(olabel)
      
      if olabel not in mapping:
           if DEBUG: print(olabel,'not in ', mapping.keys())
           continue
           raise RuntimeError('all labels should be included in the mapping dict',mapping.keys(), 'query',olabel)
      olabel = mapping[olabel]
      
      if args.skip_structure>0:
        if olabel in structure_labels:
            if DEBUG: print('skip',fname,olabel,' structure label')
            continue
      
      # if args.label_type == 'scannet20':
      if olabel not in util_label.NYU40_Label_Names:
          if DEBUG: print('skip',fname,olabel,'not in label names')
          continue
    
      if fname not in obj_by_img:
          obj_by_img[fname]=[fname,[]]
      obj_by_img[fname][1].append([oid,olabel,oc,float(x1),float(y1),float(x2),float(y2)])      
  return obj_by_img

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def print_selection(scan_id:str, node2kfs:dict, objects:dict, kfs:dict):
    '''
    show paired label and rgb images with bounding boxes, labels and occlusion level
    '''
    import os,h5py,imageio.ssg2d,zipfile,torch
    from PIL import Image
    from torchvision.ops import roi_align
    from torchvision import transforms
    from codeLib.torch.visualization import show_tensor_images
    from torchvision.utils import draw_bounding_boxes
    from ssg2d.utils.scannet.makebb_img import LabelImage
    toTensor = transforms.ToTensor()
    resize = transforms.Resize([256,256])
    pth_img = '/media/sc/SSD1TB2/dataset/scannet/images.h5'
    ffont = '/home/sc/research/PersistentSLAM/python/2DTSG/files/Raleway-Medium.ttf'
    f = h5py.File(pth_img, 'r')
    
    SCANNET_DIR = '/media/sc/space1/dataset/scannet/'
    fdata = os.path.join(SCANNET_DIR,'scans') # '/media/sc/space1/dataset/scannet/scans/'
    # pth_scannet_label = os.path.join(SCANNET_DIR, 'scannetv2-labels.combined.tsv')
    label_filepattern = '_2d-label-filt.zip'
    insta_filepattern = '_2d-instance-filt.zip'
    pth_instance_zip = os.path.join(fdata,scan_id,scan_id+insta_filepattern)
    pth_label_zip = os.path.join(fdata,scan_id,scan_id+label_filepattern)
    pth_instance_zip = os.path.join(fdata,scan_id,scan_id+insta_filepattern)
    label_prefix='label-filt/'
    insta_prefix='instance-filt/'
    
    imgs = f[scan_id]
    with zipfile.ZipFile(pth_label_zip, 'r') as arc_label, zipfile.ZipFile(pth_instance_zip, 'r') as arc_inst:
        for oid, obj in objects.items():
            fids = node2kfs[int(oid)]
            olabel = obj['label']
            
            # get image
            img_boxes = list()
            for fid in fids:
                limg_data = arc_label.read(label_prefix+str(fid)+'.png')
                iimg_data = arc_inst.read(insta_prefix+str(fid)+'.png')
                limg = LabelImage(fid, iimg_data, limg_data)
                bbox_img = limg.get_bbox_image()
                
                kf = kfs[str(fid)]
                bfid = imgs['indices'][fid] # convert frame idx to the buffer idx 
                img_data = imgs['rgb'][bfid]
                img = imageio.imread(img_data)
                img = Image.fromarray(img)
                timg = toTensor(img).unsqueeze(0)
                box = kf['bboxes'][str(oid)]
                oc  = kf['occlution'][str(oid)]
                w = int(box[2]-box[0])
                h = int(box[3]-box[1])
                box = torch.as_tensor([x1,y1,x2,y2]).float().view(1,-1)
                # extract RGB image
                timg = torch.clamp(timg*255,0,255)
                region = roi_align(timg,[box], [h,w])
                region = resize(region).squeeze(0).byte()
                # extract label image
                region_l = roi_align(bbox_img.unsqueeze(0).float(),[box], [h,w])
                region_l = resize(region_l).squeeze(0).byte()
                
                
                
                box = torch.as_tensor([0,0,255,255]).float().view(1,-1)
                soc = '{0:}_{1:.3f}'.format(fid,oc)
                result = draw_bounding_boxes(region, box, 
                                         labels=[soc],
                                         width=5,
                                         font=ffont,
                                         font_size=50)
                result = torch.stack([region_l,result])
                show_tensor_images(result, title=olabel)
                img_boxes.append(result)
                # img_boxes.append( result )
            img_boxes = torch.stack(img_boxes)
            show_tensor_images(img_boxes, title=olabel)

if __name__ == '__main__':
    args = Parse().parse_args()
    print(args)
    outdir=args.outdir
    max_oc=float(args.max_occ) # maximum occlusion rate authorised
    min_obj=float(args.min_object)
    gt2d_dir = args.gt2d_dir
    # scannet_w = 1296
    # scannet_h= 968

    '''create mapping'''
    # create name2idx mapping
    label_names, label_name_mapping, label_id_mapping = util_label.getLabelMapping(args.label_type,define.LABEL_MAPPING_FILE)
    util_label.NYU40_Label_Names
    # NYU40_Name2Idx_map = {name:idx for idx, name in enumerate(NYU40_Label_Names)}
    # NYU40_Idx2Name_map = {idx:name for idx, name in enumerate(NYU40_Label_Names)}
    
    # ''' 0Get all label names'''
    # if args.label_type.find('scannet') >=0:
    #     labelNames = sorted(SCANNET20_Label_Names)
    # else:
    #     labelNames = sorted(np.unique(values).tolist())
        
    '''create output file'''
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    #save configs
    with open(os.path.join(outdir,'args.txt'), 'w') as f:
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
        # logger_py.info(scene)
        pbar.set_description('processing {}'.format(scan_id))
        
        # load semseg
        pth_semseg = os.path.join(fdata,scan_id,define.SEMSEG_FILE_NAME)
        mapping = load_semseg(pth_semseg,label_name_mapping)
        mapping[0] = 'none'
        
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
        obj_by_img=get_bbx_wo_flatcheck(scan_id, gt2d_file, max_oc,label_name_mapping)
        
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
                kf['bboxes'][str(oid)] = [x1,y1,x2,y2]
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
        with open(os.path.join(outdir,'classes.txt'), 'w') as f:
            for cls in util_label.NYU40_Label_Names:
                f.write('{}\n'.format(cls))
    else:
        print('no scan processed')
    h5f.close()
    