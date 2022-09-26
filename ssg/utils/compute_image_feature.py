# -*- coding: utf-8 -*-
'''
Load graph file. Compute image feature for all the keyframes and save


'''
import os,logging, argparse
import ssg
from ssg import define
import torch,json
from PIL import Image
from ssg.models import encoder
from torchvision import transforms
# import open3d as o3d
# import numpy as np
# from pathlib import Path
from tqdm import tqdm
import codeLib
from codeLib.common import normalize_imagenet, create_folder
from ssg.utils import util_data
import ssg.define as define
import h5py,glob,pathlib
import numpy as np
logger_py = logging.getLogger(__name__)
debug = True
debug = False

def Parser():
    parser = argparse.ArgumentParser(
        description='compute image feature.', 
        parents=[ssg.default_parser()],
        add_help=False)
    parser.add_argument('-n','--folder_name',type=str,default='image_features', help='folder name which will be created at outdir',required=True)
    parser.add_argument('-o','--outdir',default='/media/sc/SSD1TB/dataset/3RScan/', help='where to store all image features.',required=True)
    parser.add_argument('--overwrite', type=int, default=0, help='overwrite existing file.')
    return parser

if __name__ == '__main__':
    '''load config'''
    args = Parser().parse_args()
    cfg = ssg.load_config(args)
    
    '''Logging'''
    logger_py.info('args:\n{}'.format(args.__dict__))
    logger_py.info('use backend {}'.format(cfg.model.image_encoder.backend))
    
    
    '''read label type '''
    # fdata = define.DATA_PATH
    # rgb_filepattern =  define.RGB_NAME_FORMAT
    foldername = args.folder_name
    pattern = '{}/*.h5'.format(foldername)
    pth_out = os.path.join(args.outdir,foldername)# '/media/sc/SSD1TB/dataset/3RScan/roi_images/'
    pth_link = os.path.join(args.outdir,foldername+'.h5')
    
    '''create save'''
    pathlib.Path(pth_out).mkdir(parents=True,exist_ok=True)
    
    '''load'''
    pth_filtered = os.path.join(cfg.data.path,'filtered_scans_detection_%s.h5' % (args.mode))
    filtered_data = h5py.File(pth_filtered,'r')
    
    '''image encoder'''
    logger_py.info('create image encoder')
    # feature_type = cfg.model.image_encoder.backend
    cfg.data.use_precompute_img_feature = False # force to set to false to enable backend precompute
    img_encoder = ssg.models.node_encoder_list['roi_extractor'](cfg,cfg.model.image_encoder.backend,cfg.DEVICE)
    img_encoder = img_encoder.eval()
    for param in img_encoder.parameters(): param.requires_grad = False
    img_encoder = img_encoder.to(cfg.DEVICE)
    
    '''transform'''
    if cfg.data.img_size > 0:
        transform = transforms.Compose([
                    transforms.Resize(cfg.data.img_size),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    else:
        transform = transforms.Compose([
                    # transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        
    logger_py.info('start processing')
    for scan_id, kf_indices in tqdm(filtered_data.items()):
        logger_py.info('process scan {}'.format(scan_id))
        '''check overwrite'''
        filepath = os.path.join(pth_out,scan_id+'.h5')
        logger_py.info('check file at {}'.format(filepath))
        if os.path.exists(filepath):
            if args.overwrite==0:
                logger_py.info('exist, skip')
                continue
            else:
                logger_py.info('exist skip')
                os.remove(filepath)
                
        kf_indices = [idx for idx in kf_indices]
                
        logger_py.info('create file')
        try:
            logger_py.info('get image list')
            images=list()
            for fid in kf_indices:
                pth_rgb = os.path.join(define.DATA_PATH,scan_id,'sequence', define.RGB_NAME_FORMAT.format(int(fid)))
                img_data = Image.open(pth_rgb)
                img_data = np.rot90(img_data,3)# Rotate image
                img_data = torch.as_tensor(img_data.copy()).permute(2,0,1)
                img_data = transform(img_data)
                img_data= normalize_imagenet(img_data.float()/255.0)
                images.append(img_data)
            images = torch.stack(images).to(cfg.DEVICE)
            
            logger_py.info('compute feature')
            with torch.no_grad():
                img_features = torch.cat([ img_encoder.preprocess(p_split).cpu()  for p_split in torch.split(images,int(8), dim=0) ], dim=0)
            
            logger_py.info('save')
            with h5py.File(filepath,'w') as h5f:
                for idx,fid in enumerate(kf_indices):
                    # if feature_type in h5g: del h5g[feature_type]
                    h5f.create_dataset(str(fid),data=img_features[idx].numpy())    
                logger_py.info('close')
                
            '''check '''
            with h5py.File(filepath,'r') as h5f:
                for idx,fid in enumerate(kf_indices):
                    img_data = np.asarray(h5f[str(fid)]).copy()
                    img_data = torch.from_numpy(img_data)
                    assert torch.equal(img_data,img_features[idx])
        except:
            os.remove(filepath)
            logger_py.error('error occur. delete unfinished file at {}'.format(filepath))
            raise RuntimeError()
        # break
            
    '''create a link h5 db'''
    if os.path.exists(pth_link):
        os.remove(pth_link) # always remove. in case more scenes are generated
    logger_py.info('create a link h5 library at {}'.format(pth_link))
    fbase = args.outdir
    if fbase[-1] != '/': fbase+='/'
    with h5py.File(pth_link, 'w') as h5f:
        for path in glob.glob(os.path.join(fbase,pattern)):
            # name = path.split('/')[-1]
            name = path[len(fbase):]
            scan_id = path.split('/')[-1].split('.')[0]
            h5f[scan_id] = h5py.ExternalLink(name, './')
    logger_py.info('done')