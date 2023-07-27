# -*- coding: utf-8 -*-
'''
Load graph file. Compute image feature for all the keyframes and save


'''
import os,logging, argparse
import ssg
from ssg import define
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from codeLib.common import normalize_imagenet
from ssg.utils.util_data import raw_to_data
import ssg.define as define
import h5py,pathlib
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
    # parser.add_argument('-o','--outdir',default='/media/sc/SSD1TB/dataset/3RScan/', help='where to store all image features.',required=True)
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing file.')
    return parser

if __name__ == '__main__':
    '''load config'''
    args = Parser().parse_args()
    cfg = ssg.load_config(args)
    logger_py.info(cfg)
    
    '''Logging'''
    logger_py.info('args:\n{}'.format(args.__dict__))
    logger_py.info('use backend {}'.format(cfg.model.image_encoder.backend))
    
    # if args.mode == 'eval':
    #     args.mode = 'test' #TODO: fix data generation to make this consistent
    
    '''read label type '''
    # fdata = define.DATA_PATH
    # rgb_filepattern =  define.RGB_NAME_FORMAT
    
    '''load'''
    # pth_filtered = os.path.join(cfg.data.path,'filtered_scans_detection_%s.h5' % (args.mode))
    pth_filtered = os.path.join(cfg.data.path,'filtered_scans_detection.h5')
    filtered_data = h5py.File(pth_filtered,'r')
    
    '''image encoder'''
    logger_py.info('create image encoder')
    feature_type = cfg.model.image_encoder.backend
    cfg.data.use_precompute_img_feature = False # force to set to false to enable backend precompute
    img_encoder = ssg.models.node_encoder_list['roi_extractor'](cfg,cfg.model.image_encoder.backend,cfg.DEVICE)
    img_encoder = img_encoder.eval()
    for param in img_encoder.parameters(): param.requires_grad = False
    img_encoder = img_encoder.to(cfg.DEVICE)
    
    # try to get batch size
    try:
        batch_size = cfg.model.image_encoder.img_batch_size
    except:
        batch_size = 4
    
    
    '''create folder'''
    foldername = args.folder_name
    # pattern = '{}/*.h5'.format(foldername)
    pth_link = os.path.join(args.out_dir,foldername+'.h5')
    pth_out = os.path.join(args.out_dir,foldername,feature_type)# '/media/sc/SSD1TB/dataset/3RScan/roi_images/'
    pathlib.Path(pth_out).mkdir(parents=True,exist_ok=True)

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
    for scan_id, raw in tqdm(filtered_data.items()):
        logger_py.info('process scan {}'.format(scan_id))
        kf_indices = raw_to_data(raw)[define.NAME_FILTERED_KF_INDICES]
        
        '''check overwrite'''
        filepath = os.path.join(pth_out,scan_id+'.h5')
        logger_py.info('check file at {}'.format(filepath))
        
        # also stroe a list of kf indices to be processed
        filtered_kf_indices = list()
        if os.path.exists(filepath):
            # check kf ids
            should_process=False
            
            # try to open
            try:
                with h5py.File(filepath,'r') as h5f:
                    for kfId in kf_indices:
                        if str(kfId) not in h5f:
                            should_process=True
                            # logger_py.info('missing node {}'.format(kfId))
                            filtered_kf_indices.append(kfId)
            except:
                # delete that file if failed to open it
                os.remove(filepath)

            # if all exisit and not overwriting existing feature. skip.
            if not should_process and not args.overwrite:
                logger_py.info('exist, skip')
                continue
        else:
            filtered_kf_indices = kf_indices

        logger_py.info('create file')
        try:
            torch.cuda.empty_cache()
            logger_py.info('get image list')
            '''batch process. need a lot of RAM and VRAM'''
            # images=list()
            # for fid in kf_indices:
            #     pth_rgb = os.path.join(cfg.data.path_3rscan,scan_id,'sequence', define.RGB_NAME_FORMAT.format(int(fid)))
            #     img_data = Image.open(pth_rgb)
            #     img_data = np.rot90(img_data,3)# Rotate image
            #     img_data = torch.as_tensor(img_data.copy()).permute(2,0,1)
            #     img_data = transform(img_data)
            #     img_data= normalize_imagenet(img_data.float()/255.0)
            #     images.append(img_data)
            # images = torch.stack(images)
            
            # logger_py.info('compute feature')
            # with torch.no_grad():
            #     img_features=[]
            #     for p_split in torch.split(images,int(8), dim=0):
            #         torch.cuda.empty_cache()
            #         img_features.append(img_encoder.preprocess(p_split.to(cfg.DEVICE)).cpu())
            #     img_features = torch.cat(img_features,dim=0)    
                # img_features = torch.cat([ img_encoder.preprocess(p_split.to(cfg.DEVICE)).cpu()  for p_split in torch.split(images,int(4), dim=0) ], dim=0)
                # for idx,fid in enumerate(kf_indices):
                #     # if feature_type in h5g: del h5g[feature_type]
                #     h5f.create_dataset(str(fid),data=img_features[idx].numpy())    
            '''individual process'''
            logger_py.info('save')
            with h5py.File(filepath,'a') as h5f:
                for _, fid in enumerate(filtered_kf_indices):
                    pth_rgb = os.path.join(cfg.data.path_3rscan_data,scan_id,'sequence', define.RGB_NAME_FORMAT.format(int(fid)))
                    img_data = Image.open(pth_rgb)
                    img_data = np.rot90(img_data,3)# Rotate image
                    img_data = torch.as_tensor(img_data.copy()).permute(2,0,1)
                    img_data = transform(img_data)
                    img_data= normalize_imagenet(img_data.float()/255.0).unsqueeze(0)
                    with torch.no_grad():
                        img_feature = img_encoder.preprocess(img_data.to(cfg.DEVICE)).cpu()[0].numpy()

                    h5f.create_dataset(str(fid),data=img_feature)    
                
            '''process every batch'''
            # logger_py.info('save batch')
            # # generate indices
            # filtered_kf_indices = torch.LongTensor(filtered_kf_indices)

            # for indices in torch.split(filtered_kf_indices, batch_size):
            #     images=list()
            #     for fid in indices:
            #         pth_rgb = os.path.join(cfg.data.path_3rscan,scan_id,'sequence', define.RGB_NAME_FORMAT.format(int(fid)))
            #         img_data = Image.open(pth_rgb)
            #         img_data = np.rot90(img_data,3)# Rotate image
            #         img_data = torch.as_tensor(img_data.copy()).permute(2,0,1)
            #         img_data = transform(img_data)
            #         images.append(img_data)
            #     images = torch.stack(images)
            #     images = normalize_imagenet(images.to(cfg.DEVICE).float()/255.0)
                
            #     img_features = img_encoder.preprocess(images).cpu()
                
            #     '''write'''
            #     with h5py.File(filepath,'a') as h5f:
            #         for idx, fid in enumerate(indices):
            #             fid = str(fid)
            #             if fid in h5f:
            #                 del h5f[fid]
            #             h5f.create_dataset(fid,data=img_features[idx].numpy())
                        
            #     '''check'''
            #     with h5py.File(filepath,'r') as h5f:
            #         for idx,fid in enumerate(indices):
            #             img_data = np.asarray(h5f[str(fid)]).copy()
            #             img_data = torch.from_numpy(img_data)
            #             assert torch.equal(img_data,img_features[idx])

            logger_py.info('close')
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            logger_py.error('error occur. delete unfinished file at {}'.format(filepath))
            raise RuntimeError(e)
    #     break
            
    '''create a link h5 db'''
    if os.path.exists(pth_link):
        os.remove(pth_link) # always remove. in case more scenes are generated
    logger_py.info('create a link h5 library at {}'.format(pth_link))
    fbase = args.out_dir
    if fbase[-1] != '/': fbase+='/'
    
    
    
    base = os.path.join(fbase,foldername)
    with h5py.File(pth_link, 'w') as h5f:
        for root, subdirs, files in os.walk(base):
            print(root, subdirs,files)
            if root == base: continue
            feature_type = root[len(os.path.join(fbase,foldername)):]
            feature_type = feature_type.replace('/','')
            h5g = h5f.create_group(feature_type)
            for filename in files:
                name = os.path.join(root,filename)[len(fbase):]
                scan_id = filename.split('.')[0]
                h5g[scan_id] = h5py.ExternalLink(name, './')
    logger_py.info('done')