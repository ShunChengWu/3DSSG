from genericpath import isfile
import json
import os
import numpy as np

import torch
from tqdm import tqdm
import ssg
from ssg import define
from ssg.checkpoints import CheckpointIO
import ssg.config as config
import h5py
from PIL import Image
from codeLib.common import normalize_imagenet
import torchvision,logging
logger_py = logging.getLogger(__name__)

def main():
    cfg = ssg.Parse()
    out_dir = os.path.join(cfg['training']['out_dir'], cfg.name)
    # model = config.get_model(cfg,num_obj_cls=20, num_rel_cls=8)
    # checkpoint_io = CheckpointIO(out_dir, model=model)
    # load_dict = checkpoint_io.load('model_best.pt', device=cfg.DEVICE)
    
    '''load from file'''
    with open(os.path.join(cfg.data.path,'args.json')) as f:
        tmp = json.load(f)
        label_type = tmp['label_type']
        segment_type = tmp['segment_type']
    image_feature_folder_name =define.NAME_IMAGE_FEAUTRE_FORMAT.format(segment_type,label_type)
    path_img_feature = os.path.join(cfg.data.path_image_feature,image_feature_folder_name+'.h5')
    image_feature = h5py.File(path_img_feature,'r')
    
    
    '''compute in runtime'''
    cfg.data.use_precompute_img_feature = False # force to set to false to enable backend precompute
    img_encoder = ssg.models.node_encoder_list['roi_extractor'](cfg,cfg.model.image_encoder.backend,cfg.DEVICE)
    img_encoder = img_encoder.eval()
    for param in img_encoder.parameters(): param.requires_grad = False
    img_encoder = img_encoder.to(cfg.DEVICE)
    
    '''transform'''
    if cfg.data.img_size > 0:
        transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(cfg.data.img_size),
                ])
    else:
        transform = torchvision.transforms.Compose([
                ])
        
    '''load'''
    pth_filtered = os.path.join(cfg.data.path,'filtered_scans_detection_%s.h5' % (cfg.MODE))
    filtered_data = h5py.File(pth_filtered,'r')
    
    logger_py.info('start processing')
    for scan_id, kf_indices in tqdm(filtered_data.items()):
        logger_py.info('process scan {}'.format(scan_id))
        
        fids = [idx for idx in kf_indices][:5]
        
        
        logger_py.info('get image list')
        images=list()
        for fid in fids:
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
            img_features = torch.cat([ img_encoder.preprocess(p_split).cpu() for p_split in torch.split(images,int(8), dim=0) ], dim=0)
        
        logger_py.info('save')
        with h5py.File('tmp_compare_precompute_feature_batch.h5','w') as h5f:
            for idx,fid in enumerate(fids):
                h5f.create_dataset(str(fid),data=img_features[idx].numpy())
                
                
        '''check'''
        if os.path.exists('tmp_compare_precompute_feature_batch_2.h5'):
            for idx,fid in enumerate(fids):
                with h5py.File('tmp_compare_precompute_feature_batch_2.h5','r') as h5f:
                    img_data_load = h5f[str(fid)]
                    img_data_load = np.asarray(img_data_load).copy()
                    load_2 = torch.from_numpy(img_data_load).to(cfg.DEVICE)
                    
                with h5py.File('tmp_compare_precompute_feature_batch.h5','r') as h5f:
                    img_data_load = h5f[str(fid)]
                    img_data_load = np.asarray(img_data_load).copy()
                    load_1 = torch.from_numpy(img_data_load).to(cfg.DEVICE)
                
                print(torch.equal(load_1,load_2))
                
        break
    
    return
    for scan_id, image_features in image_feature.items():
        fids = [idx for idx in image_features.keys()][:5]
        '''compute here and save'''
        images=list()
        for fid in fids:
            pth_rgb = os.path.join(define.DATA_PATH,scan_id,'sequence', define.RGB_NAME_FORMAT.format(int(fid)))
            img_data = Image.open(pth_rgb)
            img_data = np.rot90(img_data,3)# Rotate image
            img_data = torch.as_tensor(img_data.copy()).permute(2,0,1)
            img_data = transform(img_data)
            img_data= normalize_imagenet(img_data.float()/255.0)
            images.append(img_data)
        images = torch.stack(images).to(cfg.DEVICE)
        with torch.no_grad():
            img_features = torch.cat([ img_encoder.preprocess(p_split).cpu() for p_split in torch.split(images,int(8), dim=0) ], dim=0)
            
        with h5py.File('tmp_compare_precompute_feature_batch.h5','w') as h5f:
            for idx,fid in enumerate(fids):
                h5f.create_dataset(str(fid),data=img_features[idx].numpy())
                
        for fid in fids:
            with h5py.File('tmp_compare_precompute_feature_batch_2.h5','r') as h5f:
                img_data_load = h5f[str(fid)]
                img_data_load = np.asarray(img_data_load).copy()
                load_1 = torch.from_numpy(img_data_load).to(cfg.DEVICE)
            
            with h5py.File('tmp_compare_precompute_feature_batch.h5','r') as h5f:
                img_data_load = h5f[str(fid)]
                img_data_load = np.asarray(img_data_load).copy()
                load_2 = torch.from_numpy(img_data_load).to(cfg.DEVICE)
                
            print(torch.equal(load_1,load_2))
        break
        
        # for fid in fids:
        #     '''read precomputed'''
        #     img_data = image_feature[scan_id][str(fid)]
        #     img_data = np.asarray(img_data).copy()
        #     img_feature_load = torch.from_numpy(img_data).to(cfg.DEVICE).unsqueeze(0)
            
        #     '''compute in runtime'''
        #     pth_rgb = os.path.join(define.DATA_PATH,scan_id,'sequence', define.RGB_NAME_FORMAT.format(int(fid)))
        #     img_data = Image.open(pth_rgb)
        #     img_data = np.rot90(img_data,3)# Rotate image
        #     img_data = torch.as_tensor(img_data.copy()).permute(2,0,1)
        #     img_data = transform(img_data)
        #     img_data= normalize_imagenet(img_data.float()/255.0)
        #     img_data = img_data.unsqueeze(0).to(cfg.DEVICE)
            
        #     img_feature_compute = torch.cat([ img_encoder.preprocess(p_split) for p_split in torch.split(img_data,int(8), dim=0) ], dim=0)
            
        #     # img_feature_compute = img_encoder.preprocess(img_data)
            
        #     '''test IO computed batch'''
        #     with h5py.File('tmp_compare_precompute_feature_batch.h5','r') as h5f:
        #         img_data_load = h5f[str(fid)]
        #         img_data_load = np.asarray(img_data_load).copy()
        #         img_feature_load_batch = torch.from_numpy(img_data_load).to(cfg.DEVICE).unsqueeze(0)
            
        #     '''test IO'''
        #     with h5py.File('tmp_compare_precompute_feature.h5','w') as h5f:
        #         h5f.create_dataset(str(fid),data=img_feature_compute[0].cpu().numpy())
        #     with h5py.File('tmp_compare_precompute_feature.h5','r') as h5f:
        #         img_data_load = h5f[str(fid)]
        #         img_data_load = np.asarray(img_data_load).copy()
        #         img_feature_load2 = torch.from_numpy(img_data_load).to(cfg.DEVICE).unsqueeze(0)
            
        #     print('diff load',torch.equal(img_feature_compute,img_feature_load))
        #     print('diff batch',torch.equal(img_feature_compute,img_feature_load_batch))
        #     print('diff single', torch.equal(img_feature_compute,img_feature_load2))
        pass
    
    
if __name__ == '__main__':
    # logger_py.setLevel('DEBUG')
    # logger_py.debug('hello0')
    main()