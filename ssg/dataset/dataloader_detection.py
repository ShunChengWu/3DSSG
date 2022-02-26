# if __name__ == '__main__' and __package__ is None:
#     from os import sys
#     sys.path.append('../../')
import os
from ssg.utils import util_data
from codeLib.utils.util import read_txt_to_list
import codeLib.utils.string_numpy as snp
from codeLib.common import normalize_imagenet, random_drop#, load_obj
import h5py
import torch.utils.data as data
from torchvision import transforms
import ssg.utils.compute_weight as compute_weight
import torch
import json
import numpy as np
import pandas
from PIL import Image
from torchvision.io import read_image
from ssg import define
from ssg.utils.util_data import raw_to_data, cvt_all_to_dict_from_h5
import codeLib.torchvision.transforms as cltransform
from torchvision.ops import roi_align
import h5py,ast

class Graph_Loader (data.Dataset):
    def __init__(self, config, mode):
        super().__init__()
        assert mode in ['train','validation','test']
        # something to do with json. if the value is not np array. the multiprocessing in pytorch seems will copy them alot and crash
        # torch.multiprocessing.set_sharing_strategy('file_system') 
        self.cfg = config
        self._device = config.DEVICE
        # self.config = config.data
        self.path = path = config.data['path']
        self.mode=mode
        multi_rel_outputs = config.data['multi_rel']
        self.for_eval=mode != 'train'
        
        '''load from config'''
        self.sample_in_runtime = config.data.sample_in_runtime
        self.sample_num_nn  = config.data.get('sample_num_nn',1)
        self.sample_num_seed = config.data.get('sample_num_seed',1)
        self.drop_edge_eval = config.data.drop_edge_eval
        self.drop_edge = config.data.drop_edge
        self.drop_img_edge = config.data.drop_img_edge
        self.drop_img_edge_eval = config.data.drop_img_edge_eval
        self.multi_rel_outputs = config.data.multi_rel
        self.max_num_edge = config.data.max_num_edge
        self.img_feature_path = config.data.img_feature_path
        self.full_edge = config.data.full_edge
        self.normalize_weight = config.data.normalize_weight
        self.use_precompute = config.data.use_precompute_img_feature
        self.img_feature_type = config.model.image_encoder.backend
        
        self.path_h5 = os.path.join(self.path,'relationships_%s.h5' % (mode))
        self.path_mv = os.path.join(self.path,'proposals.h5')
        self.pth_filtered = os.path.join(self.path,'filtered_scans_detection_%s.h5' % (self.mode))
        self.pth_node_weights = os.path.join(self.path,'node_weights.txt')
        self.pth_edge_weights = os.path.join(self.path,'edge_weights.txt')
        
        if config.data.img_size > 0:
            self.transform = transforms.Compose([
                transforms.Resize(config.data.img_size),
                transforms.ToTensor(),
                cltransform.TrivialAugmentWide(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        ''' read classes '''
        pth_classes = os.path.join(path,'classes.txt')
        pth_relationships = os.path.join(path,'relationships.txt')
        # if mode == 'train':
        
        names_classes = read_txt_to_list(pth_classes)
        names_relationships = read_txt_to_list(pth_relationships)
        
        if not multi_rel_outputs:
            if define.NAME_NONE not in names_relationships:
                names_relationships.append(define.NAME_NONE)
        elif define.NAME_NONE in names_relationships:
            names_relationships.remove(define.NAME_NONE)
        
        if not multi_rel_outputs:
            if 'none' not in names_relationships:
                names_relationships.append('none')
        
        self.relationNames = sorted(names_relationships)
        self.classNames = sorted(names_classes)
        
        self.none_idx = self.relationNames.index(define.NAME_NONE) if not multi_rel_outputs else None
        
        
        self.preprocessing()
        
        if not self.for_eval:
            w_node_cls = np.loadtxt(self.pth_node_weights)
            w_edge_cls = np.loadtxt(self.pth_edge_weights)
            self.w_node_cls = torch.from_numpy(w_node_cls).float()
            self.w_edge_cls = torch.from_numpy(w_edge_cls).float()
            
        self.open_filtered()
        self.scans = snp.pack([k for k in self.filtered_data.keys()])
        self.size = len(self.filtered_data)
        del self.filtered_data
        
    def open_mv_graph(self):
        if not hasattr(self, 'mv_data'):
            self.mv_data = h5py.File(self.path_mv,'r')
        
    def open_data(self):
        if not hasattr(self,'sg_data'):
            self.sg_data = h5py.File(self.path_h5,'r')
        
    def __len__(self):
        return self.size
    
    def open_hdf5(self, path):
        if not hasattr(self,'img_features'):
            self.img_features = h5py.File(path,'r')
    def open_filtered(self):
        self.filtered_data = h5py.File(self.pth_filtered,'r')
            
    def preprocessing(self):
        pth_node_weights=self.pth_node_weights
        pth_edge_weights = self.pth_edge_weights
        pth_filtered = self.pth_filtered
        config = self.cfg
        
        should_process = not os.path.isfile(pth_filtered)
        if not self.for_eval:
            should_process |= not os.path.isfile(pth_node_weights) or not os.path.isfile(pth_edge_weights)
        
        if should_process:
            ''' load data '''
            selected_scans = read_txt_to_list(os.path.join(self.path,'%s_scans.txt' % (self.mode)))
            self.open_mv_graph()
            self.open_data()
            c_sg_data = cvt_all_to_dict_from_h5(self.sg_data)
            
            '''check scan_ids'''
            # filter input scans with relationship data
            tmp   = set(c_sg_data.keys())
            inter = sorted(list(tmp.intersection(selected_scans)))
            # filter input scans with image data
            tmp   = set(self.mv_data.keys())
            inter = sorted(list(tmp.intersection(inter)))
            
        if not os.path.isfile(pth_filtered):
            self.open_data()
            self.open_mv_graph()
            filtered = dict()
            for scan_id in inter:
                scan_data_raw = self.sg_data[scan_id]
                scan_data = raw_to_data(scan_data_raw)
                
                object_data = scan_data['nodes']
                # relationships_data = scan_data['relationships']        
                
                ''' build mapping '''
                instance2labelName  = { int(key): node['label'] for key,node in object_data.items()  }
                
                mv_data = self.mv_data[scan_id]
                mv_nodes = mv_data['nodes']
                kfs = mv_data['kfs']
                
                '''filter'''
                mv_node_ids = [int(x) for x in mv_nodes.keys()]
                sg_node_ids = object_data.keys()                
                inter_node_ids = set(sg_node_ids).intersection(mv_node_ids)
                object_data = {nid: object_data[nid] for nid in inter_node_ids}
                
                '''select frames with more than 1 objects'''
                kf_indices=[]
                for k in kfs.keys():
                    kf = kfs[k]
                    oids = [v[0] for v in kf.attrs['seg2idx']]
                    obj_count=0
                    if len(oids)<=1:continue
                    for oid in oids:
                        oid = int(oid)
                        if oid in instance2labelName:
                            if instance2labelName[oid] in self.classNames:
                                obj_count+=1
                    if obj_count>1:
                        kf_indices.append(int(k))
                        
                if len(kf_indices) == 0:
                    continue
                
                filtered[scan_id] = kf_indices
            # s_scan = str(filtered)
            with h5py.File(pth_filtered, 'w') as h5f:
                for scan_id in filtered:
                    h5f.create_dataset(scan_id,data=np.array(filtered[scan_id]))            
            
        if not self.for_eval:
            if not os.path.isfile(pth_node_weights) or not os.path.isfile(pth_edge_weights):
                #TODO: also filter out nodes when only with points input. this gives fair comparison on points and images methods.
                filtered_sg_data = dict()
                self.open_filtered()
                filtered = self.filtered_data.keys()
                for scan_id in filtered:
                    mv_node_ids = [int(x) for x in self.mv_data[scan_id]['nodes'].keys()]
                    sg_node_ids = c_sg_data[scan_id]['nodes'].keys()                
                    inter_node_ids = set(sg_node_ids).intersection(mv_node_ids)
                    
                    filtered_sg_data[scan_id] = dict()
                    filtered_sg_data[scan_id]['nodes'] = {nid: c_sg_data[scan_id]['nodes'][nid] for nid in inter_node_ids}
                    
                    filtered_sg_data[scan_id]['relationships'] = c_sg_data[scan_id]['relationships']
                c_sg_data = filtered_sg_data
                
                if config.data.full_edge:
                    edge_mode='fully_connected'
                else:
                    edge_mode='nn'
                # edge_mode='gt'
                # print('edge_mode:',edge_mode)
                wobjs, wrels, o_obj_cls, o_rel_cls = compute_weight.compute_sgfn(self.classNames, self.relationNames, c_sg_data, selected_scans,
                                                                            normalize=config.data.normalize_weight,
                                                                            for_BCE=self.multi_rel_outputs==True,
                                                                            edge_mode=edge_mode,
                                                                            none_index=self.none_idx,
                                                                            verbose=config.VERBOSE)           
                for idx, obj_cls_name in enumerate(self.classNames):
                    if obj_cls_name in config.data.obj_ignore_list:
                        if config.VERBOSE:
                            print('ignore object ',obj_cls_name)
                        wobjs[idx]=0
                        
                wobjs = np.array(wobjs)
                wrels = np.array(wrels)
                np.savetxt(pth_node_weights,wobjs)
                np.savetxt(pth_edge_weights,wrels)
                
                # test
                w_node_cls = np.loadtxt(pth_node_weights)
                w_edge_cls = np.loadtxt(pth_edge_weights)
                # self.w_node_cls = torch.from_numpy(np.array(wobjs)).float()
                # self.w_edge_cls = torch.from_numpy(np.array(wrels)).float()
                
        if should_process:
            del self.sg_data
            del self.mv_data   
    
    def __getitem__(self,idx):
        scan_id = snp.unpack(self.scans,idx)# self.scans[idx]
        self.open_data()
        self.open_mv_graph()
        self.open_filtered()
        
        scan_data_raw = self.sg_data[scan_id]
        scan_data = raw_to_data(scan_data_raw)
        
        object_data = scan_data['nodes']
        relationships_data = scan_data['relationships']        
        
        ''' build mapping '''
        instance2labelName  = { int(key): node['label'] for key,node in object_data.items()  }
        
        mv_data = self.mv_data[scan_id]
        mv_nodes = mv_data['nodes']
        kfs = mv_data['kfs']
                
        self.filtered_data
        kf_indices = self.filtered_data[scan_id]
        
        '''build graph base on detections'''
        if self.sample_in_runtime:
            kf_indices = random_drop(kf_indices, self.drop_img_edge)    
        
        instance2mask=dict()
        bounding_boxes = list() # bounding_boxes[node_id]{kf_id: [boxes]}
        edge_indices=list()
        cat = []
        fdata = define.DATA_PATH
        rgb_filepattern = 'frame-{0:06d}.color.jpg'
        images=list()
        for mid, fid in enumerate(kf_indices):
            pth_rgb = os.path.join(fdata,scan_id,'sequence', rgb_filepattern.format(int(fid)))
            img_data = Image.open(pth_rgb)
            img_data = np.rot90(img_data,3)# Rotate image
            img_data = self.transform(img_data.copy())
            images.append(img_data)
            width,height = img_data.shape[-1],img_data.shape[-2]
            
            kf = kfs[str(fid)]              
            kf_bid2oid = {v[1]:v[0] for v in kf.attrs['seg2idx']}
            kfdata = np.asarray(kf)
            
            # box_dict = dict()
            
            # get GT class
            per_frame_indices=list()
            for idx in range(kfdata.shape[0]):
                oid = int(kf_bid2oid[idx])
                if oid not in instance2labelName: continue
                instance_labelName = instance2labelName[oid]
                if instance_labelName in self.classNames:
                    class_id = self.classNames.index(instance_labelName)    
                
                box = kfdata[idx][:4]
                box[0]/=width
                box[1]/=height
                box[2]/=width
                box[3]/=height
                box = np.concatenate(([mid], box)).tolist()
                
                om_id = len(cat)
                bounding_boxes.append(box)
                cat.append(class_id)
                
                instance2mask[oid]=om_id
                
                per_frame_indices.append(om_id)
            for om_id1 in per_frame_indices:
                for om_id2 in per_frame_indices:
                    if om_id1 != om_id2:
                        edge_indices.append([om_id1,om_id2])
        
        '''build predicate GT'''
        if self.multi_rel_outputs:
            adj_matrix_onehot = np.zeros([len(cat), len(cat), len(self.relationNames)])
        else:
            adj_matrix = np.ones([len(cat), len(cat)]) * self.none_idx#set all to none label.
            
        for r in relationships_data:
            r_src = int(r[0])
            r_tgt = int(r[1])
            r_lid = int(r[2])
            r_cls = r[3]
            if r_src not in instance2mask or r_tgt not in instance2mask: continue
            index1 = instance2mask[r_src]
            index2 = instance2mask[r_tgt]
            assert index1>=0
            assert index2>=0
            if self.sample_in_runtime:
                if [index1,index2] not in edge_indices: continue
            if r_cls not in self.relationNames:
                continue  
            r_lid = self.relationNames.index(r_cls) # remap the index of relationships in case of custom relationNames
            # assert(r_lid == self.relationNames.index(r_cls))

            if index1 >= 0 and index2 >= 0:
                if self.multi_rel_outputs:
                    adj_matrix_onehot[index1, index2, r_lid] = 1
                else:
                    adj_matrix[index1, index2] = r_lid      
        '''edge GT to tensor'''
        if self.multi_rel_outputs:
            rel_dtype = np.float32
            adj_matrix_onehot = torch.from_numpy(np.array(adj_matrix_onehot, dtype=rel_dtype))
        else:
            rel_dtype = np.int64
            adj_matrix = torch.from_numpy(np.array(adj_matrix, dtype=rel_dtype))
        if self.multi_rel_outputs:
            gt_rels = torch.zeros(len(edge_indices), len(self.relationNames),dtype = torch.float)
        else:
            gt_rels = torch.zeros(len(edge_indices),dtype = torch.long)
        for e in range(len(edge_indices)):
            edge = edge_indices[e]
            index1 = edge[0]
            index2 = edge[1]
            if self.multi_rel_outputs:
                gt_rels[e,:] = adj_matrix_onehot[index1,index2,:]
            else:
                gt_rels[e] = adj_matrix[index1,index2]
                
                
        '''to tensor'''
        images = torch.stack(images,dim=0)
        assert len(cat) == len(bounding_boxes)
        bounding_boxes = torch.from_numpy(np.array(bounding_boxes)).float()
        gt_class = torch.from_numpy(np.array(cat))
        edge_indices = torch.tensor(edge_indices,dtype=torch.long)
        
        output = dict()
        output['scan_id'] = scan_id # str
        output['gt_rel'] = gt_rels  # tensor
        output['gt_cls'] = gt_class # tensor
        output['images'] = images# tensor
        output['node_edges'] = edge_indices # tensor
        output['instance2mask'] = instance2mask #dict
        output['image_boxes'] = bounding_boxes #list
        del self.filtered_data
        del self.sg_data
        del self.mv_data   
        return output
    
    
if __name__ == '__main__':
    import ssg
    import codeLib
    from ssg.data.collate import graph_collate
    import ssg.config as config
    path = '../../configs/default.yaml'
    path = './configs/exp2_all_test_overfit.yaml'
    path = './configs/exp2_all_gnn.yaml'
    path= './configs/exp2_all_basic_bf200.yaml'
    
    path='./experiments/config_IMP_full_l20_0.yaml'
    
    cfg= codeLib.Config(path)
    cfg.DEVICE='cpu'
    # config.model.node_encoder.backend='vgg16'
    codeLib.utils.util.set_random_seed(cfg.SEED)
    dataset = config.get_dataset(cfg,'train')
    
    # dataset.__getitem__(0)
    from tqdm import tqdm
    # for data in tqdm(iter(dataset)):
    #     continue
    
    # train_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=config['training']['batch'], num_workers=config['training']['data_workers'], shuffle=False,
    #     pin_memory=False,
    #     collate_fn=graph_collate,
    # )
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0, shuffle=False,
        pin_memory=True,
        collate_fn=graph_collate,
    )
    for epoch in tqdm(range(cfg.training.max_epoch)):
        for data in tqdm(train_loader):
            continue
        break