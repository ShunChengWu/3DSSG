# if __name__ == '__main__' and __package__ is None:
#     from os import sys
#     sys.path.append('../../')
import os
from collections import defaultdict
import pathlib
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
from codeLib.torch.visualization import show_tensor_images
from codeLib.common import run

DRAW_BBOX_IMAGE=True
DRAW_BBOX_IMAGE=False

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
        # self.use_precompute = config.data.use_precompute_img_feature
        self.img_feature_type = config.model.image_encoder.backend
        # self.use_filtered_node_list = config.data.use_filtered_node_list
        
        self.path_h5 = os.path.join(self.path,'relationships_%s.h5' % (mode))
        self.path_mv = os.path.join(self.path,'proposals.h5')
        self.pth_filtered = os.path.join(self.path,'filtered_scans_detection_%s.h5' % (self.mode))
        self.pth_node_weights = os.path.join(self.path,'node_weights.txt')
        self.pth_edge_weights = os.path.join(self.path,'edge_weights.txt')
        self.path_img_feature = os.path.join(self.path,config.data.path_image_feature)
            
        # if config.use_filtered_node_list:
        #     self.path_filtered_nodes_list = os.path.join(self.path,'filtered_node_list.h5')
        #     pass
        
        if config.data.img_size > 0:
            if not self.for_eval:
                self.transform = transforms.Compose([
                    transforms.Resize(config.data.img_size),
                    cltransform.TrivialAugmentWide(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(config.data.img_size),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            if not self.for_eval:
                self.transform = transforms.Compose([
                    cltransform.TrivialAugmentWide(),
                    # transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    # transforms.ToTensor(),
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
        
        '''check file exist'''
        if self.for_eval: # train mode can't use precmopute feature. need to do img. aug.
            if not os.path.exists(self.path_img_feature):
                # Try to generate
                os.environ['MKL_THREADING_LAYER'] = 'GNU'
                # os.environ['PYTHONPATH'] = config.PYTHONPATH
                # subprocess.call(["export PYTHONPATH={}".format(PYTHONPATH)], shell=True) 
                bashCommand=[
                    'python','ssg/utils/compute_image_feature.py',
                    "--config",config.config_path,
                    "-o",pathlib.Path(config.data.path_image_feature).parent.absolute(),
                    "--mode",mode,
                ]
                run(bashCommand)
                if not os.path.exists(self.path_img_feature):
                    raise RuntimeError('use precompute image feature is true but file not found.')
        
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
            
    def open_filtered_node_list(self):
        if not hasattr(self,'filtered_node_list'):
            self.filtered_node_list = h5py.File(self.path_filtered_nodes_list,'r')
        
    def __len__(self):
        return self.size
    
    def open_filtered(self):
        self.filtered_data = h5py.File(self.pth_filtered,'r')
        
    def open_image_feature(self):
        if not hasattr(self,'image_feature'):
            self.image_feature = h5py.File(self.path_img_feature,'r')
            
    def preprocessing(self):
        pth_node_weights=self.pth_node_weights
        pth_edge_weights = self.pth_edge_weights
        pth_filtered = self.pth_filtered
        config = self.cfg
        
        should_process = not os.path.isfile(pth_filtered)
        if not self.for_eval:
            should_process |= not os.path.isfile(pth_node_weights) or not os.path.isfile(pth_edge_weights)
        
        if should_process:
            print('generating filtered data...')
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
                scan_data = c_sg_data[scan_id]
                
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
                
                
                kf_indices=[]
                '''select frames with at least 1 objects'''
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
                    if obj_count>0:
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
        if self.for_eval:
            self.open_image_feature()
        # if self.use_filtered_node_list:
        #     self.open_filtered_node_list()
        
        scan_data_raw = self.sg_data[scan_id]
        scan_data = raw_to_data(scan_data_raw)
        
        object_data = scan_data['nodes']
        relationships_data = scan_data['relationships']        
        
        ''' build mapping '''
        instance2labelName  = { int(key): node['label'] for key,node in object_data.items()  }
        
        mv_data = self.mv_data[scan_id]
        # mv_nodes = mv_data['nodes']
        kfs = mv_data['kfs']
                
        kf_indices = self.filtered_data[scan_id]
        
        '''build graph base on detections'''
        # if self.sample_in_runtime:
        #     kf_indices = random_drop(kf_indices, self.drop_img_edge) 
        if not self.for_eval:
            kf_indices = random_drop(kf_indices, self.drop_img_edge, replace=True)
        if self.for_eval :
            kf_indices = random_drop(kf_indices, self.drop_img_edge_eval)
        
        
        per_frame_info_dict=defaultdict(dict)
        idx2iid=dict()
        bounding_boxes = list() # bounding_boxes[node_id]{kf_id: [boxes]}
        edge_indices=list()
        cat = []
        fdata = define.DATA_PATH
        rgb_filepattern = 'frame-{0:06d}.color.jpg'
        images=list()
        descriptor_generator = util_data.Node_Descriptor_24(with_bbox=self.cfg.data.img_desc_6_pts)
        node_descriptor_for_image = list()
        for mid, fid in enumerate(kf_indices):
            if self.for_eval:
                img_data = self.image_feature[scan_id][str(fid)]
                img_data = np.asarray(img_data).copy()
                img_data = torch.from_numpy(img_data)
                # images = [torch.from_numpy(img) for img in images]
            else:
                pth_rgb = os.path.join(fdata,scan_id,'sequence', rgb_filepattern.format(int(fid)))
                img_data = Image.open(pth_rgb)
                img_data = np.rot90(img_data,3)# Rotate image
                img_data = torch.as_tensor(img_data.copy()).permute(2,0,1)
                img_data = self.transform(img_data)
                img_data= normalize_imagenet(img_data.float()/255.0)
            
            images.append(img_data)
            width,height = img_data.shape[-1],img_data.shape[-2]
            
            kf = kfs[str(fid)]              
            kf_bid2oid = {v[1]:v[0] for v in kf.attrs['seg2idx']}
            kfdata = np.asarray(kf)
            
            per_frame_info = per_frame_info_dict[fid]
            per_frame_info['nodes'] = dict()
            per_frame_info['edges'] = dict()
            
            # get GT class
            per_frame_indices = list()
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
                
                idx2iid[om_id] = oid
                per_frame_indices.append(om_id)
                
                assert oid not in per_frame_info['nodes']
                per_frame_info['nodes'][oid] = om_id#.append({'oid':oid,'mid':om_id})
                
                ''' '''
                obj = object_data[oid]
                node_descriptor_for_image.append( descriptor_generator(obj) )
                
            for om_id1 in per_frame_indices:
                for om_id2 in per_frame_indices:
                    if om_id1 != om_id2:
                        em_id = len(edge_indices)
                        key_mapped = (idx2iid[om_id1],idx2iid[om_id2])
                        assert key_mapped not in per_frame_info['edges']
                        per_frame_info['edges'][key_mapped] = em_id
                        
                        key = (om_id1,om_id2)
                        edge_indices.append(key)
                        
        if DRAW_BBOX_IMAGE:
            t_img = torch.stack(images,dim=0)
            show_tensor_images(t_img.float()/255, '-')
        
        '''build predicate GT'''
        if self.multi_rel_outputs:
            adj_matrix_onehot = np.zeros([len(cat), len(cat), len(self.relationNames)])
        else:
            adj_matrix = np.ones([len(cat), len(cat)]) * self.none_idx#set all to none label.
            
        relatinoships_gt= defaultdict(list)
        for r in relationships_data:
            r_src = int(r[0])
            r_tgt = int(r[1])
            r_lid = int(r[2])
            r_cls = r[3]
            # if r_src not in instance2mask or r_tgt not in instance2mask: continue
            # index1 = instance2mask[r_src]
            # index2 = instance2mask[r_tgt]
            # assert index1>=0
            # assert index2>=0
            # if self.sample_in_runtime:
            #     if [index1,index2] not in edge_indices: continue
            if r_cls not in self.relationNames:
                continue  
            r_lid = self.relationNames.index(r_cls) # remap the index of relationships in case of custom relationNames
            # assert(r_lid == self.relationNames.index(r_cls))
            
            relatinoships_gt[(r_src,r_tgt)].append(r_lid)

            # if index1 >= 0 and index2 >= 0:
            #     if self.multi_rel_outputs:
            #         adj_matrix_onehot[index1, index2, r_lid] = 1
            #     else:
            #         adj_matrix[index1, index2] = r_lid      
        '''edge GT to tensor'''
        if self.multi_rel_outputs:
            rel_dtype = np.float32
            adj_matrix_onehot = torch.from_numpy(np.array(adj_matrix_onehot, dtype=rel_dtype))
        else:
            rel_dtype = np.int64
            adj_matrix = torch.from_numpy(np.array(adj_matrix, dtype=rel_dtype))
        if self.multi_rel_outputs:
            gt_rels = torch.zeros(len(edge_indices), len(self.relationNames),dtype = torch.float)
            # gt_rels[:,self.none_idx] = 1
        else:
            gt_rels = torch.ones(len(edge_indices),dtype = torch.long) * self.none_idx
            
        for e in range(len(edge_indices)):
            edge = edge_indices[e]
            index1 = edge[0]
            index2 = edge[1]
            
            iid1 = idx2iid[index1]
            iid2 = idx2iid[index2]
            key = (iid1,iid2)
            if key in relatinoships_gt:
                if self.multi_rel_outputs:
                    for x in relatinoships_gt[key]:
                        gt_rels[e,x] = 1# adj_matrix_onehot[index1,index2,x]
                else:
                    if len(relatinoships_gt[key])!=1:
                        print('scan_id',scan_id)
                        print('iid1,iid2',iid1,iid2)
                        print('index1,index2',index1,index2)
                        print('key, relatinoships_gt[key]',key,relatinoships_gt[key])
                        print(instance2labelName[key[0]],instance2labelName[key[1]])
                        [print(self.relationNames[x])for x in relatinoships_gt[key]]
                        assert len(relatinoships_gt[key])==1
                    gt_rels[e] = relatinoships_gt[key][0]
                
        '''build temporal node graph'''
        temporal_node_graph=list()
        temporal_edge_graph=list()
        sorted_kf_indices = sorted(kf_indices)
        for idx in range(len(sorted_kf_indices)-1):
            fid_0 = sorted_kf_indices[idx]
            fid_1 = sorted_kf_indices[idx+1]
            finfo_0,finfo_1 = per_frame_info_dict[fid_0], per_frame_info_dict[fid_1]
            
            '''check if node exist'''
            nodes_0,nodes_1 = finfo_0['nodes'],finfo_1['nodes']
            for oid_0 in nodes_0:
                if oid_0 in nodes_1:
                    temporal_node_graph.append([nodes_0[oid_0],nodes_1[oid_0]])
                    
            '''check edges'''
            edges_0,edges_1 = finfo_0['edges'],finfo_1['edges']
            for key_0 in edges_0:
                if key_0 in edges_1:
                    temporal_edge_graph.append([edges_0[key_0],edges_1[key_0]])
        
        
        # collect predictions belong to the same node
        # iid2idxes = defaultdict(list)
        # for idx,iid in idx2iid.items():iid2idxes[iid].append(idx)
        
        # # build sequential connection
        # for iid, indices in iid2idxes.items():
        #     indices = sorted(indices) # temporal
        #     for idx in range(len(indices)-1):
        #         idx1 = indices[idx]
        #         idx2 = indices[idx+1]
        #         temporal_node_graph.append([idx1,idx2])
                    
        # edgeIid_2_edgeIndices=defaultdict(list)
        # for idx, edge in enumerate(edge_indices):
        #     index1 = edge[0]
        #     index2 = edge[1]
        #     iid1 = idx2iid[index1]
        #     iid2 = idx2iid[index2]
        #     key = (iid1,iid2)
        #     edgeIid_2_edgeIndices[key].append(idx)
        # for key, indices in edgeIid_2_edgeIndices.items():
        #     for idx1 in indices:
        #         for idx2 in indices:
        #             if idx1 == idx2:continue
        #             temporal_edge_graph.append([idx1,idx2])
                
        '''to tensor'''
        assert len(bounding_boxes) > 0
        images = torch.stack(images,dim=0)
        assert len(cat) == len(bounding_boxes)
        bounding_boxes = torch.from_numpy(np.array(bounding_boxes)).float()
        gt_class = torch.from_numpy(np.array(cat))
        edge_indices = torch.tensor(edge_indices,dtype=torch.long)
        temporal_node_graph = torch.tensor(temporal_node_graph,dtype=torch.long)
        temporal_edge_graph = torch.tensor(temporal_edge_graph,dtype=torch.long)
        if len(node_descriptor_for_image)>0:
            node_descriptor_for_image = torch.stack(node_descriptor_for_image)
        else:
            node_descriptor_for_image = torch.tensor([],dtype=torch.long)
            
        # node_descriptor_for_image = torch.tensor([],dtype=torch.long)
        
        output = dict()
        output['scan_id'] = scan_id # str
        output['gt_rel'] = gt_rels  # tensor
        output['gt_cls'] = gt_class # tensor
        output['images'] = images# tensor
        output['node_edges'] = edge_indices # tensor
        output['mask2instance'] = idx2iid #dict
        output['image_boxes'] = bounding_boxes #list
        output['temporal_node_graph'] = temporal_node_graph
        output['temporal_edge_graph'] = temporal_edge_graph
        output['node_descriptor_8'] = node_descriptor_for_image
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
    
    path='./experiments/config_VGfM_full_l20_5.yaml'
    
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
        # break