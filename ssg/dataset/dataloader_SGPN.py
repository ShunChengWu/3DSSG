if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
    

import os, sys, torch, json, trimesh
from codeLib.utils.util import read_txt_to_list
# from utils import util_ply, util_data, util, define
from data_processing import compute_weight_occurrences
import numpy as np
# import torch.utils.data as data
import multiprocessing as mp
import ssg.utils.compute_weight as compute_weight
import os
# from ssg.utils import util_data
from codeLib.utils.util import read_txt_to_list
import codeLib.utils.string_numpy as snp
from codeLib.common import normalize_imagenet, random_drop#, load_obj
import h5py
# import torch.utils.data as data
from torchvision import transforms
import ssg.utils.compute_weight as compute_weight
import torch
import json
import numpy as np
import pandas
from PIL import Image
from torchvision.io import read_image
from ssg.utils import util_data, util_ply
from ssg import define


# def dataset_loading_3RScan(root:str, pth_selection:str,split:str,class_choice:list=None):    
#     pth_catfile = os.path.join(pth_selection, 'classes.txt')
#     classNames = util.read_txt_to_list(pth_catfile)
    
#     pth_relationship = os.path.join(pth_selection, 'relationships.txt')
#     util.check_file_exist(pth_relationship)
#     relationNames = util.read_relationships(pth_relationship)
    
#     selected_scans=set()
#     if split == 'train_scans' :
#         selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'train_scans.txt')))
#     elif split == 'validation_scans':
#         selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'validation_scans.txt')))
#     elif split == 'test_scans':
#         selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'test_scans.txt')))
#     else:
#         raise RuntimeError('unknown split type:',split)

#     with open(os.path.join(root, 'relationships_train.json'), "r") as read_file:
#         data1 = json.load(read_file)
#     with open(os.path.join(root, 'relationships_validation.json'), "r") as read_file:
#         data2 = json.load(read_file)
#     data = dict()
#     data['scans'] = data1['scans'] + data2['scans']
#     data['neighbors'] = {**data1['neighbors'], **data2['neighbors']}
#     return  classNames, relationNames, data, selected_scans

# def gen_modelnet_id(root):
#     classes = []
#     with open(os.path.join(root, 'train.txt'), 'r') as f:
#         for line in f:
#             classes.append(line.strip().split('/')[0])
#     classes = np.unique(classes)
#     with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
#         for i in range(len(classes)):
#             f.write('{}\t{}\n'.format(classes[i], i))

class SGPNDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode, **args
                 # split='train_scans',
                 # multi_rel_outputs=True,
                 # shuffle_objs=True,
                 # use_rgb = False,
                 # use_normal = False,
                 # load_cache = False,
                 # sample_in_runtime=True,
                 # for_eval = False,
                 # max_edges = -1
                 ):
        assert mode in ['train','validation','test']
        torch.multiprocessing.set_sharing_strategy('file_system') 
        self._device = config.DEVICE
        path = config.data['path']
        self.config = config
        self.mconfig = config.data
        self.path = config.data.path
        self.use_data_augmentation=False
        self.root_3rscan = define.DATA_PATH
        self.path_h5 = os.path.join(self.path,'relationships_%s.h5' % (mode))
        try:
            self.root_scannet = define.SCANNET_DATA_PATH
        except:
            self.root_scannet = None
        
        selected_scans = set()
        self.w_cls_obj=self.w_cls_rel=None
        self.multi_rel_outputs = multi_rel_outputs = config.model.multi_rel
        self.shuffle_objs = False
        self.use_rgb = config.model.use_rgb
        self.use_normal = config.model.use_normal
        self.sample_in_runtime = sample_in_runtime = config.data.sample_in_runtime
        self.load_cache  = False
        self.for_eval = mode != 'train'
        self.max_edges=config.data.max_num_edge
        self.full_edge = self.config.data.full_edge
        
        self.output_node = args.get('output_node', True)
        self.output_edge = args.get('output_edge', True)
        # import resource
        # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        # resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
        
        ''' read classes '''
        pth_classes = os.path.join(path,'classes.txt')
        pth_relationships = os.path.join(path,'relationships.txt')
        # if mode == 'train':
        # pth_relationship_json = os.path.join(path,'relationships_%s.json' % (mode))
        selected_scans = read_txt_to_list(os.path.join(path,'%s_scans.txt' % (mode)))
        
        names_classes = read_txt_to_list(pth_classes)
        names_relationships = read_txt_to_list(pth_relationships)
        
        if not multi_rel_outputs:
            if 'none' not in names_relationships:
                names_relationships.append('none')
        elif 'none' in names_relationships:
            names_relationships.remove('none')
        
        self.relationNames = sorted(names_relationships)
        self.classNames = sorted(names_classes)
        
        ''' load data '''
        self.open_data(self.path_h5)
        tmp = set(self.sg_data.keys())
        
        # self.path_data = pth_relationship_json # try not to load json here. it causes issue when worker is on
        # with open(pth_relationship_json) as f:
            # data = json.load(f)
        # tmp = set([s['scan'] for s in data['scans']])
        inter  = sorted(list(tmp.intersection(selected_scans)))
        
        # Convert dict and list to pandas and numpy to prevent issue in multithreading
        self.size = len(inter)
        self.scans = snp.pack(inter)#[s for s in data.keys()]
        # self.data = pandas.DataFrame.from_dict(data)
        
        
        '''compute weight  ''' #TODO: rewrite this. in runtime sampling the weight might need to be calculated in each epoch.
        if not self.for_eval:
            if config.data.full_edge:
                edge_mode='fully_connected'
            else:
                edge_mode='nn'
            # wobjs, wrels, o_obj_cls, o_rel_cls = compute_weight.compute(self.classNames, self.relationNames, data,selected_scans)
            wobjs, wrels, o_obj_cls, o_rel_cls = compute_weight.compute_sgfn(self.classNames, self.relationNames, self.sg_data, selected_scans,
                                                                        normalize=config.data.normalize_weight,
                                                                        for_BCE=multi_rel_outputs==True,
                                                                        edge_mode=edge_mode,
                                                                        verbose=config.VERBOSE)
            # self.w_cls_obj = torch.from_numpy(np.array(o_obj_cls)).float().to(self.config.DEVICE)
            # self.w_cls_rel = torch.from_numpy(np.array(o_rel_cls)).float().to(self.config.DEVICE)
            
            self.w_node_cls = torch.from_numpy(np.array(wobjs)).float()
            self.w_edge_cls = torch.from_numpy(np.array(wrels)).float()
        
        # self.relationship_json, self.objs_json, self.scans, self.nns = self.read_relationship_json(data, selected_scans)
        if self.config.VERBOSE:
            print('num of data:',len(self.scans))
        assert(len(self.scans)>0)
        if sample_in_runtime and not config.data.full_edge:
            assert(self.nns is not None)
        
        self.dim_pts = 3
        if self.use_rgb:
            self.dim_pts += 3
        if self.use_normal:
            self.dim_pts += 3
        
        del self.sg_data # will re-open it in each thread
        
        self.cache_data = dict()
        if self.config.data.load_cache:
            print('load data to cache')
            pool = mp.Pool(8)
            pool.daemon = True
            # resutls=dict()
            for scan_id in inter:
                scan_id_no_split = scan_id.rsplit('_',1)[0]
                if 'scene' in scan_id:
                    path = os.path.join(self.root_scannet, scan_id_no_split)
                else:
                    path = os.path.join(self.root_3rscan, scan_id_no_split)
                if scan_id_no_split not in self.cache_data:
                    self.cache_data[scan_id_no_split] = pool.apply_async(load_mesh,
                                                                          (path, self.mconfig.label_file,self.use_rgb,self.use_normal))
            pool.close()
            pool.join()
            for key, item in self.cache_data.items():
                self.cache_data[key] = item.get()
        
    def open_data(self, path):
        if not hasattr(self,'sg_data'):
            self.sg_data = h5py.File(path,'r')


    def __getitem__(self, index):
        scan_id = snp.unpack(self.scans,index)# self.scans[idx]
        # scan_id_no_split = scan_id.rsplit('_',1)[0]
        
        self.open_data(self.path_h5)
        scan_data = self.sg_data[scan_id]
        
        object_data = scan_data['nodes']
        relationships_data = scan_data['relationships']
        
        # build nn dict
        nns = dict()
        for oid, odata in object_data.items():
            nns[str(oid)] = [int(s) for s in odata['neighbors']]
            
        # build mapping 
        instance2labelName  = { int(key): node.attrs['label'] for key,node in object_data.items()  }
        
        # load point cloud data
        if 'scene' in scan_id:
            path = os.path.join(self.root_scannet, scan_id)
        else:
            path = os.path.join(self.root_3rscan, scan_id)
            
        if self.config.data.load_cache:
            data = self.cache_data[scan_id]
        else:
            data = load_mesh(path, self.mconfig.label_file, self.use_rgb, self.use_normal)
        points = data['points']
        instances = data['instances']
        instances_id = list(np.unique(instances))
        
        selected_nodes = list(instance2labelName.keys())

        sample_num_nn=self.mconfig.sample_num_nn
        sample_num_seed=self.mconfig.sample_num_seed
        # if sample_num_seed==0: sample_num_seed=1 # need at least 1
        # if sample_num_nn==0:sample_num_nn=1# need at least 1
        
        obj_points, rel_points, edge_indices, instance2mask, gt_rels, gt_class = \
            util_data.data_preparation(points, instances, selected_nodes, 
                         self.mconfig.node_feature_dim, self.mconfig.num_points_union,
                         # use_rgb=self.use_rgb,use_normal=self.use_normal,
                         for_train=True, instance2labelName=instance2labelName, 
                         classNames=self.classNames,
                         rel_json=relationships_data, 
                         relationships=self.relationNames,
                         multi_rel_outputs=self.multi_rel_outputs,
                         padding=0.2,num_max_rel=self.max_edges,
                         shuffle_objs=self.shuffle_objs, nns=nns,
                         sample_in_runtime=self.sample_in_runtime,
                         num_nn=sample_num_nn, num_seed=sample_num_seed,
                         use_all = self.for_eval)
        # print('edge_indices.shape',edge_indices.shape)
        obj_points = obj_points.permute(0,2,1)
        rel_points = rel_points.permute(0,2,1)
        # if edge_indices.ndim > 1:
        # print('===')
        # print('edge_indices:',edge_indices.shape)
        output = dict()
        output['scan_id'] = scan_id # str
        output['instance2mask'] = instance2mask #dict
        output['obj_points'] = obj_points
        output['rel_points'] = rel_points
        output['gt_rel'] = gt_rels  # tensor
        output['gt_cls'] = gt_class # tensor
        output['node_edges'] = edge_indices # tensor
        return output# scan_id, instance2mask, obj_points, rel_points, gt_class, gt_rels, edge_indices

    def __len__(self):
        return self.size
    
    def read_relationship_json(self, data, selected_scans:list):
        rel = dict()
        objs = dict()
        scans = list()
        nns = None
        
        if 'neighbors' in data:
            nns = data['neighbors']
        for scan in data['scans']:
            if scan["scan"] == 'fa79392f-7766-2d5c-869a-f5d6cfb62fc6':
                if self.mconfig.label_file == "labels.instances.align.annotated.v2.ply":
                    '''
                    In the 3RScanV2, the segments on the semseg file and its ply file mismatch. 
                    This causes error in loading data.
                    To verify this, run check_seg.py
                    '''
                    continue
            if scan['scan'] not in selected_scans:
                continue
                
            relationships = []
            for realationship in scan["relationships"]:
                relationships.append(realationship)
                
            objects = {}
            for k, v in scan["objects"].items():
                objects[int(k)] = v
                
            # filter scans that doesn't have the classes we care
            instances_id = list(objects.keys())
            valid_counter = 0
            for instance_id in instances_id:
                instance_labelName = objects[instance_id]
                if instance_labelName in self.classNames: # is it a class we care about?
                    valid_counter+=1
            if valid_counter < 2: # need at least two nodes
                continue

            rel[scan["scan"] + "_" + str(scan["split"])] = relationships
            scans.append(scan["scan"] + "_" + str(scan["split"]))

            
            objs[scan["scan"]+"_"+str(scan['split'])] = objects

        return rel, objs, scans, nns
    

def load_mesh(path,label_file,use_rgb,use_normal):
    result=dict()
    if label_file == 'labels.instances.align.annotated.v2.ply' or label_file == 'labels.instances.align.annotated.ply':
        if use_rgb:
            plydata = util_ply.load_rgb(path)
        else:
            plydata = trimesh.load(os.path.join(path,label_file), process=False)
            
        points = np.array(plydata.vertices.tolist())
        instances = util_ply.read_labels(plydata).flatten()
        
        if use_rgb:
            rgbs = np.array(plydata.visual.vertex_colors.tolist())[:,:3]
            points = np.concatenate((points, rgbs), axis=1)
            
        if use_normal:
            normal = plydata.vertex_normals[:,:3]
            points = np.concatenate((points, normal), axis=1)
        
        result['points']=points
        result['instances']=instances
    elif label_file == 'inseg.ply':
        plydata = trimesh.load(os.path.join(path,label_file), process=False)
        points = np.array(plydata.vertices.tolist())
        instances = plydata.metadata['ply_raw']['vertex']['data']['label'].flatten()
        
        if use_rgb:
            rgbs = np.array(plydata.colors)[:,:3] / 255.0
            points = np.concatenate((points, rgbs), axis=1)
        if use_normal:
            nx = plydata.metadata['ply_raw']['vertex']['data']['nx']
            ny = plydata.metadata['ply_raw']['vertex']['data']['ny']
            nz = plydata.metadata['ply_raw']['vertex']['data']['nz']
            
            normal = np.stack([ nx,ny,nz ]).squeeze().transpose()
            points = np.concatenate((points, normal), axis=1)
        result['points']=points
        result['instances']=instances
    else:
        raise NotImplementedError('')
    return result
    
if __name__ == '__main__':
    from config import Config
    config = Config('../config_example.json')
    config.dataset.root = '../data/example_data'
    config.dataset.label_file = 'inseg.ply'
    config.dataset_type = 'SGPN'
    config.dataset.load_cache=False
    use_rgb=True
    use_normal=True
    dataset = SGPNDataset(config,split='validation_scans',load_cache=config.dataset.load_cache,
                              use_rgb=use_rgb,use_normal=use_normal)
    scan_id, instance2mask, obj_points, rel_points, cat, target_rels, edge_indices = dataset.__getitem__(0)
    
    pass
