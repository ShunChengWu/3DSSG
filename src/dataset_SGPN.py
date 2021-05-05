if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
    

import os, sys, torch, json, trimesh
from utils import util_ply, util_data, util, define
from data_processing import compute_weight_occurrences
import numpy as np
import torch.utils.data as data
import multiprocessing as mp



def dataset_loading_3RScan(root:str, pth_selection:str,split:str,class_choice:list=None):    
    pth_catfile = os.path.join(pth_selection, 'classes.txt')
    classNames = util.read_txt_to_list(pth_catfile)
    
    pth_relationship = os.path.join(pth_selection, 'relationships.txt')
    util.check_file_exist(pth_relationship)
    relationNames = util.read_relationships(pth_relationship)
    
    selected_scans=set()
    if split == 'train_scans' :
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'train_scans.txt')))
    elif split == 'validation_scans':
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'validation_scans.txt')))
    elif split == 'test_scans':
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'test_scans.txt')))
    else:
        raise RuntimeError('unknown split type:',split)

    with open(os.path.join(root, 'relationships_train.json'), "r") as read_file:
        data1 = json.load(read_file)
    with open(os.path.join(root, 'relationships_validation.json'), "r") as read_file:
        data2 = json.load(read_file)
    data = dict()
    data['scans'] = data1['scans'] + data2['scans']
    data['neighbors'] = {**data1['neighbors'], **data2['neighbors']}
    return  classNames, relationNames, data, selected_scans

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))
            
            
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

class RIODatasetGraph(data.Dataset):
    def __init__(self,
                 config,
                 split='train_scans',
                 multi_rel_outputs=True,
                 shuffle_objs=True,
                 use_rgb = False,
                 use_normal = False,
                 load_cache = False,
                 sample_in_runtime=True,
                 for_eval = False,
                 max_edges = -1):
        assert split in ['train_scans', 'validation_scans','test_scans']
        self.config = config
        self.mconfig = config.dataset
        
        self.root = self.mconfig.root
        self.root_3rscan = define.DATA_PATH
        try:
            self.root_scannet = define.SCANNET_DATA_PATH
        except:
            self.root_scannet = None

        self.scans = []
        self.multi_rel_outputs = multi_rel_outputs
        self.shuffle_objs = shuffle_objs
        self.use_rgb = use_rgb
        self.use_normal = use_normal
        self.sample_in_runtime=sample_in_runtime
        self.load_cache = load_cache
        self.for_eval = for_eval
        self.max_edges=max_edges
        
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
        
        if isinstance(self.root, list):
            with open(os.path.join(self.root[0],'args.json'), 'r') as f:
                data = json.load(f)
                self.label_type = data['label_type']
                # self.isV2 = data['v2'] > 0
            classNames = None
            relationNames = None
            data = None
            selected_scans = None
            for i in range(len(self.root)):
                selection = self.mconfig.selection
                if selection == "":
                    selection = self.root[i]
                l_classNames, l_relationNames, l_data, l_selected_scans = \
                    dataset_loading_3RScan(self.root[i], selection, split)
                
                if classNames is None:
                    classNames, relationNames, data, selected_scans = \
                        l_classNames, l_relationNames, l_data, l_selected_scans
                else:
                    classNames = set(classNames).union(l_classNames)
                    relationNames= set(relationNames).union(l_relationNames)
                    data['scans'] = l_data['scans'] + data['scans']
                    data['neighbors'] = {**l_data['neighbors'], **data['neighbors']}
                    selected_scans = selected_scans.union(l_selected_scans)
            classNames = list(classNames)
            relationNames = list(relationNames)
        else:
            with open(os.path.join(self.root,'args.json'), 'r') as f:
                data = json.load(f)
                self.label_type = data['label_type']
                # self.isV2 = data['v2'] > 0
            
            if self.mconfig.selection == "":
                self.mconfig.selection = self.root
            classNames, relationNames, data, selected_scans = \
                dataset_loading_3RScan(self.root, self.mconfig.selection, split)        
        
        self.relationNames = sorted(relationNames)
        self.classNames = sorted(classNames)
        
        if not multi_rel_outputs:
            if 'none' not in self.relationNames:
                self.relationNames.append('none')
                
        wobjs, wrels, o_obj_cls, o_rel_cls = compute_weight_occurrences.compute(self.classNames, self.relationNames, data,selected_scans, False)
        self.w_cls_obj = torch.from_numpy(np.array(o_obj_cls)).float().to(self.config.DEVICE)
        self.w_cls_rel = torch.from_numpy(np.array(o_rel_cls)).float().to(self.config.DEVICE)
        self.w_cls_obj = torch.abs(1.0 / (torch.log(self.w_cls_obj)+1)) # +1 to prevent 1 /log(1) = inf
        self.w_cls_rel = torch.abs(1.0 / (torch.log(self.w_cls_rel)+1)) # +1 to prevent 1 /log(1) = inf
        
        if not multi_rel_outputs:
            self.w_cls_rel[-1] = 0.001
            
        print('=== {} classes ==='.format(len(self.classNames)))
        for i in range(len(self.classNames)):
            print('|{0:>2d} {1:>20s}'.format(i,self.classNames[i]),end='')
            if self.w_cls_obj is not None:
                print(':{0:>1.3f}|'.format(self.w_cls_obj[i]),end='')
            if (i+1) % 2 ==0:
                print('')
        print('')
        print('=== {} relationships ==='.format(len(self.relationNames)))
        for i in range(len(self.relationNames)):
            print('|{0:>2d} {1:>20s}'.format(i,self.relationNames[i]),end=' ')
            if self.w_cls_rel is not None:
                print('{0:>1.3f}|'.format(self.w_cls_rel[i]),end='')
            if (i+1) % 2 ==0:
                print('')
        print('')
        
        self.relationship_json, self.objs_json, self.scans, self.nns = self.read_relationship_json(data, selected_scans)
        print('num of data:',len(self.scans))
        assert(len(self.scans)>0)
        if sample_in_runtime:
            assert(self.nns is not None)
            
        self.dim_pts = 3
        if self.use_rgb:
            self.dim_pts += 3
        if self.use_normal:
            self.dim_pts += 3
        
        self.cache_data = dict()
        if load_cache:
            pool = mp.Pool(8)
            pool.daemon = True
            # resutls=dict()
            for scan_id in self.scans:
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

    def __getitem__(self, index):
        scan_id = self.scans[index]
        scan_id_no_split = scan_id.rsplit('_',1)[0]
        selected_instances = list(self.objs_json[scan_id].keys())
        map_instance2labelName = self.objs_json[scan_id]
        
        if self.load_cache:
            data = self.cache_data[scan_id_no_split]
        else:
            if 'scene' in scan_id:
                path = os.path.join(self.root_scannet, scan_id_no_split)
            else:
                path = os.path.join(self.root_3rscan, scan_id_no_split)
            data = load_mesh(path, self.mconfig.label_file, self.use_rgb, self.use_normal)
        points = data['points']
        instances = data['instances']

        sample_num_nn=1
        sample_num_seed=1
        if "sample_num_nn" in self.mconfig:
            sample_num_nn = self.mconfig.sample_num_nn
        if "sample_num_seed" in self.mconfig:
            sample_num_seed = self.mconfig.sample_num_seed

        obj_points, rel_points, edge_indices, instance2mask, gt_rels, gt_class = \
            util_data.data_preparation(points, instances, selected_instances, 
                         self.mconfig.num_points, self.mconfig.num_points_union,
                         # use_rgb=self.use_rgb,use_normal=self.use_normal,
                         for_train=True, instance2labelName=map_instance2labelName, 
                         classNames=self.classNames,
                         rel_json=self.relationship_json[scan_id], 
                         relationships=self.relationNames,
                         multi_rel_outputs=self.multi_rel_outputs,
                         padding=0.2,num_max_rel=self.max_edges,
                         shuffle_objs=self.shuffle_objs, nns=self.nns[scan_id_no_split],
                         sample_in_runtime=self.sample_in_runtime,
                         num_nn=sample_num_nn, num_seed=sample_num_seed,
                         use_all = self.for_eval)
        return scan_id, instance2mask, obj_points, rel_points, gt_class, gt_rels, edge_indices

    def __len__(self):
        return len(self.scans)
    
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
    
if __name__ == '__main__':
    from config import Config
    config = Config('../config_example.json')
    config.dataset.root = '../data/example_data'
    config.dataset.label_file = 'inseg.ply'
    config.dataset_type = 'SGPN'
    config.dataset.load_cache=False
    use_rgb=True
    use_normal=True
    dataset = RIODatasetGraph(config,split='validation_scans',load_cache=config.dataset.load_cache,
                              use_rgb=use_rgb,use_normal=use_normal)
    scan_id, instance2mask, obj_points, rel_points, cat, target_rels, edge_indices = dataset.__getitem__(0)
    
    pass
