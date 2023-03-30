if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
    
import torch.utils.data as data
import os, random, torch, json, trimesh
import numpy as np
import multiprocessing as mp

from utils import util_ply, util_data, util, define
from data_processing import compute_weight_occurrences
import op_utils


def load_mesh(path,label_file,use_rgb,use_normal):
    result=dict()
    if label_file == 'labels.instances.align.annotated.v2.ply' or label_file == 'labels.instances.align.annotated.ply':
        if use_rgb:
            plydata = util_ply.load_rgb(path)
        else:
            plydata = trimesh.load(os.path.join(path,label_file), process=False)
        
        points = np.array(plydata.vertices)
        instances = util_ply.read_labels(plydata).flatten()
        
        if use_rgb:
            r = plydata.metadata['ply_raw']['vertex']['data']['red']
            g = plydata.metadata['ply_raw']['vertex']['data']['green']
            b = plydata.metadata['ply_raw']['vertex']['data']['blue']
            rgb = np.stack([ r,g,b]).squeeze().transpose()
            points = np.concatenate((points, rgb), axis=1)
        if use_normal:
            nx = plydata.metadata['ply_raw']['vertex']['data']['nx']
            ny = plydata.metadata['ply_raw']['vertex']['data']['ny']
            nz = plydata.metadata['ply_raw']['vertex']['data']['nz']
            normal = np.stack([ nx,ny,nz ]).squeeze().transpose()
            points = np.concatenate((points, normal), axis=1)
            

        result['points']=points
        result['instances']=instances
        
    else:# label_file.find('inseg')>=0 or label_file == 'cvvseg.ply':
        plydata = trimesh.load(os.path.join(path,label_file), process=False)
        points = np.array(plydata.vertices)
        instances = plydata.metadata['ply_raw']['vertex']['data']['label'].flatten()
        
        if use_rgb:
            rgbs = np.array(plydata.colors)[:,:3] / 255.0 * 2 - 1.0
            points = np.concatenate((points, rgbs), axis=1)
        if use_normal:
            nx = plydata.metadata['ply_raw']['vertex']['data']['nx']
            ny = plydata.metadata['ply_raw']['vertex']['data']['ny']
            nz = plydata.metadata['ply_raw']['vertex']['data']['nz']
            
            normal = np.stack([ nx,ny,nz ]).squeeze().transpose()
            points = np.concatenate((points, normal), axis=1)
        result['points']=points
        result['instances']=instances
    # else:
    #     raise NotImplementedError('')
    
    return result

def dataset_loading_3RScan(root:str, pth_selection:str,split:str,class_choice:list=None):

    pth_catfile = os.path.join(pth_selection, 'classes.txt')
    classNames = util.read_txt_to_list(pth_catfile)
    
    pth_relationship = os.path.join(pth_selection, 'relationships.txt')
    util.check_file_exist(pth_relationship)
    relationNames = util.read_relationships(pth_relationship)

    selected_scans=set()
    data = dict()
    if split == 'train_scans' :
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'train_scans.txt')))
        with open(os.path.join(root, 'relationships_train.json'), "r") as read_file:
             data1 = json.load(read_file)
    elif split == 'validation_scans':
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'validation_scans.txt')))
        with open(os.path.join(root, 'relationships_validation.json'), "r") as read_file:
             data1 = json.load(read_file)
    elif split == 'test_scans':
        selected_scans = selected_scans.union(util.read_txt_to_list(os.path.join(pth_selection,'test_scans.txt')))
        with open(os.path.join(root, 'relationships_test.json'), "r") as read_file:
             data1 = json.load(read_file)
    else:
        raise RuntimeError('unknown split type.')

    # with open(os.path.join(root, 'relationships_train.json'), "r") as read_file:
    #     data1 = json.load(read_file)
    # with open(os.path.join(root, 'relationships_validation.json'), "r") as read_file:
    #     data2 = json.load(read_file)
    # with open(os.path.join(root, 'relationships_test.json'), "r") as read_file:
    #     data3 = json.load(read_file)
    
    data['scans'] = data1['scans']# + data2['scans'] + data3['scans']
    if 'neighbors' in data1:
        data['neighbors'] = data1['neighbors']#{**data1['neighbors'], **data2['neighbors'], **data3['neighbors']}
    return  classNames, relationNames, data, selected_scans

class SGFNDataset (data.Dataset):
    def __init__(self,
                 config,
                 split='train',
                 multi_rel_outputs=True,
                 shuffle_objs=True,
                 use_rgb = False,
                 use_normal = False,
                 load_cache = False,
                 sample_in_runtime=True,
                 for_eval = False,
                 max_edges = -1,
                 data_augmentation=True):
        assert split in ['train_scans', 'validation_scans','test_scans']
        self.config = config
        self.mconfig = config.dataset
        self.use_data_augmentation=data_augmentation
        self.root = self.mconfig.root
        self.root_3rscan = define.DATA_PATH
        try:
            self.root_scannet = define.SCANNET_DATA_PATH
        except:
            self.root_scannet = None
        
        selected_scans = set()
        self.w_cls_obj=self.w_cls_rel=None
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
                jf = json.load(f)
            self.label_type = jf['label_type']
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
                jf = json.load(f)
            self.label_type = jf['label_type']    
            if self.mconfig.selection == "":
                self.mconfig.selection = self.root
            classNames, relationNames, data, selected_scans = \
                dataset_loading_3RScan(self.root, self.mconfig.selection, split)                
        self.relationNames = sorted(relationNames)
        self.classNames = sorted(classNames)
        
        if not multi_rel_outputs:
            if 'none' not in self.relationNames:
                self.relationNames.append('none')


        wobjs, wrels, o_obj_cls, o_rel_cls = compute_weight_occurrences.compute(self.classNames, self.relationNames, data,selected_scans)
        self.w_cls_obj = torch.from_numpy(np.array(o_obj_cls)).float().to(self.config.DEVICE)
        self.w_cls_rel = torch.from_numpy(np.array(o_rel_cls)).float().to(self.config.DEVICE)
        
        if not multi_rel_outputs:
            self.w_cls_rel[-1] = self.w_cls_rel.max()*10
        
        if False:
            ''' 1/log(x)'''
            self.w_cls_obj = torch.abs(1.0 / (torch.log(self.w_cls_obj)+1))
            self.w_cls_rel = torch.abs(1.0 / (torch.log(self.w_cls_rel)+1))
        else:
            ''' inverse sum'''
            self.w_cls_obj = self.w_cls_obj.sum() / (self.w_cls_obj + 1) /self.w_cls_obj.sum()
            self.w_cls_rel = self.w_cls_rel.sum() / (self.w_cls_rel + 1) /self.w_cls_rel.sum()
            self.w_cls_obj /= self.w_cls_obj.max()
            self.w_cls_rel /= self.w_cls_rel.max()
        
        if self.config.VERBOSE:
            print('=== {} classes ==='.format(len(self.classNames)))
            for i in range(len(self.classNames)):
                print('|{0:>3d} {1:>20s}'.format(i,self.classNames[i]),end='')
                if self.w_cls_obj is not None:
                    print(':{0:>1.5f}|'.format(self.w_cls_obj[i]),end='')
                if (i+1) % 2 ==0:
                    print('')
            print('')
            print('=== {} relationships ==='.format(len(self.relationNames)))
            for i in range(len(self.relationNames)):
                print('|{0:>3d} {1:>20s}'.format(i,self.relationNames[i]),end=' ')
                if self.w_cls_rel is not None:
                    print('{0:>1.5f}|'.format(self.w_cls_rel[i]),end='')
                if (i+1) % 2 ==0:
                    print('')
            print('')

        self.relationship_json, self.objs_json, self.scans, self.nns = self.read_relationship_json(data, selected_scans)
        if self.config.VERBOSE:
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
        
    def data_augmentation(self, points):
        # random rotate
        matrix= np.eye(3)
        matrix[0:3,0:3] = op_utils.rotation_matrix([0,0,1], np.random.uniform(0,2*np.pi,1))
        centroid = points[:,:3].mean(0)
        points[:,:3] -= centroid
        points[:,:3] = np.dot(points[:,:3], matrix.T)
        if self.use_normal:
            ofset=3
            if self.use_rgb:
                ofset+=3
            points[:,ofset:3+ofset] = np.dot(points[:,ofset:3+ofset], matrix.T)     
            
        ## Add noise
        # ## points
        # noise = np.random.normal(0,1e-3,[points.shape[0],3]) # 1 mm std
        # points[:,:3] += noise
        
        # ## colors
        # if self.use_rgb:
        #     noise = np.random.normal(0,0.078,[points.shape[0],3])
        #     colors = points[:,3:6]
        #     colors += noise
        #     colors[np.where(colors>1)] = 1
        #     colors[np.where(colors<-1)] = -1
            
        # ## normals
        # if self.use_normal:
        #     ofset=3
        #     if self.use_rgb:
        #         ofset+=3
        #     normals = points[:,ofset:3+ofset]
        #     normals = np.dot(normals, matrix.T)     
            
        #     noise = np.random.normal(0,1e-4,[points.shape[0],3])
        #     normals += noise
        #     normals = normals/ np.linalg.norm(normals)
        return points

    def __getitem__(self, index):
        scan_id = self.scans[index]
        scan_id_no_split = scan_id.rsplit('_',1)[0]
        instance2labelName  = self.objs_json[scan_id]
            
        if self.load_cache:
            data = self.cache_data[scan_id_no_split]
        else:
            if 'scene' in scan_id:
                path = os.path.join(self.root_scannet, scan_id_no_split)
            else:
                path = os.path.join(self.root_3rscan, scan_id_no_split)
            data = load_mesh(path, self.mconfig.label_file,self.use_rgb,self.use_normal)
        points = data['points']
        instances = data['instances']
        instances_id = list(np.unique(instances))    
        
        if self.use_data_augmentation and not self.for_eval:
           points = self.data_augmentation(points)
            
        if self.sample_in_runtime:
            if not self.for_eval:
                sample_num_nn=1
                sample_num_seed=1
                if "sample_num_nn" in self.mconfig:
                    sample_num_nn = self.mconfig.sample_num_nn
                if "sample_num_seed" in self.mconfig:
                    sample_num_seed = self.mconfig.sample_num_seed
                filtered_nodes = util_data.build_neighbor(self.nns[scan_id_no_split], instance2labelName, 
                                                          sample_num_nn, sample_num_seed) # select 1 node and include their neighbor nodes n times.
            else:
                selected_nodes = list(self.objs_json[scan_id].keys())
                filtered_nodes = selected_nodes # use all nodes
            edge_indices = util_data.build_edge_from_selection(filtered_nodes, self.nns[scan_id_no_split], max_edges_per_node=-1)
            
            instances_id = list(filtered_nodes)
            
            if self.mconfig.drop_edge>0 and not self.for_eval:
                percentage = np.random.uniform(low=1-self.mconfig.drop_edge, high=1.0,size=1)[0]
                num_edge = int(float(len(edge_indices))*percentage//1)
                if num_edge > 0:
                    choices = np.random.choice(range(len(edge_indices)),num_edge,replace=False).tolist()
                    edge_indices = [edge_indices[t] for t in choices]
                    
            if self.for_eval and self.mconfig.drop_edge_eval > 0:
                percentage = 1.0-self.mconfig.drop_edge
                num_edge = int(float(len(edge_indices))*percentage//1)
                if num_edge > 0:
                    choices = np.random.choice(range(len(edge_indices)),num_edge,replace=False).tolist()
                    edge_indices = [edge_indices[t] for t in choices]

            if self.max_edges > 0 and len(edge_indices) > self.max_edges:
                choices = np.random.choice(range(len(edge_indices)),self.max_edges,replace=False).tolist()
                edge_indices = [edge_indices[t] for t in choices]
        
        
        if 0 in instances_id:
            instances_id.remove(0)
            
        if self.shuffle_objs:
            random.shuffle(instances_id)
        
        instance2mask = {}
        instance2mask[0] = 0

        rel_json = self.relationship_json[scan_id]

        ''' 
        Find instances we care abot. Build instance2mask and cat list
        instance2mask maps instances to a mask id. to randomize the order of instance in training.
        '''
        cat = []
        counter = 0
        selected_instances = list(self.objs_json[scan_id].keys())
        filtered_instances = list()
        for i in range(len(instances_id)):
            instance_id = instances_id[i]
            
            class_id = -1
            if instance_id not in selected_instances:
                instance2mask[instance_id] = 0
                continue
            instance_labelName = instance2labelName[instance_id]
            if instance_labelName in self.classNames:
                class_id = self.classNames.index(instance_labelName)

            if class_id != -1:
                counter += 1
                instance2mask[instance_id] = counter
            else:
                instance2mask[instance_id] = 0

            # mask to cat:
            if (class_id >= 0) and (instance_id > 0): # insstance 0 is unlabeled.
                filtered_instances.append(instance_id)
                cat.append(class_id)

        '''Map edge indices to mask indices'''
        if self.sample_in_runtime:
            edge_indices = [[instance2mask[edge[0]]-1,instance2mask[edge[1]]-1] for edge in edge_indices ]
        else:
            ''' Build fully connected edges '''
            edge_indices = list()
            max_edges=-1
            for n in range(len(cat)):
                for m in range(len(cat)):
                    if n == m:continue
                    edge_indices.append([n,m])
            if max_edges>0 and len(edge_indices) > max_edges :
                # for eval, do not drop out any edges.
                indices = list(np.random.choice(len(edge_indices),self.max_edges,replace=False))
                edge_indices = edge_indices[indices]

        ''' random sample points '''
        use_obj_context=False #TODO: not here
        obj_points = torch.zeros([len(cat), self.mconfig.num_points, self.dim_pts])
        descriptor = torch.zeros([len(cat), 11])
        for i in range(len(filtered_instances)):
            instance_id = filtered_instances[i]
            obj_pointset = points[np.where(instances== instance_id)[0], :]
            
            if use_obj_context:
                min_box = np.min(obj_pointset[:,:3], 0) - 0.02
                max_box = np.max(obj_pointset[:,:3], 0) + 0.02
                filter_mask = (points[:,0] > min_box[0]) * (points[:,0] < max_box[0]) \
                    * (points[:,1] > min_box[1]) * (points[:,1] < max_box[1]) \
                    * (points[:,2] > min_box[2]) * (points[:,2] < max_box[2])
                obj_pointset = points[np.where(filter_mask > 0)[0], :]
                
            if len(obj_pointset) == 0:
                print('scan_id:',scan_id)
                print('selected_instances:',len(selected_instances))
                print('filtered_instances:',len(filtered_instances))
                print('instance_id:',instance_id)
            choice = np.random.choice(len(obj_pointset), self.mconfig.num_points, replace= len(obj_pointset) < self.mconfig.num_points)
            obj_pointset = obj_pointset[choice, :]
            descriptor[i] = op_utils.gen_descriptor(torch.from_numpy(obj_pointset)[:,:3])
            obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))
            obj_pointset[:,:3] = self.norm_tensor(obj_pointset[:,:3])
            obj_points[i] = obj_pointset
        obj_points = obj_points.permute(0,2,1)
        # noise = torch.FloatTensor(obj_points.shape).normal_(0,0.005)
        # obj_points+=noise
        
        ''' Build rel class GT '''
        if self.multi_rel_outputs:
            adj_matrix_onehot = np.zeros([len(cat), len(cat), len(self.relationNames)])
        else:
            adj_matrix = np.zeros([len(cat), len(cat)])
            adj_matrix += len(self.relationNames)-1 #set all to none label.
            
        if not self.sample_in_runtime:
            edge_indices = list()
            max_edges=-1
            for n in range(len(cat)):
                for m in range(len(cat)):
                    if n == m:continue
                    edge_indices.append([n,m])
            if max_edges>0 and len(edge_indices) > max_edges and not self.for_eval: 
                # for eval, do not drop out any edges.
                indices = list(np.random.choice(len(edge_indices),max_edges,replace=False))
                edge_indices = edge_indices[indices]
            
        for r in rel_json:
            if r[0] not in instance2mask or r[1] not in instance2mask: continue
            index1 = instance2mask[r[0]]-1
            index2 = instance2mask[r[1]]-1
            if self.sample_in_runtime:
                if [index1,index2] not in edge_indices: continue
            
            if r[3] not in self.relationNames:
                continue  
            r[2] = self.relationNames.index(r[3]) # remap the index of relationships in case of custom relationNames
            # assert(r[2] == self.relationNames.index(r[3]))

            if index1 >= 0 and index2 >= 0:
                if self.multi_rel_outputs:
                    adj_matrix_onehot[index1, index2, r[2]] = 1
                else:
                    adj_matrix[index1, index2] = r[2]        
                    
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
        
        ''' Build obj class GT '''
        gt_class = torch.from_numpy(np.array(cat))
        
        edge_indices = torch.tensor(edge_indices,dtype=torch.long)
        
        
        return scan_id, instance2mask, obj_points, edge_indices, gt_class, gt_rels, descriptor

    def __len__(self):
        return len(self.scans)
    
    def norm_tensor(self, points):
        assert points.ndim == 2
        assert points.shape[1] == 3
        centroid = torch.mean(points, dim=0) # N, 3
        points -= centroid # n, 3, npts
        furthest_distance = points.pow(2).sum(1).sqrt().max() # find maximum distance for each n -> [n]
        points /= furthest_distance
        return points

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
                    # break
            if valid_counter < 2: # need at least two nodes
                continue

            rel[scan["scan"] + "_" + str(scan["split"])] = relationships
            scans.append(scan["scan"] + "_" + str(scan["split"]))

            
            objs[scan["scan"]+"_"+str(scan['split'])] = objects

        return rel, objs, scans, nns

if __name__ == '__main__':
    from config import Config
    config = Config('../config_example.json')
    config.dataset.root = "../data/example_data/"    
    config.dataset.label_file = 'inseg.ply'
    sample_in_runtime = True
    config.dataset.data_augmentation=True
    split_type = 'validation_scans' # ['train_scans', 'validation_scans','test_scans']
    dataset = SGFNDataset (config,use_rgb=True,use_normal = True, 
                                load_cache=False,sample_in_runtime=sample_in_runtime,
                                multi_rel_outputs=False,
                                for_eval=False, split=split_type,data_augmentation=config.dataset.data_augmentation)
    items = dataset.__getitem__(0)    
    print(items)