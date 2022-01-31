import torch.utils.data as data
import os, random, torch, json, trimesh, h5py, copy
import numpy as np
import multiprocessing as mp

# from utils import util_ply, util_data, util, define
from codeLib.common import random_drop
from codeLib import transformation
from ssg.utils import util_ply, util_data, util
from codeLib.utils.util import read_txt_to_list, check_file_exist
from ssg import define
# from data_processing import compute_weight_occurrences
import ssg.utils.compute_weight as compute_weight
from ssg.utils.util_data import raw_to_data, cvt_all_to_dict_from_h5
import codeLib.utils.string_numpy as snp

class SGFNDataset (data.Dataset):
    def __init__(self,config,mode, **args):
        assert mode in ['train','validation','test']
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
        self.load_cache  = load_cache = False
        self.for_eval = mode != 'train'
        self.max_edges=config.data.max_num_edge
        self.full_edge = self.config.data.full_edge
        
        self.output_node = args.get('output_node', True)
        self.output_edge = args.get('output_edge', True)    

        ''' read classes '''
        pth_classes = os.path.join(path,'classes.txt')
        pth_relationships = os.path.join(path,'relationships.txt')      
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
        c_sg_data = cvt_all_to_dict_from_h5(self.sg_data)
        del self.sg_data
        
        '''check scan_ids'''
        tmp = set(c_sg_data.keys())
        inter  = sorted(list(tmp.intersection(selected_scans)))
        
        '''pack with snp'''
        self.size = len(inter)
        self.scans = snp.pack(inter)#[s for s in data.keys()]

        '''compute weight  ''' #TODO: rewrite this. in runtime sampling the weight might need to be calculated in each epoch.
        if not self.for_eval:
            if config.data.full_edge:
                edge_mode='fully_connected'
            else:
                edge_mode='nn'
            wobjs, wrels, o_obj_cls, o_rel_cls = compute_weight.compute_sgfn(self.classNames, self.relationNames, c_sg_data, selected_scans,
                                                                        normalize=config.data.normalize_weight,
                                                                        for_BCE=multi_rel_outputs==True,
                                                                        edge_mode=edge_mode,
                                                                        verbose=config.VERBOSE)
            self.w_node_cls = torch.from_numpy(np.array(wobjs)).float()
            self.w_edge_cls = torch.from_numpy(np.array(wrels)).float()

        self.dim_pts = 3
        if self.use_rgb:
            self.dim_pts += 3
        if self.use_normal:
            self.dim_pts += 3
        
        '''cache'''
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
        
    def data_augmentation(self, points):
        # random rotate
        matrix= np.eye(3)
        matrix[0:3,0:3] = transformation.rotation_matrix([0,0,1], np.random.uniform(0,2*np.pi,1))
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
        scan_id = snp.unpack(self.scans,index)# self.scans[idx]
        
        self.open_data(self.path_h5)
        scan_data_raw = self.sg_data[scan_id]
        scan_data = raw_to_data(scan_data_raw)
        
        object_data = scan_data['nodes']
        relationships_data = scan_data['relationships']
        
        ''' build nn dict '''
        nns = dict()
        for oid, odata in object_data.items():
            nns[str(oid)] = [int(s) for s in odata['neighbors']]

        ''' build mapping '''
        instance2labelName  = { int(key): node['label'] for key,node in object_data.items()  }
            
        ''' load point cloud data '''
        if 'scene' in scan_id:
            path = os.path.join(self.root_scannet, scan_id)
        else:
            path = os.path.join(self.root_3rscan, scan_id)
            
        if self.config.data.load_cache:
            data = self.cache_data[scan_id]
        else:
            data = load_mesh(path, self.mconfig.label_file, self.use_rgb, self.use_normal)
        points = copy.deepcopy( data['points'] )
        instances = copy.deepcopy( data['instances'] )
        instances_id = list(np.unique(instances))
        
        if self.use_data_augmentation and not self.for_eval:
           points = self.data_augmentation(points)
            
        if self.sample_in_runtime:
            selected_nodes = list(object_data.keys())
            if not self.for_eval:
                sample_num_nn=self.mconfig.sample_num_nn# 1 if "sample_num_nn" not in self.config else self.config.sample_num_nn
                sample_num_seed=self.mconfig.sample_num_seed#1 if "sample_num_seed" not in self.config else self.config.sample_num_seed
                if sample_num_seed <= 0:
                    # use all nodes
                    filtered_nodes = selected_nodes
                else:
                    # random select node including neighbors
                    # filtered_nodes = util_data.build_neighbor(objects, sample_num_nn, sample_num_seed)
                    
                    filtered_nodes = util_data.build_neighbor_sgfn(nns, selected_nodes, 
                                                          sample_num_nn, sample_num_seed) # select 1 node and include their neighbor nodes n times.
            else:
                filtered_nodes = selected_nodes # use all nodes
        
        if 0 in instances_id:
            instances_id.remove(0)
            
        if self.shuffle_objs:
            random.shuffle(instances_id)
        
        instance2mask = {}

        ''' 
        Find instances we care abot. Build instance2mask and cat list
        instance2mask maps instances to a mask id. to randomize the order of instance in training.
        '''
        cat = []
        counter = 0
        selected_instances = list(object_data.keys())
        filtered_instances = list()
        for instance_id in filtered_nodes:
            # instance_id = instances_id[i]
            
            class_id = -1
            if instance_id not in selected_instances:
                # instance2mask[instance_id] = 0
                continue
            instance_labelName = instance2labelName[instance_id]
            if instance_labelName in self.classNames:
                class_id = self.classNames.index(instance_labelName)

            # if class_id != -1:
            #     counter += 1
            #     instance2mask[instance_id] = counter
            # else:
            #     instance2mask[instance_id] = 0

            # mask to cat:
            if (class_id >= 0) and (instance_id > 0): # insstance 0 is unlabeled.
                counter += 1
                instance2mask[instance_id] = counter
                filtered_instances.append(instance_id)
                cat.append(class_id)
        assert len(cat) > 0
        '''Map edge indices to mask indices'''

        ''' random sample points '''
        use_obj_context=False #TODO: not here
        obj_points = torch.zeros([len(cat), self.mconfig.node_feature_dim, self.dim_pts])
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
            choice = np.random.choice(len(obj_pointset), self.mconfig.node_feature_dim, replace= len(obj_pointset) < self.mconfig.node_feature_dim)
            obj_pointset = obj_pointset[choice, :]
            descriptor[i] = util_data.gen_descriptor_pts(torch.from_numpy(obj_pointset)[:,:3])
            obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))
            
            # util_data.save_to_ply(obj_pointset[:,:3],'./tmp_{}.ply'.format(i))
            
            obj_pointset[:,:3] = self.norm_tensor(obj_pointset[:,:3])
            obj_points[i] = obj_pointset
        obj_points = obj_points.permute(0,2,1)
        
        ''' Build rel class GT '''
        if self.multi_rel_outputs:
            adj_matrix_onehot = np.zeros([len(cat), len(cat), len(self.relationNames)])
        else:
            adj_matrix = np.zeros([len(cat), len(cat)])
            adj_matrix += len(self.relationNames)-1 #set all to none label.
        
        
        if self.sample_in_runtime:
            if not self.full_edge:
                edge_indices = util_data.build_edge_from_selection_sgfn(filtered_instances,nns,max_edges_per_node=-1)
                edge_indices = [[instance2mask[edge[0]]-1,instance2mask[edge[1]]-1] for edge in edge_indices ]
                # edge_indices = util_data.build_edge_from_selection(filtered_nodes, nns, max_edges_per_node=-1)
            else:
                edge_indices = list()
                for n in range(len(cat)):
                    for m in range(len(cat)):
                        if n == m:continue
                        edge_indices.append([n,m])
            if len(edge_indices)>0:
                if not self.for_eval:
                    edge_indices = random_drop(edge_indices, self.mconfig.drop_edge)
                    
                if self.for_eval :
                    edge_indices = random_drop(edge_indices, self.mconfig.drop_edge_eval)
                    
                if self.mconfig.max_num_edge > 0 and len(edge_indices) > self.max_num_edge:
                    choices = np.random.choice(range(len(edge_indices)),self.mconfig.max_num_edge,replace=False).tolist()
                    edge_indices = [edge_indices[t] for t in choices]
        else:
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
            
        rel_json = relationships_data
        for r in rel_json:
            r_src = int(r[0])
            r_tgt = int(r[1])
            r_lid = int(r[2])
            r_cls = r[3]
            
            if r_src not in instance2mask or r_tgt not in instance2mask: continue
            index1 = instance2mask[r_src]-1
            index2 = instance2mask[r_tgt]-1
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
        
        
        output = dict()
        output['scan_id'] = scan_id # str
        output['gt_rel'] = gt_rels  # tensor
        output['gt_cls'] = gt_class # tensor
        output['obj_points'] = obj_points
        output['descriptor'] = descriptor #tensor
        output['node_edges'] = edge_indices # tensor
        output['instance2mask'] = instance2mask #dict
        return output
        
        # return scan_id, instance2mask, obj_points, edge_indices, gt_class, gt_rels, descriptor

    def __len__(self):
        return self.size
    
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
    
    return result

def read_relationship_json(data, selected_scans:list, classNames:dict, isV2:bool=True):
        rel = dict()
        objs = dict()
        scans = list()
        nns = None
        
        if 'neighbors' in data:
            nns = data['neighbors']
        for scan in data['scans']:
            if scan["scan"] == 'fa79392f-7766-2d5c-869a-f5d6cfb62fc6':
                if isV2 == "labels.instances.align.annotated.v2.ply":
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
                if instance_labelName in classNames: # is it a class we care about?
                    valid_counter+=1
                    # break
            if valid_counter < 2: # need at least two nodes
                continue

            rel[scan["scan"] + "_" + str(scan["split"])] = relationships
            scans.append(scan["scan"] + "_" + str(scan["split"]))

            
            objs[scan["scan"]+"_"+str(scan['split'])] = objects

        return rel, objs, scans, nns

def dataset_loading_3RScan(root:str, pth_selection:str,mode:str,class_choice:list=None):
    
    pth_catfile = os.path.join(pth_selection, 'classes.txt')
    classNames = read_txt_to_list(pth_catfile)
    
    pth_relationship = os.path.join(pth_selection, 'relationships.txt')
    check_file_exist(pth_relationship)
    relationNames = read_txt_to_list(pth_relationship)
    
    selected_scans=set()
    
    selected_scans = selected_scans.union( read_txt_to_list(os.path.join(root,'%s_scans.txt' % (mode))) )
    # if split == 'train_scans' :
    #     selected_scans = selected_scans.union(read_txt_to_list(os.path.join(pth_selection,'train_scans.txt')))
    # elif split == 'validation_scans':
    #     selected_scans = selected_scans.union(read_txt_to_list(os.path.join(pth_selection,'validation_scans.txt')))
    # elif split == 'test_scans':
    #     selected_scans = selected_scans.union(read_txt_to_list(os.path.join(pth_selection,'test_scans.txt')))
    # else:
    #     raise RuntimeError('unknown split type.')

    with open(os.path.join(root, 'relationships_train.json'), "r") as read_file:
        data1 = json.load(read_file)
    with open(os.path.join(root, 'relationships_validation.json'), "r") as read_file:
        data2 = json.load(read_file)
    data = dict()
    data['scans'] = data1['scans'] + data2['scans']
    if 'neighbors' in data1:
        data['neighbors'] = {**data1['neighbors'], **data2['neighbors']}
    return  classNames, relationNames, data, selected_scans


if __name__ == '__main__':
    import codeLib
    
    path = './configs/default_sgfusion.yaml'
    config = codeLib.Config(path)
    
    # config.dataset.root = "../data/example_data/"    
    # config.dataset.label_file = 'inseg.ply'
    # sample_in_runtime = True
    # config.dataset.data_augmentation=True
    # split_type = 'validation_scans' # ['train_scans', 'validation_scans','test_scans']
    dataset = SGFNDataset (config, 'validation')
    items = dataset.__getitem__(0)    
    # print(items)