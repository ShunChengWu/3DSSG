import glob
import numpy as np
import torch,random
import copy,os
from codeLib.utils.util import read_txt_to_list
# import ssg2d
from ssg.objects import Node
import codeLib.utils.string_numpy as snp
from ssg import define
import ast
import copy
def get_all_scan_id_splits(path:str):
    files = glob.glob(os.path.join(path,'*.txt'))
    splits = {}
    for file in files:
        name = os.path.basename(file).split('.')[0]
        splits[name] = read_txt_to_list(file)
    return splits

def read_all_scan_ids(path:str):
    # get all splits
    splits = get_all_scan_id_splits(path)
    scan_ids = []
    for v in splits.values():
        scan_ids += v
    # train_ids = read_txt_to_list(os.path.join(define.PATH_FILE,'train_scans.txt'))
    # val_ids = read_txt_to_list(os.path.join(define.PATH_FILE,'validation_scans.txt'))
    # test_ids = read_txt_to_list(os.path.join(define.PATH_FILE,'test_scans.txt'))
    return scan_ids

def merge_batch_mask2inst(mask2insts):
    '''
    mask2insts: list, dict
    each mask2inst cannot have the the same mask, but can have duplicate instances
    '''
    idx2seg=dict()
    # idx2seg[0]=[0]
    if isinstance(mask2insts,list):
        for it in range(len(mask2insts)):
            mask2inst= mask2insts[it]
            for idx,iid in mask2inst.items():
                idx = idx.item() if isinstance(idx, torch.Tensor) else idx
                if iid==0:continue
                if idx <0: continue
                # print(idx)
                # assert idx not in idx2seg
                
                if idx in idx2seg:
                    print('')
                    print(iid)
                    print(idx)
                    print(mask2insts)
                    assert idx not in idx2seg
                idx2seg[idx] = iid
    else:
        mask2inst = mask2insts
        for idx,iid in mask2inst.items():
            idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            if iid==0:continue
            if idx <0: continue
            # print(idx)
            # assert idx not in idx2seg
            
            if idx in idx2seg:
                print('')
                print(iid)
                print(idx)
                print(mask2insts)
                assert idx not in idx2seg
            idx2seg[idx] = iid
    return idx2seg

def data_to_raw(data:dict):
    '''
    this converts a dictionary to string for saving purpose (e.g. h5py)
    '''
    return np.array([str(data)],dtype='S')

def raw_to_data(raw):
    '''
    this converts a string of dict back to dict
    '''
    return ast.literal_eval(raw[0].decode())

def cvt_all_to_dict_from_h5(data:dict):
    output = dict()
    for k,v in data.items():
        output[k]= raw_to_data(v)
    return output

def load_graph(data, box_filter_size: list=[]):
    # with open(pth, "r") as read_file:
    #     data = json.load(read_file)
    if 'nodes' not in data: raise RuntimeError('wrong format')
    if 'kfs' not in data: raise RuntimeError('wrong format')
    assert len(box_filter_size) in [0,1,2]
    
    ''' keyframes '''
    node2kfs = dict()
    for kf_idx in range(len(data['kfs'])):
        kf = data['kfs'][kf_idx]
        path = kf['path']
        fname = os.path.basename(path)
        fid = int(''.join([x for x in fname if x.isdigit()]))
        data['kfs'][kf_idx]['fid'] = fid
        
        for key, value in kf['bboxes'].items():
            node_id = int(key)
            
            if len(box_filter_size)>0:
                width = value[2]-value[0]
                height = value[3]-value[1]
                # print('width,height:',width,height,box_filter_size[0])
                if len(box_filter_size)==1:
                    if(width < box_filter_size[0] or height < box_filter_size[0]):
                        continue
                elif len(box_filter_size)==2:
                    if(width < box_filter_size[0] or height < box_filter_size[1]):
                        continue
            
            if node_id not in node2kfs:
                node2kfs[node_id] = list()
            node2kfs[node_id].append(kf_idx)
    
    ''' convert node to class '''
    nodes = dict()
    
    for key,node in data['nodes'].items():
        node_id = int(key)
        if node_id not in node2kfs: continue
        Rcw = np.array(node['rotation']).reshape(3,3) # It's stored in Rwc with col major but numpy read row major. so it turns to be Rcw
        Rwc = np.transpose(Rcw).tolist()
        center = node['center']
        dims = node['dimension']
        neighbors = node['neighbors']
        gtInstance = node.get('gtInstance',None)
        nodes[node_id] = Node(center = center, 
                              dimension = dims, 
                              neighbors = neighbors, 
                              Rwc = Rwc, 
                              gtInstance=gtInstance,
                              kfs = node2kfs[node_id])    
    # if len(data['nodes']) != len(nodes):
    #     print('before,filtered',len(data['nodes']), len(nodes))
        
    output = dict()        
    output = copy.deepcopy(data)
    output['nodes'] = nodes
    output['kfs'] = data['kfs']
    return output 

def save_graph(graph:dict,pth:str):
    nodes = graph['nodes']
    
    node_j = dict()
    for key,node in nodes.items():
        id_str = str(key)
        nn = dict()
        nn['center'] = node.center.tolist()
        nn['dimension'] = node.dimension.tolist()
        nn['label'] = node.label
        nn['neighbors']=list()
        nn['rotation'] = [1,0,0,0,1,0,0,0,1]
        node_j[id_str]=nn
        
    kf_j = dict()
    kf_j = graph['kfs']
    
    with open(pth,'w') as file:
        pass
    pass

def build_edge_from_selection(node_ids,objects:dict, max_edges_per_node):
    '''
    flow: an edge passes message from i to j is denoted as  [i,j]. 
    '''
    ''' build trees '''
    edge_indices = list()
    for s_idx in node_ids:
        if s_idx in objects:
            nn = objects[s_idx]['neighbors']
        else:
            nn = objects[str(s_idx)]['neighbors']
        nn = set(nn).intersection(node_ids) # only the nodes within node_ids are what we want
        if s_idx in nn: nn.remove(s_idx) # remove itself
        if max_edges_per_node>0:
            if len(nn) > max_edges_per_node:
                nn = list(np.random.choice(list(nn),max_edges_per_node))
        
        for t_idx in nn:
            edge_indices.append([s_idx, t_idx])
    return edge_indices

def build_edge_from_selection_sgfn(node_ids,neighbor_dict, max_edges_per_node):
    '''
    flow: an edge passes message from i to j is denoted as  [i,j]. 
    '''
    ''' build trees '''
    edge_indices = list()
    for s_idx in node_ids:
        if s_idx in neighbor_dict:
            nn = neighbor_dict[s_idx]
        else:
            nn = neighbor_dict[str(s_idx)]
        nn = set(nn).intersection(node_ids) # only the nodes within node_ids are what we want
        if s_idx in nn: nn.remove(s_idx) # remove itself
        if max_edges_per_node>0:
            if len(nn) > max_edges_per_node:
                nn = list(np.random.choice(list(nn),max_edges_per_node))
        
        for t_idx in nn:
            edge_indices.append([s_idx, t_idx])
    return edge_indices

def build_neighbor(objects:dict, n_times:int, n_seed = 1):
    ''' Select node'''
    selected_nodes = [int(s) for s in list(objects.keys())]
    index = np.random.choice(np.unique(selected_nodes),n_seed).tolist()
    index = list(set(index)) # make them unique
    # for n_idx in selected_nodes:
    #     if str(n_idx) not in nns:
    #         print('cannot find key',n_idx,'in',nns.keys())
    #         assert str(n_idx) in nns.keys()
    
    ''' loop over n times'''
    filtered_nodes = set()
    n_seletected_nodes = dict() # this save the neighbor with level n. level n+1 is the neighbors from level n.
    n_seletected_nodes[0] = index # first layer is the selected node.
    filtered_nodes = filtered_nodes.union(n_seletected_nodes[0])
    for n in range(n_times):
        ''' collect neighbors '''
        n_seletected_nodes[n+1]=list()
        unique_nn_found = set()
        for node_idx in n_seletected_nodes[n]:
            nns = objects[str(node_idx)]['neighbors']
            
            found = set(nns)
            found = found.intersection(selected_nodes) # only choose the node within our selections
            if len(found)==0: continue
            unique_nn_found = unique_nn_found.union(found)
            filtered_nodes = filtered_nodes.union(found)
        n_seletected_nodes[n+1] = unique_nn_found
    
    return filtered_nodes

def build_neighbor_sgfn(nns:dict, selected_nodes:list, n_times:int, n_seed = 1):
    ''' Select node'''
    # selected_nodes = list(instance2labelName.keys())
    if n_seed > len(selected_nodes):n_seed = len(selected_nodes)
    index = np.random.choice(np.unique(selected_nodes),n_seed).tolist()
    index = list(set(index)) # make them unique
    for n_idx in selected_nodes:
        if str(n_idx) not in nns:
            assert str(n_idx) in nns.keys()
    
    ''' loop over n times'''
    filtered_nodes = set()
    filtered_nodes = filtered_nodes.union(index)
    n_seletected_nodes = dict() # this save the neighbor with level n. level n+1 is the neighbors from level n.
    n_seletected_nodes[0] = index # first layer is the selected node.
    for n in range(n_times):
        ''' collect neighbors '''
        n_seletected_nodes[n+1]=list()
        unique_nn_found = set()
        for node_idx in n_seletected_nodes[n]:
            found = set(nns[str(node_idx)])
            found = found.intersection(selected_nodes) # only choose the node within our selections
            found = found.difference([0])# ignore 0
            if len(found)==0: continue
            unique_nn_found = unique_nn_found.union(found)
            filtered_nodes = filtered_nodes.union(found)
        n_seletected_nodes[n+1] = unique_nn_found
    
    return filtered_nodes

def gen_descriptor(obj:dict()):
    '''
    center,dims,volume,length
    [3,3,1,1]
    '''
    # Rinv = torch.FloatTensor(np.array(obj['rotation']).reshape(3,3))
    center = torch.FloatTensor(obj['center'])
    dims = torch.FloatTensor(obj['dimension'])
    volume = (dims[0]*dims[1]*dims[2]).unsqueeze(0)
    length = dims.max().unsqueeze(0)
    return torch.cat([center,dims,volume,length],dim=0)



def gen_descriptor_pts(pts:torch.tensor):
    '''
    centroid_pts,std_pts,segment_dims,segment_volume,segment_lengths
    [3, 3, 3, 1, 1]
    '''
    assert pts.ndim==2
    assert pts.shape[-1]==3
    # centroid [n, 3]
    centroid_pts = pts.mean(0) 
    # # std [n, 3]
    std_pts = pts.std(0)
    # dimensions [n, 3]
    segment_dims = pts.max(dim=0)[0] - pts.min(dim=0)[0]
    # volume [n, 1]
    segment_volume = (segment_dims[0]*segment_dims[1]*segment_dims[2]).unsqueeze(0)
    # length [n, 1]
    segment_lengths = segment_dims.max().unsqueeze(0)
    return torch.cat([centroid_pts,std_pts,segment_dims,segment_volume,segment_lengths],dim=0)

class Node_Descriptor_24(object):
    def __init__(self,with_bbox:bool=False):
        self.dim = 8
        self.with_bbox=with_bbox
        if with_bbox:
            self.dim+=18
        pass
    
    def __len__(self):
        return self.dim
    def __call__(self,obj:dict):
        
# def gen_descriptor_24(obj:dict()):
        '''
        center, dims,volume,length,x_max,x_min,y_max,y_min,z_max,z_min
        
        center,
        [3,3,1,1, 6*3]
        '''
        # Rinv = torch.FloatTensor(np.array(obj['rotation']).reshape(3,3))
        center = torch.FloatTensor(obj['center'])
        dims = torch.FloatTensor(obj['dimension'])
        volume = (dims[0]*dims[1]*dims[2]).unsqueeze(0)
        length = dims.max().unsqueeze(0)
        # Calculate min,max points on the roatated points
        #
        if self.with_bbox:
            six_pts = [ torch.FloatTensor([dims[0],0,0]), 
                        -torch.FloatTensor([dims[0],0,0]),
                        torch.FloatTensor([0,dims[1],0]),
                        -torch.FloatTensor([0,dims[1],0]),
                        torch.FloatTensor([0,0,dims[2]]),
                        -torch.FloatTensor([0,0,dims[2]]) ]
            six_pts = torch.stack(six_pts,dim=0)
            rotation = torch.FloatTensor(obj['normAxes'])
            six_pts = (rotation @ six_pts.t()).t() + center
        # Find x_max,x_min,y_max,y_min,z_max,z_min
        
        
        
        if self.with_bbox:
            return torch.cat([center,dims,volume,length,six_pts.flatten()],dim=0)
        else:
            return torch.cat([center,dims,volume,length],dim=0)

def zero_mean(point, normalize:bool):
    mean = torch.mean(point, dim=0)
    point -= mean.unsqueeze(0)
    ''' without norm to 1  '''
    if normalize:
        furthest_distance = point.pow(2).sum(1).sqrt().max() # find maximum distance for each n -> [n]
        point /= furthest_distance
    return point

def save_to_ply(points, path:str):
    with open(path,'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(points.shape[0]))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for i in range(points.shape[0]):
            point = points[i]
            f.write('{} {} {}\n'.format(point[0],point[1],point[2]))

def data_preparation(points, instances, selected_instances, num_points, num_points_union,
                     # use_rgb, use_normal,
                     for_train=False, instance2labelName=None, classNames=None,
                     rel_json=None, relationships=None, multi_rel_outputs=None, use_all=False,
                     padding=0.2, num_max_rel=-1, shuffle_objs=True, nns:dict = None,
                     sample_in_runtime:bool=False, num_nn=1, num_seed=1):
    if sample_in_runtime:
        assert nns is not None
        
    if for_train:
        assert(instance2labelName is not None)
        assert(rel_json is not None)
        assert(classNames is not None)
        assert(relationships is not None)
        assert(multi_rel_outputs is not None)
        
    instances = instances.flatten()

    instances_id = list(np.unique(instances))
    
    if sample_in_runtime:
        if num_nn==0 or num_seed ==0:
            use_all = True
        if not use_all:
            filtered_nodes = build_neighbor_sgfn(nns, list(instance2labelName.keys()), num_nn, num_seed)
        else:
            selected_nodes = list(instance2labelName.keys())
            filtered_nodes = selected_nodes # use all nodes
        edge_indices = build_edge_from_selection_sgfn(filtered_nodes, nns, max_edges_per_node=-1)
        
        if num_max_rel > 0 and len(edge_indices) > 0:
            choices = np.random.choice(range(len(edge_indices)),num_max_rel).tolist()
            edge_indices = [edge_indices[t] for t in choices]
        instances_id = list(filtered_nodes)
        
    if 0 in instances_id:
        instances_id.remove(0)
    if shuffle_objs:
        random.shuffle(instances_id)
        
    instance2mask = {}
    instance2mask_formask={}
    # instance2mask[0] = 0
    
    ''' Build instance2mask and their gt classes '''
    cat = []
    counter = 0
    filtered_instances = list()
    for instance_id in list(np.unique(instances)):
        
        class_id = -1 # was 0
        if instance_id not in selected_instances:
            # since we divide the whole graph into sub-graphs if the 
            # scene graph is too large to resolve the memory issue, there 
            # are instances that are not interested in the current sub-graph
            # instance2mask[instance_id] = 0
            instance2mask_formask[instance_id]=0
            continue

        instance_labelName = instance2labelName[instance_id]
        if instance_labelName in classNames: # is it a class we care about?
            class_id = classNames.index(instance_labelName)
            
        if (class_id >= 0) and (instance_id>0):
            instance2mask[instance_id] = counter
            counter += 1
            instance2mask_formask[instance_id] = counter
            filtered_instances.append(instance_id)
            cat.append(class_id)
        else:
            instance2mask_formask[instance_id] = 0
            
    '''Map edge indices to mask indices'''
    if sample_in_runtime:
        edge_indices = [[instance2mask[edge[0]],instance2mask[edge[1]]] for edge in edge_indices ]
            
    num_objects = len(instances_id) if selected_instances is None else len(selected_instances)

    masks = np.array(list(map(lambda l: instance2mask_formask[l], instances)), dtype=np.int32)
    
    dim_point = points.shape[1]
    obj_points = torch.zeros([num_objects, num_points, dim_point])
    
    # create normalized pointsets for each object, sorted like the masks
    bboxes = list()
    for i in range(num_objects):
        obj_pointset = points[np.where(masks == i+1)[0], :]
        min_box = np.min(obj_pointset[:,:3], 0) - padding
        max_box = np.max(obj_pointset[:,:3], 0) + padding
        bboxes.append([min_box,max_box])
        choice = np.random.choice(len(obj_pointset), num_points, replace=True)
        obj_pointset = obj_pointset[choice, :]
        obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))
        
        # save_to_ply(obj_pointset[:,:3],'./tmp_{}.ply'.format(i))
        
        obj_pointset[:,:3] = zero_mean(obj_pointset[:,:3], normalize=False)
        obj_points[i] = obj_pointset
        
    if not sample_in_runtime:
        # Build fully connected edges
        edge_indices = list()
        max_edges=-1
        for n in range(len(cat)):
            for m in range(len(cat)):
                if n == m:continue
                edge_indices.append([n,m])
        if max_edges>0 and len(edge_indices) > max_edges and for_train:
            # for eval, do not drop out any edges.
            indices = list(np.random.choice(len(edge_indices),max_edges))
            edge_indices = edge_indices[indices]

    if for_train:
        ''' Build rel class GT '''
        if multi_rel_outputs:
            adj_matrix_onehot = np.zeros([num_objects, num_objects, len(relationships)])
        else:
            adj_matrix = np.zeros([num_objects, num_objects])

        for r in rel_json:
            r_src = int(r[0])
            r_tgt = int(r[1])
            r_lid = int(r[2])
            r_cls = r[3]
            if r_src not in instance2mask or r_tgt not in instance2mask: continue
            index1 = instance2mask[r_src]
            index2 = instance2mask[r_tgt]
            if sample_in_runtime:
                if [index1,index2] not in edge_indices: continue
            
            if for_train:
                if r_cls not in relationships:
                    continue  
                r_lid = relationships.index(r_cls) # remap the index of relationships in case of custom relationNames

            if index1 >= 0 and index2 >= 0:
                if multi_rel_outputs:
                    adj_matrix_onehot[index1, index2, r_lid] = 1
                else:
                    adj_matrix[index1, index2] = r_lid
                    
        ''' Build rel point cloud '''            
        if multi_rel_outputs:
            rel_dtype = np.float32
            adj_matrix_onehot = torch.from_numpy(np.array(adj_matrix_onehot, dtype=rel_dtype))
        else:
            rel_dtype = np.int64
        
        if multi_rel_outputs:
            gt_rels = torch.zeros(len(edge_indices), len(relationships),dtype = torch.float)
        else:
            gt_rels = torch.zeros(len(edge_indices),dtype = torch.long)
            
    rel_points = list()
    for e in range(len(edge_indices)):
        edge = edge_indices[e]
        index1 = edge[0]
        index2 = edge[1]
        if for_train:
            if multi_rel_outputs:
                gt_rels[e,:] = adj_matrix_onehot[index1,index2,:]
            else:
                gt_rels[e] = adj_matrix[index1,index2]

        mask1 = (masks == index1+1).astype(np.int32) * 1
        mask2 = (masks == index2+1).astype(np.int32) * 2
        mask_ = np.expand_dims(mask1 + mask2, 1)
        bbox1 = bboxes[index1]
        bbox2 = bboxes[index2]
        min_box = np.minimum(bbox1[0], bbox2[0])
        max_box = np.maximum(bbox1[1], bbox2[1])
        filter_mask = (points[:,0] > min_box[0]) * (points[:,0] < max_box[0]) \
                    * (points[:,1] > min_box[1]) * (points[:,1] < max_box[1]) \
                    * (points[:,2] > min_box[2]) * (points[:,2] < max_box[2])
        points4d = np.concatenate([points, mask_], 1)

        pointset = points4d[np.where(filter_mask > 0)[0], :]
        choice = np.random.choice(len(pointset), num_points_union, replace=True)
        pointset = pointset[choice, :]
        pointset = torch.from_numpy(pointset.astype(np.float32))
        
        # save_to_ply(pointset[:,:3],'./tmp_rel_{}.ply'.format(e))
        
        pointset[:,:3] = zero_mean(pointset[:,:3],False)
        rel_points.append(pointset)

    if for_train:
        try:
            rel_points = torch.stack(rel_points, 0)
        except:
            rel_points = torch.zeros([0, num_points_union, 4])
    else:
        rel_points = torch.stack(rel_points, 0)

    cat = torch.from_numpy(np.array(cat, dtype=np.int64))
    edge_indices = torch.tensor(edge_indices,dtype=torch.long)
    
    if for_train:
        return obj_points, rel_points, edge_indices, instance2mask, gt_rels, cat
    else:
        return obj_points, rel_points, edge_indices, instance2mask
    
def match_class_info_from_two(dataset_seg,dataset_inst, multi_rel:bool):
    ''' get scan_idx mapping '''
    scanid2idx_seg = dict()
    for index in range(len(dataset_seg)):
        scan_id = snp.unpack(dataset_seg.scans,index)# self.scans[idx]
        scanid2idx_seg[scan_id] = index
        
    scanid2idx_inst = dict()
    for index in range(len(dataset_inst)):
        scan_id = snp.unpack(dataset_inst.scans,index)# self.scans[idx]
        scanid2idx_inst[scan_id] = index
        
    '''add a none class for missing instances'''
    node_cls_names = copy.copy(dataset_seg.classNames)
    edge_cls_names = copy.copy(dataset_seg.relationNames)
    if define.NAME_NONE not in dataset_seg.classNames:
        node_cls_names.append(define.NAME_NONE)
    if not multi_rel:
        if define.NAME_NONE not in dataset_seg.relationNames:
            edge_cls_names.append(define.NAME_NONE)
        # remove same part
        if define.NAME_SAME_PART in edge_cls_names:
            edge_cls_names.remove(define.NAME_SAME_PART)
            
        noneidx_edge_cls = edge_cls_names.index(define.NAME_NONE)
    else:
        noneidx_edge_cls = None
        if define.NAME_NONE in dataset_seg.relationNames:
            edge_cls_names.remove(define.NAME_NONE)
            noneidx_edge_cls = edge_cls_names.index(define.NAME_NONE)
    
    noneidx_node_cls = node_cls_names.index(define.NAME_NONE)
    
    '''
    Find index mapping. Ignore NONE for nodes since it is used for mapping missing instance.
    Ignore SAME_PART for edges.
    '''
    seg_valid_node_cls_indices = []
    inst_valid_node_cls_indices = []
    for idx in range(len(node_cls_names)):
        name = node_cls_names[idx]
        if name == define.NAME_NONE: continue
        seg_valid_node_cls_indices.append(idx)
    for idx in range(len(node_cls_names)):
        name = node_cls_names[idx]
        if name == define.NAME_NONE: continue
        inst_valid_node_cls_indices.append(idx)
    
    seg_valid_edge_cls_indices = []
    inst_valid_edge_cls_indices = []
    for idx in range(len(edge_cls_names)):
        name = edge_cls_names[idx]
        if name == define.NAME_SAME_PART: continue
        seg_valid_edge_cls_indices.append(idx)
    for idx in range(len(edge_cls_names)):
        name = edge_cls_names[idx]
        if name == define.NAME_SAME_PART: continue
        inst_valid_edge_cls_indices.append(idx)
        
    return (scanid2idx_seg, scanid2idx_inst, node_cls_names, edge_cls_names,noneidx_node_cls,noneidx_edge_cls,
            seg_valid_node_cls_indices,inst_valid_node_cls_indices,
            seg_valid_edge_cls_indices,inst_valid_edge_cls_indices)