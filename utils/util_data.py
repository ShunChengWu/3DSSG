import numpy as np
import torch,random

def build_edge_from_selection(node_ids,neighbor_dict, max_edges_per_node):
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

def build_neighbor(nns:dict, instance2labelName:dict, n_times:int, n_seed = 1):
    ''' Select node'''
    selected_nodes = list(instance2labelName.keys())
    index = np.random.choice(np.unique(selected_nodes),n_seed).tolist()
    index = list(set(index)) # make them unique
    for n_idx in selected_nodes:
        if str(n_idx) not in nns:
            assert str(n_idx) in nns.keys()
    
    ''' loop over n times'''
    filtered_nodes = set()
    n_seletected_nodes = dict() # this save the neighbor with level n. level n+1 is the neighbors from level n.
    n_seletected_nodes[0] = index # first layer is the selected node.
    for n in range(n_times):
        ''' collect neighbors '''
        n_seletected_nodes[n+1]=list()
        unique_nn_found = set()
        for node_idx in n_seletected_nodes[n]:
            found = set(nns[str(node_idx)])
            found = found.intersection(selected_nodes) # only choose the node within our selections
            if len(found)==0: continue
            unique_nn_found = unique_nn_found.union(found)
            filtered_nodes = filtered_nodes.union(found)
        n_seletected_nodes[n+1] = unique_nn_found
    
    return filtered_nodes

def zero_mean(point):
    mean = torch.mean(point, dim=0)
    point -= mean.unsqueeze(0)
    ''' without norm to 1  '''
    # furthest_distance = point.pow(2).sum(1).sqrt().max() # find maximum distance for each n -> [n]
    # point /= furthest_distance
    return point

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
        if not use_all:
            filtered_nodes = build_neighbor(nns, instance2labelName, num_nn, num_seed)
        else:
            selected_nodes = list(instance2labelName.keys())
            filtered_nodes = selected_nodes # use all nodes
        edge_indices = build_edge_from_selection(filtered_nodes, nns, max_edges_per_node=-1)
        
        if num_max_rel > 0:
            choices = np.random.choice(range(len(edge_indices)),num_max_rel).tolist()
            edge_indices = [edge_indices[t] for t in choices]
        instances_id = list(filtered_nodes)
        
    if 0 in instances_id:
        instances_id.remove(0)
    if shuffle_objs:
        random.shuffle(instances_id)
        
    instance2mask = {}
    instance2mask[0] = 0
    
    cat = []
    counter = 0
    ''' Build instance2mask and their gt classes '''
    for instance_id in list(np.unique(instances)):
        # print('instance {} size: {}'.format(instance_id,len(points[np.where(instances == instance_id)])))
        if selected_instances is not None:
            if instance_id not in selected_instances:
                # since we divide the whole graph into sub-graphs if the 
                # scene graph is too large to resolve the memory issue, there 
                # are instances that are not interested in the current sub-graph
                instance2mask[instance_id] = 0
                continue

        if for_train:
            class_id = -1 # was 0

            instance_labelName = instance2labelName[instance_id]
            if instance_labelName in classNames: # is it a class we care about?
                class_id = classNames.index(instance_labelName)
                
            if (class_id >= 0) and (instance_id > 0): # there is no instance 0?
                cat.append(class_id)
        else:
            class_id = 0

        if class_id != -1: # was 0
            counter += 1
            instance2mask[instance_id] = counter
        else:
            instance2mask[instance_id] = 0
            
    '''Map edge indices to mask indices'''
    if sample_in_runtime:
        edge_indices = [[instance2mask[edge[0]]-1,instance2mask[edge[1]]-1] for edge in edge_indices ]
            
    num_objects = len(instances_id) if selected_instances is None else len(selected_instances)

    masks = np.array(list(map(lambda l: instance2mask[l], instances)), dtype=np.int32)
    
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
        obj_pointset[:,:3] = zero_mean(obj_pointset[:,:3])
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
            if r[0] not in instance2mask or r[1] not in instance2mask: continue
            index1 = instance2mask[r[0]]-1
            index2 = instance2mask[r[1]]-1
            if sample_in_runtime:
                if [index1,index2] not in edge_indices: continue
            
            if for_train:
                if r[3] not in relationships:
                    continue  
                r[2] = relationships.index(r[3]) # remap the index of relationships in case of custom relationNames

            if index1 >= 0 and index2 >= 0:
                if multi_rel_outputs:
                    adj_matrix_onehot[index1, index2, r[2]] = 1
                else:
                    adj_matrix[index1, index2] = r[2]
                    
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
        pointset[:,:3] = zero_mean(pointset[:,:3])
        rel_points.append(pointset)

    if for_train:
        try:
            rel_points = torch.stack(rel_points, 0)
        except:
            rel_points = torch.zeros([4, num_points_union])
    else:
        rel_points = torch.stack(rel_points, 0)

    cat = torch.from_numpy(np.array(cat, dtype=np.int64))
    edge_indices = torch.tensor(edge_indices,dtype=torch.long)
    
    if for_train:
        return obj_points, rel_points, edge_indices, instance2mask, gt_rels, cat
    else:
        return obj_points, rel_points, edge_indices, instance2mask