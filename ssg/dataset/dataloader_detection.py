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
class Graph_Loader (data.Dataset):
    def __init__(self, config, mode):
        super().__init__()
        assert mode in ['train','validation','test']
        # something to do with json. if the value is not np array. the multiprocessing in pytorch seems will copy them alot and crash
        torch.multiprocessing.set_sharing_strategy('file_system') 
        self._device = config.DEVICE
        # self.config = config.data
        path = config.data['path']
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
        self.multi_rel = config.data.multi_rel
        self.max_num_edge = config.data.max_num_edge
        self.img_feature_path = config.data.img_feature_path
        self.full_edge = config.data.full_edge
        self.normalize_weight = config.data.normalize_weight
        self.use_precompute = config.data.use_precompute_img_feature
        self.img_feature_type = config.model.node_encoder.backend
        
        if config.data.img_size > 0:
            self.transform = transforms.Compose([
                transforms.Resize(config.data.img_size),
                transforms.ToTensor(),
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
        pth_relationship_json = os.path.join(path,'relationships_%s.json' % (mode))
        selected_scans = read_txt_to_list(os.path.join(path,'%s_scans.txt' % (mode)))
        # elif mode == 'validation':
        #     pth_relationship_json = os.path.join(path,'relationships_validation.json')
        #     selected_scans = read_txt_to_list(os.path.join(path,'validation_scans.txt'))
        names_classes = read_txt_to_list(pth_classes)
        names_relationships = read_txt_to_list(pth_relationships)
        
        if not multi_rel_outputs:
            if 'none' not in names_relationships:
                names_relationships.append('none')
        
        self.relationNames = sorted(names_relationships)
        self.classNames = sorted(names_classes)
        
        ''' load data '''
        self.path_data = pth_relationship_json # try not to load json here. it causes issue when worker is on
        with open(pth_relationship_json) as f:
            data = json.load(f)
        tmp = set([s for s in data.keys()])
        inter  = sorted(list(tmp.intersection(selected_scans)))
        
        # Convert dict and list to pandas and numpy to prevent issue in multithreading
        self.size = len(inter)
        self.scans = snp.pack(inter)#[s for s in data.keys()]
        self.data = pandas.DataFrame.from_dict(data)
        
        '''compute wieghts'''
        if not self.for_eval:
            if config.data.full_edge:
                edge_mode='fully_connected'
            else:
                edge_mode='nn'
            # edge_mode='gt'
            # print('edge_mode:',edge_mode)
            wobjs, wrels, o_obj_cls, o_rel_cls = compute_weight.compute(self.classNames, self.relationNames, self.data, selected_scans,
                                                                        normalize=config.data.normalize_weight,
                                                                        for_BCE=multi_rel_outputs==True,
                                                                        edge_mode=edge_mode,
                                                                        verbose=config.VERBOSE)            
            for idx, obj_cls_name in enumerate(self.classNames):
                if obj_cls_name in config.data.obj_ignore_list:
                    if config.VERBOSE:
                        print('ignore object ',obj_cls_name)
                    wobjs[idx]=0
            self.w_node_cls = torch.from_numpy(np.array(wobjs)).float()
            self.w_edge_cls = torch.from_numpy(np.array(wrels)).float()
            # print(wobjs)
            
            # self.w_node_cls[self.w_node_cls==0] = self.w_node_cls[self.w_node_cls!=0].min()
            # self.w_edge_cls[self.w_edge_cls==0] = self.w_edge_cls[self.w_edge_cls!=0].min()
        # import resource
        # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        # resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
        
    def __len__(self):
        return self.size
    
    def open_hdf5(self, path):
        if not hasattr(self,'img_features'):
            self.img_features = h5py.File(path,'r')
            
    
    def __getitem__(self,idx):
        scan_id = snp.unpack(self.scans,idx)# self.scans[idx]
        scan = self.data[scan_id]
        # return scan_id
        objects = scan['objects']
        relationships = scan['relationships']
        keyframes = scan['kfs']
        ''' sample in runtime'''
        if self.sample_in_runtime:
            selected_nodes = [int(s) for s in list(objects.keys())]
            if not self.for_eval:
                sample_num_nn=self.sample_num_nn# 1 if "sample_num_nn" not in self.config else self.config.sample_num_nn
                sample_num_seed=self.sample_num_seed#1 if "sample_num_seed" not in self.config else self.config.sample_num_seed
                if sample_num_seed <= 0:
                    # use all nodes
                    filtered_nodes = selected_nodes
                else:
                    # random select node including neighbors
                    filtered_nodes = util_data.build_neighbor(objects, sample_num_nn, sample_num_seed) # select 1 node and include their neighbor nodes n times.
            else:
                filtered_nodes = selected_nodes # use all nodes            
            instances_id = list(filtered_nodes)
        if 0 in instances_id: instances_id.remove(0)
        
        
        ''' 
        Find instances we care abot. Build instance2mask and cat list
        instance2mask maps instances to a mask id. to randomize the order of instance in training.
        '''
        instance2mask=dict()
        # instance2mask[0]=0
        cat = []
        images_indices = set()
        counter = 0
        selected_instances = [int(s) for s in list(objects.keys())]
        filtered_instances = list()
        kfs_indices = list()
        for i in range(len(instances_id)):
            instance_id = instances_id[i]
            
            class_id = -1
            if instance_id not in selected_instances:
                # instance2mask[instance_id] = 0
                continue
            
            instance_labelName = objects[str(instance_id)]['label']
            
            # instance_labelName = instance2labelName[instance_id]
            if instance_labelName in self.classNames:
                class_id = self.classNames.index(instance_labelName)

            # if class_id != -1:
            #     counter += 1
            #     instance2mask[instance_id] = counter
            # else:
                # instance2mask[instance_id] = 0

            # mask to cat:
            if (class_id >= 0) and (instance_id > 0): # insstance 0 is unlabeled.
                counter += 1
                instance2mask[instance_id] = counter
            
                filtered_instances.append(instance_id)
                cat.append(class_id)
                
                # Load image
                obj = objects[str(instance_id)]
                kf_indices = obj['kfs']
                
                if not self.for_eval:
                    kf_indices = random_drop(kf_indices, self.drop_img_edge)    
                else:
                     kf_indices = random_drop(kf_indices, self.drop_img_edge_eval)  
                kfs_indices.append(kf_indices)
                images_indices = images_indices.union(kf_indices)
            
        '''build descriptor'''
        descriptor = torch.zeros([len(cat), 8])
        for i in range(len(filtered_instances)):
            instance_id = filtered_instances[i]
            obj = objects[str(instance_id)]
            descriptor[i] = util_data.gen_descriptor(obj)
            
            

        '''Map edge indices to mask indices'''
        if self.sample_in_runtime:
            if not self.full_edge:
                edge_indices = util_data.build_edge_from_selection(filtered_nodes, objects, max_edges_per_node=-1)
            else:
                ''' Build fully connected edges '''
                edge_indices = list()
                for n in range(len(cat)):
                    for m in range(len(cat)):
                        if n == m:continue
                        edge_indices.append([n,m])
            if len(edge_indices)>0:
                if not self.for_eval:
                    edge_indices = random_drop(edge_indices, self.drop_edge)
                    
                if self.for_eval :
                    edge_indices = random_drop(edge_indices, self.drop_edge_eval)
                    
                if self.max_num_edge > 0 and len(edge_indices) > self.max_num_edge:
                    choices = np.random.choice(range(len(edge_indices)),self.max_num_edge,replace=False).tolist()
                    edge_indices = [edge_indices[t] for t in choices]
                
        
        # if not self.full_edge:
        #     edge_indices = [[instance2mask[edge[0]]-1,instance2mask[edge[1]]-1] for edge in edge_indices ]
        # else:
        #     ''' Build fully connected edges '''
        #     edge_indices = list()
        #     max_edges=-1
        #     for n in range(len(cat)):
        #         for m in range(len(cat)):
        #             if n == m:continue
        #             edge_indices.append([n,m])
        #     if max_edges>0 and len(edge_indices) > max_edges :
        #         # for eval, do not drop out any edges.
        #         indices = list(np.random.choice(len(edge_indices),self.max_edges,replace=False))
        #         edge_indices = edge_indices[indices]

        ''' Build rel class GT '''
        if self.multi_rel:
            adj_matrix_onehot = np.zeros([len(cat), len(cat), len(self.relationNames)])
        else:
            adj_matrix = np.zeros([len(cat), len(cat)])
            adj_matrix += len(self.relationNames)-1 #set all to none label.
            
        multiple_relationship_counter=dict()
        for r in relationships:
            if r[0] not in instance2mask or r[1] not in instance2mask: continue
            if r[3] not in self.relationNames: continue  
            index1 = instance2mask[r[0]]-1
            index2 = instance2mask[r[1]]-1
            assert index1>=0
            assert index2>=0
            # if self.sample_in_runtime:
            if [index1,index2] not in edge_indices: continue
        
            if (index1,index2) not in multiple_relationship_counter:
                multiple_relationship_counter[(index1,index2)] = 0
            multiple_relationship_counter[(index1,index2)] += 1
            
            r[2] = self.relationNames.index(r[3]) # remap the index of relationships in case of custom relationNames

            if index1 >= 0 and index2 >= 0:
                if self.multi_rel:
                    adj_matrix_onehot[index1, index2, r[2]] = 1
                else:
                    adj_matrix[index1, index2] = r[2]        
        if self.multi_rel:
            num_multiple_relationships=0
            for v in multiple_relationship_counter.values():
                if v > 1: num_multiple_relationships+=1
                    
        if self.multi_rel:
            rel_dtype = np.float32
            adj_matrix_onehot = torch.from_numpy(np.array(adj_matrix_onehot, dtype=rel_dtype))
        else:
            rel_dtype = np.int64
            adj_matrix = torch.from_numpy(np.array(adj_matrix, dtype=rel_dtype))
        
        ''' load images. convert image index to mask '''
        image2mask = dict()
        
        images = list()
        counter=0
        save_batch=False
        if self.use_precompute:
            pth = os.path.join(self.img_feature_path,'image_features.h5')
            self.open_hdf5(pth)
            img_features = self.img_features[scan_id][self.img_feature_type]
            for idx in images_indices:
                image2mask[idx]=counter
                counter+=1
                images.append(torch.from_numpy(img_features[idx]))
            
            # os.path.join(cfg.data.img_feature_path,'image_features.h5')
            
            # if save_batch:
            #     pth = os.path.join(self.img_feature_path,scan_id+'.pkl')
            #     kf_features = load_obj(pth)
            #     for idx in images_indices:
            #         kf = keyframes[idx]
            #         # bboxes = kf['bboxes']
            #         img_path = kf['path']
                    
            #         image2mask[idx]=counter
            #         counter+=1
                    
            #         '''load precomputed feature'''
            #         images.append(kf_features[idx])
            # else:
            #     for idx in images_indices:
            #         kf = keyframes[idx]
            #         img_path = kf['path']
            #         image2mask[idx]=counter
            #         counter+=1
                    
            #         pth = os.path.join(self.img_feature_path,scan_id,str(idx)+'.pkl')
            #         kf_feature = load_obj(pth)
            #         images.append(kf_feature)
        else:
             for idx in images_indices:
                 kf = keyframes[idx]
                 img_path = kf['path']
                 image2mask[idx]=counter
                 counter+=1
                 
                 image = Image.open(img_path) # (width, height)
                 # image = read_image(img_path) # (channel, height, width)
                 # image = torch.rot90(image,1,[-1,-2]) #TODO: maybe don't need to rotate.
                 # width,height = image.size[0], image.size[1]
                 if self.transform: image = self.transform(image)
                 image = normalize_imagenet(image)
                 images.append(image)
                 
        debug_plot=False
        bounding_boxes = list() # bounding_boxes[node_id]{kf_id: [boxes]}
        for i in range(len(filtered_instances)):
            instance_id = filtered_instances[i]
            obj = objects[str(instance_id)]
            kfs = kfs_indices[i]
            
            box_dict = dict()
            
            if debug_plot:
                imgs=list()
                tensor_boxes=list()

            for idx, kf_id in enumerate(kfs):
                # print(kf_id,keyframes[kf_id]['bboxes'].keys())
                box = keyframes[kf_id]['bboxes'][str(instance_id)]
                scaled_box  = box # already normalized during data generation
                # scaled_box = [box[0]/width, box[1]/height, box[2]/width, box[3]/height] # scale here. since image may be rescaled during data processing
                box_dict[image2mask[kf_id]] = torch.FloatTensor(scaled_box)
                # box_dict[idx,0] = image2mask[kf_id]
                # box_dict[idx,1] = scaled_box[0]
                # box_dict[idx,2] = scaled_box[1]
                # box_dict[idx,3] = scaled_box[2]
                # box_dict[idx,4] = scaled_box[3]
                if debug_plot:
                    '''debug: plot bbox and the roi cropping from roi_align'''
                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Rectangle
                    # from PIL import Image
                    # print('img_path',img_path)
                    img_path = keyframes[kf_id]['path']
                    im = Image.open(img_path)
                    box[0]*=im.size[0]
                    box[1]*=im.size[1]
                    box[2]*=im.size[0]
                    box[3]*=im.size[1]
                    
                    anchor = (box[0],box[1])
                    box_w = box[2]-box[0]
                    box_h = box[3]-box[1]
                    
                    # plt.imshow(im)
                    # Get the current reference
                    ax = plt.gca()
                    # # Create a Rectangle patch
                    rect = Rectangle(anchor,box_w,box_h,linewidth=1,edgecolor='r',facecolor='none')
                    # Add the patch to the Axes
                    ax.add_patch(rect)
                    ax.imshow(im)
                    plt.imshow(im)
                    plt.close()
                    
                    from torchvision.ops import roi_align
                    # from torchvision.io import read_image
                    box_tensor= torch.tensor(box).view(1,4)
                    image2 = read_image(img_path).float().unsqueeze(0)
                    roi_features = roi_align(image2, [box_tensor], (int(box_h),int(box_w)))
                    roi_features = roi_features.squeeze(0).long().permute(1, 2, 0)
                    # plt.imshow(  roi_features )
                    # plt.close()
                    tensor_boxes.append(box_tensor)
                    imgs.append(read_image(img_path).float())
            
            bounding_boxes.append(box_dict)
            
            if debug_plot:
                '''debug: plot roi image and the original image'''
                print('class:',self.classNames[cat[i]])
                imgs = torch.stack(imgs,dim=0)
                roi_features = roi_align(imgs, tensor_boxes, (int(250),int(250)))
                for j in range(len(imgs)):
                    img = imgs[j]
                    roi_img = roi_features[j]
                    img= img.long().permute(1, 2, 0)
                    roi_img = roi_img.long().permute(1, 2, 0)
                    plt.imshow(  img )
                    plt.imshow(  roi_img )
            

        ''' Build edge connection from each node to all the corresponding keyframes '''            
        # mapped_node_to_images = dict()
        mapped_image_indices=list()
        for map_node_idx in range(len(kfs_indices)):
            indices = kfs_indices[map_node_idx]
            [mapped_image_indices.append([map_node_idx,image2mask[idx]]) for idx in indices]
            # mapped_node_to_images[map_node_idx] = [image2mask[idx] for idx in indices]
        
        ''' Convert to tensor '''
        # Relationship
        if self.multi_rel:
            gt_rels = torch.zeros(len(edge_indices), len(self.relationNames),dtype = torch.float)
        else:
            gt_rels = torch.zeros(len(edge_indices),dtype = torch.long)
        for e in range(len(edge_indices)):
            edge = edge_indices[e]
            index1 = edge[0]
            index2 = edge[1]
            if self.multi_rel:
                gt_rels[e,:] = adj_matrix_onehot[index1,index2,:]
            else:
                gt_rels[e] = adj_matrix[index1,index2]
        # Object 
        gt_class = torch.from_numpy(np.array(cat))
        # Node Edges
        edge_indices = torch.tensor(edge_indices,dtype=torch.long)
        # Image edges
        image_indices = torch.tensor(mapped_image_indices,dtype=torch.long)
        # images
        images = torch.stack(images)
        
        
        
        output = dict()
        output['scan_id'] = scan_id # str
        output['gt_rel'] = gt_rels  # tensor
        output['gt_cls'] = gt_class # tensor
        output['images'] = images# tensor
        output['descriptor'] = descriptor #tensor
        output['node_edges'] = edge_indices # tensor
        output['image_edges'] = image_indices # tensor
        # output['node2img'] = mapped_node_to_images # dict
        output['instance2mask'] = instance2mask #dict
        output['image_boxes'] = bounding_boxes #list
        return output
    
if __name__ == '__main__':
    import ssg
    import codeLib
    from ssg.data.collate import graph_collate
    
    path = '../../configs/default.yaml'
    path = './configs/exp2_all_test_overfit.yaml'
    path = './configs/exp2_all_gnn.yaml'
    path= './configs/exp2_all_basic_bf200.yaml'
    config = codeLib.Config(path)
    config.DEVICE='cpu'
    config.model.node_encoder.backend='vgg16'
    codeLib.utils.util.set_random_seed(config.SEED)
    dataset = ssg.config.get_dataset(config,'train')
    
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
    for epoch in tqdm(range(config.training.max_epoch)):
        for data in tqdm(train_loader):
            continue
        break