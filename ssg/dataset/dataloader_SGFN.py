from collections import defaultdict
import torch.utils.data as data
import os, random, torch, json, trimesh, h5py, copy
import numpy as np
import multiprocessing as mp

# from utils import util_ply, util_data, util, define
from codeLib.common import random_drop, random_drop
from codeLib import transformation
from ssg.utils import util_ply, util_data
from codeLib.utils.util import read_txt_to_list, check_file_exist
from ssg import define
from codeLib.torch.visualization import show_tensor_images
from codeLib.common import normalize_imagenet
from torchvision import transforms
import codeLib.torchvision.transforms as cltransform
import ssg.utils.compute_weight as compute_weight
from ssg.utils.util_data import raw_to_data, cvt_all_to_dict_from_h5
import codeLib.utils.string_numpy as snp
import logging
from PIL import Image
from codeLib.common import run

logger_py = logging.getLogger(__name__)

DRAW_BBOX_IMAGE=True
DRAW_BBOX_IMAGE=False

class SGFNDataset (data.Dataset):
    def __init__(self,config,mode, **args):
        super().__init__()
        assert mode in ['train','validation','test']
        self.mode = mode
        self._device = config.DEVICE
        path = config.data['path']
        self.config = config
        self.cfg = self.config
        self.mconfig = config.data
        self.path = config.data.path
        self.label_file = config.data.label_file
        self.use_data_augmentation=self.mconfig.data_augmentation
        self.root_3rscan = define.DATA_PATH
        self.path_h5 = os.path.join(self.path,'relationships_%s.h5' % (mode))
        self.path_mv = os.path.join(self.path,'proposals.h5')
        self.path_roi_img = self.mconfig.roi_img_path
        self.pth_filtered = os.path.join(self.path,'filtered_scans_detection_%s.h5' % (mode))
        self.pth_node_weights = os.path.join(self.path,'node_weights.txt')
        self.pth_edge_weights = os.path.join(self.path,'edge_weights.txt')
        try:
            self.root_scannet = define.SCANNET_DATA_PATH
        except:
            self.root_scannet = None
            
        # SEGMENT_TYPE='GT'
        with open(os.path.join(self.cfg.data.path,'args.json')) as f:
            tmp = json.load(f)
            label_type = tmp['label_type']
            segment_type = tmp['segment_type']
        image_feature_folder_name =define.NAME_IMAGE_FEAUTRE_FORMAT.format(segment_type,label_type)
        self.path_img_feature = os.path.join(self.cfg.data.path_image_feature,image_feature_folder_name+'.h5')
        
        selected_scans = set()
        self.w_cls_obj=self.w_cls_rel=None
        self.multi_rel_outputs = multi_rel_outputs = config.model.multi_rel
        self.shuffle_objs = False
        self.use_rgb = config.model.use_rgb
        self.use_normal = config.model.use_normal
        self.sample_in_runtime= config.data.sample_in_runtime
        self.load_cache = False
        self.for_eval = mode != 'train'
        self.max_edges=config.data.max_num_edge
        self.full_edge = self.multi_rel_outputs #self.config.data.full_edge
        
        self.output_node = args.get('output_node', True)
        self.output_edge = args.get('output_edge', True)    

        ''' read classes '''
        pth_classes = os.path.join(path,'classes.txt')
        pth_relationships = os.path.join(path,'relationships.txt')      
        selected_scans = read_txt_to_list(os.path.join(path,'%s_scans.txt' % (mode)))
        
        names_classes = read_txt_to_list(pth_classes)
        names_relationships = read_txt_to_list(pth_relationships)
        
        if not multi_rel_outputs:
            if define.NAME_NONE not in names_relationships:
                names_relationships.append(define.NAME_NONE)
        elif define.NAME_NONE in names_relationships:
            names_relationships.remove(define.NAME_NONE)
        
        self.relationNames = sorted(names_relationships)
        self.classNames = sorted(names_classes)
        self.none_idx = self.relationNames.index(define.NAME_NONE) if not multi_rel_outputs else None
        
        '''set transform'''
        if self.mconfig.load_images:
            if self.mconfig.is_roi_img:
                if not self.for_eval:
                    self.transform  = transforms.Compose([
                        transforms.Resize(config.data.roi_img_size),
                        cltransform.TrivialAugmentWide(),
                        # RandAugment(),
                        ])
                else:
                    self.transform = transforms.Compose([
                        transforms.Resize(config.data.roi_img_size),
                        ])
            else:
                if not self.for_eval:
                    if config.data.img_size > 0:
                        self.transform = transforms.Compose([
                                    transforms.Resize(config.data.img_size),
                                ])
                    else:
                        self.transform = transforms.Compose([
                                ])
                else:
                    self.transform = transforms.Compose([
                                ])
        
        # Generate filtered data and compute weights
        self.__preprocessing()
        
        '''compute channel dims'''
        self.dim_pts = 3
        if self.use_rgb:
            self.dim_pts += 3
        if self.use_normal:
            self.dim_pts += 3
            
        '''pack with snp'''
        self.open_filtered()
        self.scans = snp.pack([k for k in self.filtered_data.keys()])
        self.size = len(self.filtered_data)
            
        '''check file exist'''
        if self.for_eval and self.mconfig.load_images: # train mode can't use precmopute feature. need to do img. aug.
            # self.open_filtered()
            should_compute_image_feature=False
            if not os.path.exists(self.path_img_feature):
                should_compute_image_feature=True
            else:
                self.open_image_feature()
                for scan_id in self.filtered_data:
                    if scan_id not in self.image_feature:
                        should_compute_image_feature=True
                        break
            
            if should_compute_image_feature:
                # Try to generate
                os.environ['MKL_THREADING_LAYER'] = 'GNU'
                # os.environ['PYTHONPATH'] = config.PYTHONPATH
                # subprocess.call(["export PYTHONPATH={}".format(PYTHONPATH)], shell=True) 
                bashCommand=[
                    'python','ssg/utils/compute_image_feature.py',
                    "--config",config.config_path,
                    '-n',image_feature_folder_name,
                    "-o",self.cfg.data.path_image_feature,
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
        
        '''cache'''
        self.cache_data = dict()
        if self.config.data.load_cache and self.mconfig.load_points:
            print('load data to cache')
            pool = mp.Pool(8)
            pool.daemon = True

            for scan_id in self.filtered_data:
                scan_id_no_split = scan_id.rsplit('_',1)[0]
                if 'scene' in scan_id:
                    path = os.path.join(self.root_scannet, scan_id_no_split)
                else:
                    path = os.path.join(self.root_3rscan, scan_id_no_split)
                if scan_id_no_split not in self.cache_data:
                    self.cache_data[scan_id_no_split] = pool.apply_async(load_mesh,
                                                                          (path, self.label_file,self.use_rgb,self.use_normal))
            pool.close()
            pool.join()
            for key, item in self.cache_data.items():
                self.cache_data[key] = item.get()
                
        # del self.sg_data
        del self.filtered_data
        # if self.mconfig.load_images:
            # del self.roi_imgs
            # del self.mv_data
                
    def open_filtered(self):
        self.filtered_data = h5py.File(self.pth_filtered,'r')
        
    def open_mv_graph(self):
        if not hasattr(self, 'mv_data'):
            self.mv_data = h5py.File(self.path_mv,'r')
                
    def open_data(self):
        if not hasattr(self,'sg_data'):
            self.sg_data = h5py.File(self.path_h5,'r')
            
    def open_img(self):
        if not hasattr(self, 'roi_imgs'):
            self.roi_imgs = h5py.File(self.path_roi_img,'r')
            
    def open_image_feature(self):
        if not hasattr(self,'image_feature'):
            self.image_feature = h5py.File(self.path_img_feature,'r')
           
    def __getitem__(self, index):
        scan_id = snp.unpack(self.scans,index)# self.scans[idx]
        
        self.open_data()
        
        scan_data_raw = self.sg_data[scan_id]
        scan_data = raw_to_data(scan_data_raw)
        
        object_data = scan_data['nodes']
        relationships_data = scan_data['relationships']
        mv_data = None
        if self.mconfig.load_images:
            self.open_mv_graph()
            
            mv_data = self.mv_data[scan_id]
            mv_nodes = mv_data['nodes'] # contain kf ids of a given node
            if self.mconfig.is_roi_img:
                self.open_img()
                roi_imgs = self.roi_imgs[scan_id]
            else:
                self.open_filtered()
                if self.for_eval:
                    self.open_image_feature()
            
            '''filter'''
            mv_node_ids = [int(x) for x in mv_data['nodes'].keys()]
            
            sg_node_ids = object_data.keys()                
            inter_node_ids = set(sg_node_ids).intersection(mv_node_ids)
            
            object_data = {nid: object_data[nid] for nid in inter_node_ids}
            
        ''' build nn dict '''
        nns = dict()
        seg2inst = dict()
        for oid, odata in object_data.items():
            nns[str(oid)] = [int(s) for s in odata['neighbors']]
            
            '''build instance dict'''
            if 'instance_id' in odata:
                seg2inst[oid] = odata['instance_id']
            
        ''' load point cloud data '''
        if self.mconfig.load_points:
            if 'scene' in scan_id:
                path = os.path.join(self.root_scannet, scan_id)
            else:
                path = os.path.join(self.root_3rscan, scan_id)
                
            if self.config.data.load_cache:
                data = self.cache_data[scan_id]
            else:
                data = load_mesh(path, self.label_file, self.use_rgb, self.use_normal)
            points = copy.deepcopy( data['points'] )
            instances = copy.deepcopy( data['instances'] )
        
            if self.use_data_augmentation and not self.for_eval:
               points = self.data_augmentation(points)
               
        '''extract 3D node classes and instances'''
        cat,oid2idx,idx2oid,filtered_instances = self.__sample_3D_nodes(object_data,mv_data,nns)
         
        '''sample 3D node connections'''
        edge_indices_3D = self.__sample_3D_node_edges(cat,oid2idx,filtered_instances,nns)
        
        '''extract relationships data'''
        relationships_3D = self.__extract_relationship_data(relationships_data,edge_indices_3D)
        
        '''sample 3D edges'''
        gt_rels_3D = self.__sample_relationships(relationships_3D,idx2oid,edge_indices_3D)

        ''' random sample points '''
        if self.mconfig.load_points:
            obj_points, descriptor, bboxes = self.__sample_points(scan_id,points,instances,cat,filtered_instances)
            
            '''build rel points'''
            if self.mconfig.rel_data_type == 'points':
                rel_points =    self.__sample_rel_points(points,instances,idx2oid,bboxes,edge_indices_3D)
            
        if self.mconfig.load_images:
            '''load images'''
            if self.mconfig.is_roi_img:
                roi_images, node_descriptor_for_image = self.__load_roi_images(cat,idx2oid,mv_nodes,roi_imgs,
                                       object_data,filtered_instances)
            else:
                images, img_bounding_boxes, bbox_cat, node_descriptor_for_image, \
                    image_edge_indices, img_idx2oid, temporal_node_graph, temporal_edge_graph = \
                        self.__load_full_images(scan_id,idx2oid,cat,scan_data,mv_data)
                relationships_img = self.__extract_relationship_data(relationships_data,image_edge_indices)
                gt_rels_2D = self.__sample_relationships(relationships_img,img_idx2oid,image_edge_indices)
                        
                '''to tensor'''
                assert len(img_bounding_boxes) > 0
                images = torch.stack(images,dim=0)
                assert len(bbox_cat) == len(img_bounding_boxes)
                img_bounding_boxes = torch.from_numpy(np.array(img_bounding_boxes)).float()
                gt_class_image = torch.from_numpy(np.array(bbox_cat))
                image_edge_indices = torch.tensor(image_edge_indices,dtype=torch.long)
                temporal_node_graph = torch.tensor(temporal_node_graph,dtype=torch.long)
                temporal_edge_graph = torch.tensor(temporal_edge_graph,dtype=torch.long)
                if len(node_descriptor_for_image)>0:
                    node_descriptor_for_image = torch.stack(node_descriptor_for_image)
                else:
                    node_descriptor_for_image = torch.tensor([],dtype=torch.long)
        
        ''' to tensor '''
        gt_class_3D = torch.from_numpy(np.array(cat))
        edge_indices_3D = torch.tensor(edge_indices_3D,dtype=torch.long)
        
        '''release'''
        del self.sg_data
        if self.mconfig.load_images:
            if self.mconfig.is_roi_img:
                del self.roi_imgs
            else:
                del self.filtered_data
                if self.for_eval:
                    del self.image_feature
            del self.mv_data
        
        output = dict()
        output['scan_id'] = scan_id # str
        output['gt_rel'] = gt_rels_3D
        output['gt_cls'] = gt_class_3D # tensor
        output['node_edges'] = edge_indices_3D # tensor
        output['mask2instance'] = idx2oid
        output['seg2inst'] = seg2inst
        
        if self.mconfig.load_points:
            output['obj_points'] = obj_points
            output['descriptor'] = descriptor #tensor
            if self.mconfig.rel_data_type == 'points':
                output['rel_points'] = rel_points
        if self.mconfig.load_images:
            if self.mconfig.is_roi_img:
                output['roi_imgs'] = roi_images #list
            else:
                output['images'] = images
                output['image_boxes'] = img_bounding_boxes
                output['temporal_node_graph'] = temporal_node_graph
                output['temporal_edge_graph'] = temporal_edge_graph
                output['image_node_edges'] = image_edge_indices
                
                output['image_gt_cls'] = gt_class_image
                output['image_gt_rel'] = gt_rels_2D
                output['image_mask2instance'] = img_idx2oid
                
            output['node_descriptor_8'] = node_descriptor_for_image

        return output
        
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
                if self.label_file == "labels.instances.align.annotated.v2.ply":
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
    
    def __preprocessing(self):
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
    
    def __sample_points(self, scan_id, points, instances, cat:list, filtered_instances:list):
        bboxes = list()
        use_obj_context=False #TODO: not here
        obj_points = torch.zeros([len(cat), self.mconfig.node_feature_dim, self.dim_pts])
        descriptor = torch.zeros([len(cat), 11])
        for i in range(len(filtered_instances)):
            instance_id = filtered_instances[i]
            obj_pointset = points[np.where(instances== instance_id)[0], :]
            
            min_box = np.min(obj_pointset[:,:3], 0)
            max_box = np.max(obj_pointset[:,:3], 0)
            if use_obj_context:
                min_box -= 0.02
                max_box += 0.02
                filter_mask = (points[:,0] > min_box[0]) * (points[:,0] < max_box[0]) \
                    * (points[:,1] > min_box[1]) * (points[:,1] < max_box[1]) \
                    * (points[:,2] > min_box[2]) * (points[:,2] < max_box[2])
                obj_pointset = points[np.where(filter_mask > 0)[0], :]
            bboxes.append([min_box,max_box])
                
            if len(obj_pointset) == 0:
                print('scan_id:',scan_id)
                # print('selected_instances:',len(selected_instances))
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
        return obj_points,descriptor, bboxes
           
    def __sample_rel_points(self,points,instances,idx2oid,bboxes, edge_indices):
        rel_points = list()
        for e in range(len(edge_indices)):
            edge = edge_indices[e]
            index1 = edge[0]
            index2 = edge[1]
            
            mask1 = (instances==idx2oid[index1]).astype(np.int32) * 1
            mask2 = (instances==idx2oid[index2]).astype(np.int32) * 2
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
            choice = np.random.choice(len(pointset), self.mconfig.num_points_union, replace=True)
            pointset = pointset[choice, :]
            pointset = torch.from_numpy(pointset.astype(np.float32))
            
            # save_to_ply(pointset[:,:3],'./tmp_rel_{}.ply'.format(e))
            
            pointset[:,:3] = zero_mean(pointset[:,:3],False)
            rel_points.append(pointset)
            
        if not self.for_eval:
            try:
                rel_points = torch.stack(rel_points, 0)
            except:
                rel_points = torch.zeros([0, self.mconfig.num_points_union, 4])
        else:
            if len(rel_points) == 0:
                # print('len(edge_indices)',len(edge_indices))
                # sometimes tere will have no edge because of only 1 ndoe exist. 
                # this is due to the label mapping/filtering process in the data generation
                rel_points = torch.zeros([0, self.mconfig.num_points_union, 4])
            else:
                rel_points = torch.stack(rel_points, 0)
        rel_points = rel_points.permute(0,2,1)
        return rel_points
    
    def __sample_3D_nodes(self, object_data:dict, mv_data:dict, nns:dict):
        instance2labelName  = { int(key): node['label'] for key,node in object_data.items()  }
        
        '''sample training set'''  
        instances_ids = list(instance2labelName.keys())
        if 0 in instances_ids: instances_ids.remove(0)
        if self.sample_in_runtime and not self.for_eval:
            selected_nodes = list(object_data.keys())
            if self.mconfig.load_images:
                mv_node_ids = [int(x) for x in mv_data['nodes'].keys()]
                selected_nodes = list( set(selected_nodes).intersection(mv_node_ids) )
            
            use_all=False
            sample_num_nn=self.mconfig.sample_num_nn# 1 if "sample_num_nn" not in self.config else self.config.sample_num_nn
            sample_num_seed=self.mconfig.sample_num_seed#1 if "sample_num_seed" not in self.config else self.config.sample_num_seed
            if sample_num_nn==0 or sample_num_seed ==0:
                use_all=True
                
            if not use_all:
                filtered_nodes = util_data.build_neighbor_sgfn(nns, selected_nodes, sample_num_nn, sample_num_seed) # select 1 node and include their neighbor nodes n times.
            else:
                filtered_nodes = selected_nodes # use all nodes
                
            instances_ids = list(filtered_nodes)
            if 0 in instances_ids: instances_ids.remove(0)
            
            if 'max_num_node' in self.mconfig and self.mconfig.max_num_node>0 and len(instances_ids)>self.mconfig.max_num_node:
                instances_ids = random_drop(instances_ids, self.mconfig.max_num_node )
        
            if self.shuffle_objs:
                random.shuffle(instances_ids)

        ''' 
        Find instances we care abot. Build oid2idx and cat list
        oid2idx maps instances to a mask id. to randomize the order of instance in training.
        '''
        oid2idx = {} # map instance_id to idx
        idx2oid = {} # map idx to instance_id
        cat = []
        counter = 0
        filtered_instances = list()
        for instance_id in instances_ids:    
            class_id = -1
            instance_labelName = instance2labelName[instance_id]
            if instance_labelName in self.classNames:
                class_id = self.classNames.index(instance_labelName)

            # mask to cat:
            if (class_id >= 0) and (instance_id > 0): # insstance 0 is unlabeled.
                oid2idx[int(instance_id)] = counter
                idx2oid[counter] = int(instance_id)
                counter += 1
                filtered_instances.append(instance_id)
                cat.append(class_id)
                
        return cat,oid2idx,idx2oid,filtered_instances
    
    def __extract_relationship_data(self, relationships_data,edge_indices):
        '''build relaitonship data'''
        relatinoships_gt= defaultdict(list)
        for r in relationships_data:
            r_src = int(r[0])
            r_tgt = int(r[1])
            r_lid = int(r[2])
            r_cls = r[3]
            
            if r_cls not in self.relationNames: continue # only keep the relationships we want
            r_lid = self.relationNames.index(r_cls) # remap the index of relationships in case of custom relationNames
            
            key = (r_src,r_tgt)
            
            # if r_src not in oid2idx or r_tgt not in oid2idx: continue
            # index1 = oid2idx[r_src]
            # index2 = oid2idx[r_tgt]
            # key = (index1,index2)
            # assert index1>=0
            # assert index2>=0
            # if self.sample_in_runtime:
                # print('index1,index2',index1,index2, type(edge_indices),edge_indices)
                # if key not in edge_indices: continue
            relatinoships_gt[key].append(r_lid)
        return relatinoships_gt
    
    def __sample_relationships(self, relatinoships_gt:dict, idx2oid:dict, edge_indices:list):
        if self.multi_rel_outputs:
            gt_rels = torch.zeros(len(edge_indices), len(self.relationNames),dtype = torch.float)
        else:
            gt_rels = torch.zeros(len(edge_indices),dtype = torch.long)
        for e in range(len(edge_indices)):
            edge = edge_indices[e]
            index1 = edge[0]
            index2 = edge[1]
            oid1 = idx2oid[index1]
            oid2 = idx2oid[index2]
            key = (oid1,oid2)
            if key in relatinoships_gt:
                if self.multi_rel_outputs:
                    for x in relatinoships_gt[key]:
                        gt_rels[e,x]=1
                else:
                    if len(relatinoships_gt[key])!=1:
                        # print('scan_id',scan_id)
                        # print('iid1,iid2',idx2oid[index1],idx2oid[index2])
                        print('index1,index2',index1,index2)
                        print('key, relatinoships_gt[key]',key,relatinoships_gt[key])
                        # print(instance2labelName[key[0]],instance2labelName[key[1]])
                        [print(self.relationNames[x])for x in relatinoships_gt[key]]
                        assert len(relatinoships_gt[key])==1
                    gt_rels[e] = relatinoships_gt[key][0]
        return gt_rels
    
    def __sample_3D_node_edges(self, cat:list, oid2idx:dict,filtered_instances:list,nns:dict):
        if self.sample_in_runtime:
            if self.full_edge:
                '''use dense'''
                edge_indices = list()
                for n in range(len(cat)):
                    for m in range(len(cat)):
                        if n == m:continue
                        edge_indices.append((n,m))
            else:
                if not self.for_eval:
                    '''sample from neighbor'''
                    edge_indices = util_data.build_edge_from_selection_sgfn(filtered_instances,nns,max_edges_per_node=-1)
                    edge_indices = [(oid2idx[edge[0]],oid2idx[edge[1]]) for edge in edge_indices ]
                else:
                    '''dense neighbor'''
                    edge_indices = set()
                    for k,v in nns.items():
                        k=int(k)
                        if k not in oid2idx:continue
                        mask_k = oid2idx[k]
                        for vv in v:
                            vv = int(vv)
                            if vv not in oid2idx:continue
                            mask_vv = oid2idx[vv]
                            edge_indices.add((mask_k,mask_vv))
                    edge_indices = [(e[0],e[1]) for e in edge_indices]

            '''edge dropout'''
            if len(edge_indices)>0:
                if not self.for_eval:
                    edge_indices = random_drop(edge_indices, self.mconfig.drop_edge)       
                if self.for_eval :
                    edge_indices = random_drop(edge_indices, self.mconfig.drop_edge_eval)
                    
                if self.mconfig.max_num_edge > 0 and len(edge_indices) > self.mconfig.max_num_edge and not self.for_eval:
                    choices = np.random.choice(range(len(edge_indices)),self.mconfig.max_num_edge,replace=False).tolist()
                    edge_indices = [edge_indices[t] for t in choices]
        else:
            edge_indices = list()
            max_edges=-1
            for n in range(len(cat)):
                for m in range(len(cat)):
                    if n == m:continue
                    edge_indices.append((n,m))
            if max_edges>0 and len(edge_indices) > max_edges and not self.for_eval: 
                # for eval, do not drop out any edges.
                indices = list(np.random.choice(len(edge_indices),max_edges,replace=False))
                edge_indices = edge_indices[indices]
        return edge_indices
    
    def __load_roi_images(self, cat:list, idx2oid:dict,mv_nodes:dict, roi_imgs:dict,
                          object_data:dict, filtered_instances:list):
        descriptor_generator = util_data.Node_Descriptor_24(with_bbox=self.mconfig.img_desc_6_pts)
        
        roi_images = list()
        node_descriptor_for_image = torch.zeros([len(cat), len(descriptor_generator)])
        
        '''get roi images'''
        for idx in range(len(cat)):
            oid = str(idx2oid[idx])
            node = mv_nodes[oid]
            cls_label = node.attrs['label']
            if cls_label == 'unknown':
                cls_label = self.classNames[cat[idx]]
                
            img_ids=range(len(roi_imgs[oid]))
            
            if not self.for_eval:
                img_ids = random_drop(img_ids, self.mconfig.drop_img_edge, replace=True)
            if self.for_eval :
                img_ids = random_drop(img_ids, self.mconfig.drop_img_edge_eval)
                
            img = [roi_imgs[oid][x] for x in img_ids]
            # else:
            #     kf_indices = [idx for idx in range(img.shape[0])]
            
            img = torch.as_tensor(np.array(img))#.clone()
            img = torch.clamp((img*255).byte(),0,255).byte()
            t_img = torch.stack([self.transform(x) for x in img],dim=0)
            if DRAW_BBOX_IMAGE:
                show_tensor_images(t_img.float()/255, cls_label)
            t_img= normalize_imagenet(t_img.float()/255.0)
            roi_images.append( t_img)
            
            #TODO: add temporal graph
            
        '''compute node description'''
        for i in range(len(filtered_instances)):
            instance_id = filtered_instances[i]
            obj = object_data[instance_id]
            # obj = objects[str(instance_id)]
            
            '''augmentation'''
            # random scale dim with up to 0.3
            if not self.for_eval and self.mconfig.bbox_aug_ratio>0:
                center = np.array(obj['center'])
                dim = np.array(obj['dimension'])
                
                max_ratio=self.mconfig.bbox_aug_ratio
                dim_scales = np.random.uniform(low=-max_ratio,high=max_ratio,size=3)
                reduce_amount = dim * dim_scales
                center += reduce_amount
                
                dim_scales = np.random.uniform(low=-max_ratio,high=max_ratio,size=3)
                reduce_amount = dim * dim_scales
                dim += reduce_amount
                obj['center'] = center.tolist()
                obj['dimension'] = dim.tolist()
            node_descriptor_for_image[i] = descriptor_generator(obj)
            
        return roi_images, node_descriptor_for_image
    
    def __load_full_images(self, scan_id, idx2oid:dict, cat:list, 
                           scan_data:dict, mv_data:dict):
        images=list()
        bounding_boxes = list() # bounding_boxes[node_id]{kf_id: [boxes]}
        bbox_cat=list()
        node_descriptor_for_image = list()
        image_edge_indices=list()
        img_idx2oid = dict() # from image object index to object isntance
        per_frame_info_dict=defaultdict(dict)
        
        '''alias'''
        object_data = scan_data['nodes']
        mv_nodes = mv_data['nodes']
        mv_kfs = mv_data['kfs']
        
        # 
        descriptor_generator = util_data.Node_Descriptor_24(with_bbox=self.cfg.data.img_desc_6_pts)
        
        '''collect frame idx'''
        fids = set()
        oid2cls = dict()
        for idx in range(len(cat)):
            oid = idx2oid[idx]
            oid2cls[oid] = cat[idx]
            mv_node = mv_nodes[str(oid)]
            cls_label = mv_node.attrs['label']
            if cls_label == 'unknown':
                cls_label = self.classNames[cat[idx]]
                
            kf_indices = np.asarray(mv_node)
            fids = fids.union(kf_indices)
        
        filtered_kf_indices = self.filtered_data[scan_id]
        fids = fids.intersection(filtered_kf_indices)
        
        # drop images for memory sack 
        fids = list(fids)
        if not self.for_eval:
            fids = random_drop(fids, self.mconfig.drop_img_edge, replace=True)
        else:
            fids = random_drop(fids, self.mconfig.drop_img_edge_eval)
        # print(len(fids))        
        
        '''load'''
        for mid, fid in enumerate(fids):
            '''read data'''
            kf = mv_kfs[str(fid)]
            kf_oid2idx = {v[0]:v[1] for v in kf.attrs['seg2idx']}
            
            '''get boxes of selected objects'''
            filtered_kf_oid2idx = dict()
            for k in kf_oid2idx:
                if k in idx2oid.values():
                    filtered_kf_oid2idx[k] = kf_oid2idx[k]
            if len(filtered_kf_oid2idx)==0:continue
            
            '''load image'''
            if self.for_eval:
                img_data = self.image_feature[scan_id][str(fid)]
                img_data = np.asarray(img_data).copy()
                img_data = torch.from_numpy(img_data)
            else:
                pth_rgb = os.path.join(define.DATA_PATH,scan_id,'sequence', define.RGB_NAME_FORMAT.format(int(fid)))
                img_data = Image.open(pth_rgb)
                img_data = np.rot90(img_data,3)# Rotate image
                img_data = torch.as_tensor(img_data.copy()).permute(2,0,1)
                img_data = self.transform(img_data)
                img_data= normalize_imagenet(img_data.float()/255.0)
            width,height = img_data.shape[-1],img_data.shape[-2]
            images.append(img_data)
            
            '''collecting info for temporal edge'''            
            per_frame_info = per_frame_info_dict[fid]
            per_frame_info['nodes'] = dict()
            per_frame_info['edges'] = dict()

            '''build image box and gt'''
            kfdata = np.asarray(kf)
            per_frame_indices = list()
            for kf_oid, kf_idx in filtered_kf_oid2idx.items():
                box = kfdata[kf_idx][:4] 
                box[0]/=width
                box[1]/=height
                box[2]/=width
                box[3]/=height
                box = np.concatenate(([mid], box)).tolist() # ROIAlign format
                
                om_id = len(bbox_cat)
                img_idx2oid[om_id] = kf_oid
                per_frame_indices.append(om_id)
                
                '''build input and gt'''
                bbox_cat.append(oid2cls[kf_oid])
                bounding_boxes.append(box)
                node_descriptor_for_image.append( descriptor_generator(object_data[kf_oid]) )
                
                # for temporal node edge
                assert kf_oid not in per_frame_info['nodes'] # per framae should have only one oid to one om_id
                per_frame_info['nodes'][kf_oid] = om_id
                
            '''build image edges'''
            for om_id1 in per_frame_indices:
                for om_id2 in per_frame_indices:
                    if om_id1 != om_id2:
                        em_id = len(image_edge_indices)
                        key_mapped = (img_idx2oid[om_id1],img_idx2oid[om_id2])
                        assert key_mapped not in per_frame_info['edges']
                        per_frame_info['edges'][key_mapped] = em_id
                        
                        key = (om_id1,om_id2)
                        image_edge_indices.append(key)
        
        '''build temporal node graph'''
        temporal_node_graph=list()
        temporal_edge_graph=list()
        sorted_kf_indices = sorted(fids)
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
                        
        # if DRAW_BBOX_IMAGE:
        # t_img = torch.stack(images,dim=0)
        # show_tensor_images(t_img.float()/255, '-')
            
        return images, bounding_boxes, bbox_cat, node_descriptor_for_image, \
            image_edge_indices, img_idx2oid, temporal_node_graph, temporal_edge_graph

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

def zero_mean(point, normalize:bool):
    mean = torch.mean(point, dim=0)
    point -= mean.unsqueeze(0)
    ''' without norm to 1  '''
    if normalize:
        furthest_distance = point.pow(2).sum(1).sqrt().max() # find maximum distance for each n -> [n]
        point /= furthest_distance
    return point

if __name__ == '__main__':
    import codeLib
    path = './experiments/config_2DSSG_ORBSLAM3_l20_6_1.yaml'
    config = codeLib.Config(path)
    
    config.DEVICE='1'
    # config.dataset.root = "../data/example_data/"    
    # config.dataset.label_file = 'inseg.ply'
    # sample_in_runtime = True
    # config.dataset.data_augmentation=True
    # split_type = 'validation_scans' # ['train_scans', 'validation_scans','test_scans']
    dataset = SGFNDataset (config, 'validation')
    items = dataset.__getitem__(0)    
    # print(items)