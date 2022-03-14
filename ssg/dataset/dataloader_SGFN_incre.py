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
from torchvision.utils import draw_bounding_boxes
from codeLib.common import color_rgb, rand_24_bit
from codeLib.torch.visualization import show_tv_grid
import matplotlib.pyplot as plt
from torchvision.ops import roi_align
import os,json,trimesh, argparse
import open3d as o3d
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ssg.utils import util_ply, util_label, util, util_3rscan, util_data
from ssg import define
from codeLib.utils.util import set_random_seed, read_txt_to_list

logger_py = logging.getLogger(__name__)

debug=True
debug=False
DRAW_BBOX_IMAGE=True
DRAW_BBOX_IMAGE=False
random_clr_i = [color_rgb(rand_24_bit()) for _ in range(1500)]
random_clr_i[0] = (0,0,0)
ffont = '/home/sc/research/PersistentSLAM/python/2DTSG/files/Raleway-Medium.ttf'
rgb_filepattern = 'frame-{0:06d}.color.jpg'
toTensor = transforms.ToTensor()
resize = transforms.Resize([256,256])


class SGFNIDataset (data.Dataset):
    def __init__(self,config,mode, **args):
        super().__init__()
        assert mode in ['train','validation','test']
        self._device = config.DEVICE
        path = config.data['path']
        self.config = config
        self.mconfig = config.data
        self.path = config.data.path
        self.label_file = config.data.label_file
        self.use_data_augmentation=self.mconfig.data_augmentation
        self.root_3rscan = define.DATA_PATH
        self.path_h5 = os.path.join(self.path,'relationships_%s.h5' % (mode))
        self.path_mv = os.path.join(self.path,'proposals.h5')
        self.path_roi_img = self.mconfig.roi_img_path
        self.pth_node_weights = os.path.join(self.path,'node_weights.txt')
        self.pth_edge_weights = os.path.join(self.path,'edge_weights.txt')
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
        self.sample_in_runtime= config.data.sample_in_runtime
        self.load_cache = False
        self.for_eval = mode != 'train'
        self.max_edges=config.data.max_num_edge
        self.full_edge = self.config.data.full_edge
        
        self.output_node = args.get('output_node', True)
        self.output_edge = args.get('output_edge', True)    

        ''' read classes '''
        pth_classes = os.path.join(path,'classes.txt')
        pth_relationships = os.path.join(path,'relationships.txt')      
        selected_scans = list()
        selected_scans += read_txt_to_list(os.path.join(path,'train_scans.txt'))
        selected_scans += read_txt_to_list(os.path.join(path,'validation_scans.txt'))
        selected_scans += read_txt_to_list(os.path.join(path,'test_scans.txt'))
        
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
        
        '''pack with snp'''
        self.size = len(selected_scans)
        self.scans = snp.pack(selected_scans)#[s for s in data.keys()]

    def open_mv_graph(self):
        if not hasattr(self, 'mv_data'):
            self.mv_data = h5py.File(self.path_mv,'r')
                
    def open_data(self):
        if not hasattr(self,'sg_data'):
            self.sg_data = h5py.File(self.path_h5,'r')
            
    def open_img(self):
        if not hasattr(self, 'roi_imgs'):
            self.roi_imgs = h5py.File(self.path_roi_img,'r')
           
    def __getitem__(self, index):
        scan_id = snp.unpack(self.scans,index)# self.scans[idx]
        # scan_id = '095821f7-e2c2-2de1-9568-b9ce59920e29'
        # self.open_data()
        # scan_data_raw = self.sg_data[scan_id]
        # scan_data = raw_to_data(scan_data_raw)
        
        # object_data = scan_data['nodes']
        # relationships_data = scan_data['relationships']        


        filtered_data = list()
        with open(os.path.join(define.FILE_PATH + "relationships.json"), "r") as read_file:
            data = json.load(read_file)
            for s in data["scans"]:
                if scan_id != s["scan"]: continue
                filtered_data.append(s)
                break
            gt_relationships = s["relationships"]
        ''''''''''''''''''''''''
        ''''''''''''''''''''''''
        # if self.mconfig.load_images:
        #     self.open_mv_graph()
        #     self.open_img()
        #     mv_data = self.mv_data[scan_id]
        #     mv_nodes = mv_data['nodes']
        #     roi_imgs = self.roi_imgs[scan_id]
            
        #     '''filter'''
        #     mv_node_ids = [int(x) for x in mv_data['nodes'].keys()]
            
        #     sg_node_ids = object_data.keys()                
        #     inter_node_ids = set(sg_node_ids).intersection(mv_node_ids)
            
        #     object_data = {nid: object_data[nid] for nid in inter_node_ids}
            
        
        fdata = os.path.join(define.DATA_PATH)
        # if self.mconfig.load_incremental:
        '''load'''
        with open(os.path.join(fdata,scan_id,'graph_2dssg_seq.json'),'r') as f:
            sdata = json.load(f)[scan_id]
            
            
        ''' load instance mapping'''
        _, label_name_mapping, _ = util_label.getLabelMapping('ScanNet20')
        segseg_file_name = 'semseg.v2.json'
        pth_semseg_file = os.path.join(define.DATA_PATH, scan_id, segseg_file_name)
        instance2labelName_ = util_3rscan.load_semseg(pth_semseg_file, label_name_mapping,True)
        
        min_3D_bbox_size=0.2*0.2*0.2
        occ_thres=0.5
        
        outputs = list()
        keys = [int(k) for k in sdata]
        keys = sorted(keys)
        for key in keys:
            data = sdata[str(key)]
            nodes = data['nodes']

            '''filter input and build mapping'''            
            map_segment_pd_2_gt=dict()
            gt_segments_2_pd_segments = dict() # how many segment_pd corresponding to this segment_gt
            segs_neighbors=dict()
            segments_pd_filtered=list()
            kf_ids_from_nodes = set()
            for seg_id in nodes.keys():
                node = nodes[seg_id]
                if len(node['kfs']) == 0:continue
                
                size = node['dimension'][0]*node['dimension'][1]*node['dimension'][2]
                if size <= min_3D_bbox_size :
                    # name = instance2labelName.get(seg_id,'unknown')
                    if debug: print('node',seg_id,'too small (', size,'<',min_3D_bbox_size,')')
                    continue
                
                '''find GT instance'''
                # get maximum
                max_v=0
                max_k=0
                for k,v in node['gtInstance'].items():
                    if v>max_v:
                        max_v=v
                        max_k=int(k)
                if max_v < occ_thres:
                    if debug: print('node',seg_id,'has too small overlappign to GT instance', max_v,'<',occ_thres)
                    continue
                
                '''skip nonknown'''
                if instance2labelName_[max_k] == '-' or instance2labelName_[max_k] =='none':
                    if debug: print('node',seg_id,'has unknown GT instance',max_k)
                    continue
                
                '''  '''
                map_segment_pd_2_gt[int(seg_id)] = int(max_k)
                
                if max_k not in gt_segments_2_pd_segments:
                    gt_segments_2_pd_segments[max_k] = list()
                gt_segments_2_pd_segments[max_k].append(seg_id)        
                
                segs_neighbors[int(seg_id)] = node['neighbors']
        
                segments_pd_filtered.append(seg_id)
                
                [kf_ids_from_nodes.add(x) for x in node['kfs']]
            if debug: print('map_segment_pd_2_gt.keys():',map_segment_pd_2_gt.keys())

            '''build kf ans nodes ampping'''
            kfs = dict()
            objects = dict()
            node2kfs = dict()
            min_size =60
            for kf_ in data['kfs']:
                bboxes = kf_['bboxes']
                # if len(bboxes) < min_obj: continue
                width = kf_['rgb_dims'][0]
                height = kf_['rgb_dims'][1]
                path = kf_['path']
                fname = os.path.basename(path)
                fid = int(''.join([x for x in fname if x.isdigit()]))
                if fid not in kf_ids_from_nodes: continue
                if str(fid) not in kfs: kfs[str(fid)] = dict()
                kf = kfs[str(fid)]
                kf['idx'] = fid
                kf['bboxes'] = dict()
                
                
                img = np.array(Image.open(path))
                img = np.rot90(img,3).copy()# Rotate image
                
                boxes=list()
                clrs =list()
                labelNames=list()
                
                scale = [kf_['rgb_dims'][0]/kf_['mask_dims'][0],kf_['rgb_dims'][1]/kf_['mask_dims'][1] ]
                for oid in bboxes:
                    if int(oid) == 0: continue
                    if str(oid ) not in segments_pd_filtered:continue
                    # print('oid',oid)
                    
                    '''scale bounding box back'''
                    box = bboxes[oid] # xmin,ymin,xmax,ymax
                    box[0] *= scale[0]
                    box[2] *= scale[0]
                    box[1] *= scale[1]
                    box[3] *= scale[1]
                    
                    '''Check width and height'''
                    w_ori = box[2]-box[0]
                    h_ori = box[3]-box[1]
                    if w_ori  < min_size or h_ori < min_size: continue
                
                    '''check format is correct'''
                    assert 0 <= box[0] < box[2]
                    assert 0 <= box[1] < box[3]
                    
                    # check boundary
                    box[0] = max(0,box[0])
                    box[1] = max(0,box[1])
                    box[2] = min(width, box[2])
                    box[3] = min(height, box[3])
                    
                    '''
                    The coordinate system is top-left corner (openCV). clockwise 90 degree with OpenGL(Pytorch) bottom left is equalivelent to counter clockwise 90.
                    ------->x
                    |
                    | 
                    v
                    y
                    
                    to
                    
                    x
                    ^
                    |
                    |
                    ------>y
                    
                    '''
                    # c.clockwise 90
                    box_r = [0,0,0,0]
                    box_r[0] = height-box[1]
                    box_r[1] = box[0]
                    box_r[2] = height-box[3]
                    box_r[3] = box[2]
                    
                    box[0] = min(box_r[0],box_r[2])
                    box[2] = max(box_r[0],box_r[2])
                    box[1] = min(box_r[1],box_r[3])
                    box[3] = max(box_r[1],box_r[3])
                    
                    boxes.append(box)
                    labelNames.append('unknown')
                    clrs.append((255,255,255))
                    
                    kf['bboxes'][oid] = box
                    
                    if str(oid) not in objects: objects[str(oid)] = dict()
                    obj = objects[str(oid)]
                    obj['label'] = 'unknown'
                    
                    if int(oid) not in node2kfs:
                        node2kfs[int(oid)]=list()
                    node2kfs[int(oid)].append(fid)
                    
                    # break
                # if DRAW_BBOX_IMAGE:
                #     torch_img = torch.from_numpy(img).permute(2,0,1)
                #     boxes = torch.tensor(boxes, dtype=torch.float)
                #     result = draw_bounding_boxes(torch_img, boxes, 
                #                                   labels=labelNames,
                #                                   colors=clrs, 
                #                                   width=5,
                #                                   font=ffont,
                #                                   font_size=50)
                #     show_tv_grid(result)
                #     plt.show()

            '''check if each node has at least one kf'''
            to_deletes = []
            for k,v in objects.items():
                if int(k) not in node2kfs or len(node2kfs[int(k)])==0 or k not in segments_pd_filtered:
                    to_deletes.append(k)
            for idx in to_deletes:
                objects.pop(idx)
            
            for oid, obj in objects.items():
                obj['label'] = instance2labelName_[map_segment_pd_2_gt[int(oid)]]
            
            ''' build mapping '''
            instance2labelName  = { int(key): node['label'] for key,node in objects.items()  }
            # print('instance2labelName',instance2labelName)
                
            '''check if empty'''
            if len(objects) == 0:
                    # invalid_scans.append(scan_id)
                continue
            instances_ids = [int(k) for k in objects.keys()]
            

            ''' 
            Find instances we care abot. Build oid2idx and cat list
            oid2idx maps instances to a mask id. to randomize the order of instance in training.
            '''
            oid2idx = {}
            idx2oid = {}
            cat = []
            counter = 0
            filtered_instances = list()
            # kf_indices = set()
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
                    
                    # [kf_indices.add(x) for x in node2kfs[instance_id]]
            if len(cat) == 0:
                # logger_py.debug('filtered_nodes: {}'.format(filtered_nodes))
                logger_py.debug('cat: {}'.format(cat))
                logger_py.debug('self.classNames: {}'.format(self.classNames))
                # logger_py.debug('list(object_data.keys()): {}'.format(list(object_data.keys())))
                # logger_py.debug('nn: {}'.format(nns))
                assert len(cat) > 0
            ''' build nn dict '''
            nns = dict()
            seg2inst = dict()
            for oid in filtered_instances:
                nns[str(oid)] = [int(s) for s in nodes[str(oid)]['neighbors']]
                
                '''build instance dict'''
                seg2inst[oid] = map_segment_pd_2_gt[oid]
                # if 'instance_id' in odata:
                #     seg2inst[oid] = odata['instance_id']
                
                
            bounding_boxes = list()
            for idx, oid in idx2oid.items():
                kf_indices = node2kfs[int(oid)]
                img_boxes = list()
                fidx2idx=list()
                # poses=list()
                counter=0
                for fid in kf_indices:
                    pth_rgb = os.path.join(fdata,scan_id,'sequence', rgb_filepattern.format(fid))
                    # pth_pos = os.path.join(fdata,scan_id,'sequence', pose_filepattern.format(fid))
                    '''load data'''
                    img_data = Image.open(pth_rgb)
                    img_data = np.rot90(img_data,3)# Rotate image
                    # pos_data = np.loadtxt(pth_pos)
                    
                    # bfid = imgs['indices'][fid] # convert frame idx to the buffer idx 
                    # pose = torch.from_numpy(pos_data)
                    
                    # if pose.isnan().any() or pose.isinf().any(): continue
                    
                    # if is_close(pose, poses):
                        # continue
                    # poses.append(pose)
                    
                    
                    #  bounding box
                    kf = kfs[str(fid)]                
                    # kf_seg2idx = {v[0]:v[1] for v in kf.attrs['seg2idx']}
                    # bid = kf_seg2idx[int(oid)]
                    # kfdata = np.asarray(kf)
                    # try:
                    box = np.asarray(kf['bboxes'][str(oid)])[:4]# kfdata[bid,:4]
                    # except:
                    #     print(oid)
                    # oc  = kfdata[bid,4]
                    # print(oc)
                    box = torch.from_numpy(box).float().view(1,-1)
                    timg = toTensor(img_data.copy()).unsqueeze(0)
                    w = box[:,2] - box[:,0]
                    h = box[:,3] - box[:,1]
                    # if n_workers==0: logger_py.info('box: {}, dim: {}'.format(box,[h,w]))
                    region = roi_align(timg,[box], [h,w])
                    region = resize(region).squeeze(0)
                    img_boxes.append( region )
                    fidx2idx.append( (fid, counter) )
                    counter+=1
                    # plt.imshow(region.permute(1,2,0))
                    # plt.show()
                if len(img_boxes)==0: 
                    raise RuntimeError("scan:{}.node_id:{} has 0 img boxes".format(scan_id,oid))
                img_boxes = torch.stack(img_boxes)
                
                img_boxes = torch.clamp((img_boxes*255).byte(),0,255).byte()
                img_boxes = torch.stack([self.transform(x) for x in img_boxes],dim=0)
                if DRAW_BBOX_IMAGE:
                    show_tensor_images(img_boxes, title=objects[str(oid)]['label'])
                img_boxes= normalize_imagenet(img_boxes.float()/255.0)
                bounding_boxes.append(img_boxes)
                # h5d = h5f.create_dataset(oid,data=img_boxes.numpy(), compression="gzip", compression_opts=9)
                # h5d.attrs['seg2idx'] = fidx2idx          
            
            descriptor_generator = util_data.Node_Descriptor_24(with_bbox=self.mconfig.img_desc_6_pts)
            node_descriptor_for_image = torch.zeros([len(cat), len(descriptor_generator)])
            for i in range(len(filtered_instances)):
                instance_id = filtered_instances[i]
                obj = nodes[str(instance_id)]
                obj['normAxes'] =  copy.deepcopy( np.array(obj['rotation']).reshape(3,3).transpose().tolist() )
                node_descriptor_for_image[i] = descriptor_generator(obj)
            
        
            ''' Build rel class GT '''
            if self.multi_rel_outputs:
                adj_matrix_onehot = np.zeros([len(cat), len(cat), len(self.relationNames)])
            else:
                adj_matrix = np.ones([len(cat), len(cat)]) * self.none_idx#set all to none label.
                # adj_matrix += self.none_idx 
            
            # '''sample connections'''
            # if self.sample_in_runtime:
            #     if not self.for_eval:
            #         edge_indices = util_data.build_edge_from_selection_sgfn(filtered_instances,nns,max_edges_per_node=-1)
            #         edge_indices = [[oid2idx[edge[0]],oid2idx[edge[1]]] for edge in edge_indices ]
            #         # edge_indices = util_data.build_edge_from_selection(filtered_nodes, nns, max_edges_per_node=-1)
            #     else:
            #         edge_indices = list()
            #         for n in range(len(cat)):
            #             for m in range(len(cat)):
            #                 if n == m:continue
            #                 edge_indices.append([n,m])
                            
            #     if len(edge_indices)>0:
            #         if not self.for_eval:
            #             edge_indices = random_drop(edge_indices, self.mconfig.drop_edge)       
            #         if self.for_eval :
            #             edge_indices = random_drop(edge_indices, self.mconfig.drop_edge_eval)
                        
            #         if self.mconfig.max_num_edge > 0 and len(edge_indices) > self.mconfig.max_num_edge:
            #             choices = np.random.choice(range(len(edge_indices)),self.mconfig.max_num_edge,replace=False).tolist()
            #             edge_indices = [edge_indices[t] for t in choices]
            # else:
            #     edge_indices = list()
            #     max_edges=-1
            #     for n in range(len(cat)):
            #         for m in range(len(cat)):
            #             if n == m:continue
            #             edge_indices.append([n,m])
            #     if max_edges>0 and len(edge_indices) > max_edges and not self.for_eval: 
            #         # for eval, do not drop out any edges.
            #         indices = list(np.random.choice(len(edge_indices),max_edges,replace=False))
            #         edge_indices = edge_indices[indices]
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
            edge_indices = [[l[0],l[1]] for l in edge_indices]
            
            '''generate relationships '''    
            relationships = gen_relationship(scan_id, 0, gt_relationships, map_segment_pd_2_gt, instance2labelName_, gt_segments_2_pd_segments)
            
            rel_json = relationships['relationships']
            for r in rel_json:
                r_src = int(r[0])
                r_tgt = int(r[1])
                r_lid = int(r[2])
                r_cls = r[3]
                
                if r_src not in oid2idx or r_tgt not in oid2idx: continue
                index1 = oid2idx[r_src]
                index2 = oid2idx[r_tgt]
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
            
            ''' to tensor '''
            gt_class = torch.from_numpy(np.array(cat))
            edge_indices = torch.tensor(edge_indices,dtype=torch.long)
            
            
        
            
        
        
        
            output = dict()
            output['fid'] = key
            output['scan_id'] = scan_id # str
            output['gt_rel'] = gt_rels  # tensor
            output['gt_cls'] = gt_class # tensor
            if self.mconfig.load_images:
                output['roi_imgs'] = bounding_boxes #list
                output['node_descriptor_8'] = node_descriptor_for_image
            output['node_edges'] = edge_indices # tensor
            # output['instance2mask'] = oid2idx #dict
            output['mask2instance'] = idx2oid
            output['seg2inst'] = seg2inst
            outputs.append(output)
            # break
        # del self.sg_data
        # if self.mconfig.load_images:
            # del self.roi_imgs
            # del self.mv_data
        return outputs
        
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

def gen_relationship(scan_id:str,split:int,gt_relationships, map_segment_pd_2_gt:dict,instance2labelName:dict,gt_segments_2_pd_segments:dict,
                     target_segments:list=None) -> dict:
    '''' Save as relationship_*.json '''
    relationships = dict() #relationships_new["scans"].append(s)
    relationships["scan"] = scan_id
    relationships["split"] = split
    
    objects = dict()
    for seg, segment_gt in map_segment_pd_2_gt.items():
        if target_segments is not None:
            if seg not in target_segments: continue
        name = instance2labelName[segment_gt]
        assert(name != '-' and name != 'none')
        objects[int(seg)] = dict()
        objects[int(seg)]['label'] = name
        objects[int(seg)]['instance_id'] = segment_gt
    if debug: print(objects)
    relationships["objects"] = objects
    
    target_relationships = ['supported by', 'attached to','standing on','hanging on','connected to','part of','build in']
    target_relationships.append(define.NAME_SAME_PART)
    
    split_relationships = list()
    ''' Inherit relationships from ground truth segments '''
    if gt_relationships is not None:
        relationships_names = read_txt_to_list(os.path.join(define.FILE_PATH, "relationships.txt"))

        for rel in gt_relationships:
            id_src = rel[0]
            id_tar = rel[1]
            num = rel[2]
            name = rel[3]
            idx_in_txt = relationships_names.index(name)
            assert(num==idx_in_txt)
            if name not in target_relationships: 
                continue
            if id_src == id_tar:
                if debug: print('ignore relationship (',name,') between',id_src,'and',id_tar,'that has the same source and target')
                continue
            idx_in_txt_new = target_relationships.index(name)
            
            if id_src in gt_segments_2_pd_segments and id_tar in gt_segments_2_pd_segments:
                segments_src = gt_segments_2_pd_segments[id_src]
                segments_tar = gt_segments_2_pd_segments[id_tar]
                for segment_src in segments_src:
                    if int(segment_src) not in objects:
                        if debug: print('filter',name,'segment_src', instance2labelName[id_src],' is not in objects')
                        continue
                    for segment_tar in segments_tar:        
                        if int(segment_tar) not in objects:
                            if debug: print('filter',name,'segment_tar', instance2labelName[id_tar], ' is not in objects')
                            continue
                        if target_segments is not None:
                            ''' skip if they not in the target_segments'''
                            if segment_src not in target_segments: continue
                            if segment_tar not in target_segments: continue
                        # if segment_tar == segments_src:continue
                        ''' check if they are neighbors '''
                        split_relationships.append([ int(segment_src), int(segment_tar), idx_in_txt_new, name ])
                        if debug: print('inherit', instance2labelName[id_src],'(',id_src,')',name, instance2labelName[id_tar],'(',id_tar,')')
            # else:
            #     if debug:
            #         if id_src in gt_segments_2_pd_segments:
            #             print('filter', instance2labelName[id_src],name, instance2labelName[id_tar],'id_src', id_src, 'is not in the gt_segments_2_pd_segments')
            #         if id_tar in gt_segments_2_pd_segments:
            #             print('filter', instance2labelName[id_src],name, instance2labelName[id_tar],'id_tar', id_tar, 'is not in the gt_segments_2_pd_segments')
    
    ''' Build "same part" relationship '''
    idx_in_txt_new = target_relationships.index(define.NAME_SAME_PART)
    for _, groups in gt_segments_2_pd_segments.items():
        if target_segments is not None:
            filtered_groups = list()
            for g in groups:
                if g in target_segments:
                    filtered_groups.append(g)
            groups = filtered_groups
        if len(groups) <= 1: continue
                    
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                split_relationships.append([int(groups[i]),int(groups[j]), idx_in_txt_new, define.NAME_SAME_PART])
                split_relationships.append([int(groups[j]),int(groups[i]), idx_in_txt_new, define.NAME_SAME_PART])
    
    relationships["relationships"] = split_relationships
    if debug: print(split_relationships)
    return relationships

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
    config.DEVICE='cpu'
    # config.dataset.root = "../data/example_data/"    
    # config.dataset.label_file = 'inseg.ply'
    # sample_in_runtime = True
    # config.dataset.data_augmentation=True
    # split_type = 'validation_scans' # ['train_scans', 'validation_scans','test_scans']
    dataset = SGFNIDataset (config, 'test')
    items = dataset.__getitem__(0)    
    # print(items)