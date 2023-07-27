from collections import defaultdict
import torch.utils.data as data
import os
import random
import torch
import json
import trimesh
import h5py
import copy
import numpy as np
import multiprocessing as mp

# from utils import util_ply, util_data, util, define
from codeLib.common import random_drop
from codeLib import transformation
from ssg.utils import util_ply, util_data
from codeLib.utils.util import read_txt_to_list, check_file_exist
from ssg import define
# from codeLib.torch.visualization import show_tensor_images
from codeLib.common import normalize_imagenet
from torchvision import transforms
import codeLib.torchvision.transforms as cltransform
import ssg.utils.compute_weight as compute_weight
from ssg.utils.util_data import raw_to_data, data_to_raw, cvt_all_to_dict_from_h5
import codeLib.utils.string_numpy as snp
import logging
from PIL import Image
from codeLib.common import run
from pytictoc import TicToc
from torch_geometric.data import HeteroData
from torchvision.ops import roi_align

logger_py = logging.getLogger(__name__)

DRAW_BBOX_IMAGE = True
DRAW_BBOX_IMAGE = False


class Dataset (data.Dataset):
    def __init__(self, config, mode, **args):
        super().__init__()
        assert mode in ['train', 'validation', 'test']
        self.mode = mode
        self._device = config.DEVICE
        # path = config.data['path_seq']
        self.config = config
        self.cfg = self.config
        self.mconfig = config.data
        self.path = config.data.path_seq
        self.label_file = config.data.label_file_seq
        self.use_data_augmentation = self.mconfig.data_augmentation
        self.root_3rscan = config.data.path_3rscan
        self.path_h5 = os.path.join(self.path, 'relationships_%s.h5' % (mode))
        self.path_mv = os.path.join(self.path, 'proposals.h5')
        self.path_roi_img = self.mconfig.roi_img_path
        self.pth_filtered = os.path.join(
            self.path, 'filtered_scans_detection_%s.h5' % (mode))
        self.pth_node_weights = os.path.join(self.path, 'node_weights.txt')
        self.pth_edge_weights = os.path.join(self.path, 'edge_weights.txt')
        try:
            self.root_scannet = config.data.path_scannet
        except:
            self.root_scannet = None

        # SEGMENT_TYPE='GT'
        with open(os.path.join(self.cfg.data.path, 'args.json')) as f:
            tmp = json.load(f)
            label_type = tmp['label_type']
            segment_type = tmp['segment_type']
        image_feature_folder_name = define.NAME_IMAGE_FEAUTRE_FORMAT.format(
            segment_type, label_type)
        self.path_img_feature = os.path.join(
            self.cfg.data.path_image_feature, image_feature_folder_name+'.h5')

        # selected_scans = set()
        self.w_cls_obj = self.w_cls_rel = None
        self.multi_rel_outputs = multi_rel_outputs = config.model.multi_rel
        self.shuffle_objs = False
        self.use_rgb = config.model.use_rgb
        self.use_normal = config.model.use_normal
        self.sample_in_runtime = config.data.sample_in_runtime
        self.load_cache = False
        self.for_eval = mode != 'train'
        self.max_edges = config.data.max_num_edge
        self.full_edge = self.multi_rel_outputs  # self.config.data.full_edge

        self.output_node = args.get('output_node', True)
        self.output_edge = args.get('output_edge', True)

        ''' read classes '''
        pth_classes = os.path.join(self.path, 'classes.txt')
        pth_relationships = os.path.join(self.path, 'relationships.txt')
        # selected_scans = read_txt_to_list(os.path.join(path,'%s_scans.txt' % (mode)))

        names_classes = read_txt_to_list(pth_classes)
        names_relationships = read_txt_to_list(pth_relationships)

        if not multi_rel_outputs:
            if define.NAME_NONE not in names_relationships:
                names_relationships.append(define.NAME_NONE)
        elif define.NAME_NONE in names_relationships:
            names_relationships.remove(define.NAME_NONE)

        self.relationNames = sorted(names_relationships)
        self.classNames = sorted(names_classes)
        self.none_idx = self.relationNames.index(
            define.NAME_NONE) if not multi_rel_outputs else None

        '''set transform'''
        if self.mconfig.load_images:
            if self.mconfig.is_roi_img:
                if not self.for_eval:
                    self.transform = transforms.Compose([
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
                            cltransform.TrivialAugmentWide(),
                        ])
                    else:
                        self.transform = transforms.Compose([
                            cltransform.TrivialAugmentWide(),
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

        '''check if pre-computed global image featur exist'''
        # if self.for_eval and not self.mconfig.is_roi_img and self.mconfig.load_images: # train mode can't use precmopute feature. need to do img. aug.
        # if not self.mconfig.is_roi_img and self.mconfig.load_images and self.cfg.data.use_precompute_img_feature: # loading and memory issue. try to use precomputed
        # self.open_filtered()
        # should_compute_image_feature=False
        # if not os.path.exists(self.path_img_feature):
        #     should_compute_image_feature=True
        # else:
        #     feature_type = self.cfg.model.image_encoder.backend
        #     self.open_image_feature()
        #     image_feature = self.image_feature[feature_type]
        #     for scan_id in self.filtered_data:
        #         # Check scan exist
        #         if scan_id not in image_feature:
        #             should_compute_image_feature=True
        #         else:
        #             # check image exist
        #             filtered_data = raw_to_data(self.filtered_data[scan_id])[define.NAME_FILTERED_KF_INDICES]

        #             # try to open it
        #             try:
        #                 for kfId in filtered_data:
        #                     if str(kfId) not in image_feature[scan_id]:
        #                         should_compute_image_feature=True
        #                         break
        #             except:
        #                 should_compute_image_feature=True

        #         if should_compute_image_feature: break
        # if should_compute_image_feature:
        #     # Try to generate
        #     os.environ['MKL_THREADING_LAYER'] = 'GNU'
        #     # os.environ['PYTHONPATH'] = config.PYTHONPATH
        #     # subprocess.call(["export PYTHONPATH={}".format(PYTHONPATH)], shell=True)
        #     mode_ = 'eval' if mode == 'test' else mode
        #     bashCommand=[
        #         'python','ssg/utils/compute_image_feature.py',
        #         "--config",config.config_path,
        #         '-n',image_feature_folder_name,
        #         "-o",self.cfg.data.path_image_feature,
        #         "--mode",mode_,
        #     ]
        #     run(bashCommand)
        #     if not os.path.exists(self.path_img_feature):
        #         raise RuntimeError('use precompute image feature is true but file not found.')

        if not self.for_eval:
            w_node_cls = np.loadtxt(self.pth_node_weights)
            w_edge_cls = np.loadtxt(self.pth_edge_weights)
            self.w_node_cls = torch.from_numpy(w_node_cls).float()
            self.w_edge_cls = torch.from_numpy(w_edge_cls).float()

        '''cache'''
        self.cache_data = dict()
        # if self.load_cache:
        #     print('load data to cache')
        #     pool = mp.Pool(8)
        #     pool.daemon = True

        #     for scan_id in self.filtered_data:
        #         scan_id_no_split = scan_id.rsplit('_',1)[0]
        #         if 'scene' in scan_id:
        #             path = os.path.join(self.root_scannet, scan_id_no_split)
        #         else:
        #             path = os.path.join(self.root_3rscan, scan_id_no_split)
        #         if scan_id_no_split not in self.cache_data:
        #             self.cache_data[scan_id_no_split] = pool.apply_async(load_mesh,
        #                                                                   (path, self.label_file,self.use_rgb,self.use_normal))
        #     pool.close()
        #     pool.join()
        #     for key, item in self.cache_data.items():
        #         self.cache_data[key] = item.get()

        # del self.sg_data
        del self.filtered_data
        # if self.mconfig.load_images:
        # del self.roi_imgs
        # del self.mv_data

    def open_filtered(self):
        self.filtered_data = h5py.File(self.pth_filtered, 'r')

    def open_mv_graph(self):
        if not hasattr(self, 'mv_data'):
            self.mv_data = h5py.File(self.path_mv, 'r')

    def open_data(self):
        if not hasattr(self, 'sg_data'):
            self.sg_data = h5py.File(self.path_h5, 'r')

    def open_img(self):
        if not hasattr(self, 'roi_imgs'):
            self.roi_imgs = h5py.File(self.path_roi_img, 'r')

    def open_image_feature(self):
        if not hasattr(self, 'image_feature'):
            self.image_feature = h5py.File(self.path_img_feature, 'r')

    def __getitem__(self, index):
        timers = dict()
        timer = TicToc()
        scan_id = snp.unpack(self.scans, index)  # self.scans[idx]

        '''open data'''
        timer.tic()
        # open
        self.open_filtered()
        self.open_data()
        self.open_mv_graph()

        # get data
        scan_data_raw = self.sg_data[scan_id]
        scan_data_seq = raw_to_data(scan_data_raw)

        filtered_data_seq = raw_to_data(self.filtered_data[scan_id])

        mv_data_seq = self.mv_data[scan_id]

        '''process each timestamp'''
        output_seq = defaultdict(HeteroData)

        sorted_keys = [str(k) for k in sorted([int(k)
                                               for k in filtered_data_seq])]
        # sorted_keys = sorted_keys[-1:] #TODO: comment me after debug
        for timestamp in sorted_keys:
            scan_data = scan_data_seq[timestamp]
            filtered_data = filtered_data_seq[timestamp]
            mv_data = mv_data_seq[timestamp]
            # shortcut
            object_data = scan_data['nodes']
            relationships_data = scan_data['relationships']
            filtered_node_indices = filtered_data[define.NAME_FILTERED_OBJ_INDICES]
            filtered_kf_indices = filtered_data[define.NAME_FILTERED_KF_INDICES]

            if self.mconfig.load_images:
                self.open_mv_graph()

                # mv_data = self.mv_data[scan_id]
                mv_nodes = mv_data['nodes']  # contain kf ids of a given node
                if self.mconfig.is_roi_img:
                    self.open_img()
                    roi_imgs = self.roi_imgs[scan_id]

            '''filter node data'''
            object_data = {nid: object_data[nid]
                           for nid in filtered_node_indices}

            timers['open_data'] = timer.tocvalue()

            ''' build nn dict '''
            timer.tic()
            nns = dict()
            seg2inst = dict()
            for oid, odata in object_data.items():
                nns[str(oid)] = [int(s) for s in odata['neighbors']]

                '''build instance dict'''
                if 'instance_id' in odata:
                    seg2inst[oid] = odata['instance_id']
            timers['build_nn_dict'] = timer.tocvalue()

            ''' load point cloud data '''
            timer.tic()
            if self.mconfig.load_points:
                if 'scene' in scan_id:
                    path = os.path.join(self.root_scannet, scan_id)
                else:
                    path = os.path.join(self.root_3rscan, scan_id)

                if self.load_cache:
                    data = self.cache_data[scan_id]
                else:
                    data = load_mesh(
                        path, self.label_file+'_{}.ply'.format(timestamp), self.use_rgb, self.use_normal)
                points = copy.deepcopy(data['points'])
                instances = copy.deepcopy(data['instances'])

                if self.use_data_augmentation and not self.for_eval:
                    points = self.data_augmentation(points)
            timers['load_pc'] = timer.tocvalue()

            '''extract 3D node classes and instances'''
            timer.tic()
            cat, oid2idx, idx2oid, filtered_instances = self.__sample_3D_nodes(object_data,
                                                                               mv_data,
                                                                               nns)
            timers['sample_3D_nodes'] = timer.tocvalue()

            '''sample 3D node connections'''
            timer.tic()
            edge_indices_3D = self.__sample_3D_node_edges(
                cat, oid2idx, filtered_instances, nns)
            timers['sample_3D_node_edges'] = timer.tocvalue()

            '''extract relationships data'''
            timer.tic()
            relationships_3D = self.__extract_relationship_data(
                relationships_data)
            timers['extract_relationship_data'] = timer.tocvalue()

            ''' 
            Generate mapping from selected entity buffer to the ground truth entity buffer (for evaluation)
            Save the mapping in edge_index format to allow PYG to rearrange them.
            '''
            instance2labelName = {
                int(key): node['label'] for key, node in object_data.items()}
            # Collect GT entity list
            gt_entities = set()
            gtIdx_entities_cls = []
            gtIdx2ebIdx = []
            for key, value in relationships_3D.items():
                sub_o = key[0]
                tgt_o = key[1]
                gt_entities.add(sub_o)
                gt_entities.add(tgt_o)
            gt_entities = [k for k in gt_entities]
            # assert len(gt_entities) > 0
            for gtIdx, k in enumerate(gt_entities):
                if k in oid2idx:
                    idx = oid2idx[k]
                    gtIdx2ebIdx.append([gtIdx, idx])
                    label = instance2labelName[k]
                    gtIdx_entities_cls.append(self.classNames.index(label))
                else:
                    # Add negative index to indicate missing
                    gtIdx2ebIdx.append([gtIdx, -1])

            gtIdx_edge_index = []
            gtIdx_edge_cls = []
            for key, value in relationships_3D.items():
                sub_o = key[0]
                tgt_o = key[1]
                # sub_cls = instance2labelName[sub_o]
                # tgt_cls = instance2labelName[tgt_o]
                # sub_cls_id = self.classNames.index(sub_cls)
                # tgt_cls_id = self.classNames.index(tgt_cls)
                # relationships_3D_mask.append([sub_o,tgt_o,sub_cls_id,tgt_cls_id,value])

                sub_ebIdx = oid2idx[sub_o]
                tgt_ebIdx = oid2idx[tgt_o]
                sub_gtIdx = gt_entities.index(sub_o)
                tgt_gtIdx = gt_entities.index(tgt_o)
                gtIdx_edge_index.append([sub_gtIdx, tgt_gtIdx])
                gtIdx_edge_cls.append(value)

            # gtIdx_entities_cls = torch.from_numpy(np.array(gtIdx_entities_cls))
            gtIdx2ebIdx = torch.tensor(
                gtIdx2ebIdx, dtype=torch.long).t().contiguous()
            # gtIdx_edge_cls = torch.from_numpy(np.array(gtIdx_edge_cls))
            gtIdx_edge_index = torch.tensor(
                gtIdx_edge_index, dtype=torch.long).t().contiguous()

            '''sample 3D edges'''
            timer.tic()
            gt_rels_3D, edge_index_has_gt_3D = self.__sample_relationships(
                relationships_3D, idx2oid, edge_indices_3D)
            timers['sample_relationships'] = timer.tocvalue()

            '''drop edges'''  # to fit memory
            gt_rels_3D, edge_indices_3D = self.__drop_edge(
                gt_rels_3D, edge_indices_3D, edge_index_has_gt_3D)

            ''' random sample points '''
            if self.mconfig.load_points:
                timer.tic()
                obj_points, descriptor, bboxes = self.__sample_points(
                    scan_id, points, instances, cat, filtered_instances)
                timers['sample_points'] = timer.tocvalue()

                '''build rel points'''
                timer.tic()
                if self.mconfig.rel_data_type == 'points':
                    rel_points = self.__sample_rel_points(
                        points, instances, idx2oid, bboxes, edge_indices_3D)
                timers['sample_rel_points'] = timer.tocvalue()

            '''load images'''
            if self.mconfig.load_images:
                timer.tic()

                '''extract roi images'''
                mv_nodes = mv_data['nodes']
                mv_kfs = mv_data['kfs']

                roi_images, node_descriptor_for_image, edge_indices_img_to_obj = \
                    self.__crop_roi_images(scan_id,
                                           cat,
                                           idx2oid,
                                           mv_nodes,
                                           mv_kfs,
                                           filtered_instances,
                                           filtered_kf_indices,
                                           object_data)

                # descriptor_generator = util_data.Node_Descriptor_24(with_bbox=self.mconfig.img_desc_6_pts)
                # node_descriptor_for_image = torch.zeros([len(cat), len(descriptor_generator)])

                # roi_images = list()
                # edge_indices_img_to_obj=[]
                # for idx in range(len(cat)):
                #     oid = str(idx2oid[idx])
                #     node = mv_nodes[oid]
                #     kf_indices = np.asarray(node)

                #     for fid in kf_indices:
                #         pth_rgb = os.path.join(self.cfg.data.path_3rscan,scan_id,'sequence', define.RGB_NAME_FORMAT.format(int(fid)))

                #         '''load data'''
                #         img_data = Image.open(pth_rgb)
                #         img_data = np.rot90(img_data,3)# Rotate image

                #         height,width = img_data.shape[0],img_data.shape[1]
                #         # bfid = imgs['indices'][fid] # convert frame idx to the buffer idx

                #         #  bounding box
                #         kf = mv_kfs[str(fid)]
                #         kf_seg2idx = {v[0]:v[1] for v in kf.attrs['seg2idx']}
                #         bid = kf_seg2idx[int(oid)]
                #         kfdata = np.asarray(kf)
                #         box = kfdata[bid,:4]
                #         oc  = kfdata[bid,4]

                #         assert box[0]<=1
                #         assert box[0]>=0
                #         assert box[1]<=1
                #         assert box[1]>=0
                #         assert box[2]<=1
                #         assert box[2]>=0
                #         assert box[3]<=1
                #         assert box[3]>=0

                #         # Denormalize
                #         box[0] *= width
                #         box[2] *= width
                #         box[1] *= height
                #         box[3] *= height

                #         box = torch.from_numpy(box).float().view(1,-1)
                #         timg = transforms.ToTensor()(img_data.copy()).unsqueeze(0)
                #         w = box[:,2] - box[:,0]
                #         h = box[:,3] - box[:,1]
                #         # if n_workers==0: logger_py.info('box: {}, dim: {}'.format(box,[h,w]))
                #         region = roi_align(timg,[box], [h,w])
                #         region = self.transform(region).squeeze(0)

                #         roi_images.append(region)
                #         edge_indices_img_to_obj.append([len(roi_images)-1, idx])

                # roi_images = torch.stack(roi_images,dim=0)
                # roi_images= normalize_imagenet(roi_images.float()/255.0)

                # edge_indices_img_to_obj = torch.LongTensor(edge_indices_img_to_obj).t().contiguous()

                # '''compute node description'''
                # for i in range(len(filtered_instances)):
                #     instance_id = filtered_instances[i]
                #     obj = object_data[instance_id]
                #     # obj = objects[str(instance_id)]

                #     '''augmentation'''
                #     # random scale dim with up to 0.3
                #     if not self.for_eval and self.mconfig.bbox_aug_ratio>0:
                #         center = np.array(obj['center'])
                #         dim = np.array(obj['dimension'])

                #         max_ratio=self.mconfig.bbox_aug_ratio
                #         dim_scales = np.random.uniform(low=-max_ratio,high=max_ratio,size=3)
                #         reduce_amount = dim * dim_scales
                #         center += reduce_amount

                #         dim_scales = np.random.uniform(low=-max_ratio,high=max_ratio,size=3)
                #         reduce_amount = dim * dim_scales
                #         dim += reduce_amount
                #         obj['center'] = center.tolist()
                #         obj['dimension'] = dim.tolist()
                #     node_descriptor_for_image[i] = descriptor_generator(obj)

                # '''extract information'''
                # roi_images, node_descriptor_for_image, edge_indices_img_to_obj = \
                #     self.__load_roi_images(cat,idx2oid,mv_nodes,roi_imgs,
                #                         object_data,filtered_instances)
                # else:
                #     images, img_bounding_boxes, bbox_cat, node_descriptor_for_image, \
                #         image_edge_indices, img_idx2oid, temporal_node_graph, temporal_edge_graph = \
                #             self.__load_full_images(scan_id,idx2oid,cat,scan_data,mv_data, filtered_kf_indices)
                #     relationships_img = self.__extract_relationship_data(relationships_data)
                #     gt_rels_2D, edge_index_has_gt_2D = self.__sample_relationships(relationships_img,img_idx2oid,image_edge_indices)

                #     '''to tensor'''
                #     assert len(img_bounding_boxes) > 0
                #     images = torch.stack(images,dim=0)
                #     assert len(bbox_cat) == len(img_bounding_boxes)
                #     img_bounding_boxes = torch.from_numpy(np.array(img_bounding_boxes)).float()
                #     gt_class_image = torch.from_numpy(np.array(bbox_cat))
                #     image_edge_indices = torch.tensor(image_edge_indices,dtype=torch.long)
                #     temporal_node_graph = torch.tensor(temporal_node_graph,dtype=torch.long)
                #     temporal_edge_graph = torch.tensor(temporal_edge_graph,dtype=torch.long)
                #     if len(node_descriptor_for_image)>0:
                #         node_descriptor_for_image = torch.stack(node_descriptor_for_image)
                #     else:
                #         node_descriptor_for_image = torch.tensor([],dtype=torch.long)
                timers['load_images'] = timer.tocvalue()

            '''collect attribute for nodes'''
            inst_indices = [seg2inst[k] for k in idx2oid.values(
            )]  # for inseg the segment instance should be converted back to the GT instances

            ''' to tensor '''
            gt_class_3D = torch.from_numpy(np.array(cat))
            edge_indices_3D = torch.tensor(edge_indices_3D, dtype=torch.long) if len(
                edge_indices_3D) > 0 else torch.zeros([0, 2], dtype=torch.long)
            tensor_oid = torch.from_numpy(np.array(inst_indices))
            # idx2iid = seg2inst

            # output = dict()
            output = output_seq[timestamp]
            output['scan_id'] = scan_id  # str
            output['timestamp'] = timestamp

            output['node'].x = torch.zeros([gt_class_3D.shape[0], 1])  # dummy
            output['node'].y = gt_class_3D
            output['node'].oid = tensor_oid

            output['node_gt'].x = torch.zeros(
                [len(gtIdx_entities_cls), 1])  # dummy
            output['node_gt'].clsIdx = gtIdx_entities_cls
            output['node_gt', 'to', 'node'].edge_index = gtIdx2ebIdx
            output['node_gt', 'to', 'node_gt'].clsIdx = gtIdx_edge_cls
            output['node_gt', 'to', 'node_gt'].edge_index = gtIdx_edge_index

            output['node', 'to', 'node'].edge_index = edge_indices_3D.t().contiguous()
            output['node', 'to', 'node'].y = gt_rels_3D

            # output['node'].x = torch.zeros([gt_class_3D.shape[0],1]) # dummy
            # output['edge'].x = torch.zeros([gt_rels_3D.shape[0],1])

            # output['node'].y = gt_class_3D
            # output['edge'].y = gt_rels_3D

            # if output['edge'].y.nelement()==0:
            #     print('debug')

            # output['node','to','node'].edge_index = edge_indices_3D.t().contiguous()

            # output['node'].idx2oid = [idx2oid]
            # output['node'].idx2iid = [idx2iid]
            # print(output['node'].idx2iid)

            if self.mconfig.load_points:
                output['node'].pts = obj_points

                if 'edge_desc' not in self.mconfig or self.mconfig['edge_desc'] == 'pts':
                    output['node'].desp = descriptor

                if self.mconfig.rel_data_type == 'points':
                    output['node', 'to', 'node'].pts = rel_points

            if self.mconfig.load_images:
                if self.mconfig.is_roi_img:
                    output['roi'].x = torch.zeros([roi_images.size(0), 1])
                    output['roi'].img = roi_images
                    output['roi', 'sees',
                           'node'].edge_index = edge_indices_img_to_obj

                    if 'edge_desc' not in self.mconfig or self.mconfig['edge_desc'] == 'roi':
                        output['node'].desp = node_descriptor_for_image

                    # if not self.mconfig.load_points:
                    #     output['node'].desp = node_descriptor_for_image
                else:
                    output['roi'].x = torch.zeros([len(img_bounding_boxes), 1])
                    output['roi'].y = gt_class_image
                    output['roi'].box = img_bounding_boxes
                    output['roi'].img = images
                    output['roi'].desp = node_descriptor_for_image
                    output['roi'].idx2oid = [img_idx2oid]
                    output['edge2D'].x = torch.zeros([gt_rels_2D.size(0), 1])
                    output['edge2D'].y = gt_rels_2D

                    output['roi', 'to', 'roi'].edge_index = image_edge_indices.t(
                    ).contiguous()
                    output['roi', 'temporal',
                           'roi'].edge_index = temporal_node_graph.t().contiguous()
                    output['edge2D', 'temporal', 'edge2D'].edge_index = temporal_edge_graph.t(
                    ).contiguous()

        '''release'''
        if hasattr(self, 'filtered_data'):
            del self.filtered_data
        if hasattr(self, 'image_feature'):
            del self.image_feature
        if hasattr(self, 'sg_data'):
            del self.sg_data
        if hasattr(self, 'roi_imgs'):
            del self.roi_imgs
        if hasattr(self, 'mv_data'):
            del self.mv_data
        return output_seq

    def __len__(self):
        return self.size

    def norm_tensor(self, points):
        assert points.ndim == 2
        assert points.shape[1] == 3
        centroid = torch.mean(points, dim=0)  # N, 3
        points -= centroid  # n, 3, npts
        # find maximum distance for each n -> [n]
        furthest_distance = points.pow(2).sum(1).sqrt().max()
        points /= furthest_distance
        return points

    def data_augmentation(self, points):
        # random rotate
        matrix = np.eye(3)
        matrix[0:3, 0:3] = transformation.rotation_matrix(
            [0, 0, 1], np.random.uniform(0, 2*np.pi, 1))
        centroid = points[:, :3].mean(0)
        points[:, :3] -= centroid
        points[:, :3] = np.dot(points[:, :3], matrix.T)
        if self.use_normal:
            ofset = 3
            if self.use_rgb:
                ofset += 3
            points[:, ofset:3 +
                   ofset] = np.dot(points[:, ofset:3+ofset], matrix.T)

        # Add noise
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
        pth_node_weights = self.pth_node_weights
        pth_edge_weights = self.pth_edge_weights
        pth_filtered = self.pth_filtered
        config = self.cfg

        should_process = not os.path.isfile(pth_filtered)
        if not self.for_eval:
            should_process |= not os.path.isfile(
                pth_node_weights) or not os.path.isfile(pth_edge_weights)

        if should_process:
            '''
            This is to make sure the 2D and 3D methdos have the same amount of data for training 
            '''
            print('generating filtered data...')
            ''' load data '''
            selected_scans = read_txt_to_list(os.path.join(
                self.path, '%s_scans.txt' % (self.mode)))
            self.open_mv_graph()
            self.open_data()
            c_sg_data = cvt_all_to_dict_from_h5(self.sg_data)

            '''check scan_ids'''
            # filter input scans with relationship data
            tmp = set(c_sg_data.keys())
            inter = sorted(list(tmp.intersection(selected_scans)))
            # filter input scans with image data
            tmp = set(self.mv_data.keys())
            inter = sorted(list(tmp.intersection(inter)))

        if not os.path.isfile(pth_filtered):
            self.open_data()
            self.open_mv_graph()
            filtered_data = defaultdict(dict)
            # filtered_kf_indices = dict()
            # filtered_node_indices = dict()

            for scan_id in inter:
                scan_data = c_sg_data[scan_id]
                mv_data = self.mv_data[scan_id]

                for timestamp in scan_data:
                    scan_data_seq = scan_data[timestamp]
                    mv_data_seq = mv_data[timestamp]

                    # if 'nodes' not in

                    object_data = scan_data_seq['nodes']
                    # relationships_data = scan_data['relationships']

                    ''' build mapping '''
                    instance2labelName = {
                        int(key): node['label'] for key, node in object_data.items()}

                    mv_nodes = mv_data_seq['nodes']
                    kfs = mv_data_seq['kfs']

                    '''filter'''
                    # get the intersection between point and multi-view data
                    mv_node_ids = [int(x) for x in mv_nodes.keys()]
                    sg_node_ids = object_data.keys()
                    inter_node_ids = set(sg_node_ids).intersection(mv_node_ids)
                    # object_data = {nid: object_data[nid] for nid in inter_node_ids}
                    filtered_object_indices = [nid for nid in inter_node_ids]

                    if len(filtered_object_indices) == 0:
                        continue  # skip if no intersection

                    dict_objId_kfId = dict()  # make sure each object has at least a keyframe
                    kf_indices = []
                    '''select frames with at least 1 objects'''
                    for k in kfs.keys():
                        kf = kfs[k]
                        oids = [v[0] for v in kf.attrs['seg2idx']]

                        # filter object bbox with the intersection of the object_data
                        oids = set(object_data.keys()).intersection(oids)
                        if len(oids) == 0:
                            continue  # skip if no object available

                        # filter keyframe by checking there is at least one object exist
                        obj_count = 0
                        for oid in oids:
                            oid = int(oid)
                            if oid in instance2labelName:
                                if instance2labelName[oid] in self.classNames:
                                    dict_objId_kfId[oid] = k
                                    obj_count += 1
                        if obj_count > 0:
                            kf_indices.append(int(k))

                    if len(kf_indices) == 0:
                        continue  # skip if no keyframe available

                    filtered_object_indices = [
                        k for k in dict_objId_kfId.keys()]

                    filtered_data[scan_id][timestamp] = dict()
                    filtered_data[scan_id][timestamp][define.NAME_FILTERED_KF_INDICES] = kf_indices
                    filtered_data[scan_id][timestamp][define.NAME_FILTERED_OBJ_INDICES] = filtered_object_indices

            with h5py.File(pth_filtered, 'w') as h5f:
                for scan_id in filtered_data:
                    buffer = data_to_raw(filtered_data[scan_id])
                    h5f.create_dataset(scan_id, data=buffer,
                                       compression='gzip')

        if not self.for_eval:
            if not os.path.isfile(pth_node_weights) or not os.path.isfile(pth_edge_weights):
                # TODO: also filter out nodes when only with points input. this gives fair comparison on points and images methods.
                filtered_sg_data = dict()
                self.open_filtered()
                for scan_id in self.filtered_data.keys():
                    filtered_data = raw_to_data(self.filtered_data[scan_id])
                    node_indices = filtered_data[define.NAME_FILTERED_OBJ_INDICES]

                    # mv_node_ids = [int(x) for x in self.mv_data[scan_id]['nodes'].keys()]
                    # sg_node_ids = c_sg_data[scan_id]['nodes'].keys()
                    # inter_node_ids = set(sg_node_ids).intersection(mv_node_ids)

                    filtered_sg_data[scan_id] = dict()
                    filtered_sg_data[scan_id]['nodes'] = {
                        nid: c_sg_data[scan_id]['nodes'][nid] for nid in node_indices}

                    filtered_sg_data[scan_id]['relationships'] = c_sg_data[scan_id]['relationships']
                c_sg_data = filtered_sg_data

                if self.full_edge:
                    edge_mode = 'fully_connected'
                else:
                    edge_mode = 'nn'
                # edge_mode='gt'
                # print('edge_mode:',edge_mode)
                wobjs, wrels, o_obj_cls, o_rel_cls = compute_weight.compute_sgfn(self.classNames, self.relationNames, c_sg_data, selected_scans,
                                                                                 normalize=config.data.normalize_weight,
                                                                                 for_BCE=self.multi_rel_outputs == True,
                                                                                 edge_mode=edge_mode,
                                                                                 none_index=self.none_idx,
                                                                                 verbose=config.VERBOSE)
                for idx, obj_cls_name in enumerate(self.classNames):
                    if obj_cls_name in config.data.obj_ignore_list:
                        if config.VERBOSE:
                            print('ignore object ', obj_cls_name)
                        wobjs[idx] = 0

                wobjs = np.array(wobjs)
                wrels = np.array(wrels)
                np.savetxt(pth_node_weights, wobjs)
                np.savetxt(pth_edge_weights, wrels)

                # test
                w_node_cls = np.loadtxt(pth_node_weights)
                w_edge_cls = np.loadtxt(pth_edge_weights)
                # self.w_node_cls = torch.from_numpy(np.array(wobjs)).float()
                # self.w_edge_cls = torch.from_numpy(np.array(wrels)).float()

        if should_process:
            del self.sg_data
            del self.mv_data

    def __sample_points(self, scan_id, points, instances, cat: list, filtered_instances: list):
        bboxes = list()
        use_obj_context = False  # TODO: not here
        obj_points = torch.zeros(
            [len(cat), self.mconfig.node_feature_dim, self.dim_pts])
        descriptor = torch.zeros([len(cat), 11])
        for i in range(len(filtered_instances)):
            instance_id = filtered_instances[i]
            obj_pointset = points[np.where(instances == instance_id)[0], :]

            min_box = np.min(obj_pointset[:, :3], 0)
            max_box = np.max(obj_pointset[:, :3], 0)
            if use_obj_context:
                min_box -= 0.02
                max_box += 0.02
                filter_mask = (points[:, 0] > min_box[0]) * (points[:, 0] < max_box[0]) \
                    * (points[:, 1] > min_box[1]) * (points[:, 1] < max_box[1]) \
                    * (points[:, 2] > min_box[2]) * (points[:, 2] < max_box[2])
                obj_pointset = points[np.where(filter_mask > 0)[0], :]
            bboxes.append([min_box, max_box])

            if len(obj_pointset) == 0:
                print('scan_id:', scan_id)
                # print('selected_instances:',len(selected_instances))
                print('filtered_instances:', len(filtered_instances))
                print('instance_id:', instance_id)
            choice = np.random.choice(len(obj_pointset), self.mconfig.node_feature_dim, replace=len(
                obj_pointset) < self.mconfig.node_feature_dim)
            obj_pointset = obj_pointset[choice, :]
            descriptor[i] = util_data.gen_descriptor_pts(
                torch.from_numpy(obj_pointset)[:, :3])
            obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))

            # util_data.save_to_ply(obj_pointset[:,:3],'./tmp_{}.ply'.format(i))

            obj_pointset[:, :3] = self.norm_tensor(obj_pointset[:, :3])
            obj_points[i] = obj_pointset
        obj_points = obj_points.permute(0, 2, 1)
        return obj_points, descriptor, bboxes

    def __sample_rel_points(self, points, instances, idx2oid, bboxes, edge_indices):
        rel_points = list()
        for e in range(len(edge_indices)):
            edge = edge_indices[e]
            index1 = edge[0]
            index2 = edge[1]

            mask1 = (instances == idx2oid[index1]).astype(np.int32) * 1
            mask2 = (instances == idx2oid[index2]).astype(np.int32) * 2
            mask_ = np.expand_dims(mask1 + mask2, 1)
            bbox1 = bboxes[index1]
            bbox2 = bboxes[index2]
            min_box = np.minimum(bbox1[0], bbox2[0])
            max_box = np.maximum(bbox1[1], bbox2[1])
            filter_mask = (points[:, 0] > min_box[0]) * (points[:, 0] < max_box[0]) \
                * (points[:, 1] > min_box[1]) * (points[:, 1] < max_box[1]) \
                * (points[:, 2] > min_box[2]) * (points[:, 2] < max_box[2])
            points4d = np.concatenate([points, mask_], 1)

            pointset = points4d[np.where(filter_mask > 0)[0], :]
            choice = np.random.choice(
                len(pointset), self.mconfig.num_points_union, replace=True)
            pointset = pointset[choice, :]
            pointset = torch.from_numpy(pointset.astype(np.float32))

            # save_to_ply(pointset[:,:3],'./tmp_rel_{}.ply'.format(e))

            pointset[:, :3] = zero_mean(pointset[:, :3], False)
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
        rel_points = rel_points.permute(0, 2, 1)
        return rel_points

    def __sample_3D_nodes(self, object_data: dict, mv_data: dict, nns: dict):
        instance2labelName = {int(key): node['label']
                              for key, node in object_data.items()}

        '''sample training set'''
        instances_ids = list(instance2labelName.keys())
        if 0 in instances_ids:
            instances_ids.remove(0)

        if self.sample_in_runtime and not self.for_eval:
            selected_nodes = list(object_data.keys())
            if self.mconfig.load_images:
                mv_node_ids = [int(x) for x in mv_data['nodes'].keys()]
                selected_nodes = list(
                    set(selected_nodes).intersection(mv_node_ids))
            # selected_nodes = list(set(selected_nodes).intersection(filtered_data))
            # if len(selected_nodes)==0:
            #     print('object_data.keys():',sorted(list(object_data.keys())))
            #     if self.mconfig.load_images:
            #         print('mv_node_ids',sorted(mv_node_ids))
            #         print('filtered_data:',sorted(np.asarray(filtered_data)))
            #     raise RuntimeError('no node available!')

            use_all = False
            # 1 if "sample_num_nn" not in self.config else self.config.sample_num_nn
            sample_num_nn = self.mconfig.sample_num_nn
            # 1 if "sample_num_seed" not in self.config else self.config.sample_num_seed
            sample_num_seed = self.mconfig.sample_num_seed
            if sample_num_nn == 0 or sample_num_seed == 0:
                use_all = True

            if not use_all:
                # select 1 node and include their neighbor nodes n times.
                filtered_nodes = util_data.build_neighbor_sgfn(
                    nns, selected_nodes, sample_num_nn, sample_num_seed)
            else:
                filtered_nodes = selected_nodes  # use all nodes

            instances_ids = list(filtered_nodes)
            if 0 in instances_ids:
                instances_ids.remove(0)

            if 'max_num_node' in self.mconfig and self.mconfig.max_num_node > 0 and len(instances_ids) > self.mconfig.max_num_node:
                instances_ids = random_drop(
                    instances_ids, self.mconfig.max_num_node)

            if self.shuffle_objs:
                random.shuffle(instances_ids)

        ''' 
        Find instances we care abot. Build oid2idx and cat list
        oid2idx maps instances to a mask id. to randomize the order of instance in training.
        '''
        oid2idx = {}  # map instance_id to idx
        idx2oid = {}  # map idx to instance_id
        cat = []
        counter = 0
        filtered_instances = list()
        for instance_id in instances_ids:
            class_id = -1
            instance_labelName = instance2labelName[instance_id]
            if instance_labelName in self.classNames:
                class_id = self.classNames.index(instance_labelName)

            # mask to cat:
            # insstance 0 is unlabeled.
            if (class_id >= 0) and (instance_id > 0):
                oid2idx[int(instance_id)] = counter
                idx2oid[counter] = int(instance_id)
                counter += 1
                filtered_instances.append(instance_id)
                cat.append(class_id)

        return cat, oid2idx, idx2oid, filtered_instances

    def __extract_relationship_data(self, relationships_data):
        '''build relaitonship data'''
        relatinoships_gt = defaultdict(list)
        for r in relationships_data:
            r_src = int(r[0])
            r_tgt = int(r[1])
            r_lid = int(r[2])
            r_cls = r[3]

            if r_cls not in self.relationNames:
                continue  # only keep the relationships we want
            # remap the index of relationships in case of custom relationNames
            r_lid = self.relationNames.index(r_cls)

            key = (r_src, r_tgt)

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

    def __sample_relationships(self, relatinoships_gt: dict, idx2oid: dict, edge_indices: list):
        if self.multi_rel_outputs:
            gt_rels = torch.zeros(len(edge_indices), len(
                self.relationNames), dtype=torch.float)
        else:
            gt_rels = torch.ones(
                len(edge_indices), dtype=torch.long)*self.none_idx
        edges_has_gt = []
        for e in range(len(edge_indices)):
            edge = edge_indices[e]
            index1 = edge[0]
            index2 = edge[1]
            oid1 = idx2oid[index1]
            oid2 = idx2oid[index2]
            key = (oid1, oid2)
            if key in relatinoships_gt:
                if self.multi_rel_outputs:
                    for x in relatinoships_gt[key]:
                        gt_rels[e, x] = 1
                else:
                    if len(relatinoships_gt[key]) != 1:
                        # print('scan_id',scan_id)
                        # print('iid1,iid2',idx2oid[index1],idx2oid[index2])
                        print('index1,index2', index1, index2)
                        print(
                            'key, relatinoships_gt[key]', key, relatinoships_gt[key])
                        # print(instance2labelName[key[0]],instance2labelName[key[1]])
                        [print(self.relationNames[x])
                         for x in relatinoships_gt[key]]
                        assert len(relatinoships_gt[key]) == 1
                    gt_rels[e] = relatinoships_gt[key][0]
                edges_has_gt.append(e)
        return gt_rels, edges_has_gt

    def __sample_3D_node_edges(self, cat: list, oid2idx: dict, filtered_instances: list, nns: dict):
        if self.sample_in_runtime:
            if self.full_edge:
                '''use dense'''
                edge_indices = list()
                for n in range(len(cat)):
                    for m in range(len(cat)):
                        if n == m:
                            continue
                        edge_indices.append((n, m))
            else:
                if not self.for_eval:
                    '''sample from neighbor'''
                    edge_indices = util_data.build_edge_from_selection_sgfn(
                        filtered_instances, nns, max_edges_per_node=-1)
                    edge_indices = [(oid2idx[edge[0]], oid2idx[edge[1]])
                                    for edge in edge_indices]
                else:
                    '''dense neighbor'''
                    edge_indices = set()
                    for k, v in nns.items():
                        k = int(k)
                        if k not in oid2idx:
                            continue
                        mask_k = oid2idx[k]
                        for vv in v:
                            vv = int(vv)
                            if vv not in oid2idx:
                                continue
                            mask_vv = oid2idx[vv]
                            edge_indices.add((mask_k, mask_vv))
                    edge_indices = [(e[0], e[1]) for e in edge_indices]

            '''edge dropout'''
            # if len(edge_indices)>0:
            #     if not self.for_eval:
            #         edge_indices = random_drop(edge_indices, self.mconfig.drop_edge)
            #     if self.for_eval :
            #         edge_indices = random_drop(edge_indices, self.mconfig.drop_edge_eval)

            #     if self.mconfig.max_num_edge > 0 and len(edge_indices) > self.mconfig.max_num_edge and not self.for_eval:
            #         choices = np.random.choice(range(len(edge_indices)),self.mconfig.max_num_edge,replace=False).tolist()
            #         edge_indices = [edge_indices[t] for t in choices]
        else:
            edge_indices = list()
            max_edges = -1
            for n in range(len(cat)):
                for m in range(len(cat)):
                    if n == m:
                        continue
                    edge_indices.append((n, m))
            # if max_edges>0 and len(edge_indices) > max_edges and not self.for_eval:
            #     # for eval, do not drop out any edges.
            #     indices = list(np.random.choice(len(edge_indices),max_edges,replace=False))
            #     edge_indices = edge_indices[indices]
        return edge_indices

    def __drop_edge(self, gt_rels: torch.Tensor, edge_indices: list, edge_index_has_gt: list):
        if len(edge_indices) == 0:  # no edges
            return gt_rels, edge_indices

        all_indices = set(range(len(edge_indices)))
        edge_index_wo_gt = all_indices.difference(edge_index_has_gt)
        if len(edge_index_wo_gt) == 0:
            return gt_rels, edge_indices  # all edges are needed

        edge_index_wo_gt = list(edge_index_wo_gt)
        if not self.for_eval:
            edge_index_wo_gt = random_drop(
                edge_index_wo_gt, self.mconfig.drop_edge)
        if self.for_eval:
            edge_index_wo_gt = random_drop(
                edge_index_wo_gt, self.mconfig.drop_edge_eval)

        num_edges = len(edge_index_wo_gt)+len(edge_index_has_gt)
        if self.mconfig.max_num_edge > 0 and num_edges > self.mconfig.max_num_edge:
            # only process with max_num_ede is set, and the total number is larger
            # and the edges with gt is smaller
            if len(edge_index_has_gt) < self.mconfig.max_num_edge:
                number_to_sample = self.mconfig.max_num_edge - \
                    len(edge_index_has_gt)
                edge_index_wo_gt = np.random.choice(
                    edge_index_wo_gt, number_to_sample, replace=False).tolist()
            else:
                edge_index_wo_gt = []
        final_edge_indices = list(edge_index_has_gt)+list(edge_index_wo_gt)
        edge_indices = [edge_indices[t] for t in final_edge_indices]
        gt_rels = gt_rels[final_edge_indices]

        return gt_rels, edge_indices

    def __crop_roi_images(self,
                          scan_id: str,
                          cat: list,
                          idx2oid: dict,
                          mv_nodes: dict,
                          mv_kfs: dict,
                          filtered_instances: list,
                          filtered_kf_indices: list,
                          object_data):
        descriptor_generator = util_data.Node_Descriptor_24(
            with_bbox=self.mconfig.img_desc_6_pts)
        node_descriptor_for_image = torch.zeros(
            [len(cat), len(descriptor_generator)])

        roi_images = list()
        edge_indices_img_to_obj = []
        for idx in range(len(cat)):
            oid = str(idx2oid[idx])
            node = mv_nodes[oid]
            kf_indices = np.asarray(node)

            for fid in kf_indices:
                pth_rgb = os.path.join(
                    self.cfg.data.path_3rscan, scan_id, 'sequence', define.RGB_NAME_FORMAT.format(int(fid)))

                '''load data'''
                img_data = Image.open(pth_rgb)
                img_data = np.rot90(img_data, 3)  # Rotate image

                height, width = img_data.shape[0], img_data.shape[1]
                # bfid = imgs['indices'][fid] # convert frame idx to the buffer idx

                #  bounding box
                kf = mv_kfs[str(fid)]
                kf_seg2idx = {v[0]: v[1] for v in kf.attrs['seg2idx']}
                bid = kf_seg2idx[int(oid)]
                kfdata = np.asarray(kf)
                box = kfdata[bid, :4]
                oc = kfdata[bid, 4]

                assert box[0] <= 1
                assert box[0] >= 0
                assert box[1] <= 1
                assert box[1] >= 0
                assert box[2] <= 1
                assert box[2] >= 0
                assert box[3] <= 1
                assert box[3] >= 0

                # Denormalize
                box[0] *= width
                box[2] *= width
                box[1] *= height
                box[3] *= height

                box = torch.from_numpy(box).float().view(1, -1)
                # timg = torch.as_tensor(img_data.copy()).permute(2,0,1)
                timg = transforms.ToTensor()(img_data.copy()).unsqueeze(0)
                w = box[:, 2] - box[:, 0]
                h = box[:, 3] - box[:, 1]
                # if n_workers==0: logger_py.info('box: {}, dim: {}'.format(box,[h,w]))
                region = roi_align(timg, [box], [h, w])
                region = self.transform(region).squeeze(0)

                roi_images.append(region)
                edge_indices_img_to_obj.append([len(roi_images)-1, idx])

        roi_images = torch.stack(roi_images, dim=0)
        roi_images = normalize_imagenet(roi_images.float())

        edge_indices_img_to_obj = torch.LongTensor(
            edge_indices_img_to_obj).t().contiguous()

        '''compute node description'''
        for i in range(len(filtered_instances)):
            instance_id = filtered_instances[i]
            obj = object_data[instance_id]
            # obj = objects[str(instance_id)]

            '''augmentation'''
            # random scale dim with up to 0.3
            if not self.for_eval and self.mconfig.bbox_aug_ratio > 0:
                center = np.array(obj['center'])
                dim = np.array(obj['dimension'])

                max_ratio = self.mconfig.bbox_aug_ratio
                dim_scales = np.random.uniform(
                    low=-max_ratio, high=max_ratio, size=3)
                reduce_amount = dim * dim_scales
                center += reduce_amount

                dim_scales = np.random.uniform(
                    low=-max_ratio, high=max_ratio, size=3)
                reduce_amount = dim * dim_scales
                dim += reduce_amount
                obj['center'] = center.tolist()
                obj['dimension'] = dim.tolist()
            node_descriptor_for_image[i] = descriptor_generator(obj)

        return roi_images, node_descriptor_for_image, edge_indices_img_to_obj

    def __load_roi_images(self, cat: list, idx2oid: dict, mv_nodes: dict, roi_imgs: dict,
                          object_data: dict, filtered_instances: list):
        descriptor_generator = util_data.Node_Descriptor_24(
            with_bbox=self.mconfig.img_desc_6_pts)

        roi_images = list()
        node_descriptor_for_image = torch.zeros(
            [len(cat), len(descriptor_generator)])
        edge_indices_img_to_obj = []

        '''get roi images'''
        for idx in range(len(cat)):
            oid = str(idx2oid[idx])
            node = mv_nodes[oid]
            cls_label = node.attrs['label']
            if cls_label == 'unknown':
                cls_label = self.classNames[cat[idx]]

            img_ids = range(len(roi_imgs[oid]))
            if not self.for_eval:
                img_ids = random_drop(
                    img_ids, self.mconfig.drop_img_edge, replace=True)
            if self.for_eval:
                img_ids = random_drop(img_ids, self.mconfig.drop_img_edge_eval)

            for img_id in img_ids:
                img = roi_imgs[oid][img_id]
                img = torch.as_tensor(np.array(img))
                img = torch.clamp((img*255).byte(), 0, 255).byte()
                img = self.transform(img)
                roi_images.append(img)
                edge_indices_img_to_obj.append([len(roi_images)-1, idx])

        # images = torch.as_tensor(np.array(images))#.clone()
        # images = torch.clamp((images*255).byte(),0,255).byte()
        # images = torch.stack([self.transform(x) for x in images],dim=0)
        roi_images = torch.stack(roi_images, dim=0)
        roi_images = normalize_imagenet(roi_images.float()/255.0)

        edge_indices_img_to_obj = torch.LongTensor(
            edge_indices_img_to_obj).t().contiguous()

        # for idx in range(len(cat)):
        #     oid = str(idx2oid[idx])
        #     node = mv_nodes[oid]
        #     cls_label = node.attrs['label']
        #     if cls_label == 'unknown':
        #         cls_label = self.classNames[cat[idx]]

        #     img_ids=range(len(roi_imgs[oid]))

        #     if not self.for_eval:
        #         img_ids = random_drop(img_ids, self.mconfig.drop_img_edge, replace=True)
        #     if self.for_eval :
        #         img_ids = random_drop(img_ids, self.mconfig.drop_img_edge_eval)

        #     img = [roi_imgs[oid][x] for x in img_ids]
        #     # else:
        #     #     kf_indices = [idx for idx in range(img.shape[0])]

        #     img = torch.as_tensor(np.array(img))#.clone()
        #     img = torch.clamp((img*255).byte(),0,255).byte()
        #     t_img = torch.stack([self.transform(x) for x in img],dim=0)
        #     if DRAW_BBOX_IMAGE:
        #         show_tensor_images(t_img.float()/255, cls_label)
        #     t_img= normalize_imagenet(t_img.float()/255.0)

        #     roi_images.append( t_img)

        '''compute node description'''
        for i in range(len(filtered_instances)):
            instance_id = filtered_instances[i]
            obj = object_data[instance_id]
            # obj = objects[str(instance_id)]

            '''augmentation'''
            # random scale dim with up to 0.3
            if not self.for_eval and self.mconfig.bbox_aug_ratio > 0:
                center = np.array(obj['center'])
                dim = np.array(obj['dimension'])

                max_ratio = self.mconfig.bbox_aug_ratio
                dim_scales = np.random.uniform(
                    low=-max_ratio, high=max_ratio, size=3)
                reduce_amount = dim * dim_scales
                center += reduce_amount

                dim_scales = np.random.uniform(
                    low=-max_ratio, high=max_ratio, size=3)
                reduce_amount = dim * dim_scales
                dim += reduce_amount
                obj['center'] = center.tolist()
                obj['dimension'] = dim.tolist()
            node_descriptor_for_image[i] = descriptor_generator(obj)

        return roi_images, node_descriptor_for_image, edge_indices_img_to_obj

    def __load_full_images(self, scan_id, idx2oid: dict, cat: list,
                           scan_data: dict, mv_data: dict, filtered_kf_indices: dict):
        if self.cfg.data.use_precompute_img_feature:
            self.open_image_feature()

        '''containers'''
        images = list()
        bounding_boxes = list()  # bounding_boxes[node_id]{kf_id: [boxes]}
        bbox_cat = list()
        node_descriptor_for_image = list()
        image_edge_indices = list()
        img_idx2oid = dict()  # from image object index to object isntance
        per_frame_info_dict = defaultdict(dict)

        '''alias'''
        object_data = scan_data['nodes']
        mv_nodes = mv_data['nodes']
        mv_kfs = mv_data['kfs']
        feature_type = self.cfg.model.image_encoder.backend

        '''descriptor generator'''
        descriptor_generator = util_data.Node_Descriptor_24(
            with_bbox=self.cfg.data.img_desc_6_pts)

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
            kf_indices = set(kf_indices).intersection(filtered_kf_indices)
            kf_indices = list(kf_indices)

            if len(kf_indices) == 0:
                continue
            if not self.for_eval:
                kf_indices = random_drop(
                    kf_indices, self.mconfig.drop_img_edge, replace=True)
            else:
                kf_indices = random_drop(
                    kf_indices, self.mconfig.drop_img_edge_eval)

            fids = fids.union(kf_indices)

        if len(fids) == 0:
            print()
            raise RuntimeError(
                'there is no bounding boxes found. (len(flids)==0)')
        # fids = fids.intersection(filtered_kf_indices)

        # drop images for memory sack
        fids = list(fids)
        fids = random_drop(fids, self.mconfig.max_full_img, replace=False)
        # if not self.for_eval:
        #     fids = random_drop(fids, self.mconfig.drop_img_edge, replace=True)
        # else:
        #     fids = random_drop(fids, self.mconfig.drop_img_edge_eval)
        # fids = random_drop(fids, self.mconfig.drop_img_edge, replace=True)

        '''load'''
        for mid, fid in enumerate(fids):
            '''read data'''
            kf = mv_kfs[str(fid)]
            # convert obj_idx to mask_idx
            kf_oid2idx = {v[0]: v[1] for v in kf.attrs['seg2idx']}

            '''get boxes of selected objects'''
            filtered_kf_oid2idx = dict()
            for k in kf_oid2idx:
                if k in idx2oid.values():  # object should exist in selected object indices
                    filtered_kf_oid2idx[k] = kf_oid2idx[k]
            if len(filtered_kf_oid2idx) == 0:
                continue

            '''load image'''
            if self.cfg.data.use_precompute_img_feature:
                # if self.for_eval:
                img_data = self.image_feature[feature_type][scan_id][str(fid)]
                img_data = np.asarray(img_data).copy()
                img_data = torch.from_numpy(img_data)
            else:
                pth_rgb = os.path.join(
                    self.cfg.data.path_3rscan, scan_id, 'sequence', define.RGB_NAME_FORMAT.format(int(fid)))
                img_data = Image.open(pth_rgb)
                img_data = np.rot90(img_data, 3)  # Rotate image
                img_data = torch.as_tensor(img_data.copy()).permute(2, 0, 1)
                img_data = self.transform(img_data)
                img_data = normalize_imagenet(img_data.float()/255.0)

            # width,height = img_data.shape[-1],img_data.shape[-2]
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
                # box[0]/=width #NOTE: already scaled in make_obj_graph_3rscan
                # box[1]/=height
                # box[2]/=width
                # box[3]/=height
                box = np.concatenate(([mid], box))
                assert not (box[1:] > 1).any()

                box = box.tolist()  # ROIAlign format

                om_id = len(bbox_cat)
                img_idx2oid[om_id] = kf_oid
                per_frame_indices.append(om_id)

                '''build input and gt'''
                bbox_cat.append(oid2cls[kf_oid])
                bounding_boxes.append(box)
                node_descriptor_for_image.append(
                    descriptor_generator(object_data[kf_oid]))

                # for temporal node edge
                # per framae should have only one oid to one om_id
                assert kf_oid not in per_frame_info['nodes']
                per_frame_info['nodes'][kf_oid] = om_id

            '''build image edges'''
            for om_id1 in per_frame_indices:
                for om_id2 in per_frame_indices:
                    if om_id1 != om_id2:
                        em_id = len(image_edge_indices)
                        key_mapped = (img_idx2oid[om_id1], img_idx2oid[om_id2])
                        assert key_mapped not in per_frame_info['edges']
                        per_frame_info['edges'][key_mapped] = em_id

                        key = (om_id1, om_id2)
                        image_edge_indices.append(key)

            # if self.mconfig.max_num_edge > 0 and len(image_edge_indices) > self.mconfig.max_num_edge and not self.for_eval:
            #     choices = np.random.choice(range(len(image_edge_indices)),self.mconfig.max_num_edge,replace=False).tolist()
            #     image_edge_indices = [image_edge_indices[t] for t in choices]

        '''build temporal node graph'''
        temporal_node_graph = list()
        temporal_edge_graph = list()
        sorted_kf_indices = sorted(fids)
        for idx in range(len(sorted_kf_indices)-1):
            fid_0 = sorted_kf_indices[idx]
            fid_1 = sorted_kf_indices[idx+1]
            finfo_0, finfo_1 = per_frame_info_dict[fid_0], per_frame_info_dict[fid_1]

            '''check if node exist'''
            nodes_0, nodes_1 = finfo_0['nodes'], finfo_1['nodes']
            for oid_0 in nodes_0:
                if oid_0 in nodes_1:
                    temporal_node_graph.append(
                        [nodes_0[oid_0], nodes_1[oid_0]])

            '''check edges'''
            edges_0, edges_1 = finfo_0['edges'], finfo_1['edges']
            for key_0 in edges_0:
                if key_0 in edges_1:
                    temporal_edge_graph.append(
                        [edges_0[key_0], edges_1[key_0]])

        # if DRAW_BBOX_IMAGE:
        # t_img = torch.stack(images,dim=0)
        # show_tensor_images(t_img.float()/255, '-')
        return images, bounding_boxes, bbox_cat, node_descriptor_for_image, \
            image_edge_indices, img_idx2oid, temporal_node_graph, temporal_edge_graph


def load_mesh(path, label_file, use_rgb, use_normal):
    result = dict()
    if label_file == 'labels.instances.align.annotated.v2.ply' or label_file == 'labels.instances.align.annotated.ply':
        if use_rgb:
            plydata = util_ply.load_rgb(path)
        else:
            plydata = trimesh.load(os.path.join(
                path, label_file), process=False)

        points = np.array(plydata.vertices)
        instances = util_ply.read_labels(plydata).flatten()

        if use_rgb:
            r = plydata.metadata['ply_raw']['vertex']['data']['red']
            g = plydata.metadata['ply_raw']['vertex']['data']['green']
            b = plydata.metadata['ply_raw']['vertex']['data']['blue']
            rgb = np.stack([r, g, b]).squeeze().transpose()
            points = np.concatenate((points, rgb), axis=1)
        if use_normal:
            nx = plydata.metadata['ply_raw']['vertex']['data']['nx']
            ny = plydata.metadata['ply_raw']['vertex']['data']['ny']
            nz = plydata.metadata['ply_raw']['vertex']['data']['nz']
            normal = np.stack([nx, ny, nz]).squeeze().transpose()
            points = np.concatenate((points, normal), axis=1)

        result['points'] = points
        result['instances'] = instances

    else:  # label_file.find('inseg')>=0 or label_file == 'cvvseg.ply':
        plydata = trimesh.load(os.path.join(path, label_file), process=False)
        points = np.array(plydata.vertices)
        text_ply_raw = 'ply_raw' if 'ply_raw' in plydata.metadata else '_ply_raw'
        instances = plydata.metadata[text_ply_raw]['vertex']['data']['label'].flatten(
        )

        if use_rgb:
            rgbs = np.array(plydata.colors)[:, :3] / 255.0 * 2 - 1.0
            points = np.concatenate((points, rgbs), axis=1)
        if use_normal:
            nx = plydata.metadata[text_ply_raw]['vertex']['data']['nx']
            ny = plydata.metadata[text_ply_raw]['vertex']['data']['ny']
            nz = plydata.metadata[text_ply_raw]['vertex']['data']['nz']

            normal = np.stack([nx, ny, nz]).squeeze().transpose()
            points = np.concatenate((points, normal), axis=1)
        result['points'] = points
        result['instances'] = instances

    return result


def zero_mean(point, normalize: bool):
    mean = torch.mean(point, dim=0)
    point -= mean.unsqueeze(0)
    ''' without norm to 1  '''
    if normalize:
        # find maximum distance for each n -> [n]
        furthest_distance = point.pow(2).sum(1).sqrt().max()
        point /= furthest_distance
    return point


if __name__ == '__main__':
    import codeLib
    path = './experiments/config_2DSSG_ORBSLAM3_l20_6_1.yaml'
    config = codeLib.Config(path)

    config.DEVICE = '1'
    # config.dataset.root = "../data/example_data/"
    # config.dataset.label_file = 'inseg.ply'
    # sample_in_runtime = True
    # config.dataset.data_augmentation=True
    # split_type = 'validation_scans' # ['train_scans', 'validation_scans','test_scans']
    dataset = SGFNDataset(config, 'validation')
    items = dataset.__getitem__(0)
    # print(items)
