import logging
#import trimesh
import numpy as np

import argparse,os
import codeLib
import torch
import ssg.config as config
from ssg.data.collate import graph_collate#, batch_graph_collate
from ssg.checkpoints import CheckpointIO
import torch.multiprocessing
import cProfile
import matplotlib
import copy
import json
import codeLib.utils.string_numpy as snp
from torchvision import transforms
from torchvision.ops import roi_align
from codeLib.common import normalize_imagenet
from codeLib.torch.visualization import show_tensor_images
from ssg.utils import util_data
from collections import defaultdict
import time
from ssg import define
from PIL import Image
from ssg.data.collate import graph_collate
from ssg.utils import util_label, util_merge_same_part
from ssg.utils.util_data import raw_to_data, cvt_all_to_dict_from_h5
from codeLib.common import rgb_2_hex
from codeLib.common import rand_24_bit, color_rgb
from ssg.utils.graph_vis import DrawSceneGraph, to_name_dict, process_pd
from ssg.utils.util_data import merge_batch_mask2inst
from codeLib.geoemetry.common import create_box
import trimesh

matplotlib.pyplot.set_loglevel("CRITICAL")
logging.getLogger('PIL').setLevel('CRITICAL')
logging.getLogger('trimesh').setLevel('CRITICAL')
logger_py = logging.getLogger(__name__)

def parse():
    r"""loads model config

    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='./configs/default.yaml', help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('-f','--filename', type=str, default='/home/sc/research/ORB_SLAM3/bin/test/graph_2dssg.json', 
                        help='filepath to graph_2dssg_seq.json',required=True)
    parser.add_argument('-vis','--vis', action='store_true',help='show scene graph')
    args = parser.parse_args()
    
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise RuntimeError('Targer config file does not exist. {}' & config_path)
        
    # load config file
    config = codeLib.Config(config_path)
    # return config
    config.LOADBEST = True
    config.MODE = 'test'
    config.filename = args.filename
    config.vis = args.vis
    
    # check if name exist
    if 'name' not in config:
        config_name = os.path.basename(args.config)
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['name'] = name 
    
    # init device
    if torch.cuda.is_available() and len(config.GPU) > 0:
        config.DEVICE = torch.device("cuda")
    else:
        config.DEVICE = torch.device("cpu")      
        
    config.log_level = 'DEBUG'
    return config

def render(nodes,predictions:dict, COLOR_MODE:str):
    assert COLOR_MODE in ['segment','semantic']
    meshes=list()
    pds = process_pd(**predictions)
    color_map = util_label.get_NYU40_color_palette()
    for key,node in nodes.items():
        node_id = int(key)
        if node_id == 0:continue
        if key not in pds['nodes']:continue
        
        Rinv = np.array(node['rotation']).reshape(3,3)
        R = np.transpose(Rinv)
        center = np.array(node['center']) 
        dims = np.array(node['dimension'])
        box = create_box(dims,0.05)
        
        mat44 = np.eye(4)
        mat44[:3,:3] = R
        mat44[:3,3] = center
        box.apply_transform(mat44)
        
        if COLOR_MODE == 'segment':
            if 'color' in node:
                color = [node['color'][2],node['color'][1],node['color'][0] ]
            else:
                color = color_rgb(rand_24_bit())
        elif COLOR_MODE =='semantic':
            color = color_map[util_label.nyu40_name_to_id(pds['nodes'][key])+1]
        else:
            raise RuntimeError('unknown type {}'.format(COLOR_MODE))
            
        box.visual.vertex_colors[:,:3] = color
        if center[2]>1:continue
        if center[2]+dims[2]>1:continue
        meshes.append(box)
    return meshes

class BBoxDrawer(object):
    def __init__(self, MODE:str='texture'):
        assert MODE in ['texture','label']
        if MODE == 'texture':
            self.GT_NAME = 'mesh.refined.obj'
        else:
            self.GT_NAME = 'labels.instances.annotated.v2.ply'
        # load transformation
        scan_dict = dict()
        with open(os.path.join(define.DATA_PATH,'3RScan.json'),'r') as f:
            scan3r_data = json.load(f)
            for scan_data in scan3r_data:
                scan_dict[scan_data['reference']] = scan_data
                for sscan in scan_data['scans']:
                    scan_dict[sscan['reference']] = sscan
        self.scan_dict= scan_dict
    def __call__(self,scan_id:str,pth_json:str,predictions:dict):
        with open(pth_json) as f: 
            data = json.load(f)
        if 'nodes' not in data:
            nodes = data[scan_id]['nodes']
        else:
            nodes = data['nodes']
        meshes = render(nodes,predictions,'semantic')

        '''load GT mesh'''    
        pth_folder = '/media/sc/SSD1TB/dataset/3RScan/data/3RScan/'+scan_id+'/'+ self.GT_NAME
        mesh = trimesh.load(pth_folder, process=False)
        if 'transform' in self.scan_dict[scan_id]:
            T = np.asarray(self.scan_dict[scan_id]['transform']).reshape(4,4).transpose()
        else:
            T = np.eye(4,4)
        mesh = mesh.apply_transform(T)
        meshes.append(mesh)
        return meshes
        
        

def main():
    cfg = parse()
    
    # Shorthands
    out_dir = os.path.join(cfg['training']['out_dir'], cfg.name)
    
    # set random seed
    codeLib.utils.util.set_random_seed(cfg.SEED)
    
    # Output directory
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    # Log
    logging.basicConfig(filename=os.path.join(out_dir,'log'), level=cfg.log_level)
    logger_py.setLevel(cfg.log_level)

    ''' Get segment dataset '''
    cfg.data.input_type = 'sgfn_incre'
    dataset_seg  = config.get_dataset(cfg,'test')    

    ''' Get logger '''
    logger = config.get_logger(cfg)
    if logger is not None: logger, _ = logger
    
    ''' Create model '''
    node_cls_names = dataset_seg.classNames
    edge_cls_names = dataset_seg.relationNames
    num_obj_cls = len(node_cls_names)
    num_rel_cls = len(edge_cls_names) if edge_cls_names is not None else 0
    
    model = config.get_model(cfg,num_obj_cls=num_obj_cls, num_rel_cls=num_rel_cls)
    '''check ckpt'''
    checkpoint_io = CheckpointIO(out_dir, model=model)
    load_dict = checkpoint_io.load('model_best.pt', device=cfg.DEVICE)
    it = load_dict.get('it', -1)
    
    with_loader = False
    bbox_drawer = BBoxDrawer()
    if not with_loader:
        dataProcessor = InputProcessor(cfg,cfg.DEVICE)
        acc_time = 0
        timer_counter=0
        model.eval()
        
        nodes_pds_all = dict()
        edges_pds_all = dict()
        node_colors_all = dict()
        path = cfg.filename
        for data in dataProcessor(path):
            # data_seg = graph_collate([data_seg])
            '''process seg'''
            eval_dict={}
            with torch.no_grad():
                data = dataProcessor.process_data_dict(data)
                
                scan_id = data['scan_id']
                graphDrawer = DrawSceneGraph(scan_id,node_cls_names, edge_cls_names,debug=True)
                nodes_w=defaultdict(int)
                edges_w=defaultdict(int)
                
                def fuse(old:dict,w_old:dict,new:dict):
                    for k,v in new.items():
                        if k in old:
                            old[k] = (old[k]*w_old[k]+new[k]) / (w_old[k]+1)
                            w_old[k]+=1
                        else:
                            old[k] = new[k]
                            w_old[k] = 1
                    return old,w_old
                def process(data):
                    data = dataProcessor.process_data_dict(data)
                    # Shortcuts
                    scan_id = data['scan_id']
                    # gt_cls = data['gt_cls']
                    # gt_rel = data['gt_rel']
                    mask2seg = data['mask2instance']
                    node_edges_ori = data['node_edges']
                    data['node_edges'] = data['node_edges'].t().contiguous()
                    # seg2inst = data['seg2inst']
                    
                    # check input valid
                    if node_edges_ori.ndim==1: return {},{}, -1
                    
                    ''' make forward pass through the network '''
                    tick = time.time()
                    node_cls, edge_cls = model(**data)
                    tock = time.time()
                    
                    '''collect predictions on inst and edge pair'''
                    node_pds = dict()
                    edge_pds = dict()
                    
                    '''merge prediction from seg to instance (in case of "same part")'''
                    # inst2masks = defaultdict(list)          
                    mask2seg= merge_batch_mask2inst(mask2seg) 
                    tmp_dict = defaultdict(list)
                    for mask, seg in mask2seg.items():
                        # inst = seg2inst[seg]
                        # inst2masks[inst].append(mask)
                        
                        tmp_dict[seg].append(node_cls[mask])
                    for seg, l in tmp_dict.items():
                        if seg in node_pds: 
                            raise RuntimeError()
                        pd =  torch.stack(l,dim=0)
                        pd = torch.softmax(pd,dim=1).mean(dim=0)
                        node_pds[seg] = pd
                        
                    tmp_dict = defaultdict(list)
                    for idx in range(len(node_edges_ori)):
                        src_idx,tgt_idx = data['node_edges'][0,idx].item(),data['node_edges'][1,idx].item()
                        seg_src,seg_tgt = mask2seg[src_idx],mask2seg[tgt_idx]
                        # inst_src,inst_tgt = seg2inst[seg_src],seg2inst[seg_tgt]
                        key = (seg_src,seg_tgt)
                        
                        
                        tmp_dict[key].append(edge_cls[idx])
                        
                    for key, l in tmp_dict.items():
                        if key in edge_pds: 
                            raise RuntimeError()
                        pd =  torch.stack(l,dim=0)
                        pd = torch.softmax(pd,dim=1).mean(0)
                        edge_pds[key] = pd
                        # src_inst_idx, tgt_inst_idx = inst_mask2inst[src_idx], inst_mask2inst[tgt_idx]
                        # inst_gt_pairs.add((src_inst_idx, tgt_inst_idx))
                    
                    return node_pds, edge_pds, tock-tick
                
                
                node_pds, edge_pds, pt = process(data)
                if pt>0:
                    acc_time += pt
                    timer_counter+=1
                
                    fuse(nodes_pds_all,nodes_w,node_pds)
                    fuse(edges_pds_all,edges_w,edge_pds)
                    
                    
                    inst_mask2instance = data['mask2instance']
                    
                    gts = None
                    node_colors = data['node_colors']
                    node_colors_all = {**node_colors_all,**node_colors}
                    
                    g = graphDrawer.draw({'nodes':nodes_pds_all, 'edges':edges_pds_all},
                                          node_colors=node_colors_all)
                    g.render(os.path.join(cfg['training']['out_dir'], 
                                              cfg.name, 'full_graph'),view=cfg.vis)
                    
                    
                    '''draw 3DOBB'''
                    pds = to_name_dict({'nodes':nodes_pds_all, 'edges':edges_pds_all},
                                       node_cls_names,
                                       edge_cls_names)
                    mesh = bbox_drawer(scan_id, cfg.filename, pds)
                    trimesh.Scene(mesh).show()
                    
                    
                    
        print('done')
    else:
        model_trainer = config.get_trainer(cfg, model, node_cls_names, edge_cls_names,
                                            w_node_cls=None,
                                            w_edge_cls=None
                                            )
        
        '''start eval'''
        logger_py.info('start evaluation')
        pr = cProfile.Profile()
        pr.enable()    
        eval_dict, eval_tool = model_trainer.visualize_inst_incre(dataset_seg, topk=cfg.eval.topK)
        pr.disable()
        # logger_py.info('save time profile to {}'.format(os.path.join(out_dir,'tp_eval_inst.dmp')))
        # pr.dump_stats(os.path.join(out_dir,'tp_eval_inst.dmp'))
        
        '''log'''
        # ignore_missing=cfg.eval.ignore_missing
        prefix='incre_inst' if not cfg.eval.ignore_missing else 'incre_inst_ignore'
        
        print(eval_tool.gen_text())
        _ = eval_tool.write(out_dir, prefix)
        if logger:
            for k,v in eval_dict['visualization'].items(): 
                logger.add_figure('test/'+prefix+'_'+k, v, global_step=it)
            for k, v in eval_dict.items():
                if isinstance(v,dict): continue
                logger.add_scalar('test/'+prefix+'_'+'%s' % k, v, it)

class InputProcessor(object):
    def __init__(self, cfg, device):
        self.cfg=cfg
        self._device=device
        self.transform = transforms.Compose([
                    transforms.Resize([256,256]),
                    ])
        self.DRAW_BBOX_IMAGE=False
        self.min_3D_bbox_size=0.2*0.2*0.2
        # self.occ_thres=0.5
        self.debug = False
    def __call__(self,path_graph):
        '''load'''
        with open(path_graph,'r') as f:
            sdata = json.load(f)
            
        min_3D_bbox_size = self.min_3D_bbox_size
        # occ_thres = self.occ_thres
        debug = self.debug
        for scan_id, data in sdata.items():
            nodes = data['nodes']

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

                segs_neighbors[int(seg_id)] = node['neighbors']
        
                segments_pd_filtered.append(seg_id)
                
                [kf_ids_from_nodes.add(x) for x in node['kfs']]
            if len(segments_pd_filtered) == 0:
                continue

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
            '''check if each node has at least one kf'''
            to_deletes = []
            for k,v in objects.items():
                if int(k) not in node2kfs or len(node2kfs[int(k)])==0 or k not in segments_pd_filtered:
                    to_deletes.append(k)
            for idx in to_deletes:
                objects.pop(idx)

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
            # cat = []
            counter = 0
            filtered_instances = list()
            kf_indices = set()
            for instance_id in instances_ids:    
                # class_id = -1
                # instance_labelName = instance2labelName[instance_id]
                # if instance_labelName in self.classNames:
                #     class_id = self.classNames.index(instance_labelName)
    
                # mask to cat:
                if instance_id > 0: # insstance 0 is unlabeled.
                    oid2idx[int(instance_id)] = counter
                    idx2oid[counter] = int(instance_id)
                    counter += 1
                    filtered_instances.append(instance_id)
                    # cat.append(class_id)
                    
                    # [kf_indices.add(x) for x in node2kfs[instance_id]]

            ''' build nn dict '''
            nns = dict()
            # seg2inst = dict()
            for oid in filtered_instances:
                nns[str(oid)] = [int(s) for s in nodes[str(oid)]['neighbors']]
                
                '''build instance dict'''
                # seg2inst[oid] = map_segment_pd_2_gt[oid]
                # if 'instance_id' in odata:
                #     seg2inst[oid] = odata['instance_id']
                
            toTensor = transforms.ToTensor()
            fdata = os.path.join(define.DATA_PATH)                
            resize = transforms.Resize([256,256])
            rgb_filepattern = 'frame-{0:06d}.color.jpg'
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

                    #  bounding box
                    kf = kfs[str(fid)]                
                    box = np.asarray(kf['bboxes'][str(oid)])[:4]# kfdata[bid,:4]
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
                if self.DRAW_BBOX_IMAGE:
                    show_tensor_images(img_boxes, title=objects[str(oid)]['label'])
                img_boxes= normalize_imagenet(img_boxes.float()/255.0)
                bounding_boxes.append(img_boxes)
                # h5d = h5f.create_dataset(oid,data=img_boxes.numpy(), compression="gzip", compression_opts=9)
                # h5d.attrs['seg2idx'] = fidx2idx          
            
            descriptor_generator = util_data.Node_Descriptor_24(with_bbox=self.cfg.data.img_desc_6_pts)
            node_descriptor_for_image = torch.zeros([len(filtered_instances), len(descriptor_generator)])
            for i in range(len(filtered_instances)):
                instance_id = filtered_instances[i]
                obj = nodes[str(instance_id)]
                obj['normAxes'] =  copy.deepcopy( np.array(obj['rotation']).reshape(3,3).transpose().tolist() )
                node_descriptor_for_image[i] = descriptor_generator(obj)

            '''sample connections'''
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
            # edge_indices = list()
            # for n in range(len(filtered_instances)):
            #     for m in range(len(filtered_instances)):
            #         if n == m:continue
            #         edge_indices.append([n,m])
                    
            edge_indices = [[l[0],l[1]] for l in edge_indices]

            edge_indices = torch.tensor(edge_indices,dtype=torch.long)
            
            
        
            '''get node color if available'''
            color_dict = dict()
            for iid in instance2labelName:
                # color = [0,0,0]
                if 'color' in nodes[str(iid)]:
                    color = nodes[str(iid)]['color']
                    color_dict[iid] = [color[2],color[1],color[0]]
                    # color_dict[iid] = torch.tensor(color,dtype=torch.long)
        
        
            output = dict()
            output['scan_id'] = scan_id
            output['roi_imgs'] = bounding_boxes #list
            output['node_descriptor_8'] = node_descriptor_for_image
            output['node_edges'] = edge_indices # tensor
            output['node_colors'] = color_dict
            output['instance2mask'] = oid2idx #dict
            output['mask2instance'] = idx2oid
            # output['seg2inst'] = seg2inst
            yield output
            
                # outputs.append(output)

    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors
    
        Args:
            data (dictionary): data dictionary
        '''
        try:
            data =  dict(zip(data.keys(), self.toDevice(*data.values()) ))
        except:
            '''if failed, so more info'''
            print('')
            # print('type(data)',type(data))
            if not isinstance(data,dict):
                raise RuntimeError('expect input data with type dict but got {}'.format(type(data)))
            '''convert individually until error happen'''
            for k, v in data.items():
                try:
                    self.toDevice(v)
                except:
                    raise RuntimeError('unable to convert the object of {} with type {} to device {}'.format(k,type(v),self._device))
        return data
    def toDevice(self, *args):
        output = list()
        for item in args:
            if isinstance(item,  torch.Tensor):
                output.append(item.to(self._device,non_blocking=True))
            elif isinstance(item,  dict):
                ks = item.keys()
                vs = self.toDevice(*item.values())
                item = dict(zip(ks, vs))
                output.append(item)
            elif isinstance(item, list):
                output.append(self.toDevice(*item))
            else:
                output.append(item)
        return output


if __name__ == '__main__':
    # logger_py.setLevel('DEBUG')
    # logger_py.debug('hello0')
    main()