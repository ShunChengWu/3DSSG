import json,trimesh,os,argparse
import pickle
import pandas as pd
import numpy as np
from ssg.utils import util_3rscan
from codeLib.common import rand_24_bit, color_rgb
from codeLib.geoemetry.common import create_box
from collections import defaultdict
from  box import Box
from iou import IoU
from pathlib import Path
from tqdm import tqdm


'''define'''
base_path = './data/3RScan/data/3RScan'
filename_3rscan_json = os.path.join(base_path,'3RScan.json')
# filename='2dssg_orbslam3.json'
OBJ_NAME = 'mesh.refined.obj'

def Parser(add_help=True):
    parser = argparse.ArgumentParser(description='Compute IoU curve for all input sequences.', formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     add_help=add_help)
    parser.add_argument('--pth_out', type=str,default='./evalLabelMapping_ratio_gt/', help='pth to output directory',required=False)    
    parser.add_argument('--filename', type=str,default='2dssg_orbslam3',required=False)    
    parser.add_argument('--overwrite',action='store_true')
    parser.add_argument('--vis',action='store_true')
    return parser

def calculate_label_size(pts,labels):
    output=dict()
    unique_labels = np.unique(labels)
    for l in unique_labels:
        lpts = pts[np.where(labels==l)]
        output[l] = len(lpts)
    return output


def eval_one_scene_ratio(scan_id,filename)->dict:
    pth_est = os.path.join(base_path,scan_id,filename+'.json')
    pth_ply = os.path.join(base_path,scan_id,filename+'.ply')
    pth_semseg= os.path.join(base_path,scan_id,'semseg.v2.json')    
    
    # Load gt data
    with open(pth_semseg) as f: 
        data_gt = json.load(f)
    # GT
    gt_dict = dict()
    for group in data_gt['segGroups']:
        gt_dict[group['objectId']] = group['obb']
        
    # Load est data
    with open(pth_est) as f: 
        data_est = json.load(f)
    if 'nodes' not in data_est:
        nodes = data_est[scan_id]['nodes']
    else:
        nodes = data_est['nodes']
        
    # load pcs
    pt_size_pd={}
    pt_size_gt={}
    dominant_gt_ratio=dict()
    dominant_ratio=dict()
    gt2seg_dict=defaultdict(list)
    if len(nodes) > 0:
        plydata = trimesh.load(pth_ply, process=False)
        points = np.array(plydata.vertices)
        text_ply_raw = 'ply_raw' if 'ply_raw' in plydata.metadata else '_ply_raw'
        labels = plydata.metadata[text_ply_raw]['vertex']['data']['label'].flatten()
        gt_labels = plydata.metadata[text_ply_raw]['vertex']['data']['gtinstance'].flatten()
        
        pt_size_pd = calculate_label_size(points,labels)
        pt_size_gt = calculate_label_size(points,gt_labels)
        
        '''est dist'''
        for idx,node in nodes.items():
            idx = int(idx)
            gt_insts = node['gtInstance']
            
            max_v=0
            max_k=0
            for k,v in gt_insts.items():
                if v>max_v:
                    max_v=v
                    max_k=int(k)
            dominant_ratio[idx] = max_v
            
            gt2seg_dict[max_k].append(idx)
            
            # take the maximum one
            # assert max_k not in est_dict # otherwise need to do merging
            # est_dict[idx] = max_v
            if max_k not in dominant_gt_ratio or max_v > dominant_gt_ratio[max_k]:
                dominant_gt_ratio[max_k] = max_v    
        
    '''find dominant'''
    dominant_gt_ratio_all=dict()
    for gt_inst in gt_dict:
        if gt_inst not in dominant_gt_ratio:
            dominant_gt_ratio_all[gt_inst] = 0
            continue
        dominant_gt_ratio_all[gt_inst] = dominant_gt_ratio[gt_inst]
    
    output=dict()
    output['pt_size_pd'] = pt_size_pd
    output['pt_size_gt'] = pt_size_gt
    output['dominant_gt_ratio'] = dominant_gt_ratio_all
    output['dominant_pd_ratio'] = dominant_ratio
    output['gt2seg_dict'] = gt2seg_dict
    return output

def segmentation_covering(size_dict:dict, max_overlap_dict:dict):
    sc=0
    totoal_pt_size=0
    for idx in size_dict.keys():
        totoal_pt_size += size_dict[idx]
        sc += size_dict[idx] * max_overlap_dict[idx]
    sc /= totoal_pt_size
    return sc
        
def eval_ratio(train_list,output_path):
    '''eval'''
    for scan_id in tqdm(train_list,desc='compute...'):
        # scan_id = '4acaebcc-6c10-2a2a-858b-29c7e4fb410d'
        
        pth_out = os.path.join(output_path,scan_id+'.pkl')
        
        # check exist
        if not args.overwrite and os.path.exists(pth_out):
            continue
        
        data = eval_one_scene_ratio(scan_id,filename)
        with open(pth_out,'wb') as f:
            pickle.dump(data,f)
        
        # with open(pth_out,'w') as f:
        #     for k,v in ious.items():
        #         f.write('{}\t{}\n'.format(k,v))
                
    '''generate cruve'''
    # read all into a single array
    data_array = list()
    per_scan_dict = dict()
    perscene_sc = dict()
    purity_ratio=list() # dominant gt ratio
    for scan_id in tqdm(train_list,desc='load...'):
        pth_out = os.path.join(output_path,scan_id+'.pkl')
        try:
            with open(pth_out,'rb') as f:
                data = pickle.load(f)
            
            '''compute segmentation covering'''
            perscene_sc[scan_id] = segmentation_covering(data['pt_size_pd'],data['dominant_pd_ratio'])
            
            purity_ratio += data['dominant_pd_ratio'].values()
            
            # data_array+= list(data[1])
            # per_scan_dict[scan_id] = np.asarray(data[1]).mean()
        except:
            pass
    purity_ratio = np.asarray(purity_ratio).mean()
    
    '''calculate mean sc'''
    all_sc=0
    
    num_valid_scan = 0
    num_scan = 0
    for scan_id in tqdm(train_list,desc='calculate mean sc'):
        if scan_id in perscene_sc:
            all_sc += perscene_sc[scan_id]
            num_valid_scan+=1
        num_scan+=1
    mean_sc_valid = all_sc / num_valid_scan
    mean_sc       = all_sc / num_scan
    
    # data_array = np.array(data_array)
    
    # mean_score = data_array.mean()
    # with open(os.path.join(args.pth_out, filename+'.txt'),'w') as f:
    #     f.write('all {}\n'.format(mean_score))
        
    #     tmp=list()
    #     for k,v in per_scan_dict.items():
    #         tmp.append(v)
    #     tmp = np.asarray(tmp).mean()
                
    #     f.write('per-scene mean {}'.format(tmp))
    
    # recall_at_k = dict()
    # for th in np.linspace(0,100,101):
    #     th /= 100
    #     recall = (data_array>=th).sum()/data_array.size
    #     recall_at_k[str(th)] = recall    
    with open(os.path.join(args.pth_out, filename+'.txt'),'w') as f:
        f.write('sc:\t{}\n'.format(mean_sc))
        f.write('sc_valid:\t{}\n'.format(mean_sc_valid))
        f.write('valid ratio: {}({}/{})\n'.format(num_valid_scan/num_scan,num_valid_scan,num_scan))
        f.write('purity_ratio:\t{}\n'.format(purity_ratio))
        
        # for k,v in recall_at_k.items():
        #         f.write('{}\t{}\n'.format(k,v))
    
    
def eval_one_scene_iou(scan_id,filename, vis:bool) -> dict:
    pth_est = os.path.join(base_path,scan_id,filename+'.json')
    pth_semseg= os.path.join(base_path,scan_id,'semseg.v2.json')    
    
    # Load gt data
    with open(pth_semseg) as f: 
        data_gt = json.load(f)
        
    # Load est data
    with open(pth_est) as f: 
        data_est = json.load(f)
    if 'nodes' not in data_est:
        nodes = data_est[scan_id]['nodes']
    else:
        nodes = data_est['nodes']

    if vis: 
        meshes = list()
        '''load GT mesh'''
        pth_folder = os.path.join(base_path,scan_id,OBJ_NAME)
        mesh = trimesh.load(pth_folder, process=False)
        meshes.append(mesh)

    gt_dict = dict()
    # print('=====')
    for group in data_gt['segGroups']:
        gt_dict[group['objectId']] = group['obb']
    # print('=====')
    est_dict = dict()
    for idx,node in nodes.items():
        gt_insts = node['gtInstance']
        
        max_v=0
        max_k=0
        for k,v in gt_insts.items():
            if v>max_v:
                max_v=v
                max_k=int(k)
                
        # take the maximum one
        # assert max_k not in est_dict # otherwise need to do merging
        est_dict[max_k] = idx
        
    '''draw'''
    iou_list = dict()
    for gt_inst in gt_dict:
        if gt_inst not in est_dict:
            iou_list[gt_inst] = 0
            # print('gt_inst not exist')
            continue

        clr = color_rgb(rand_24_bit())
        clr_est = [int(c*0.5) for c in clr]
        

        '''GT'''
        # get info
        obb = gt_dict[gt_inst]
        mat44 = np.eye(4)
        mat44[:3,:3] = np.array(obb['normalizedAxes']).reshape(3,3).transpose()
        mat44[:3,3] = obb['centroid']
        # create iou box
        iou_box_gt = Box.from_transformation(mat44[:3,:3],mat44[:3,3],obb['axesLengths'])
        # create trimesh box
        if vis: 
            box_gt = create_box(obb['axesLengths'],0.05)
            box_gt.apply_transform(mat44)
            box_gt.visual.vertex_colors[:,:3] = clr
        
        '''Est'''
        # get info
        node = nodes[est_dict[gt_inst]]
        Rinv = np.array(node['rotation']).reshape(3,3)
        R = np.transpose(Rinv)
        center = np.array(node['center']) 
        dims = np.array(node['dimension'])
        mat44 = np.eye(4)
        mat44[:3,:3] = R
        mat44[:3,3] = center
        # create iou box
        iou_box_est = Box.from_transformation(mat44[:3,:3],mat44[:3,3],dims)
        if dims[0] == 0 or dims[1] == 0 or dims[2] == 0:
            iou_list[gt_inst] = 0
            # print('dim==0')
            continue
        
        # create trimesh box
        if vis: 
            box_est = create_box(dims,0.05)
            box_est.apply_transform(mat44)
            box_est.visual.vertex_colors[:,:3] = clr_est
        
        '''measure'''
        iou = IoU(iou_box_gt,iou_box_est).iou()
        # print('iou',iou)
        
        if gt_inst not in iou_list or iou > iou_list[gt_inst]:
            iou_list[gt_inst] = iou

        if vis: 
            clr = [int(255*iou) for _ in range(3)]
            box_gt.visual.vertex_colors[:,:3] = clr
            box_est.visual.vertex_colors[:,:3] = clr
            meshes.append(box_gt)
            meshes.append(box_est)
    if vis:
        trimesh.Scene(meshes).show()
    return iou_list
         
def eval_ious(train_list, output_path):
    '''eval'''
    for scan_id in tqdm(train_list,desc='compute...'):
        # scan_id = '4acaebcc-6c10-2a2a-858b-29c7e4fb410d'
        
        pth_out = os.path.join(output_path,scan_id+'.txt')
        
        # check exist
        if not args.overwrite and os.path.exists(pth_out):
            continue
        
        ious = eval_one_scene_iou(scan_id,filename,vis)
        
        with open(pth_out,'w') as f:
            for k,v in ious.items():
                f.write('{}\t{}\n'.format(k,v))
        # break
        
    '''generate cruve'''
    # read all into a single array
    data_array = list()
    for scan_id in tqdm(train_list,desc='load...'):
        pth_out = os.path.join(output_path,scan_id+'.txt')
        data = pd.read_csv(pth_out,sep='\t',header=None)
        data_array+= list(data[1])
        # print(data)
    data_array = np.array(data_array)
    
    recall_at_k = dict()
    for th in np.linspace(0,100,101):
        th /= 100
        recall = (data_array>=th).sum()/data_array.size
        recall_at_k[str(th)] = recall
        
    with open(os.path.join(args.pth_out, filename+'.txt'),'w') as f:
        for k,v in recall_at_k.items():
                f.write('{}\t{}\n'.format(k,v))
    
if __name__ == '__main__':
    args = Parser().parse_args()
    vis = args.vis
    filename = args.filename
    # create
    output_path = os.path.join(args.pth_out,filename)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    '''get split'''
    train_list, _ = util_3rscan.get_train_val_split(pth_3rscan_json=filename_3rscan_json)
    
    # eval_ious(train_list,output_path)
        
    eval_ratio(train_list,output_path)