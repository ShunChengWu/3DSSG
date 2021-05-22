if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
from utils import util_ply
import trimesh
import open3d as o3d
import numpy as np
from utils import define, util
from utils import util_ply, util_label, util, define
from utils.util_search import SAMPLE_METHODS,find_neighbors
from tqdm import tqdm
from pathlib import Path
import os,json    
import argparse

def Parser(add_help=True):
    parser = argparse.ArgumentParser(description='Process some integers.', formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                     add_help=add_help)
    parser.add_argument('--scans', type=str,default='/media/sc/SSD1TB/dataset/3RScan/data/3RScan/')
    parser.add_argument('--type', type=str, default='train', choices=['train', 'test', 'validation'], help="allow multiple rel pred outputs per pair",required=False)
    parser.add_argument('--pth_out', type=str,default='../data/tmp', help='pth to output directory',required=True)
    parser.add_argument('--relation', type=str,default='relationships', choices=['relationships_extended', 'relationships'])
    parser.add_argument('--target_scan', type=str, default='', help='')
    parser.add_argument('--label_type', type=str,default='3RScan160', choices=['3RScan160'], help='label',required=False)
    
    # options
    parser.add_argument('--mapping',type=int,default=1,
                        help='map label from 3RScan to label_type. otherwise filter out labels outside label_type.')
    parser.add_argument('--v2', type=int,default=1,
                        help='v2 version')
    parser.add_argument('--verbose', type=bool, default=False, help='verbal',required=False)
    parser.add_argument('--debug', type=int, default=0, help='debug',required=False)
    
    # neighbor search parameters
    parser.add_argument('--search_method', type=str, choices=['BBOX','KNN'],default='BBOX',help='How to split the scene.')
    parser.add_argument('--radius_receptive', type=float,default=0.5,help='The receptive field of each seed.')
    
    # split parameters
    parser.add_argument('--split', type=int,default=0,help='Split scene into groups.')
    parser.add_argument('--radius_seed', type=float,default=1,help='The minimum distance between two seeds.')
    parser.add_argument('--min_segs', type=int,default=5,help='Minimum segments for each segGroup')
    parser.add_argument('--split_method', type=str, choices=['BBOX','KNN'],default='BBOX',help='How to split the scene.')
    
    return parser

name_same_segment = 'same part'

def generate_groups(cloud:trimesh.points.PointCloud, distance:float=1, bbox_distance:float=0.75, 
                    min_seg_per_group = 5, segs_neighbors=None):
    points = np.array(cloud.vertices.tolist())
    segments = cloud.metadata['ply_raw']['vertex']['data']['label'].flatten()
    seg_ids = np.unique(segments)
    selected_indices = list()
    
    index = np.random.choice(range(len(points)),1)
    selected_indices.append(index)
    should_continue = True
    while should_continue:
        distances_pre=None
        for index in selected_indices:
            point = points[index]
            distances = np.linalg.norm(points[:,0:2]-point[:,0:2],axis=1) # ignore z axis.
            if distances_pre is not None:
                distances = np.minimum(distances, distances_pre)
            distances_pre = distances
        selectable = np.where(distances > distance)[0]
        if len(selectable) < 1: 
            should_continue=False
            break
        index = np.random.choice(selectable,1)
        selected_indices.append(index)
        
    if args.verbose:print('num of selected point seeds:',len(selected_indices))

    
    if debug:
        seg_colors = dict()
        for index in seg_ids:
            seg_colors[index] = util.color_rgb(util.rand_24_bit())
        counter=0
    '''Get segment groups'''
    seg_group = list()
    
    ''' Building Box Method '''  
    from enum import Enum
    class SAMPLE_METHODS(Enum):
        BBOX=1
        RADIUS=2
    if args.split_method == 'BBOX':
        sample_method = SAMPLE_METHODS.BBOX
    elif args.split_method == 'KNN':
        sample_method = SAMPLE_METHODS.RADIUS
    
    if sample_method == SAMPLE_METHODS.BBOX:
        for index in selected_indices:
            point = points[index]
            min_box = (point-bbox_distance)[0]
            max_box = (point+bbox_distance)[0]
            
            filter_mask = (points[:,0] > min_box[0]) * (points[:,0] < max_box[0]) \
                            * (points[:,1] > min_box[1]) * (points[:,1] < max_box[1]) \
                            * (points[:,2] > min_box[2]) * (points[:,2] < max_box[2])
                            
            filtered_segments = segments[np.where(filter_mask > 0)[0]]
            segment_ids = np.unique(filtered_segments) 
            # print('segGroup {} has {} segments.'.format(index,len(segment_ids)))
            if len(segment_ids) < min_seg_per_group: continue
            seg_group.append(segment_ids.tolist())
            
            if debug:
                '''Visualize the segments involved'''
                cloud.visual.vertex_colors = [0,0,0,255]
                for segment_id in segment_ids:
                    segment_indices = np.where(segments == segment_id )[0]
                    for idx in segment_indices:
                        cloud.visual.vertex_colors[idx][:3] = seg_colors[segment_id]
                cloud.export('tmp'+str(counter)+'.ply')
                counter+=1
    elif sample_method == SAMPLE_METHODS.RADIUS:
        radknn = 0.1
        n_layers = 2
        trees = dict()
        segs  = dict()
        bboxes = dict()
        for idx in seg_ids:
            segs[idx] = points[np.where(segments==idx)]
            trees[idx] = o3d.geometry.KDTreeFlann(segs[idx].transpose())
            bboxes[idx] = [segs[idx].min(0)-radknn,segs[idx].max(0)+radknn]

        # search neighbor for each segments
        if segs_neighbors is None:
            segs_neighbors = find_neighbors(points, segments, search_method,receptive_field=args.radius_receptive)

        def cat_neighbors(idx:int, neighbor_list:dict):
            output = set()
            for n in neighbor_list[idx]:
                output.add(n)
            return output
        
        for idx in selected_indices:
            seg_id =segments[idx][0]
            neighbors = set()
            neighbors.add(seg_id)
            nn_layers = dict()
            for i in range(n_layers):
                nn_layers[i] = set()
                for j in neighbors:
                    new_nn = cat_neighbors(j, segs_neighbors)
                    nn_layers[i] = nn_layers[i].union(new_nn)
                neighbors = neighbors.union(nn_layers[i])
            
            # print(idx, nn_layers)
            for i in range(n_layers):
                for j in range(i+1, n_layers):
                    nn_layers[j] = nn_layers[j].difference(nn_layers[i])
            # print(idx, nn_layers)
            
            if len(neighbors) < min_seg_per_group: continue
            seg_group.append(neighbors)
            
            if debug:
                '''Visualize the segments involved'''
                cloud.visual.vertex_colors = [0,0,0,255]
                for segment_id in neighbors:
                    segment_indices = np.where(segments == segment_id )[0]
                    for idx in segment_indices:
                        cloud.visual.vertex_colors[idx][:3] = seg_colors[segment_id]
                cloud.export('tmp'+str(counter)+'.ply')
                counter+=1
    return seg_group

def process(pth_3RScan, scan_id,
            target_relationships:list,
            gt_relationships:dict=None, verbose=False,split_scene=True) -> list:
    pth_gt = os.path.join(pth_3RScan,scan_id, define.LABEL_FILE_NAME)
    segseg_file_name = define.SEMSEG_FILE_NAME

    # load gt
    cloud_gt = trimesh.load(pth_gt, process=False)
    points_gt = np.array(cloud_gt.vertices.tolist())
    segments_gt = util_ply.get_label(cloud_gt, '3RScan', 'Segment').flatten()
    
    segs_neighbors = find_neighbors(points_gt, segments_gt, search_method,receptive_field=args.radius_receptive)
    relationships_new['neighbors'][scan_id] = segs_neighbors
    
    segment_ids = np.unique(segments_gt) 
    segment_ids = segment_ids[segment_ids!=0]

    if split_scene:
        seg_groups = generate_groups(cloud_gt,args.radius_seed,args.radius_receptive,args.min_segs,
                                     segs_neighbors=segs_neighbors)
        if args.verbose:
            print('final segGroups:',len(seg_groups))
    else:    
        seg_groups = None

    _, label_name_mapping, _ = util_label.getLabelMapping(args.label_type)
    pth_semseg_file = os.path.join(pth_3RScan, scan_id, segseg_file_name)
    instance2labelName = util.load_semseg(pth_semseg_file, label_name_mapping,args.mapping)
    
    ''' Find and count all corresponding segments'''
    size_segments_gt = dict()
    map_segment_pd_2_gt = dict() # map segment_pd to segment_gt
    for segment_id in segment_ids:
        segment_indices = np.where(segments_gt == segment_id)[0]
        segment_points = points_gt[segment_indices]        
        size_segments_gt[segment_id] = len(segment_points)
        map_segment_pd_2_gt[segment_id]=segment_id
    
    ''' Save as ply '''
    if debug:
        for seg, label_name in instance2labelName.items():
            segment_indices = np.where(segments_gt == seg)[0]
            if label_name != 'none':
                continue
            for index in segment_indices:
                cloud_gt.visual.vertex_colors[index][:3] = [0,0,0]
        cloud_gt.export('tmp_gtcloud.ply')

    
    '''' Save as relationship_*.json '''
    list_relationships = list()
    if seg_groups is not None:
        for split_id in range(len(seg_groups)):
            seg_group = seg_groups[split_id]
            relationships = gen_relationship(scan_id,split_id,map_segment_pd_2_gt, instance2labelName,seg_group)
            if len(relationships["objects"]) == 0 or len(relationships['relationships']) == 0:
                continue
            list_relationships.append(relationships)
            
            ''' check '''
            for obj in relationships['objects']:
                assert(obj in seg_group)
            for rel in relationships['relationships']:
                assert(rel[0] in relationships['objects'])
                assert(rel[1] in relationships['objects'])
    else:
        relationships = gen_relationship(scan_id,0, map_segment_pd_2_gt, instance2labelName)
        if len(relationships["objects"]) != 0 and len(relationships['relationships']) != 0:
                list_relationships.append(relationships)
    
    return list_relationships, segs_neighbors


def gen_relationship(scan_id:str,split:int, map_segment_pd_2_gt:dict,instance2labelName:dict,
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
        if name == '-' or name == 'none':
            continue
        objects[int(seg)] = name #labels_utils.NYU40_Label_Names[label-1]
    relationships["objects"] = objects
    
    
    split_relationships = list()
    ''' Inherit relationships from ground truth segments '''
    if gt_relationships is not None:
        relationships_names = util.read_relationships(os.path.join(define.FILE_PATH, args.relation + ".txt"))
        for rel in gt_relationships:
            id_src = rel[0]
            id_tar = rel[1]
            num = rel[2]
            name = rel[3]
            idx_in_txt = relationships_names.index(name)
            assert(num==idx_in_txt)
            if name not in target_relationships: 
                # if debug:print('filter',name,'it is not in the target relationships')
                continue
            idx_in_txt_new = target_relationships.index(name)
            
            
            
            split_relationships.append([ int(id_src), int(id_tar), idx_in_txt_new, name ])
            
    relationships["relationships"] = split_relationships
    return relationships
    
if __name__ == '__main__':
    args = Parser().parse_args()
    debug = args.debug > 0
    if args.search_method == 'BBOX':
        search_method = SAMPLE_METHODS.BBOX
    elif args.search_method == 'KNN':
        search_method = SAMPLE_METHODS.RADIUS
    
    util.set_random_seed(2020)
    
    ''' Map label to 160'''
    label_names = sorted(util.read_classes(define.CLASS160_FILE))
    # target_relationships = sorted(util.read_classes(define.RELEASE_PATH + '/classes160.txt'))
    target_relationships = ['supported by', 'attached to','standing on','hanging on','connected to','part of','build in']
    classes_json = list()
    for name in label_names:
        if name == '-':continue
        classes_json.append(name)
        
    ''' Read Scan and their type=['train', 'test', 'validation'] '''
    scan2type = {}
    with open(define.Scan3RJson_PATH, "r") as read_file:
        data = json.load(read_file)
        for scene in data:
            scan2type[scene["reference"]] = scene["type"]
            for scan in scene["scans"]:
                scan2type[scan["reference"]] = scene["type"]
                
    target_scan=[]
    if args.target_scan != '':
        target_scan = util.read_txt_to_list(args.target_scan)
            
    valid_scans=list()
    relationships_new = dict()
    relationships_new["scans"] = list()
    relationships_new['neighbors'] = dict()
    counter= 0
    with open(os.path.join(define.FILE_PATH + args.relation + ".json"), "r") as read_file:
        data = json.load(read_file)
        for s in tqdm(data["scans"]):
        # for s in data["scans"]:
            scan_id = s["scan"]
            
            if len(target_scan) ==0:
                if scan2type[scan_id] != args.type: 
                    if args.verbose:
                        print('skip',scan_id,'not validation type')
                    continue
            else:
                if scan_id not in target_scan: continue
            
            gt_relationships = s["relationships"]
            if debug:print('processing scene',scan_id)
            valid_scans.append(scan_id)
            relationships, segs_neighbors = process(args.scans, scan_id, target_relationships,
                                    gt_relationships = gt_relationships,
                                    split_scene = args.split,
                                    verbose = args.verbose)
            if len(relationships) == 0:
                print('skip',scan_id,'due to not enough objs and relationships')
                continue
            else:
                print('no skip', scan_id)
            
            relationships_new["scans"] += relationships
            relationships_new['neighbors'][scan_id] = segs_neighbors
            
            if debug:
                break
            
    Path(args.pth_out).mkdir(parents=True, exist_ok=True)
    pth_args = os.path.join(args.pth_out,'args.json')
    with open(pth_args, 'w') as f:
            tmp = vars(args)
            json.dump(tmp, f, indent=2)
    
    pth_relationships_json = os.path.join(args.pth_out, "relationships_" + args.type + ".json")
    with open(pth_relationships_json, 'w') as f:
        json.dump(relationships_new, f)
    pth_classes = os.path.join(args.pth_out, 'classes.txt')
    with open(pth_classes,'w') as f:
        for name in classes_json:
            if name == '-': continue
            f.write('{}\n'.format(name))
    pth_relation = os.path.join(args.pth_out, 'relationships.txt')
    with open(pth_relation,'w') as f:
        for name in target_relationships:
            f.write('{}\n'.format(name))
    pth_split = os.path.join(args.pth_out, args.type+'_scans.txt')
    with open(pth_split,'w') as f:
        for name in valid_scans:
            f.write('{}\n'.format(name))