from collections import defaultdict
import os
import json
import trimesh
import argparse
import pathlib
import open3d as o3d
import numpy as np
from pathlib import Path
from tqdm import tqdm
import codeLib
from ssg import define
from ssg.utils import util_3rscan, util_label, util_ply
from ssg.utils.util_search import SAMPLE_METHODS, find_neighbors
import h5py
import ast
import copy
import logging


def Parser(add_help=True):
    parser = argparse.ArgumentParser(description='Generate custom scene graph dataset from the 3RScan dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     add_help=add_help)
    parser.add_argument(
        '-c', '--config', default='./configs/config_default.yaml', required=False)
    parser.add_argument('-o', '--pth_out', type=str, default='../data/tmp',
                        help='pth to output directory', required=True)
    parser.add_argument('--target_scan', type=str, default='', help='')
    parser.add_argument('-l', '--label_type', type=str, default='3RScan160',
                        choices=['3RScan160', 'ScanNet20'], help='label', required=False)
    parser.add_argument('--only_support_type', action='store_true',
                        help='use only support type of relationship')

    parser.add_argument('--debug', action='store_true',
                        help='debug', required=False)
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite or not.')

    # constant
    parser.add_argument('--segment_type', type=str, default='InSeg')
    return parser


class GenerateSceneGraph(object):
    def __init__(self, cfg: dir, target_relationships: list, label_type: str):
        self.cfg = cfg
        self.target_relationships = target_relationships
        self.label_type = label_type

    def __call__(self, scan_id, gt_relationships):
        # some params
        pth_3RScan_data = self.cfg.data.path_3rscan_data
        lcfg = self.cfg.data.scene_graph_generation
        target_relationships = self.target_relationships
        pth_gt = os.path.join(pth_3RScan_data, scan_id,
                              self.cfg.data.label_file_gt)
        pth_pd = os.path.join(pth_3RScan_data, scan_id,
                              self.cfg.data.label_file)
        max_distance = lcfg.max_dist
        # if the num of points within a segment below this threshold, discard this
        filter_segment_size = lcfg.min_seg_size
        # if percentage of the corresponding label must exceed this value to accept the correspondence
        filter_corr_thres = lcfg.corr_thres
        filter_occ_ratio = lcfg.occ_thres

        # load gt
        cloud_gt = trimesh.load(pth_gt, process=False)
        points_gt = np.array(cloud_gt.vertices.tolist()).transpose()
        segments_gt = util_ply.get_label(
            cloud_gt, '3RScan', 'Segment').flatten()

        # load pd
        cloud_pd = trimesh.load(pth_pd, process=False)
        cloud_pd.apply_scale(lcfg.point_cloud_scale)
        points_pd = np.array(cloud_pd.vertices.tolist())
        segments_pd = util_ply.get_label(
            cloud_pd, '3RScan', 'Segment').flatten()

        # get num of segments
        segment_ids = np.unique(segments_pd)
        segment_ids = segment_ids[segment_ids != 0]

        # Filter segments
        # if cfg.VERBOSE: print('filtering input segments.. (ori num of segments:',len(segment_ids),')')
        segments_pd_filtered = list()
        for seg_id in segment_ids:
            pts = points_pd[np.where(segments_pd == seg_id)]
            if len(pts) > filter_segment_size:
                segments_pd_filtered.append(seg_id)
        segment_ids = segments_pd_filtered
        # if cfg.VERBOSE: print('there are',len(segment_ids), 'segemnts:\n', segment_ids)

        # Find neighbors of each segment
        segs_neighbors = find_neighbors(
            points_pd, segments_pd, search_method,
            receptive_field=lcfg.radius_receptive,
            selected_keys=segment_ids)
        # if cfg.VERBOSE: print('segs_neighbors:\n',segs_neighbors.keys())

        # get label mapping
        _, label_name_mapping, _ = util_label.getLabelMapping(self.label_type)
        pth_semseg = os.path.join(
            pth_3RScan_data, scan_id, define.SEMSEG_FILE_NAME)
        instance2labelName = util_3rscan.load_semseg(
            pth_semseg, label_name_mapping)

        '''extract object bounding box info'''
        objs_obbinfo = dict()
        pth_obj_graph = os.path.join(pth_3RScan_data, scan_id, lcfg.graph_name)
        with open(pth_obj_graph) as f:
            data = json.load(f)
        for nid, node in data[scan_id]['nodes'].items():
            obj_obbinfo = objs_obbinfo[int(nid)] = dict()
            obj_obbinfo['center'] = copy.deepcopy(node['center'])
            obj_obbinfo['dimension'] = copy.deepcopy(node['dimension'])
            obj_obbinfo['normAxes'] = copy.deepcopy(
                np.array(node['rotation']).reshape(3, 3).transpose().tolist())
        del data

        # count gt segment size
        size_segments_gt = dict()
        for segment_id in segments_gt:
            segment_indices = np.where(segments_gt == segment_id)[0]
            size_segments_gt[segment_id] = len(segment_indices)

        ''' Find and count all corresponding segments'''
        tree = o3d.geometry.KDTreeFlann(points_gt)
        # counts each segment_pd to its corresonding segment_gt
        count_seg_pd_2_corresponding_seg_gts = dict()

        size_segments_pd = dict()
        instance2labelName_filtered = dict()
        for segment_id in segment_ids:
            if int(segment_id) not in objs_obbinfo:
                continue

            segment_indices = np.where(segments_pd == segment_id)[0]
            segment_points = points_pd[segment_indices]

            size_segments_pd[segment_id] = len(segment_points)

            if filter_segment_size > 0:
                if size_segments_pd[segment_id] < filter_segment_size:
                    # print('skip segment',segment_id,'with size',size_segments_pd[segment_id],'that smaller than',filter_segment_size)
                    continue

            for i in range(len(segment_points)):
                point = segment_points[i]
                # [k, idx, distance] = tree.search_radius_vector_3d(point,0.001)
                k, idx, distance = tree.search_knn_vector_3d(point, 1)
                if distance[0] > max_distance:
                    continue
                # label_gt = labels_gt[idx][0]
                segment_gt = segments_gt[idx][0]

                if segment_gt not in instance2labelName:
                    continue
                if instance2labelName[segment_gt] == 'none':
                    continue
                instance2labelName_filtered[segment_gt] = instance2labelName[segment_gt]

                if segment_id not in count_seg_pd_2_corresponding_seg_gts:
                    count_seg_pd_2_corresponding_seg_gts[segment_id] = dict()
                if segment_gt not in count_seg_pd_2_corresponding_seg_gts[segment_id]:
                    count_seg_pd_2_corresponding_seg_gts[segment_id][segment_gt] = 0
                count_seg_pd_2_corresponding_seg_gts[segment_id][segment_gt] += 1

        instance2labelName = instance2labelName_filtered

        # if debug:
        #     print('There are {} segments have found their correponding GT segments.'.format(len(count_seg_pd_2_corresponding_seg_gts)))
        #     for k,i in count_seg_pd_2_corresponding_seg_gts.items():
        #         print('\t{}: {}'.format(k,len(i)))

        ''' Save as ply '''
        # if debug:
        #     if args.label_type == 'NYU40':
        #         colors = util_label.get_NYU40_color_palette()
        #         cloud_gt.visual.vertex_colors = [0,0,0,255]
        #         for seg, label_name in instance2labelName.items():
        #             segment_indices = np.where(segments_gt == seg)[0]
        #             if label_name == 'none':continue
        #             label = util_label.NYU40_Label_Names.index(label_name)+1
        #             for index in segment_indices:
        #                 cloud_gt.visual.vertex_colors[index][:3] = colors[label]
        #         cloud_gt.export('tmp_gtcloud.ply')
        #     else:
        #         for seg, label_name in instance2labelName.items():
        #             segment_indices = np.where(segments_gt == seg)[0]
        #             if label_name != 'none':
        #                 continue
        #             for index in segment_indices:
        #                 cloud_gt.visual.vertex_colors[index][:3] = [0,0,0]
        #         cloud_gt.export('tmp_gtcloud.ply')

        ''' Find best corresponding segment '''
        map_segment_pd_2_gt = dict()  # map segment_pd to segment_gt
        # how many segment_pd corresponding to this segment_gt
        gt_segments_2_pd_segments = defaultdict(list)
        for segment_id, cor_counter in count_seg_pd_2_corresponding_seg_gts.items():
            size_pd = size_segments_pd[segment_id]
            # if cfg.VERBOSE: print('segment_id', segment_id, size_pd)

            max_corr_ratio = -1
            max_corr_seg = -1
            list_corr_ratio = list()
            for segment_gt, count in cor_counter.items():
                size_gt = size_segments_gt[segment_gt]
                corr_ratio = count/size_pd
                list_corr_ratio.append(corr_ratio)
                if corr_ratio > max_corr_ratio:
                    max_corr_ratio = corr_ratio
                    max_corr_seg = segment_gt
                # if cfg.VERBOSE or debug: print('\t{0:s} {1:3d} {2:8d} {3:2.3f} {4:2.3f}'.\
                #                         format(instance2labelName[segment_gt],segment_gt,count, count/size_gt, corr_ratio))
            if len(list_corr_ratio) > 2:
                list_corr_ratio = sorted(list_corr_ratio, reverse=True)
                occ_ratio = list_corr_ratio[1]/list_corr_ratio[0]
            else:
                occ_ratio = 0

            if max_corr_ratio > filter_corr_thres and occ_ratio < filter_occ_ratio:
                '''
                This is to prevent a segment is almost equally occupied two or more gt segments. 
                '''
                # if cfg.VERBOSE or debug: print('add correspondence of segment {:s} {:4d} to label {:4d} with the ratio {:2.3f} {:1.3f}'.\
                #     format(instance2labelName[segment_gt],segment_id,max_corr_seg,max_corr_ratio,occ_ratio))
                map_segment_pd_2_gt[segment_id] = max_corr_seg
                gt_segments_2_pd_segments[max_corr_seg].append(segment_id)
            else:
                pass
                # if cfg.VERBOSE or debug: print('filter correspondence segment {:s} {:4d} to label {:4d} with the ratio {:2.3f} {:1.3f}'.\
                #     format(instance2labelName[segment_gt],segment_id,max_corr_seg,max_corr_ratio,occ_ratio))

        # if cfg.VERBOSE:
        #     print('final correspondence:')
        #     print('  pd  gt')
        #     for segment, label in sorted(map_segment_pd_2_gt.items()):
        #         print("{:4d} {:4d}".format(segment,label))
        #     print('final pd segments within the same gt segment')
        #     for gt_segment, pd_segments in sorted(gt_segments_2_pd_segments.items()):
        #         print('{:4d}:'.format(gt_segment),end='')
        #         for pd_segment in pd_segments:
        #             print('{} '.format(pd_segment),end='')
        #         print('')

        ''' Save as ply '''
        # if debug:
        #     if args.label_type == 'NYU40':
        #         colors = util_label.get_NYU40_color_palette()
        #         cloud_pd.visual.vertex_colors = [0, 0, 0, 255]
        #         for segment_pd, segment_gt in map_segment_pd_2_gt.items():
        #             segment_indices = np.where(segments_pd == segment_pd)[0]
        #             label = util_label.NYU40_Label_Names.index(
        #                 instance2labelName[segment_gt])+1
        #             color = colors[label]
        #             for index in segment_indices:
        #                 cloud_pd.visual.vertex_colors[index][:3] = color
        #         cloud_pd.export('tmp_corrcloud.ply')
        #     else:
        #         cloud_pd.visual.vertex_colors = [0, 0, 0, 255]
        #         for segment_pd, segment_gt in map_segment_pd_2_gt.items():
        #             segment_indices = np.where(segments_pd == segment_pd)[0]
        #             for index in segment_indices:
        #                 cloud_pd.visual.vertex_colors[index] = [
        #                     255, 255, 255, 255]
        #         cloud_pd.export('tmp_corrcloud.ply')

        '''' Save as relationship_*.json '''
        relationships = self.generate_relationship(
            scan_id,
            target_relationships,
            gt_relationships,
            map_segment_pd_2_gt,
            instance2labelName,
            gt_segments_2_pd_segments)
        
        for oid in relationships['objects'].keys():
            try:
                relationships['objects'][oid] = {
                    **objs_obbinfo[oid], **relationships['objects'][oid]}
            except:
                print('oid({}) in objs_obbinfo'.format(
                    oid), oid in objs_obbinfo)
                print('oid({}) in relationships[\'objects\']'.format(
                    oid), oid in relationships['objects'])
                print(objs_obbinfo)
                relationships['objects'][oid] = {
                    **objs_obbinfo[oid], **relationships['objects'][oid]}

        return relationships, segs_neighbors

    def generate_relationship(
            self,
            scan_id: str,
            target_relationships: list,
            gt_relationships: list,
            map_segment_pd_2_gt: dict,
            instance2labelName: dict,
            gt_segments_2_pd_segments: dict,
            target_segments: list = None) -> dict:
        '''' Save as relationship_*.json '''
        relationships = dict()  # relationships_new["scans"].append(s)
        relationships["scan"] = scan_id

        objects = dict()
        for seg, segment_gt in map_segment_pd_2_gt.items():
            if target_segments is not None:
                if seg not in target_segments:
                    continue
            name = instance2labelName[segment_gt]
            assert (name != '-' and name != 'none')
            objects[int(seg)] = dict()
            objects[int(seg)]['label'] = name
            objects[int(seg)]['instance_id'] = segment_gt
        relationships["objects"] = objects

        split_relationships = list()
        ''' Inherit relationships from ground truth segments '''
        if gt_relationships is not None:
            relationships_names = codeLib.utils.util.read_txt_to_list(
                os.path.join(define.PATH_FILE, lcfg.relation + ".txt"))

            for rel in gt_relationships:
                id_src = rel[0]
                id_tar = rel[1]
                num = rel[2]
                name = rel[3]
                
                if name not in target_relationships:
                    # logger_py.debug('filter ' + name +
                    #                 '. it is not in the target relationships')
                    continue
                
                idx_in_txt = relationships_names.index(name)
                idx_in_txt_new = target_relationships.index(name)
                assert (num == idx_in_txt)

                if id_src == id_tar:
                    continue  # an edge canno self connect
                    print('halloe', print(rel))

                if id_src in gt_segments_2_pd_segments and id_tar in gt_segments_2_pd_segments:
                    segments_src = gt_segments_2_pd_segments[id_src]
                    segments_tar = gt_segments_2_pd_segments[id_tar]
                    for segment_src in segments_src:
                        if segment_src not in objects:
                            continue
                        for segment_tar in segments_tar:
                            if segment_tar not in objects:
                                # if debug:
                                #     print(
                                #         'filter', name, 'segment_tar', instance2labelName[id_tar], ' is not in objects')
                                continue
                            if target_segments is not None:
                                ''' skip if they not in the target_segments'''
                                if segment_src not in target_segments:
                                    continue
                                if segment_tar not in target_segments:
                                    continue

                            ''' check if they are neighbors '''
                            split_relationships.append(
                                [int(segment_src), int(segment_tar), idx_in_txt_new, name])
                            # if debug:
                            #     print('segment', segment_src, '(', id_src, ')', segment_tar, '(', id_tar, ')',
                            #           'inherit', instance2labelName[id_src], name, instance2labelName[id_tar])
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
            if len(groups) <= 1:
                continue

            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    split_relationships.append([int(groups[i]), int(
                        groups[j]), idx_in_txt_new, define.NAME_SAME_PART])
                    split_relationships.append([int(groups[j]), int(
                        groups[i]), idx_in_txt_new, define.NAME_SAME_PART])

        '''check if a pair has multiple relationsihps'''
        relatinoships_gt_dict = defaultdict(list)
        for r in split_relationships:
            r_src = int(r[0])
            r_tgt = int(r[1])
            r_lid = int(r[2])
            r_cls = r[3]
            relatinoships_gt_dict[(r_src, r_tgt)].append(r_cls)
        invalid_keys = list()
        for key, value in relatinoships_gt_dict.items():
            if len(value) != 1:
                invalid_keys.append(key)
        for key in invalid_keys:
            print('key:', key, 'has more than one predicates:',
                  relatinoships_gt_dict[key])
            print(objects[key[0]]['label'], objects[key[1]]['label'])
        assert len(invalid_keys) == 0

        relationships["relationships"] = split_relationships
        return relationships


if __name__ == '__main__':
    args = Parser().parse_args()
    cfg = codeLib.Config(args.config)
    lcfg = cfg.data.scene_graph_generation
    outdir = args.pth_out
    debug = args.debug

    '''create log'''
    pathlib.Path(outdir).mkdir(exist_ok=True, parents=True)
    name_log = os.path.split(__file__)[-1].replace('.py', '.log')
    path_log = os.path.join(outdir, name_log)
    logging.basicConfig(filename=path_log, level=logging.INFO, force=True)
    logger_py = logging.getLogger(name_log)
    logger_py.info(f'create log file at {path_log}')
    if debug:
        logger_py.setLevel('DEBUG')
    else:
        logger_py.setLevel('INFO')

    if lcfg.neighbor_search_method == 'BBOX':
        search_method = SAMPLE_METHODS.BBOX
    elif args.neighbor_search_method == 'KNN':
        search_method = SAMPLE_METHODS.RADIUS

    codeLib.utils.util.set_random_seed(2020)

    '''create mapping'''
    label_names, * \
        _ = util_label.getLabelMapping(
            args.label_type, define.PATH_LABEL_MAPPING)

    '''get relationships'''
    target_relationships = sorted(codeLib.utils.util.read_txt_to_list(os.path.join(define.PATH_FILE, lcfg.relation + ".txt"))
                                  if not args.only_support_type else define.SUPPORT_TYPE_RELATIONSHIPS)
    target_relationships.append(define.NAME_SAME_PART)

    ''' get all classes '''
    classes_json = list()
    for key, value in label_names.items():
        if value == '-':
            continue
        classes_json.append(value)

    ''' read target scan'''
    target_scan = []
    if args.target_scan != '':
        target_scan = codeLib.utils.util.read_txt_to_list(args.target_scan)

    '''filter scans according to the target type'''
    with open(os.path.join(cfg.data.path_3rscan_data, lcfg.relation + ".json"), "r") as read_file:
        data = json.load(read_file)
        filtered_data = list()
        '''skip scan'''
        for s in data["scans"]:
            scan_id = s["scan"]
            if len(target_scan) > 0 and scan_id not in target_scan:
                continue
            filtered_data.append(s)

    '''create output'''
    pth_relationships_json = os.path.join(
        args.pth_out, define.NAME_RELATIONSHIPS)
    try:
        h5f = h5py.File(pth_relationships_json, 'a')
    except:
        os.remove(pth_relationships_json)
        h5f = h5py.File(pth_relationships_json, 'a')

    processor = GenerateSceneGraph(cfg, target_relationships, args.label_type)

    # Path(args.pth_out).mkdir(parents=True, exist_ok=True)
    # logging.basicConfig(filename=os.path.join(args.pth_out,'gen_data_'+args.type+'.log'), level=logging.DEBUG)
    # logger_py = logging.getLogger(__name__)

    # debug |= args.debug>0
    # args.verbose |= debug
    # if args.search_method == 'BBOX':
    #     search_method = SAMPLE_METHODS.BBOX
    # elif args.search_method == 'KNN':
    #     search_method = SAMPLE_METHODS.RADIUS

    # label_names, _, _ = util_label.getLabelMapping(args.label_type)
    # classes_json = list()
    # for key,value in label_names.items():
    #     if value == '-':continue
    #     classes_json.append(value)

    # ''' Read Scan and their type=['train', 'test', 'validation'] '''
    # scan2type = {}
    # with open(define.Scan3RJson_PATH, "r") as read_file:
    #     data = json.load(read_file)
    #     for scene in data:
    #         scan2type[scene["reference"]] = scene["type"]
    #         for scan in scene["scans"]:
    #             scan2type[scan["reference"]] = scene["type"]

    # '''read relationships'''
    # target_relationships=list()
    # if args.inherit:
    #     # target_relationships += ['supported by', 'attached to','standing on', 'lying on','hanging on','connected to',
    #                             # 'leaning against','part of','build in','standing in','lying in','hanging in']
    #     target_relationships += ['supported by', 'attached to','standing on','hanging on','connected to','part of','build in']
    # target_relationships.append(define.NAME_SAME_PART)

    # target_scan=[]
    # if args.target_scan != '':
    #     target_scan = util.read_txt_to_list(args.target_scan)

    # valid_scans=list()
    # relationships_new = dict()
    # relationships_new["scans"] = list()
    # relationships_new['neighbors'] = dict()
    # counter= 0
    ''' generate data '''
    valid_scans = list()
    for s in tqdm(filtered_data):
        scan_id = s["scan"]
        valid_scans.append(scan_id)
        # Check exist
        if scan_id in h5f:
            if not args.overwrite:
                logger_py.debug(f'{scan_id} exist. skip')
                continue
            else:
                del h5f[scan_id]

        gt_relationships = s["relationships"]
        logger_py.info('processing scene {}'.format(scan_id))
        relationships, segs_neighbors = processor(
            scan_id, gt_relationships
        )
        if len(relationships) == 0:
            logger_py.info(
                'skip {} due to not enough objs and relationships'.format(scan_id))
            continue
        else:
            logger_py.debug('no skip', scan_id)

        '''save to h5'''
        # save everything to dict. convert it to str. decode it back to dict later
        objects = relationships['objects']
        d_scan = dict()
        d_nodes = d_scan['nodes'] = dict()

        # Nodes
        for idx, data in enumerate(objects.items()):
            oid, obj_info = data
            ascii_nn = [str(n).encode("ascii", "ignore")
                        for n in segs_neighbors[oid]]
            d_nodes[oid] = dict()
            d_nodes[oid] = obj_info
            d_nodes[oid]['neighbors'] = ascii_nn

        # Relationships
        str_relationships = list()
        for rel in relationships['relationships']:
            str_relationships.append([str(s) for s in rel])
        d_scan['relationships'] = str_relationships

        s_scan = str(d_scan)
        h5_scan = h5f.create_dataset(scan_id, data=np.array(
            [s_scan], dtype='S'), compression='gzip')
        # test decode
        tmp = h5_scan[0].decode()
        assert isinstance(ast.literal_eval(tmp), dict)
    h5f.close()

    '''Save'''
    Path(args.pth_out).mkdir(parents=True, exist_ok=True)
    pth_args = os.path.join(args.pth_out, 'args.json')
    with open(pth_args, 'w') as f:
        tmp = vars(args)
        json.dump(tmp, f, indent=2)
    pth_classes = os.path.join(args.pth_out, 'classes.txt')
    with open(pth_classes, 'w') as f:
        for name in classes_json:
            if name == '-':
                continue
            f.write('{}\n'.format(name))
    pth_relation = os.path.join(args.pth_out, 'relationships.txt')
    with open(pth_relation, 'w') as f:
        for name in target_relationships:
            f.write('{}\n'.format(name))
    pth_split = os.path.join(args.pth_out, 'scan_ids.txt')
    with open(pth_split, 'w') as f:
        for name in valid_scans:
            f.write('{}\n'.format(name))

    # '''Save'''
    # pth_args = os.path.join(args.pth_out,'args.json')
    # with open(pth_args, 'w') as f:
    #         tmp = vars(args)
    #         json.dump(tmp, f, indent=2)

    # pth_classes = os.path.join(args.pth_out, 'classes.txt')
    # with open(pth_classes,'w') as f:
    #     for name in classes_json:
    #         if name == '-': continue
    #         f.write('{}\n'.format(name))
    # pth_relation = os.path.join(args.pth_out, 'relationships.txt')
    # with open(pth_relation,'w') as f:
    #     for name in target_relationships:
    #         f.write('{}\n'.format(name))
    # pth_split = os.path.join(args.pth_out, args.type+'_scans.txt')
    # with open(pth_split,'w') as f:
    #     for name in valid_scans:
    #         f.write('{}\n'.format(name))
    # # '''Save to json'''
    # # pth_relationships_json = os.path.join(args.pth_out, "relationships_" + args.type + ".json")
    # # with open(pth_relationships_json, 'w') as f:
    # #     json.dump(relationships_new, f)

    # '''Save to h5'''
    # pth_relationships_json = os.path.join(args.pth_out, "relationships_" + args.type + ".h5")
    # h5f = h5py.File(pth_relationships_json, 'w')
    # # reorganize scans from list to dict
    # scans = dict()
    # for s in relationships_new['scans']:
    #     scans[s['scan']] = s
    # all_neighbors = relationships_new['neighbors']
    # for scan_id in scans.keys():
    #     scan_data = scans[scan_id]
    #     neighbors = all_neighbors[scan_id]
    #     objects = scan_data['objects']

    #     d_scan = dict()
    #     d_nodes = d_scan['nodes'] = dict()

    #     ## Nodes
    #     for idx, data in enumerate(objects.items()):
    #         oid, obj_info = data
    #         ascii_nn = [str(n).encode("ascii", "ignore") for n in neighbors[oid]]
    #         d_nodes[oid] = dict()
    #         d_nodes[oid] = obj_info
    #         d_nodes[oid]['neighbors'] = ascii_nn

    #     ## Relationships
    #     str_relationships = list()
    #     for rel in scan_data['relationships']:
    #         str_relationships.append([str(s) for s in rel])
    #     d_scan['relationships']= str_relationships

    #     s_scan = str(d_scan)
    #     h5_scan = h5f.create_dataset(scan_id,data=np.array([s_scan],dtype='S'),compression='gzip')
    #     # test decode
    #     tmp = h5_scan[0].decode()
    #     assert isinstance(ast.literal_eval(tmp),dict)

    #     # ast.literal_eval(h5_scan)
    # h5f.close()
