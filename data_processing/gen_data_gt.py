import pathlib
import codeLib
from ssg.utils import util_3rscan, util_label, util_ply
# from utils import util_ply
import trimesh
import open3d as o3d
import numpy as np
# from utils import define, util
# from utils import util_ply, util_label, util, define
from ssg import define
# from utils.util_search import SAMPLE_METHODS,find_neighbors
from ssg.utils.util_search import SAMPLE_METHODS, find_neighbors
from tqdm import tqdm
from pathlib import Path
import os
import json
import argparse
import h5py
import ast
import copy
import logging


def Parser(add_help=True):
    helpmsg = 'Generate custom scene graph dataset from the 3RScan dataset.'
    parser = argparse.ArgumentParser(
        description=helpmsg, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-c', '--config', default='./configs/config_default.yaml', required=False)
    parser.add_argument('-o', '--pth_out', type=str, default='../data/tmp',
                        help='pth to output directory', required=True)
    parser.add_argument('--target_scan', type=str, default='', help='')
    parser.add_argument('-l', '--label_type', type=str, default='3RScan160',
                        choices=['NYU40', 'Eigen13', 'Rio27', 'Rio7', '3RScan', '3RScan160', 'ScanNet20'], help='label', required=False)
    parser.add_argument('--only_support_type', action='store_true',
                        help='use only support type of relationship')

    # options
    parser.add_argument('--debug', action='store_true',
                        help='debug', required=False)
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite or not.')

    # neighbor search parameters
    # parser.add_argument('--search_method', type=str, choices=['BBOX','KNN'],default='BBOX',help='How to split the scene.')
    # parser.add_argument('--radius_receptive', type=float,default=0.5,help='The receptive field of each seed.')

    # constant
    parser.add_argument('--segment_type', type=str, default='GT')
    return parser


class GenerateSceneGraph_GT(object):
    def __init__(self, cfg: dir, target_relationships: list, label_type: str):
        self.cfg = cfg
        self.target_relationships = target_relationships
        self.label_type = label_type

    def __call__(self, scan_id, gt_relationships):
        pth_3RScan_data = self.cfg.data.path_3rscan_data
        lcfg = self.cfg.data.scene_graph_generation
        target_relationships = self.target_relationships
        pth_gt = os.path.join(pth_3RScan_data, scan_id,
                              self.cfg.data.label_file_gt)
        # segseg_file_name = define.SEMSEG_FILE_NAME

        # load gt
        cloud_gt = trimesh.load(pth_gt, process=False)
        points_gt = np.array(cloud_gt.vertices.tolist())
        segments_gt = util_ply.get_label(
            cloud_gt, '3RScan', 'Segment').flatten()

        # find neighbors
        segs_neighbors = find_neighbors(
            points_gt, segments_gt, search_method, receptive_field=lcfg.radius_receptive)

        # get segment ids
        segment_ids = np.unique(segments_gt)
        segment_ids = segment_ids[segment_ids != 0]

        # get label mapping
        _, label_name_mapping, _ = util_label.getLabelMapping(self.label_type)
        pth_semseg = os.path.join(
            pth_3RScan_data, scan_id, define.SEMSEG_FILE_NAME)
        instance2labelName = util_3rscan.load_semseg(
            pth_semseg, label_name_mapping)

        '''extract object bounding box info'''
        objs_obbinfo = dict()
        with open(pth_semseg) as f:
            data = json.load(f)
        for group in data['segGroups']:
            obb = group['obb']
            obj_obbinfo = objs_obbinfo[group["id"]] = dict()
            obj_obbinfo['center'] = copy.deepcopy(obb['centroid'])
            obj_obbinfo['dimension'] = copy.deepcopy(obb['axesLengths'])
            obj_obbinfo['normAxes'] = copy.deepcopy(
                np.array(obb['normalizedAxes']).reshape(3, 3).transpose().tolist())
        del data

        ''' Find and count all corresponding segments'''
        map_segment_pd_2_gt = dict()  # map segment_pd to segment_gt
        for segment_id in segment_ids:
            map_segment_pd_2_gt[segment_id] = segment_id

        ''' Save as ply '''
        # if debug:
        #     for seg, label_name in instance2labelName.items():
        #         segment_indices = np.where(segments_gt == seg)[0]
        #         if label_name != 'none':
        #             continue
        #         for index in segment_indices:
        #             cloud_gt.visual.vertex_colors[index][:3] = [0,0,0]
        #     cloud_gt.export('tmp_gtcloud.ply')

        '''' Save as relationship_*.json '''
        relationships = self.generate_relationship(
            scan_id,
            target_relationships,
            gt_relationships,
            map_segment_pd_2_gt,
            instance2labelName)

        for oid in relationships['objects'].keys():
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
            if name == '-' or name == 'none':
                continue
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
                assert (num == idx_in_txt)
                idx_in_txt_new = target_relationships.index(name)
                split_relationships.append(
                    [int(id_src), int(id_tar), idx_in_txt_new, name])

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

    '''Create processor'''
    if args.segment_type == "GT":
        processor = GenerateSceneGraph_GT(
            cfg, target_relationships, args.label_type)
    else:
        raise NotImplementedError()

    '''generate data'''
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
        relationships, segs_neighbors = processor(scan_id, gt_relationships)
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

        # relationships_new["scans"] += relationships
        # relationships_new['neighbors'][scan_id] = segs_neighbors
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
