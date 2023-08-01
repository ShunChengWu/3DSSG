#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 19:35:06 2022

@author: sc
"""
import graphviz
import operator
import logging
import torch
from collections import defaultdict
from ssg.utils import util_label, util_merge_same_part
from ssg.utils.util_data import raw_to_data, cvt_all_to_dict_from_h5
# from codeLib.common import rgb_2_hex
from codeLib.common import rand_24_bit, color_hex_rgb, rgb_2_hex
from codeLib.geoemetry.common import create_box
from ssg import define
import math

logger_py = logging.getLogger(__name__)


def process_node_pd(nodes):
    ns = dict()
    for k, v in nodes.items():
        if isinstance(v, dict):  # full output
            vv = max(v.items(), key=operator.itemgetter(1))[0]
            ns[k] = vv
        else:
            ns[k] = v
    return ns


def process_edge_pd(edges):
    es = dict()
    for k, v in edges.items():
        if isinstance(v, dict):
            vv = max(v.items(), key=operator.itemgetter(1))[0]
            es[k] = vv
        else:
            es[k] = v
    return es


def process_pd(nodes, edges):
    '''
    The input is expected to be a dictionary with key:node_id, value: the full probability of all class
    '''
    output = dict()
    output = {'nodes': process_node_pd(nodes),
              'edges': process_edge_pd(edges)}
    return output


def process_gt(nodes, edges):
    nodes_gts = dict()
    for idx, pd in nodes.items():
        nodes_gts[str(idx)] = pd
        pass
    edges_gts = defaultdict(list)
    for edge in edges:
        name = edge
        predicates = edges[edge]
        if isinstance(predicates, list):
            for predicate in predicates:
                edges_gts[name].append(predicate)
        else:
            edges_gts[name].append(predicates)

    output = dict()
    output = {'nodes': nodes_gts,
              'edges': edges_gts}
    return output


def to_name_dict(x, node_names: list, edge_names: list):
    output = dict()
    nodes = output['nodes'] = dict()
    edges = output['edges'] = dict()
    for k, v in x['nodes'].items():
        k = str(k)
        nodes[k] = dict()
        if v.ndim == 0:
            nodes[k] = node_names[v]
        else:
            assert v.shape[0] == len(node_names)
            for idx, prob in enumerate(v):
                nodes[k][node_names[idx]] = prob

    for k, v in x['edges'].items():
        if isinstance(k, tuple):
            assert len(k) == 2
            k = str(k[0])+'_'+str(k[1])
        else:
            k = str(k)
        edges[k] = dict()
        if v.ndim == 0:
            edges[k] = edge_names[v]
        else:
            assert v.shape[0] == len(edge_names)
            for idx, prob in enumerate(v):
                edges[k][edge_names[idx]] = prob
    return output


class DrawSceneGraph(object):
    def __init__(self, scan_id: str, node_names: list, edge_names: list, debug: bool):
        self.node_names = node_names
        self.edge_names = edge_names
        self.color_correct = '#0CF369'  # green
        self.color_wrong = '#FF0000'  # red
        self.color_missing_pd = '#A9A2A2'  # gray
        self.color_missing_gt = '#0077FF'  # blue
        self.debug = debug
        self.scan_id = scan_id

        '''load segment color'''
        self.node_colors = dict()
        # os.path.join(define.DATA_PATH,scan_id,)
        # pth_ply = os.path.join(pth_pd,'node_segment.ply')
        # # pth_ply = os.path.join(pth_pd,'node_panoptic.ply')
        # cloud_pd= trimesh.load(pth_ply , process=False)
        # colors  = cloud_pd.colors
        # segments_pd = cloud_pd.metadata['ply_raw']['vertex']['data']['label'].flatten()
        # seg_colors = dict()
        # segment_ids = np.unique(segments_pd)
        # min_size = 512*8
        # for idx in segment_ids:
        #     indices = np.where(segments_pd == idx)[0]
        #     if min_size > 0:
        #         if len(indices) > min_size:
        #             seg_colors[idx] = colors[indices[0]].tolist()

    def to_name_dict(self, x):
        return to_name_dict(x, self.node_names, self.edge_names)

    def draw(self, pds: dict, gts: dict = None, node_colors=None):
        '''
        pds:dict {
            'nodes':,
            'edges':,
            }
        gts:dict {
            'nodes':,
            'edges':,
            }
        '''
        pds = self.to_name_dict(pds)
        if gts is not None:
            gts = self.to_name_dict(gts)
        processed_pds = process_pd(**pds)
        # gts = process_pd(**gts)

        '''find instance group (connected by "same part"'''
        inst_groups = util_merge_same_part.collect(
            processed_pds['nodes'], processed_pds['edges'], define.NAME_SAME_PART)

        seg_to_inst = dict()
        for k, v in inst_groups.items():
            for vv in v:
                seg_to_inst[vv] = k

        ''' collect predictions on inst level '''
        inst_edge_pred = dict()
        inst_edge_gt = dict()
        counter = dict()
        for k, v in pds['edges'].items():
            if isinstance(k, str):
                sp = k.split('_')
            else:
                sp = k
            source, target = sp
            if source not in seg_to_inst or target not in seg_to_inst:
                continue

            ''' check if prediction is correct'''
            s, t = seg_to_inst[source], seg_to_inst[target]

            ''' set name to cluster if the node on that cluster has the same label'''
            s = 'cluster'+str(s) if source in pds['nodes'] and gts is not None and source in gts['nodes'] and pds['nodes'][source] == gts['nodes'][source] \
                else source
            t = 'cluster'+str(t)
            name = str(s)+'_'+str(t)

            if name not in inst_edge_pred:
                inst_edge_pred[name] = dict()
                counter[name] = dict()
            for v_k, v_v in v.items():
                if v_k not in inst_edge_pred[name]:
                    inst_edge_pred[name][v_k] = 0
                    counter[name][v_k] = 0
                inst_edge_pred[name][v_k] += v_v  # math.exp(v_v)
                counter[name][v_k] += 1

        if gts is not None:
            for k, v in gts['edges'].items():
                if isinstance(k, str):
                    sp = k.split('_')
                else:
                    sp = k
                source, target = sp
                if source not in seg_to_inst or target not in seg_to_inst:
                    continue

                ''' check if prediction is correct'''
                s, t = seg_to_inst[source], seg_to_inst[target]

                ''' set name to cluster if the node on that cluster has the same label'''
                s = 'cluster'+str(s) if source in pds['nodes'] and source in gts['nodes'] and pds['nodes'][source] == gts['nodes'][source] \
                    else source
                # t = 'cluster'+str(t) if target in pds['nodes'] and target in gts['nodes'] and pds['nodes'][target] == gts['nodes'][target] \
                #     else target
                t = 'cluster'+str(t)
                name = str(s)+'_'+str(t)
                if name in inst_edge_gt:
                    if inst_edge_gt[name] != gts['edges'][k]:
                        inst_edge_gt[name] = inst_edge_gt[name].union(
                            set(gts['edges'][k]))  # = inst_edge_gt[name]+','+
                        # raise RuntimeError('duplicate',inst_edge_gt[name], gts['edges'][k])
                else:
                    inst_edge_gt[name] = set(gts['edges'][k])

        for k, v in inst_edge_pred.items():
            for v_k, v_v in v.items():
                inst_edge_pred[k][v_k] /= counter[k][v_k]

        # processed_edge_pds = process_edge_pd(inst_edge_pred)
        if gts is not None:
            edge_gts = inst_edge_gt
        processed_pds['edges'] = process_edge_pd(
            inst_edge_pred)  # replace to grouped edges

        '''try to load segment color from point cloud'''
        # cloud_pd= trimesh.load(pth_ply , process=False)
        # colors  = cloud_pd.colors
        # segments_pd = cloud_pd.metadata['ply_raw']['vertex']['data']['label'].flatten()
        # seg_colors = dict()
        # segment_ids = np.unique(segments_pd)
        # min_size = 512*8
        # for idx in segment_ids:
        #     indices = np.where(segments_pd == idx)[0]
        #     if min_size > 0:
        #         if len(indices) > min_size:
        #             seg_colors[idx] = colors[indices[0]].tolist()

        seg_colors = None
        # g = self.draw_evaluation(self.scan_id, pds=processed_pds,gts=gts  ,none_name =define.NAME_NONE, inst_groups=inst_groups,
        #                 label_color=util_label.get_NYU40_color_palette(),seg_colors = seg_colors)
        g = self.draw_pd_only(self.scan_id, pds=processed_pds, inst_groups=inst_groups,
                              label_color=util_label.get_NYU40_color_palette(),
                              node_colors=node_colors,
                              grouping=True)
        return g

    def draw_pd_only(self, scan_id, pds, inst_groups: dict = None, label_color=None, node_colors: dict = None, grouping: bool = True):
        node_pds, edge_pds = pds['nodes'], pds['edges']
        none_name = define.NAME_NONE
        if not grouping:
            g = graphviz.Digraph(comment=scan_id,
                                 format='png',
                                 # node_attr={'shape': 'circle',
                                 #            'style': 'filled',
                                 #            'fontname':'helvetica',
                                 #            'color': 'lightblue2',
                                 #            'width':'1',
                                 #            'fontsize':'24',
                                 #            },
                                 # edge_attr={
                                 #            'fontsize':'18',
                                 #            },
                                 #  graph_attr={'rankdir':'LR',
                                 #              # 'center':'true',
                                 #                'splines':'compound',
                                 #              'margin':'0.01',
                                 #              'fontsize':'24',
                                 #              'ranksep':'0.1',
                                 #              'nodesep':'0.1',
                                 #               'width':'1',
                                 #              # 'height':'20',
                                 #              },
                                 #  # graph_attr={'rankdir': 'TB'},
                                 #   engine='fdp'
                                 )
        else:
            g = graphviz.Digraph(comment=scan_id,
                                 format='png',
                                 node_attr={'shape': 'circle',
                                            'style': 'filled',
                                            'fontname': 'helvetica',
                                            'color': 'lightblue2',
                                            'width': '1',
                                            'fontsize': '24',
                                            },
                                 edge_attr={
                                     'fontsize': '18',
                                 },
                                 graph_attr={'rankdir': 'LR',
                                             # 'center':'true',
                                             'splines': 'curved',  # 'compound',
                                             # 'margin':'0.01',
                                             'fontsize': '24',
                                             'ranksep': '0.1',
                                             'nodesep': '0.1',
                                             'width': '1',
                                             # 'height':'20',
                                             },
                                 # graph_attr={'rankdir': 'TB'},
                                 engine='fdp',  # 'dot'#fdp
                                 )

        seg_to_inst = dict()
        drawed_nodes = set()
        # drawed_insts = set()
        # wrong_segs = list()
        DRAW_MODE = 'SEMANTIC'
        SHOW_LABEL_NAME = False
        if not grouping:
            sorted_keys = [str(k) for k in sorted([int(k) for k in node_pds])]
            for k in sorted_keys:
                name_pd = node_pds[k]
                if DRAW_MODE == 'SEMANTIC':
                    color = '#%02x%02x%02x' % label_color[util_label.nyu40_name_to_id(
                        name_pd)+1]
                elif DRAW_MODE == 'SEGMENT':
                    if node_colors is None:
                        if k not in self.node_colors:
                            self.node_colors[k] = color_hex_rgb(rand_24_bit())
                        color = self.node_colors[k]
                    else:
                        try:
                            color = rgb_2_hex(node_colors[int(k)])
                        except:
                            color = rgb_2_hex(node_colors[int(k)])

                # g.node(k,str(k) + '_' + name_pd,color=color)
                if SHOW_LABEL_NAME:
                    show_name = str(k) + '_' + name_pd
                else:
                    show_name = str(k)
                g.node(k, show_name, color=color)
                drawed_nodes.add(k)

            i = 0
            for edge in edge_pds.keys():
                '''
                For each edge there may have multiple labels. 
                If non ground truth labels are given, set to missing_gt
                If gt labels are given, find the union and difference.
                '''
                f = edge.split('_')[0]
                t = edge.split('_')[1]

                if 'cluster' in f:
                    f = f[7:]
                if f not in drawed_nodes:
                    continue
                if 'cluster' in t:
                    t = t[7:]
                if t not in drawed_nodes:
                    continue

                names_pd = list()
                names_pd = edge_pds[edge] if isinstance(
                    edge_pds[edge], list) else [edge_pds[edge]]
                names_pd = set(names_pd).difference([none_name, 'same part'])

                for name_pd in names_pd:
                    # color_missing_pd
                    g.edge(f, t, label=name_pd, color=self.color_missing_gt)
                i += 1
        else:
            for k, v in inst_groups.items():
                inst_name = 'cluster'+k
                with g.subgraph(name=inst_name) as c:
                    # name =  ''#node_pds[k]
                    # color = None
                    # color = '#%02x%02x%02x' % label_color[util_label.nyu40_name_to_id(name)+1]

                    # if not self.debug: c.attr(label=name)
                    # else: c.attr(label=name+'_'+k)
                    # c.attr(style='filled', color=color)

                    pre = []
                    for idx in v:
                        name_pd = none_name
                        if idx in node_pds:
                            name_pd = node_pds[idx]
                        seg_to_inst[idx] = inst_name

                        '''color'''
                        color = None
                        if DRAW_MODE == 'SEMANTIC':
                            color = '#%02x%02x%02x' % label_color[util_label.nyu40_name_to_id(
                                name_pd)+1]
                        elif DRAW_MODE == 'SEGMENT':
                            if node_colors is None:
                                if k not in self.node_colors:
                                    self.node_colors[k] = color_hex_rgb(
                                        rand_24_bit())
                                color = self.node_colors[k]
                            else:
                                try:
                                    color = rgb_2_hex(node_colors[int(idx)])
                                except:
                                    color = rgb_2_hex(node_colors[int(idx)])

                        '''display'''
                        if SHOW_LABEL_NAME:
                            show_name = str(idx) + '_' + name_pd
                        else:
                            show_name = str(idx)
                        show_name = ''
                        c.node(idx, show_name, color='white', fillcolor=color)
                        if len(pre) > 0:
                            g.edge(pre[-1], idx)

            i = 0
            for edge in edge_pds.keys():
                '''
                For each edge there may have multiple labels. 
                If non ground truth labels are given, set to missing_gt
                If gt labels are given, find the union and difference.
                '''
                f = edge.split('_')[0]
                t = edge.split('_')[1]

                # if f not in drawed_nodes:
                #     continue
                # if t not in drawed_nodes:
                #     continue

                names_pd = list()
                if edge in edge_pds:
                    names_pd = edge_pds[edge] if isinstance(
                        edge_pds[edge], list) else [edge_pds[edge]]
                names_pd = set(names_pd).difference([none_name, 'same part'])

                for name_pd in names_pd:
                    # color_missing_pd
                    g.edge(f, t, label=name_pd, color=self.color_missing_gt)
                i += 1
        return g

    def draw_nogroup(self, pds: dict, node_colors):
        '''
        make every node as a group. outer color is the estimated label, inner is the node color
        pds:dict {
            'nodes':,
            'edges':,
            }
        gts:dict {
            'nodes':,
            'edges':,
            }
        '''
        pds = self.to_name_dict(pds)
        processed_pds = process_pd(**pds)
        node_pds, edge_pds = processed_pds['nodes'], processed_pds['edges']
        label_color = util_label.get_NYU40_color_palette()

        g = graphviz.Digraph(comment=self.scan_id,
                             format='png',
                             node_attr={'shape': 'circle',
                                        'style': 'filled',
                                        'fontname': 'helvetica',
                                        'color': 'lightblue2',
                                        'width': '1',
                                        'fontsize': '24',
                                        },
                             edge_attr={
                                 'fontsize': '18',
                             },
                             graph_attr={'rankdir': 'LR',
                                         # 'center':'true',
                                         'splines': 'curved',  # 'compound',
                                         # 'margin':'0.01',
                                         'fontsize': '24',
                                         'ranksep': '0.1',
                                         'nodesep': '0.1',
                                         'width': '1',
                                         # 'height':'20',
                                         },
                             # graph_attr={'rankdir': 'TB'},
                             engine='fdp',  # 'dot'#fdp
                             )
        DRAW_MODE = 'SEMANTIC'
        SHOW_LABEL_NAME = False

        sorted_keys = [str(k) for k in sorted([int(k) for k in node_pds])]
        drawed_nodes = set()
        for k in sorted_keys:
            inst_name = 'cluster'+k
            with g.subgraph(name=inst_name) as c:
                name_pd = node_pds[k]
                color = node_colors[int(k)]
                color = rgb_2_hex(color)
                c.attr(style='filled', color=color)

                '''display'''
                if SHOW_LABEL_NAME:
                    show_name = str(k) + '_' + name_pd
                else:
                    show_name = str(k)
                show_name = ''

                color = '#%02x%02x%02x' % label_color[util_label.nyu40_name_to_id(
                    name_pd)+1]
                c.node(k, show_name, color='white', fillcolor=color)
                drawed_nodes.add(str(k))
        i = 0
        for edge in edge_pds.keys():
            '''
            For each edge there may have multiple labels. 
            If non ground truth labels are given, set to missing_gt
            If gt labels are given, find the union and difference.
            '''
            f = edge.split('_')[0]
            t = edge.split('_')[1]

            if f not in drawed_nodes:
                continue
            if t not in drawed_nodes:
                continue
            f = 'cluster'+f
            t = 'cluster'+t

            names_pd = list()
            if edge in edge_pds:
                names_pd = edge_pds[edge] if isinstance(
                    edge_pds[edge], list) else [edge_pds[edge]]
            names_pd = set(names_pd).difference(
                [define.NAME_NONE, 'same part'])

            for name_pd in names_pd:
                # color_missing_pd
                g.edge(f, t, label=name_pd, color=self.color_missing_gt)
            i += 1
        return g

    def draw_evaluation(self, scan_id, pds, gts, none_name=define.NAME_NONE,
                        pd_only=False, gt_only=False, inst_groups: dict = None, label_color=None, seg_colors: dict = None):
        node_pds, edge_pds = pds['nodes'], pds['edges']
        if gts is not None:
            node_gts, edge_gts = gts['nodes'], gts['edges']
        g = graphviz.Digraph(comment=scan_id, format='png',
                             node_attr={'shape': 'circle',
                                        'style': 'filled',
                                        'fontname': 'helvetica',
                                        'color': 'lightblue2',
                                        'width': '1',
                                        'fontsize': '24',
                                        },
                             edge_attr={
                                 'fontsize': '18',
                             },
                             graph_attr={'rankdir': 'LR',
                                         # 'center':'true',
                                         'splines': 'compound',
                                         'margin': '0.01',
                                         'fontsize': '24',
                                         'ranksep': '0.1',
                                         'nodesep': '0.1',
                                         'width': '1',
                                         # 'height':'20',
                                         },
                             # graph_attr={'rankdir': 'TB'},
                             engine='fdp'
                             )

        nodes = set(node_pds.keys()).union(node_gts.keys())
        selected_insts = []
        exclude_selected_insts = []
        selected_nodes = []

        drawed_nodes = set()
        drawed_insts = set()
        if inst_groups is None:
            for idx in nodes:
                name_gt = none_name
                name_pd = none_name
                if idx not in node_pds:
                    color = self.color_missing_pd
                else:
                    name_pd = node_pds[idx]
                if idx not in node_gts:
                    color = self.color_missing_gt
                else:
                    name_gt = node_gts[idx]
                if idx in node_pds and idx in node_gts:
                    color = self.color_correct if node_gts[idx] == node_pds[idx] else self.color_wrong
                g.node(idx, str(idx) + '_' + name_pd +
                       '('+name_gt+')', color=color)
                drawed_nodes.add(idx)
        else:
            seg_to_inst = dict()
            wrong_segs = list()
            for k, v in inst_groups.items():
                if len(selected_insts) > 0:
                    if k not in selected_insts:
                        continue
                if len(exclude_selected_insts) > 0:
                    if k in exclude_selected_insts:
                        continue
                inst_name = 'cluster'+k
                with g.subgraph(name=inst_name) as c:
                    name = node_gts[k]
                    color = None
                    color = '#%02x%02x%02x' % label_color[util_label.nyu40_name_to_id(
                        name)+1]

                    if not self.debug:
                        c.attr(label=name)
                    else:
                        c.attr(label=name+'_'+k)
                    c.attr(style='filled', color=color)

                    pre = []
                    for idx in v:
                        if seg_colors is not None and int(idx) not in seg_colors:
                            continue
                        if len(selected_nodes) > 0:
                            if idx not in selected_nodes:
                                continue

                        name_gt = none_name
                        name_pd = none_name
                        if idx in node_pds:
                            name_pd = node_pds[idx]
                        if idx in node_gts:
                            name_gt = node_gts[idx]

                        color = None
                        if label_color is not None and seg_colors is not None:
                            # color = '#%02x%02x%02x' % label_color[util_label.nyu40_name_to_id(name)+1]
                            color = '#%02x%02x%02x' % tuple(
                                seg_colors[int(idx)][:3])
                        else:
                            if k not in self.node_colors:
                                self.node_colors[k] = color_hex_rgb(
                                    rand_24_bit())
                            color = self.node_colors[k]

                        seg_to_inst[idx] = inst_name
                        node_label = '' if name_pd == name_gt else name_gt
                        if self.debug:
                            node_label = str(idx)+'_'+node_label
                        # node_label = name_pd+'_'+name_gt
                        c.node(idx, node_label, color='white', fillcolor=color)
                        drawed_nodes.add(idx)
                        drawed_insts.add(inst_name)
                        if node_label == '':
                            wrong_segs.append(idx)

                        if len(pre) > 0:
                            g.edge(pre[-1], idx)
        # return g

        edges = set(edge_pds.keys()).union(edge_gts.keys())
        i = 0
        for edge in edges:
            '''
            For each edge there may have multiple labels. 
            If non ground truth labels are given, set to missing_gt
            If gt labels are given, find the union and difference.
            '''
            f = edge.split('_')[0]
            t = edge.split('_')[1]

            if f == 'cluster901' and t == 'cluster312':
                print('helo')
                # continue
            if f == 'cluster527':
                print('2')

            names_gt = list()
            names_pd = list()
            if edge in edge_pds:
                names_pd = edge_pds[edge] if isinstance(
                    edge_pds[edge], list) else [edge_pds[edge]]

            if edge in edge_gts:
                names_gt = edge_gts[edge] if isinstance(
                    edge_gts[edge], list) else [edge_gts[edge]]

            names_pd = set(names_pd).difference([none_name, 'same part'])

            names_gt = set(names_gt).difference([none_name, 'same part'])

            if len(names_gt) > 0:
                intersection = set(names_gt).intersection(
                    names_pd)  # match prediction
                diff_gt = set(names_gt).difference(
                    intersection)  # unmatched gt
                diff_pd = set(names_pd).difference(
                    intersection)  # unmatched pd

                ''' same part is in a box alearly only use outline color to indicate right or wrong'''

                for name in intersection:
                    g.edge(f, t, label=name, color=self.color_correct)

                pds = ''
                if not gt_only:
                    for name_pd in diff_pd:  # in pd but not in gt
                        if pds == '':
                            pds = name_pd
                        else:
                            pds = pds + ','+name_pd
                        # g.edge(f,t,label=name_pd,color=color_wrong)
                gts = ''
                if not pd_only:
                    for name_gt in diff_gt:  # in gt but not in pd
                        gts = name_gt if gts == '' else gts + ','+name_gt
                        # g.edge(f,t,label=none_name+'\n('+name_gt+')',color=color_wrong) # color_missing_pd
                if pds != '' or gts != '':
                    if pds != '':
                        pds+'\n'
                    g.edge(f, t, label=pds+'('+gts+')',
                           color=self.color_wrong)  # color_missing_pd
            elif len(names_gt) == 0:  # missing gt
                if not gt_only:
                    for name_pd in names_pd:
                        # color_missing_pd
                        g.edge(f, t, label=name_pd,
                               color=self.color_missing_gt)
            i += 1
        return g
