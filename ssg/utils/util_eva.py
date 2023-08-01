import copy
import os
import math
import json
import pathlib
import torch
import numpy as np
# import ssg
from codeLib.utils.plot_confusion_matrix import plot_confusion_matrix
from collections import defaultdict
from torch import Tensor
from torch_geometric.data import HeteroData
import itertools
# from utils.plot_confusion_matrix import plot_confusion_matrix


def merged_prediction_to_node(data: HeteroData):
    node = data['node']
    # oid2idx
    oids = node.oid.tolist()
    oid2idx = {oid: idx for idx, oid in enumerate(oids)}

    # collect prediction from roi
    if len(data['roi']) > 0 and 'y' in data['roi']:
        roi = data['roi']
        roi_oid = roi.oid.tolist()
        assert 'x' in roi, "should assign prediction to roi.x"

        roi_pred_dict = defaultdict(list)
        for idx_roi, oid in enumerate(roi_oid):
            roi_pred_dict[oid].append(roi.pd[idx_roi])

        # merge prediction
        for key, pds in roi_pred_dict.items():
            pds = torch.stack(pds).mean(0)
            roi_pred_dict[key] = pds

    # build new node list
    node_pds_new = []
    oids_new = []
    oid2idxNew = {}
    node_y_new = []
    for idx_node_new, oid in enumerate(roi_pred_dict):
        node_pds_new.append(roi_pred_dict[oid])
        oids_new.append(oid)
        oid2idxNew[oid] = idx_node_new
        node_y_new.append(oid2idx[oid])
    node_y_new = node.y[node_y_new]

    edge_index = data['node', 'to', 'node'].edge_index.tolist()
    edge_index_new = []
    edge_index_y_new = []
    for idx in range(len(edge_index[0])):
        src_idx, tgt_idx = edge_index[0][idx], edge_index[1][idx]
        src_oid, tgt_oid = oids[src_idx], oids[tgt_idx]
        if src_oid not in oid2idxNew or tgt_oid not in oid2idxNew:
            continue
        src_idx_new, tgt_idx_new = oid2idxNew[src_oid], oid2idxNew[tgt_oid]
        key = (src_idx_new, tgt_idx_new)
        if key in edge_index_new:
            continue
        edge_index_new.append(key)
        edge_index_y_new.append(data['node', 'to', 'node'].y[idx])

    # Modify node_gt
    node_gt = data['node_gt']
    if len(node_gt.x) > 0:
        # modify 'node_gt','to','node'
        edge_index_gt = data['node_gt', 'to', 'node'].edge_index.tolist()
        for idx in range(len(edge_index_gt[0])):
            idx_node = edge_index_gt[1][idx]

            if idx_node >= 0:
                oid = oids[idx_node]
                if oid not in oid2idxNew:
                    idx_node = -1
                else:
                    idx_node = oid2idxNew[oid]
            data['node_gt', 'to', 'node'].edge_index[1, idx] = idx_node
            # edge_index_gt[1][idx] = idx_node

    # write back
    device = node.y.device
    data['node'].pd = torch.stack(
        node_pds_new, 0).detach().cpu()  # .to(device)
    data['node'].y = node_y_new.detach().cpu()
    data['node'].oid = torch.tensor(oids_new).detach().cpu()
    data['node', 'to', 'node'].edge_index = torch.tensor(
        edge_index_new, dtype=torch.long).t().contiguous().detach().cpu()
    data['node', 'to', 'node'].y = torch.stack(edge_index_y_new).detach(
    ).cpu() if len(edge_index_y_new) > 0 else torch.tensor(edge_index_new)
    # data['node_gt','to','node'].edge_index = torch.tensor(edge_index_gt,dtype=torch.long,device=device)


def get_metrics(label_id, confusion, VALID_CLASS_IDS: list = None):
    if VALID_CLASS_IDS is not None:
        if not label_id in VALID_CLASS_IDS:
            return float('nan')
    # #true positives
    tp = np.longlong(confusion[label_id, label_id])
    # #false negatives
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    # #false positives
    if VALID_CLASS_IDS is not None:
        not_ignored = [l for l in VALID_CLASS_IDS if not l == label_id]
        fp = np.longlong(confusion[not_ignored, label_id].sum())
    else:
        not_ignored = [l for l in range(
            confusion.shape[1]) if not l == label_id]
        fp = np.longlong(confusion[not_ignored, label_id].sum())

    denom = (tp + fp + fn)  # denominator

    iou = float('nan') if denom == 0 else (float(tp) / denom, tp, denom)
    precision = float('nan') if (
        tp+fp) == 0 else (float(tp) / (tp+fp), tp, tp+fp)
    recall = float('nan') if (tp+fn) == 0 else (float(tp) / (tp+fn), tp, tp+fn)
    return iou, precision, recall


def evaluate_topk_single_prediction(gts, pds, k=-1):
    top_k = list()
    pds = pds.detach().cpu()
    gts = gts.detach().cpu()
    size_o = len(pds)

    # calculate topk
    if k < 1:
        maxk = min(len(pds.shape[1]), 1)
    else:
        maxk = k

    for obj in range(size_o):
        pd = pds[obj]
        sorted_conf, sorted_args = torch.sort(pd, descending=True)  # 1D
        # if k<1:
        #     maxk=len(sorted_conf)
        #     maxk=min(len(sorted_conf),maxk)
        # else:
        #     maxk=k
        # sorted_conf=sorted_conf[:maxk]
        sorted_args = sorted_args[:maxk]

        gt = gts[obj]
        if gt in sorted_args:
            index = sorted(torch.where(sorted_args == gt)[0])[0].item()+1
        else:
            index = maxk+1
        top_k.append(index)

    return top_k

# def evaluate_topk_recall_single_prediction(gts:dict, pds, mask2inst:dict, k=-1):
#     top_k=list()
#     pds = pds.detach().cpu()
#     size_o = len(gts)

#     # calculate topk
#     if k<1:
#         maxk=min(len(pds.shape[1]),1)
#     else:
#         maxk=k

#     # Put prediction to a dict for easy query
#     pds_inst = dict()
#     for mask,inst in mask2inst.items():
#         assert inst not in pds_inst
#         pds_inst[inst] = pds[mask]

#     for inst, label in gts.items():
#         if inst not in pds_inst:
#             top_k.append(maxk+1)
#             continue

#         pd = pds_inst[inst]
#         sorted_conf, sorted_args = torch.sort(pd, descending=True) # 1D

#         # sorted_conf=sorted_conf[:maxk]
#         sorted_args=sorted_args[:maxk]

#         if label in sorted_args:
#             index = sorted(torch.where(sorted_args == label)[0])[0].item()+1
#         else:
#             index = maxk+1
#         top_k.append(index)

#     return top_k


def evaluate_topk_recall_single_prediction(gts, pds, edge_index_gtNode2pdNode, k=-1):
    top_k = list()
    pds = pds.detach().cpu()

    # calculate topk
    if k < 1:
        maxk = pds.shape[1]
    else:
        maxk = k

    for idx in range(edge_index_gtNode2pdNode.shape[1]):
        gtNodeIdx = edge_index_gtNode2pdNode[0, idx]
        pdNodeIdx = edge_index_gtNode2pdNode[1, idx]
        if pdNodeIdx < 0:  # missing prediction
            index = maxk+1
        else:
            pd = pds[pdNodeIdx]
            gt = gts[gtNodeIdx]
            _, sorted_args = torch.sort(pd, descending=True)  # 1D
            sorted_args = sorted_args[:maxk]

            if gt in sorted_args:  # check if the target id is truncated by maxk
                index = sorted(torch.where(sorted_args == gt)[0])[0].item()+1
            else:
                index = maxk+1
        top_k.append(index)
    return top_k


def evaluate_topk_multi_prediction(gts, pds, k=-1):
    top_k = list()
    pds = pds.detach().cpu()
    gts = gts.detach().cpu()
    size_o = len(pds)
    for obj in range(size_o):
        gt = gts[obj]

        if gt.sum() == 0:  # happens when use multi-rel. nothing to match. skip
            continue

        pd = pds[obj]
        sorted_conf, sorted_args = torch.sort(pd, descending=True)  # 1D
        if k < 1:
            maxk = len(sorted_conf)
            maxk = min(len(sorted_conf), maxk)
        else:
            maxk = k
        # sorted_conf=sorted_conf[:maxk]
        sorted_args = sorted_args[:maxk]

        indices = torch.where(gt > 0)[0].tolist()
        for index in indices:
            if index in sorted_args:
                index = sorted(torch.where(sorted_args == index)[0])[
                    0].item()+1
                index = math.ceil(index / len(indices))
            else:
                index = maxk+1
            top_k.append(index)
    return top_k


def evaluate_topk_predicate(gt_edges, rels_pred, threshold=0.5, k=-1):
    '''
    Find the distance of the predicted probability to the target probability. 
    Sorted the propagated predictions. The index of  the predicted probability is the distance.
    '''
    top_k = list()
    # if multi_rel_threshold:
    #     rels_pred = rels_pred.detach().cpu() # from sigmoid
    # else:
    rels_pred = rels_pred.detach().cpu()
    size_p = len(rels_pred)

    for rel in range(size_p):
        rel_pred = rels_pred[rel]
        rels_target = gt_edges[rel][2]
        if len(rels_target) == 0:  # Ground truth is None
            continue

        temp_topk = []
        sorted_args = []
        maxk = k
        if rel_pred.sum() == 0:  # happens when use multi-rel. nothing to match. skip
            pass
        else:
            # if len(rels_target) == 0: continue
            sorted_conf, sorted_args = torch.sort(
                rel_pred, descending=True)  # 1D
            if k < 1:
                maxk = min(len(sorted_conf), 1)  # take top1
            sorted_conf = sorted_conf[:maxk]
            sorted_args = sorted_args[:maxk]

        if len(rels_target) == 0:  # Ground truth is None
            # continue
            '''If gt is none, find the first prediction that is below threshold (which predicts "no relationship")'''
            indices = torch.where(sorted_conf < threshold)[0]
            if len(indices) == 0:
                index = maxk+1
            else:
                index = sorted(indices)[0].item()+1
            temp_topk.append(index)
        else:
            for gt in rels_target:
                '''if has gt, find the location of the gt label prediction.'''
                if gt in sorted_args:
                    index = sorted(torch.where(sorted_args == gt)[0])[
                        0].item()+1
                    index = math.ceil(index / len(rels_target))
                else:
                    index = maxk+1

                # if index != 1:
                #     print('hallo')

                # if len(indices) == 0:
                #     index = len(sorted_conf)+1
                # else:
                #     index = sorted(indices)[0].item()+1
                temp_topk.append(index)
        temp_topk = sorted(temp_topk)  # ascending I hope/think
        top_k += temp_topk
    return top_k

# def evaluate_topk_recall_predicate(gt_rel, rels_pred,edges,  mask2inst:dict, threshold=0.5,k=-1):
#     top_k=list()
#     rels_pred = rels_pred.detach().cpu()

#     assert edges.shape[0] == 2
#     num_edges = edges.shape[1]
#     all_indices = [idx for idx in range(num_edges)]

#     '''convert edge_index from mask to instance id'''
#     edges = edges.tolist()
#     edge_index_inst = list()
#     for idx in all_indices:
#         o_m = edges[0][idx]
#         s_m = edges[1][idx]
#         o_i = mask2inst[o_m]
#         s_i = mask2inst[s_m]
#         edge_index_inst.append((o_i,s_i))
#     # edge_index_inst = [(mask2inst[edges[0,idx].item()], mask2inst[edges[1,idx].item()]) for idx in all_indices]

#     '''calculate max k'''
#     if k<1:
#         maxk=min(rels_pred.shape[1],1) # take top1
#     else:
#         maxk=k

#     '''build lookup table'''
#     for line in gt_rel:
#         sub_index = line[0]
#         obj_index = line[1]
#         sub_cls_index = line[2]
#         obj_cls_index = line[3]
#         gt_r = line[4]

#         key = (sub_index,obj_index)
#         if key not in edge_index_inst:
#             #TODO: append maxk
#             for _ in gt_r: # for the multi rel case
#                 top_k += [maxk+1]
#             continue

#         # Get predicate prediction
#         edge_index = edge_index_inst.index(key)
#         rel_pred = rels_pred[edge_index]

#         sorted_conf, sorted_args = torch.sort(rel_pred, descending=True)  # 1D
#         # sorted_conf=sorted_conf[:maxk]
#         sorted_args=sorted_args[:maxk]

#         temp_topk = []
#         if len(gt_r) == 0:# Ground truth is None
#             # continue
#             '''If gt is none, find the first prediction that is below threshold (which predicts "no relationship")'''
#             indices = torch.where(sorted_conf < threshold)[0]
#             if len(indices) == 0:
#                 index = maxk+1
#             else:
#                 index = sorted(indices)[0].item()+1
#             temp_topk.append(index)
#         else:
#             for gt in gt_r:
#                 '''if has gt, find the location of the gt label prediction.'''
#                 if gt in sorted_args:
#                     index = sorted(torch.where(sorted_args == gt)[0])[0].item()+1
#                     index = math.ceil(index / len(gt_r))
#                 else:
#                     index = maxk+1
#                 temp_topk.append(index)
#             temp_topk = sorted(temp_topk)  # ascending I hope/think
#         top_k += temp_topk
#     return top_k


def evaluate_topk_recall_predicate(pds, gts,
                                   nodeGtIdx2nodeIdx,
                                   edge_index,
                                   edge_gt_index,
                                   threshold=0.5, k=-1):
    top_k = list()

    # calculate max k
    if k < 1:
        maxk = pds.shape[1]  # take top1
    else:
        maxk = k

    # Convert list to map
    nodeGtIdx2nodeIdx_map = dict()
    for idx in range(nodeGtIdx2nodeIdx.shape[1]):
        src_idx = nodeGtIdx2nodeIdx[0, idx].item()
        tgt_idx = nodeGtIdx2nodeIdx[1, idx].item()
        nodeGtIdx2nodeIdx_map[src_idx] = tgt_idx

    # calculate topk
    edge_index_list = edge_index.t().tolist()
    for idx in range(edge_gt_index.shape[1]):
        missing = False
        gt_rels = gts[idx]
        src_idx = edge_gt_index[0, idx].item()
        tgt_idx = edge_gt_index[1, idx].item()
        # check if mapping exist
        if src_idx not in nodeGtIdx2nodeIdx_map or tgt_idx not in nodeGtIdx2nodeIdx_map:
            missing = True
        else:
            src_idx = nodeGtIdx2nodeIdx_map[src_idx]
            tgt_idx = nodeGtIdx2nodeIdx_map[tgt_idx]
            key = [src_idx, tgt_idx]
            if key not in edge_index_list:
                missing = True
        if missing == True:  # If missing add maximum top_k
            for _ in gt_rels:
                top_k += [maxk+1]
            continue

        # compute top_k
        idx = edge_index_list.index(key)
        rel_pred = pds[idx]
        sorted_conf, sorted_args = torch.sort(rel_pred, descending=True)  # 1D
        sorted_conf = sorted_conf[:maxk]
        sorted_args = sorted_args[:maxk]

        temp_topk = []
        if len(gt_rels) == 0:  # Ground truth is None
            # continue
            '''If gt is none, find the first prediction that is below threshold (which predicts "no relationship")'''
            indices = torch.where(sorted_conf < threshold)[0]
            if len(indices) == 0:
                index = maxk+1
            else:
                index = sorted(indices)[0].item()+1
            temp_topk.append(index)
        else:
            for gt in gt_rels:
                '''if has gt, find the location of the gt label prediction.'''
                if gt in sorted_args:
                    index = sorted(torch.where(sorted_args == gt)[0])[
                        0].item()+1
                    index = math.ceil(index / len(gt_rels))
                else:
                    index = maxk+1
                temp_topk.append(index)
            temp_topk = sorted(temp_topk)  # ascending I hope/think
        top_k += temp_topk
    return top_k


def get_gt(objs_target, rels_target, edges, mask2inst, multiple_prediction: bool):
    gt_edges = []  # initialize
    idx2instance = torch.zeros_like(objs_target)
    for idx, iid in mask2inst.items():
        # if idx > 0:
        idx2instance[idx] = iid

    for edge_index in range(len(edges)):
        idx_eo = edges[edge_index][0].cpu().numpy().item()
        idx_os = edges[edge_index][1].cpu().numpy().item()

        target_eo = objs_target[idx_eo].cpu().numpy().item()
        target_os = objs_target[idx_os].cpu().numpy().item()
        target_rel = []  # there might be multiple
        if multiple_prediction:
            assert rels_target.ndim == 2
            for i in range(rels_target.size(1)):
                if rels_target[edge_index, i] == 1:
                    target_rel.append(i)
        else:
            assert rels_target.ndim == 1
            if rels_target[edge_index] > 0:
                target_rel.append(rels_target[edge_index].cpu().numpy().item())
        gt_edges.append([target_eo, target_os, target_rel,
                         idx2instance[idx_eo], idx2instance[idx_os]])
    return gt_edges


def build_gt_triplet(objs_target, rels_target, edges, multiple_prediction: bool):
    gt_edges = []  # initialize
    assert edges.shape[0] == 2
    num_connections = edges.shape[1]
    for edge_index in range(num_connections):
        idx_eo = edges[0, edge_index].cpu().numpy().item()
        idx_os = edges[1, edge_index].cpu().numpy().item()

        target_eo = objs_target[idx_eo].cpu().numpy().item()
        target_os = objs_target[idx_os].cpu().numpy().item()
        target_rel = []  # there might be multiple
        if multiple_prediction:
            assert rels_target.ndim == 2
            for i in range(rels_target.size(1)):
                if rels_target[edge_index, i] == 1:
                    target_rel.append(i)
        else:
            assert rels_target.ndim == 1
            target_rel.append(rels_target[edge_index].cpu().numpy().item())
        gt_edges.append([target_eo, target_os, target_rel])
    return gt_edges


def evaluate_topk(gt_rel, objs_pred, rels_pred, edges, threshold=0.5, k=40, ignore_none: bool = False):
    top_k = list()
    device = objs_pred.device
    objs_pred = objs_pred.detach().cpu()
    rels_pred = rels_pred.detach().cpu()

    batch_size = 64
    assert edges.shape[0] == 2
    num_edges = edges.shape[1]
    all_indices = [idx for idx in range(num_edges)]

    for indices in torch.split(torch.LongTensor(all_indices), batch_size):
        sub_preds = objs_pred[edges[0, indices]]
        obj_preds = objs_pred[edges[1, indices]]
        rel_preds = rels_pred[indices]
        so_preds = torch.einsum('bn,bm->bnm', sub_preds, obj_preds)
        conf_matrix = torch.einsum('bnm, bk->bnmk', so_preds, rel_preds)
        batch, dim_n, dim_m, dim_k = conf_matrix.shape

        conf_matrix = conf_matrix.reshape(
            so_preds.shape[0], -1).to(device)  # use CUDA if available
        torch.cuda.empty_cache()
        sorted_conf_matrix, sorted_args_1d = torch.sort(
            conf_matrix, descending=True)  # 1D
        sorted_conf_matrix, sorted_args_1d = sorted_conf_matrix.cpu(), sorted_args_1d.cpu()
        torch.cuda.empty_cache()
        if k < 1:
            maxk = min(len(sorted_conf_matrix), 1)  # take top1
        else:
            maxk = k
        sorted_conf_matrix = sorted_conf_matrix[:, :maxk]
        sorted_args_1d = sorted_args_1d[:, :maxk]

        for mask, idx in enumerate(indices):
            e = gt_rel[idx]
            gt_s = e[0]
            gt_t = e[1]
            gt_r = e[2]
            temp_topk = []
            if len(gt_r) == 0:
                if ignore_none:
                    continue
                # Ground truth is None
                indices = torch.where(sorted_conf_matrix[mask] < threshold)[0]
                if len(indices) == 0:
                    index = maxk+1
                else:
                    index = sorted(indices)[0].item()+1
                temp_topk.append(index)
            else:
                for predicate in gt_r:  # for the multi rel case
                    index_1d = (gt_s*dim_m+gt_t)*dim_k+predicate
                    indices = torch.where(sorted_args_1d[mask] == index_1d)[0]
                    if len(indices) == 0:
                        index = maxk+1
                    else:
                        index = sorted(indices)[0].item()+1
                        index = math.ceil(index/len(gt_r))
                    temp_topk.append(index)
            temp_topk = sorted(temp_topk)
            top_k += temp_topk
    # for idx in range(len(edges)):
    #     edge_from = edges[idx][0]
    #     edge_to = edges[idx][1]
    #     e = gt_rel[idx]
    #     gt_s = e[0]
    #     gt_t = e[1]
    #     gt_r = e[2]
    #     # if len(gt_r) == 0: continue

    #     rel_predictions = rels_pred[idx]
    #     objs_pred_1 = objs_pred[edge_from]
    #     objs_pred_2 = objs_pred[edge_to]
    #     node_score = torch.einsum('n,l->nl',objs_pred_1,objs_pred_2)
    #     conf_matrix = torch.einsum('nl,m->nlm',node_score,rel_predictions)
    #     conf_matrix_1d = conf_matrix.reshape(-1)
    #     sorted_conf_matrix, sorted_args_1d = torch.sort(conf_matrix_1d, descending=True) # 1D
    #     if k<1:
    #         maxk=len(sorted_conf_matrix)
    #         maxk=min(len(sorted_conf_matrix),maxk)
    #     else:
    #         maxk=k
    #     sorted_conf_matrix=sorted_conf_matrix[:maxk]
    #     sorted_args_1d=sorted_args_1d[:maxk]

    #     temp_topk = []

    #     if len(gt_r) == 0:
    #         # Ground truth is None
    #         indices = torch.where(sorted_conf_matrix < threshold)[0]
    #         if len(indices) == 0:
    #             index = maxk+1 #
    #         else:
    #             index = sorted(indices)[0].item()+1
    #         temp_topk.append(index)
    #     for predicate in gt_r: # for the multi rel case
    #         index_1d = (gt_s*conf_matrix.shape[1]+gt_t)*conf_matrix.shape[2]+predicate
    #         indices = torch.where(sorted_args_1d == index_1d)[0]
    #         # gt_conf = conf_matrix[gt_s, gt_t, predicate]
    #         # indices = torch.where(sorted_conf_matrix == gt_conf)[0]
    #         if len(indices) == 0:
    #             index = maxk+1
    #         else:
    #             index = sorted(indices)[0].item()+1
    #         temp_topk.append(index)
    #     temp_topk = sorted(temp_topk)
    #     top_k += temp_topk
    return top_k

# def evaluate_topk_recall(gt_rel, objs_pred, rels_pred, edges, mask2inst:dict, threshold=0.5, k=40, ignore_none:bool=False):
#     top_k=list()
#     device = objs_pred.device
#     objs_pred = objs_pred.detach().cpu()
#     rels_pred = rels_pred.detach().cpu()

#     batch_size = 64
#     assert edges.shape[0] == 2
#     num_edges = edges.shape[1]
#     all_indices = [idx for idx in range(num_edges)]

#     '''convert edge_index from mask to instance id'''
#     edges = edges.tolist()
#     edge_index_inst = list()
#     for idx in all_indices:
#         o_m = edges[0][idx]
#         s_m = edges[1][idx]
#         o_i = mask2inst[o_m]
#         s_i = mask2inst[s_m]
#         edge_index_inst.append((o_i,s_i))
#     # edge_index_inst = [(mask2inst[edges[0,idx].item()], mask2inst[edges[1,idx].item()]) for idx in all_indices]

#     '''calculate max k'''
#     if k<1:
#         maxk=min(sub_preds.shape[1]*2+rel_preds.shape[1],1) # take top1
#     else:
#         maxk=k

#     '''build lookup table'''
#     for line in gt_rel:
#         sub_index = line[0]
#         obj_index = line[1]
#         sub_cls_index = line[2]
#         obj_cls_index = line[3]
#         gt_r = line[4]

#         key = (sub_index,obj_index)
#         if key not in edge_index_inst:
#             #TODO: append maxk
#             for _ in gt_r: # for the multi rel case
#                 top_k += [maxk+1]
#             continue

#         '''calculate topK'''
#         edge_index = edge_index_inst.index(key)
#         sub_preds = objs_pred[edges[0][edge_index]]
#         obj_preds = objs_pred[edges[1][edge_index]]
#         rel_preds = rels_pred[edge_index]

#         so_preds = torch.einsum('n,m->nm',sub_preds,obj_preds)
#         conf_matrix = torch.einsum('nm, k->nmk',so_preds,rel_preds)
#         dim_n, dim_m, dim_k = conf_matrix.shape

#         conf_matrix = conf_matrix.reshape(-1).to(device) # use CUDA if available
#         torch.cuda.empty_cache()
#         sorted_conf_matrix, sorted_args_1d = torch.sort(conf_matrix, descending=True) # 1D
#         sorted_conf_matrix, sorted_args_1d=sorted_conf_matrix.cpu(),sorted_args_1d.cpu()
#         torch.cuda.empty_cache()

#         sorted_conf_matrix=sorted_conf_matrix[:maxk]
#         sorted_args_1d=sorted_args_1d[:maxk]

#         temp_topk = []
#         if len(gt_r) == 0:
#             if ignore_none:
#                 continue
#             # Ground truth is None
#             indices = torch.where(sorted_conf_matrix < threshold)[0]
#             if len(indices) == 0:
#                 index = maxk+1 #
#             else:
#                 index = sorted(indices)[0].item()+1
#             temp_topk.append(index)
#         else:
#             for predicate in gt_r: # for the multi rel case
#                 index_1d = (sub_cls_index*dim_m+obj_cls_index)*dim_k+predicate
#                 indices = torch.where(sorted_args_1d == index_1d)[0]
#                 if len(indices) == 0:
#                     index = maxk+1
#                 else:
#                     index = sorted(indices)[0].item()+1
#                     index = math.ceil(index/len(gt_r))
#                 temp_topk.append(index)
#             temp_topk = sorted(temp_topk)
#         top_k += temp_topk
#     return top_k


def evaluate_topk_recall(entity_pds, edge_pds,
                         entity_gts, edge_gts,
                         nodeGtIdx2nodeIdx,
                         edge_index,
                         edge_gt_index,
                         threshold=0.5, k=-1):
    device = entity_pds.device
    top_k = list()

    '''calculate max k'''
    if k < 1:
        maxk = entity_pds.shape[1]*entity_pds.shape[1] * \
            edge_pds.shape[1]  # take top1
    else:
        maxk = k

    # Convert list to map
    nodeGtIdx2nodeIdx_map = dict()
    for idx in range(nodeGtIdx2nodeIdx.shape[1]):
        gt_idx = nodeGtIdx2nodeIdx[0, idx].item()
        pd_idx = nodeGtIdx2nodeIdx[1, idx].item()
        nodeGtIdx2nodeIdx_map[gt_idx] = pd_idx

    # calculate topk
    edge_index_list = edge_index.t().tolist()
    for idx in range(edge_gt_index.shape[1]):
        missing = False
        gt_rels = edge_gts[idx]
        src_gt_idx = edge_gt_index[0, idx].item()
        tgt_gt_idx = edge_gt_index[1, idx].item()
        # check if mapping exist
        if src_gt_idx not in nodeGtIdx2nodeIdx_map or tgt_gt_idx not in nodeGtIdx2nodeIdx_map:
            missing = True
        else:
            src_pd_idx = nodeGtIdx2nodeIdx_map[src_gt_idx]
            tgt_pd_idx = nodeGtIdx2nodeIdx_map[tgt_gt_idx]
            key = [src_pd_idx, tgt_pd_idx]
            if key not in edge_index_list:
                missing = True
        if missing == True:  # If missing add maximum top_k
            for _ in gt_rels:
                top_k += [maxk+1]
            continue

        # Collect GT cls idx
        src_gt_cls = entity_gts[src_gt_idx]
        tgt_gt_cls = entity_gts[tgt_gt_idx]  # TODO: ck if this is a list

        # collect PD feature
        idx = edge_index_list.index(key)
        rel_pd = edge_pds[idx]
        src_pd = entity_pds[src_pd_idx]
        tgt_pd = entity_pds[tgt_pd_idx]

        # concat src-tgt-pred
        so_preds = torch.einsum('n,m->nm', src_pd, tgt_pd)
        conf_matrix = torch.einsum('nm, k->nmk', so_preds, rel_pd)
        sorted_conf, sorted_args = torch.sort(
            conf_matrix, descending=True)  # 1D
        dim_n, dim_m, dim_k = conf_matrix.shape

        # use CUDA if available
        conf_matrix = conf_matrix.reshape(-1).to(device)
        torch.cuda.empty_cache()
        sorted_conf_matrix, sorted_args_1d = torch.sort(
            conf_matrix, descending=True)  # 1D
        sorted_conf_matrix, sorted_args_1d = sorted_conf_matrix.cpu(), sorted_args_1d.cpu()
        torch.cuda.empty_cache()

        sorted_conf_matrix = sorted_conf_matrix[:maxk]
        sorted_args_1d = sorted_args_1d[:maxk]

        temp_topk = []
        if len(gt_rels) == 0:
            # Ground truth is None
            indices = torch.where(sorted_conf_matrix < threshold)[0]
            if len(indices) == 0:
                index = maxk+1
            else:
                index = sorted(indices)[0].item()+1
            temp_topk.append(index)
        else:
            for predicate in gt_rels:  # for the multi rel case
                index_1d = (src_gt_cls*dim_m+tgt_gt_cls)*dim_k+predicate
                indices = torch.where(sorted_args_1d == index_1d)[0]
                if len(indices) == 0:
                    index = maxk+1
                else:
                    index = sorted(indices)[0].item()+1
                    index = math.ceil(index/len(gt_rels))
                temp_topk.append(index)
            temp_topk = sorted(temp_topk)
        top_k += temp_topk
    return top_k

# def evaluate_topk_recall(
#         top_k, top_k_obj, top_k_predicate,
#         objs_pred:torch.tensor, objs_target:torch.tensor,
#         rels_pred:torch.tensor, rels_target:torch.tensor,
#         edges, instance2mask):

#     gt_edges = get_gt(objs_target, rels_target, edges, instance2mask)
#     top_k += evaluate_topk(gt_edges, objs_pred, rels_pred, edges) # class_labels, relationships_dict)
#     top_k_obj += evaluate_topk_single_prediction(objs_target, objs_pred)
#     top_k_predicate += evaluate_topk_predicate(gt_edges, rels_pred)
#     return top_k, top_k_obj, top_k_predicate


def cal_mean(values, VALID_CLASS_IDS: list, CLASS_LABELS: list):
    if len(VALID_CLASS_IDS) == 0:
        VALID_CLASS_IDS = [i for i in range(len(CLASS_LABELS))]
    if isinstance(values, list):
        sums = list()
        for v in values:
            sum = 0
            counter = 0
            for i in range(len(VALID_CLASS_IDS)):
                label_name = CLASS_LABELS[i]
                if isinstance(v[label_name], tuple):
                    sum += v[label_name][0]
                    counter += 1
            sum /= (counter+1e-12)
            sums.append(sum)
    elif isinstance(values, dict):
        sums = dict()
        for k, v in values.items():
            # sums[k] = dict()
            sum = 0
            counter = 0
            for i in range(len(VALID_CLASS_IDS)):
                label_name = CLASS_LABELS[i]
                if isinstance(v[label_name], tuple):
                    sum += v[label_name][0]
                    counter += 1
            sum /= (counter+1e-12)
            sums[k] = sum
            # sums[k].append(sum)
    return sums


def get_metrics_all(confusion: np.array, VALID_CLASS_IDS: list, CLASS_LABELS: list):
    '''
    return mean_iou, mean_precision, mean_recall
    '''
    outputs = dict()
    ious = dict()
    precisions = dict()
    recalls = dict()
    if len(VALID_CLASS_IDS) == 0:
        VALID_CLASS_IDS = [i for i in range(len(CLASS_LABELS))]
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        label_id = VALID_CLASS_IDS[i]
        ious[label_name], precisions[label_name], recalls[label_name] \
            = get_metrics(label_id, confusion, VALID_CLASS_IDS)

    outputs['iou'] = ious
    outputs['precisions'] = precisions
    outputs['recalls'] = recalls
    return outputs  # (ious, precisions, recalls)
    # return cal_mean(ious),cal_mean(precisions),cal_mean(recalls)


def write_result_file(confusion: np.array,
                      filename: str,
                      VALID_CLASS_IDS: list, CLASS_LABELS: list):
    ious = dict()
    precisions = dict()
    recalls = dict()

    if len(VALID_CLASS_IDS) == 0:
        VALID_CLASS_IDS = [i for i in range(len(CLASS_LABELS))]
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        label_id = VALID_CLASS_IDS[i]
        ious[label_name], precisions[label_name], recalls[label_name] \
            = get_metrics(label_id, confusion, VALID_CLASS_IDS)

    with open(filename, 'w') as f:
        def write_metric(name, values):
            f.write('{} scores\n'.format(name))
            sum = 0
            counter = 0
            for i in range(len(VALID_CLASS_IDS)):
                label_id = VALID_CLASS_IDS[i]
                label_name = CLASS_LABELS[i]
                if isinstance(values[label_name], tuple):
                    # value = values[label_name][0]
                    f.write('{0:<14s}({1:<2d}): {2:>5.3f}   ({3:>6d}/{4:<6d})\n'.format(label_name,
                            label_id, values[label_name][0], values[label_name][1], values[label_name][2]))
                    sum += values[label_name][0]
                    counter += 1
                else:
                    f.write('{0:<14s}({1:<2d}): nan\n'.format(
                        label_name, label_id))
            sum /= (counter+1e-12)
            f.write('{0:<18s}: {1:>5.3f}\n'.format('Average', sum))

            for i in range(len(VALID_CLASS_IDS)):
                if i > 0:
                    f.write(' & ')
                label_id = VALID_CLASS_IDS[i]
                label_name = CLASS_LABELS[i]
                if isinstance(values[label_name], tuple):
                    value = values[label_name][0]
                    f.write('{:>5.3f}'.format(value))

                else:
                    f.write('nan')

            f.write(' & {:>5.3f}\n'.format(sum))
            return sum
        mean_iou = write_metric("IoU", ious)
        mean_pre = write_metric("Precision", precisions)
        mean_rec = write_metric("Recall", recalls)
        f.write('{0:<14s}: {1:>5.3f}   ({2:>6f}/{3:<6f})\n\n'.format('overall recall',
                                                                     confusion.trace()/confusion.sum(),
                                                                     confusion.trace(),
                                                                     confusion.sum()))
        f.write('{0:<14s}: {1:>5.3f}   ({2:>6f}/{3:<6f})\n\n'.format('mean recall',
                                                                     (confusion.diagonal(
                                                                     )/confusion.sum(1)).sum(),
                                                                     confusion.trace(),
                                                                     confusion.sum()))

        f.write('\nconfusion matrix\n')
        f.write('\t\t\t')
        for i in range(len(VALID_CLASS_IDS)):
            # f.write('\t{0:<14s}({1:<2d})'.format(CLASS_LABELS[i], VALID_CLASS_IDS[i]))
            f.write('{0:<8d}'.format(VALID_CLASS_IDS[i]))
        f.write('\n')
        for r in range(len(VALID_CLASS_IDS)):
            f.write('{0:<14s}({1:<2d})'.format(
                CLASS_LABELS[r], VALID_CLASS_IDS[r]))
            for c in range(len(VALID_CLASS_IDS)):
                f.write('\t{0:>5.3f}'.format(
                    confusion[VALID_CLASS_IDS[r], VALID_CLASS_IDS[c]]))
            f.write('\n')
    print('wrote results to', filename)
    return [mean_iou, mean_pre, mean_rec]


def build_seg2name(pds: torch.tensor, mask2inst: list, names):
    '''
    pds: [n]
    '''
    s2n = dict()
    for n in range(len(pds)):
        try:
            s2n[str(mask2inst[n])] = names[pds[n]]
        except:
            # print(edge)
            # print(pds[n])
            raise RuntimeError('')

    return s2n


def build_edge2name(pds: torch.tensor, edges: torch.tensor,
                    idx2seg: dict, names: list):
    if edges.shape[0] == 2:
        edges = edges.t()

    edges = edges.cpu().tolist()
    # pds=pds.cpu()

    s2n = dict()

    if pds.ndim == 2:  # has multiple prediction
        s2n = defaultdict(list)
        for n in range(pds.shape[0]):
            n_i = idx2seg[edges[n][0]]
            n_j = idx2seg[edges[n][1]]
            edge = str(n_i)+'_'+str(n_j)

            # pd_names=list()
            indices = pds[n].nonzero().tolist()
            if len(indices) == 0:
                continue
            pd_names = [names[x[0]] for x in indices]

            # if edge not in s2n:
            #     s2n[edge]=list()
            # else
            s2n[edge] += pd_names
            #     if pds[n][c]>0:
            #         s2n[edge].append( names[c] )
    else:
        for n in range(pds.shape[0]):
            n_i = idx2seg[edges[n][0]]
            n_j = idx2seg[edges[n][1]]
            edge = str(n_i)+'_'+str(n_j)
            try:
                s2n[edge] = names[pds[n]]
            except:
                print(edge)
                print(names)
                print(pds[n])
                raise RuntimeError('')

    return s2n


def build_edge2name_value(values: torch.tensor, edges: torch.tensor, idx2gtcls: dict, names: list):
    if edges.shape[0] == 2:
        edges = edges.t()
    edges = edges.cpu().tolist()

    nn2v = dict()
    for n in range(edges.shape[0]):
        n_i = idx2gtcls[edges[n][0]]
        n_j = idx2gtcls[edges[n][1]]
        if names[n_i] not in nn2v:
            nn2v[names[n_i]] = dict()
        if names[n_j] not in nn2v[names[n_i]]:
            nn2v[names[n_i]][names[n_j]] = list()
        nn2v[names[n_i]][names[n_j]].append(values[n])
    return nn2v


class EvaPairWeight():
    def __init__(self, class_names: list):
        self.class_names = class_names
        self.c_mat = np.zeros(
            [len(self.class_names), len(self.class_names)], dtype=float)

    def update(self, values, edge_indices, idx2gtcls):
        nn2vs = build_edge2name_value(
            values, edge_indices, idx2gtcls, self.class_names)
        for name1, n2vs in nn2vs.items():
            for name2, vs in n2vs.items():
                if len(vs) == 0:
                    continue
                a_vs = np.array(vs).mean()
                idx1 = self.class_names.index(name1)
                idx2 = self.class_names.index(name2)
                self.c_mat[idx1][idx2] += a_vs

    def reset(self):
        self.c_mat = np.zeros(
            [len(self.class_names), len(self.class_names)], dtype=float)

# class EvaClassificationSimple(object):
#     def __init__(self,class_names):
#         '''
#         This class takes the prediction and ground truth indices as the input,
#         and keep updating a confusion matrix.

#         Returns
#         -------
#         None.

#         '''
#         self.class_names = class_names
#         self.c_mat = np.zeros([len(self.class_names),len(self.class_names)], dtype=float) # cmat[gt][pd]
#         pass
#     def __call__(self,pds, gts):
#         self.update(pds,gts)
#     def update(self,pds, gts):
#         assert len(pds) == len(gts)
#         for i in range(len(pds)):
#             pd = pds[i]
#             gt = gts[i]
#             self.c_mat[gt][pd]+=1

#     def get_recall(self):
#         return self.c_mat.diagonal().sum() / self.c_mat.sum()
#     def get_all_metrics(self):
#         return get_metrics_all(self.c_mat, [], self.class_names)
#     def get_mean_metrics(self):
#         return cal_mean(self.get_all_metrics(), [], self.class_names)
#     def reset(self):
#         self.c_mat = np.zeros([len(self.class_names),len(self.class_names)], dtype=float)
#     def draw(self, **args):
#         args['y_labels']=self.class_names
#         args['x_labels']=self.class_names
#         return plot_confusion_matrix(self.c_mat,
#                           **args)
#     def write_result_file(self, filename, VALID_CLASS_IDS):
#         return write_result_file(self.c_mat, filename, VALID_CLASS_IDS, self.class_names)
#     def gen_text(self):
#         c_recall = self.get_recall()
#         txt = "recall obj cls {}".format(c_recall) +'\n'
#         return txt

#     def write(self, path, model_name):
#         pathlib.Path(path).mkdir(parents=True, exist_ok=True)

#         obj_results = self.write_result_file(os.path.join(path,model_name+'_results_obj.txt'), [])

#         r_o = {k: v for v, k in zip(obj_results, ['Obj_IOU','Obj_Precision', 'Obj_Recall']) }
#         results = r_o

#         self.draw(
#             title='object confusion matrix',
#             normalize='log',
#             plot_text=False,
#             plot=False,
#             grid=False,
#             pth_out=os.path.join(path, model_name + "_obj_cmat.png")
#         )
#         return results


class ConfusionMatrix():
    def __init__(self, class_names: list):
        self.class_names = class_names
        self.c_mat = np.zeros([len(self.class_names), len(
            self.class_names)], dtype=float)  # cmat[gt][pd]

    def get_recall(self):
        return self.c_mat.diagonal().sum() / self.c_mat.sum()

    def get_all_metrics(self):
        return get_metrics_all(self.c_mat, [], self.class_names)

    def get_mean_metrics(self):
        return cal_mean(self.get_all_metrics(), [], self.class_names)

    def reset(self):
        self.c_mat = np.zeros(
            [len(self.class_names), len(self.class_names)], dtype=float)

    def draw(self, **args):
        args['y_labels'] = self.class_names
        args['x_labels'] = self.class_names
        return plot_confusion_matrix(self.c_mat,
                                     **args)

    def write_result_file(self, filename, VALID_CLASS_IDS):
        return write_result_file(self.c_mat, filename, VALID_CLASS_IDS, self.class_names)

    def __call__(self, *args, **kwards):
        self.update(*args, **kwards)

    def update(*args, **kwards):
        raise NotImplementedError()


class EvaClassificationSimple(ConfusionMatrix):
    def __init__(self, class_names: list):
        super().__init__(class_names)

    def __call__(self, pd_tensor: Tensor, gt_tensor: Tensor):
        self.update(pd_tensor, gt_tensor)

    def update(self, pd_tensor: Tensor, gt_tensor: Tensor):
        assert len(pd_tensor) == len(gt_tensor)
        for i in range(len(pd_tensor)):
            pd_idx, gt_idx = pd_tensor[i], gt_tensor[i]
            self.c_mat[gt_idx][pd_idx] += 1


class EvaClassification(ConfusionMatrix):
    def __init__(self, class_names: list, none_name: str = 'UN'):
        '''find if none is already included'''
        if none_name in class_names:
            self.unknown = class_names.index(
                none_name)  # TODO: remove none_name
            class_names = class_names
        else:
            self.unknown = len(class_names)
            class_names = class_names + [none_name]

        super().__init__(class_names)
        self.none_name = none_name

    def update(self, pd_indices: dict, gt_indices: dict, gt_only=False, pd_only=False):
        union_indices = set(pd_indices.keys()).union(gt_indices.keys())
        multi_pred = True
        for k, v in pd_indices.items():
            multi_pred = isinstance(v, list)
            break

        for idx in union_indices:  # since it's union, there is no way to get both unknown.
            if not multi_pred:
                pd = self.unknown if idx not in pd_indices else pd_indices[idx]
                pd_idx = self.class_names.index(
                    pd) if pd in self.class_names else self.unknown
                gt = self.unknown if idx not in gt_indices else gt_indices[idx]
                gt_idx = self.class_names.index(
                    gt) if gt in self.class_names else self.unknown
                self.c_mat[gt_idx][pd_idx] += 1
            else:
                def get_indices(indices) -> list:
                    if idx not in indices:
                        idxes = [self.unknown]
                    else:
                        assert isinstance(indices[idx], list)
                        idxes = [self.class_names.index(
                            i) for i in indices[idx]]
                    return idxes
                pd_indices_set = set(get_indices(
                    pd_indices)).difference([self.unknown])
                gt_indices_set = set(get_indices(
                    gt_indices)).difference([self.unknown])

                if len(gt_indices_set) == 0 and len(pd_indices_set) == 0:
                    # true negative
                    self.c_mat[self.unknown][self.unknown] += 1

                # for every unmatched prediction, generate confusion to the other unmatched, otherwise treat them as unknown
                if len(gt_indices_set) > 0:
                    # TP
                    intersection = set(pd_indices_set).intersection(
                        gt_indices_set)  # match prediction
                    diff_gt = set(gt_indices_set).difference(
                        intersection)  # unmatched gt
                    diff_pd = set(pd_indices_set).difference(
                        intersection)  # unmatched pd
                    for i in intersection:
                        self.c_mat[i][i] += 1  # true positive
                    if not gt_only:
                        for pd_idx in diff_pd:  # false positive
                            self.c_mat[self.unknown][pd_idx] += 1
                            # if len(diff_gt) > 0:
                            #     # pass
                            #     for gt_idx in diff_gt:
                            #         self.c_mat[gt_idx][pd_idx] += 1/len(diff_gt)
                            # else:
                            #     # pass
                            #     # print('hello!', self.unknown, pd_idx)
                            # self.c_mat[self.unknown][pd_idx] += 1
                    if not pd_only:
                        for gt_idx in diff_gt:  # false negative
                            self.c_mat[gt_idx][self.unknown] += 1
                            # if len(diff_pd) > 0:
                            #     for pd_idx in diff_pd:
                            #         self.c_mat[gt_idx][pd_idx] += 1/len(diff_pd)
                            # else:
                            #     pass
                            # self.c_mat[gt_idx][self.unknown] += 1
                elif len(gt_indices_set) == 0:
                    for idx in pd_indices_set:
                        self.c_mat[self.unknown][idx] += 1


class EvaMultiBinaryClassification(object):
    def __init__(self, class_names: list):
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.c_mat = np.zeros([len(self.class_names), 4],
                              dtype=int)  # cmat[gt][pd]

    def __call__(self, pds, gts):
        self.update(pds, gts)

    def update(self, pds, gts):
        '''

        Parameters
        ----------
        pds : TYPE
            dim: [n,n_class]
        gts : TYPE
            dim: [n,n_class]
        Returns
        -------
        None.

        '''
        TP = ((gts == True) * (pds == True)).sum(0)
        TN = ((gts == False) * (pds == False)).sum(0)
        FP = ((gts == False) * (pds == True)).sum(0)
        FN = ((gts == True) * (pds == False)).sum(0)

        if isinstance(TP, torch.Tensor):
            TP = TP.cpu().numpy()
            TN = TN.cpu().numpy()
            FP = FP.cpu().numpy()
            FN = FN.cpu().numpy()

        self.c_mat[:, 0] += TP
        self.c_mat[:, 1] += TN
        self.c_mat[:, 2] += FP
        self.c_mat[:, 3] += FN

    def get_recall(self, full=False):
        # TP / (TP+FN)
        if not full:
            return self.c_mat[:, 0] / (self.c_mat[:, 0] + self.c_mat[:, 3])
        else:
            return (self.c_mat[:, 0] / (self.c_mat[:, 0] + self.c_mat[:, 3]), self.c_mat[:, 0], self.c_mat[:, 0] + self.c_mat[:, 3])

    def get_precision(self, full=False):
        # TP / (TP+FP)
        if not full:
            return self.c_mat[:, 0] / (self.c_mat[:, 0]+self.c_mat[:, 2])
        else:
            return (self.c_mat[:, 0] / (self.c_mat[:, 0]+self.c_mat[:, 2]), self.c_mat[:, 0], self.c_mat[:, 0]+self.c_mat[:, 2])

    def get_accuracy(self, full=False):
        # (TP+TN) / (ALL)
        if not full:
            return (self.c_mat[:, 0]+self.c_mat[:, 1]) / (self.c_mat.sum(1))
        else:
            return ((self.c_mat[:, 0]+self.c_mat[:, 1]) / (self.c_mat.sum(1)), self.c_mat[:, 0]+self.c_mat[:, 1], self.c_mat.sum(1))

    def get_mean_metric(self, VALID_CLASS_IDS: list = None):
        '''
        return mean_iou, mean_precision, mean_recall
        '''
        CLASS_LABELS = self.class_names
        if VALID_CLASS_IDS is None:
            VALID_CLASS_IDS = [i for i in range(len(CLASS_LABELS))]

        metrics = self.get_all_metrics()
        metrics = {k: v[VALID_CLASS_IDS].mean() for k, v in metrics.items()}
        # mean_iou = ious[VALID_CLASS_IDS].mean()
        # mean_pre = pres[VALID_CLASS_IDS].mean()
        # mean_rec = recs[VALID_CLASS_IDS].mean()
        # outputs = dict()
        # outputs['iou'] = mean_iou
        # outputs['precisions'] = mean_pre
        # outputs['recalls'] = mean_rec
        return metrics

    def get_all_metrics(self):
        '''
        return mean_iou, mean_precision, mean_recall
        '''
        # for binary classification, IoU = Accuracy (intersection=TP+TN, union=P+N)
        ious = self.get_accuracy()
        pres = self.get_precision()
        recs = self.get_recall()

        outputs = dict()
        outputs['iou'] = ious
        outputs['precisions'] = pres
        outputs['recalls'] = recs

        outputs = {k: np.nan_to_num(v) for k, v in outputs.items()}

        return outputs

    def get_mean_metrics(self):
        # cal_mean(self.get_all_metrics(), [], self.class_names)
        return self.get_mean_metric()

    def draw(self, **args):
        args['y_labels'] = self.class_names
        args['x_labels'] = ['TP', 'TN', 'FP', 'FN']
        return plot_confusion_matrix(self.c_mat,
                                     **args)

    def write_result_file(self, filename, VALID_CLASS_IDS):
        CLASS_LABELS = self.class_names
        if len(VALID_CLASS_IDS) == 0:
            VALID_CLASS_IDS = [i for i in range(len(CLASS_LABELS))]

        ious = self.get_accuracy(True)
        pres = self.get_precision(True)
        recs = self.get_recall(True)
        ious = {CLASS_LABELS[idx]: (ious[0][idx], ious[1][idx], ious[2][idx])
                if ious[2][idx] > 0 else float('nan') for idx in VALID_CLASS_IDS}
        precisions = {CLASS_LABELS[idx]: (pres[0][idx], pres[1][idx], pres[2][idx])
                      if pres[2][idx] > 0 else float('nan') for idx in VALID_CLASS_IDS}
        recalls = {CLASS_LABELS[idx]: (recs[0][idx], recs[1][idx], recs[2][idx])
                   if recs[2][idx] > 0 else float('nan') for idx in VALID_CLASS_IDS}

        # for i in range(len(VALID_CLASS_IDS)):
        #     label_name = CLASS_LABELS[i]
        #     label_id = VALID_CLASS_IDS[i]
        #     ious[label_name], precisions[label_name], recalls[label_name] \
        #         = get_metrics(label_id, confusion,VALID_CLASS_IDS)

        with open(filename, 'w') as f:
            def write_metric(name, values):
                f.write('{} scores\n'.format(name))
                sum = 0
                counter = 0
                for i in range(len(VALID_CLASS_IDS)):
                    label_id = VALID_CLASS_IDS[i]
                    label_name = CLASS_LABELS[i]
                    if isinstance(values[label_name], tuple):
                        # value = values[label_name][0]
                        f.write('{0:<14s}({1:<2d}): {2:>5.3f}   ({3:>6d}/{4:<6d})\n'.format(label_name,
                                label_id, values[label_name][0], values[label_name][1], values[label_name][2]))
                        sum += values[label_name][0]
                        counter += 1
                    else:
                        f.write('{0:<14s}({1:<2d}): nan\n'.format(
                            label_name, label_id))
                sum /= (counter+1e-12)
                f.write('{0:<18s}: {1:>5.3f}\n'.format('Average', sum))

                for i in range(len(VALID_CLASS_IDS)):
                    if i > 0:
                        f.write(' & ')
                    label_id = VALID_CLASS_IDS[i]
                    label_name = CLASS_LABELS[i]
                    if isinstance(values[label_name], tuple):
                        value = values[label_name][0]
                        f.write('{:>5.3f}'.format(value))

                    else:
                        f.write('nan')

                f.write(' & {:>5.3f}\n'.format(sum))
                return sum
            mean_iou = write_metric("IoU", ious)
            mean_pre = write_metric("Precision", precisions)
            mean_rec = write_metric("Recall", recalls)

            # f.write('{0:<14s}: {1:>5.3f}   ({2:>6f}/{3:<6f})\n\n'.format('accuracy', \
            #                                                           confusion.trace()/confusion.sum(),  \
            #                                                           confusion.trace(), \
            #                                                           confusion.sum()) )

            # f.write('\nconfusion matrix\n')
            # f.write('\t\t\t')
            # for i in range(len(VALID_CLASS_IDS)):
            #     #f.write('\t{0:<14s}({1:<2d})'.format(CLASS_LABELS[i], VALID_CLASS_IDS[i]))
            #     f.write('{0:<8d}'.format(VALID_CLASS_IDS[i]))
            # f.write('\n')
            # for r in range(len(VALID_CLASS_IDS)):
            #     f.write('{0:<14s}({1:<2d})'.format(CLASS_LABELS[r], VALID_CLASS_IDS[r]))
            #     for c in range(len(VALID_CLASS_IDS)):
            #         f.write('\t{0:>5.3f}'.format(confusion[VALID_CLASS_IDS[r],VALID_CLASS_IDS[c]]))
            #     f.write('\n')
        print('wrote results to', filename)
        return [mean_iou, mean_pre, mean_rec]


class EvalSceneGraphBase():
    def __init__(self):
        pass
        self.eva_o_cls = ConfusionMatrix(['node'])
        self.eva_p_cls = ConfusionMatrix(['none'])
        self.top_k_triplet = list()
        self.top_k_triplet_recall = list()
        self.top_k_obj = list()
        self.top_k_obj_recall = list()
        self.top_k_rel = list()
        self.top_k_rel_recall = list()
        self.ignore_none = False

    def get_recall(self):
        return self.eva_o_cls.get_recall(), self.eva_p_cls.get_recall()

    def get_mean_metrics(self):
        return self.eva_o_cls.get_mean_metrics(), self.eva_p_cls.get_mean_metrics()

    def gen_text(self):
        c_recall = self.eva_o_cls.get_recall()
        r_recall = self.eva_p_cls.get_recall()

        txt = "recall obj cls {}".format(c_recall) + '\n'
        txt += "recall rel cls {}".format(r_recall) + '\n'
        if self.k > 0:
            # print("Recall@k for relationship triplets: ")
            txt += "Recall@k for relationship triplets: "+'\n'
            ntop_k = np.asarray(self.top_k_triplet)
            ks = set([1, 2, 3, 5, 10, 50, 100])
            for i in [0, 0.05, 0.1, 0.2, 0.5, 0.9]:
                ks.add(int(math.ceil(self.k*i+1e-9)))
            for k in sorted(ks):
                R = (ntop_k <= k).sum() / len(ntop_k)
                # print("top-k R@" + str(k), "\t", R)
                txt += "top-k R@{}\t {}".format(k, R)+'\n'
            # print(len(self.top_k_triplet))
            txt += str(len(self.top_k_triplet)) + '\n'

            # print("Recall@k for objects: ")
            txt += 'Recall@k for objects: \n'
            ntop_k_obj = np.asarray(self.top_k_obj)
            ks = set([1, 2, 3, 4, 5, 10, 50, 100])
            for i in [0, 0.05, 0.1, 0.2, 0.5]:
                ks.add(int(math.ceil(len(self.obj_class_names)*i+1e-9)))
            for k in sorted(ks):
                R = (ntop_k_obj <= k).sum() / len(ntop_k_obj)
                # print("top-k R@" + str(k), "\t", R)
                txt += "top-k R@{}\t {}".format(k, R)+'\n'
            txt += str(len(self.top_k_obj)) + '\n'

            # print("Recall@k for predicates: ")
            txt += "Recall@k for predicates: \n"
            ntop_k_predicate = np.asarray(self.top_k_rel)
            ks = set([1, 2, 3, 4, 5, 10])
            for i in [0, 0.05, 0.1, 0.2, 0.5]:
                ks.add(int(math.ceil(len(self.rel_class_names)*i + 1e-9)))
            for k in sorted(ks):
                R = (ntop_k_predicate <= k).sum() / len(ntop_k_predicate)
                # print("top-k R@" + str(k), "\t", R)
                txt += "top-k R@{}\t {}".format(k, R)+'\n'
            # print(len(self.top_k_rel))
            txt += str(len(self.top_k_rel)) + '\n'

            txt += "New Recall@k for relationship triplets: "+'\n'
            ntop_k = np.asarray(self.top_k_triplet_recall)
            ks = set([1, 2, 3, 5, 10, 50, 100])
            for i in [0, 0.05, 0.1, 0.2, 0.5, 0.9]:
                ks.add(int(math.ceil(self.k*i+1e-9)))
            for k in sorted(ks):
                R = (ntop_k <= k).sum() / len(ntop_k)
                # print("top-k R@" + str(k), "\t", R)
                txt += "top-k R@{}\t {}".format(k, R)+'\n'
            txt += str(len(self.top_k_triplet_recall)) + '\n'

            txt += 'Mew Recall@k for objects: \n'
            ntop_k_obj = np.asarray(self.top_k_obj_recall)
            ks = set([1, 2, 3, 4, 5, 10, 50, 100])
            for i in [0, 0.05, 0.1, 0.2, 0.5]:
                ks.add(int(math.ceil(len(self.obj_class_names)*i+1e-9)))
            for k in sorted(ks):
                R = (ntop_k_obj <= k).sum() / len(ntop_k_obj)
                # print("top-k R@" + str(k), "\t", R)
                txt += "top-k R@{}\t {}".format(k, R)+'\n'
            txt += str(len(self.top_k_obj_recall)) + '\n'

            txt += "New Recall@k for predicates: \n"
            ntop_k_predicate = np.asarray(self.top_k_rel_recall)
            ks = set([1, 2, 3, 4, 5, 10])
            for i in [0, 0.05, 0.1, 0.2, 0.5]:
                ks.add(int(math.ceil(len(self.rel_class_names)*i + 1e-9)))
            for k in sorted(ks):
                R = (ntop_k_predicate <= k).sum() / len(ntop_k_predicate)
                # print("top-k R@" + str(k), "\t", R)
                txt += "top-k R@{}\t {}".format(k, R)+'\n'
            txt += str(len(self.top_k_rel_recall)) + '\n'
        return txt

    def draw(self, **args):
        fig_o = self.eva_o_cls.draw(
            title='object confusion matrix',
            **args
        )
        fig_r = self.eva_p_cls.draw(
            title='predicate confusion matrix',
            **args
        )
        return fig_o, fig_r

    def write(self, path, model_name):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(path, 'predictions.json'), 'w') as f:
            json.dump(self.predictions, f, indent=4)

        obj_results = self.eva_o_cls.write_result_file(
            os.path.join(path, model_name+'_results_obj.txt'), [])
        rel_results = self.eva_p_cls.write_result_file(
            os.path.join(path, model_name+'_results_rel.txt'), [])

        r_o = {k: v for v, k in zip(
            obj_results, ['Obj_IOU', 'Obj_Precision', 'Obj_Recall'])}
        r_r = {k: v for v, k in zip(
            rel_results, ['Rel_IOU', 'Rel_Precision', 'Rel_Recall'])}
        results = {**r_o, **r_r}

        self.eva_o_cls.draw(
            title='object confusion matrix',
            normalize='log',
            plot_text=False,
            plot=False,
            grid=False,
            pth_out=os.path.join(path, model_name + "_obj_cmat.png")
        )
        self.eva_p_cls.draw(
            title='predicate confusion matrix',
            normalize='log',
            plot_text=False,
            plot=False,
            grid=False,
            pth_out=os.path.join(path, model_name + "_rel_cmat.png")
        )

        # if  opt.eval_topk:
        with open(os.path.join(path, model_name + '_topk.txt'), 'w+') as f:
            f.write(self.gen_text())
        return results


class EvalSceneGraphBatch(EvalSceneGraphBase):
    def __init__(self, obj_class_names: list, rel_class_names: list, multi_rel_threshold: float = 0.5, k=100,
                 save_prediction: bool = False,
                 multi_rel_prediction: bool = True, none_name='none',
                 ignore_none: bool = False):
        super().__init__()
        self.obj_class_names = obj_class_names
        self.rel_class_names = rel_class_names
        self.multi_rel_threshold = multi_rel_threshold
        self.multi_rel_prediction = multi_rel_prediction
        self.k = k
        self.none_name = none_name
        self.save_prediction = save_prediction
        self.ignore_none = ignore_none

        # object cls
        self.eva_o_cls = EvaClassificationSimple(obj_class_names)
        # predicate cls
        self.eva_p_cls = EvaClassificationSimple(
            rel_class_names) if not multi_rel_prediction else EvaMultiBinaryClassification(rel_class_names)

        self.top_k_triplet = list()
        self.top_k_triplet_recall = list()
        self.top_k_obj = list()
        self.top_k_obj_recall = list()
        self.top_k_rel = list()
        self.top_k_rel_recall = list()
        self.predictions = dict()

    def reset(self):
        self.eva_o_cls.reset()
        self.eva_p_cls.reset()
        self.top_k_triplet = list()
        self.top_k_triplet_recall = list()
        self.top_k_obj = list()
        self.top_k_obj_recall = list()
        self.top_k_rel = list()
        self.top_k_rel_recall = list()
        self.predictions = dict()

    def add(self,
            data: HeteroData,
            # scan_id:str, obj_pds, bobj_gts, rel_pds, brel_gts, mask2insts:dict, edge_indices,
            # gt_relationships:list=None
            ):
        '''
        obj_pds: [n, n_cls]: softmax
        obj_gts: [n, 1]: long tensor
        rel_pds: [m,n_cls]: torch.sigmoid(x) if multi_rel_threshold>0 else softmax
        rel_gts: [m,n_cls] if multi_rel_threshold>0 else [m,1]
        '''

        node_data = data['node']
        edge_data = data['node', 'to', 'node']
        node_gt_data = data['node_gt']
        edge_gt_data = data['node_gt', 'to', 'node_gt']
        node_gt_to_node = data['node_gt', 'to', 'node']

        # Shortcut
        node_pds = node_data.pd
        node_gts = node_data.y
        node_pds = node_pds.detach()

        edge_gts = edge_data.y
        has_rel = 'pd' in edge_data and len(edge_gts) > 0
        if has_rel:
            edge_pds = edge_data.pd

            # if self.ignore_none and not self.multi_rel_prediction:
            #     none_index = self.rel_class_names.index(self.none_name)
            #     non_none_index = torch.where(edge_gts != none_index)
            #     edge_gts = edge_gts[non_none_index]
            #     edge_pds = edge_pds[non_none_index]

            #     # edge_indices_bk = copy.copy(edge_indices)
            #     edge_indices = edge_indices[:,non_none_index[0]]
            # has_rel = edge_pds is not None and len(edge_pds)>0 and edge_pds.shape[0] > 0
            # if has_rel:
            edge_pds = edge_pds.detach()

        # Obj
        o_pds = node_pds.max(1)[1]
        self.eva_o_cls(o_pds, node_gts)

        # Rel
        if has_rel:
            r_pds = edge_pds > self.multi_rel_threshold if self.multi_rel_prediction else edge_pds.max(1)[
                1]
            self.eva_p_cls(r_pds, edge_gts)

        # Topk
        if self.k > 0:
            self.top_k_obj += evaluate_topk_single_prediction(
                node_gts, node_pds, k=self.k)

            if has_rel:

                # if self.ignore_none:
                #     # Filter out none predictions
                #     none_index = self.rel_class_names.index(self.none_name)
                #     non_none_index = torch.where(edge_gts != none_index)
                #     non_none_brel_gts = edge_gts[non_none_index]
                #     non_none_brel_pds = edge_pds[non_none_index]

                #     self.top_k_rel += evaluate_topk_single_prediction(non_none_brel_gts,non_none_brel_pds,k=self.k)

                if not self.multi_rel_prediction:
                    self.top_k_rel += evaluate_topk_single_prediction(
                        edge_gts, edge_pds, k=self.k)
                else:
                    self.top_k_rel += evaluate_topk_multi_prediction(
                        edge_gts, edge_pds, k=self.k)

                gt_rel_triplet = build_gt_triplet(
                    node_gts, edge_gts, edge_data.edge_index, self.multi_rel_prediction)

                self.top_k_triplet += evaluate_topk(gt_rel_triplet, node_pds, edge_pds, edge_data.edge_index,
                                                    threshold=self.multi_rel_threshold, k=self.k, ignore_none=self.ignore_none)  # class_labels, relationships_dict)

        # return
        '''calculate topK recall'''
        # Build mask2instance
        # mask2instance = dict()
        # for idx in range(node_gt_to_node.shape[1]):
        #     idx_node_gt = node_gt_to_node[0,idx].item()
        #     idx_node = node_gt_to_node[1,idx].item()
        #     assert idx_node not in mask2instance
        #     mask2instance[idx_node] = idx_node_gt

        # compute object topk recall

        # Merge list from batches to one
        if len(node_gt_data.x) > 0:
            # in batch process list objects will be merged into a larger list.
            if 'batch' in node_gt_data:
                node_gt_data.clsIdx = list(
                    itertools.chain.from_iterable(node_gt_data.clsIdx))
                edge_gt_data.clsIdx = list(
                    itertools.chain.from_iterable(edge_gt_data.clsIdx))
            self.top_k_obj_recall += evaluate_topk_recall_single_prediction(
                node_gt_data.clsIdx, node_pds, node_gt_to_node.edge_index, self.k)
            if has_rel:
                # compute predicate topk recall
                self.top_k_rel_recall += evaluate_topk_recall_predicate(edge_pds, edge_gt_data.clsIdx,
                                                                        node_gt_to_node.edge_index,
                                                                        edge_data.edge_index, edge_gt_data.edge_index,
                                                                        threshold=self.multi_rel_threshold, k=self.k)
                # compute triplet
                self.top_k_triplet_recall += evaluate_topk_recall(
                    node_pds, edge_pds,
                    node_gt_data.clsIdx, edge_gt_data.clsIdx,
                    node_gt_to_node.edge_index,
                    edge_data.edge_index, edge_gt_data.edge_index,
                    threshold=self.multi_rel_threshold, k=self.k)

        # Collect object list
        # if gt_relationships is not None:
        #     gt_obj_ids = dict()
        #     for line in gt_relationships:
        #         (sub_o,tgt_o,sub_cls_id,tgt_cls_id,_) = line
        #         if sub_o not in gt_obj_ids:
        #             gt_obj_ids[sub_o] = sub_cls_id
        #         else: # check is label consistent
        #             assert gt_obj_ids[sub_o] == sub_cls_id
        #         if tgt_o not in gt_obj_ids:
        #             gt_obj_ids[tgt_o] = tgt_cls_id
        #         else:
        #             assert gt_obj_ids[tgt_o] == tgt_cls_id

        #     if self.k>0:
        #         # compute object topk recall
        #         self.top_k_obj_recall += evaluate_topk_recall_single_prediction(gt_obj_ids,node_pds,mask2insts[0],self.k)
        #         if has_rel:
        #             # compute predicate topk recall
        #             self.top_k_rel_recall += evaluate_topk_recall_predicate(gt_relationships,edge_pds, edge_indices,mask2insts[0],threshold=self.multi_rel_threshold,k=self.k)

        #             # compute triplet
        #             self.top_k_triplet_recall += evaluate_topk_recall(gt_relationships, node_pds, edge_pds, edge_indices, mask2insts[0],
        #                                 threshold=self.multi_rel_threshold, k=self.k, ignore_none=self.ignore_none)

        # Write prediction
        return
        # TODO: update to suppport new input
        if self.save_prediction:
            if len(scan_id) != 1:
                raise RuntimeError('batch size must be 1')
            scan_id = scan_id[0]
            mask2insts = mask2insts[0]

            pd = dict()
            gt = dict()
            '''update node'''
            pd['nodes'] = dict()
            gt['nodes'] = dict()
            pd['nodes'] = build_seg2name(
                o_pds,   mask2insts, self.obj_class_names)
            gt['nodes'] = build_seg2name(
                node_gts, mask2insts, self.obj_class_names)

            self.predictions[scan_id] = dict()
            self.predictions[scan_id]['pd'] = pd
            self.predictions[scan_id]['gt'] = gt


class EvalSceneGraph(EvalSceneGraphBase):
    def __init__(self, obj_class_names: list, rel_class_names: list, multi_rel_threshold: float = 0.5, k=100, multi_rel_prediction: bool = True,
                 save_prediction: bool = False,
                 none_name='none'):
        super().__init__()
        # params
        self.obj_class_names = obj_class_names
        self.rel_class_names = rel_class_names
        self.multi_rel_threshold = multi_rel_threshold
        self.multi_rel_prediction = multi_rel_prediction
        self.k = k
        self.none_name = none_name
        self.save_prediction = save_prediction

        # object cls
        self.eva_o_cls = EvaClassification(
            obj_class_names, none_name=none_name)
        # predicate cls
        self.eva_p_cls = EvaClassification(
            rel_class_names, none_name=none_name) if not multi_rel_prediction else EvaMultiBinaryClassification(rel_class_names)

        self.predictions = dict()
        self.top_k_triplet = list()
        self.top_k_obj = list()
        self.top_k_obj_recall = list()
        self.top_k_rel = list()

    def reset(self):
        self.eva_o_cls.reset()
        self.eva_p_cls.reset()
        self.top_k_triplet = list()
        self.top_k_obj = list()
        self.top_k_obj_recall = list()
        self.top_k_rel = list()
        self.predictions = dict()

    def add(self, scan_ids, obj_pds, bobj_gts, rel_pds, brel_gts, mask2insts: list, bedge_indices):
        '''
        obj_pds: [n, n_cls]: softmax
        obj_gts: [n, 1]: long tensor
        rel_pds: [m,n_cls]: torch.sigmoid(x) if multi_rel_threshold>0 else softmax
        rel_gts: [m,n_cls] if multi_rel_threshold>0 else [m,1]
        '''
        obj_pds = obj_pds.detach()
        has_rel = rel_pds is not None and len(
            rel_pds) > 0 and rel_pds.shape[0] > 0
        if has_rel:
            rel_pds = rel_pds.detach()

        o_pds = obj_pds.max(1)[1]
        if has_rel:
            r_pds = rel_pds > self.multi_rel_threshold if self.multi_rel_prediction else rel_pds.max(1)[
                1]

        '''build idx mapping'''
        mask2insts_flat = dict()

        for it in range(len(mask2insts)):
            mask2inst = mask2insts[it]
            for idx, iid in mask2inst.items():
                idx = idx.item() if isinstance(idx, torch.Tensor) else idx
                if iid == 0:
                    continue
                if idx < 0:
                    continue
                # print(idx)
                # assert idx not in idx2seg

                if idx in mask2insts_flat:
                    print('')
                    print(iid)
                    print(idx)
                    print(mask2insts)
                    assert idx not in mask2insts_flat
                mask2insts_flat[idx] = iid

        b_pd = dict()
        b_gt = dict()
        b_pd['nodes'] = build_seg2name(
            o_pds,   mask2insts_flat, self.obj_class_names)
        b_gt['nodes'] = build_seg2name(
            bobj_gts, mask2insts_flat, self.obj_class_names)
        self.eva_o_cls.update(b_pd['nodes'], b_gt['nodes'], False)

        for it in range(len(mask2insts)):
            mask2inst = mask2insts[it]
            scan_id = scan_ids[it]

            indices = torch.tensor(list(mask2inst.keys()))
            indices = indices[indices >= 0]

            # idx2seg=dict()
            # for iid,idx in seg2idx.items():
            #     if isinstance(idx, torch.Tensor):
            #         idx2seg[idx.item()] = iid
            #     else:
            #         idx2seg[idx] = iid

            pd = dict()
            gt = dict()
            '''update node'''
            # o_pd = o_pds[indices]
            # obj_gts = bobj_gts[indices]
            # pd['nodes'] = build_seg2name(o_pd,   idx2seg,self.obj_class_names)
            # gt['nodes'] = build_seg2name(obj_gts,idx2seg,self.obj_class_names)
            pd['nodes'] = dict()
            gt['nodes'] = dict()
            for idx in indices.tolist():
                seg_id = str(mask2inst[idx])
                pd['nodes'][seg_id] = b_pd['nodes'][seg_id]
                gt['nodes'][seg_id] = b_gt['nodes'][seg_id]
            # pd['nodes'] = {str(idx2seg[idx]): b_pd['nodes'][str(idx2seg[idx])] for idx in indices.tolist()   }#  b_pd['nodes'][indices]
            # gt['nodes'] = b_gt['nodes'][indices]
            self.eva_o_cls.update(pd['nodes'], gt['nodes'], False)

            '''update edge'''
            # get relavent edges
            # print('')
            # print('bedge_indices.shape',bedge_indices.shape)
            if bedge_indices.nelement() > 1:
                if bedge_indices.shape[0] == 2:
                    bedge_indices = bedge_indices.t().contiguous()
                edge_indices_ = torch.where((bedge_indices[:, 0] <= max(
                    indices)) & (bedge_indices[:, 0] >= min(indices)))[0]
                edge_indices = bedge_indices[edge_indices_]
                r_pd = r_pds[edge_indices_]
                rel_gts = brel_gts[edge_indices_]

                if has_rel:
                    if self.multi_rel_prediction:
                        assert self.multi_rel_threshold > 0
                        pd['edges'] = build_edge2name(
                            r_pd,    edge_indices, mask2inst, self.rel_class_names)
                        gt['edges'] = build_edge2name(
                            rel_gts, edge_indices, mask2inst, self.rel_class_names)
                        '''for multiple rel prediction. every predicate should be treated individually'''
                        self.eva_p_cls.update(r_pd, rel_gts)
                    else:
                        pd['edges'] = build_edge2name(
                            r_pd,    edge_indices, mask2inst, self.rel_class_names)
                        gt['edges'] = build_edge2name(
                            rel_gts, edge_indices, mask2inst, self.rel_class_names)
                        self.eva_p_cls.update(pd['edges'], gt['edges'], False)

                if self.k > 0:
                    self.top_k_obj += evaluate_topk_single_prediction(
                        bobj_gts, obj_pds, k=self.k)

                    if has_rel:
                        gt_edges = get_gt(
                            bobj_gts, rel_gts, edge_indices, mask2inst, self.multi_rel_prediction)
                        self.top_k_rel += evaluate_topk_predicate(gt_edges, rel_pds,
                                                                  threshold=self.multi_rel_threshold,
                                                                  k=self.k)

                        self.top_k_triplet += evaluate_topk(gt_edges, obj_pds, rel_pds, edge_indices,
                                                            threshold=self.multi_rel_threshold, k=self.k)  # class_labels, relationships_dict)
            if self.save_prediction:
                self.predictions[scan_id] = dict()
                self.predictions[scan_id]['pd'] = pd
                self.predictions[scan_id]['gt'] = gt


class EvalUpperBound():
    def __init__(self, node_cls_names, edge_cls_names, noneidx_node_cls, noneidx_edge_cls, multi_rel: bool, topK: int, none_name: str):
        # if multi_rel: raise NotImplementedError()
        self.multi_rel = multi_rel
        self.node_cls_names = node_cls_names
        self.edge_cls_names = edge_cls_names
        self.noneidx_node_cls = noneidx_node_cls
        self.noneidx_edge_cls = noneidx_edge_cls
        '''evaluate'''
        self.eval_tool = EvalSceneGraphBatch(node_cls_names, edge_cls_names,
                                             multi_rel_prediction=multi_rel, k=topK,
                                             save_prediction=True,
                                             none_name=none_name)

    def __call__(self, data_seg, data_inst, is_eval_image: bool):
        # Shortcuts
        # scan_id = data_inst['scan_id']
        inst_oids = data_inst['node'].oid  # data_inst['mask2instance']
        inst_gt_cls = data_inst['node'].y  # data_inst['gt_cls']
        inst_gt_rel = data_inst['node', 'to', 'node'].y  # data_inst['gt_rel']
        # data_inst['node_edges']
        inst_node_edges = data_inst['node', 'to', 'node'].edge_index
        # gt_relationships = data_inst['relationships']

        if data_seg is None:
            node_pred = torch.zeros_like(torch.nn.functional.one_hot(
                inst_gt_cls, len(self.node_cls_names))).float()
            node_pred[:, self.noneidx_node_cls] = 1.0

            if not self.multi_rel:
                edge_pred = torch.zeros_like(torch.nn.functional.one_hot(
                    inst_gt_rel, len(self.edge_cls_names))).float()
                edge_pred[:, self.noneidx_edge_cls] = 1.0
            else:
                edge_pred = torch.zeros_like(inst_gt_rel).float()

            data_inst['node'].pd = node_pred.detach()
            data_inst['node', 'to', 'node'].pd = edge_pred.detach()

            self.eval_tool.add(data_inst)
            return

        # Get data from data_seg
        if not is_eval_image:
            seg_gt_cls = data_seg['node'].y
            seg_oids = data_seg['node'].oid  # instance ID # idx2oid[0]

            seg_node_edges = data_seg['node', 'to', 'node'].edge_index
        else:
            seg_gt_cls = data_seg['roi'].y
            seg_oids = data_seg['roi'].oid
            seg_node_edges = data_seg['roi', 'to', 'roi'].edge_index

        # convert them to list
        assert inst_node_edges.shape[0] == 2
        # assert seg_node_edges.shape[0] == 2
        inst_node_edges = inst_node_edges.tolist()
        seg_node_edges = seg_node_edges.tolist()
        seg_oids = seg_oids.tolist()
        inst_oids = inst_oids.tolist()

        # collect inst in seg
        seg_instance_set = set()
        for idx in range(len(seg_oids)):
            seg_instance_set.add(seg_oids[idx])
        # for _, inst in seg_mask2inst.items():
        #     seg_instance_set.add(inst)

        # Find missing nodes
        missing_nodes = list()
        for idx in range(len(inst_oids)):
            inst = inst_oids[idx]
            if inst not in seg_instance_set:
                missing_nodes.append(idx)

        # build seg inst key

        seg_inst_pair_edges = set()
        if len(seg_node_edges) > 0:
            for idx in range(len(seg_node_edges[0])):
                src, tgt = seg_node_edges[0][idx], seg_node_edges[1][idx]
                # seg_mask2inst[src],seg_mask2inst[tgt]
                src, tgt = seg_oids[src], seg_oids[tgt]
                seg_inst_pair_edges.add((src, tgt))

        # Find missing edges

        missing_edges = list()
        for mask in range(len(inst_node_edges[0])):
            src, tgt = inst_node_edges[0][mask], inst_node_edges[1][mask]
            # inst_mask2instance[src], inst_mask2instance[tgt]
            src, tgt = inst_oids[src], inst_oids[tgt]
            key = (src, tgt)
            if key not in seg_inst_pair_edges:
                missing_edges.append(mask)

        ''' Create Foo prdiction vector '''
        # Nodes
        node_pred = torch.nn.functional.one_hot(
            inst_gt_cls, len(self.node_cls_names)).float()
        node_pred[missing_nodes] = 0
        node_pred[missing_nodes, self.noneidx_node_cls] = 1.0

        # edges
        if not self.multi_rel:
            edge_pred = torch.nn.functional.one_hot(
                inst_gt_rel, len(self.edge_cls_names)).float()
            edge_pred[missing_edges] = 0
            edge_pred[missing_edges, self.noneidx_edge_cls] = 1.0
        else:
            edge_pred = inst_gt_rel.clone().float()
            edge_pred[missing_edges] = 0

        data_inst['node'].pd = node_pred.detach()
        data_inst['node', 'to', 'node'].pd = edge_pred.detach()
        self.eval_tool.add(data_inst)

        return len(missing_nodes)/len(seg_instance_set), len(missing_edges)/len(inst_gt_rel)


if __name__ == '__main__':
    tt = EvaClassification(['1', '2'], [0, 1])
    pd = dict()
    gt = dict()
    pd[0] = '1'
    pd[1] = '1'
    gt[0] = '1'
    gt[1] = '2'
    tt.update(pd, gt, False)
