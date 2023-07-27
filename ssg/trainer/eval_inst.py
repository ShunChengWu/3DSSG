from collections import defaultdict
import numpy as np
import torch

from tqdm import tqdm
from codeLib.utils import moving_average
from ssg import define
from ssg.utils.util_data import match_class_info_from_two, merge_batch_mask2inst
from ssg.utils.util_eva import EvalSceneGraphBatch, EvalUpperBound
import codeLib.utils.string_numpy as snp


class EvalInst(object):
    def __init__(self):
        pass

    def evaluate_inst(self, dataset_seg, dataset_inst, topk):
        is_eval_image = self.cfg.model.method in ['imp']
        ignore_missing = self.cfg.eval.ignore_missing

        '''add a none class for missing instances'''
        (scanid2idx_seg, _, node_cls_names, edge_cls_names, noneidx_node_cls, noneidx_edge_cls,
            seg_valid_node_cls_indices, inst_valid_node_cls_indices,
            seg_valid_edge_cls_indices, inst_valid_edge_cls_indices) = \
            match_class_info_from_two(
                dataset_seg, dataset_inst, multi_rel=self.cfg.model.multi_rel)

        '''all'''
        eval_tool_all = EvalSceneGraphBatch(node_cls_names, edge_cls_names,
                                            multi_rel_prediction=self.cfg.model.multi_rel, k=topk, save_prediction=True,
                                            none_name=define.NAME_NONE, ignore_none=False)

        '''ignore none'''
        eval_tool_ignore_none = EvalSceneGraphBatch(node_cls_names, edge_cls_names,
                                                    multi_rel_prediction=self.cfg.model.multi_rel, k=topk, save_prediction=True,
                                                    none_name=define.NAME_NONE, ignore_none=True)
        # , 'ignore_none': eval_tool_ignore_none}
        eval_tools = {'all': eval_tool_all}

        # eval_upper_bound
        eval_UpperBound = EvalUpperBound(node_cls_names, edge_cls_names, noneidx_node_cls, noneidx_edge_cls,
                                         multi_rel=self.cfg.model.multi_rel, topK=topk, none_name=define.NAME_NONE)

        eval_list = defaultdict(moving_average.MA)

        ''' get scan_idx mapping '''
        scanid2idx_seg = dict()
        for index in range(len(dataset_seg)):
            scan_id = snp.unpack(dataset_seg.scans, index)  # self.scans[idx]
            scanid2idx_seg[scan_id] = index

        scanid2idx_inst = dict()
        for index in range(len(dataset_inst)):
            scan_id = snp.unpack(dataset_inst.scans, index)  # self.scans[idx]
            scanid2idx_inst[scan_id] = index

        '''start eval'''
        self.model.eval()
        for index in tqdm(range(len(dataset_inst))):
            data_inst = dataset_inst.__getitem__(index)
            scan_id_inst = data_inst['scan_id']

            # Find the same scan in dataset_seg
            if scan_id_inst not in scanid2idx_seg:
                data_seg = None
            else:
                index_seg = scanid2idx_seg[scan_id_inst]
                data_seg = dataset_seg.__getitem__(index_seg)
                assert data_seg['scan_id'] == data_inst['scan_id']

            '''process seg'''
            eval_dict = {}
            with torch.no_grad():
                data_seg = self.process_data_dict(data_seg)
                data_inst = self.process_data_dict(data_inst)
                # record in eval_UB
                eval_UpperBound(data_seg, data_inst, is_eval_image)
                # continue

                # Shortcuts
                inst_oids = data_inst['node'].oid  # data_inst['mask2instance']
                inst_gt_cls = data_inst['node'].y  # data_inst['gt_cls']
                # data_inst['seg_gt_rel']
                inst_gt_rel = data_inst['node', 'to', 'node'].y
                # data_inst['node_edges']
                inst_node_edges = data_inst['node', 'to', 'node'].edge_index

                if data_seg is None:
                    '''
                    If no target scan in dataset_seg is found, set all prediction to none
                    '''
                    # Nodes
                    node_pred = torch.zeros_like(torch.nn.functional.one_hot(
                        inst_gt_cls, len(node_cls_names))).float()
                    node_pred[:, noneidx_node_cls] = 1.0

                    # Edges
                    if not self.cfg.model.multi_rel:
                        edge_pred = torch.zeros_like(torch.nn.functional.one_hot(
                            inst_gt_rel, len(edge_cls_names))).float()
                        edge_pred[:, noneidx_edge_cls] = 1.0
                    else:
                        edge_pred = torch.zeros_like(inst_gt_rel).float()

                    # log
                    data_inst['node'].pd = node_pred.detach()
                    data_inst['node', 'to', 'node'].pd = edge_pred.detach()
                    for eval_tool in eval_tools.values():
                        eval_tool.add(data_inst)
                    continue

                # Get data from data_seg
                if not is_eval_image:
                    # seg_gt_cls = data_seg['node'].y
                    seg_gt_rel = data_seg['node', 'to', 'node'].y
                    seg_oids = data_seg['node'].oid
                    seg_node_edges = data_seg['node', 'to', 'node'].edge_index
                else:
                    # seg_gt_cls = data_seg['roi'].y
                    seg_gt_rel = data_seg['roi', 'to', 'roi'].y
                    seg_oids = data_seg['roi'].oid
                    seg_node_edges = data_seg['roi', 'to', 'roi'].edge_index
                    # seg2inst = data_seg['roi'].get('idx2iid',None)

                ''' make forward pass through the network '''
                node_cls, edge_cls = self.model(data_seg)

                # convert them to list
                assert inst_node_edges.shape[0] == 2
                # assert seg_node_edges.shape[0] == 2
                inst_node_edges = inst_node_edges.tolist()
                seg_node_edges = seg_node_edges.tolist()
                seg_oids = seg_oids.tolist()
                inst_oids = inst_oids.tolist()

                '''merge prediction from seg to instance (in case of "same part")'''
                # use list bcuz may have multiple predictions on the same object instance
                seg_oid2idx = defaultdict(list)
                for idx in range(len(seg_oids)):
                    seg_oid2idx[seg_oids[idx]].append(idx)

                '''merge nodes'''
                merged_idx2oid = dict()
                merged_oid2idx = dict()
                merged_node_cls = torch.zeros(
                    len(inst_oids), len(node_cls_names)).to(self.cfg.DEVICE)
                # set all initial prediction to none
                merged_node_cls_gt = (torch.ones(
                    len(inst_oids), dtype=torch.long) * noneidx_node_cls).to(self.cfg.DEVICE)
                counter = 0
                # go over all instances and merge their predictions
                for idx in range(len(inst_oids)):
                    oid = inst_oids[idx]
                    if not ignore_missing:
                        merged_oid2idx[oid] = idx
                        merged_idx2oid[idx] = oid
                        # use GT class
                        merged_node_cls_gt[idx] = inst_gt_cls[idx]
                        if oid in seg_oid2idx:
                            '''merge nodes'''
                            predictions = node_cls[seg_oid2idx[oid]
                                                   ]  # get all predictions on that instance
                            node_cls_pred = torch.softmax(predictions, dim=1).mean(
                                dim=0)  # averaging the probability
                            # assign and ignor
                            merged_node_cls[idx,
                                            inst_valid_node_cls_indices] = node_cls_pred[seg_valid_node_cls_indices]

                        else:
                            assert noneidx_node_cls is not None
                            merged_node_cls[idx, noneidx_node_cls] = 1.0
                            # merged_node_cls_gt[mask_new]=noneidx_node_cls

                            pass
                    else:
                        if oid not in seg_oid2idx:
                            continue
                        merged_idx2oid[counter] = oid
                        merged_oid2idx[oid] = counter
                        predictions = node_cls[seg_oid2idx[oid]]
                        node_cls_pred = torch.softmax(
                            predictions, dim=1).mean(dim=0)
                        merged_node_cls[counter,
                                        inst_valid_node_cls_indices] = node_cls_pred[seg_valid_node_cls_indices]
                        merged_node_cls_gt[counter] = inst_gt_cls[idx]
                        counter += 1
                if ignore_missing:
                    merged_node_cls = merged_node_cls[:counter]
                    merged_node_cls_gt = merged_node_cls_gt[:counter]

                '''merge batched dict to one single dict'''
                # mask2seg= merge_batch_mask2inst(mask2seg)
                # inst_mask2inst=merge_batch_mask2inst(inst_mask2instance)

                ''' build search list for inst GT edge pairs'''
                inst_gt_pairs = set()

                # For instance level
                # This collects "from" and "to" instances pair as key  -> predicate label
                inst_gt_rel_dict = dict()
                for idx in range(len(inst_gt_rel)):
                    src_idx, tgt_idx = inst_node_edges[0][idx], inst_node_edges[1][idx]
                    src_oid, tgt_oid = inst_oids[src_idx], inst_oids[tgt_idx]
                    inst_gt_pairs.add((src_oid, tgt_oid))
                    inst_gt_rel_dict[(src_oid, tgt_oid)] = inst_gt_rel[idx]
                # For segment level
                # map edge predictions on the same pair of instances.
                merged_edge_cls_dict = defaultdict(list)
                for idx in range(len(seg_gt_rel)):
                    src_idx, tgt_idx = seg_node_edges[0][idx], seg_node_edges[1][idx]
                    src_oid, tgt_oid = seg_oids[src_idx], seg_oids[tgt_idx]
                    pair = (src_oid, tgt_oid)
                    if pair in inst_gt_pairs:
                        merged_edge_cls_dict[pair].append(edge_cls[idx])
                    else:
                        # print('cannot find seg:{}(inst:{}) to seg:{}(inst:{}) with relationship:{}.'.format(src_seg_idx,src_inst_idx,tgt_seg_idx,tgt_inst_idx,relname))
                        pass

                '''merge predictions'''
                merged_edge_cls = torch.zeros(
                    len(inst_gt_rel), len(edge_cls_names)).to(self.cfg.DEVICE)
                if not self.cfg.model.multi_rel:
                    merged_edge_cls_gt = (torch.ones(
                        len(inst_gt_rel), dtype=torch.long) * noneidx_edge_cls).to(self.cfg.DEVICE)
                else:
                    merged_edge_cls_gt = inst_gt_rel.clone().float()
                merged_node_edges = list()  # new edge_indices
                counter = 0
                for idx, (pair, inst_edge_cls) in enumerate(inst_gt_rel_dict.items()):
                    if ignore_missing:
                        if pair[0] not in merged_oid2idx:
                            continue
                        if pair[1] not in merged_oid2idx:
                            continue
                    # merge edge index to the new mask ids
                    src_idx = merged_oid2idx[pair[0]]
                    tgt_idx = merged_oid2idx[pair[1]]
                    merged_node_edges.append([src_idx, tgt_idx])

                    if pair in merged_edge_cls_dict:
                        edge_pds = torch.stack(merged_edge_cls_dict[pair])
                        # ignore same part
                        edge_pds = edge_pds[:, seg_valid_edge_cls_indices]

                        if not self.cfg.model.multi_rel:
                            edge_pds = torch.softmax(edge_pds, dim=1).mean(0)
                        else:
                            edge_pds = torch.sigmoid(edge_pds).mean(0)
                        merged_edge_cls[counter,
                                        inst_valid_edge_cls_indices] = edge_pds
                    elif not self.cfg.model.multi_rel:
                        merged_edge_cls[counter, noneidx_edge_cls] = 1.0

                    if not self.cfg.model.multi_rel:
                        merged_edge_cls_gt[counter] = inst_edge_cls
                    counter += 1
                if ignore_missing:
                    merged_edge_cls = merged_edge_cls[:counter]
                    merged_edge_cls_gt = merged_edge_cls_gt[:counter]
                merged_node_edges = torch.tensor(
                    merged_node_edges, dtype=torch.long)

            merged_node_edges = merged_node_edges.t().contiguous()

            data_inst['node'].pd = merged_node_cls.detach()
            data_inst['node'].y = merged_node_cls_gt.detach()
            data_inst['node', 'to', 'node'].pd = merged_edge_cls.detach()
            data_inst['node', 'to', 'node'].y = merged_edge_cls_gt.detach()
            data_inst['node', 'to', 'node'].edge_index = merged_node_edges
            data_inst['node'].clsIdx = torch.from_numpy(
                np.array([k for k in merged_idx2oid.values()]))
            for eval_tool in eval_tools.values():
                eval_tool.add(data_inst)

        eval_dict = dict()
        eval_dict['visualization'] = dict()
        for eval_type, eval_tool in eval_tools.items():

            obj_, edge_ = eval_tool.get_mean_metrics()
            for k, v in obj_.items():
                # print(k)
                eval_dict[eval_type+'_'+k+'_node_cls'] = v
            for k, v in edge_.items():
                # print(k)
                eval_dict[eval_type+'_'+k+'_edge_cls'] = v

            for k, v in eval_list.items():
                eval_dict[eval_type+'_'+k] = v.avg

            vis = self.visualize(eval_tool=eval_tool)

            vis = {eval_type+'_'+k: v for k, v in vis.items()}

            eval_dict['visualization'].update(vis)

        return eval_dict, eval_tools, eval_UpperBound.eval_tool
