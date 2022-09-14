from collections import defaultdict
import torch

from tqdm import tqdm
from codeLib.utils import moving_average
from ssg import define
from ssg.utils.util_data import match_class_info_from_two, merge_batch_mask2inst
from ssg.utils.util_eva import EvalSceneGraph, EvalUpperBound
import codeLib.utils.string_numpy as snp

class EvalInst(object):
    def __init__(self):
        pass
    def evaluate_inst(self,dataset_seg,dataset_inst,topk):
        ignore_missing=self.cfg.eval.ignore_missing
        
        '''add a none class for missing instances'''
        (scanid2idx_seg, node_cls_names, edge_cls_names,noneidx_node_cls,noneidx_edge_cls,
            seg_valid_node_cls_indices,inst_valid_node_cls_indices,
            seg_valid_edge_cls_indices,inst_valid_edge_cls_indices) = \
            match_class_info_from_two(dataset_seg,dataset_inst, multi_rel=self.cfg.model.multi_rel)

        eval_tool = EvalSceneGraph(node_cls_names, edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=topk,save_prediction=True,
                                   none_name=define.NAME_NONE) 
        # eval_upper_bound
        eval_UpperBound = EvalUpperBound(node_cls_names,edge_cls_names,noneidx_node_cls,noneidx_edge_cls,
                                         multi_rel=self.cfg.model.multi_rel,topK=topk,none_name=define.NAME_NONE)
        
        eval_list = defaultdict(moving_average.MA)
        
        '''check'''
        #length
        # print('len(dataset_seg), len(dataset_inst):',len(dataset_seg), len(dataset_inst))
        # print('ignore missing',ignore_missing)
        #classes
        
        
        ''' get scan_idx mapping '''
        scanid2idx_seg = dict()
        for index in range(len(dataset_seg)):
            scan_id = snp.unpack(dataset_seg.scans,index)# self.scans[idx]
            scanid2idx_seg[scan_id] = index
            
        scanid2idx_inst = dict()
        for index in range(len(dataset_inst)):
            scan_id = snp.unpack(dataset_inst.scans,index)# self.scans[idx]
            scanid2idx_inst[scan_id] = index

        '''start eval'''
        self.model.eval()
        for index in tqdm(range(len(dataset_inst))):
            # if index>10: break
        # for data_inst in seg_dataloader:
            data_inst = dataset_inst.__getitem__(index)                
            scan_id_inst = data_inst['scan_id']
            # print('scan_id_inst',scan_id_inst)
            
            if scan_id_inst not in scanid2idx_seg:
                #TODO: what should we do if missing scans?
                raise RuntimeError('')
                continue
            
            index_seg = scanid2idx_seg[scan_id_inst]
            data_seg  = dataset_seg.__getitem__(index_seg)
            
            assert data_seg['scan_id'] == data_inst['scan_id']
            
            '''process seg'''
            eval_dict={}
            with torch.no_grad():
                logs = {}
        
                # Process data dictionary
                data = self.process_data_dict(data_seg)
                data_inst = self.process_data_dict(data_inst)
                
                
                # record in eval_UB
                eval_UpperBound(data,data_inst)
                
                # Shortcuts
                scan_id = data['scan_id']
                # gt_cls = data['gt_cls']
                gt_rel = data['gt_rel']
                mask2seg = data['mask2instance']
                node_edges_ori = data['node_edges']
                data['node_edges'] = data['node_edges'].t().contiguous()
                seg2inst = data.get('seg2inst',None)
                
                inst_mask2instance = data_inst['mask2instance']
                inst_gt_cls = data_inst['gt_cls']
                inst_gt_rel = data_inst['gt_rel']
                data_inst['node_edges'] = data_inst['node_edges'].t().contiguous()
                
                # print(scan_id, inst_gt_cls.shape, inst_gt_rel.shape)
                # continue
                # import sys
                # sys.exit()
                
                # check input valid
                if node_edges_ori.ndim==1: 
                    raise RuntimeError('')
                    continue
                
                ''' make forward pass through the network '''
                node_cls, edge_cls = self.model(**data)
                
                '''merge prediction from seg to instance (in case of "same part")'''
                inst2masks = defaultdict(list)                
                for mask, seg in mask2seg.items():
                    if seg2inst is not None:
                        inst = seg2inst[seg]
                    else:
                        inst = seg
                    inst2masks[inst].append(mask)
                
                '''merge nodes'''
                merged_mask2instance = dict()
                merged_instance2idx = dict()
                merged_node_cls = torch.zeros(len(inst_mask2instance),len(node_cls_names)).to(self.cfg.DEVICE)
                merged_node_cls_gt = (torch.ones(len(inst_mask2instance),dtype=torch.long) * noneidx_node_cls).to(self.cfg.DEVICE)
                counter=0
                for mask_new, (mask_old, inst)in enumerate(inst_mask2instance.items()):                    
                    # merge predictions
                    if not ignore_missing:
                        merged_instance2idx[inst]=mask_new
                        merged_mask2instance[mask_new] = inst
                        merged_node_cls_gt[mask_new] = inst_gt_cls[mask_old] # use GT class
                        if inst in inst2masks:
                            '''merge nodes'''
                            predictions = node_cls[inst2masks[inst]]# get all predictions on that instance
                            node_cls_pred = torch.softmax(predictions, dim=1).mean(dim=0)# averaging the probability
                            merged_node_cls[mask_new,inst_valid_node_cls_indices] = node_cls_pred[seg_valid_node_cls_indices] # assign and ignor
                            
                        else:
                            #TODO: handle missing inst
                            merged_node_cls[mask_new,noneidx_node_cls] = 1.0
                            # merged_node_cls_gt[mask_new]=noneidx_node_cls
                            
                            
                            pass
                    else:
                        if inst not in inst2masks:
                            continue
                        merged_mask2instance[counter] = inst
                        merged_instance2idx[inst]=counter
                        predictions = node_cls[inst2masks[inst]]
                        node_cls_pred = torch.softmax(predictions, dim=1).mean(dim=0)                        
                        merged_node_cls[counter,inst_valid_node_cls_indices] = node_cls_pred[seg_valid_node_cls_indices]
                        merged_node_cls_gt[counter] = inst_gt_cls[mask_old]
                        counter+=1
                if ignore_missing:
                    merged_node_cls = merged_node_cls[:counter]
                    merged_node_cls_gt = merged_node_cls_gt[:counter]
                    
                '''merge batched dict to one single dict'''
                mask2seg= merge_batch_mask2inst(mask2seg) 
                inst_mask2inst=merge_batch_mask2inst(inst_mask2instance)
                
                # build search list for GT edge pairs
                inst_gt_pairs = set() # 
                inst_gt_rel_dict = dict() # This collects "from" and "to" instances pair as key  -> predicate label
                for idx in range(len(inst_gt_rel)):
                    src_idx, tgt_idx = data_inst['node_edges'][0,idx].item(),data_inst['node_edges'][1,idx].item()
                    src_inst_idx, tgt_inst_idx = inst_mask2inst[src_idx], inst_mask2inst[tgt_idx]
                    inst_gt_pairs.add((src_inst_idx, tgt_inst_idx))
                    inst_gt_rel_dict[(src_inst_idx, tgt_inst_idx)] = inst_gt_rel[idx]
               
                merged_edge_cls_dict = defaultdict(list) # map edge predictions on the same pair of instances.
                for idx in range(len(gt_rel)):
                    src_idx, tgt_idx = data['node_edges'][0,idx].item(), data['node_edges'][1,idx].item()
                    # relname = self.edge_cls_names[gt_rel[idx].item()]
                    
                    src_seg_idx = mask2seg[src_idx]
                    src_inst_idx = seg2inst[src_seg_idx] if seg2inst is not None else src_seg_idx
                    
                    tgt_seg_idx = mask2seg[tgt_idx]
                    tgt_inst_idx = seg2inst[tgt_seg_idx] if seg2inst is not None else tgt_seg_idx
                    pair = (src_inst_idx,tgt_inst_idx)
                    if  pair in inst_gt_pairs:
                        merged_edge_cls_dict[pair].append( edge_cls[idx] )
                    else:
                        # print('cannot find seg:{}(inst:{}) to seg:{}(inst:{}) with relationship:{}.'.format(src_seg_idx,src_inst_idx,tgt_seg_idx,tgt_inst_idx,relname))
                        pass
                # check missing rels
                # for pair in inst_gt_rel_dict:
                #     if pair not in merged_edge_cls_dict:
                #         relname = self.edge_cls_names[inst_gt_rel_dict[pair]]
                #         src_name = self.node_cls_names[inst_gt_cls[inst_instance2mask[pair[0]]]]
                #         tgt_name = self.node_cls_names[inst_gt_cls[inst_instance2mask[pair[1]]]]
                #         print('missing inst:{}({}) to inst:{}({}) with relationship:{}'.format(pair[0],src_name,pair[1],tgt_name,relname))
                        
                '''merge predictions'''
                merged_edge_cls = torch.zeros(len(inst_gt_rel),len(edge_cls_names)).to(self.cfg.DEVICE)
                if not self.cfg.model.multi_rel:
                    merged_edge_cls_gt = (torch.ones(len(inst_gt_rel),dtype=torch.long) * noneidx_edge_cls).to(self.cfg.DEVICE)
                else:
                    merged_edge_cls_gt = inst_gt_rel.clone().float()
                merged_node_edges = list() # new edge_indices
                counter=0
                for idx, (pair,inst_edge_cls) in enumerate(inst_gt_rel_dict.items()):
                    if ignore_missing:
                        if pair[0] not in merged_instance2idx: continue
                        if pair[1] not in merged_instance2idx: continue
                    # merge edge index to the new mask ids
                    mask_src = merged_instance2idx[pair[0]]
                    mask_tgt = merged_instance2idx[pair[1]]
                    merged_node_edges.append([mask_src,mask_tgt])
                    
                    if pair in merged_edge_cls_dict:
                        edge_pds = torch.stack(merged_edge_cls_dict[pair])
                        edge_pds = edge_pds[:,seg_valid_edge_cls_indices]
                        # seg_valid_edge_cls_indices
                        
                        #ignore same part
                        if not self.cfg.model.multi_rel:
                            edge_pds = torch.softmax(edge_pds,dim=1).mean(0)
                        else:
                            edge_pds = torch.sigmoid(edge_pds).mean(0)
                        merged_edge_cls[counter,inst_valid_edge_cls_indices] = edge_pds
                    else:
                        merged_edge_cls[counter,noneidx_edge_cls] = 1.0
                        
                    merged_edge_cls_gt[counter] = inst_edge_cls
                    counter+=1
                if ignore_missing:
                    merged_edge_cls=merged_edge_cls[:counter]
                    merged_edge_cls_gt = merged_edge_cls_gt[:counter]
                merged_node_edges = torch.tensor(merged_node_edges,dtype=torch.long)
            
            eval_tool.add([scan_id], 
                          merged_node_cls,
                          merged_node_cls_gt, 
                          merged_edge_cls,
                          merged_edge_cls_gt,
                          [merged_mask2instance],
                          merged_node_edges)
            # break
        eval_dict = dict()
        obj_, edge_ = eval_tool.get_mean_metrics()
        for k,v in obj_.items():
            # print(k)
            eval_dict[k+'_node_cls'] = v
        for k,v in edge_.items():
            # print(k)
            eval_dict[k+'_edge_cls'] = v
        
        for k, v in eval_list.items():
            eval_dict[k] = v.avg
        
        vis = self.visualize(eval_tool=eval_tool)
        eval_dict['visualization'] = vis
        return eval_dict, eval_tool, eval_UpperBound.eval_tool