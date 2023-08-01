import argparse, codeLib
import torch
from ssg import define
import ssg.config as config
from ssg.utils.util_data import match_class_info_from_two
from ssg.utils.util_eva import EvalUpperBound

def process(cfg1,cfg2):
    db_1  = config.get_dataset(cfg1,'test')
    db_2  = config.get_dataset_inst(cfg2,'test')
    
    topk = cfg1.eval.topK
    
    (scanid2idx_seg, node_cls_names, edge_cls_names,noneidx_node_cls,noneidx_edge_cls,
            seg_valid_node_cls_indices,inst_valid_node_cls_indices,
            seg_valid_edge_cls_indices,inst_valid_edge_cls_indices) = \
            match_class_info_from_two(db_1,db_2, multi_rel=cfg1.model.multi_rel)

    # eval_tool = EvalSceneGraph(node_cls_names, edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=topk,save_prediction=True,
                            #    none_name=define.NAME_NONE) 
    # eval_upper_bound
    eval_UpperBound = EvalUpperBound(node_cls_names,edge_cls_names,noneidx_node_cls,noneidx_edge_cls,
                                        multi_rel=cfg1.model.multi_rel,topK=topk,none_name=define.NAME_NONE)
    
    
    for index in range(len(db_2)):
        data_inst = db_2.__getitem__(index)                
        scan_id_inst = data_inst['scan_id']
        # print('scan_id_inst',scan_id_inst)
        
        if scan_id_inst not in scanid2idx_seg:
            #TODO: what should we do if missing scans?
            # raise RuntimeError('')
            # continue
            data_seg = None
        else:
            index_seg = scanid2idx_seg[scan_id_inst]
            data_seg  = db_1.__getitem__(index_seg)
        eval_UpperBound(data_seg,data_inst)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config1')
    parser.add_argument('config2')
    args = parser.parse_args()
    
    cfg1 = codeLib.Config(args.config1)
    cfg2 = codeLib.Config(args.config2)
    # init device
    device = 'cuda' if torch.cuda.is_available() and len(cfg1.GPU) > 0 else 'cpu'
    cfg1.DEVICE=torch.device(device)
    cfg2.DEVICE=torch.device(device)
    
    process(cfg1,cfg2)