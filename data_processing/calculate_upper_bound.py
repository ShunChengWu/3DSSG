'''
This script estimates the upperbound of a subset of dataset compared to its full GT dataset.
The generated data may have missing instances which leads to imperfect GT. 
'''
import os,torch,codeLib,ssg
from tqdm import tqdm
from ssg import config,define
# from ssg.utils.util_eva import EvalSceneGraph



def calculate(args:codeLib.Config, topK:int=10):
    '''
    return upperbound of triplet Recall, object recalls and predicate recalls
    '''
    ''' Get segment dataset '''
    is_eval_image = args.model.method in ['imp']
    
    dataset_seg  = config.get_dataset(cfg,'test')
    ''' Get instance dataset'''
    dataset_inst = config.get_dataset_inst(cfg,'test')
    
    (scanid2idx_seg, scanid2idx_inst, node_cls_names, edge_cls_names,noneidx_node_cls,noneidx_edge_cls,
            seg_valid_node_cls_indices,inst_valid_node_cls_indices,
            seg_valid_edge_cls_indices,inst_valid_edge_cls_indices) = \
        ssg.utils.util_data.match_class_info_from_two(dataset_seg,dataset_inst,multi_rel=args.model.multi_rel)
        
    eval_UB = ssg.utils.util_eva.EvalUpperBound(node_cls_names,edge_cls_names,
                                                noneidx_node_cls,noneidx_edge_cls,
                                                multi_rel=args.model.multi_rel,topK=topK,none_name=define.NAME_NONE)
    
    for idx in tqdm(range(len(dataset_inst))):
        data_inst = dataset_inst.__getitem__(idx)                
        scan_id_inst = data_inst['scan_id']
        
        index_seg = scanid2idx_seg[scan_id_inst]
        data_seg  = dataset_seg.__getitem__(index_seg)
        
        
        eval_UB(data_seg,data_inst,is_eval_image)
    return eval_UB.eval_tool

if __name__ == '__main__':
    cfg = ssg.Parse()
    eval_tool = calculate(cfg)
    
    
    out_dir = os.path.join(cfg['training']['out_dir'], cfg.name)
    
    prefix='upper_bound'
    print(eval_tool.gen_text())
    _ = eval_tool.write(out_dir, prefix)