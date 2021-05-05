if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import os,math,json,pathlib,torch
import numpy as np


from utils.plot_confusion_matrix import plot_confusion_matrix

def get_metrics(label_id, confusion, VALID_CLASS_IDS:list=None):
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
        fp = np.longlong(confusion[:, label_id].sum())
    
    denom = (tp + fp + fn)
    
    iou = float('nan') if denom == 0 else (float(tp) / denom, tp, denom)
    precision = float('nan') if (tp+fp)==0 else (float(tp) / (tp+fp), tp,tp+fp)
    recall = float('nan') if (tp+fn)==0 else (float(tp) / (tp+fn),tp,tp+fn)
    return iou,precision,recall

def evaluate_topk_object(objs_target, objs_pred, k=-1):
    top_k=list()
    objs_pred = np.exp(objs_pred.detach().cpu())
    size_o = len(objs_pred)
    for obj in range(size_o):
        obj_pred = objs_pred[obj]
        sorted_conf, sorted_args = torch.sort(obj_pred, descending=True) # 1D
        if k<1:
            maxk=len(sorted_conf)
            maxk=min(len(sorted_conf),maxk)
        else:
            maxk=k
        sorted_conf=sorted_conf[:maxk]
        sorted_args=sorted_args[:maxk]
        
        gt = objs_target[obj]
        index = sorted(torch.where(sorted_conf == obj_pred[gt])[0])[0].item()+1
        top_k.append(index)

    return top_k


def evaluate_topk_predicate(gt_edges, rels_pred, multi_rel_outputs, threshold=0.5,k=-1):
    '''
    Find the distance of the predicted probability to the target probability. 
    Sorted the propagated predictions. The index of  the predicted probability is the distance.
    '''
    top_k=list()
    if multi_rel_outputs:
        rels_pred = rels_pred.detach().cpu() # from sigmoid
    else:
        rels_pred = np.exp(rels_pred.detach().cpu()) # log_softmax -> softmax
    size_p = len(rels_pred)

    for rel in range(size_p):
        rel_pred = rels_pred[rel]
        sorted_conf, sorted_args = torch.sort(rel_pred, descending=True)  # 1D
        if k<1:
            maxk=len(sorted_conf)
            maxk=min(len(sorted_conf),maxk)
        else:
            maxk=k
        sorted_conf=sorted_conf[:maxk]
        sorted_args=sorted_args[:maxk]

        temp_topk = []
        rels_target = gt_edges[rel][2]
        
        if len(rels_target) == 0:# Ground truth is None
            indices = torch.where(sorted_conf < threshold)[0]
            if len(indices) == 0:
                index = len(sorted_conf)+1
            else:
                index = sorted(indices)[0].item()+1
            temp_topk.append(index)
        for gt in rels_target:
            indices = torch.where(sorted_conf == rel_pred[gt])[0]
            if len(indices) == 0:
                index = len(sorted_conf)+1
            else:
                index = sorted(indices)[0].item()+1
            temp_topk.append(index)
        temp_topk = sorted(temp_topk)  # ascending I hope/think
        top_k += temp_topk
    return top_k

def get_gt(objs_target, rels_target, edges, instance2mask,multi_rel_outputs):
    gt_edges = [] # initialize
    idx2instance = torch.zeros_like(objs_target)
    for key, value in instance2mask.items():
        if value > 0:
            idx2instance[value - 1] = key

    for edge_index in range(len(edges)):
        idx_eo = edges[edge_index][0].cpu().numpy().item()
        idx_os = edges[edge_index][1].cpu().numpy().item()

        target_eo = objs_target[idx_eo].cpu().numpy().item()
        target_os = objs_target[idx_os].cpu().numpy().item()
        target_rel = [] # there might be multiple
        if multi_rel_outputs:
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

def evaluate_topk(gt_rel, objs_pred, rels_pred, edges, multi_rel_outputs, threshold=0.5, k=40):
    top_k=list()
    objs_pred = np.exp(objs_pred.detach().cpu())
    if multi_rel_outputs:
        # sigmoids
        rels_pred = rels_pred.detach().cpu()
    else:
        # log_softmax
        rels_pred = np.exp(rels_pred.detach().cpu())

    for edge in range(len(edges)):

        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        objs_pred_1 = objs_pred[edge_from]
        objs_pred_2 = objs_pred[edge_to]
        node_score = torch.einsum('n,m->nm',objs_pred_1,objs_pred_2)
        conf_matrix = torch.einsum('nl,m->nlm',node_score,rel_predictions)
        conf_matrix_1d = conf_matrix.reshape(-1)
        sorted_conf_matrix, sorted_args_1d = torch.sort(conf_matrix_1d, descending=True) # 1D
        if k<1:
            maxk=len(sorted_conf_matrix)
            maxk=min(len(sorted_conf_matrix),maxk)
        else:
            maxk=k
        sorted_conf_matrix=sorted_conf_matrix[:maxk]
        sorted_args_1d=sorted_args_1d[:maxk]
        e = gt_rel[edge]
        gt_s = e[0]
        gt_t = e[1]
        gt_r = e[2]
        temp_topk = []
        
        if len(gt_r) == 0:
            # Ground truth is None
            indices = torch.where(sorted_conf_matrix < threshold)[0]
            if len(indices) == 0:
                index = maxk+1
            else:
                index = sorted(indices)[0].item()+1
            temp_topk.append(index)
        for predicate in gt_r: # for the multi rel case
            gt_conf = conf_matrix[gt_s, gt_t, predicate]
            indices = torch.where(sorted_conf_matrix == gt_conf)[0]
            if len(indices) == 0:
                index = maxk+1
            else:
                index = sorted(indices)[0].item()+1
            temp_topk.append(index)
        temp_topk = sorted(temp_topk)
        top_k += temp_topk
    return top_k

def evaluate_topk_recall(
        top_k, top_k_obj, top_k_predicate, 
        multi_rel_outputs,
        objs_pred:torch.tensor, objs_target:torch.tensor,
        rels_pred:torch.tensor, rels_target:torch.tensor, 
        edges, instance2mask):
    
    gt_edges = get_gt(objs_target, rels_target, edges, instance2mask,multi_rel_outputs)
    top_k += evaluate_topk(gt_edges, objs_pred, rels_pred, edges, multi_rel_outputs) # class_labels, relationships_dict)
    top_k_obj += evaluate_topk_object(objs_target, objs_pred)
    top_k_predicate += evaluate_topk_predicate(gt_edges, rels_pred,multi_rel_outputs)
    return top_k, top_k_obj, top_k_predicate

def get_mean_metric(confusion:np.array, VALID_CLASS_IDS:list, CLASS_LABELS:list):
    ious=dict()
    precisions=dict()
    recalls=dict()
    
    if len(VALID_CLASS_IDS) == 0:
        VALID_CLASS_IDS = [i for i in range(len(CLASS_LABELS))]
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        label_id = VALID_CLASS_IDS[i]
        ious[label_name], precisions[label_name], recalls[label_name] \
            = get_metrics(label_id, confusion,VALID_CLASS_IDS)
    def cal_mean(values):
        sum = 0
        counter=0
        for i in range(len(VALID_CLASS_IDS)):
            label_name = CLASS_LABELS[i]
            if isinstance(values[label_name],tuple):
                sum += values[label_name][0]
                counter += 1
        sum /= (counter+1e-12)
        return sum
    return cal_mean(ious),cal_mean(precisions),cal_mean(recalls)

def write_result_file(confusion:np.array, 
                      filename:str,
                      VALID_CLASS_IDS:list, CLASS_LABELS:list):
    ious=dict()
    precisions=dict()
    recalls=dict()
    
    if len(VALID_CLASS_IDS) == 0:
        VALID_CLASS_IDS = [i for i in range(len(CLASS_LABELS))]
    for i in range(len(VALID_CLASS_IDS)):
        label_name = CLASS_LABELS[i]
        label_id = VALID_CLASS_IDS[i]
        ious[label_name], precisions[label_name], recalls[label_name] \
            = get_metrics(label_id, confusion,VALID_CLASS_IDS)

    with open(filename, 'w') as f:
        def write_metric(name, values):
            f.write('{} scores\n'.format(name))
            sum = 0
            counter=0
            for i in range(len(VALID_CLASS_IDS)):
                label_id = VALID_CLASS_IDS[i]
                label_name = CLASS_LABELS[i]
                if isinstance(values[label_name],tuple):
                    # value = values[label_name][0]
                    f.write('{0:<14s}({1:<2d}): {2:>5.3f}   ({3:>6d}/{4:<6d})\n'.format(label_name, label_id, values[label_name][0], values[label_name][1], values[label_name][2]))
                    sum += values[label_name][0]
                    counter += 1
                else:
                    f.write('{0:<14s}({1:<2d}): nan\n'.format(label_name, label_id))
            sum /= (counter+1e-12)
            f.write('{0:<18s}: {1:>5.3f}\n'.format('Average', sum))
            
            for i in range(len(VALID_CLASS_IDS)):
                if i > 0:
                    f.write(' & ')
                label_id = VALID_CLASS_IDS[i]
                label_name = CLASS_LABELS[i]
                if isinstance(values[label_name],tuple):
                    value = values[label_name][0]
                    f.write('{:>5.3f}'.format(value))
                    
                else:
                    f.write('nan')
            
            f.write(' & {:>5.3f}\n'.format(sum))
            return sum
        mean_iou = write_metric("IoU", ious)
        mean_pre = write_metric("Precision", precisions)
        mean_rec = write_metric("Recall", recalls)
        f.write('{0:<14s}: {1:>5.3f}   ({2:>6f}/{3:<6f})\n\n'.format('accuracy', \
                                                                  confusion.trace()/confusion.sum(),  \
                                                                  confusion.trace(), \
                                                                  confusion.sum()) )
        
        f.write('\nconfusion matrix\n')
        f.write('\t\t\t')
        for i in range(len(VALID_CLASS_IDS)):
            #f.write('\t{0:<14s}({1:<2d})'.format(CLASS_LABELS[i], VALID_CLASS_IDS[i]))
            f.write('{0:<8d}'.format(VALID_CLASS_IDS[i]))
        f.write('\n')
        for r in range(len(VALID_CLASS_IDS)):
            f.write('{0:<14s}({1:<2d})'.format(CLASS_LABELS[r], VALID_CLASS_IDS[r]))
            for c in range(len(VALID_CLASS_IDS)):
                f.write('\t{0:>5.3f}'.format(confusion[VALID_CLASS_IDS[r],VALID_CLASS_IDS[c]]))
            f.write('\n')
    print ('wrote results to', filename)
    return [mean_iou, mean_pre,mean_rec]
    
def build_seg2name(pds:torch.tensor, idx2seg, names):
    '''
    pds: [n]
    '''
    s2n = dict()
    for n in range(len(pds)):
        s2n[str(idx2seg[n])] = names[pds[n]]  
    return s2n
def build_edge2name(pds:torch.tensor,edges:torch.tensor,
                    idx2seg:dict,names:list, none_name = "UN"):
    if edges.shape[0] == 2:
        edges = edges.t()
    
    s2n=dict()
    if pds.ndim == 2: # has multiple prediction
        for n in range(pds.shape[0]):
            n_i = idx2seg[edges[n][0].item()]
            n_j = idx2seg[edges[n][1].item()]
            edge = str(n_i)+'_'+str(n_j)
            for c in range(pds.shape[1]):
                if edge not in s2n:
                    s2n[edge]=list()
                if pds[n][c]>0:
                    s2n[edge].append( names[c] )
    else: 
        for n in range(pds.shape[0]):
            n_i = idx2seg[edges[n][0].item()]
            n_j = idx2seg[edges[n][1].item()]
            edge = str(n_i)+'_'+str(n_j)
            s2n[edge] = names[pds[n]]
            
    return s2n

def build_edge2name_value(values:torch.tensor,edges:torch.tensor, idx2gtcls:dict,names:list):
    if edges.shape[0] == 2:
        edges = edges.t()
    nn2v=dict()
    for n in range(edges.shape[0]):
        n_i = idx2gtcls[edges[n][0].item()]
        n_j = idx2gtcls[edges[n][1].item()]
        if names[n_i] not in nn2v:
            nn2v[names[n_i]]=dict()
        if names[n_j] not in nn2v[names[n_i]]:
            nn2v[names[n_i]][names[n_j]]=list()
        nn2v[names[n_i]][names[n_j]].append(values[n].item())
    return nn2v

class EvaPairWeight():
    def __init__(self,class_names:list):
        self.class_names = class_names
        self.c_mat = np.zeros([len(self.class_names),len(self.class_names)], dtype=np.float)
    def update(self, values, edge_indices, idx2gtcls):
        nn2vs = build_edge2name_value(values, edge_indices, idx2gtcls, self.class_names)
        for name1, n2vs in nn2vs.items():
            for name2, vs in n2vs.items():
                if len(vs) == 0: continue
                a_vs = np.array(vs).mean()
                idx1 = self.class_names.index(name1)
                idx2 = self.class_names.index(name2)
                self.c_mat[idx1][idx2] += a_vs
    def reset(self):
        self.c_mat = np.zeros([len(self.class_names),len(self.class_names)], dtype=np.float)
        
class EvaClassification():
    def __init__(self,class_names:list, none_name:str = 'UN'):
        self.none_name = none_name
        self.unknown = len(class_names)
        self.class_names = class_names + [none_name]
        self.c_mat = np.zeros([len(self.class_names),len(self.class_names)], dtype=np.float)
    def update(self, pd_indices:dict, gt_indices:dict, gt_only=False, pd_only=False):
        union_indices = set(pd_indices.keys()).union(gt_indices.keys())
        multi_pred = True
        for k,v in pd_indices.items():
            multi_pred = isinstance(v, list)
            break
            
        for idx in union_indices: # since it's union, there is no way to get both unknown.
            if not multi_pred:
                pd = self.unknown if idx not in pd_indices else pd_indices[idx]
                pd_idx= self.class_names.index(pd) if pd in self.class_names else self.unknown
                gt = self.unknown if idx not in gt_indices else gt_indices[idx]
                gt_idx = self.class_names.index(gt) if gt in self.class_names else self.unknown
                self.c_mat[gt_idx][pd_idx] += 1
            else:
                def get_indices(indices) -> list:
                    if idx not in indices:
                        idxes = [self.unknown]
                    else:
                        assert isinstance(indices[idx], list)
                        idxes = [self.class_names.index(i) for i in indices[idx]]
                    return idxes
                pd_indices_set = set(get_indices(pd_indices)).difference([self.unknown])
                gt_indices_set = set(get_indices(gt_indices)).difference([self.unknown])
                
                if len(gt_indices_set)==0 and len(pd_indices_set) == 0: 
                    self.c_mat[self.unknown][self.unknown] += 1
                # for every unmatched prediction, generate confusion to the other unmatched, otherwise treat them as unknown
                if len(gt_indices_set) > 0:
                    intersection = set(pd_indices_set).intersection(gt_indices_set) # match prediction
                    diff_gt = set(gt_indices_set).difference(intersection) # unmatched gt
                    diff_pd = set(pd_indices_set).difference(intersection) # unmatched pd
                    for i in intersection:
                        self.c_mat[i][i] += 1
                    if not gt_only:
                        for pd_idx in diff_pd:
                            if len(diff_gt) > 0:
                                for gt_idx in diff_gt:
                                    self.c_mat[gt_idx][pd_idx] += 1/len(diff_gt)
                            else:
                                self.c_mat[self.unknown][pd_idx] += 1
                    if not pd_only:
                        for gt_idx in diff_gt:
                            if len(diff_pd) > 0:
                                for pd_idx in diff_pd:
                                    self.c_mat[gt_idx][pd_idx] += 1/len(diff_pd)
                            else:
                                self.c_mat[gt_idx][self.unknown] += 1
                elif len(gt_indices_set) == 0:
                    for idx in pd_indices_set:
                        self.c_mat[self.unknown][idx] += 1
                
                
                
    def get_recall(self):
        return self.c_mat.c_cmat.diagonal().sum() / self.c_mat.sum()
    def get_mean_metrics(self):
        return get_mean_metric(self.c_mat, [], self.class_names)
    def reset(self):
        self.c_mat = np.zeros([len(self.class_names),len(self.class_names)], dtype=np.float)
    def draw(self, title='Confusion matrix'):
        return plot_confusion_matrix(self.c_mat, 
                          target_names=self.class_names, 
                          title=title,
                          plot_text=False,)
    
class EvalSceneGraph():
    def __init__(self, obj_class_names:list, rel_class_names:list, multi_rel_outputs:float=0.5, k=100, multi_rel_prediction:bool=True):
        # params
        self.obj_class_names=obj_class_names
        self.rel_class_names=rel_class_names
        self.multi_rel_outputs=multi_rel_outputs
        self.multi_rel_prediction=multi_rel_prediction
        self.k=k
        # containers
        self.eva_o_cls = EvaClassification(obj_class_names)
        self.eva_r_cls = EvaClassification(rel_class_names)
        self.predictions=dict()
        self.top_k_triplet=list()
        self.top_k_obj=list()
        self.top_k_rel=list()
    def reset(self):
        self.eva_o_cls.reset()
        self.eva_r_cls.reset()
        self.top_k_triplet=list()
        self.top_k_obj=list()
        self.top_k_rel=list()
        self.predictions=dict()
        
    def add(self,scan_id, obj_pds, obj_gts, rel_pds,rel_gts, seg2idx:dict, edge_indices):
        '''
        obj_pds: [n, n_cls]: log_softmax
        obj_gts: [n, 1]: long tensor
        rel_pds: [m,n_cls]: torch.sigmoid(x) if multi_rel_outputs>0 else log_softmax
        rel_gts: [m,n_cls] if multi_rel_outputs>0 else [m,1]
        '''
        obj_pds=obj_pds.detach()
        if rel_pds is not None:
            rel_pds=rel_pds.detach()
        o_pd = obj_pds.max(1)[1]
        # correct_array = o_pd.eq(obj_gts.data).cpu()
        
        idx2seg=dict()
        for key,item in seg2idx.items():
            if isinstance(item, torch.Tensor):
                idx2seg[item.item()-1] = key
            else:
                idx2seg[item-1] = key
        pd=dict()
        gt=dict()
        pd['nodes'] = build_seg2name(o_pd,idx2seg,self.obj_class_names)
        gt['nodes'] = build_seg2name(obj_gts,idx2seg,self.obj_class_names)
        self.eva_o_cls.update(pd['nodes'], gt['nodes'], False)
        
        if rel_pds is not None and rel_pds.shape[0] > 0:
            if self.multi_rel_prediction:
                assert self.multi_rel_outputs>0
                r_pd = rel_pds > self.multi_rel_outputs
            else:
                r_pd = rel_pds.max(1)[1]
                
            pd['edges'] = build_edge2name(r_pd, edge_indices, idx2seg, self.rel_class_names)
            gt['edges'] = build_edge2name(rel_gts, edge_indices, idx2seg, self.rel_class_names)
            self.eva_r_cls.update(pd['edges'],gt['edges'],False)
        
        self.predictions[scan_id]=dict()
        self.predictions[scan_id]['pd']=pd
        self.predictions[scan_id]['gt']=gt
        
        if self.k>0:
            # top_k_predicate, top_k_obj = [], []
            self.top_k_obj += evaluate_topk_object(obj_gts, obj_pds)
            
            if rel_pds is not None:
                gt_edges = get_gt(obj_gts, rel_gts, edge_indices, seg2idx, self.multi_rel_prediction)
                self.top_k_rel += evaluate_topk_predicate(gt_edges, rel_pds, 
                                                          multi_rel_outputs = self.multi_rel_prediction, 
                                                          threshold=self.multi_rel_outputs, k = self.k)
                
                self.top_k_triplet += evaluate_topk(gt_edges, obj_pds, rel_pds, edge_indices, 
                                      multi_rel_outputs=self.multi_rel_prediction,
                                      threshold=self.multi_rel_outputs, k=self.k) # class_labels, relationships_dict)
    def get_recall(self):
        return self.eva_o_cls.get_recall(), self.eva_r_cls.get_recall()
    
    def get_mean_metrics(self):
        return self.eva_o_cls.get_mean_metrics(),self.eva_r_cls.get_mean_metrics()
        
    def gen_text(self):
        c_cmat = self.eva_o_cls.c_mat
        r_cmat = self.eva_r_cls.c_mat
        
        c_TP = c_cmat.diagonal().sum()
        c_P  = c_cmat.sum(axis=0).sum()
        
        r_TP = r_cmat.diagonal().sum()
        r_P  = r_cmat.sum(axis=0).sum()
        txt = "recall obj cls {}".format(c_TP / float(c_P)) +'\n'
        txt += "recall rel cls {}".format(r_TP / float(r_P)) +'\n'
        if  self.k>0:
            # print("Recall@k for relationship triplets: ")
            txt += "Recall@k for relationship triplets: "+'\n'
            ntop_k = np.asarray(self.top_k_triplet)
            ks = set([1,2,3,5,10,50,100])
            for i in [0,0.05,0.1,0.2,0.5,0.9]:
                ks.add( int(math.ceil(self.k*i+1e-9)) )
            for k in sorted(ks):
                R = (ntop_k <= k).sum() / len(ntop_k)
                # print("top-k R@" + str(k), "\t", R)
                txt += "top-k R@{}\t {}".format(k,R)+'\n'
            # print(len(self.top_k_triplet))
            txt += str(len(self.top_k_triplet)) +'\n'

            # print("Recall@k for objects: ")
            txt+='Recall@k for objects: \n'
            ntop_k_obj = np.asarray(self.top_k_obj)
            ks = set([1,2,3,4,5,10,50,100])
            for i in [0,0.05,0.1,0.2,0.5]:
                ks.add( int(math.ceil(len(self.obj_class_names)*i+1e-9)) )
            for k in sorted(ks):
                R = (ntop_k_obj <= k).sum() / len(ntop_k_obj)
                # print("top-k R@" + str(k), "\t", R)
                txt += "top-k R@{}\t {}".format(k,R)+'\n'
            # print(len(self.top_k_obj))
            txt += str(len(self.top_k_obj)) +'\n'

            # print("Recall@k for predicates: ")
            txt += "Recall@k for predicates: \n"
            ntop_k_predicate = np.asarray(self.top_k_rel)
            ks = set([1,2,3,4,5,10])
            for i in [0,0.05,0.1,0.2,0.5]:
                ks.add( int(math.ceil(len(self.rel_class_names)*i +1e-9)) )
            for k in sorted(ks):
                R = (ntop_k_predicate <= k).sum() / len(ntop_k_predicate)
                # print("top-k R@" + str(k), "\t", R)
                txt += "top-k R@{}\t {}".format(k,R)+'\n'
            # print(len(self.top_k_rel))
            txt += str(len(self.top_k_rel)) +'\n'
        return txt

    def write(self, path, model_name):
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        
        
        with open(os.path.join(path,'predictions.json'),'w') as f:
            json.dump(self.predictions,f, indent=4)   
    
        obj_results = write_result_file(self.eva_o_cls.c_mat, os.path.join(path,model_name+'_results_obj.txt'), [], self.eva_o_cls.class_names)
        rel_results = write_result_file(self.eva_r_cls.c_mat, os.path.join(path,model_name+'_results_rel.txt'), [], self.eva_r_cls.class_names)
        
        r_o = {k: v for v, k in zip(obj_results, ['Obj_IOU','Obj_Precision', 'Obj_Recall']) }
        r_r = {k: v for v, k in zip(rel_results, ['Rel_IOU','Rel_Precision', 'Rel_Recall']) }
        results = {**r_o, **r_r}
        
        plot_confusion_matrix(self.eva_o_cls.c_mat,
                          target_names=self.eva_o_cls.class_names,
                          title='object confusion matrix',
                          normalize=True,
                          plot_text=False,
                          plot=False,
                          pth_out=os.path.join(path, model_name + "_obj_cmat.png"))
        plot_confusion_matrix(self.eva_r_cls.c_mat,
                          target_names=self.eva_r_cls.class_names,
                          title='predicate confusion matrix',
                          normalize=True,
                          plot_text=False,
                          plot=False,
                          pth_out=os.path.join(path, model_name + "_rel_cmat.png"))
       
        # if  opt.eval_topk:
        with open(os.path.join(path, model_name + '_topk.txt'),'w+') as f:
            f.write(self.gen_text())
        return results
            
if __name__ == '__main__':
    tt = EvaClassification(['1','2'], [0,1])
    pd=dict()
    gt=dict()
    pd[0]='1'
    pd[1]='1'
    gt[0]='1'
    gt[1]='2'
    tt.update(pd,gt,False)
    