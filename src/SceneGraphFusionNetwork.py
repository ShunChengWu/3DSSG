if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
import os,torch,time
from DataLoader import CustomDataLoader
from model_SGFN import SGFNModel
from model_SGPN import SGPNModel
from dataset_builder import build_dataset
from torch.utils.tensorboard import SummaryWriter
from config import Config
import op_utils
from utils import plot_confusion_matrix
from utils import util_eva

class SGFN():
    def __init__(self, config):
        self.config = config
        
        try:
            self.model_name = self.config.NAME
        except:
            self.model_name = 'SceneGraphFusionNetwork'

        self.mconfig = mconfig = config.MODEL
        ''' set dataset to SGFN if handcrafted edge descriptor is used '''
        self.use_edge_descriptor=self.config.dataset.dataset_type == "SGFN"
        
        ''' Build dataset '''
        dataset = None
        if config.MODE  == 'train':
            if config.VERBOSE: print('build train dataset')
            self.dataset_train = build_dataset(self.config,split_type='train_scans', shuffle_objs=True,
                                               multi_rel_outputs=mconfig.multi_rel_outputs,
                                               use_rgb=mconfig.USE_RGB,
                                               use_normal=mconfig.USE_NORMAL)
            self.dataset_train.__getitem__(0)
            self.w_cls_obj=self.w_cls_rel=None
            if self.config.WEIGHTING:
                self.w_cls_obj=self.dataset_train.w_cls_obj
                self.w_cls_rel=self.dataset_train.w_cls_rel
                
        if config.MODE  == 'train' or config.MODE  == 'trace':
            if config.VERBOSE: print('build valid dataset')
            self.dataset_valid = build_dataset(self.config,split_type='validation_scans', shuffle_objs=False, 
                                      multi_rel_outputs=mconfig.multi_rel_outputs,
                                      use_rgb=mconfig.USE_RGB,
                                      use_normal=mconfig.USE_NORMAL)
            num_obj_class = len(self.dataset_valid.classNames)
            num_rel_class = len(self.dataset_valid.relationNames)
            dataset = self.dataset_valid

        try:
            if config.VERBOSE: print('build test dataset')
            self.dataset_eval = build_dataset(self.config,split_type='test_scans', shuffle_objs=False, 
                                      multi_rel_outputs=mconfig.multi_rel_outputs,
                                      use_rgb=mconfig.USE_RGB,
                                      use_normal=mconfig.USE_NORMAL)
            num_obj_class = len(self.dataset_eval.classNames)
            num_rel_class = len(self.dataset_eval.relationNames)
            dataset = self.dataset_eval
        except:
            print('canno build eval dataset.')
            self.dataset_eval = None
            
        ''' Build Model '''
        if self.use_edge_descriptor:
            self.model = SGFNModel(config,self.model_name,num_obj_class, num_rel_class).to(config.DEVICE)
        else:
            # raise NotImplementedError('not yet cleaned.')
            self.model = SGPNModel(config,self.model_name,num_obj_class, num_rel_class).to(config.DEVICE)
        
        self.samples_path = os.path.join(config.PATH, self.model_name, 'samples')
        self.results_path = os.path.join(config.PATH, self.model_name, 'results')
        self.trace_path = os.path.join(config.PATH, self.model_name, 'traced')
        
        if config.MODE == 'train' or config.MODE == 'eval':
            pth_log = os.path.join(config.PATH, "logs", self.model_name)
            self.writter = SummaryWriter(pth_log)
            
            # Plot data graph to tensorboard
            all_logs = op_utils.get_tensorboard_logs(pth_log)
            if len(all_logs) == 1:
                if self.use_edge_descriptor:
                    data = dataset.__getitem__(0)
                    obj_points, edge_indices, gt_class, gt_rels, descriptor = self.data_processing(data[2:])
                    self.writter.add_graph(self.model,[obj_points, edge_indices.t().contiguous(), descriptor])
                else:
                    data = dataset.__getitem__(0)
                    obj_points, rel_points, edge_indices, *_ = self.data_processing(data[2:])
                    self.writter.add_graph(self.model,[obj_points,rel_points,edge_indices.t().contiguous()])
        
    def load(self, best=False):
        return self.model.load(best)
    
    def data_processing(self, items, max_edges=-1):
        if self.use_edge_descriptor:
            with torch.no_grad():
                items = [item.squeeze(0) for item in items]
                items = self.cuda(*items)
            return items
        else:
            with torch.no_grad():
                obj_points, rel_points, gt_class, gt_rels, edge_indices = items
                obj_points = obj_points.squeeze(0)
                rel_points = rel_points.squeeze(0)
                edge_indices = edge_indices.squeeze(0)
                gt_class   = gt_class.squeeze(0).flatten().long()
                gt_rels    = gt_rels.squeeze(0)
                
                obj_points = obj_points.permute(0,2,1)
                rel_points = rel_points.permute(0,2,1)
                obj_points, rel_points, edge_indices, gt_class, gt_rels = \
                    self.cuda(obj_points, rel_points, edge_indices, gt_class, gt_rels)
            return obj_points, rel_points, edge_indices, gt_class, gt_rels
        
  
    def train(self):
        ''' create data loader '''
        drop_last = True
        
        train_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_train,
            batch_size=1,
            num_workers=self.config.WORKERS,
            drop_last=drop_last,
            shuffle=True,
        )
        
        epoch = 1
        keep_training = True
        total = len(self.dataset_train) \
            if drop_last is True else  len(self.dataset_train)
        max_iteration = int(float(self.config.MAX_EPOCHES)*len(self.dataset_train))
                
        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        
        progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
        
        if self.model.iteration >= max_iteration:
            keep_training = False
            print('Read maximum training iteration (',max_iteration,').')
                
        ''' Resume data loader to the last read location '''
        loader = iter(train_loader)
        
        if self.model.iteration > 0:
            print('\n Resuming dataloader to last iteration...')
            import math
            floored = math.floor(self.model.iteration / len(self.dataset_train))
            # remains = self.model.iteration % len(self.dataset_train)
            iteration=floored*len(self.dataset_train)
            epoch+=floored
            while True:                
                iter_local = 0
                for idx in loader.IndexIter():
                    progbar.add(1, silent=True)
                    iter_local += 1
                    iteration += 1
                    # print(iteration , self.model.iteration)
                    if iteration >= self.model.iteration:
                        break
                if iteration >= self.model.iteration:
                    break
                epoch+=1
                progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
                loader = iter(train_loader)
            
        ''' Train '''
        eva_tool = util_eva.EvalSceneGraph(self.dataset_train.classNames, self.dataset_train.relationNames,
                                  multi_rel_outputs=0.5, k=0,
                                  multi_rel_prediction=self.model.mconfig.multi_rel_outputs)
        eva_tool_prob = [util_eva.EvaPairWeight(self.dataset_valid.classNames) for i in range(self.model.mconfig.N_LAYERS)]
        while(keep_training):
            print('\n\nTraining epoch: %d' % epoch)
            for items in loader:
                self.model.train()
                
                scan_id = items[0][0]
                instance2mask = items[1]
                if items[2].ndim < 4: continue
                
                ''' get data '''
                tick = time.time()                
                if self.use_edge_descriptor:
                    obj_points, edge_indices, gt_obj_cls, gt_rel_cls, descriptor = self.data_processing(items[2:])
                else:
                    obj_points, rel_points, edge_indices, gt_obj_cls, gt_rel_cls = self.data_processing(items[2:])                    
                tock = time.time()

                if edge_indices.ndim == 1: 
                    # print('no edges found. skip this.')
                    continue
                if obj_points.shape[0] < 2 : 
                    # print('need at least two nodes. skip this one. (got ',obj_points.shape,')')
                    continue
                if edge_indices.shape[0] <= 1:
                    # print('no edges! skip')
                    continue

                ignore_none_rel = False
                w_rel = self.dataset_train.w_cls_rel
                
                if 'scene' in scan_id:
                    if 'ignore_scannet_rel' in self.config.dataset:
                        ignore_none_rel = self.config.dataset.ignore_scannet_rel
                    else:
                        ignore_none_rel = True
                    
                tick = time.time()
                if self.use_edge_descriptor:
                    logs, pred_obj_cls, pred_rel_cls, probs = self.model.process(obj_points, edge_indices.t().contiguous(), descriptor,
                                                                      gt_obj_cls, gt_rel_cls,
                                                                      weights_obj=self.dataset_train.w_cls_obj, 
                                                                      weights_rel=w_rel,
                                                                      ignore_none_rel = ignore_none_rel)
                else:
                    logs, pred_obj_cls, pred_rel_cls, probs = self.model.process(obj_points, rel_points, edge_indices.t().contiguous(), 
                                                                      gt_obj_cls, gt_rel_cls,
                                                                      weights_obj=self.dataset_train.w_cls_obj, 
                                                                      weights_rel=self.dataset_train.w_cls_rel)
                tock = time.time()
                
                tick = time.time()
                eva_tool.add(scan_id, pred_obj_cls, gt_obj_cls, pred_rel_cls, gt_rel_cls, instance2mask, edge_indices)
                if probs is not None:
                    if self.model.mconfig.GCN_TYPE == 'EGCN':
                        for l in range(self.model.mconfig.N_LAYERS):                
                            eva_tool_prob[l].update(probs[l].mean(dim=1), edge_indices,gt_obj_cls)
                            
                ''' calculate metrics '''
                cls_, rel_ = eva_tool.get_mean_metrics()
                logs += [("IoU/train_obj_cls",cls_[0]), 
                        ("Precision/train_obj_cls",cls_[1]), 
                        ("Recall/train_obj_cls",cls_[2]), 
                        ("IoU/train_rel_cls",rel_[0]), 
                        ("Precision/train_rel_cls",rel_[1]), 
                        ("Recall/train_rel_cls",rel_[2])]                            

                iteration = self.model.iteration
                logs = [
                    ("Misc/epo", int(epoch)),
                    ("Misc/it", int(iteration)),
                ] + logs
                
                progbar.add(1, values=logs \
                            if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])
                    
                ''' save model at checkpoints '''
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs, iteration)
                if self.config.LOG_IMG_INTERVAL and iteration % self.config.LOG_IMG_INTERVAL == 0:
                    img_confusion_matrix = plot_confusion_matrix.plot_confusion_matrix(eva_tool.eva_o_cls.c_mat, 
                                                                 eva_tool.eva_o_cls.class_names,
                                                                 title='Object Confusion matrix',
                                                                 plot_text=False,
                                                                 plot = False)
                    self.writter.add_figure('train_obj_confusion_matrix', img_confusion_matrix, global_step=self.model.iteration)
                    
                    img_confusion_matrix = plot_confusion_matrix.plot_confusion_matrix(eva_tool.eva_r_cls.c_mat, 
                                                    eva_tool.eva_r_cls.class_names,
                                                    title='Predicate Confusion Matrix',
                                                    plot_text=False,
                                                    plot = False)
                    self.writter.add_figure('train_rel_confusion_matrix', img_confusion_matrix, global_step=self.model.iteration)
                    # r_c_matrix = np.zeros((num_rel, num_rel), dtype=np.ulonglong)
                    eva_tool.reset()
                    
                    for i in range(self.model.mconfig.N_LAYERS):
                        c_mat = eva_tool_prob[i].c_mat
                        img_score_matrix = plot_confusion_matrix.plot_confusion_matrix(
                            c_mat, 
                            eva_tool_prob[i].class_names, 
                            title='prob_'+str(i),
                            plot_text=False,
                            plot = False,
                            normalize=False)
                        self.writter.add_figure('train_probs_'+str(i), img_score_matrix,global_step=self.model.iteration)
                        eva_tool_prob[i].reset()
                tock = time.time()
                if self.model.iteration >= max_iteration:
                    break
            epoch+=1
            
            if 'VALID_INTERVAL' in self.config and self.config.VALID_INTERVAL > 0 and epoch % self.config.VALID_INTERVAL == 0:
                print('start validation...')
                m_obj, m_rel = self.validation()
                self.model.eva_iou = m_obj[0]
            self.save()
            
            if epoch > self.config.MAX_EPOCHES:
                break
            progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
            loader = iter(train_loader)
        self.save()
        print('')
        self.eval()
       
    def cuda(self, *args):
        return [item.to(self.config.DEVICE) for item in args]
    
    def log(self, logs, iteration):
        # Tensorboard
        if self.writter is not None:
            for i in logs:
                if not i[0].startswith('Misc'):
                    self.writter.add_scalar(i[0], i[1], iteration)
                    
    def save(self):
        self.model.save()
        
    def validation(self, debug_mode = False):
        val_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_valid,
            batch_size=1,
            num_workers=self.config.WORKERS,
            drop_last=False,
            shuffle=False
        )
        from utils import util_eva
        eva_tool = util_eva.EvalSceneGraph(self.dataset_valid.classNames, self.dataset_valid.relationNames,
                                  multi_rel_outputs=0.5, k=0, 
                                  multi_rel_prediction=self.model.mconfig.multi_rel_outputs)
       
        total = len(self.dataset_valid)
        progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/it'])
        
        print('===   start evaluation   ===')
        self.model.eval()
        for i, items in enumerate(val_loader, 0):
            scan_id = items[0][0]
            instance2mask = items[1]                        
            if items[2].ndim < 4: continue
            
            ''' get data '''
            tick = time.time()
            if self.use_edge_descriptor:
                obj_points, edge_indices, gt_obj_cls, gt_rel_cls, descriptor = self.data_processing(items[2:])
            else:
                obj_points, rel_points, edge_indices, gt_obj_cls, gt_rel_cls = self.data_processing(items[2:])                
            tock = time.time()
            
            if edge_indices.ndim == 1: 
                # print('no edges found. skip this.')
                continue
            if obj_points.shape[0] < 2 : 
                # print('need at least two nodes. skip this one. (got ',obj_points.shape,')')
                continue
            if edge_indices.shape[0] == 0:
                # print('no edges! skip')
                continue
                
            tick = time.time()
            with torch.no_grad():
                if self.use_edge_descriptor:
                    pred_obj_cls, pred_rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs = \
                            self.model(obj_points, edge_indices.t().contiguous(), descriptor, return_meta_data=True)
                else:
                    pred_obj_cls, pred_rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs = \
                        self.model(obj_points, rel_points, edge_indices.t().contiguous(), return_meta_data=True)
                        
            ''' calculate metrics '''
            logs = self.model.calculate_metrics([pred_obj_cls, pred_rel_cls], [gt_obj_cls, gt_rel_cls])
            
            ignore_rel = False
            if 'scene' in scan_id:
                if 'ignore_scannet_rel' in self.config.dataset:
                    ignore_rel = self.config.dataset.ignore_scannet_rel
                else:
                    ignore_rel = True
                    
            if ignore_rel:
                pred_rel_cls = gt_rel_cls = None
            eva_tool.add(scan_id, pred_obj_cls, gt_obj_cls, pred_rel_cls, gt_rel_cls, instance2mask, edge_indices)
            
            
            idx2seg=dict()
            for key,item in instance2mask.items():
                idx2seg[item.item()-1] = key
            
            logs = [
               
                ] + logs
            progbar.add(1, values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])

            if debug_mode:
                if i > 0:
                    break
            # break
        img_confusion_matrix = plot_confusion_matrix.plot_confusion_matrix(eva_tool.eva_o_cls.c_mat, 
                                                                 eva_tool.eva_o_cls.class_names,
                                                                 title='Object Confusion matrix',
                                                                 plot_text=False,
                                                                 plot = False)
        self.writter.add_figure('vali_obj_confusion_matrix', img_confusion_matrix, global_step=self.model.iteration)
        img_confusion_matrix = plot_confusion_matrix.plot_confusion_matrix(eva_tool.eva_r_cls.c_mat, 
                                        eva_tool.eva_r_cls.class_names,
                                        title='Predicate Confusion Matrix',
                                        plot_text=False,
                                        plot = False)
        self.writter.add_figure('vali_rel_confusion_matrix', img_confusion_matrix, global_step=self.model.iteration)
       
        cls_, rel_ = eva_tool.get_mean_metrics()
        logs = [("IoU/val_obj_cls",cls_[0]), 
                ("Precision/val_obj_cls",cls_[1]), 
                ("Recall/val_obj_cls",cls_[2]), 
                ("IoU/val_rel_cls",rel_[0]), 
                ("Precision/val_rel_cls",rel_[1]), 
                ("Recall/val_rel_cls",rel_[2])]  
        self.log(logs,self.model.iteration)
        return cls_, rel_

        
    def eval(self, debug_mode=False):
        if self.dataset_eval is None:
            print('no evaludation dataset was built!')
            return 
        val_loader = CustomDataLoader(
            config = self.config,
            dataset=self.dataset_eval,
            batch_size=1,
            num_workers=0,
            drop_last=False,
            shuffle=False
        )
        from utils import util_eva
        eva_tool = util_eva.EvalSceneGraph(self.dataset_eval.classNames, self.dataset_eval.relationNames,
                                  multi_rel_outputs=0.5, k=100,multi_rel_prediction=self.model.mconfig.multi_rel_outputs)
        
        total = len(self.dataset_eval)
        progbar = op_utils.Progbar(total, width=20, stateful_metrics=['Misc/it'])
        
        print('===   start evaluation   ===')
        list_feature_maps = dict()
        list_feature_maps['node_feature'] = list()
        list_feature_maps['edge_feature'] = list()
        list_feature_maps['gcn_node_feature'] = list()
        list_feature_maps['gcn_edge_feature'] = list()
        list_feature_maps['node_names'] = list()
        list_feature_maps['edge_names'] = list()

        self.model.eval()
        for i, items in enumerate(val_loader, 0):
            scan_id = items[0][0]
            instance2mask = items[1]                        
            if items[2].ndim < 4: continue
            
            ''' get data '''
            if self.use_edge_descriptor:
                obj_points, edge_indices, gt_obj_cls, gt_rel_cls, descriptor = self.data_processing(items[2:])
            else:
                obj_points, rel_points, edge_indices, gt_obj_cls, gt_rel_cls = self.data_processing(items[2:])
            
            if edge_indices.ndim == 1: 
                # print('no edges found. skip this.')
                continue
            if obj_points.shape[0] < 2 : 
                # print('need at least two nodes. skip this one. (got ',obj_points.shape,')')
                continue
            if edge_indices.shape[0] == 0:
                # print('no edges! skip')
                continue
            
            with torch.no_grad():
                if self.use_edge_descriptor:
                    pred_obj_cls, pred_rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs = \
                            self.model(obj_points, edge_indices.t().contiguous(), descriptor, return_meta_data=True)
                else:
                    pred_obj_cls, pred_rel_cls, obj_feature, rel_feature, gcn_obj_feature, gcn_rel_feature, probs = \
                        self.model(obj_points, rel_points, edge_indices.t().contiguous(), return_meta_data=True)
            
            ''' calculate metrics '''
            logs = self.model.calculate_metrics([pred_obj_cls, pred_rel_cls], [gt_obj_cls, gt_rel_cls])
            
            
            ignore_rel = False
            if 'scene' in scan_id:
                if 'ignore_scannet_rel' in self.config.dataset:
                    ignore_rel = self.config.dataset.ignore_scannet_rel
                else:
                    ignore_rel = True
            if ignore_rel:
                pred_rel_cls = gt_rel_cls = None
            
            eva_tool.add(scan_id, pred_obj_cls, gt_obj_cls, pred_rel_cls, gt_rel_cls, instance2mask, edge_indices)
            
            
            idx2seg=dict()
            for key,item in instance2mask.items():
                idx2seg[item.item()-1] = key
                
            [list_feature_maps['node_names'].append(self.dataset_eval.classNames[aa]) for aa in gt_obj_cls.tolist()]            
            list_feature_maps['node_feature'].append(obj_feature.detach().cpu())
            list_feature_maps['edge_feature'].append(rel_feature.detach().cpu())
            if gcn_obj_feature is not None:
                list_feature_maps['gcn_node_feature'].append(gcn_obj_feature.detach().cpu())
            
            if not ignore_rel:
                if gcn_rel_feature is not None:
                    list_feature_maps['gcn_edge_feature'].append(gcn_rel_feature.detach().cpu())                
                if self.model.mconfig.multi_rel_outputs:
                    for a in range(gt_rel_cls.shape[0]):
                        name = ''
                        for aa in range(gt_rel_cls.shape[1]):
                            if gt_rel_cls[a][aa] > 0:
                                name += self.dataset_eval.relationNames[aa] + '_'
                        if name == '':
                            name = 'none'
                        list_feature_maps['edge_names'].append(name)
                else:
                    for a in range(gt_rel_cls.shape[0]):
                        list_feature_maps['edge_names'].append(self.dataset_eval.relationNames[gt_rel_cls[a]])

            logs = [
               
                ] + logs
            progbar.add(1, values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])

            if debug_mode:
                if i > 0:
                    break

        print(eva_tool.gen_text())
        pth_out = os.path.join(self.results_path,str(self.model.iteration))
        op_utils.create_dir(pth_out)
        
        result_metrics = eva_tool.write(pth_out, self.model_name)
        # result_metrics = {'hparam/'+key: value for key,value in result_metrics.items()}
        tmp_dict = dict()
        for key,item in self.model.mconfig.items():
            if isinstance(item, int) or isinstance(item, float) or isinstance(item, str) or isinstance(item, bool) or \
                isinstance(item, torch.Tensor):
                 tmp_dict[key]=item
        self.writter.add_hparams(tmp_dict, metric_dict = result_metrics)
        
        img_confusion_matrix = plot_confusion_matrix.plot_confusion_matrix(eva_tool.eva_o_cls.c_mat, 
                                                                 eva_tool.eva_o_cls.class_names,
                                                                 title='Object Confusion matrix',
                                                                 plot_text=False,
                                                                 plot = False)
        self.writter.add_figure('eval_obj_confusion_matrix', img_confusion_matrix, global_step=self.model.iteration)
        img_confusion_matrix = plot_confusion_matrix.plot_confusion_matrix(eva_tool.eva_r_cls.c_mat, 
                                        eva_tool.eva_r_cls.class_names,
                                        title='Predicate Confusion Matrix',
                                        plot_text=False,
                                        plot = False)
        self.writter.add_figure('eval_rel_confusion_matrix', img_confusion_matrix, global_step=self.model.iteration)
        
        for name, list_tensor in list_feature_maps.items():
            # if name == 'label_names':continue    
            if not isinstance(list_tensor, list): continue
            if len(list_tensor) == 0:continue
            if not isinstance(list_tensor[0], torch.Tensor): continue
            if len(list_tensor) < 1: continue
            tmp = torch.cat(list_tensor,dim=0)
            if name.find('node')>=0:
                names = list_feature_maps['node_names']
            elif name.find('edge')>=0:
                names = list_feature_maps['edge_names']
            else:
                continue
            print(name)
            self.writter.add_embedding(tmp,metadata=names,tag=self.model_name+'_'+name,
                                       global_step=self.model.iteration)
        
    def trace(self):
        op_utils.create_dir(self.trace_path)
        args = self.model.trace(self.trace_path)
        with open(os.path.join(self.trace_path, 'classes.txt'), 'w') as f:
            for c in self.dataset_valid.classNames:
                f.write('{}\n'.format(c))
                
        ''' save relation file'''
        with open(os.path.join(self.trace_path, 'relationships.txt'), 'w') as f:
            for c in self.dataset_valid.relationNames:
                f.write('{}\n'.format(c))
                
        import json
        with open(os.path.join(self.trace_path, 'args.json'), 'w') as f:
            args['label_type'] = self.dataset_valid.label_type
            json.dump(args, f, indent=2)
        pass
            
if __name__ == '__main__':
    TEST_CUDA=True
    TEST_EVAL=False
    TEST_TRACE=False
    
    config = Config('../config_example.json')
    config.dataset.root = "../data/example_data"
    config.MODEL.GCN_TYPE = 'EAN'
    config.MODEL.multi_rel_outputs=False
    config['NAME'] = 'test'
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    
    # init device
    if TEST_CUDA and torch.cuda.is_available() and len(config.GPU) > 0:
        config.DEVICE = torch.device("cuda")
    else:
        config.DEVICE = torch.device("cpu")
        
    config.MODE = 'train' if not TEST_EVAL else 'eval'
    
    pg = SGFN(config)
    if TEST_TRACE:
        pg.trace()
    elif not TEST_EVAL:
        pg.train()
    else:
        pg.eval(debug_mode = True)
