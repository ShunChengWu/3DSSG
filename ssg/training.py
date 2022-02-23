import os
import numpy as np
from tqdm import tqdm, tnrange
from collections import defaultdict
import ssg
from codeLib.models import BaseTrainer
from codeLib.common import check_weights, check_valid, convert_torch_to_scalar
import torch
import torch.nn.functional as F
from ssg.utils.util_eva import EvalSceneGraph, plot_confusion_matrix
import time
import logging
from ssg.checkpoints import CheckpointIO
import torch.optim as optim
import codeLib.utils.moving_average as moving_average 
import torch.multiprocessing
logger_py = logging.getLogger(__name__)
        
class Trainer():
    def __init__(self, cfg, model_trainer:BaseTrainer, #node_cls_names, edge_cls_names,
                 logger=None,
                 device=None, 
                 **kwargs):
        super().__init__()
        self._device=device if device is not None else 'cpu'
        self.cfg = cfg
        self.model_trainer = model_trainer
        #self.node_cls_names = node_cls_names#kwargs['node_cls_names']
        #self.edge_cls_names = edge_cls_names#kwargs['edge_cls_names']
        self.logger = logger        
        # self.eva_tool = EvalSceneGraph(self.node_cls_names, self.edge_cls_names,multi_rel_prediction=self.cfg.model.multi_rel,k=0) # do not calculate topK in training mode
        
        # self.loss_node_cls = torch.nn.CrossEntropyLoss(weight=self.w_node_cls)
        # if self.cfg.model.multi_rel:
        #     self.loss_rel_cls = torch.nn.BCEWithLogitsLoss(pos_weight=self.w_edge_cls)
        # else:
        #     self.loss_rel_cls = torch.nn.CrossEntropyLoss(weight=self.w_edge_cls)
        #     pass
        self.scheduler = ssg.config.get_schedular(cfg, self.model_trainer.optimizer, last_epoch=-1)
        
        ''' load model and previous information '''
        out_dir = os.path.join(self.cfg['training']['out_dir'], cfg.name)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        ckpt_io = CheckpointIO(out_dir, model=self.model_trainer.model, optimizer=self.model_trainer.optimizer,scheduler=self.scheduler)
        
        if cfg['training']['model_selection_mode'] == 'maximize':
            model_selection_sign = 1
        elif cfg['training']['model_selection_mode'] == 'minimize':
            model_selection_sign = -1
        else:
            raise ValueError('model_selection_mode must be '
                            'either maximize or minimize.')
            
        self.ckpt_io = ckpt_io
        self.metric_sign = model_selection_sign
        self.selected_metric = cfg['training']['model_selection_metric']
    
    def fit(self,**args):
        '''shortcut'''
        logger_py.setLevel(self.cfg.log_level)
        cfg = self.cfg
        self.patient = 0
        
        max_epoch = self.cfg['training']['max_epoch']
        max_patient = self.cfg['training']['patient']
        
        '''get'''
        logger = self.logger#args.get('logger', None)
        train_loader = args['train_loader']
        val_loader = args['val_loader']
        
        # '''init'''
        try:
            load_dict = self.ckpt_io.load('model.pt', device=cfg.DEVICE)
        except FileExistsError:
            load_dict = dict()
        epoch_it = load_dict.get('epoch_it', -1)
        it = load_dict.get('it', -1)
        self.metric_val_best = load_dict.get('loss_val_best', -self.metric_sign * np.inf)
            
        # self.scheduler = ssg.config.get_schedular(cfg, self.model_trainer.optimizer, last_epoch=epoch_it)
        
        
        '''  '''
        if logger:
            try:
                logger.watch(self.model_trainer.model,
                              log_freq=cfg.logging.log_grad_freq,
                              log_graph=cfg.logging.log_graph) #TODO: add tensorboard version of logging gradient
            except: pass
        if epoch_it <0:epoch_it=0
        pbar = tqdm(range(epoch_it, max_epoch), desc='[Epoch %02d]' % (epoch_it))
        avg_loss=0
        for epoch in pbar:
            # logger_py.info('Train epoch')
            pbar.set_description('[Epoch %02d] loss=%.4f' % (epoch, avg_loss))
            it, epo_time, _, avg_loss = self.train(
                train_loader=train_loader,
                val_loader=val_loader,
                epoch_it=epoch,
                it_start=it,
                logger=logger
            )
            
            if logger:
                metrics = self.model_trainer.get_log_metrics()
                for k,v in metrics.items(): logger.add_scalar('train/'+k, v,it)
                logger.add_scalar('train/epoch', epoch,it)
                logger.add_scalar('train/lr', self.model_trainer.optimizer.param_groups[0]['lr'],it)
            
            
            # Visualization
            # if (visualize_every > 0 and (it % visualize_every) == 0):
            figs = self.model_trainer.visualize()
            if logger:
                for k,v in figs.items(): logger.add_figure('train/'+k, v,it)
                
            # Save checkpoint
            # logger_py.info('Saving checkpoint')
            self.ckpt_io.save('model.pt', epoch_it=epoch, it=it,
                            loss_val_best=self.metric_val_best)
                        
            # validate
            pbar.set_description('[Epoch %02d] loss=%.4f it=%03d,Run Validation' % (epoch, avg_loss, it))
            eval_dict = self.run_validation(val_loader, logger, epoch,it)
            
            # check patient
            if max_patient > 0 and self.patient > max_patient:
                logger_py.info('reach maximum patient! {} > {}. Stop training'.\
                               format(self.patient, max_patient))
                break
            
            if cfg.training.scheduler.method.lower() == 'reduceluronplateau':
                metric_val = eval_dict[self.selected_metric]
                self.scheduler.step(metric_val) # maybe make a EMA otherwise it's too sensitive to outliers.
            else:
                self.scheduler.step()
            
            
        logger_py.info('Training finished.')

    def train(self, train_loader,val_loader, epoch_it, it_start, **args):
        log_every = self.cfg['training']['log_every']
        logger = args.get('logger', None)
        scalar_list = defaultdict(moving_average.CMA)
        it = it_start
        avg_time = moving_average.CMA()
        avg_loss = moving_average.CMA()
        
        epo_time = time.time()
        it_time = time.time()
        time.sleep(2)# Prevent possible deadlock during epoch transition
        torch.multiprocessing.set_sharing_strategy('file_system')
        it_dataset = train_loader.__iter__()
        pbar = tqdm(it_dataset,leave=False)
        self.model_trainer.zero_metrics()
        for data in pbar:
            # continue
            it+=1
            logs = self.model_trainer.train_step(data, it)
            
            if 'loss' not in logs: continue
            # logs = convert_torch_to_scalar(logs)            
            loss = logs['loss']
            
            #update time
            avg_time.update(time.time()-it_time)
            it_time = time.time()
            
            avg_loss.update(loss)
            pbar.set_description('it=%03d, loss=%.4f, time=%.4f' % (it, avg_loss.avg, avg_time.avg))
            
            # accumulate scalars
            for k, v in logs.items():
                if isinstance(v,dict):continue
                scalar_list['train/'+k].update(v)
              
            # log scalars
            if logger and log_every > 0 and (it % log_every) == 0:
                for k, v in scalar_list.items():
                    logger.add_scalar(k, v.avg,it)
                scalar_list = defaultdict(moving_average.CMA)
            break
        epo_time = time.time() - epo_time
        del it_dataset
        return it, epo_time, avg_time.avg, avg_loss.avg
    
    def run_validation(self,val_loader,logger,
                       epoch_it, it):
        '''
        Run validation with logging and checkpoint management.
    
        '''
        torch.cuda.empty_cache()
        eval_dict, *_ = self.model_trainer.evaluate(val_loader, topk=0) # only calculate topK in evaluation. it's slow.
        metric_val = eval_dict[self.selected_metric]
        torch.cuda.empty_cache()

        if logger:
            logger.add_scalar('val/epoch', epoch_it,it)
            for k,v in eval_dict['visualization'].items(): 
                logger.add_figure('val/'+k, v,it)
            for k, v in eval_dict.items():
                if isinstance(v,dict): continue
                logger.add_scalar('val/%s' % k, v,it)
        
        if self.metric_sign * (metric_val - self.metric_val_best) > 0:
            self.metric_val_best = metric_val
            self.patient = 0
            logger_py.info('New best model (%s %.4f)' % (self.metric_sign, metric_val))
            # self.ckpt_io.backup_model_best('model_best.pt')
            self.ckpt_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                loss_val_best=metric_val)
        else:
            logger_py.info('new val metric is worse (%.4f %.4f).increase patient to %d (max patient %d)' % \
                           (metric_val, self.metric_val_best,self.patient, self.cfg['training']['patient']))
            self.patient += 1
            
        return eval_dict