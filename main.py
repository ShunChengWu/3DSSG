import logging
#import trimesh
import argparse,os
import codeLib
import ssg
import torch
import ssg.config as config
from ssg.data.collate import graph_collate#, batch_graph_collate
from ssg.checkpoints import CheckpointIO
import torch.multiprocessing
import cProfile
import matplotlib
import PIL
import copy
import codeLib.utils.string_numpy as snp
matplotlib.pyplot.set_loglevel("CRITICAL")
logging.getLogger('PIL').setLevel('CRITICAL')
logging.getLogger('trimesh').setLevel('CRITICAL')
# import resource
# resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))
# logging.basicConfig()

logger_py = logging.getLogger(__name__)

# logging.basicConfig()
# logger_py.setLevel(logging.DEBUG)

def main():
    cfg = ssg.Parse()
    
    # Shorthands
    out_dir = os.path.join(cfg['training']['out_dir'], cfg.name)
    
    # set random seed
    codeLib.utils.util.set_random_seed(cfg.SEED)
    
    # Output directory
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
    # Log
    logging.basicConfig(filename=os.path.join(out_dir,'log'), level=cfg.log_level)
    logger_py.setLevel(cfg.log_level)
    # logger_py.info('info')
    # logger_py.debug('debug')
    # import sys
    # sys.exit()
    
    if cfg.MODE == 'train':
        logger_py.info('train')
        n_workers = cfg['training']['data_workers']
        ''' create dataset and loaders '''
        dataset_train = config.get_dataset(cfg,'train')
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=cfg['training']['batch'], num_workers=n_workers, shuffle=True,
            pin_memory=True,
            collate_fn=graph_collate,
        )
        dataset_val  = config.get_dataset(cfg,'validation')
        val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=1, num_workers=n_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=graph_collate,
        )
        # try to load one data
        dataset_train.__getitem__(0)
        
        # Get logger
        logger = config.get_logger(cfg)
        if logger is not None: logger, _ = logger
        
        ''' Create model '''
        relationNames = dataset_train.relationNames
        classNames = dataset_train.classNames
        num_obj_cls = len(dataset_train.classNames)
        num_rel_cls = len(dataset_train.relationNames) if relationNames is not None else 0
        
        model = config.get_model(cfg,num_obj_cls=num_obj_cls, num_rel_cls=num_rel_cls)
        
        if cfg.VERBOSE:
            print(model)
        
        # crreate optimizer    
        # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        # optimizer = config.get_optimizer(cfg, trainable_params)
        
        # trainer
        model_trainer = config.get_trainer(cfg, model, classNames, relationNames,
                                           w_node_cls=dataset_train.w_node_cls,
                                           w_edge_cls=dataset_train.w_edge_cls)
        
        trainer = ssg.Trainer(
            cfg = cfg, 
            model_trainer = model_trainer, 
            node_cls_names = classNames, 
            edge_cls_names=relationNames,
            logger=logger,
            )
        
        pr = cProfile.Profile()
        pr.enable()
        trainer.fit(train_loader=train_loader,val_loader=val_loader)
        pr.disable()
        logger_py.info('save time profile to {}'.format(os.path.join(out_dir,'tp_train.dmp')))
        pr.dump_stats(os.path.join(out_dir,'tp_train.dmp'))
    elif cfg.MODE == 'eval':
        cfg.data.load_cache=False
        eval_mode = cfg.eval.mode
        assert eval_mode in ['segment','instance']
        if eval_mode == 'segment':
            dataset_test  = config.get_dataset(cfg,'test')
            val_loader = torch.utils.data.DataLoader(
                dataset_test, batch_size=1, num_workers=cfg['eval']['data_workers'],
                shuffle=False, drop_last=False,
                pin_memory=True,
                collate_fn=graph_collate,
            )
            dataset_test.__getitem__(0)
            
            # Get logger
            # logger=None
            logger = config.get_logger(cfg)
            if logger is not None: logger, _ = logger
            
            ''' Create model '''
            relationNames = dataset_test.relationNames
            classNames = dataset_test.classNames
            num_obj_cls = len(dataset_test.classNames)
            num_rel_cls = len(dataset_test.relationNames) if relationNames is not None else 0
            
            model = config.get_model(cfg,num_obj_cls=num_obj_cls, num_rel_cls=num_rel_cls)
            
            model_trainer = config.get_trainer(cfg, model, classNames, relationNames,
                                               w_node_cls=None,
                                               w_edge_cls=None
                                               )
            
            checkpoint_io = CheckpointIO(out_dir, model=model)
            load_dict = checkpoint_io.load('model_best.pt', device=cfg.DEVICE)
            it = load_dict.get('it', -1)
            
            # use the evaluation function inside trainer
            # trainer = ssg2d.Trainer(cfg,model,device=cfg.DEVICE,
            #                         node_cls_names = classNames,
            #                         edge_cls_names = relationNames)
            
            #
            logger_py.info('start evaluation')
            pr = cProfile.Profile()
            pr.enable()
            eval_dict, eval_tool = model_trainer.evaluate(val_loader, topk=cfg.eval.topK)
            pr.disable()
            logger_py.info('save time profile to {}'.format(os.path.join(out_dir,'tp_eval.dmp')))
            pr.dump_stats(os.path.join(out_dir,'tp_eval.dmp'))
            
            #
            print(eval_tool.gen_text())
            _ = eval_tool.write(out_dir, cfg.name)
    
            if logger:
                for k,v in eval_dict['visualization'].items(): 
                    logger.add_figure('test/'+k, v, global_step=it)
                for k, v in eval_dict.items():
                    if isinstance(v,dict): continue
                    logger.add_scalar('test/%s' % k, v, it)
        elif eval_mode == 'instance':
            ''' Get segment dataset '''
            dataset_seg  = config.get_dataset(cfg,'test')
            ''' Get instance dataset'''
            dataset_inst = config.get_dataset_inst(cfg,'test')
            # write back
            
            '''check'''
            # assert len(dataset_seg.relationNames) == len(dataset_inst.relationNames)+1
            assert len(dataset_seg.classNames)==len(dataset_inst.classNames)
            
            ''' Get logger '''
            logger = config.get_logger(cfg)
            if logger is not None: logger, _ = logger
            
            ''' Create model '''
            relationNames = dataset_seg.relationNames
            classNames = dataset_seg.classNames
            num_obj_cls = len(dataset_seg.classNames)
            num_rel_cls = len(dataset_seg.relationNames) if relationNames is not None else 0
            
            model = config.get_model(cfg,num_obj_cls=num_obj_cls, num_rel_cls=num_rel_cls)
            
            model_trainer = config.get_trainer(cfg, model, classNames, relationNames,
                                               w_node_cls=None,
                                               w_edge_cls=None
                                               )
            '''check ckpt'''
            checkpoint_io = CheckpointIO(out_dir, model=model)
            load_dict = checkpoint_io.load('model_best.pt', device=cfg.DEVICE)
            it = load_dict.get('it', -1)
            
            '''start eval'''
            logger_py.info('start evaluation')
            pr = cProfile.Profile()
            pr.enable()    
            eval_dict, eval_tool,eval_tool_upperbound = model_trainer.evaluate_inst(dataset_seg,dataset_inst, topk=cfg.eval.topK)
            pr.disable()
            logger_py.info('save time profile to {}'.format(os.path.join(out_dir,'tp_eval_inst.dmp')))
            pr.dump_stats(os.path.join(out_dir,'tp_eval_inst.dmp'))
            
            '''log'''
            # ignore_missing=cfg.eval.ignore_missing
            prefix='inst' if not cfg.eval.ignore_missing else 'inst_ignore'
            
            print(eval_tool.gen_text())
            _ = eval_tool.write(out_dir, prefix)
            
            eval_tool_upperbound.write(out_dir,'upper_bound')
            
            if logger:
                for k,v in eval_dict['visualization'].items(): 
                    logger.add_figure('test/'+prefix+'_'+k, v, global_step=it)
                for k, v in eval_dict.items():
                    if isinstance(v,dict): continue
                    logger.add_scalar('test/'+prefix+'_'+'%s' % k, v, it)
            
            
    elif cfg.MODE == 'sample':
        dataset_test  = config.get_dataset(cfg,'test')
        val_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, num_workers=0,
            shuffle=False, drop_last=False,
            pin_memory=True,
            collate_fn=graph_collate,
        )
        
        ''' Create model '''
        relationNames = dataset_test.relationNames
        classNames = dataset_test.classNames
        num_obj_cls = len(dataset_test.classNames)
        num_rel_cls = len(dataset_train.relationNames) if relationNames is not None else 0
        
        model = config.get_model(cfg,num_obj_cls=num_obj_cls, num_rel_cls=num_rel_cls)
        
        model_trainer = config.get_trainer(cfg, model, classNames, relationNames,
                                           w_node_cls=None,
                                           w_edge_cls=None
                                           )
        
        checkpoint_io = CheckpointIO(out_dir, model=model)
        load_dict = checkpoint_io.load('model_best.pt', device=cfg.DEVICE)
        it = load_dict.get('it', -1)
        
        logger_py.info('start sample')
        model_trainer.sample(val_loader)
        logger_py.info('sample finished')
    elif cfg.MODE == 'trace':
        logger_py.info('trace')
        logger_py.info('build model')
        
        cfg.DEVICE = torch.device("cpu")      
        dataset  = config.get_dataset(cfg,'validation')
        # val_loader = torch.utils.data.DataLoader(
        #     dataset, batch_size=1, num_workers=n_workers,
        #     shuffle=False,
        #     drop_last=False,
        #     pin_memory=True,
        #     collate_fn=graph_collate,
        # )
        
        relationNames = dataset.relationNames
        classNames = dataset.classNames
        num_obj_cls = len(dataset.classNames)
        num_rel_cls = len(dataset.relationNames) if relationNames is not None else 0
        
        model = config.get_model(cfg,num_obj_cls=num_obj_cls, num_rel_cls=num_rel_cls)
        logger_py.info('load checkpoint')
        checkpoint_io = CheckpointIO(out_dir, model=model)
        load_dict = checkpoint_io.load('model_best.pt', device=cfg.DEVICE)
        
        logger_py.info('trace')
        model.trace(out_dir)
        pass
        
        
    else:
        raise NotImplementedError('unknown input mode')
        
if __name__ == '__main__':
    # logger_py.setLevel('DEBUG')
    # logger_py.debug('hello0')
    main()