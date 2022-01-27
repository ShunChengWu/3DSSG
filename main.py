import logging
import argparse,os
import codeLib
import ssg
import torch
import ssg.config as config
from ssg.data.collate import graph_collate, batch_graph_collate
from ssg.checkpoints import CheckpointIO
import torch.multiprocessing
import cProfile

# import resource
# resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))
logging.basicConfig()
logger_py = logging.getLogger(__name__)
# logging.basicConfig()
# logger_py.setLevel(logging.DEBUG)

def main():
    cfg = parse()
    logger_py.setLevel(cfg.log_level)
    
    # Shorthands
    out_dir = os.path.join(cfg['training']['out_dir'], cfg.name)
    
    # set random seed
    codeLib.utils.util.set_random_seed(cfg.SEED)
    
    # Output directory
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    
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
        dataset_test  = config.get_dataset(cfg,'test')
        val_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, num_workers=0,
            shuffle=False, drop_last=False,
            pin_memory=True,
            collate_fn=graph_collate,
        )
        
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
        eval_dict, eval_tool = model_trainer.evaluate(val_loader, topk=100)
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
        

def parse():
    r"""loads model config

    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='./configs/default.yaml', help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('--mode', type=str, choices=['train','trace','eval','sample'], default='train', help='mode. can be [train,trace,eval]',required=False)
    parser.add_argument('--loadbest', type=int, default=0,choices=[0,1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')
    parser.add_argument('--log', type=str, default='DEBUG',choices=['DEBUG','INFO','WARNING','CRITICAL'], help='')
    args = parser.parse_args()
    
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise RuntimeError('Targer config file does not exist. {}' & config_path)
        
    # load config file
    config = codeLib.Config(config_path)
    # return config
    config.LOADBEST = args.loadbest
    config.MODE = args.mode
    
    # check if name exist
    if 'name' not in config:
        config_name = os.path.basename(args.config)
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['name'] = name 
    
    # init device
    if torch.cuda.is_available() and len(config.GPU) > 0:
        config.DEVICE = torch.device("cuda")
    else:
        config.DEVICE = torch.device("cpu")      
        
    config.log_level = args.log
    # logging.basicConfig(level=config.log_level)
    # logging.setLevel(config.log_level)
    return config

if __name__ == '__main__':
    # logger_py.setLevel('DEBUG')
    # logger_py.debug('hello0')
    main()