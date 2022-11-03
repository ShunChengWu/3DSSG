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
import torch_geometric
# disable GUI
matplotlib.pyplot.switch_backend('agg')
# change log setting
matplotlib.pyplot.set_loglevel("CRITICAL")
logging.getLogger('PIL').setLevel('CRITICAL')
logging.getLogger('trimesh').setLevel('CRITICAL')
logger_py = logging.getLogger(__name__)


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

    ''' Get segment dataset '''
    logger_py.info('create loader')
    cfg.data.input_type = 'sgfn_seq'
    # cfg.data.is_roi_img=True
    dataset_seg  = config.get_dataset(cfg,'test')    
    
    dataset_inst = config.get_dataset_inst(cfg,'test')
    
    logger_py.info('test loader')
    dataset_seg.__getitem__(0)
    for i,data in enumerate(dataset_seg):
        # print(i)
        break
    dataset_inst.__getitem__(0)
    for i,data in enumerate(dataset_inst):
        # print(i)
        break
    
    '''check'''
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
    eval_dict, eval_tools,eval_tool_upperbound = model_trainer.evaluate_inst_incre(dataset_seg,dataset_inst, topk=cfg.eval.topK)
    pr.disable()
    logger_py.info('save time profile to {}'.format(os.path.join(out_dir,'tp_eval_incre.dmp')))
    pr.dump_stats(os.path.join(out_dir,'tp_eval_incre.dmp'))
    
    '''log'''
    # ignore_missing=cfg.eval.ignore_missing
    prefix='incre_inst' if not cfg.eval.ignore_missing else 'incre_inst_ignore'
    
    eval_tool_upperbound.write(out_dir,'incre_upper_bound')
    
    for eval_type, eval_tool in eval_tools.items():
        print('======={}======'.format(eval_type))
        print(eval_tool.gen_text())
        _ = eval_tool.write(out_dir, eval_type+'_'+prefix)
    
    # if logger:
    #     for k,v in eval_dict['visualization'].items(): 
    #         logger.add_figure('test/'+prefix+'_'+k, v, global_step=it)
    #     for k, v in eval_dict.items():
    #         if isinstance(v,dict): continue
    #         logger.add_scalar('test/'+prefix+'_'+'%s' % k, v, it)


def parse():
    r"""loads model config

    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='./configs/default.yaml', help='configuration file name. Relative path under given path (default: config.yml)')
    args = parser.parse_args()
    
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise RuntimeError('Targer config file does not exist. {}' & config_path)
        
    # load config file
    config = codeLib.Config(config_path)
    # return config
    config.LOADBEST = True
    config.MODE = 'test'
    
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
        
    config.log_level = 'DEBUG'
    # logging.basicConfig(level=config.log_level)
    # logging.setLevel(config.log_level)
    return config

if __name__ == '__main__':
    # logger_py.setLevel('DEBUG')
    # logger_py.debug('hello0')
    main()