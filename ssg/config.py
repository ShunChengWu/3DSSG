import os
import copy
import logging
import ssg
from ssg import SGFN, SGPN, IMP, JointSG
from torch.utils.tensorboard import SummaryWriter
from codeLib.loggers import WandbLogger
from codeLib.common import filter_args_create
import torch.optim as optim
from ssg.trainer import trainer_dict
from copy import deepcopy

logger_py = logging.getLogger(__name__)

method_dict = {
    'sgfn': SGFN,
    'sgpn': SGPN,
    'imp': IMP,
    'jointsg': JointSG,
}

optimizer_dict = {
    'adam': optim.Adam,
    'adamw': optim.AdamW
}


def get_optimizer(cfg, training_parameters):
    optimizer_name = cfg.training.optimizer
    assert optimizer_name in ['adam', 'adamw']
    weight_decay = cfg.training.get('weight_decay', 0)
    opt = filter_args_create(optimizer_dict[optimizer_name],
                             {'params': training_parameters,
                              'lr': float(cfg.training.lr),
                              'amsgrad': cfg.training.amsgrad,
                              'weight_decay': weight_decay
                              })
    return opt


def get_schedular(cfg, optimizer, **args):
    method = cfg.training.scheduler.method.lower()
    tmp = deepcopy(cfg.training.scheduler)
    tmp['optimizer'] = optimizer
    tmp = {**tmp, **args, **cfg.training.scheduler.args}
    if method == 'multisteplr':
        return filter_args_create(optim.lr_scheduler.MultiStepLR, tmp)
    elif method == 'reduceluronplateau':
        return filter_args_create(optim.lr_scheduler.ReduceLROnPlateau, tmp)
    elif method == "none":
        return None


def get_dataset(cfg, mode='train'):
    # get multi_rel config from model and write it to data for convenient
    multi_rel = cfg.model.multi_rel
    cfg.data.multi_rel = multi_rel
    if 'num_points_union' in cfg.model:
        cfg.data.num_points_union = cfg.model.num_points_union
    if 'node_feature_dim' in cfg.model:
        cfg.data.node_feature_dim = cfg.model.node_feature_dim
    return ssg.dataset.dataset_dict[cfg.data.input_type](cfg, mode=mode)


def get_dataset_inst(cfg, mode='test'):
    tmp_cfg = copy.deepcopy(cfg)
    tmp_cfg.model.use_rgb = False
    tmp_cfg.model.use_normal = False
    tmp_cfg.data.input_type = 'sgfn'
    tmp_cfg.data.load_images = False
    tmp_cfg.data.load_points = False
    tmp_cfg.data.path = cfg.data.path_gt
    tmp_cfg.data.label_file = cfg.data.label_file_gt
    tmp_cfg.data.load_cache = False
    dataset_inst = get_dataset(tmp_cfg, mode)
    return dataset_inst


def get_model(cfg, num_obj_cls, num_rel_cls):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    if cfg.model.method == 'sgfn' or cfg.model.method == 'sgpn' or cfg.model.method == 'jointsg':
        return method_dict[cfg.model.method](
            cfg=cfg,
            num_obj_cls=num_obj_cls,
            num_rel_cls=num_rel_cls,
            device=cfg.DEVICE).to(cfg.DEVICE)
    elif cfg.model.method == 'mv':
        return method_dict[cfg.model.method](
            cfg=cfg,
            num_obj_cls=num_obj_cls,
            device=cfg.DEVICE).to(cfg.DEVICE)
    elif cfg.model.method == 'sv':
        return method_dict[cfg.model.method](
            cfg=cfg,
            num_obj_cls=num_obj_cls,
            device=cfg.DEVICE).to(cfg.DEVICE)
    elif cfg.model.method == 'imp':
        # logger_py.info('use IMP implementation')
        return method_dict[cfg.model.method](
            cfg=cfg,
            num_obj_cls=num_obj_cls,
            num_rel_cls=num_rel_cls,
            device=cfg.DEVICE).to(cfg.DEVICE)

    node_encoder = get_node_encoder(cfg, cfg.DEVICE)
    edge_encoder = get_edge_encoder(cfg, cfg.DEVICE)
    gnn = get_gnn(cfg, cfg.DEVICE)

    model = method_dict[cfg.model.method](
        cfg=cfg,
        num_obj_cls=num_obj_cls,
        num_rel_cls=num_rel_cls,
        node_encoder=node_encoder,
        edge_encoder=edge_encoder,
        gnn=gnn,
        device=cfg.DEVICE).to(cfg.DEVICE)

    return model


def get_node_encoder(cfg, device):
    return ssg.models.node_encoder_list[cfg.model.node_encoder.method](cfg, device).to(device)


def get_edge_encoder(cfg, device):
    return ssg.models.edge_encoder_list[cfg.model.edge_encoder.method](cfg, device).to(device)


def get_gnn(cfg, device):
    if cfg.model.gnn.method == 'none':
        return None
    return ssg.models.gnn_list[cfg.model.gnn.method](
        dim_node=cfg.model.node_feature_dim,
        dim_edge=cfg.model.edge_feature_dim,
        dim_atten=cfg.model.gnn.hidden_dim,
        num_layers=cfg.model.gnn.num_layers,
        num_heads=cfg.model.gnn.num_heads,
        aggr='max',
        DROP_OUT_ATTEN=cfg.model.gnn.drop_out
    )


def get_logger(cfg):
    method = cfg.logging.method.lower()
    name = cfg.name
    log_dir = os.path.join(cfg['training']['out_dir'], name, 'logs')

    if method == 'tensorboard':
        print('use logger:', method)
        return SummaryWriter(log_dir), cfg
    elif method == 'wandb':
        if cfg.wandb.dry_run is True:
            print('use logger: none (dry_run is true)')
            return None
        print('use logger:', method)

        log_dir = cfg.wandb.dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        cfg.wandb.name = cfg.wandb.id = name
        logger = filter_args_create(WandbLogger, {"cfg": cfg, **cfg.wandb})
        logger.log_config(cfg)
        cfg = logger.config
        return logger, cfg
    elif method == 'none':
        print('use logger: none')
        return None
    else:
        raise logger_py.error('unknown logger type.')


def get_trainer(cfg, model, node_cls_names, edge_cls_names, **kwargs):
    return trainer_dict[cfg.model.method](
        cfg=cfg, model=model, node_cls_names=node_cls_names, edge_cls_names=edge_cls_names,
        device=cfg.DEVICE,
        **kwargs)
