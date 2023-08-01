import argparse
import os
import codeLib
import torch
from .training import Trainer
from . import dataset
from .sgfn import SGFN
from .sgpn import SGPN
from .jointSG import JointSG
from .imp import IMP

__all__ = ['SGFN', 'SGPN', 'dataset',
           'Trainer', 'IMP', 'JointSG']


def default_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='./configs/config_default.yaml',
                        help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'validation', 'trace', 'eval',
                        'sample', 'trace'], default='train', help='mode. can be [train,trace,eval]', required=False)
    parser.add_argument('--loadbest', type=int, default=0, choices=[
                        0, 1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')
    parser.add_argument('--log', type=str, default='DEBUG',
                        choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL'], help='')
    parser.add_argument('-o', '--out_dir', type=str, default='',
                        help='overwrite output directory given in the config file.')
    parser.add_argument('--dry_run', action='store_true',
                        help='disable logging in wandb (if that is the logger).')
    parser.add_argument('--cache', action='store_true',
                        help='load data to RAM.')
    return parser


def load_config(args):
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise RuntimeError(
            'Targer config file does not exist. {}'.format(config_path))

    # load config file
    config = codeLib.Config(config_path)
    # configure config based on the input arguments
    config.config_path = config_path
    config.LOADBEST = args.loadbest
    config.MODE = args.mode
    if len(args.out_dir) > 0:
        config.training.out_dir = args.out_dir
    if args.dry_run:
        config.wandb.dry_run = True
    if args.cache:
        config.data.load_cache = True

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
    return config


def Parse():
    r"""loads model config

    """
    args = default_parser().parse_args()
    return load_config(args)
