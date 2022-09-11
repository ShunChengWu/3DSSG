# from . import define
import argparse,os, codeLib,torch
from .training import Trainer
# from .models import *
# from .objects import *
# from .dataset import dataset_dict
# from .utils import *
from . import dataset
from .ssg3d import SSG3D
from .sgfn import SGFN
from .sgpn import SGPN
from .mvenc import MVEnc
from .svenc import SVEnc
from .imp import IMP #iterative message passing
# from .destcmp import DestCmp
# from . import dataset
# # from . import config as config


__all__ = ['SSG3D','SGFN', 'SGPN','dataset','Trainer','MVEnc','SVEnc','IMP']
# __all__ = ['define', 'Trainer', 'dataset_dict', 'SSG2D','SGFN',
#            'DestCmp']

def Parse():
    r"""loads model config

    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='./configs/default.yaml', help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('--mode', type=str, choices=['train','trace','eval','sample','trace'], default='train', help='mode. can be [train,trace,eval]',required=False)
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