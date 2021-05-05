#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
if __name__ == '__main__':
    os.sys.path.append('./src')
from src.SceneGraphFusionNetwork import SGFN
from src.config import Config
from utils import util
import torch
import argparse

def main():
    config = load_config()
    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    
    util.set_random_seed(config.SEED)
    
    # init device
    if torch.cuda.is_available() and len(config.GPU) > 0:
        config.DEVICE = torch.device("cuda")
    else:
        config.DEVICE = torch.device("cpu")
        
    if config.MODE == 'trace':
        config.DEVICE = torch.device("cpu")    
    if config.VERBOSE:
        print(config)
    
    if config.MODE != 'train':
        config.dataset.load_cache=False
    
    model = SGFN(config)                         
    
    if config.MODE == 'train':
        try:
            model.load()
        except:
            print('unable to load previous model.')
        print('\nstart training...\n')
        model.train()
    if config.MODE == 'eval':
        if model.load(config.LOADBEST) is False:
            raise Exception('\nCannot find saved model!\n')
        model.eval()
    if config.MODE == 'trace':
        if model.load(config.LOADBEST) is False:
            raise Exception('\nCannot find saved model!\n')
        model.trace()
        print('done')
    
def load_config():
    r"""loads model config

    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config_example.json', help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('--loadbest', type=int, default=0,choices=[0,1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')
    parser.add_argument('--mode', type=str, choices=['train','trace','eval'], help='mode. can be [train,trace,eval]',required=True)

    args = parser.parse_args()
    config_path = os.path.abspath(args.config)

    if not os.path.exists(config_path):
        raise RuntimeError('Targer config file does not exist. {}' & config_path)
    
    # load config file
    config = Config(config_path)
    
    if 'NAME' not in config:
        config_name = os.path.basename(args.config)
        if len(config_name) > len('config_'):
            name = config_name[len('config_'):]
            name = os.path.splitext(name)[0]
            translation_table = dict.fromkeys(map(ord, '!@#$'), None)
            name = name.translate(translation_table)
            config['NAME'] = name            
    config.LOADBEST = args.loadbest
    config.MODE = args.mode

    return config
if __name__ == '__main__':
    main()