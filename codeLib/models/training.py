import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
class BaseTrainer(object):
    '''
    Base trainer for all networks.
    '''
    def __init__(self, device):
        self._device=device
    
    def evaluate(self, val_loader):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        eval_list = defaultdict(list)

        for data in tqdm(val_loader):
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_list[k].append(v)

        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict
    
    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError
        
    def get_log_metrics(self):
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError
        
    def sample(self, *args, **kwargs):
        '''
        '''
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        ''' Performs  visualization.
        '''
        raise NotImplementedError
        
    def toDevice(self, *args):
        output = list()
        for item in args:
            if isinstance(item,  torch.Tensor):
                output.append(item.to(self._device,non_blocking=True))
            elif isinstance(item,  dict):
                ks = item.keys()
                vs = self.toDevice(*item.values())
                item = dict(zip(ks, vs))
                output.append(item)
            elif isinstance(item, list):
                output.append(self.toDevice(*item))
            else:
                output.append(item)
        return output
    
    def zero_metrics(self):
        '''reset metric accumulators'''
        raise NotImplementedError