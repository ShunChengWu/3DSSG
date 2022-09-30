import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
import importlib

# try to import pytorch_geometric
has_pyg = importlib.util.find_spec('torch_geometric') is not None
if has_pyg:
    import torch_geometric

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
    
    def process_data_dict(self, data):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        if data is None: return data
        if has_pyg and isinstance(data,torch_geometric.data.hetero_data.HeteroData):
            return data.to(self._device)
        
        try:
            data =  dict(zip(data.keys(), self.toDevice(*data.values()) ))
        except:
            '''if failed, so more info'''
            print('')
            # print('type(data)',type(data))
            if not isinstance(data,dict):
                raise RuntimeError('expect input data with type dict but got {}'.format(type(data)))
            '''convert individually until error happen'''
            for k, v in data.items():
                try:
                    self.toDevice(v)
                except:
                    raise RuntimeError('unable to convert the object of {} with type {} to device {}'.format(k,type(v),self._device))
                
            
        return data
    
    def zero_metrics(self):
        '''reset metric accumulators'''
        raise NotImplementedError