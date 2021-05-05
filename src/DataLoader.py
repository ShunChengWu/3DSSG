import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
import numpy as np

class CustomSingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):
    def __init__(self,loader):
        super().__init__(loader)
    def IndexIter(self):
        return self._sampler_iter
    
class CustomMultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):
    def __init__(self,loader):
        super().__init__(loader)
    def IndexIter(self):
        return self._sampler_iter


class CustomDataLoader(DataLoader):
    def __init__(self, config, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        if worker_init_fn is None:
            worker_init_fn = self.init_fn
        super().__init__(dataset, batch_size, shuffle, sampler,
                 batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context)
        self.config = config
        
    def init_fn(self, worker_id):
        np.random.seed(self.config.SEED + worker_id)
        
    def __iter__(self):
        if self.num_workers == 0:
            return CustomSingleProcessDataLoaderIter(self)
        else:
            return CustomMultiProcessingDataLoaderIter(self)