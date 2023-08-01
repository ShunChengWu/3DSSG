#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 10:46:16 2021

@author: sc
"""
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict

class TBLogger:
    """
    Wandb logger class to monitor training.

    Parameters
    ----------
    name : str
        Run name (if empty, uses a fancy Wandb name, highly recommended)
    dir : str
        Folder where wandb information is stored
    id : str
        ID for the run
    anonymous : bool
        Anonymous mode
    version : str
        Run version
    project : str
        Wandb project where the run will live
    tags : list of str
        List of tags to append to the run
    log_model : bool
        Log the model to wandb or not
    experiment : wandb
        Wandb experiment
    entity : str
        Wandb entity
    """
    def __init__(self, cfg, log_dir,
                 ):
        self.logger = SummaryWriter(log_dir)
        self._metrics = OrderedDict()
        self._figures = OrderedDict()
    def add_scalar(self, name, value):
        self._metrics.update({name: value})
        
    def add_figure(self,name,image):
        self._figures.update({name: image})
        
    def commit(self, step:int):
        for k,v in self._metrics.items():
            self.logger.add_scalar(k, v,global_step=step)
        self._metrics.clear()
        
        for k,v in self._figures.items():
            self.logger.add_figure(k, v,global_step=step)
        self._figures.clear()