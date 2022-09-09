#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:30:59 2022

@author: sc
"""
from torch.nn import Module

class BaseModel(Module):
    def __init__(self):
        super().__init__()
    def forward(self, **args):
        raise NotImplementedError()
    def calculate_metrics(self, **args):
        raise NotImplementedError()
    def trace(self, **args):
        raise NotImplementedError()
