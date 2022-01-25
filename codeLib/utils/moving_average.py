#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:05:47 2021

@author: sc
"""

# class SMA(object):
#     def __init__(self):
#         '''
#         Simple moving average
#         '''
#         self.count=0
#         self.avg=0
#     def update(self, v):
#         self.count+=1
#         self.avg+=(v-self.avg)/(self.count)
class CMA(object):
    def __init__(self):
        '''
        Cumulative moving average
        '''
        self.reset()
    def update(self, v):
        self.count+=1
        self.avg+=(v-self.avg)/(self.count)
    def reset(self):
        self.count=0
        self.avg=0
    def __repr__(self):
        return str(self.avg)
    def __str__(self):
        return str(self.avg)