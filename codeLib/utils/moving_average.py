#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:05:47 2021

@author: sc
"""
from codeLib.common import filter_args_create
__all__ = ['Identity','MA','EMA']
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

class Identity(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.count=0
        self.avg=0
    def __call__(self,x):
        self.avg=x
        self.count+=1
        return x
    def state_dict(self):
        return {'count':self.count,
                'avg':self.avg}
    def load_state_dict(self, sd):
        self.count=sd['count']
        self.avg=sd['avg']

class MA(object):
    def __init__(self):
        '''
        moving average
        '''
        self.reset()
    def __call__(self,v):
        return self.update(v)
    def update(self, v):
        self.count+=1
        self.avg+=(v-self.avg)/(self.count)
        return self.avg
    def reset(self):
        self.count=0
        self.avg=0
    def __repr__(self):
        return str(self.avg)
    def __str__(self):
        return str(self.avg)
    def state_dict(self):
        return {'count':self.count,
                'avg':self.avg}
    def load_state_dict(self, sd):
        self.count=sd['count']
        self.avg=sd['avg']
    
    
class EMA(object):
    def __init__(self, alpha:float=0.8,correction:bool=True):
        '''
        Exponential moving average

        Parameters
        ----------
        alpha : float, optional
            The importanese of the past. The default is 0.5.
        correction : bool, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        '''
        self.alpha=alpha
        self.correction=correction
        self.reset()
    def reset(self):
        self.count=0
        self.avg=0
        self.alphaExp=1
    def __call__(self,v):
        return self.update(v)
    def update(self,v):
        self.count+=1
        self.avg=self.alpha*self.avg+(1-self.alpha)*v
        if self.correction and self.alpha<1.0:
            self.alphaExp*=self.alpha
            return self.avg / (1-self.alphaExp)
        else:
            return self.avg
    def count(self):
        return self.count
    def value(self):
        return self.avg
    def __repr__(self):
        return str(self.avg)
    def __str__(self):
        return str(self.avg)
    def state_dict(self):
        return {'count':self.count,
                'avg':self.avg,
                'alpha':self.alpha,
                'alphaExp':self.alphaExp,
                'correction':self.correction,
                }
    def load_state_dict(self, sd):
        self.count=sd['count']
        self.avg=sd['avg']
        self.alpha=sd['alpha']
        self.alphaExp=sd['alphaExp']
        self.correction=sd['correction']

smoother_dict = {
    'none': Identity,
    'ma':MA,
    'ema':EMA
    }
def get_smoother(method,**args):
    smoother = smoother_dict[method.lower()]
    return filter_args_create(smoother,args)#  smoother(args)