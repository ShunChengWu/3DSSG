#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 17:05:00 2020

@author: sc
"""
import numpy as np
from collections.abc import Iterable

class Element:
    def __init__(self, x, parent=None):
        self.idx = x
        self.parent = x if parent is None else parent
        self.childs = set([x])
    def __repr__(self):
        return "{} {}".format(self.idx, self.parent)
    
    def add_child(self, x):
        # self.childs.add(x.idx)
        self.childs = self.childs.union(x.childs)
    
class Disjset:
    def __init__(self):
        self.elements = dict()
        self.n_set = 0
        self.is_child = set()
    def add(self,x):
        if isinstance(x, list) or isinstance(x,Iterable):
            for xx in x:
                self.elements[xx] = Element(xx)
                self.n_set+=1
        else:
            self.elements[x] = Element(x)
            self.n_set+=1
        
    def union(self,x,y):
        set_x = self.find_set(x)
        set_y = self.find_set(y)
        if set_x == None or set_y == None:
            raise RuntimeError('')
        if set_x == set_y: return
        if set_x < set_y:
            self.elements[y].parent=set_x
            self.is_child.add(set_y)
            self.elements[set_x].add_child(self.elements[set_y])
        else:
            self.elements[x].parent=set_y
            self.is_child.add(set_x)
            self.elements[set_y].add_child(self.elements[set_x])
        self.n_set-=1
        
    def find_set(self,x) -> Element:
        if x in self.elements:
            if x == self.elements[x].parent:
                return x
            else:
                return self.find_set(self.elements[x].parent)
        else :
            raise RuntimeError('')
            
    def get_sets(self):
        return [x for x in self.elements.keys() if x not in self.is_child]
        
            
    def __str__(self):
        return 'size: ' + self.n_set + ", elements."
    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)
    
def merge(pd_instances, node_predictions:dict, edge_predictions:dict, same_part_name:str='same part'):
    unique_inst_pd_idx = np.unique(pd_instances)
    
    djset = Disjset()
    djset.add(unique_inst_pd_idx.tolist())
    for k,v in edge_predictions.items():
        if v == same_part_name:
            sp = k.split('_')
            source = int(sp[0])
            target = int(sp[1])
            # check whether label type are the same
            l_s = node_predictions[source]
            l_t = node_predictions[target]
            if l_s != l_t: continue # ignore
            djset.union(source, target)
            
    merged_inst_pd = [djset.find_set(pd_instances[i])  for i in range(len(pd_instances))]
    unique_merged_inst_pd = np.unique(merged_inst_pd).tolist()
    # diff = set(unique_inst_pd_idx.tolist()).difference(unique_merged_inst_pd)
    ''' map instance to continuous integers '''
    for i in range(len(merged_inst_pd)):
        merged_inst_pd[i] = unique_merged_inst_pd.index(merged_inst_pd[i])
        
        
def collect(node_predictions:dict, edge_predictions:dict, same_part_name:str='same part') ->dict:
    '''
    Find nodes that are connected with same part relationship

    Parameters
    ----------
    node_predictions : dict
        DESCRIPTION.
    edge_predictions : dict
        DESCRIPTION.
    same_part_name : str, optional
        DESCRIPTION. The default is 'same part'.

    Returns
    -------
    None.

    '''
    is_str = isinstance(next(node_predictions.keys().__iter__()),str)
    
    djset = Disjset()
    djset.add(node_predictions.keys())
    for k,v in edge_predictions.items():
        if v == same_part_name:
            sp = k.split('_')
            source, target = sp
            if not is_str:
                source = str(sp[0])
                target = str(sp[1])
            # check whether label type are the same
            l_s = node_predictions[source]
            l_t = node_predictions[target]
            if l_s != l_t: continue # ignore
            djset.union(source, target)
    
    sets = djset.get_sets()
    instance_groups = dict()
    for s in sets:
        instance_groups[s] = djset.elements[s].childs
    
    # for k in node_predictions.keys():
    #     inst = djset.find_set(k)
    #     if inst not in instance_groups:
    #         instance_groups[inst] = list()
    #     instance_groups[inst].append(k)
        
        
    return instance_groups
