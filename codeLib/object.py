#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 15:36:06 2021

@author: sc
"""


class BoundingBox(object):
    box=[0.0,0.0,0.0,0.0]
    def __init__(self,boundaries=[0.0,0.0,0.0,0.0]):
        if isinstance(boundaries, BoundingBox):
            self.box = boundaries.box
        self.box = boundaries
    def is_valid(self)->bool:
        return (self[2]>self[0]) and (self[3]>self[1])
    def tolist(self):
        return [self.box[0],self.box[1],self.box[2],self.box[3]]
    def size(self):
        return (self[2]-self[0])*(self[3]-self[1])
    def __call__(self):
        return self.box
    def __getitem__(self, key):
        return self.box[key]
    def x_min(self):
        return self.box[0]
    def y_min(self):
        return self.box[1]
    def x_max(self):
        return self.box[2]
    def y_max(self):
        return self.box[3]
    def width(self):
        return self[2]-self[0]
    def height(self):
        return self[3]-self[1]
    def get_intersection(self, box):
        box = BoundingBox(box)
        if self.x_min() > box.x_max() or box.x_min() > self.x_max() or \
        self.y_min() > box.y_max() or box.y_min() > self.y_max(): return BoundingBox()
        
        x_min = max(self[0], box[0])
        x_max = min(self[2], box[2])
        # if x_min>x_max:
        #     x_min,x_max = x_max,x_min
        y_min = max(self[1], box[1])
        y_max = min(self[3], box[3])
        # if y_min>y_max:
        #     y_min,y_max = y_max,y_min
        return BoundingBox([x_min,y_min,x_max,y_max])
    def get_iou(self,box):
        assert box.is_valid()
        assert self.is_valid()
        box = BoundingBox(box)
        a1 = self.size()
        a2 = box.size()
        inter = self.get_intersection(box).size()
        union = a1+a2-inter
        assert union>=inter
        assert union>=0
        return inter/union
    def __repr__(self):
        return '[x_min,y_min,x_max,y_max]: {}'.format(self.box)