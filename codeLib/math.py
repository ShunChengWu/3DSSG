#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:20:41 2022

@author: sc
"""
import numpy as np
import math
def Euler_from_rotation(rotation:np.array):
    th_x = math.atan2(rotation[3,2], rotation[3,3])
    th_y = math.atan2(-rotation[2,0], math.sqrt(rotation[2,1]**2+rotation[2,2]**2))
    th_z = math.atan2(rotation[1,0],rotation[0,0])
    return [th_x,th_y,th_z]