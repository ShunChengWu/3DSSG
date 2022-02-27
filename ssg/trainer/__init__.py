#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 18:44:20 2021

@author: sc
"""
# from .trainer_2DSSG import Trainer_2DSSG
from .trainer_SGFN import Trainer_SGFN
# from .trainer_DesCmp import Trainer_DCMP
from .trainer_mvenc import Trainer_MVENC
from .trainer_svenc import Trainer_SVENC
from .trainer_IMP import Trainer_IMP
trainer_dict = {
    # 'ssg2d': Trainer_2DSSG,
    'sgfn': Trainer_SGFN,
    'sgpn': Trainer_SGFN,
    # 'dcmp': Trainer_DCMP,
    'mv': Trainer_MVENC,
    'sv' : Trainer_SVENC,
    'imp': Trainer_IMP,
    }