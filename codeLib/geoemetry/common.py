#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:55:11 2022

@author: sc
"""
import trimesh
def create_box(dimensions, width = 0.01):
    lines = list()
    lines.append( trimesh.creation.box( (dimensions[0]+width,width,width) ).apply_translation(((0,-0.5*dimensions[1],-0.5*dimensions[2]))) )
    lines.append( trimesh.creation.box( (dimensions[0]+width,width,width) ).apply_translation(((0,-0.5*dimensions[1],0.5*dimensions[2]))) )
    lines.append( trimesh.creation.box( (dimensions[0]+width,width,width) ).apply_translation(((0,0.5*dimensions[1],-0.5*dimensions[2]))) )
    lines.append( trimesh.creation.box( (dimensions[0]+width,width,width) ).apply_translation(((0,0.5*dimensions[1],0.5*dimensions[2]))) )
    
    lines.append( trimesh.creation.box( (width,dimensions[1],width) ).apply_translation((-0.5*dimensions[0],0,-0.5*dimensions[2])) )
    lines.append( trimesh.creation.box( (width,dimensions[1],width) ).apply_translation((-0.5*dimensions[0],0,0.5*dimensions[2])) )
    lines.append( trimesh.creation.box( (width,dimensions[1],width) ).apply_translation((0.5*dimensions[0],0,-0.5*dimensions[2])) )
    lines.append( trimesh.creation.box( (width,dimensions[1],width) ).apply_translation((0.5*dimensions[0],0,0.5*dimensions[2])) )
    
    lines.append( trimesh.creation.box( (width,width,dimensions[2]) ).apply_translation((-0.5*dimensions[0],-0.5*dimensions[1],0)) )
    lines.append( trimesh.creation.box( (width,width,dimensions[2]) ).apply_translation((-0.5*dimensions[0],0.5*dimensions[1],0)) )
    lines.append( trimesh.creation.box( (width,width,dimensions[2]) ).apply_translation((0.5*dimensions[0],-0.5*dimensions[1],0)) )
    lines.append( trimesh.creation.box( (width,width,dimensions[2]) ).apply_translation((0.5*dimensions[0],0.5*dimensions[1],0)) )
    
    box = trimesh.util.concatenate(lines )
    return box