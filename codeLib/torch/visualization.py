#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 18:47:46 2021

@author: sc
"""
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def show_tv_grid(imgs):
    '''
    Take the output of torchvision.utils.make_grid. 
    return a plt image
    '''
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return fix
    # plt.show()

def show_tensor_images(imgs, title=None):
    grid = torchvision.utils.make_grid(imgs)
    if not isinstance(grid, list):
        grid = [grid]
    
    fix, axs = plt.subplots(ncols=len(grid), squeeze=False)
    for i, img in enumerate(grid):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()

def save_tensor_images(imgs, path, title=None):
    grid = torchvision.utils.make_grid(imgs)
    if not isinstance(grid, list):
        grid = [grid]
    
    fix, axs = plt.subplots(ncols=len(grid), squeeze=False)
    for i, img in enumerate(grid):
        img = img.detach()
        img = torchvision.transforms.functional.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if title is not None:
        plt.title(title)
    fix.savefig(path)
    plt.close()