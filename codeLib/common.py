# -*- coding: utf-8 -*-
import math
import numpy as np
import subprocess
import torch
import torch.nn as nn
import logging
import pickle
from inspect import signature
from pathlib import Path
logger_py = logging.getLogger(__name__)

def create_folder(path, parents=True, exist_ok=True):
    Path(path).mkdir(parents=parents, exist_ok=exist_ok)

def check_weights(params):
    ''' Checks weights for illegal values.

    Args:
        params (tensor): parameter tensor
    '''
    hasNan=False
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any():
                logger_py.error('NaN Values detected in model weight %s.' % k)
                hasNan=True
            if torch.isinf(v).any():
                logger_py.error('Nan Values detected in model weight %s.' % k)
                hasNan=True
        elif isinstance(v, np.ndarray):
            if np.isnan(v).any():
                logger_py.error('Inf Values detected in model weight %s.' % k)
                hasNan=True
            if np.isinf(v).any():
                logger_py.error('Inf Values detected in model weight %s.' % k)
                hasNan=True
    return hasNan

def check_valid(*xx):
    '''
    check if input is a valid value
    '''
    hasNan=False
    for x in xx:
        if isinstance(x, torch.Tensor):
            if torch.isnan(x).any():
                logger_py.error('NaN Values detected.')
                hasNan=True
            if torch.isinf(x).any():
                logger_py.error('Inf Values detected.')
                hasNan=True
        elif isinstance(x, dict):
            hasNan |= check_weights(x)
        elif isinstance(x, np.ndarray):
            if np.isnan(x).any():
                logger_py.error('NaN Values detected.')
                hasNan=True
            if np.isinf(x).any():
                logger_py.error('Inf Values detected')
                hasNan=True
        elif isinstance(x, list):
            hasNan |= check_valid(*x)
        elif isinstance(x, str):
            pass
        else:
            logger_py.error('unhandled type {}'.format(type(x).__name__))
            hasNan=True
    return hasNan

def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.

    Args:
        x (tensor): input images
    '''
    x = x.clone()
    if x.shape[0] == 3:
        x[0] = (x[0]-0.485)/0.229
        x[1] = (x[1]-0.456)/0.224
        x[2] = (x[2]-0.406)/0.225
    elif x.shape[1] == 3:
        x[:,0] = (x[:,0]-0.485)/0.229
        x[:,1] = (x[:,1]-0.456)/0.224
        x[:,2] = (x[:,2]-0.406)/0.225
    elif x.shape[2] == 3:
        x[:,:,0] = (x[:,:,0]-0.485)/0.229
        x[:,:,1] = (x[:,:,1]-0.456)/0.224
        x[:,:,2] = (x[:,:,2]-0.406)/0.225
    elif x.shape[3] == 3:
        x[:,:,:,0] = (x[:,:,:,0]-0.485)/0.229
        x[:,:,:,1] = (x[:,:,:,1]-0.456)/0.224
        x[:,:,:,2] = (x[:,:,:,2]-0.406)/0.225
    else:
        raise NotImplementedError()
    return x

def denormalize_imagenet(x):
    assert x.shape[1] == 3
    x = x.clone()
    x[:, 0] = x[:,0]*0.229+0.485
    x[:, 1] = x[:,1]*0.224+0.456
    x[:, 2] = x[:,2]*0.225+0.406
    return x

def save_obj(obj, file_path ):
    # if isinstance(obj, torch.Tensor):
    #     torch.save(obj,file_path)
    # else:
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def random_drop(x, p, replace:bool=False):
    '''
    random drop p percentage of the input list.
    if p is int and >0, select random p elements.
    if p is float/double, >0 and <=1.0, random select p percentage #in [1-p, 1.0]
    '''
    n_elms = len(x)
    if isinstance(p, list):
        # random select between given [low, high] bounds
        assert len(p)==2
        low = p[0]
        high = p[1]
        assert low < high
        
        if isinstance(low, int):
            if not replace:
                if low >= n_elms: return x
                if high > n_elms: high=n_elms
            select_p = np.random.choice(range(low,high),1)[0]
            choices  = np.random.choice(range(n_elms),select_p,replace=replace ).tolist()
            x = [x[y] for y in choices]
        else:
            assert low>=0
            assert high<=1
            percentage = np.random.uniform(low=low, high=high,size=1)[0]
            num_edge = int(float(len(x))*percentage//1)
            if num_edge == 0:
                num_edge = 1
            assert len(x)>0
            assert num_edge > 0
            choices = np.random.choice(range(len(x)),num_edge,replace=replace).tolist()
            x = [x[y] for y in choices]
    elif p > 0:
        if isinstance(p, int):
            if p < n_elms or replace: # if p >= n_elms, just return the same input
                choices = np.random.choice(range(n_elms),p,replace=replace ).tolist()
                x = [x[y] for y in choices]
        else:
            percentage = p# np.random.uniform(low=1-p, high=1.0,size=1)[0]
            num_edge = int(float(len(x))*percentage//1)
            if num_edge == 0:
                num_edge = 1
            assert len(x)>0
            assert num_edge > 0
            choices = np.random.choice(range(len(x)),num_edge,replace=replace).tolist()
            x = [x[y] for y in choices]
    return x

def convert_torch_to_scalar(o):
    if isinstance(o, dict):
        for k,v in o.items():
            o[k]=convert_torch_to_scalar(v)
        return o
    elif isinstance(o, torch.Tensor) and o.nelement() == 1:
        return o.item()
    else:
        return o
    
    
def filter_args(func, keys):
    """
    Filters a dictionary so it only contains keys that are arguments of a function
    Parameters
    ----------
    func : Function
        Function for which we are filtering the dictionary
    keys : dict
        Dictionary with keys we are filtering
    Returns
    -------
    filtered : dict
        Dictionary containing only keys that are arguments of func
    """
    filtered = {}
    sign = list(signature(func).parameters.keys())
    for k, v in {**keys}.items():
        if k in sign:
            filtered[k] = v
    return filtered


def filter_args_create(func, keys):
    """
    Filters a dictionary so it only contains keys that are arguments of a function
    and creates a function with those arguments
    Parameters
    ----------
    func : Function
        Function for which we are filtering the dictionary
    keys : dict
        Dictionary with keys we are filtering
    Returns
    -------
    func : Function
        Function with filtered keys as arguments
    """
    return func(**filter_args(func, keys))

def reset_parameters_with_activation(mm,nonlinearity):
    def process(m,nonlinearity):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5),nonlinearity=nonlinearity)
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5),nonlinearity=nonlinearity)
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)
        else:
            logger_py.error('unknown input type', m.__name__)
    
    if isinstance(mm , list):
        for m in mm: 
            process(m,nonlinearity)
    else:
        process(mm,nonlinearity)
    
        
def freeze(x:torch.nn.Module):
    for param in x.parameters(): param.requires_grad = False
        

def rand_24_bit():
    import random
    """Returns a random 24-bit integer"""
    return random.randrange(0, 16**6)
def color_dec():
    """Alias of rand_24 bit()"""
    return rand_24_bit()
def color_hex(num=rand_24_bit()):
    """Returns a 24-bit int in hex"""
    return "%06x" % num
def color_rgb(num=rand_24_bit()):
    """Returns three 8-bit numbers, one for each channel in RGB"""
    hx = color_hex(num)
    barr = bytearray.fromhex(hx)
    return (barr[0], barr[1], barr[2])
def rgb_2_hex(rgb):
    if isinstance(rgb,list):
        assert len(rgb) == 3
        rgb = (rgb[0],rgb[1],rgb[2])
    return '#%02x%02x%02x' % rgb
def color_hex_rgb(num=rand_24_bit()):
    rgb = color_rgb(num)
    """Returns a 24-bit int in hex"""
    return rgb_2_hex(rgb)



def execute(cmd,cwd):
    popen = subprocess.Popen(cmd,cwd=cwd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        print(cmd)
        raise subprocess.CalledProcessError(return_code, cmd)

def run(bashCommand,cwd='./',verbose:bool=False):
    for path in execute(bashCommand,cwd):
        if verbose:
            print(path, end="")
        else:
            pass