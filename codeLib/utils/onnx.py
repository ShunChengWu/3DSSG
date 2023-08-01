# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import onnxruntime as ort

def check(x,y):
    x = x if isinstance(x, list) or isinstance(x, tuple) else [x]
    y = y if isinstance(y, list) or isinstance(y, tuple) else [y]
    [np.testing.assert_allclose(x[i].flatten(), y[i].flatten(), rtol=1e-03, atol=1e-05) for i in range(len(x))]
        
def export(model:torch.nn.Module, inputs:list,pth:str, input_names:list, output_names:list, dynamic_axes:dict):
    inputs = inputs if isinstance(inputs, list) or isinstance(inputs, tuple) else [inputs]
    torch.onnx.export(model = model, args = tuple(inputs), f=pth,
              verbose=False,export_params=True,
              do_constant_folding=True,
              input_names=input_names, output_names=output_names,
              dynamic_axes=dynamic_axes,opset_version=12)
    with torch.no_grad():
        model.eval()
        sess = ort.InferenceSession(pth)
        x = model(*inputs)
        ins = {input_names[i]: inputs[i].numpy() for i in range(len(inputs))}
        y = sess.run(None, ins)
        check(x,y)
        
        inputs = [torch.cat([input,input],dim=0) for input in inputs]
        x = model(*inputs)
        ins = {input_names[i]: inputs[i].numpy() for i in range(len(inputs))}
        y = sess.run(None, ins)
        check(x,y)
        
        
def Linear_layer_wrapper(model:torch.nn.Linear,name, pth = './tmp', name_prefix=''):
    in_features = model.in_features
    out_features = model.out_features
    
    inputs = torch.rand(1, in_features)
    input_names = ['x']
    output_names = ['y']
    
    model(inputs)
                    
    dynamic_axes = {input_names[0]:{0:'n_node'}}
    export(model, inputs, os.path.join(pth, name), 
                        input_names=input_names, output_names=output_names, 
                        dynamic_axes = dynamic_axes)

    names = dict()
    name = name_prefix+'_'+name
    names['model_'+name] = dict()
    names['model_'+name]['path'] = name
    names['model_'+name]['input']=input_names
    names['model_'+name]['output']=output_names
    return names
    