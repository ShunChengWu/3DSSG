# -*- coding: utf-8 -*-
import torch
import numpy as np

def check(x,y):
    x = x if isinstance(x, list) or isinstance(x, tuple) else [x]
    y = y if isinstance(y, list) or isinstance(y, tuple) else [y]
    [np.testing.assert_allclose(x[i].flatten(), y[i].flatten(), rtol=1e-03, atol=1e-05) for i in range(len(x))]
        
def export(model:torch.nn.Module, inputs:list,pth:str, input_names:list, output_names:list, dynamic_axes:dict):
    import onnxruntime as ort
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