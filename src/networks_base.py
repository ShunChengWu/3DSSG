if __name__ == '__main__' and __package__ is None:
    from os import sys#, path
    sys.path.append('../')

import torch.nn as nn
class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
    def init_weights(self, init_type='normal', gain=0.02, bias_value=0.0,
                     target_op = None):
        '''
        initialize network's weights
        init_type: normal | xavier_normal | kaiming | orthogonal | xavier_unifrom
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        
        def init_func(m):
            classname = m.__class__.__name__
                    
            if target_op is not None:
                if classname.find(target_op) == -1:
                    return False
                
            if hasattr(m, 'param_inited'):
                return 
                
            # print('classname',classname)    
            if hasattr(m, 'weight'):# and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_unifrom':
                    nn.init.xavier_uniform_(m.weight.data, gain=gain)
                elif init_type == 'constant':
                    nn.init.constant_(m.weight.data, gain)
                else:
                    raise NotImplementedError()

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, bias_value)
            m.param_inited = True
        self.init_apply(init_func)
        
    def getParamList(self,x):
        return list(x.parameters())
    def init_apply(self, fn):
        for m in self.children():
            if hasattr(m, 'param_inited'):
                if m.param_inited is False:
                    m.init_apply(fn)
            else:
                m.apply(fn)    
        fn(self)
        return self
    
class mySequential(nn.Sequential, BaseNetwork):
    def __init__(self, *args):
        super(mySequential, self).__init__(*args)
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
if __name__ == "__main__":
    pass