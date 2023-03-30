if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')
    
import os
import json

class Config(dict):
    def __init__(self, config_path):
        super().__init__()
        if isinstance(config_path, str):
            with open(config_path, 'r') as f:
                d = Config(json.load(f))
                for k,v in d.items():
                    self[k]=v
                self['PATH'] = os.path.dirname(config_path)
                # check input fields
                self.check_keys(self)
                        
            try: 
                import torch
                if torch.cuda.is_available() and len(self['GPU']) > 0:
                    self['DEVICE'] = torch.device("cuda")
                else:
                    self['DEVICE'] = torch.device("cpu")
            except:
                pass
        elif isinstance(config_path, dict):
            for k,v in config_path.items():
                self[k]=v
        else:
            raise RuntimeError('input must be str or dict')
            
    def check_keys(self,d:dict()):
        for key,value in d.items():
            if isinstance(value, dict):
                self.check_keys(value)
            else:    
                comment_key = '_'+key
                if comment_key in d.keys():
                    if d[key] not in d[comment_key]:
                        raise RuntimeError('value for',key,'should be one of',d[comment_key],'got',d[key])
            if isinstance(value, dict):
                self[key]= Config(value)
    def __dir__(self):
        return self.keys()
                
    def __getattr__(self, name):
        if self.get(name) is not None:
            return self[name]
        raise RuntimeError('key',name,'is not defined!')

        return None
    
    def __setattr__(self, name, value):
        self[name]=value
    
    def __get__(self, instance, owner):
        print('get')
        return self.value
    def __set__(self, instance, value):
        print('hello')
        self.value = float(value)

    def get_format_str_from_dict(self,name,iitem, indent=0):
        text = str()
        if isinstance(iitem, dict):
            text += 'dict:' + name + '\n'
            for key,value in iitem.items():
                text += self.get_format_str_from_dict(key,value,indent+2)
        else:
            for i in range(indent): text += ' '
            text += '{} {}\n'.format(name,iitem)
        return text
            
            
    def __repr__(self):
        text = str()
        for key, item in self.items():
            text +=self.get_format_str_from_dict(key,item)    
        return text

if __name__ == '__main__':    
    config = Config('../config_example.json')
    print(config)