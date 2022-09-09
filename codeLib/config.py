import os
import yaml
from copy import copy, deepcopy
# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.Loader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg

def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v
            
class Config(dict):
    def __init__(self, config_path=None):
        super().__init__()
        if isinstance(config_path, str):
            config = load_config(config_path)
            for k,v in config.items():
                self[k]=self.check_value(v)
            # self.check_keys(self)
            
        elif isinstance(config_path, dict):
            for k,v in config_path.items():
                self[k]=self.check_value(v)
            # raise RuntimeError("wrong!")
            # for k,v in config_path.items():
            #     self[k]= v
        self.config_path=config_path
    def check_value(self,v):
        if isinstance(v, dict):
            tmp = Config()
            for key,value in v.items():
                tmp[key] = self.check_value(value)
            return tmp
        else:
            return v
                
    # def check_keys(self,d:dict()):
    #     tmp = Config()
    #     for key,value in d.items():
    #         if isinstance(value, dict):
    #             tmp[key] = Config(value)
    #             tmp.check_keys(value)

    #     # for key,value in tmp.items():
    #     #     if isinstance(d[key], dict):
    #     #         # tmp[key] = Config(value)
    #     #         self[key]= Config(d[key])
    #     for k,v in tmp.items():
    #         self[k] = v
    
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.items():
            setattr(result, k, deepcopy(v, memo))
        return result
    
    # def __copy__(self):
    #     return Config(self)
    # def __deepcopy__(self, memo):
    #     return Config(self) # not sure about this
    
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
            text += '{}: {}\n'.format(name,iitem)
        return text
    def __repr__(self):
        text = str()
        for key, item in self.items():
            text +=self.get_format_str_from_dict(key,item)    
        return text    

if __name__ == '__main__':    
    config = Config('../configs/default.yaml')
    print(config)