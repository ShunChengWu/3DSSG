import os,json
import random,torch
import numpy as np
    
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def check_file_exist(path):
    if not os.path.exists(path):
            raise RuntimeError('Cannot open file. (',path,')')

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def read_label_files(file, split=' '):
    '''
    file format should be 
    idx1 label1
    idx2 label2
    
    can also be:
    idx1: label1
    
    specify the split
    '''
    lines = read_txt_to_list(file)
    out = dict()
    for line in lines:
        s = line.split(split)
        out[int(s[0])] = s[1]
    return out

def load_semseg(json_file, name_mapping_dict=None, mapping = True):    
    '''
    Create a dict that maps instance id to label name.
    If name_mapping_dict is given, the label name will be mapped to a corresponding name.
    If there is no such a key exist in name_mapping_dict, the label name will be set to '-'

    Parameters
    ----------
    json_file : str
        The path to semseg.json file
    name_mapping_dict : dict, optional
        Map label name to its corresponding name. The default is None.
    mapping : bool, optional
        Use name_mapping_dict as name_mapping or name filtering.
        if false, the query name not in the name_mapping_dict will be set to '-'
    Returns
    -------
    instance2labelName : dict
        Map instance id to label name.

    '''
    instance2labelName = {}
    with open(json_file, "r") as read_file:
        data = json.load(read_file)
        for segGroups in data['segGroups']:
            # print('id:',segGroups["id"],'label', segGroups["label"])
            # if segGroups["label"] == "remove":continue
            labelName = segGroups["label"]
            if name_mapping_dict is not None:
                if mapping:
                    if not labelName in name_mapping_dict:
                        labelName = 'none'
                    else:
                        labelName = name_mapping_dict[labelName]
                else:
                    if not labelName in name_mapping_dict.values():
                        labelName = 'none'

            instance2labelName[segGroups["id"]] = labelName.lower()#segGroups["label"].lower()
    return instance2labelName


def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.

    Args:
        x (tensor): input images
    '''
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x

def pytorch_count_params(model, trainable=True):
    "count number trainable parameters in a pytorch model"
    s = 0
    for p in model.parameters():
        if trainable:
            if not p.requires_grad: continue
        try:
            s += p.numel()
        except:
            pass
    return s
