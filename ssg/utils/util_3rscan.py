# -*- coding: utf-8 -*-
import json


def read_3rscan_info(pth):
    with open(pth, 'r') as f:
        lines = f.readlines()
        output = dict()
        for line in lines:
            split = line.rstrip().split(' = ')
            output[split[0]] = split[1]
    return output


def load_semseg(json_file, name_mapping_dict=None, mapping=True):
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

            # segGroups["label"].lower()
            instance2labelName[segGroups["id"]] = labelName.lower()
    return instance2labelName


def get_train_val_split(pth_3rscan_json):
    with open(pth_3rscan_json, 'r') as f:
        scan3r = json.load(f)

    train_list = list()
    val_list = list()
    for scan in scan3r:
        ref_id = scan['reference']

        if scan['type'] == 'train':
            l = train_list
        elif scan['type'] == 'validation':
            l = val_list
        else:
            continue
        l.append(ref_id)
        for sscan in scan['scans']:
            l.append(sscan['reference'])

    return train_list, val_list
