#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:26:58 2021

@author: sc
"""

import torch
import re
import collections
from torch._six import string_classes
# from torch.utils.data.dataloader import default_collate

int_classes=int
# string_classes=str

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)

        # for x in batch:
        #     print(x.shape)
        return torch.cat(batch,0,out=out)
        # return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
            
def graph_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    out = dict()
    
    
    n_nodes_acc=0
    n_images_acc=0
    instance2mask=list()
    image_boxes=list()
    roi_imgs = list()
    for b in batch:
        if 'node_edges' in b:
            x = b['node_edges']
            x += n_nodes_acc
        
        if 'image_edges' in b:
            x = b['image_edges']
            x[:,0] += n_nodes_acc
            x[:,1] += n_images_acc
        
        # TODO: need to change evaluation tool for this
        if 'instance2mask' in b:
            x = b['instance2mask']
            instance2mask.append( { inst: v+n_nodes_acc for inst, v in x.items() } )
        # b['instance2mask'] = x
        
        if 'image_boxes' in b:
            x = b['image_boxes']
            x[:,0] += n_nodes_acc
            # image_boxes += [{ k+n_images_acc: v for k,v in xx.items() } for xx in x]
        # b['image_boxes'] = [{ k+n_images_acc: v for k,v in xx.items() } for xx in x]
            
        if 'gt_cls' in b:
            n_nodes = len(b['gt_cls'])
            n_nodes_acc += n_nodes
        if 'images' in b:
            n_images = len(b['images'])
            n_images_acc += n_images
            
        if 'roi_imgs' in b:
            roi_imgs += b['roi_imgs']
    
    if len(image_boxes)>0:
        out['image_boxes'] = image_boxes
    if len(instance2mask)>0:
        out['instance2mask'] = instance2mask
    if len(roi_imgs)>0:
        out['roi_imgs'] = roi_imgs
        
    for x in ['node_edges','image_edges']:
        if x in elem:
            out[x] = torch.cat([d[x] for d in batch])
        
    
    for x in ['scan_id','gt_rel','gt_cls','images','descriptor','node_descriptor_8', 'obj_points', 'rel_points']:
        if x in elem:
            # print('===')
            # print(x)
            out[x] = collate([d[x] for d in batch])
        
    return out
    # {key: graph_collate([d[key] for d in batch]) for key in elem}
    
    # output['scan_id'] = scan_id # str
    # output['gt_rel'] = gt_rels  # tensor
    # output['gt_cls'] = gt_class # tensor
    # output['images'] = images# tensor
    # output['descriptor'] = descriptor #tensor
    
    # output['node_edges'] = edge_indices # tensor
    # output['image_edges'] = image_indices # tensor
    
    # output['instance2mask'] = instance2mask #dict
    # output['image_boxes'] = bounding_boxes #list

    raise TypeError(default_collate_err_msg_format.format(elem_type))
    
def batch_graph_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    out = list()
    for b in batch:
        out.append(b)
    return out
    
if __name__ == '__main__':
    import ssg2d
    import codeLib
    
    config = codeLib.Config('../../configs/default.yaml')
    config.path = '/home/sc/research/PersistentSLAM/python/2DTSG/data/test/'
    config.DEVICE='cpu'
    dataset = ssg2d.config.get_dataset(config,'train')
    
    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, num_workers=0, shuffle=True,
            pin_memory=False,
            collate_fn=graph_collate,
        )
    for batch in train_loader:
        print(batch)
        break