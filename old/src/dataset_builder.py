#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataset_SGPN import RIODatasetGraph
from dataset_SGFN import SGFNDataset

def build_dataset(config, split_type, shuffle_objs, multi_rel_outputs,
                  use_rgb, use_normal):
    if split_type != 'train_scans' and split_type != 'validation_scans' and split_type != 'test_scans':
        raise RuntimeError(split_type)
    if config.dataset.dataset_type  not in ['SGPN', "SGFN"]:
        raise RuntimeError('unknown dataset type.')
    
    if config.dataset.dataset_type == 'SGPN':
        # raise NotImplementedError('')
        if config.VERBOSE: print('build rio_graph dataset')
        dataset = RIODatasetGraph(config,
                          split=split_type,
                          multi_rel_outputs=multi_rel_outputs,
                          shuffle_objs=shuffle_objs,
                          use_rgb=use_rgb,
                          use_normal=use_normal,
                          load_cache=config.dataset.load_cache,
                          sample_in_runtime=config.dataset.sample_in_runtime,
                          for_eval = split_type == 'test_scans',
                          max_edges = config.dataset.max_edges)
    elif config.dataset.dataset_type == 'SGFN':
        if config.VERBOSE: print('build point_graph dataset')
        dataset = SGFNDataset(
            config,
            split=split_type,
            multi_rel_outputs=multi_rel_outputs,
            shuffle_objs=shuffle_objs,
            use_rgb = use_rgb,
            use_normal = use_normal,
            load_cache = config.dataset.load_cache,
            sample_in_runtime = config.dataset.sample_in_runtime,
            for_eval = split_type == 'test_scans',
            max_edges = config.dataset.max_edges,
            data_augmentation = config.dataset.data_augmentation,
            )
            
    return dataset


if __name__ == '__main__':
    from config import Config
    config = Config('../config_example.json')
    config.dataset.root = '../data/example_data'
    build_dataset(config, split_type = 'train_scans', shuffle_objs=True, multi_rel_outputs=False,use_rgb=True,use_normal=True)