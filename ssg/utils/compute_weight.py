# -*- coding: utf-8 -*-
import numpy as np


def compute_weights(labels, per_class_count, normalize=False, verbose=False):
    if verbose:
        print("-------------")
    sum_weights = 0
    count = per_class_count.sum()
    weights = list()
    for c in range(len(per_class_count)):
        if per_class_count[c] > 0:
            weight = count / per_class_count[c]
            sum_weights += weight
            weights.append(weight)
        else:
            weights.append(0)

    if normalize:
        if verbose:
            print('normalize weight')
        for c in range(len(weights)):
            weights[c] = weights[c] / sum_weights * len(labels)

    weights = np.array(weights)
    # print('weights:',weights)
    # print('weights[weights!=0].min():', weights[weights>0].min())
    weights[weights == 0] = weights[weights > 0].min()
    if verbose:
        for c in range(len(weights)):
            weight = weights[c]
            print('{0:>20s} {1:>1.3f} {2:>6d}'.format(
                labels[c], weight, int(per_class_count[c])))
        print("-------------")
    return weights


def compute_weights_forBCE(labels, per_class_count, num_samples, normalize=False, verbose=False):
    if verbose:
        print("-------------")
    weights = list()
    for c in range(len(per_class_count)):
        pos_samples = per_class_count[c]
        if pos_samples > 0:
            # pos_weight = negative_samples/positive_samples
            neg_samples = num_samples - pos_samples
            weight = neg_samples / pos_samples
            weights.append(weight)
        else:
            weights.append(0)

    if normalize:
        if verbose:
            print('normalize weight')
        for c in range(len(weights)):
            weights[c] = weights[c] / max(weights) * len(labels)

    weights = np.array(weights)
    weights[weights == 0] = weights[weights > 0].min()
    if verbose:
        for c in range(len(weights)):
            weight = weights[c]
            print('{0:>20s} {1:>1.3f} {2:>6d}'.format(
                labels[c], weight, int(per_class_count[c])))
        print("-------------")
    return weights


def compute(classNames, relationNames, relationship_data, selections: list = None,
            normalize=False, for_BCE=False, edge_mode: str = 'gt', verbose=False):
    '''

    Parameters
    ----------
    classNames : TYPE
        DESCRIPTION.
    relationNames : TYPE
        DESCRIPTION.
    relationship_data : TYPE
        DESCRIPTION.
    selections : list, optional
        DESCRIPTION. The default is None.
    normalize : TYPE, optional
        DESCRIPTION. The default is False.
    for_BCE : TYPE, optional
        DESCRIPTION. The default is False.
    edge_mode : str, optional
        Edge mode affect the total number of edges during training. When using BCE 
        loss, edge mode affect the weight on each class dramatically. 
        Can be ['gt','fully_connected','nn'].
        when use 'nn', need to pass object neighbors.
         The default is 'gt'.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    RuntimeError
        DESCRIPTION.

    Returns
    -------
    wobjs : TYPE
        DESCRIPTION.
    wrels : TYPE
        DESCRIPTION.
    o_obj_cls : TYPE
        DESCRIPTION.
    o_rel_cls : TYPE
        DESCRIPTION.

    '''
    o_rel_cls = np.zeros((len(relationNames)))
    o_obj_cls = np.zeros((len(classNames)))

    assert edge_mode in ['gt', 'fully_connected', 'nn']
    if edge_mode == 'nn':
        print('edge_mode nn. is not calculated correctly! \
              The current method count all the NN, but it may have duplicate \
                  (i.e. 1 is neighobr of 2 but 2 is also neighbor of 1')

    exceed_ids = dict()
    scene_analysis = dict()
    n_edge_with_gt = 0
    n_edge_nn = 0
    n_edges_fully_connected = 0

    for scan_id, scan in relationship_data.items():

        if selections is not None:
            if scan_id not in selections:
                continue
        instance2LabelName = {}

        n_obj = len(scan["objects"])
        for k, v in scan["objects"].items():
            n_edge_nn += len(v.get('neighbors', 0))
            label = v['label']
            instance2LabelName[int(k)] = label
            if label not in classNames:
                if verbose:
                    print(label, 'not in classNames')
            o_obj_cls[classNames.index(label)] += 1

        n_edges_fully_connected += n_obj*n_obj-n_obj

        nnk = dict()
        for relationship in scan["relationships"]:
            if relationship[3] not in relationNames:
                if verbose:
                    print(relationship[3], 'not in relationNames')
                continue

            obj = relationship[0]  # id object
            sub = relationship[1]  # id subject
            rel = relationship[2]  # id relationship

            if obj == 0 or sub == 0:
                raise RuntimeError('found obj or sub is 0')

            if not obj in instance2LabelName:
                RuntimeWarning('key not found:', obj)
                continue

            if not sub in instance2LabelName:
                RuntimeWarning('key not found:', sub)
                continue

            if relationNames.index(relationship[3]) >= len(relationNames):
                if rel not in exceed_ids:
                    exceed_ids[relationNames.index(relationship[3])] = 0
                else:
                    exceed_ids[relationNames.index(relationship[3])] += 1
                continue
            o_rel_cls[relationNames.index(relationship[3])] += 1
            # rel_cls_count += 1

            nn = str(obj)+'_'+str(sub)
            if nn not in nnk:
                nnk[nn] = 0
            nnk[str(obj)+'_'+str(sub)] += 1
        for v in nnk.values():
            if v > 1:
                n_edge_with_gt += 1

        scene_analysis[scan_id] = dict()
        scene_analysis[scan_id]['num objects'] = len(scan["objects"].items())
        scene_analysis[scan_id]['num relationships'] = len(
            scan['relationships'])
    if verbose:
        print('num multi predicates:', n_edge_with_gt)

    if len(exceed_ids) > 1:
        print('exceed map')
        for id, count in exceed_ids.items():
            print('\t', id, count)

    if verbose:
        print("objects:")
    wobjs = compute_weights(classNames, o_obj_cls,
                            normalize=normalize, verbose=verbose)
    if verbose:
        print("relationships:")
    if for_BCE:
        if verbose:
            print('use bce weighting for relationships')
        total_num_edges = edge_mode
        if edge_mode == 'gt':
            total_num_edges = n_edge_with_gt
        elif edge_mode == 'fully_connected':
            total_num_edges = n_edges_fully_connected
        elif edge_mode == 'nn':
            total_num_edges = n_edge_nn
        wrels = compute_weights_forBCE(
            relationNames, o_rel_cls, total_num_edges, normalize=normalize, verbose=verbose)
    else:
        wrels = compute_weights(relationNames, o_rel_cls,
                                normalize=normalize, verbose=verbose)
    return wobjs, wrels, o_obj_cls, o_rel_cls


def compute_sgfn(classNames, relationNames, relationship_data, selections: list = None,
                 normalize=False, for_BCE=False, edge_mode: str = 'gt', none_index: int = None, verbose=False):
    if relationNames is not None:
        o_rel_cls = np.zeros((len(relationNames)))
    else:
        o_rel_cls = None
    o_obj_cls = np.zeros((len(classNames)))

    assert edge_mode in ['gt', 'fully_connected', 'nn']
    if edge_mode == 'nn':
        print('edge_mode nn. is not calculated correctly! \
              The current method count all the NN, but it may have duplicate \
                  (i.e. 1 is neighobr of 2 but 2 is also neighbor of 1')

    # classes_count = 0
    # counter = 0

    exceed_ids = dict()
    scene_analysis = dict()
    n_edge_with_gt = 0
    n_edge_nn = 0
    n_edges_fully_connected = 0

    for scan_id, scan_data in relationship_data.items():
        for node_id, node_data in scan_data['nodes'].items():
            n_edge_nn += len(node_data['neighbors'])

    # for scan_id, v in relationship_data['neighbors'].items():
    #     for node_idx, nns in v.items():
    #         n_edge_nn += len(nns)

    # cnn=0
    for scan_id, scan_data in relationship_data.items():
        if selections is not None:
            if scan_id not in selections:
                continue
        instance2LabelName = {}

        nodes = scan_data['nodes']
        if relationNames is not None:
            relationships = scan_data['relationships']

        n_obj = 0
        for k, v in nodes.items():
            obj_label = v['label']
            instance2LabelName[int(k)] = obj_label
            if obj_label not in classNames:
                if verbose:
                    print(obj_label, 'not in classNames')
                continue
            o_obj_cls[classNames.index(obj_label)] += 1
            n_obj += 1

        n_edges_fully_connected += n_obj*n_obj-n_obj

        if relationNames is not None:
            nnk = dict()
            n_rel = 0
            for relationship in relationships:
                obj = int(relationship[0])
                sub = int(relationship[1])
                rel = int(relationship[2])
                name = relationship[3]
                if name not in relationNames:
                    if verbose:
                        print(relationship[3], 'not in relationNames')
                    continue

                if obj == 0 or sub == 0:
                    raise RuntimeError('found obj or sub is 0')

                if not obj in instance2LabelName:
                    RuntimeWarning('key not found:', obj)
                    continue

                if not sub in instance2LabelName:
                    RuntimeWarning('key not found:', sub)
                    continue

                if relationNames.index(name) >= len(relationNames):
                    if rel not in exceed_ids:
                        exceed_ids[relationNames.index(name)] = 0
                    else:
                        exceed_ids[relationNames.index(name)] += 1
                    continue
                o_rel_cls[relationNames.index(name)] += 1
                # classes_count += 1

                nn = str(obj)+'_'+str(sub)
                if nn not in nnk:
                    nnk[nn] = 0
                nnk[str(obj)+'_'+str(sub)] += 1
                n_rel += 1

            n_edge_with_gt = len(nnk)
        # for v in nnk.values():
        #     if v > 1:
        #         n_edge_with_gt+=1

        # counter += 1

        scene_analysis[scan_id] = dict()
        scene_analysis[scan_id]['num objects'] = n_obj
        if relationNames is not None:
            scene_analysis[scan_id]['num relationships'] = n_rel
    if verbose:
        print('num multi predicates:', n_edge_with_gt)

    if len(exceed_ids) > 1:
        print('exceed map')
        for id, count in exceed_ids.items():
            print('\t', id, count)

    if verbose:
        print("objects:")
    wobjs = compute_weights(classNames, o_obj_cls,
                            normalize=normalize, verbose=verbose)
    if relationNames is not None:
        if verbose:
            print("relationships:")
        if for_BCE:
            if verbose:
                print('use bce weighting for relationships')
            total_num_edges = edge_mode
            if edge_mode == 'gt':
                total_num_edges = n_edge_with_gt
            elif edge_mode == 'fully_connected':
                total_num_edges = n_edges_fully_connected
            elif edge_mode == 'nn':
                total_num_edges = n_edge_nn
            if verbose:
                print('edge mode:', edge_mode)
            wrels = compute_weights_forBCE(
                relationNames, o_rel_cls, total_num_edges, normalize=normalize, verbose=verbose)
        else:
            # if none_index is not None:
            #     o_rel_cls[none_index] = n_edges_fully_connected-o_rel_cls.sum()
            wrels = compute_weights(
                relationNames, o_rel_cls, normalize=normalize, verbose=verbose)
    else:
        wrels = None
    # wrels = compute_weights(relationNames, o_rel_cls, classes_count,verbose)
    return wobjs, wrels, o_obj_cls, o_rel_cls
