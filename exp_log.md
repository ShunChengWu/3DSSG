# SSG with GT Segmentation. Multiple predicate prediction. 160-26
## SGPN (3DSSG)
For `config_3DSSG_test_*.json`.
- 0~3: try to make it work
- 4: train with batchsize 4. ->doesn't train
- 5: train with batchsize 1. ->doesn't train
- 6: no GCN ->has better curve than 4,5 at the begining. The problem may be the number of GCN layers.
- 7: 2 layers GCN
- 8: 2 layers GCN. fix imp. with adding residual and aggr.
- 9: fix hidden layer size in gnn. fix network init
- 10: normalize edge weight
- 11: 5 layers
- 12: 5l. node pd. w/. gnn
- 13: node_dim: 256->512. num_points_union: 512->1024
- 14: ndoe_dim: 1024, num_points_union: 2048
- 15: test with the optimized version of the dataset -> can reproduce 14.

Note:
* 4 and 5 are used to see if batch process is working.
* 9: the final result is worse than the one in the paper. This may due to the weighting method. and the number of GNN iterations. -> the lr schedular dropping too fast

## SGFN
- 0: test. batchsize 0
- 1: batchsize 4

## 2DSSG
img_batchsize: 8
- exp0: MVCNN+VGG16. no GNN
- exp1: MVCNN+Res18. no gnn
- exp2: MVCNN+Res18. FAN -> forgot to enable gnn
- 2DSSG_exp2_1: MVCNN+Res18. FAN 2 layer GNN
- 2DSSG_exp_2_2: with 0.15 bounding box augmentation
- 2DSSG_exp3: MVCNN+Res18, FAN, 1 layer GNN
- 2DSSG_exp_2_3: with LR schedular setup1

#### relationships
| method             | R@50 | R@100 | R@5  | R@10 | R@3  | R@10 |
| ------------------ | ---- | ----- | ---- | ---- | ---- | ---- |
| 3DSSG_obj_wo_gcn   | 0.40 | 0.66  | 0.68 | 0.78 | 0.89 | 0.93 |
| 3DSSG_obj_from_gcn | 0.30 | 0.60  | 0.60 | 0.73 | 0.79 | 0.91 |
| exp0_9             | 0.75 | 0.77  | 0.49 | 0.64 | 0.36 | 0.51 |
| exp0_10            | 0.73 | 0.75  | 0.5  | 0.65 | 0.83 | 0.86 |
| exp0_12            | 0.75 | 0.77  | 0.57 | 0.70 | 0.81 | 0.92 |
| exp0_13            | 0.77 | 0.80  | 0.63 | 0.76 | 0.85 | 0.97 |
| exp0_14            | 0.72 | 0.79  | 0.60 | 0.74 | 0.83 | 0.95 |
| SGFN               | 0.85 | 0.87  | 0.7  | 0.8  | 0.97 | 0.99 |
| exp1_0             | 0.91 | 0.92  | 0.71 | 0.82 | 0.94 | 1.00 |
| exp1_1             | 0.90 | 0.91  | 0.69 | 0.81 | 0.94 | 1.00 |
| 2DSSG_exp2_1       | 0.91 | 0.93  | 0.78 | 0.87 | 0.93 | 1.00 |
| 2DSSS_exp2_3       | 0.92 | 0.93  | 0.78 | 0.87 | 0.94 | 1.00 |

#### Relationships (R@1)
| method         | Relationships | Objects | Predicate |
| -------------- | ------------- | ------- | --------- |
| 3DSSG baseline | 5.4           | 35.1    | 15.0      |
| 3DSSG GCNfeat  | 20.3          | 39.9    | 58.5      |
| 3DSSG_PN       | 31.5          | 33.4    | 64.2      |
| Jo_IJCV        | 42.5          | 52.0    | 71.2      |
| 2DSSS_exp2_3   | 84.4          | 50.5    | 87.5      |

Note: why ours looks much better in relationships? Our topK relationship includes true positive. maybe this is not the case for johanna? But the "none" relationships should also be correct. otherwise the network can always predicate something.

#### objects
| method       | IoU   | Precision | Recall |
| ------------ | ----- | --------- | ------ |
| 2DSSS_exp2_3 | 0.159 | 0.291     | 0.269  |

#### predicates
| method       | IoU   | Precision | Recall |
| ------------ | ----- | --------- | ------ |
| 2DSSS_exp2_3 | 0.993 | 0.421     | 0.242  |

Note: the IoU is extremely high because of the overwhelming true negatives (none relationship).


# SSG with Incremental Segments. 20-8
3RSan with ScanNet20.
- SGFN_inseg_0: SceneGraphFusion
- SGFN_inseg_0_1: with Pts,RGB,Normal
- SGFN_inseg_0_2: with Pts,RGB,Normal. change stop crite. to acc.
- SGFN_inseg_0_3: same as 0_1. just to check if reproducable.
- SGFN_inseg_0_4: same as 0_3. adjust schedular. ()
- SGFN_inseg_1: with img (should be renamed to 2DSSG_inseg_1 but this will break wandb)
- SGFN_full_0: use GT segments. wrong obj&rel classes.
- SGFN_full_0_1: with LRrule:0. failed. drop too fast
- SGFN_full_0_2: with LRrule:1
- SGFN_full_0_3: with LRrule:1. fix predicates class.
- 2DSSG_full_0: try to use image, multi-predicates.
- 2DSSG_full_1: single predicates
- 2DSSG_full_1_1: single predicates. fix predicates class. ->abort
- 2DSSG_full_1_2: single predicates. fix predicates class. fix class weighting.


## Segment level
| method        | R@1  | R@3  | R@1  | R@3  | R@1  | R@2  |
| ------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| SGFN(cvpr)(f) | 0.55 | 0.78 | 0.75 | 0.93 | 0.86 | 0.98 |
| inseg_0       | 0.27 | 0.41 | 0.55 | 0.84 | 0.88 | 0.96 |
| inseg_0_1     | 0.43 | 0.61 | 0.69 | 0.91 | 0.89 | 0.97 |
| inseg_0_4     | 0.46 | 0.63 | 0.72 | 0.91 | 0.9  | 0.97 |

## Instance level
#### Relationship
| method         | R@1  | R@3  | R@1  | R@3  | R@1  | R@2  |
| -------------- | ---- | ---- | ---- | ---- | ---- | ---- |
| SGFN_full_0_2  | 0.32 | 0.56 | 0.56 | 0.85 | 0.96 | 1.00 |
| SGFN_inseg_0_1 | 0.31 | 0.45 | 0.56 | 0.74 | 0.54 | 0.57 |
| SGFN_full_0_3  | 0.31 | 0.55 | 0.58 | 0.85 | 0.92 | 0.99 |
| 2DSSG_full_1_2 | 0.46 | 0.59 | 0.72 | 0.92 | 0.87 | 0.96 |

for the instance case, maybe it is better to show precision, since a lot of
objects and predicates are missing due to the missing nodes.

#### Object
| method         | IoU   | Precision | Recall |
| -------------- | ----- | --------- | ------ |
| SGFN_full_0_2  | 0.316 | 0.426     | 0.506  |
| SGFN_inseg_0_1 | 0.283 | 0.716     | 0.304  |
| SGFN_full_0_3  | 0.326 | 0.457     | 0.482  |
| 2DSSG_full_1_2 | 0.510 | 0.603     | 0.754  |

#### Predicate
| method         | IoU   | Precision | Recall |
| -------------- | ----- | --------- | ------ |
| SGFN_full_0_3  | 0.325 | 0.329     | 0.656  |
| 2DSSG_full_1_2 | 0.308 | 0.329     | 0.802  |

remember to recalculate average. (ignore none in recall and iou)

# Some Setup
## LR schedular
- 0: reduceluronplateau, factor=0.1
- 1: reduceluronplateau, factor=0.9
