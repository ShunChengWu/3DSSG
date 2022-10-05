# New
There was a problem with the edge connection and network setup. 
3DSSG should have exactly the same parameters apart from the use of input rel. representation and the GNN architecture. 
The edge connection for multi-predicate estimation should be fully connected. For this need to redo experiments
- [x] 3DSSG_full_l160_0
- [x] 2DSSG_full_l160_1

Use SGFN loader to train VGfM and IMP. 
Change loader from graph to sgfn.
remember to change drop_img_edge
- [ ] IMP_FULL_l160_2_1
- [x] IMP_full_l20_4
- [ ] IMP_INSEG_l20_2
- [ ] IMP_ORBSLAM3_l20_2

- [x] VGfM_FULL_l160_3_1
- [ ] VGfM_full_l20_6 # 
- [ ] VGFM_INSEG_l20_3
- [ ] VGFM_ORBSLAM3_l20_4

Test new loader
- [x] config_SGFN_full_l20_0.yaml
- [x] config_2DSSG_full_l20_3_1.yaml

# TODO: Re-eval all methods. 
# TODO: run again. there was a bug in eval using edge indicex (N,2)->(2,N)
# TODO: check instance ID conversion
# TODO: there was a bug in the GT rel generator. for single prediction.
3RScan160, Multi-Pred.
- [ ] IMP (IMP_FULL_l160_2_1)
- [x] VGfM (VGfM_FULL_l160_3_1)
- [ ] 3DSSG (3DSSG_full_l160_1)
- [ ] SGFN (SGFN_full_l160_3)
- [ ] 2DSSG (2DSSG_full_l160_2)
ScanNet20, Single, Full
- [x] IMP (IMP_full_l20_4)
- [ ] IMP (IMP_full_l20_5)
- [x] VGfM (VGfM_full_l20_6)
- [ ] VGfM (VGfM_full_l20_7)
- [ ] 3DSSG (3DSSG_INSEG_l20_2)
- [ ] SGFN
- [ ] 2DSSG
ScanNet20, Single, Inseg
- [ ] IMP
- [ ] VGfM
- [ ] 3DSSG
- [ ] SGFN
- [ ] 2DSSG
ScanNet20, Single, ORBSLAM

## 3RScan160, Multiple Predicates
| method             | Trip      | Obj       | Pred      | mRecall_O | mRecall_P |
| ------------------ | --------- | --------- | --------- | --------- | --------- |
| IMP_FULL_l160_1    | 0.0/73.7  | 3.5/95.9  | 1.4/74.0  | 1.1/94.4  | 18.5/81.5 |
| VGfM_FULL_l160_3_1 | 5.0/81.9  | 43.5/85.1 | 10.5/100  | 18.5/82.4 | 11.0/100  |
| 3DSSG_full_l160_0  | 8.2/100   | 30.4/100  | 50.2/100  | 10.4/100  | 17.4/100  |
| SGFN_full_l160_2   | 6.3/100   | 34.6/100  | 37.2/100  | 15.6/100  | 10.8/100  |
| SGFN_full_l160_3   | 3.4       | 30.0      | 25.6      | 13.3      | 12.2      |
| 2DSSG_full_l160_1  | 11.7/96.4 | 49.9/95.9 | 42.7/95.6 | 28.5/94.5 | 25.4/95.5 |

## ScanNet20, Single Predicate, full
| method             | Trip      | Obj       | Pred      | mRecall_O | mRecall_P |
| ------------------ | --------- | --------- | --------- | --------- | --------- |
| IMP_full_l20_3     | 8.8/97.3  | 28.1/97.9 | 94.3/99.9 | 4.9/98.1  | 12.5/98.2 |
| IMP_full_l20_4     | 22.7/85.8 | 47.4/87.5 | 29.3/98.0 | 50.1/92.1 | 59.6/98.6 |
| VGfM_full_l20_5    | 16.8/97.3 | 38.4/97.9 | 94.3/99.9 | 24.1/98.1 | 18.9/98.2 |
| VGfM_full_l20_6    | 35.6/86.9 | 67.6/92.2 | 60.1/98.6 | 55.5/92.1 | 25.5/98.0 |
| 3DSSG_full_l20_2   | 10.9/100  | 35.1/100  | 88.1/100  | 17.3/100  | 28.3/100  |
| SGFN_full_0_3      | 35.6/100  | 59.8/100  | 89.3/100  | 53.0/100  | 66.1/100  |
| 2DSSG_full_l20_2   | 61.3/97.4 | 76.6/97.9 | 95.6/99.9 | 79.1/98.1 | 70.5/98.5 |
| SGFN_full_l20_0    | 42.7/100  | 62.9/100  | 67.6/100  | 55.4/100  | 57.1/100  |
| 2DSSG_full_l20_3_1 | 57.7/97.4 | 75.3/97.9 | 76.8/99.9 | 80.6/98.1 | 71.9/98.5 |

## ScanNet20, Single Predicate, inseg
| method            | Trip      | Obj       | Pred      | mRecall_O | mRecall_P |
| ----------------- | --------- | --------- | --------- | --------- | --------- |
| IMP_INSEG_l20_1   | 0.7/6.8   | 7.6/24.9  | 94.3/94.9 | 1.3/23.2  | 12.5/18.9 |
| VGfM_INSEG_l20_2  | 0.0/6.8   | 2.5/24.9  | 93.4/94.9 | 1.2/23.2  | 12.5/18.9 |
| 3DSSG_INSEG_1     | 17.2/63.8 | 41.7/75.1 | 91.4/98.3 | 34.7/75.7 | 33.1/67.5 |
| SGFN_inseg_0_5    | 29.3/63.8 | 55.1/75.1 | 84.3/98.3 | 46.8/75.7 | 37.4/67.5 |
| 2DSSG_INSEG_l20_1 | 31.4/63.7 | 54.3/75.0 | 90.6/98.3 | 47.9/75.7 | 33.9/67.4 |
TODO: investigate why IMP and VGfM have so low upper bound

## ScanNet20, Single Predicate, ORBSLAM3
| method                 | Trip     | Obj       | Pred      | mRecall_O | mRecall_P |
| ---------------------- | -------- | --------- | --------- | --------- | --------- |
| IMP_ORBSLAM3_l20_1     | 0.0/0.1  | 0.8/3.2   | 94.3/94.4 | 0.1/2.3   | 12.5/12.6 |
| VGfM_ORBSLAM3_l20_3    | 0.0/0.1  | 1.0/3.2   | 94.3/94.4 | 0.6/2.3   | 12.5/12.6 |
| 3DSSG_ORBSLAM3_l20_0   | 1.8/25.5 | 12.9/44.2 | 93.2/96.1 | 8.7/48.6  | 17.5/35.2 |
| SGFN_ORBSLAM3_l20_0    | 2.5/25.5 | 15.5/44.2 | 94.0/96.1 | 6.9/48.6  | 13.2/35.2 |
| 2DSSG_ORBSLAM3_l20_6_1 | 8.7/25.5 | 27.0/44.2 | 93.2/96.1 | 25.1/48.6 | 19.6/35.2 |
TODO: investigate why IMP and VGfM have so low upper bound

##
segment-level, instance-level
| method                 | Trip | Obj  | Pred | mO   | mR   | Trip      | Obj       | Pred      | mO        | mR        |
| ---------------------- | ---- | ---- | ---- | ---- | ---- | --------- | --------- | --------- | --------- | --------- |
| 2DSSG_full_l20_2       | 58.6 | 77.4 | 97.6 | 80.9 | 71.4 | 56.5/96.4 | 75.8/97.9 | 97.7/99.9 | 79.4/98.1 | 70.3/98.5 |
| SGFN_full_0_3          | 32.1 | 58.6 | 93.1 | 47.6 | 63.1 | 33.0/100  | 59.4/100  | 93.8/100  | 49.4/100  | 67.2/100  |
| 3DSSG_full_l20_1       |
| IMP_ORBSLAM3_l20_1     | 2.55 | 25.6 | 82.0 | 5.0  | 11.1 | 0.1/1.3   | 0.8/3.2   | 97.1/97.2 | 0.1/2.3   | 12.5/12.6 |
| VGfM_ORBSLAM3_l20_3    | 7.8  | 35.4 | 72.1 | 8.7  | 17.4 | 0.1/1.3   | 1.0/3.2   | 97.1/97.2 | 0.6/2.3   | 12.5/12.6 |
| 2DSSG_ORBSLAM3_l20_6_1 | 27.2 | 57.0 | 86.0 | 55.4 | 49.3 | 6.5/21.8  | 26.2/44.2 | 96.3/98.2 | 28.1/48.6 | 21.3/39.0 |
| SGFN_ORBSLAM3_l20_0    | 8.6  | 35.6 | 86.2 | 10.8 | 15.2 | 2.3/21.8  | 16.4/44.2 | 97.0/98.2 | 7.3/48.6  | 13.2/39.0 |
| 3DSSG_ORBSLAM3_l20_0   | 7.1  | 31.0 | 86.3 | 16.8 | 34.9 | 1.6/21.8  | 13.7/44.2 | 96.4/98.2 | 9.3/48.6  | 17.3/39.0 |
| 2DSSG_INSEG_l20_1      | 42.6 | 67.9 | 91.7 | 59.1 | 56.7 | 26.6/57.2 | 53.3/74.7 | 95.1/99.1 | 47.1/75.5 | 34.2/68.0 |
| SGFN_inseg_0_5         | 39.7 | 67.3 | 84.6 | 59.3 | 61.6 | 27.2/58.1 | 54.7/75.1 | 91.0/99.2 | 47.9/75.7 | 38.5/68.5 |
| 3DSSG_INSEG_0          |

Note: VGfM and IMP in ORBSLAM3, INSEG should be retrained due to the filtering.

Note: Why in ORBSLAM3 2DSSG has higher UB than VGfM?

3D method should have the same UB. 

2D methods should have the same UB in object estimation. for predicate IMP and VGfM should have way smaller UB due to the limitation of edge building on image space.

all those methods are evaluated with fully-connected edge, which not be.


# ============= OLD ===================
# SSG with GT Segmentation. Multiple predicate prediction. 160-26
full_edge. multi_rel.
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
- SGFN_full_l160_0: forgot to use full_edge
- SGFN_full_l160_1: use full edge

## 2DSSG
img_batchsize: 8
- exp0: MVCNN+VGG16. no GNN
- exp1: MVCNN+Res18. no gnn
- exp2: MVCNN+Res18. FAN -> forgot to enable gnn
- 2DSSG_exp2_1: MVCNN+Res18. FAN 2 layer GNN
- 2DSSG_exp_2_2: with 0.15 bounding box augmentation
- 2DSSG_exp3: MVCNN+Res18, FAN, 1 layer GNN
- 2DSSG_exp_2_3: with LR schedular setup1
- 2DSSG_full_l160_0: try to train with the best setup

## IMP 
- [x] IMP_FULL_l160_0:
- [ ] IMP_FULL_l160_1: fixing training procedure

## VGfM
- [x] VGfM_FULL_l160_0:
- [x] VGfM_FULL_l160_1: add geometric feature

#### relationships
Note taht there is an inconsistency in the reported number. At some points the
number is R@1.
| method             | R@50 | R@100 | R@5  | R@10 | R@3  | R@10 |
| ------------------ | ---- | ----- | ---- | ---- | ---- | ---- |
| 3DSSG_obj_wo_gcn   | 0.40 | 0.66  | 0.68 | 0.78 | 0.89 | 0.93 |
| 3DSSG_obj_from_gcn | 0.30 | 0.60  | 0.60 | 0.73 | 0.79 | 0.91 |
| exp0_9             | 0.75 | 0.77  | 0.49 | 0.64 | 0.36 | 0.51 |
| exp0_10            | 0.73 | 0.75  | 0.5  | 0.65 | 0.83 | 0.86 |
| exp0_12            | 0.75 | 0.77  | 0.57 | 0.70 | 0.81 | 0.92 |
| exp0_13            | 0.77 | 0.80  | 0.63 | 0.76 | 0.85 | 0.97 |
| 3DSSG_test_14      | 0.72 | 0.79  | 0.60 | 0.74 | 0.83 | 0.95 |
| SGFN               | 0.85 | 0.87  | 0.7  | 0.8  | 0.97 | 0.99 |
| exp1_0             | 0.91 | 0.92  | 0.71 | 0.82 | 0.94 | 1.00 |
| exp1_1             | 0.90 | 0.91  | 0.69 | 0.81 | 0.94 | 1.00 |
| 2DSSG_exp2_1       | 0.91 | 0.93  | 0.78 | 0.87 | 0.93 | 1.00 |
| 2DSSS_exp2_3       | 0.92 | 0.93  | 0.78 | 0.87 | 0.94 | 1.00 |
| SGFN_full_l160_1   | 88.7 | 90.0  | 65.1 | 76.0 | 93.0 | 99.5 |
| VGfM_FULL_l160_0   |      |       |      |      |      |      |
| IMP_FULL_l160_0    |      |       |      |      |      |      |

#### Relationships (R@1)
| method                | Relationships | Objects   | Predicate |
| --------------------- | ------------- | --------- | --------- |
| 3DSSG baseline        | 15.0          | 35.1      | 5.4       |
| 3DSSG GCNfeat         | 58.5          | 39.9      | 20.3      |
| 3DSSG_PN              | 64.2          | 33.4      | 31.5      |
| Jo_IJCV               | 71.2          | 52.0      | 42.5      |
| 3DSSG_test_14         | 70.2          | 31.1      | 74.2      |
| 2DSSS_exp2_3          | 84.4          | 50.5      | 87.5      |
| SGFN_full_l160_1      | 84.2          | 34.6      | 87.5      |
| 2DSSG_full_l160_0     | 84.2          | 52.3      | 86.9      |
| VGfM_FULL_l160_0      | 65.0          | 32.9      | 66.4      |
| IMP_FULL_l160_0       | 63.2          | 37.0      | 66.6      |
| VGfM_FULL_l160_1      | 63.1          | 54.0      | 65.3      |
| IMP_FULL_l160_1       |               |           |           |
| VGfM_FULL_l160_2_inst | 82.3          | 17.0      | 29.5      |
| VGfM_FULL_l160_UB     | 83.4          | 95.9      | 86.9      |
| SGFN_full_l160_UB     | 83.8          | 100       | 90.3      |
| SGFN_full_l160_1_inst | 83.7          | 34.9      | 87.6      |
| VGfM_FULL_l160_2      | 0.1/73.7      | 17.0/95.9 | 9.8/74.0  | 4.6/94.4 | 19.0/81.5 |
Note: why ours looks much better in relationships? Our topK relationship includes true positive. maybe this is not the case for johanna? But the "none" relationships should also be correct. otherwise the network can always predicate something.

#### objects
| method                | IoU  | Precision | Recall |
| --------------------- | ---- | --------- | ------ |
| 3DSSG_test_14         | 8.0  | 13.3      | 16.2   |
| SGFN_full_l160_1      | 7.7  | 15.3      | 13.9   |
| 2DSSS_exp2_3          | 15.9 | 29.1      | 26.9   |
| 2DSSG_full_l160_0     | 19.1 | 30.9      | 31.9   |
| VGfM_FULL_l160_0      | 6.6  | 13.9      | 15.3   |
| IMP_FULL_l160_0       | 9.5  | 16.4      | 22.6   |
| IMP_FULL_l160_1       |      |           |        |
| VGfM_FULL_l160_2_inst | 1.8  | 11.3      | 4.6    |
| VGfM_FULL_l160_UB     | 93.9 | 99.4      | 94.4   |
| SGFN_full_l160_UB     | 100  | 100       | 100    |
| SGFN_full_l160_1_inst | 7.9  | 16.7      | 13.8   |

#### predicates
| method                | IoU  | Precision | Recall |
| --------------------- | ---- | --------- | ------ |
| 3DSSG_test_14         | 98.6 | 34.3      | 11.6   |
| SGFN_full_l160_1      | 99.3 | 32.4      | 11.4   |
| 2DSSS_exp2_3          | 99.3 | 42.1      | 24.2   |
| 2DSSG_full_l160_0     | 99.3 | 38.3      | 26.3   |
| VGfM_FULL_l160_0      | 98.3 | 13.0      | 3.9    |
| IMP_FULL_l160_0       | 98.3 | 17.5      | 6.1    |
| VGfM_FULL_l160_1      | 98.2 | 16.2      | 8.9    |
| IMP_FULL_l160_1       |      |           |        |
| VGfM_FULL_l160_2_inst | 39.4 | 0.3       | 19.0   |
| VGfM_FULL_l160_UB     | 99.5 | 100       | 31.7   |
| SGFN_full_l160_UB     | -    | -         | 59.5   |
| SGFN_full_l160_1_inst | -    | -         | 11.7   |

Note: the IoU is extremely high because of the overwhelming true negatives (none relationship).


# SSG with Incremental Segments. 20-8
3RSan with ScanNet20.
- SGFN_inseg_0: SceneGraphFusion
- SGFN_inseg_0_1: with Pts,RGB,Normal
- SGFN_inseg_0_2: with Pts,RGB,Normal. change stop crite. to acc.
- SGFN_inseg_0_3: same as 0_1. just to check if reproducable.
- SGFN_inseg_0_4: same as 0_3. adjust schedular.
- SGFN_inseg_0_5: with the correct predicate labels
- SGFN_inseg_1: with img (should be renamed to 2DSSG_inseg_1 but this will break wandb)

- SGFN_full_0: use GT segments. wrong obj&rel classes.
- SGFN_full_0_1: with LRrule:0. failed. drop too fast
- SGFN_full_0_2: with LRrule:1
- SGFN_full_0_3: with LRrule:1. fix predicates class.
- [x] SGFN_ORBSLAM3_l20_0:


- 3DSSG_full_l20_0: -> not valid. was using the full edge in training.
- 3DSSG_full_l20_1: don't use full edge
- [ ] 3DSSG_full_l20_1: retrain. model doesn't match.
- [x] 3DSSG_INSEG_0:
- [x] 3DSSG_ORBSLAM3_l20_0:

- [x] IMP_full_l20_0: iterative message passing  -> doesn't train
- [x] IMP_full_l20_1: use dyanmic ratio. fix rel GT.
  - [x] IMG_full_l20_1_1: fix image loader -> doesn't work.
- [x] IMP_full_l20_2: _1 + use global feature
  - [x] IMP_full_l20_2_1: fix loader loader
  - [x] IMP_full_l20_2_2: fix loader loader. turn off full_edge
- [x] IMP_full_l20_3: train with filtered node list, without relationship filtering on nodes.
- [x] IMP_ORBSLAM3_l20_0: train IMP on ORBSLAM3 entities
- [x] IMP_ORBSLAM3_l20_1: train IMP on ORBSLAM3 entities. turn off full_edge
- [x] IMP_INSEG_l20_0:


- VGfM_full_l20_0: there was a bug. using `msg_t_node` for both node and edge
- VGfM_full_l20_1: with the bug.
- [x] VGfM_full_l20_2: fix bug and turn off full_edge
- [x] VGfM_full_l20_3: fix temporal summ
- [x] VGfM_full_l20_4: add geometric feature
- [x] VGfM_INSEG_l20_0:
- [x] VGfM_INSEG_l20_1: add geometric feature
- [x] VGfM_full_l20_5: train with filtered node list, without relationship filtering on nodes.
- [X] VGfM_full_l20_UB: UpperBound. Only consider missing instances and predicates 
- [x] VGfM_FULL_l160_2: fixing training procedure

- VGfM_ORBSLAM3_l20_0:there was a bug. using `msg_t_node` for both node and edge
- VGfM_ORBSLAM3_l20_1: fix the bug
- [x] VGfM_ORBSLAM3_l20_2: turn off full_edge
- [x] VGfM_ORBSLAM3_l20_3: add geometric feature

- 2DSSG_full_0: try to use image, multi-predicates.
- 2DSSG_full_1: single predicates
- 2DSSG_full_1_1: single predicates. fix predicates class. ->abort
- 2DSSG_full_1_2: single predicates. fix predicates class. fix class weighting.
- 2DSSG_full_1_3: try to roll back to the previous predicate class weighting.
- 2DSSG_full_l20_1: train with the best setup.
- [x] 2DSSG_full_l20_2: train with the best setup. turn off full_edge

- 2DSSG_INSEG_l20_0: train on InSeg segmentation
- [x] 2DSSG_INSEG_l20_1: train on InSeg segmentation. without full_edge

- 2DSSG_ORBSLAM3_l20_0: with est. entities from ORBSLAM3 (forgot to finetune)
- 2DSSG_ORBSLAM3_l20_1: with improved bboxes
- 2DSSG_ORBSLAM3_l20_2: with improved bboxes. w/o finetune
- 2DSSG_ORBSLAM3_l20_3: bbox augmentation 0.3. use acc_node_cls instaed of iou as the stop and schedular.
- 2DSSG_ORBSLAM3_l20_4: use spatial encoder (fc,dim=128). bbox augmentation 0.3.
- 2DSSG_ORBSLAM3_l20_5: try to use different edge descriptor
- 2DSSG_ORBSLAM3_l20_6: with dyanmic ratio between node and edge
- 2DSSG_ORBSLAM3_l20_7: old edge description+ dynamic
- [x] 2DSSG_ORBSLAM3_l20_6_1: turn off full_edge
- [x] 2DSSG_ORBSLAM3_l20_7_1: turn off full_edge

## Segment level
#### Relationship
| method                 | rel.R@1 | rel.R@3 | obj.R@1 | obj.R@3 | pred.R@1 | pred.R@2 |
| ---------------------- | ------- | ------- | ------- | ------- | -------- | -------- |
| SGFN(cvpr)(f)          | 0.55    | 0.78    | 0.75    | 0.93    | 0.86     | 0.98     |
| inseg_0                | 0.27    | 0.41    | 0.55    | 0.84    | 0.88     | 0.96     |
| inseg_0_1              | 0.43    | 0.61    | 0.69    | 0.91    | 0.89     | 0.97     |
| inseg_0_4              | 0.46    | 0.63    | 0.72    | 0.91    | 0.9      | 0.97     |
| SGFN_inseg_0_5         | 39.3    | 54.9    | 67.3    | 89.6    | 82.8     | 94.7     |
| 2DSSG_ORBSLAM3_l20_0   | 22.7    | 32.3    | 53.8    | 80.3    | 82.1     | 91.6     |
| 2DSSG_ORBSLAM3_l20_1   | 21.5    | 29.3    | 48.2    | 72.7    | 80.5     | 90.6     |
| 2DSSG_ORBSLAM3_l20_2   | 25.5    | 34.8    | 53.6    | 81.2    | 79.9     | 90.0     |
| 2DSSG_ORBSLAM3_l20_3   | 29.4    | 39.5    | 58.0    | 84.1    | 80.8     | 91.1     |
| 2DSSG_ORBSLAM3_l20_4   | 31.8    | 40.8    | 60.3    | 83.8    | 78.2     | 90.3     |
| 2DSSG_ORBSLAM3_l20_5   | 32.7    | 42.3    | 61.2    | 84.6    | 80.5     | 90.9     |
| 2DSSG_ORBSLAM3_l20_6   | 32.7    | 40.9    | 60.6    | 84.9    | 81.5     | 91.3     |
| 2DSSG_ORBSLAM3_l20_7   | 31.3    | 35.8    | 60.3    | 86.5    | 81.3     | 91.3     |
| 2DSSG_INSEG_l20_0      | 42.9    | 52.8    | 69.1    | 89.8    | 89.5     | 96.5     |
| IMP_full_l20_2_1       | 31.0    | 40.9    | 58.4    | 82.2    | 83.9     | 90.9     |
| VGfM_full_l20_0        | 28.6    | 38.7    | 57.5    | 80.9    | 83.7     | 90.3     |
| 2DSSG_full_l20_2       | 54.5    | 77.0    | 75.8    | 92.5    | 95.9     | 99.5     |
| VGfM_full_l20_2        | 29.8    | 40.2    | 59.2    | 82.7    | 83.5     | 89.9     |
| 2DSSG_ORBSLAM3_l20_6_1 | 29.5    | 38.7    | 58.0    | 86.8    | 80.4     | 91.2     |
| 2DSSG_ORBSLAM3_l20_7_1 | 31.6    | 40.2    | 59.9    | 85.1    | 80.6     | 91.1     |
| IMP_ORBSLAM3_l20_1     | 26.8    | 32.7    | 52.9    | 79.1    | 72.2     | 85.5     |
| VGfM_ORBSLAM3_l20_2    | 26.4    | 35.7    | 55.0    | 80.4    | 71.7     | 83.2     |
| VGfM_INSEG_l20_0       | 26.1    | 37.9    | 57.7    | 81.4    | 67.1     | 82.6     |
| IMP_INSEG_l20_0        | 28.7    | 37.4    | 58.8    | 82.4    | 69.4     | 86.6     |
| 2DSSG_INSEG_l20_1      | 42.2    | 49.8    | 67.9    | 89.0    | 89.6     | 96.4     |
| SGFN_ORBSLAM3_l20_0    | 13.6    | 22.8    | 35.9    | 57.1    | 81.5     | 87.4     |
| 3DSSG_ORBSLAM3_l20_0   | 12.3    | 18.4    | 31.0    | 62.4    | 81.6     | 89.7     |
| VGfM_full_l20_4        | 46.7    | 57.7    | 74.7    | 92.4    | 84.8     | 91.6     |
| 3DSSG_INSEG_0          | 18.2    | 32.3    | 42.2    | 68.1    | 93.4     | 97.4     |
| VGfM_INSEG_l20_1       | 39.9    | 49.7    | 68.6    | 88.2    | 73.4     | 89.0     |
| VGfM_ORBSLAM3_l20_3    | 29.9    | 37.3    | 57.6    | 82.0    | 74.3     | 86.0     |
| IMP_full_l20_3         | 11.71   | 11.71   | 30.14   | 46.25   | 84.39    | 85.19    |

#### Object
| method                 | IoU  | Precision | Recall |
| ---------------------- | ---- | --------- | ------ |
| 2DSSG_ORBSLAM3_l20_0   | 28.9 | 41.8      | 47.4   |
| 2DSSG_full_1_3         | 51.0 | 61.1      | 75.8   |
| SGFN_inseg_0_5         | 41.7 | 52.1      | 59.3   |
| 2DSSG_ORBSLAM3_l20_1   | 22.1 | 33.6      | 39.9   |
| 2DSSG_ORBSLAM3_l20_2   | 29.1 | 40.6      | 54.8   |
| 2DSSG_ORBSLAM3_l20_3   | 31.0 | 43.2      | 53.4   |
| 2DSSG_ORBSLAM3_l20_4   | 29.5 | 40.5      | 52.3   |
| 2DSSG_ORBSLAM3_l20_5   | 30.9 | 40.4      | 50.5   |
| 2DSSG_ORBSLAM3_l20_6   | 31.1 | 42.1      | 54.9   |
| 2DSSG_ORBSLAM3_l20_7   | 31.5 | 43.5      | 52.4   |
| 2DSSG_INSEG_l20_0      | 42.4 | 54.0      | 60.6   |
| VGfM_full_l20_0        | 27.3 | 38.2      | 52.0   |
| IMP_full_l20_2_1       | 29.2 | 41.8      | 50.0   |
| 2DSSG_full_l20_2       | 55.1 | 66.6      | 79.4   |
| VGfM_full_l20_2        | 26.4 | 36.0      | 50.7   |
| 2DSSG_ORBSLAM3_l20_6_1 | 30.4 | 40.1      | 52.9   |
| 2DSSG_ORBSLAM3_l20_7_1 | 29.7 | 39.5      | 51.4   |
| IMP_ORBSLAM3_l20_1     | 23.1 | 33.3      | 45.0   |
| VGfM_ORBSLAM3_l20_2    | 27.2 | 42.4      | 42.8   |
| VGfM_INSEG_l20_0       | 23.6 | 36.2      | 39.0   |
| IMP_INSEG_l20_0        | 23.9 | 33.0      | 39.7   |
| 2DSSG_INSEG_l20_1      | 41.3 | 52.9      | 59.1   |
| SGFN_ORBSLAM3_l20_0    | 6.3  | 26.8      | 10.8   |
| 3DSSG_ORBSLAM3_l20_0   | 9.1  | 21.7      | 16.8   |
| VGfM_full_l20_4        | 40.5 | 52.7      | 65.5   |
| 3DSSG_INSEG_0          | 19.4 | 35.6      | 33.0   |
| VGfM_INSEG_l20_1       | 37.5 | 48.5      | 57.2   |
| VGfM_ORBSLAM3_l20_3    | 26.6 | 39.1      | 41.9   |
| IMP_full_l20_3         | 1.40 | 14.3      | 5.00   |

#### Predicates
| method                 | IoU  | Precision | Recall |
| ---------------------- | ---- | --------- | ------ |
| 2DSSG_ORBSLAM3_l20_0   | 21.2 | 28.9      | 32.0   |
| 2DSSG_full_1_3         | 40.9 | 46.0      | 73.0   |
| SGFN_inseg_0_5         | 31.1 | 34.7      | 61.6   |
| 2DSSG_ORBSLAM3_l20_1   | 24.6 | 35.6      | 41.5   |
| 2DSSG_ORBSLAM3_l20_2   | 25.4 | 37.6      | 48.5   |
| 2DSSG_ORBSLAM3_l20_4   | 25.8 | 34.3      | 48.6   |
| 2DSSG_ORBSLAM3_l20_3   | 27.6 | 38.7      | 45.5   |
| 2DSSG_ORBSLAM3_l20_5   | 27.3 | 36.2      | 49.6   |
| 2DSSG_ORBSLAM3_l20_6   | 27.0 | 36.8      | 47.9   |
| 2DSSG_ORBSLAM3_l20_7   | 26.6 | 36.9      | 46.4   |
| 2DSSG_INSEG_l20_0      | 37.5 | 44.4      | 56.3   |
| VGfM_full_l20_0        | 23.1 | 36.1      | 35.3   |
| IMP_full_l20_2_1       | 32.3 | 44.1      | 49.2   |
| 2DSSG_full_l20_2       | 45.2 | 51.4      | 70.3   |
| VGfM_full_l20_2        | 23.4 | 36.5      | 39.1   |
| 2DSSG_ORBSLAM3_l20_6_1 | 27.0 | 38.7      | 51.3   |
| 2DSSG_ORBSLAM3_l20_7_1 | 26.3 | 35.2      | 47.9   |
| IMP_ORBSLAM3_l20_1     | 18.2 | 26.9      | 31.4   |
| VGfM_ORBSLAM3_l20_2    | 14.0 | 22.1      | 22.4   |
| VGfM_INSEG_l20_0       | 16.2 | 33.5      | 25.0   |
| IMP_INSEG_l20_0        | 27.7 | 36.5      | 46.0   |
| 2DSSG_INSEG_l20_1      | 37.1 | 43.9      | 56.7   |
| SGFN_ORBSLAM3_l20_0    | 12.9 | 69.1      | 15.2   |
| 3DSSG_ORBSLAM3_l20_0   | 21.4 | 35.9      | 34.9   |
| VGfM_full_l20_4        | 36.4 | 43.0      | 58.8   |
| 3DSSG_INSEG_0          | 23.9 | 37.7      | 28.3   |
| VGfM_INSEG_l20_1       | 31.9 | 41.8      | 49.6   |
| VGfM_ORBSLAM3_l20_3    | 24.0 | 31.5      | 46.7   |
| IMP_full_l20_3         | 11.4 | 91.3      | 12.5   |

## Instance level
#### Relationship
| method                  | rel.R@1 | rel.R@3 | obj.R@1 | obj.R@3 | pred.R@1 | pred.R@2 |
| ----------------------- | ------- | ------- | ------- | ------- | -------- | -------- |
| SGFN_full_0_2           | 0.32    | 0.56    | 0.56    | 0.85    | 0.96     | 1.00     |
| SGFN_inseg_0_1          | 0.31    | 0.45    | 0.56    | 0.74    | 0.54     | 0.57     |
| SGFN_full_0_3           | 31.4    | 55.3    | 58.4    | 84.8    | 92.0     | 98.6     |
| SGFN_inseg_0_5          | 28.0    | 41.5    | 54.6    | 72.8    | 94.2     | 98.2     |
| 3DSSG_full_l20_1        | 26.5    | 39.6    | 52.3    | 81.8    | 91.3     | 95.7     |
| 2DSSG_full_1_2          | 45.5    | 58.9    | 72.5    | 91.9    | 87.2     | 95.8     |
| 2DSSG_full_1_3          | 49.9    | 58.3    | 73.0    | 92.4    | 95.2     | 97.4     |
| 2DSSG_ORBSLAM3_l20_0    | 3.6     | 7.0     | 19.2    | 37.5    | 95.3     | 97.8     |
| 2DSSG_ORBSLAM3_l20_1    | 5.6     | 10.0    | 24.2    | 43.4    | 95.0     | 98.0     |
| 2DSSG_ORBSLAM3_l20_1*   | 21.5    | 29.3    | 48.2    | 72.7    | 80.5     | 90.6     |
| 2DSSG_ORBSLAM3_l20_2*   | 27.5    | 48.2    | 55.6    | 82.8    | 89.8     | 99.0     |
| 2DSSG_ORBSLAM3_l20_2    | 6.4     | 11.8    | 26.4    | 46.8    | 95.0     | 98.1     |
| SGFN_inseg_0_5*         | 49.4    | 72.7    | 72.8    | 92.1    | 92.0     | 98.3     |
| 2DSSG_ORBSLAM3_l20_4*   | 35.6    | 55.5    | 62.4    | 85.7    | 88.5     | 98.9     |
| 2DSSG_ORBSLAM3_l20_4    | 8.2     | 13.4    | 29.7    | 48.2    | 94.7     | 98.1     |
| 2DSSG_ORBSLAM3_l20_3    | 7.3     | 13.2    | 28.6    | 48.1    | 95.2     | 95.2     |
| 2DSSG_ORBSLAM3_l20_3*   | 31.9    | 54.5    | 60.1    | 85.4    | 90.7     | 99.2     |
| 2DSSG_ORBSLAM3_l20_5    | 8.7     | 14.3    | 30.8    | 48.7    | 95.1     | 98.1     |
| 2DSSG_ORBSLAM3_l20_5*   | 38.4    | 59.9    | 64.7    | 86.8    | 90.4     | 99.0     |
| 2DSSG_ORBSLAM3_l20_6    | 8.7     | 14.3    | 30.4    | 48.6    | 95.3     | 98.2     |
| 2DSSG_ORBSLAM3_l20_6*   | 38.5    | 59.6    | 63.9    | 86.5    | 91.1     | 99.2     |
| 2DSSG_ORBSLAM3_l20_7    | 8.1     | 14.3    | 29.5    | 49.4    | 95.2     | 98.1     |
| 2DSSG_ORBSLAM3_l20_7*   | 35.8    | 59.8    | 62.0    | 88.2    | 90.9     | 99.1     |
| IMG_full_l20_2_1        | 8.1     | 14.3    | 31.7    | 51.5    | 95.4     | 98.1     |
| IMG_full_l20_2_1*       | 30.4    | 51.3    | 57.5    | 82.5    | 91.4     | 98.1     |
| IMG_full_l20_1_1        | 2.3     | 3.3     | 9.6     | 29.3    | 95.3     | 98.1     |
| IMP_ORBSLAM3_l20_0      |         |         |         |         |          |          |
| 2DSSG_full_l20_1        | 53.3    | 77.7    | 75.0    | 92.7    | 95.7     | 99.6     |
| 2DSSG_INSEG_l20_0       | 26.9    | 41.0    | 53.8    | 72.5    | 95.7     | 98.7     |
| 2DSSG_INSEG_l20_0*      | 48.3    | 72.9    | 71.8    | 91.7    | 94.6     | 99.2     |
| VGfM_full_l20_0         | 4.2     | 9.2     | 22.6    | 47.5    | 93.9     | 97.8     |
| VGfM_full_l20_0*        | 15.5    | 31.7    | 41.0    | 75.4    | 85.6     | 97.0     |
| 2DSSG_full_l20_2        | 54.5    | 77.0    | 75.8    | 92.5    | 95.9     | 99.5     |
| IMP_full_l20_2_2        | 8.1     | 14.3    | 31.6    | 51.5    | 95.6     | 98.1     |
| IMP_full_l20_2_2*       | 30.4    | 51.1    | 57.3    | 82.6    | 92.1     | 98.1     |
| VGfM_full_l20_2*        | 19.0    | 36.4    | 46.8    | 77.4    | 83.2     | 96.4     |
| VGfM_full_l20_2         | 5.1     | 10.4    | 25.8    | 48.6    | 93.3     | 97.6     |
| 2DSSG_ORBSLAM3_l20_6_1  | 7.7     | 14.0    | 28.9    | 49.6    | 94.9     | 98.2     |
| 2DSSG_ORBSLAM3_l20_6_1* | 33.6    | 58.5    | 60.7    | 88.6    | 89.5     | 99.3     |
| 2DSSG_ORBSLAM3_l20_7_1  | 8.3     | 14.3    | 29.8    | 48.9    | 95.0     | 98.1     |
| 2DSSG_ORBSLAM3_l20_7_1* | 36.8    | 59.6    | 62.7    | 87.1    | 90.1     | 99.0     |
| IMP_ORBSLAM3_l20_1      | 0.1     | 18.5    | 0.6     | 13.3    | 95.4     | 97.2     |
| IMP_ORBSLAM3_l20_1*     | 12.5    | 20.8    | 22.2    | 50.0    | 75.0     | 91.7     |
| VGfM_ORBSLAM3_l20_2     | 0.1     | 1.8     | 0.9     | 13.5    | 95.4     | 97.2     |
| VGfM_ORBSLAM3_l20_2*    | 8.3     | 8.3     | 33.3    | 55.5    | 62.5     | 83.3     |
| VGfM_full_l20_4         | 11.1    | 18.7    | 38.1    | 55.1    | 95.4     | 98.2     |
| VGfM_full_l20_5         | 13.49   | 24.78   | 38.40   | 59.81   | 95.34    | 98.56    |
| VGfM_full_l20_UB        | 91.9    | 93.8    | 97.9    | 98.2    | 95.6     | 97.5     |

Note: `2DSSG_ORBSLAM3_l20_1*` is ignore missing objects and scans.

for the instance case, maybe it is better to show precision, since a lot of
objects and predicates are missing due to the missing nodes.

#### Object
| method                  | IoU   | Precision | Recall |
| ----------------------- | ----- | --------- | ------ |
| SGFN_full_0_2           | 0.316 | 0.426     | 0.506  |
| SGFN_inseg_0_1          | 0.283 | 0.716     | 0.304  |
| SGFN_full_0_3           | 0.326 | 0.457     | 0.482  |
| SGFN_inseg_0_5          | 33.5  | 52.1      | 47.2   |
| 3DSSG_full_l20_1        | 28.1  | 39.0      | 47.6   |
| 2DSSG_full_1_2          | 51.0  | 60.3      | 75.4   |
| 2DSSG_full_1_3          | 51.0  | 61.1      | 75.8   |
| 2DSSG_ORBSLAM3_l20_0    | 15.7  | 41.6      | 20.0   |
| 2DSSG_ORBSLAM3_l20_1    | 15.2  | 35.6      | 23.6   |
| 2DSSG_ORBSLAM3_l20_1*   | 22.1  | 33.6      | 39.9   |
| 2DSSG_ORBSLAM3_l20_2*   | 32.9  | 45.0      | 57.7   |
| 2DSSG_ORBSLAM3_l20_2    | 19.0  | 42.8      | 30.2   |
| SGFN_inseg_0_5*         | 46.8  | 57.4      | 62.4   |
| 2DSSG_ORBSLAM3_l20_4*   | 33.8  | 45.7      | 53.4   |
| 2DSSG_ORBSLAM3_l20_4    | 19.0  | 43.5      | 27.6   |
| 2DSSG_ORBSLAM3_l20_3    | 20.2  | 45.5      | 28.8   |
| 2DSSG_ORBSLAM3_l20_3*   | 34.4  | 47.8      | 55.4   |
| 2DSSG_ORBSLAM3_l20_5*   | 36.8  | 50.0      | 52.1   |
| 2DSSG_ORBSLAM3_l20_5    | 20.1  | 47.5      | 26.3   |
| 2DSSG_ORBSLAM3_l20_6    | 21.6  | 49.2      | 29.8   |
| 2DSSG_ORBSLAM3_l20_6*   | 37.5  | 51.6      | 58.2   |
| 2DSSG_ORBSLAM3_l20_7    | 20.7  | 46.3      | 27.6   |
| 2DSSG_ORBSLAM3_l20_7*   | 35.9  | 48.6      | 54.8   |
| IMG_full_l20_2_1*       | 32.1  | 44.5      | 53.3   |
| IMG_full_l20_2_1        | 20.6  | 42.4      | 30.8   |
| IMG_full_l20_1_1        | 1.5   | -         | 6      |
| 2DSSG_INSEG_l20_0       | 32.9  | 53.5      | 47.2   |
| 2DSSG_INSEG_l20_0*      | 43.6  | 56.1      | 61.1   |
| 2DSSG_full_l20_1        | 52.2  | 64.4      | 75.9   |
| VGfM_full_l20_0*        | 21.4  | 36.5      | 34.7   |
| VGfM_full_l20_0         | 12.7  | 34.8      | 20.7   |
| 2DSSG_full_l20_2        | 55.1  | 66.6      | 79.4   |
| IMP_full_l20_2_2        | 19.7  | 42.8      | 30.6   |
| IMP_full_l20_2_2*       | 30.7  | 45.1      | 52.5   |
| VGfM_full_l20_2*        | 21.7  | 41.4      | 36.6   |
| VGfM_full_l20_2         | 12.9  | 39.3      | 21.8   |
| 2DSSG_ORBSLAM3_l20_6_1  | 20.7  | 44.0      | 27.9   |
| 2DSSG_ORBSLAM3_l20_6_1* | 35.2  | 46.2      | 55.8   |
| 2DSSG_ORBSLAM3_l20_7_1  | 18.9  | 41.9      | 26.1   |
| 2DSSG_ORBSLAM3_l20_7_1* | 33.5  | 44.0      | 52.4   |
| IMP_ORBSLAM3_l20_1      | 0.3   | 6.7       | 0.4    |
| IMP_ORBSLAM3_l20_1*     | 31.3  | 43.7      | 38.2   |
| VGfM_ORBSLAM3_l20_2     | 0.5   | 11.4      | 0.6    |
| VGfM_ORBSLAM3_l20_2*    | 4.6   | 13.3      | 8      |
| VGfM_full_l20_4         | 25.6  | 47.2      | 39.1   |
| VGfM_full_l20_5         | 11.97 | 25.5      | 25.3   |
| VGfM_full_l20_UB        | 93.4  | 95.2      | 98.1   |

#### Predicate
| method                  | IoU   | Precision | Recall |
| ----------------------- | ----- | --------- | ------ |
| SGFN_full_0_3           | 0.325 | 0.329     | 0.656  |
| SGFN_inseg_0_5          | 32.1  | 38.9      | 48.4   |
| 3DSSG_full_l20_1        | 0.194 | 0.304     | 0.283  |
| 2DSSG_full_1_2          | 0.308 | 0.329     | 0.802  |
| 2DSSG_full_1_3          | 0.409 | 0.460     | 0.730  |
| 2DSSG_ORBSLAM3_l20_0    | 16.2  | 27.6      | 17.3   |
| 2DSSG_ORBSLAM3_l20_1    | 20.8  | 33.2      | 25.2   |
| 2DSSG_ORBSLAM3_l20_1*   | 24.6  | 35.6      | 41.5   |
| 2DSSG_ORBSLAM3_l20_2*   | 30.6  | 39.0      | 58.6   |
| 2DSSG_ORBSLAM3_l20_2    | 21.8  | 38.9      | 28.3   |
| SGFN_inseg_0_5*         | 36.2  | 38.9      | 68.0   |
| 2DSSG_ORBSLAM3_l20_4*   | 30.4  | 36.3      | 56.3   |
| 2DSSG_ORBSLAM3_l20_4    | 22.5  | 36.2      | 27.6   |
| 2DSSG_ORBSLAM3_l20_3    | 22.2  | 39.0      | 25.6   |
| 2DSSG_ORBSLAM3_l20_3*   | 31.5  | 39.1      | 50.6   |
| 2DSSG_ORBSLAM3_l20_5*   | 31.8  | 37.7      | 56.5   |
| 2DSSG_ORBSLAM3_l20_5    | 22.8  | 376       | 27.7   |
| 2DSSG_ORBSLAM3_l20_6    | 22.7  | 39.0      | 27.4   |
| 2DSSG_ORBSLAM3_l20_6*   | 32.4  | 39.1      | 55.8   |
| 2DSSG_ORBSLAM3_l20_7    | 21.6  | 37.7      | 25.1   |
| 2DSSG_ORBSLAM3_l20_7*   | 30.2  | 37.4      | 49.8   |
| IMG_full_l20_2_1*       | 30.7  | 43.7      | 45.3   |
| IMG_full_l20_2_1        | 23.4  | 43.8      | 27.5   |
| IMG_full_l20_1_1        | 11.9  | -         | 12.5   |
| 2DSSG_INSEG_l20_0       | 34.8  | 50.2      | 42.3   |
| 2DSSG_INSEG_l20_0*      | 43.1  | 50.3      | 57.7   |
| 2DSSG_full_l20_1        | 44.7  | 50.4      | 71.2   |
| VGfM_full_l20_0*        | 21.6  | 28.2      | 34.0   |
| VGfM_full_l20_0         | 18.7  | 28.2      | 23.4   |
| 2DSSG_full_l20_2        | 45.2  | 51.4      | 70.3   |
| IMP_full_l20_2_2        | 24.3  | 43.8      | 28.2   |
| IMP_full_l20_2_2*       | 31.6  | 43.6      | 45.4   |
| VGfM_full_l20_2*        | 21.1  | 26.9      | 41.8   |
| VGfM_full_l20_2         | 19.0  | 26.9      | 27.7   |
| 2DSSG_ORBSLAM3_l20_6_1  | 22.5  | 38.7      | 27.8   |
| 2DSSG_ORBSLAM3_l20_6_1* | 31.8  | 38.8      | 59.2   |
| 2DSSG_ORBSLAM3_l20_7_1  | 23.0  | 35.9      | 29.2   |
| 2DSSG_ORBSLAM3_l20_7_1* | 31.4  | 36.0      | 58.6   |
| IMP_ORBSLAM3_l20_1      | 11.9  | 48.5      | 12.5   |
| IMP_ORBSLAM3_l20_1*     | 31.3  | 43.7      | 38.2   |
| VGfM_ORBSLAM3_l20_2     | 12.0  | 33.8      | 12.5   |
| VGfM_ORBSLAM3_l20_2*    | 22.6  | 31.7      | 39.5   |
| VGfM_full_l20_4         | 26.9  | 43.8      | 33.6   |
| VGfM_full_l20_5         | 17.4  | 45.2      | 18.9   |
| VGfM_full_l20_UB        | 36.7  | 99.5      | 37.3   |



Note: remember to recalculate average. (ignore none in recall and iou)

# Some Setup
## LR schedular
- 0: reduceluronplateau, factor=0.1
- 1: reduceluronplateau, factor=0.9





# MVCNN
- mv_res18_3rscan: trainng.
