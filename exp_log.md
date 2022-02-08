# SSG with GT Segmentation
## SGPN
For `config_3DSSG_test_*.json`. 
- 0~3: try to make it work
- 4: train with batchsize 4.
- 5: train with batchsize 1.
- 6: no GCN
- 7: 2 layers GCN
- 8: 2 layers GCN. fix imp. with adding residual and aggr. 
- 9: fix hidden layer size in gnn. fix network init
4 and 5 are used to see if batch process is working.

4: doesn't train  
5: doesn't train  
6: has better curve than 4,5 at the begining. The problem may be the number of GCN layers.  

9: the final result is worse than the one in the paper. 

This may due to the weighting method. and the number of GNN iterations.

- 10: normalize edge weight 
- 11: 5 layers
- 12: 5l. node pd. w/. gnn

The result is still worse than the one reported from the paper. maybe increase input point size?

- 13: node_dim: 256->512. num_points_union: 512->1024
- 14: ndoe_dim: 1024, num_points_union: 2048
- 15: test with the optimized version of the dataset -> can reproduce 14. 

## SGFN
- 0: test. batchsize 0
- 1: batchsize 4


| method             | R@50 | R@100 | R@5  | R@10 | R@3  | R@10 |
|--------------------|------|------|------|------|------|------|
| 3DSSG_obj_wo_gcn   | 0.40 | 0.66 | 0.68 | 0.78 | 0.89 | 0.93 |
| 3DSSG_obj_from_gcn | 0.30 | 0.60 | 0.60 | 0.73 | 0.79 | 0.91 |
| exp0_9             | 0.75 | 0.77 | 0.49 | 0.64 | 0.36 | 0.51 |
| exp0_10            | 0.73 | 0.75 | 0.5  | 0.65 | 0.83 | 0.86 |
| exp0_12            | 0.75 | 0.77 | 0.57 | 0.70 | 0.81 | 0.92 | 
| exp0_13            | 0.77 | 0.80 | 0.63 | 0.76 | 0.85 | 0.97 |
| exp0_14            | 0.72 | 0.79 | 0.60 | 0.74 | 0.83 | 0.95 |
| SGFN               | 0.85 | 0.87 | 0.7  | 0.8  | 0.97 | 0.99 |
| exp1_0             | 0.91 | 0.92 | 0.71 | 0.82 | 0.94 | 1.00 |
| exp1_1             | 0.90 | 0.91 | 0.69 | 0.81 | 0.94 | 1.00 | 


## 2DSSG
img_batchsize: 8
exp0: MVCNN+VGG16. no GNN 
