
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

9: the final result is worse than the one in the paper. Rel. Pred. R@50:0.75>0.4. R@100:0.77>0.66. Obj. 
Pred.: R@5:0.49<0.68. R@10:0.64<0.78. Pred. Pred.:R@30.36<0.89. R@5: 0.51<0.93.

This may due to the weighting method. and the number of GNN iterations.

- 10: normalize edge weight 
- 11: 5 layers
