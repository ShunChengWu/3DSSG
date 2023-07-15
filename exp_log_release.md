# GT
- [x] IMP
- [x] VGfM
- [x] 3DSSG
- [x] SGFN
- [ ] JointSSG
# INSEG
- [x] IMP
- [x] VGfM
- [x] 3DSSG
- [x] SGFN
- [ ] JointSSG

| Name     | Input  | Re.Trip. | Re.Obj. | Re.Pred. | Pre.Trip. | Pre.Obj. | Pre.Pred. | mRe.Obj. | mRe.Pred. |
| -------- | ------ | -------- | ------- | -------- | --------- | -------- | --------- | -------- | --------- |
| IMP      | GT   | 45.3     | 65.4    | 94.0     | 44.3      | 66.0     | 56.6      | 56.2     | 41.8      |
| VGfM     | GT   | 52.9     | 70.8    | 95.0     | 51.5      | 71.4     | 62.8      | 59.5     | 46.8      |
| 3DSSG    | GT   | 31.8     | 55.1    | 95.4     | 39.7      | 55.6     | 71.0      | 47.7     | 61.5      |
| SGFN     | GT   | 42.7     | 63.6    | 94.3     | 47.6      | 64.4     | 69.0      | 53.6     | 63.1      |
| JointSSG(not finished yet) | GT   | 52.3     | 71.9    | 91.0     | 57.3      | 71.6     | 78.2      | 79.5     | 78.2      |
| IMP      | DENSE  | 24.6     | 47.7    | 89.2     | 19.7      | 49.5     | 20.9      | 34.7     | 23.9      |
| VGfM     | DENSE  | 25.9     | 48.4    | 90.4     | 19.6      | 50.0     | 20.4      | 34.8     | 21.5      |
| 3DSSG    | DENSE  | 14.5     | 37.0    | 88.0     | 12.9      | 37.4     | 22.0      | 26.2     | 23.7      |
| SGFN     | DENSE  | 27.7     | 49.7    | 89.9     | 22.0      | 51.6     | 27.5      | 37.7     | 32.6      |
| JointSSG(not finished yet) | DENSE  | 20.7     | 45.0    | 86.0     | 20.5      | 46.7     | 27.3      | 37.0     | 33.3      |
| IMP      | SPARSE |
| VGfM     | SPARSE |
| 3DSSG    | SPARSE |
| SGFN     | SPARSE |
| JointSSG | SPARSE |

The number of `GT` and `DENSE` are not the same as reported in the paper, but they follow the share the same trend. (Haven't compared jointSSG since the training is not yet finished. JointSSG should have the highest number in GT and Dense)
