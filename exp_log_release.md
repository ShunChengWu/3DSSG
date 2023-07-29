# GT
- [x] IMP
- [x] VGfM
- [x] 3DSSG
- [x] SGFN
- [x] JointSSG
# INSEG
- [x] IMP
- [x] VGfM
- [x] 3DSSG     
- [x] SGFN
- [x] JointSSG
# ORBSLAM
- [x] IMP
- [x] VGfM
- [x] 3DSSG     
- [x] SGFN
- [x] JointSSG
| Name     | Input  | Trip.    | Obj.     | Pred.    | Trip.*   | Obj.*    | Pred.*   | mRe.Obj. | mRe.Pred. |
| -------- | ------ | -------- | -------- | -------- | -------- | -------- | -------- | -------- | --------- |
| IMP      | GT     | 45.3     | 65.4     | 94.0     | 44.3     | 66.0     | 56.6     | 56.2     | 41.8      |
| VGfM     | GT     | 52.9     | 70.8     | 95.0     | 51.5     | 71.4     | 62.8     | 59.5     | 46.8      |
| 3DSSG    | GT     | 31.8     | 55.1     | 95.4     | 39.7     | 55.6     | 71.0     | 47.7     | 61.5      |
| SGFN     | GT     | 42.7     | 63.6     | 94.3     | 47.6     | 64.4     | 69.0     | 53.6     | 63.1      |
| JointSSG | GT     | **63.9** | **79.4** | **95.6** | **63.4** | **80.0** | **76.0** | **78.2** | **64.8**  |
|          |        |          |          |          |          |          |          |          |           |
| IMP      | DENSE  | 24.6     | 47.7     | 89.2     | 19.7     | 49.5     | 20.9     | 34.7     | 23.9      |
| VGfM     | DENSE  | 25.9     | 48.4     | **90.4** | 19.6     | 50.0     | 20.4     | 34.8     | 21.5      |
| 3DSSG    | DENSE  | 14.5     | 37.0     | 88.0     | 12.9     | 37.4     | 22.0     | 26.2     | 23.7      |
| SGFN     | DENSE  | 27.7     | 49.7     | 89.9     | 22.0     | 51.6     | 27.5     | 37.7     | 32.6      |
| JointSSG | DENSE  | **29.5** | **52.0** | 88.6     | **23.3** | **53.8** | **28.4** | **43.8** | **35.8**  |
|          |        |          |          |          |          |          |          |          |           |
| IMP      | SPARSE | 8.6      | 27.7     | **90.9** | 3.6      | 24.5     | 4.0      | 20.2     | 14.7      |
| VGfM     | SPARSE | 9.0      | 28.0     | 90.7     | 4.0      | 28.8     | 4.4      | 24.3     | 13.9      |
| 3DSSG    | SPARSE | 1.3      | 11.1     | 90.2     | 1.0      | 11.7     | 4.6      | 6.1      | 13.9      |
| SGFN     | SPARSE | 2.5      | 15.4     | 88.3     | 3.4      | 15.9     | 7.0      | 8.9      | 14.5      |
| JointSSG | SPARSE | **9.9**  | **28.7** | 89.8     | **6.8**  | **29.5** | **8.2**  | **27.0** | **17.6**  |


The number of `GT` and `DENSE` are not the same as reported in the paper, but they follow the share the same trend. (Haven't compared jointSSG since the training is not yet finished. JointSSG should have the highest number in GT and Dense)

# 160
- [x] IMP
- [x] VGfM
- [x] 3DSSG     
- [x] SGFN
- [x] JointSSG
| Name     | Input | Trip. | Obj. | Pred. | Trip.* | Obj.* | Pred.* | mRe.Obj. | mRe.Pred. |
| -------- | ----- | ----- | ---- | ----- | ------ | ----- | ------ | -------- | --------- |
| IMP      | GT    | 64.2  | 43.0 | 16.2  | 4.9    | 42.9  | 16.4   | 16.0     | 3.6       |
| VGfM     | GT    | 64.5  | 46.0 | 17.4  | 5.9    | 46.0  | 17.6   | 19.1     | 5.5       |
| 3DSSG    | GT    | 64.8  | 28.0 | 67.1  | 6.9    | 27.9  | 67.1   | 12.1     | 20.9      |
| SGFN     | GT    | 64.7  | 36.9 | 48.4  | 6.6    | 36.8  | 48.4   | 16.2     | 14.4      |
| JointSSG | GT    | 67.6  | 53.4 | 48.1  | 14.8   | 53.2  | 48.1   | 28.9     | 24.7      |