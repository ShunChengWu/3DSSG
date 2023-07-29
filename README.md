# Setup
## Environment.
```
# if you don't have miniconda
source setup_conda.sh 

# setup
source setup.sh

mkdir data
ln -s /path/to/your/3RScan ./data/

source Init.sh # This will set PYTHONPATH and activate the environment for you.
```

## Prepare 3RScan dataset
Please make sure you agree the [3RScan Terms of Use](https://forms.gle/NvL5dvB4tSFrHfQH6) first, and get the download script and put it right at the 3RScan main directory.

Then run
```
python script/Run_prepare_dataset_3RScan.py --download --thread 8
```

## Generate Experiment data
```
# For GT
# This script downloads preprocessed data for GT data generation, and generate GT data.
python scripts/RUN_prepare_GT_setup_3RScan.py --thread 16

# For Dense
# This script downloads the inseg.ply files and unzip them to your 3rscan folder, and 
generates training data.
python scripts/RUN_prepare_Dense_setup_3RScan.py -c configs/dataset/config_base_3RScan_inseg_l20.yaml --thread 16

# For Sparse
# This script downloads the 2dssg_orbslam3.[json,ply] files and unzip them to your 3rscan folder, and 
generates training data.
python scripts/RUN_prepare_Sparse_setup_3RScan.py -c configs/dataset/config_base_3RScan_orbslam_l20.yaml --thread 16
```

# Train 
The first time you may need want to chagne the wandb account in `configs/config_default.yaml`. Change the `wanb.entity` and `wanb.project` to yours. Or you can disable logging by passing `--dry_run`.
```
source Init.sh

# Train and eval everything. 
python scripts/RUN_traineval_all.py

# Train single
python main.py --mode train --config /path/to/your/config/file

# Eval one
python main.py --mode eval --config /path/to/your/config/file
```

# Trained models
We provide trained model using the optimized code (this one), instead of the one reported in our CVPR23 paper. Although the numbers are different but all methods follow the same trend. We encourage people compare to the results obtained by yourself using this repo.

<details>
  <summary>Results</summary>


The first **Trip. Obj. Pred.** are the result including all the predictions. The second **Trip.*, Obj.*, Pred.*** without considering `None` relationship.

With the same setup as the Table 1. 3RSca dataset with 20 objects and 8 predicate classes.
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

With the same setup as the Table 2. 3RSca dataset with 160 objects and 26 predicate classes.
| Name     | Input | Trip. | Obj. | Pred. | Trip.* | Obj.* | Pred.* | mRe.Obj. | mRe.Pred. |
| -------- | ----- | ----- | ---- | ----- | ------ | ----- | ------ | -------- | --------- |
| IMP      | GT    | 64.2  | 43.0 | 16.2  | 4.9    | 42.9  | 16.4   | 16.0     | 3.6       |
| VGfM     | GT    | 64.5  | 46.0 | 17.4  | 5.9    | 46.0  | 17.6   | 19.1     | 5.5       |
| 3DSSG    | GT    | 64.8  | 28.0 | 67.1  | 6.9    | 27.9  | 67.1   | 12.1     | 20.9      |
| SGFN     | GT    | 64.7  | 36.9 | 48.4  | 6.6    | 36.8  | 48.4   | 16.2     | 14.4      |
| JointSSG | GT    | 67.6  | 53.4 | 48.1  | 14.8   | 53.2  | 48.1   | 28.9     | 24.7      |

</details>




# ==================================================
This repo has been used for following publiscations

# Incremental 3D Semantic Scene Graph Prediction from RGB Sequences
**Authors**: [Shun-Cheng Wu][sc], [Keisuke Tateno][keisu], [Nassir Navab][nassir] and [Federico Tombari][fede]

# SceneGraphFusion
![teaser](img/teaser_SGFN.png)
**Authors**: [Shun-Cheng Wu][sc], [Johanna Wald][jojo], [Keisuke Tateno][keisu], [Nassir Navab][nassir] and [Federico Tombari][fede]

[sc]:http://campar.in.tum.de/Main/ShunChengWu
[keisu]:http://campar.in.tum.de/Main/KeisukeTateno
[jojo]:http://campar.in.tum.de/Main/JohannaWald
[nassir]:http://campar.in.tum.de/Main/NassirNavabCv
[fede]:http://campar.in.tum.de/Main/FedericoTombari

This repository contains the network part of the SceneGraphFusion work. For the incremental framework,
please check [here](https://github.com/ShunChengWu/SceneGraphFusion).




# Run
Run a toy example:
```
python main.py --mode [train,eval,trace] --config ./config_example.json
```
The main.py file will create a folder at the same directory of config with the NAME from config_[NAME] and a log folder
stors the logging from Tensorboard. The trained models/ evaluation results/ traced models will all be stored within the
NAME folder.

We provide a trained model [here](https://drive.google.com/file/d/1a2q7yMNNmEpUfC1_5Wuor0qDM-sBStFZ/view?usp=sharing). The model is able to perform equivelent result as reported in the SceneGraphFusion [paper][1].
Note: The model is trained with 20 NYUv2 object classes used in ScanNet benchmark, and with 8 support types of predicates.


# License
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

files under `./src/pointnet/*` are with Apache License, Version 2.0

`./src/network_PointNet.py` is modifed from https://github.com/charlesq34/pointnet under MIT License


# Repository structure
```
main.py # main file
src/             # main codes
src/network_*    # basic network/layers/operations
src/model_*      # a network model consists of multiple layers
src/dataset_*    # data related
src/*_base.py    # basic class templates
src/*_util*.py   # utilities
src/[name].py    # top-level class to train/eval/trace a model

data_processing/ # the codes to generate training data from 3RScan/ScanNet
utils/           # utilities
scripts/         # evaluation script/ scene reconstruction script/ etc.
```

### Paper
If you find the code useful please consider citing our [paper](https://arxiv.org/pdf/2103.14898.pdf):

```
@inproceedings{Wu2021,
    title = {{SceneGraphFusion: Incremental 3D Scene Graph Prediction from RGB-D Sequences}},
    author = {Shun-Cheng Wu and Johanna Wald and Keisuke Tateno and Nassir Navab and Federico Tombari},
    booktitle = {Proceedings IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}

@inproceedings{Wald2020,
    title = {{Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions}},
    author = {Wald, Johanna and Dhamo, Helisa and Navab, Nassir and Tombari, Federico},
    booktitle = {Proceedings IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2020}
}
```


[1]: https://arxiv.org/pdf/2103.14898.pdf