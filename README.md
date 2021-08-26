# SceneGraphFusion
![teaser](img/teaser.png)
**Authors**: [Shun-Cheng Wu][sc], [Johanna Wald][jojo], [Keisuke Tateno][keisu], [Nassir Navab][nassir] and [Federico Tombari][fede]

[sc]:http://campar.in.tum.de/Main/ShunChengWu
[keisu]:http://campar.in.tum.de/Main/KeisukeTateno
[jojo]:http://campar.in.tum.de/Main/JohannaWald
[nassir]:http://campar.in.tum.de/Main/NassirNavabCv
[fede]:http://campar.in.tum.de/Main/FedericoTombari

This repository contains the network part of the SceneGraphFusion work. For the incremental framework, 
please check [here](https://github.com/ShunChengWu/SceneGraphFusion).

# Dependencies
The code has been tested on Ubuntu 18.04 and gcc 7.5. You can either create a conda environment by 
```
conda env create --name <env_name> --file environment.yml
```
or install the dependnecies manually

```
###
# Dependencies
###
# for training and evaluation:
# - Pytorch, Pytorch Geometric, Trimesh, Tensorboard
# for tracing:
# - onnxruntime
# for data generation:
# - open3d
###
# Install commends 
###
# Main env
conda create -n 3dssg pytorch=1.8.9 cudatoolkit=10.2 -c pytorch tensorboard trimesh -c conda-forge
# Onnxruntime
pip install onnxruntime
# Pytorch Geometric
export CUDA=10.2
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-geometric
# open3d
pip install open3d
```

Instructions to use a Docker image: [instructions](Docker_instructions.md)

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


# Trace
The trained model can be traced and then be used on our [SceneGraphFusion](https://github.com/ShunChengWu/SceneGraphFusion) framework.

```
python main.py --mode trace --config ./path/to/config
```

For example, to trace our pre-trained model
```
python main.py --mode trace --config ./CVPR21/config_CVPR21.json
```

The traced model will be stored at
`./CVPR21/CVPR21/traced/`

# Generate Training Data
See README.md under [data_processing](data_processing/) folder

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
