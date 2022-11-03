# TODO
- [ ] write a loader that load the sequential data
- [ ] allow jointSSG takes not cropped ROI input.

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

# Setup 
```
# this will create a conda environment 2dssg 
bash -i setup.sh 
```

# Data generation
## Generate aligned RGB
This steps is required if you want to train SceneGraphFusion with RGB.
```
cd script
python RUN_replace_color_to_real.py --help # check arugments
```
## Image graph
Please use `--help` to check the required input arguments on each script.
```
cd ssg/utils
# calculate occlusion & instance label information
python makebb_img_3rscan.py
# build up a img graph dataset connect with relationships
python make_obj_graph_3rscan.py -o output/dir -l [labeltype]
# extract object bounding boxes.
python extract_mv_box_image_3rscan.py
```
## InSeg
```
# Geenrate `inseg.ply` and `graph.json`
python script/RUN_GenSeg.py --dataset 3RScan --type validation --thread 8 --overwrite 1
python script/RUN_GenSeg.py --dataset 3RScan --type train --thread 8 --overwrite 1

# Generate training data
cd data_processing
python gen_data.py --pth_out ../data/3RScan_ScanNet20/ --target_scan ../data/3RScan_3RScan/train_scans.txt --type train --scan_name inseg.ply; python gen_data.py --pth_out ../data/3RScan_ScanNet20/ --target_scan ../data/3RScan_3RScan/validation_scans.txt --type validation --scan_name inseg.ply;
python gen_data.py --pth_out ../data/3RScan_ScanNet20/ --target_scan ../data/3RScan_3RScan/test_scans.txt --type test

# Generate object-image graph
PYTHONPATH=./ python ssg/utils/make_obj_graph_incremental.py -o ./data/3RScan_ScanNet20/ --overwrite 0

# extract mv images
PYTHONPATH=./ python ssg/utils/extract_mv_box_image_3rscan.py -o /media/sc/SSD1TB/dataset/3RScan/incremental/ -f ./data/3RScan_ScanNet20/proposals.h5 --thread 0 --overwrite 1
```

## 2DSSG
```
# Generate training data
PYTHONPATH=./ python script/RUN_Gen2DSSG.py --thread 4 --dataset 3RScan --type train --overwrite 0;
PYTHONPATH=./ python script/RUN_Gen2DSSG.py --thread 4 --dataset 3RScan --type validation --overwrite 0;

# generate test data with checkpoitns
PYTHONPATH=./ python script//RUN_Gen2DSSG_incre.py --thread 4

# generate point data
python gen_data_obj.py\
 --pth_out ../data/3RScan_ScanNet20_2DSSG_ORBSLAM3_3/\
 --target_scan ../data/3RScan_3RScan160/test_scans.txt\
 --type test \
 --label_type ScanNet20\
 --scan_name 2dssg_orbslam3 

# geneate object-image graph
PYTHONPATH=./ python ssg/utils/make_obj_graph_incremental.py -o ./data/3RScan_ScanNet20_2DSSG_ORBSLAM3/ --target_name 2dssg_orbslam3.json

# extract mv images
PYTHONPATH=./ python ssg/utils/extract_mv_box_image_3rscan.py -o /media/sc/SSD4TB/roi_2dssg_orbslam3 -f ./data/3RScan_ScanNet20_2DSSG_ORBSLAM3/proposals.h5 --thread 4
```


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


## Run on cluster
Copy data to cluster

```
rsync -ahz /home/sc/research/PersistentSLAM/python/3DSSG/data/3RScan_ScanNet20_gt /media/fileserver/gpucluster/workfiles/wsh/3DSSG/data/
```
Setup ssh key
```
bash /mnt/nfs-user/scripts/ssh_key.sh
```

Run setup
```
bash /mnt/nfs-user/scripts/setup_3dssg.sh
```