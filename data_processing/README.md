# Data preparation
### Download data from 3RScan
```
# at root direcotry of this repo
cd files
bash preparation.sh
```
* To download the 3RScan 3D data, visit their [project page](https://waldjohannau.github.io/RIO).
* After receiving the download script, run:
```
python download.py -o /path/to/3RScan/ --type semseg.v2.json
python download.py -o /path/to/3RScan/ --type labels.instances.annotated.v2.ply
``` 

* change DATA_PATH in `./utils/define.py` to your path to the 3RScan directory and ROOT_PATH to the path of this repository.

* Run `transform_ply.py` under `./data_processing/` to generate `labels.instances.align.annotated.ply`

# Generate training/evaluation data with GT segmentations
Use `gen_data_gt.py` to generate the training data in the 3DSSG paper.

# Generate training/evaluation data from estimated segmentations
1. Generate aligned pose and mesh.  
To generate incremental segmented scene, you will need to run our reconstruction framework with aligined pose and rendered view from 3RScan.  
Use one of the executable `rio_renderer_render_all` and `align_poses`  in the [3RScan repo](https://github.com/WaldJohannaU/3RScan) to generate them.
```
# ./rio_renderer_render_all <3RScan_path> <scan_id> <output_folder> <render_mode>
# <render_mode>: 0 = default (occlusion) = all; 1 = only images, 2 = only depth; 3 = images and bounding boxes; 4 = only bounding boxes
./rio_renderer_render_all ../../../data/3RScan 754e884c-ea24-2175-8b34-cead19d4198d sequence 2

# ./align_poses <3RScan_path> <scan_id> <output_folder>
./align_poses ../../../data/3RScan 754e884c-ea24-2175-8b34-cead19d4198d sequence
```

2. Generate estimated segment map.   
   Run InSeg incremental segmentation method to generate `inseg.ply` files. 
    1. clone `https://github.com/ShunChengWu/SceneGraphFusion` to `${PATH_SGFUSION}`and compile
    2. change `exe_path` in `./script/RUN_GenSeg.py` to `${PATH_SGFUSION}/bin/exe_GraphSLAM`
    3. Go to `./script` and run
```
python RUN_GenSeg.py --dataset 3RScan --type validation --thread 8
python RUN_GenSeg.py --dataset 3RScan --type train --thread 8
```
3. Generate data with 3RScan dataset 
```
python gen_data.py 
  --scans /path/to/3RScan/folder/ 
  --type [train/test/validation] 
  --label_type ['ScanNet20', '3RScan160']
  --pth_out /path/to/an/output/folder/
```

You can also use the script `generate_train_valid_test_splits.py` to generate custom splits for training the data. 
Example:
```
# Generate splits
python generate_train_valid_test_splits.py --pth_out ./tmp/ 
# Train
python gen_data.py --type train --label_type ScanNet20 --pth_out ../data/example_data/ --target_scan tmp/train_scans.txt --min_seg_size 256;
# Valid
python gen_data.py --type validation --label_type ScanNet20 --pth_out ../data/example_data/ --target_scan tmp/validation_scans.txt --min_seg_size 256;
# Test
python gen_data.py --type test --label_type ScanNet20 --pth_out ../data/example_data/ --target_scan tmp/test_scans.txt --min_seg_size 256;
```

4. Generate data with ScanNet dataset
```
python gen_data_scannet.py 
  --scans /path/to/scannet/scans/ 
  --txt /path/to/scan/split.txt
  --pth_out /path/to/an/output/folder/
```

[3rscan]: https://waldjohannau.github.io/RIO/
[scannet]: http://www.scan-net.org/
