CONFIG="experiments/config_IMP_FULL_l160_2.yaml"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ python main.py --mode train --config $CONFIG

CONFIG="experiments/config_IMP_full_l20_4.yaml"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ python main.py --mode train --config $CONFIG

CONFIG="experiments/config_IMP_INSEG_l20_2.yaml"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ python main.py --mode train --config $CONFIG

CONFIG="experiments/config_IMP_ORBSLAM3_l20_2.yaml"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ python main.py --mode train --config $CONFIG

CONFIG="experiments/config_VGfM_FULL_l160_3.yaml"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ python main.py --mode train --config $CONFIG

CONFIG="experiments/config_VGfM_full_l20_6.yaml"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ python main.py --mode train --config $CONFIG

CONFIG="experiments/config_VGfM_INSEG_l20_3.yaml"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ python main.py --mode train --config $CONFIG

CONFIG="experiments/config_VGfM_ORBSLAM3_l20_4.yaml"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ python main.py --mode train --config $CONFIG
