# Build conda env
source ~/.bashrc
false | conda create -n 2dssg pytorch=1.11 torchvision cudatoolkit=11.3 pyg -c pytorch -c pyg
conda activate 2dssg
python -m pip install -U wandb trimesh matplotlib h5py plyfile open3d\
 onnxruntime tensorboard graphviz pytictoc
python -m pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Download dat afiles
cd files; bash preparation.sh; cd ..;