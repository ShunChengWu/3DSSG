source ~/.bashrc
false | conda create -n 2dssg pytorch=1.11 torchvision cudatoolkit=10.2 pyg -c pytorch -c pyg
conda activate 2dssg;
python -m pip install -U wandb trimesh matplotlib h5py plyfile open3d onnxruntime tensorboard graphviz