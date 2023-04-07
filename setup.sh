# Build conda env
source ~/.bashrc
false | conda create -n 2dssg pytorch=1.11 torchvision cudatoolkit=11.3 pyg -c pytorch -c pyg
conda activate 2dssg
python -m pip install -U wandb trimesh matplotlib h5py plyfile open3d\
 onnxruntime tensorboard graphviz pytictoc

# Download dat afiles
cd files; bash preparation.sh; cd ..;

# Build RIO renderer
git clone git@github.com:ShunChengWu/rio_renderer.git
cd rio_renderer;
cmake . -B build;
cmake --build build --config RelWithDebInfo -j