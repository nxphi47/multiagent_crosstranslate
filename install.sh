
#export CUDA_HOME=/usr/local/cuda
export cur=$PWD
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd $PWD