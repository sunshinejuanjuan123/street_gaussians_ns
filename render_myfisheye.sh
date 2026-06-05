#!/usr/bin/env bash
PS4='+$LINENO: '
source /mnt/iag/yuanweizhong/miniconda3/bin/activate /mnt/iag/yuanweizhong/miniconda3/envs/gs-reconstruction


# rm -rf ~/.cache/torch_extensions/py310_cu*/gsplat_cuda
# A100，可选按 GPU 推断
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export TORCH_CUDA_ARCH_LIST=8.0
which nvcc
python -c "from gsplat.cuda._backend import _C; print('cached load OK:', _C is not None)"

pip install mediapy
pip install -e . --no-deps
CONFIG='/mnt/iag/yuanweizhong/datasets/pvb_delivery_11v_86125_uniscene/2026_03_07_14_11_11_dlp_pilotGtParser/2026_03_07_14_11_11_dlp_pilotGtParser_unknown_1772863892910819000_1772863912910819000/3dgs_format/v1/street-gaussians-ns/2026-06-04_092419/config.yml'
NVS="/mnt/iag/yuanweizhong/code/street_gaussians_ns/scripts/nvs_template_pvb_11v.json"
bash /mnt/iag/yuanweizhong/code/street_gaussians_ns/scripts/shells/render.sh "$CONFIG" "$NVS" 0