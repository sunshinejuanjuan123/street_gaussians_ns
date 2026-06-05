#!/usr/bin/env bash
# Generate LiDAR depth maps for 3DGS training depth supervision.
set -euo pipefail

data_root=$1
gs_data_root=$2

python scripts/pythons/generate_lidar_depth.py \
    --data_root "$data_root" \
    --gs_data_root "$gs_data_root"
