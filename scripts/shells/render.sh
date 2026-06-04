#!/usr/bin/env bash
set -euo pipefail

config_path=$1
vehicle_config=$2
cuda_id=$3

setup_gsplat_cuda() {
    local cuda_candidates=(
        "${CUDA_HOME:-}"
        "/usr/local/cuda-12.4"
        "/usr/local/cuda-12"
        "/usr/local/cuda"
        "/usr/local/cuda-11.8"
    )
    for cuda_root in "${cuda_candidates[@]}"; do
        [ -z "$cuda_root" ] && continue
        if [ -x "${cuda_root}/bin/nvcc" ]; then
            export CUDA_HOME="$cuda_root"
            export PATH="${cuda_root}/bin:${PATH}"
            return 0
        fi
    done
    if command -v nvcc >/dev/null 2>&1; then
        export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
        return 0
    fi
    return 1
}

warmup_gsplat() {
    setup_gsplat_cuda || return 1
    python - <<'PY' || return 1
from gsplat.cuda._backend import _C
if _C is None:
    raise SystemExit("gsplat CUDA extension (_C) is None; install nvcc or set CUDA_HOME")
PY
}

setup_gsplat_cuda
warmup_gsplat || exit 1

CUDA_VISIBLE_DEVICES=$cuda_id sgn-render \
    --load-config "$config_path" \
    --vehicle-config "$vehicle_config" \
    --render-camera-model fisheye
