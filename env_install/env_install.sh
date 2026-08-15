#!/usr/bin/env bash
set -euo pipefail

# Runtime and evaluation dependencies for the public Boba branch.
# Assumes:
#   1. a Linux NVIDIA machine with a CUDA 12.1-compatible toolkit/driver
#   2. the target conda environment is named "phystwin"
#   3. desktop OpenGL / X11 system libraries are installed outside conda

ENV_NAME="phystwin"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda install -y -n "${ENV_NAME}" \
  numpy==1.26.4 \
  scipy \
  pyyaml \
  pillow \
  matplotlib \
  opencv \
  pytorch==2.4.0 \
  torchvision==0.19.0 \
  torchaudio==2.4.0 \
  pytorch-cuda=12.1 \
  -c pytorch \
  -c nvidia

conda run -n "${ENV_NAME}" env PYTHONNOUSERSITE=1 python -m pip install \
  termcolor \
  imageio \
  imageio-ffmpeg \
  warp-lang \
  open3d \
  glfw \
  PyOpenGL \
  pycuda \
  kornia \
  plyfile

conda run -n "${ENV_NAME}" env PYTHONNOUSERSITE=1 python -m pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html

conda run -n "${ENV_NAME}" env PYTHONNOUSERSITE=1 BUILD_NO_CUDA=1 \
  python -m pip install -e "${REPO_ROOT}/gaussian_splatting/submodules/gsplat"

conda run -n "${ENV_NAME}" env PYTHONNOUSERSITE=1 python -m pip install \
  "${REPO_ROOT}/gaussian_splatting/submodules/simple-knn/"
