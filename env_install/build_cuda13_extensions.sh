#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTIVE_PREFIX="${CONDA_PREFIX:-}"
RUNTIME_HOOK_ROOT="${REPO_ROOT}/env_install/conda"

EXPECTED_ENV="phystwin-cu132"
EXPECTED_TORCH_VERSION="2.12.1+cu132"
EXPECTED_TORCH_CUDA="13.2"
CACHE_TAG="cu132"

if [[ -z "${ACTIVE_PREFIX}" || "$(basename "${ACTIVE_PREFIX}")" != "${EXPECTED_ENV}" ]]; then
  echo "Activate ${EXPECTED_ENV} before rebuilding CUDA extensions." >&2
  exit 2
fi
if [[ ! -f "${ACTIVE_PREFIX}/include/GL/gl.h" ]]; then
  echo "Missing GL/gl.h; update ${EXPECTED_ENV} from env_install/${EXPECTED_ENV}.yml." >&2
  exit 3
fi

# Keep the workstation's system CUDA installation out of this CUDA-13 build.
export PYTHONNOUSERSITE=1
export CUDA_HOME="${ACTIVE_PREFIX}"
export PATH="${ACTIVE_PREFIX}/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
ACTIVE_SITE_PACKAGES="$("${ACTIVE_PREFIX}/bin/python" -c \
  'import sysconfig; print(sysconfig.get_path("platlib"))')"
export LD_LIBRARY_PATH="${ACTIVE_SITE_PACKAGES}/nvidia/cu13/lib:${ACTIVE_SITE_PACKAGES}/torch/lib:${ACTIVE_PREFIX}/lib:${ACTIVE_PREFIX}/targets/x86_64-linux/lib"
export TORCH_CUDA_ARCH_LIST="$(python "${REPO_ROOT}/env_install/cuda_arch.py")"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_EXTENSIONS_DIR="${ACTIVE_PREFIX}/var/cache/torch_extensions-boba-${CACHE_TAG}"
export WARP_CACHE_PATH="${ACTIVE_PREFIX}/var/cache/warp-boba-${CACHE_TAG}"

mkdir -p "${TORCH_EXTENSIONS_DIR}" "${WARP_CACHE_PATH}"
echo "CUDA extension architectures: ${TORCH_CUDA_ARCH_LIST}"

python - "${EXPECTED_TORCH_VERSION}" "${EXPECTED_TORCH_CUDA}" <<'PY'
import importlib.metadata
import os
import sys

import torch
from torch.utils.cpp_extension import CUDA_HOME

expected_torch_version = sys.argv[1]
expected_torch_cuda = sys.argv[2]
if torch.__version__ != expected_torch_version or torch.version.cuda != expected_torch_cuda:
    raise SystemExit(
        f"Expected torch {expected_torch_version} / CUDA {expected_torch_cuda}, found "
        f"{torch.__version__} / {torch.version.cuda}"
    )
if os.path.realpath(CUDA_HOME or "") != os.path.realpath(sys.prefix):
    raise SystemExit(f"CUDA_HOME resolved to {CUDA_HOME!r}, expected {sys.prefix!r}")

cuda12_packages = sorted(
    distribution.metadata["Name"]
    for distribution in importlib.metadata.distributions()
    if "cu12" in (distribution.metadata.get("Name") or "").lower()
)
if cuda12_packages:
    raise SystemExit(f"CUDA-12 Python packages are installed: {cuda12_packages}")
PY

nvcc --version

# PyPI wheels do not expose the CUDA/OpenGL interop API used by Boba's PBO
# renderer. Rebuild the pinned release from source with GL support enabled.
BOBA_CUDA_DRIVER_LIBRARY="$({ ldconfig -p 2>/dev/null || true; } | awk '
  $1 == "libcuda.so" && $0 ~ /x86-64/ && !found { print $NF; found = 1 }
')"
if [[ -z "${BOBA_CUDA_DRIVER_LIBRARY}" || ! -f "${BOBA_CUDA_DRIVER_LIBRARY}" ]]; then
  echo "Unable to locate the x86-64 NVIDIA driver library (libcuda.so)." >&2
  exit 4
fi
BOBA_CUDA_DRIVER_DIR="$(dirname "${BOBA_CUDA_DRIVER_LIBRARY}")"
PYCUDA_BUILD_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${PYCUDA_BUILD_ROOT}"' EXIT

python -m pip download pycuda==2026.1 --no-binary=pycuda --no-deps \
  --dest "${PYCUDA_BUILD_ROOT}"
tar -xf "${PYCUDA_BUILD_ROOT}/pycuda-2026.1.tar.gz" \
  -C "${PYCUDA_BUILD_ROOT}"
(
  cd "${PYCUDA_BUILD_ROOT}/pycuda-2026.1"
  python configure.py \
    --cuda-root="${ACTIVE_PREFIX}" \
    --cuda-inc-dir="${ACTIVE_PREFIX}/targets/x86_64-linux/include" \
    --cuda-enable-gl \
    --cudadrv-lib-dir="${BOBA_CUDA_DRIVER_DIR}" \
    --cudart-lib-dir="${ACTIVE_PREFIX}/lib" \
    --curand-lib-dir="${ACTIVE_PREFIX}/lib"
  python -m pip install --no-build-isolation --no-deps --force-reinstall -v .
)

python -m pip install --no-build-isolation --no-deps --force-reinstall -v \
  "${REPO_ROOT}/gaussian_splatting/submodules/simple-knn"
python -m pip install --no-build-isolation --no-deps --force-reinstall -v \
  "${REPO_ROOT}/gaussian_splatting/submodules/fused-ssim"
python -m pip install --no-build-isolation --no-deps --force-reinstall -v -e \
  "${REPO_ROOT}/gaussian_splatting/submodules/gsplat"

python - "${TORCH_CUDA_ARCH_LIST}" "${REPO_ROOT}" <<'PY'
import importlib
import sys
import subprocess

import torch
import pycuda._driver as pycuda_driver

sys.path.insert(0, sys.argv[2])
from env_install.cuda_arch import expected_sm_targets, parse_arch_list

requested_architectures = parse_arch_list(sys.argv[1])
requested_sm_targets = expected_sm_targets(requested_architectures)

if not pycuda_driver.have_gl_ext():
    raise SystemExit("PyCUDA was built without required CUDA/OpenGL interop support")
importlib.import_module("pycuda.gl")

modules = {
    "simple_knn": importlib.import_module("simple_knn._C"),
    "fused_ssim": importlib.import_module("fused_ssim_cuda"),
}
gsplat_backend = importlib.import_module("gsplat.cuda._backend")
modules["gsplat"] = gsplat_backend._C

for name, module in modules.items():
    path = module.__file__
    ldd = subprocess.run(
        ["ldd", path], text=True, stdout=subprocess.PIPE, check=True
    ).stdout
    if "libcudart.so.12" in ldd or "libcudart.so.13" not in ldd:
        raise SystemExit(f"{name} has an unexpected CUDA runtime mapping:\n{ldd}")
    cubins = subprocess.run(
        ["cuobjdump", "--list-elf", path],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    missing_targets = [target for target in requested_sm_targets if target not in cubins]
    if missing_targets:
        raise SystemExit(
            f"{name} is missing requested cubins {missing_targets}:\n{cubins}"
        )
    print(
        f"{name}: {path} (libcudart.so.13, "
        f"{', '.join(requested_sm_targets)})"
    )

probe = torch.eye(3, dtype=torch.float32, device="cuda").unsqueeze(0)
torch.backends.cuda.preferred_linalg_library("cusolver")
from gaussian_splatting.rotation_utils import eigh_3x3
eigh_3x3(probe)
torch.cuda.synchronize()
PY

# Persist the environment-local runtime paths only after every build and
# verification step has succeeded. Conda sources these hooks for both
# interactive activation and `conda run`.
ACTIVATE_HOOK_DIR="${ACTIVE_PREFIX}/etc/conda/activate.d"
DEACTIVATE_HOOK_DIR="${ACTIVE_PREFIX}/etc/conda/deactivate.d"
install -d "${ACTIVATE_HOOK_DIR}" "${DEACTIVATE_HOOK_DIR}"
install -m 0644 \
  "${RUNTIME_HOOK_ROOT}/activate.d/boba-cu130-runtime.sh" \
  "${ACTIVATE_HOOK_DIR}/boba-cu130-runtime.sh"
install -m 0644 \
  "${RUNTIME_HOOK_ROOT}/deactivate.d/boba-cu130-runtime.sh" \
  "${DEACTIVATE_HOOK_DIR}/boba-cu130-runtime.sh"

echo "Installed portable CUDA 13 runtime hooks in ${ACTIVE_PREFIX}."
echo "Reactivate ${EXPECTED_ENV} before running Boba in the current shell."
