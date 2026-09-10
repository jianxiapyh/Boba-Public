"""Shared production environment and cuSOLVER policy."""

import sys
from pathlib import Path

EXPECTED_CONDA_ENV = "phystwin-cu132"
CUSOLVER_BACKEND = "cusolver"


def require_runtime(torch_module=None):
    # The executing interpreter is authoritative, including under conda run.
    if Path(sys.prefix).resolve().name != EXPECTED_CONDA_ENV:
        raise RuntimeError(
            "Boba requires the 'phystwin-cu132' conda environment. "
            f"Python prefix: {sys.prefix!r}. Run: conda activate phystwin-cu132"
        )
    if torch_module is None:
        import torch as torch_module
    build = torch_module.version.cuda
    try:
        version = tuple(int(part) for part in build.split(".")[:2])
    except (AttributeError, ValueError):
        version = ()
    if version < (13, 2):
        raise RuntimeError(
            "Boba requires PyTorch built with CUDA 13.2 or newer in "
            f"phystwin-cu132; found torch.version.cuda={build!r}. "
            "The CUDA version shown by nvidia-smi describes the driver."
        )


def configure_linalg_backend(torch_module):
    require_runtime(torch_module)
    torch_module.backends.cuda.preferred_linalg_library(CUSOLVER_BACKEND)
    return CUSOLVER_BACKEND
