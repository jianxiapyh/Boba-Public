"""Fixed production CUDA linear-algebra policy for Boba."""

from __future__ import annotations

from typing import Optional, Tuple


CUSOLVER_BACKEND = "cusolver"
MIN_CUDA_MAJOR_FOR_COMPUTE_CAPABILITY_12 = 13


def cuda_build_major(cuda_build: Optional[str]) -> Optional[int]:
    if cuda_build is None:
        return None
    try:
        return int(str(cuda_build).strip().split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def normalize_device_capability(device_capability) -> Tuple[int, int]:
    try:
        major, minor = device_capability
        major = int(major)
        minor = int(minor)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Unable to interpret CUDA device capability {device_capability!r}."
        ) from exc
    if major < 0 or minor < 0:
        raise RuntimeError(
            f"Unable to interpret CUDA device capability {device_capability!r}."
        )
    return major, minor


def validate_cuda_compatibility(
    cuda_build: Optional[str],
    device_capability,
) -> Tuple[int, int]:
    capability = normalize_device_capability(device_capability)
    build_major = cuda_build_major(cuda_build)
    if build_major is None:
        raise RuntimeError(
            "Boba requires a CUDA-enabled PyTorch build with a parseable "
            f"torch.version.cuda value; found {cuda_build!r}. Use the supported "
            "phystwin environment, or phystwin-cu130 for the CUDA 13 "
            "compatibility path."
        )
    if (
        capability[0] >= 12
        and build_major < MIN_CUDA_MAJOR_FOR_COMPUTE_CAPABILITY_12
    ):
        raise RuntimeError(
            f"CUDA device capability {capability[0]}.{capability[1]} requires "
            "a PyTorch build against CUDA 13 or newer for Boba's cuSOLVER "
            f"runtime; found torch.version.cuda={cuda_build!r}. Use the "
            "phystwin-cu130 environment. The CUDA version shown by nvidia-smi "
            "is driver capability, not the CUDA version used to build PyTorch."
        )
    return capability


def configure_linalg_backend(torch_module) -> str:
    try:
        device_capability = torch_module.cuda.get_device_capability()
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "Boba could not read the active CUDA device capability. Confirm "
            "that a supported NVIDIA GPU is visible to PyTorch."
        ) from exc

    validate_cuda_compatibility(
        cuda_build=getattr(getattr(torch_module, "version", None), "cuda", None),
        device_capability=device_capability,
    )
    try:
        torch_module.backends.cuda.preferred_linalg_library(CUSOLVER_BACKEND)
    except AttributeError as exc:
        raise RuntimeError(
            "The active PyTorch build does not expose "
            "torch.backends.cuda.preferred_linalg_library()."
        ) from exc
    return CUSOLVER_BACKEND


__all__ = [
    "CUSOLVER_BACKEND",
    "MIN_CUDA_MAJOR_FOR_COMPUTE_CAPABILITY_12",
    "configure_linalg_backend",
    "cuda_build_major",
    "normalize_device_capability",
    "validate_cuda_compatibility",
]
