import importlib
import os

import torch

from .cuda_linalg import configure_linalg_backend


VALID_DEVICE_CHOICES = {"auto", "orin", "desktop"}


def _read_device_override():
    value = os.environ.get("BOBA_DEVICE", "auto").strip().lower()
    if not value:
        value = "auto"

    if value not in VALID_DEVICE_CHOICES:
        allowed = ", ".join(sorted(VALID_DEVICE_CHOICES))
        raise RuntimeError(
            f"Invalid BOBA_DEVICE value {value!r}. Expected one of: {allowed}."
        )

    return value


def _detect_device_name():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "Boba_OpenSource requires a CUDA-capable NVIDIA GPU. "
            "No supported CUDA device was detected."
        )

    try:
        device_name = torch.cuda.get_device_name(0).strip()
    except Exception as exc:
        raise RuntimeError(
            "Boba_OpenSource could not read the active CUDA device name."
        ) from exc

    if not device_name:
        raise RuntimeError(
            "Boba_OpenSource could not identify the active CUDA device."
        )

    return device_name


def _select_variant(override, device_name):
    if override == "orin":
        return "orin"
    if override == "desktop":
        return "desktop"

    lowered = device_name.lower()
    if "orin" in lowered or "jetson" in lowered:
        return "orin"
    return "desktop"


def _configure_linalg_backend():
    return configure_linalg_backend(torch_module=torch)


BOBA_DEVICE = _read_device_override()
DETECTED_DEVICE_NAME = _detect_device_name()
SELECTED_DYNAMIC_UTIL_VARIANT = _select_variant(BOBA_DEVICE, DETECTED_DEVICE_NAME)
SELECTED_LINALG_BACKEND = _configure_linalg_backend()
SELECTED_DYNAMIC_UTIL_MODULE = (
    "gaussian_splatting.dynamic_utils_fp16_no_profiling_orin"
    if SELECTED_DYNAMIC_UTIL_VARIANT == "orin"
    else "gaussian_splatting.dynamic_utils_fp16_no_profiling"
)

_IMPL = importlib.import_module(SELECTED_DYNAMIC_UTIL_MODULE)
_EXPORTED_NAMES = getattr(
    _IMPL,
    "__all__",
    [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))],
)
for _name in _EXPORTED_NAMES:
    globals()[_name] = getattr(_IMPL, _name)

__all__ = sorted(
    set(
        _EXPORTED_NAMES
        + [
            "BOBA_DEVICE",
            "DETECTED_DEVICE_NAME",
            "SELECTED_LINALG_BACKEND",
            "SELECTED_DYNAMIC_UTIL_VARIANT",
            "SELECTED_DYNAMIC_UTIL_MODULE",
            "VALID_DEVICE_CHOICES",
        ]
    )
)
