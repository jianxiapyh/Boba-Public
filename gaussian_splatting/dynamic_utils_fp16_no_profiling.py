import importlib.util
from pathlib import Path

import torch


_ORIN_MODULE_PATH = Path(__file__).with_name("dynamic_utils_fp16_no_profiling_orin.py")
_SPEC = importlib.util.spec_from_file_location(
    "gaussian_splatting._dynamic_utils_fp16_no_profiling_desktop_base",
    _ORIN_MODULE_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load dynamic utils implementation from {_ORIN_MODULE_PATH}")

_IMPL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPL)

try:
    _IMPL.quat_mul_norm_fused = torch.compile(dynamic=True)(_IMPL.quat_mul_norm_fused)
except AttributeError as exc:
    raise RuntimeError(
        "BOBA_DEVICE=desktop requires a PyTorch build with torch.compile support."
    ) from exc


IMPLEMENTATION_VARIANT = "desktop"
SOURCE_MODULE = _ORIN_MODULE_PATH.name

_EXPORTED_NAMES = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
for _name in _EXPORTED_NAMES:
    globals()[_name] = getattr(_IMPL, _name)

__all__ = sorted(set(_EXPORTED_NAMES + ["IMPLEMENTATION_VARIANT", "SOURCE_MODULE"]))
