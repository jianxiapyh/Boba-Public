"""Direct small-matrix cuSOLVER binding; built once in PyTorch's extension cache."""

import threading
from pathlib import Path

from .cuda_linalg import require_runtime

_extension = None
_build_lock = threading.Lock()


def extension():
    global _extension
    if _extension is not None:
        return _extension
    with _build_lock:
        if _extension is None:
            require_runtime()
            import torch
            from torch.utils.cpp_extension import CUDA_HOME, load

            if CUDA_HOME is None:
                raise RuntimeError(
                    "CUDA toolkit not found. Activate phystwin-cu132 and set "
                    "CUDA_HOME=$CONDA_PREFIX before starting Boba."
                )
            toolkit = Path(CUDA_HOME)
            includes = [str(toolkit / "include")]
            includes.extend(str(path) for path in sorted(toolkit.glob("targets/*/include")))
            _extension = load(
                name="boba_cusolver_eigh",
                sources=[str(Path(__file__).parent / "csrc" / "eigh_3x3.cpp")],
                extra_include_paths=includes,
                extra_cflags=["-O2"],
                extra_ldflags=[
                    "-ltorch_cuda_linalg", "-lcusolver",
                    f"-Wl,-rpath,{Path(torch.__file__).parent / 'lib'}",
                ],
                with_cuda=True,
                verbose=False,
            )
    return _extension


if __name__ == "__main__":
    import torch
    from .cuda_linalg import configure_linalg_backend
    configure_linalg_backend(torch)
    extension().eigh_3x3(torch.eye(3, device="cuda").unsqueeze(0))
    print(f"Boba cuSOLVER syevjBatched ready on {torch.cuda.get_device_name()}")
