# Batched rotation eigendecomposition

The current runtime requires the `phystwin-cu132` interpreter and a PyTorch
CUDA 13.2+ build on every GPU. The public Boba_Batched, phone demo, and Quest
demo use the same `eigh_3x3` implementation. Old `BOBA_LINALG_BACKEND` values
no longer override the production cuSOLVER policy.

CUDA inference calls cuSOLVER's dedicated `syevjBatched` API once for the
whole batch of real symmetric 3x3 matrices, on Ampere, Ada and Blackwell alike.
It does not use a GPU-name exception or automatically split at 4,096 matrices.
This avoids the much larger temporary workspace observed when PyTorch chose
its generic eigensolver on Ada. CUDA 13.2 also avoids the older stack's batch
count limitation. Other simulation and rendering memory still scales with
the number of instances.

This changes the library entry point, not the physical simulation, the Gram
matrix construction, or the subsequent polar-rotation/LBS formulas. Each
branch keeps its existing regularization, clamping and rotation-cache policy.
The Jacobi solver sorts eigenvalues in ascending order, uses machine-precision
tolerance, allows 100 sweeps and checks convergence through PyTorch's error
handling. CPU and differentiable calls retain `torch.linalg.eigh` and its
backward implementation.

cuSOLVER overwrites its input with eigenvectors. The wrapper therefore copies
the input into a column-major output tensor, preserving the caller's matrix.
It is not a zero-copy solver. There is no new CPU decomposition or CPU copy
of the full matrix batch. The convergence check synchronizes with the CPU,
as `torch.linalg.eigh` does. Eigenvector signs and bases for repeated
eigenvalues are not unique; switching algorithms can change rounding,
especially for nearly singular deformations. Bitwise equivalence to every
previous PyTorch dispatch is not promised.

The small C++ binding uses PyTorch's device/stream-specific cuSOLVER handle
and caching allocator. It is compiled once per extension cache. A failed
build or solver call is reported; there is no silent fallback to the large
generic CUDA workspace. The first build requires Ninja, a C++ compiler and
CUDA 13.2+ headers/libraries in the existing environment.

Warm the extension before an event or latency measurement, from this repo:

```bash
conda activate phystwin-cu132
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_HOME="$CONDA_PREFIX"
PYTHONNOUSERSITE=1 python -m gaussian_splatting._cusolver_eigh
```

The phone preflight and public CUDA extension installer also warm this path.
The `test_eigh_3x3.py` suite covers FP32/FP64 residuals, singular and repeated
eigenvalues, lower-triangle semantics, input preservation, empty batches,
autograd, concurrent streams/threads, and up to 2,427,000 matrices with less
than 1 GiB of incremental allocated workspace plus outputs. Run it in the
required environment with a visible CUDA GPU; CUDA tests skip otherwise.

See the [production validation record](CUSOLVER_VALIDATION.md) for measured
results and the limits of the hardware/end-to-end checks.
