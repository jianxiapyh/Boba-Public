# Boba: Batched Simulation for Physics-Based Gaussian Digital Twins (ECCV 2026)

**Accepted to ECCV 2026.**

[Project Page](https://jianxiapyh.github.io/Boba-project-page/) | [Paper](https://rsim.cs.illinois.edu/Pubs/Boba_ECCV.pdf)

> This repository contains the source code for Boba, and this branch currently includes `Boba-Local` and `Boba-Batched`.
> For `Boba-Distributed`, switch to the future `Boba-Distributed` branch and follow the README there.

Boba is based on PhysTwin. In this branch, the main public paths are `Boba-Local` and `Boba-Batched`, corresponding to the system designs shown below.

This branch contains the spring-mass simulation and skinning pipeline described in the paper, together with the rendering optimizations used for the visualization path. `Boba-Distributed` and its transmission optimizations will be documented in the future `Boba-Distributed` branch.

## System Designs

![Boba system designs](./assets/system_designs.png)

- `Boba-Local`: single-instance spring-mass simulation, skinning, and visualization.
- `Boba-Batched`: batched spring-mass simulation and skinning, with headless and rendered benchmark paths.
- `Boba-Distributed`: planned as a separate public branch.
- In this branch, `batch_size=1` / `instance=1` follows the `Boba-Local` equivalent path.

## Setup

### Prerequisites

- Linux on an NVIDIA GPU
- CUDA-compatible driver and toolkit
- Linux build tools for compiled extensions
- Desktop OpenGL / X11 for interactive windowed runs
- A valid `DISPLAY` when launching rendering paths

On a minimal Ubuntu/NVIDIA machine, install system packages such as `build-essential`, `libglfw3`, `libglfw3-dev`, and the usual desktop OpenGL / X11 runtime libraries before continuing.

For production linear algebra, Boba always selects cuSOLVER. There is no
runtime backend-selection environment variable and no MAGMA fallback.

### Standard CUDA 12 environment

The existing install script is written around a CUDA 12.1 desktop setup. We
have also tested Boba successfully on CUDA 11.8 and CUDA 12.2. Use this path
for supported GPUs with compute capability below 12 unless you encounter a
CUDA solver compatibility problem.

```bash
export PATH={YOUR_DIR}/cuda/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH={YOUR_DIR}/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH

conda create -y -n phystwin python=3.10

bash ./env_install/env_install.sh
```

`env_install/env_install.sh` installs the Python packages required by the public Boba scripts in this branch:

- core runtime and evaluation packages
- PyTorch 2.4.0 with CUDA 12.1
- Warp, Open3D, OpenGL / GLFW / PyCUDA, vendored `gsplat`, and kornia
- `gsplat` pinned to upstream `v1.5.3` and installed into the active supported
  `phystwin` or `phystwin-cu130` conda environment
- `gsplat` installed in editable mode from `gaussian_splatting/submodules/gsplat`, with CUDA compiled on first use
- compiled KNN extension

The standard examples below use `phystwin`. Run Boba rendering commands from
the supported environment in which the extensions were built, or via:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 python ...
```

Boba validates that runtime imports resolve to the vendored `gsplat` copy. A system-level or user-level `gsplat` install is not a supported configuration.

### CUDA 13 compatibility environment

Use `phystwin-cu130` for GPUs with compute capability 12.x or newer, or when a
supported older GPU encounters a solver failure with its CUDA 12/PyTorch
stack:

```bash
conda env create -f env_install/phystwin-cu130.yml
conda activate phystwin-cu130
./env_install/build_cuda13_extensions.sh
conda deactivate
conda activate phystwin-cu130
```

The CUDA 13 builder compiles only `simple-knn`, `fused-ssim`, PyCUDA with
OpenGL support, and Boba's vendored `gsplat`. It detects all visible GPU
compute capabilities, removes duplicates, and builds the matching native
cubins. For example, a machine exposing compute capabilities 8.9 and 12.0
builds both `sm_89` and `sm_120`.

The builder also installs environment-local Conda activation hooks for the
CUDA 13, PyTorch, and C++ runtime libraries. Reactivating the environment
makes those paths available without a machine-specific `LD_LIBRARY_PATH`;
the hooks derive them from `CONDA_PREFIX` and restore the previous library
path on deactivation. Commands launched with `conda run -n phystwin-cu130`
receive the same activation hook automatically.

For a headless build, or to prepare one environment for GPUs that are not
visible during installation, provide a numeric architecture list explicitly:

```bash
TORCH_CUDA_ARCH_LIST="8.9;12.0+PTX" \
  ./env_install/build_cuda13_extensions.sh
```

The builder accepts numeric capabilities separated by spaces, semicolons, or
commas, with an optional `+PTX` suffix. It verifies that every Boba extension
links CUDA 13 and contains cubins for every requested architecture.

At runtime Boba reads the active device capability through PyTorch. Capability
12.x and newer requires a PyTorch build against CUDA 13 or newer under Boba's
compatibility policy; this is based on capability rather than a GPU product
name. `nvidia-smi` reports the maximum CUDA version supported by the installed
driver, while `torch.version.cuda` reports the CUDA version used to build the
active PyTorch package.

## Required Assets

Replace the placeholders below with the public download links before release, then place the downloaded folders at the repository root.

- [data](https://example.com/boba/data)
- [gaussian_output](https://example.com/boba/gaussian_output)
- [experiments](https://example.com/boba/experiments)
- [experiments_optimization](https://example.com/boba/experiments_optimization)

Expected layout:

```text
data/
experiments/
experiments_optimization/
gaussian_output/
```

## Run Boba-Local

Performance mode:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 python interactive_playground.py --mode perf --case_name double_lift_cloth_3
```

Quality mode:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 python interactive_playground.py --mode quality --case_name double_lift_cloth_3
```

`quality` mode is the single-instance evaluation path with calibrated multi-view rendering.
It writes `inference.pkl` and rendered frames under `results/quality/<case>/`.
For quality comparison against the original PhysTwin paper numbers, use one camera view.
The quality script defaults to PhysTwin-style render aggregation:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 bash benchmarks/Boba_Local_single_inst_quality.sh --num_views 1
```

Render evaluation supports two `OVERALL` aggregation modes:
- `phystwin`: PhysTwin-compatible mode. `OVERALL` is averaged over every evaluated frame/view sample directly, so longer sequences contribute more samples.
- `scene_mean`: Equal-scene summary mode. This is often the fairer Boba summary because each scene/case sequence gets equal weight in the final `OVERALL` row.

Here, `scene` means one data sequence/case such as `double_lift_cloth_1`, and `view` means one calibrated camera view. `--num_views 2` and `--num_views 3` are Boba multi-view extensions; use `--num_views 1` for one-to-one comparison with original PhysTwin paper numbers.

## Run Boba-Batched

Headless spring-mass + LBS batch scaling:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 bash benchmarks/run_sim_lbs_batch_scaling.sh --batch_sizes 1 2 4 8
```

Headless spring-mass + LBS best-throughput search for one case:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 bash benchmarks/run_sim_lbs_best_throughput.sh single_lift_rope
```

This autotune benchmark searches batch sizes automatically instead of requiring a fixed `--batch_sizes` list.
It reports the best measured instance count, batch FPS, and throughput under `results/batch_autotune/`.

```bash
NUM_RUNS=1 MAX_BATCH_SIZE=256 REFINE_SAMPLES=9 REFINE_ROUNDS=2 FINAL_DENSE_WINDOW=8 \
  conda run -n phystwin env PYTHONNOUSERSITE=1 bash benchmarks/run_sim_lbs_best_throughput.sh single_lift_rope
```

Batched full runtime for one case:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 python benchmarks/run_batched_full_runtime_case.py \
  --case_name double_lift_cloth_3 \
  --batch_size 4 \
  --batched_render_variant batch_optimized

conda run -n phystwin env PYTHONNOUSERSITE=1 python benchmarks/run_batched_full_runtime_case.py \
  --case_name double_lift_cloth_3 \
  --batch_size 4 \
  --render_mode instance \
  --instance_id 2 \
  --save_video
```

Batched full runtime across cases:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 bash benchmarks/run_batched_full_runtime.sh --batch_size 4 --batched_render_variant batch_optimized
conda run -n phystwin env PYTHONNOUSERSITE=1 bash benchmarks/run_batched_full_runtime.sh --batch_size 4 --render_mode instance --instance_id 2 --save_video
```

## Boba-Distributed

`Boba-Distributed` will be documented in the future `Boba-Distributed` branch. Use the README in that branch for the setup and execution commands for the distributed design.

## Benchmark Scripts

For the full benchmark entrypoint reference, including every script option,
environment override, and output layout, see
[`benchmarks/README.md`](benchmarks/README.md).

Full-runtime performance benchmark:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 bash benchmarks/Boba_Local_single_inst_perf.sh
```

Full-runtime quality benchmark:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 bash benchmarks/Boba_Local_single_inst_quality.sh
```

PhysTwin-compatible render reporting for paper-number comparison:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 bash benchmarks/Boba_Local_single_inst_quality.sh --num_views 1
```

Headless sim+LBS batch-scaling benchmark:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 bash benchmarks/run_sim_lbs_batch_scaling.sh --batch_sizes 1 2 4 8
```

Headless sim+LBS best-throughput autotune benchmark:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 bash benchmarks/run_sim_lbs_best_throughput.sh single_lift_rope
```

Batched full-runtime benchmark:

```bash
conda run -n phystwin env PYTHONNOUSERSITE=1 bash benchmarks/run_batched_full_runtime.sh --batch_size 4 --batched_render_variant batch_optimized
```

Batched full-runtime scaling benchmark:

```bash
DISPLAY=:1 conda run -n phystwin env PYTHONNOUSERSITE=1 bash benchmarks/run_batched_full_runtime_batch_scaling.sh --batch_sizes 1 2 4 8 16 32 64
```

Defaults when unspecified:
- `render_mode=batch_images`
- `num_views=1`
- `overall_mode=phystwin`
- `save_video=false`
- `NUM_RUNS=3`
- best-throughput autotune: `NUM_RUNS=1`, `MAX_BATCH_SIZE=256`, `REFINE_SAMPLES=9`, `REFINE_ROUNDS=2`, `FINAL_DENSE_WINDOW=8`

This path still requires an X11/OpenGL display because the full-runtime renderer creates a GLFW window.

## Outputs

- `results/perf`: full-runtime performance summaries and logs
- `results/quality`: rendered outputs, evaluation artifacts, and metrics
- `results/batch_scaling`: headless sim+LBS batch-scaling outputs
- `results/batch_autotune`: best-throughput search outputs, including `best_throughput_table.csv` and `candidate_table.csv`
- `results/batched_render`: batched full-runtime render and benchmark outputs

## Citation

If you find Boba useful, please cite:

```bibtex
@inproceedings{pang2026boba,
  title     = {Boba: Batched Simulation for Physics-Based Gaussian Digital Twins},
  author    = {Pang, Yihan and Jiang, Hanxiao and Kondguli, Sushant and Adve, Sarita and Wang, Shenlong},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
