# cuSOLVER production validation — 2026-09-10

All runs used Python 3.10, PyTorch 2.12.1+cu132 and the `phystwin-cu132`
environment. The C++ binding and runtime policy are identical in the public
Boba_Batched, phone demo and Quest demo branches.

| Check | Result |
| --- | --- |
| Phone regression suite, RTX 5090 | 77 passed |
| Public regression suite, RTX 5090 | 28 passed |
| Quest regression suite, RTX 5090 | 134 passed; 5 Garden rendering tests skipped because optional Garden model/cache assets were unavailable |
| Native solver and environment tests, RTX 4090 | 14 passed |
| Phone preflight, RTX 5090 | Passed, including dedicated solver warmup and CUDA/OpenGL interop |
| Phone demo, RTX 5090 | Batch 100, 640x480, completed a bounded 410-frame launch |
| Simulation + LBS, RTX 4090 | Batches 1 and 100 each completed 410 frames and two replay resets; sampled Gaussian outputs finite |
| Large native solver batches, both GPUs | 4,097, 242,700 and 2,427,000 FP32 matrices passed residual checks with incremental workspace/output allocation below 1 GiB |

The public full-runtime command was:

```bash
NUM_RUNS=1 bash benchmarks/run_batched_full_runtime_batch_scaling.sh \
  --batch_sizes 1 100 --batch_image_resolution 640x480 \
  --batched_render_variant batch_optimized double_stretch_sloth
```

Both sizes passed on RTX 5090: 211.18 FPS at batch 1 and 29.34 FPS at batch
100 (2,934 instances/s). These are single-run validation measurements, not a
new throughput claim or a paired comparison with the previous implementation.

For a downstream numerical comparison on RTX 5090, a 100-instance Sloth
simulation ran 410 frames. At frames 0, 95, 191, 192, 383, 384 and 409, LBS
was evaluated twice from identical simulation inputs and independently copied
rotation caches: once with the production binding and once with PyTorch's
previous cuSOLVER `torch.linalg.eigh` call. All Gaussian positions and
quaternions matched bitwise. The selected instance's 640x480 rendered images
also matched bitwise at every sampled frame, including both reset boundaries.
This does not promise bitwise equivalence to generic eigensolver algorithms
on other architectures, particularly for nearly singular matrices.

The RTX 4090 replay check was headless. Earlier windowed controls on that
machine failed CUDA/OpenGL buffer registration even without the new solver;
4090 windowed rendering is not certified by this validation. Quest CUDA
rendering/runtime tests passed, but a live headset session was not exercised.
The legacy Jetson checkout and its environment were not updated.

The existing environment already contained the native build dependencies.
The public installer was checked for shell syntax and environment rejection;
its package-rebuild operations were not rerun over the working environment.
