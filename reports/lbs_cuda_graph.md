# CUDA graphs for public Boba LBS

`/home/boba/Research/Boba-Public` was a source snapshot at GitHub commit
`77a3aa309d325fe3ae022d47c91d9b3037cdeb00`, with no Git metadata. It now tracks
`origin/Boba_Batched`. All existing assets and three local edited files were
preserved. Copies of those original edits are stored locally in
`../.boba-public-local-before-git-20260929`.

The private robot-articulation commits do not apply directly: the public branch
has no scanned xArm Gaussian transform. This change applies their graph-replay
and buffer-reuse approach to public rope/sloth LBS instead.

## Release policy

LBS automatically uses CUDA graphs for batches up to 64 and ordinary CUDA kernel
launches for larger batches. There is no runtime selector. The existing physics
CUDA graph remains in use at every batch size.

The timing tables below record the initial experiment with LBS graphs at every
batch size, before the release cutoff was introduced. Graph results above 64
describe that experiment, not the current release behavior. The JSON and CSV
retain those original measurements and source hashes.

The automatic cutoff passes 39 repository tests, including batches 64 and 65.
Both rope and sloth have bit-exact geometry and cache state on matched physics
inputs at that boundary, with 774 pixel-identical composited instance images.
Two complete rendered rope episodes confirm graph replay at B64 and ordinary
CUDA kernels at B65. Logs are in `results/lbs_release_policy_validation`.

## Implementation

Two fixed-shape stages use cached CUDA graphs: preparing deformation gradients
and the rotation-reuse mask, then blending Gaussian positions and quaternions.
The variable-size selection, cuSOLVER eigendecomposition, reflection correction,
quaternion sign stabilization, and selective rotation-cache writes retain the
original code and threshold. No extra bones are recomputed to make capture work.

Runtime callers use borrowed output buffers on the same CUDA stream, avoiding
intermediate output clones. The public function keeps independently owned
outputs by default. Cache shape, allocation, precision and stream changes are
accounted for. First-use capture and library initialization are outside warmed
timings.

## LBS timing

Milliseconds per entire batch on RTX PRO 6000 Blackwell Max-Q, Torch 2.12.1 /
CUDA 13.2, public desktop implementation with its normal TF32 setting. The
unmodified baseline LBS function is loaded from GitHub revision `77a3aa3`.
These are first 50 rope / 64 sloth states from actual physics replays, with
8 warmup states excluded and three timed repeats in alternating mode order.
Each entry is the median of the three per-repeat mean times. Physics recording
and image validation are outside these LBS timings.

| Batch | Rope: original → graph, ms | Time reduction | Sloth: original → graph, ms | Time reduction |
|---|---:|---:|---:|---:|
| 1 | 2.593 → 2.135 | 17.6% | 2.735 → 2.364 | 13.6% |
| 4 | 2.582 → 2.111 | 18.3% | 2.820 → 2.483 | 11.9% |
| 16 | 2.808 → 2.454 | 12.6% | 3.038 → 2.880 | 5.2% |
| 64 | 3.304 → 3.131 | 5.2% | 4.978 → 4.852 | 2.5% |
| 128 | 4.159 → 4.003 | 3.7% | 7.830 → 7.745 | 1.1% |
| 512 | 10.762 → 10.636 | 1.2% | 26.984 → 27.134 | -0.6% |

At B1–B4 the reduction is approximately 12–18%. At larger batches the gain
shrinks; sloth B512 is 0.6% slower, so no large-batch speedup is claimed there.
These are initial measurements for two assets, not the full 22-case paper suite.
Rope uses 432 mass nodes / 38,476 Gaussians; sloth uses 2,427 / 102,245.

## Complete rendered runtime

The existing benchmark runs spring-mass simulation, LBS, rendering and
compositing: `batch_optimized`, shared templates, 640×480, one camera per
instance, the original force policy (atomic accumulation at B1, gather at
B128), and cyclic controller trajectories when needed.
Two complete episodes per mode were run in alternating order: 50 rope frames
and 192 sloth frames, excluding the original benchmark's first two frames.
Numbers below are the mean frame time over the two runs.

| Asset | Batch | Original → graph frame time, ms | Aggregate throughput increase |
|---|---:|---:|---:|
| Rope | 1 | 8.653 → 8.187 | 5.7% |
| Rope | 128 | 17.163 → 16.978 | 1.1% |
| Sloth | 1 | 6.881 → 6.431 | 7.0% |
| Sloth | 128 | 41.432 → 41.391 | 0.1% |

Full runtime gains approximately 6–7% throughput at B1. At B128 the difference
is approximately 0–1%, within small run-to-run variation for the sloth case.
The preserved local runtime edits are present in both comparison modes.

Measured B128 peak Torch allocations increased slightly: rope 2.531 → 2.565 GiB
and sloth 4.072 → 4.121 GiB. Reserved-memory samples and end-of-runtime allocations
are included in the numeric report. Maximum feasible batch capacity was not
retested, so these timings do not establish unchanged maximum capacity.

## Correctness

- All 12 asset/batch configurations have bit-exact Gaussian positions,
  quaternions and all five mutable LBS cache tensors against the original code.
- 774 composited instance images are pixel-identical, covering three frames
  of each asset at B1 and B128.
- All full-runtime final states are finite. Physics positions/velocities and
  final Gaussian geometry are bit-identical across the four B128 runs per asset.
  At B1 even the two original-code runs have different final-state hashes:
  that existing solver path uses atomic force accumulation. The exact LBS and
  image comparisons above use identical recorded physics inputs, so they do
  not depend on independent physics replays matching. No claim is made about
  the magnitude of the B1 replay differences.
- At the measured commit, the repository test suite passed 38 tests, including output-lifetime,
  changing-node, cache-invalidation, precision and stream-ordering checks.
  GPU tests also skip cleanly when no CUDA device is visible.
- GPU process monitoring found no competing compute process. Per-run manifests
  record source hashes and commands for the historical measured implementation.

## Reproduce

The current scripts validate the automatic release policy. Activate the
configured Boba environment, or use this machine's `run_boba.sh`:

```bash
python -m unittest discover -s tests -v
python benchmarks/profile_lbs_cuda_graph.py --case single_lift_rope --batch 1 --render-check --output-dir /tmp/rope-lbs-graph
python benchmarks/profile_lbs_full_runtime.py --case single_lift_rope --batch 64 --output-dir /tmp/rope-full-b64
python benchmarks/profile_lbs_full_runtime.py --case single_lift_rope --batch 65 --output-dir /tmp/rope-full-b65
```

Use fresh output directories. The full-runtime benchmark uses the existing
OpenGL/X11 setup. Raw logs, per-frame timings, commands and GPU monitoring are
stored locally in `results/lbs_cuda_graph_validation`. Those artifacts record
the original unrestricted graph comparison; the current scripts validate the
automatic release policy.

[Numeric report](lbs_cuda_graph.json) · [LBS timing CSV](lbs_cuda_graph.csv)
