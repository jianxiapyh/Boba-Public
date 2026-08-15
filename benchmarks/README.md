# Benchmark Entrypoints

This directory contains the benchmark runners, one-case helpers, offline pruning
helpers, and post-processing scripts used by Boba. Run commands from the repo
root unless a script says otherwise.

The full-runtime search runners use config-named output folders by default. The
default full-runtime render path is:

- Simulation: `gather` (default)
- Rendering: `batch_images`, `640x480`, `shared_template`, `batch_prune` (default)
- Template-state atomic simulation: pass `--sim_force_mode template_state_batched_atomic`

Default config folder names:

```text
sim_gather_render_batch_images_640x480_shared_template_batch_prune
sim_template_state_batched_atomic_render_batch_images_640x480_shared_template_batch_prune
```

The old dedicated template-state wrappers were removed. Template-state selection
is now sim-only: it does not silently change retry count, success threshold, max
batch size, or output root. If you want the old retry/cap behavior, pass it
explicitly, for example:

```bash
NUM_RUNS=3 MIN_SUCCESSES=2 MAX_BATCH_SIZE=5096 \
bash benchmarks/run_batched_full_runtime_best_throughput.sh \
  --sim_force_mode template_state_batched_atomic
```

## Full-Runtime Batch Search

### `benchmarks/run_batched_full_runtime_best_throughput.sh`

Purpose: autotunes each case for the highest full-runtime throughput across
batch sizes.

Canonical/default command:

```bash
bash benchmarks/run_batched_full_runtime_best_throughput.sh
```

Template-state command:

```bash
bash benchmarks/run_batched_full_runtime_best_throughput.sh \
  --sim_force_mode template_state_batched_atomic
```

Command-line options:

- `--batched_render_variant batch_original|batch_optimized|batch_prune`:
  selects the batch-images render preset (default: `batch_prune`). Deprecated
  aliases are accepted with warnings: `baseline`, `optimized`,
  `optimized_pruned`.
- `--sim_force_mode gather|template_state_batched_atomic`: selects the
  simulation force path (default: `gather`).
- `case_name ...`: optional case names; when omitted, cases are read from
  `CASES_FILE` (default: `data_config.csv`).
- `--help`, `-h`: print usage.

Environment overrides:

- `NUM_RUNS`: runs per candidate (default: `1`).
- `MIN_SUCCESSES`: minimum successful runs for candidate eligibility
  (default: `1`).
- `CASES_FILE`: case CSV used when no case names are passed
  (default: `data_config.csv`).
- `BASE_PATH`: case data root (default: `./data/different_types`).
- `GAUSSIAN_PATH`: baseline Gaussian root (default: `./gaussian_output`).
- `PRUNED_GAUSSIAN_PATH`: pruned Gaussian root
  (default: `./gaussian_output_pruned_policy_30_55`).
- `BG_IMG_PATH`: background image path (default: `./data/bg.png`).
- `MIN_BATCH_SIZE`: lower search bound (default: `1`).
- `MAX_BATCH_SIZE`: upper search bound (default: `2048`).
- `REFINE_SAMPLES`: sampled candidates per refine round (default: `9`).
- `REFINE_ROUNDS`: zoom refinement rounds (default: `2`).
- `FINAL_DENSE_WINDOW`: dense search radius around the best sample
  (default: `8`).
- `RESULTS_ROOT`: output root. If set, it is used exactly as provided
  (default: `results/batched_full_runtime_autotune/<config>`).
- `RENDER_MODE`: render mode (default: `batch_images`; this autotune runner
  currently supports only `batch_images`).
- `GAUSSIAN_RENDER_MODE`: Gaussian render mode (default: `shared_template`).
- `BATCH_IMAGE_RESOLUTION`: batch-image resolution (default: `640x480`).
- `BATCHED_RENDER_VARIANT`: render variant (default: `batch_prune`).
- `SIM_FORCE_MODE`: simulation force path (default: `gather`).
- `NUM_VIEWS`: camera views passed to the case runner (default: `1`).

Output location: config-aware by default.

```text
results/batched_full_runtime_autotune/<config>/
  best_throughput_table.csv
  candidate_table.csv
  attempted_candidates.csv
  logs/
  run_01/
```

### `benchmarks/run_batched_full_runtime_find_boundary.sh`

Purpose: finds the maximum full-runtime batch size each case can run before
failure.

Canonical/default command:

```bash
bash benchmarks/run_batched_full_runtime_find_boundary.sh
```

Template-state command:

```bash
bash benchmarks/run_batched_full_runtime_find_boundary.sh \
  --sim_force_mode template_state_batched_atomic
```

Command-line options:

- `--render_mode instance|batch_images`: render one instance or a batch image
  grid (default: `batch_images`).
- `--gaussian_render_mode shared_template|duplicated`: Gaussian render storage
  mode (default: `shared_template`).
- `--batched_render_variant batch_original|batch_optimized|batch_prune`:
  batch-images render preset (default: `batch_prune`). Deprecated aliases:
  `baseline`, `optimized`, `optimized_pruned`.
- `--sim_force_mode gather|template_state_batched_atomic`: simulation force
  path (default: `gather`).
- `--instance_id I`: instance index, required for `--render_mode instance`
  (default: unset).
- `--num_views N`: number of camera views, valid values `1`, `2`, `3`
  (default: `1`).
- `--save_video`: save PNG folders and MP4 videos (default: off).
- `--save_batch_images`: save per-instance composited images in
  `batch_images` mode (default: off).
- `--save_batch_grid`: save tiled batch preview images in `batch_images` mode
  (default: off).
- `--batch_image_resolution native|640x480`: batch-image resolution
  (default: `640x480`).
- `--batch_grid_cols N`: columns in tiled previews
  (default: `ceil(sqrt(batch_size))`).
- `case_name ...`: optional case names; when omitted, cases are read from
  `CASES_FILE` (default: `data_config.csv`).
- `--help`, `-h`: print usage.

Environment overrides:

- `CASES_FILE`: case CSV (default: `data_config.csv`).
- `BASE_PATH`: case data root (default: `./data/different_types`).
- `GAUSSIAN_PATH`: baseline Gaussian root (default: `./gaussian_output`).
- `PRUNED_GAUSSIAN_PATH`: pruned Gaussian root
  (default: `./gaussian_output_pruned_policy_30_55`).
- `BG_IMG_PATH`: background image path (default: `./data/bg.png`).
- `BASE_BATCH_SIZE`: first boundary candidate (default: `64`).
- `GROWTH_NUM`: growth numerator (default: `3`).
- `GROWTH_DEN`: growth denominator (default: `2`).
- `MAX_BATCH_SIZE`: upper growth bound (default: `16384`).
- `RETRIES`: attempts per batch (default: `NUM_RUNS` if set, otherwise `1`).
- `NUM_RUNS`: used as `RETRIES` only when `RETRIES` is unset (default: unset).
- `MIN_SUCCESSES`: successes required for a batch to count as supported
  (default: `RETRIES`).
- `TIMEOUT_SEC`: timeout per attempt; `0` disables timeout (default: `0`).
- `SIM_FORCE_MODE`: simulation force path (default: `gather`).
- `BATCHED_RENDER_VARIANT`: render variant (default: `batch_prune`).
- `BATCH_IMAGE_RESOLUTION`: batch-image resolution (default: `640x480`).
- `OUT_CSV`: output CSV. If set, used exactly as provided
  (default: `results/batched_full_runtime_boundary/<config>/boundary.csv`).
- `LOG_DIR`: log directory. If set, used exactly as provided
  (default: `results/batched_full_runtime_boundary/<config>/logs`).
- `RUN_ROOT`: candidate output root. If set, used exactly as provided
  (default: `results/batched_full_runtime_boundary/<config>/runs`).

Output location: config-aware by default.

```text
results/batched_full_runtime_boundary/<config>/
  boundary.csv
  logs/
  runs/
```

## Full-Runtime Batch Measurement

### `benchmarks/run_batched_full_runtime.sh`

Purpose: measures batched full-runtime performance for fixed batch size(s)
across cases.

Canonical/default command:

```bash
bash benchmarks/run_batched_full_runtime.sh --batch_size 4
```

Live batch-grid example (optionally add `--save_batch_grid` to save the same
grid as preview frames):

```bash
bash benchmarks/run_batched_full_runtime.sh --batch_size 4 --render_mode batch_images --display_batch_grid
```

Command-line options:

- `--batch_size N`: required positive batch size.
- `--render_mode instance|batch_images`: render one instance or a batch image
  grid (default: `batch_images`).
- `--gaussian_render_mode shared_template|duplicated`: Gaussian render mode
  (default: `shared_template`).
- `--batched_render_variant batch_original|batch_optimized|batch_prune`:
  batch-images render preset (default: unset). Deprecated aliases:
  `baseline`, `optimized`, `optimized_pruned`.
- `--instance_id I`: instance index, required for `--render_mode instance`
  (default: unset).
- `--num_views N`: number of camera views, valid values `1`, `2`, `3`
  (default: `1`).
- `--save_video`: save PNG folders and MP4 videos (default: off).
- `--save_batch_images`: save per-instance composited images in
  `batch_images` mode (default: off).
- `--save_batch_grid`: save tiled batch previews in `batch_images` mode
  (default: off).
- `--display_batch_grid`: show the tiled batch preview in a screen-sized live
  window (default: off).
- `--batch_image_resolution native|640x480`: batch-image resolution
  (default: `native`).
- `--batch_grid_cols N`: preview-grid columns
  (default: `ceil(sqrt(batch_size))`).
- `--profile_render_components`: write render component timing JSON/CSV
  (default: off).
- `case_name ...`: optional case names; otherwise read from `CASES_FILE`
  (default: `data_config.csv`).
- `--help`, `-h`: print usage.

Environment overrides:

- `NUM_RUNS`: runs per case (default: `3`).
- `CASES_FILE`: case CSV (default: `data_config.csv`).
- `BASE_PATH`: case data root (default: `./data/different_types`).
- `GAUSSIAN_PATH`: baseline Gaussian root (default: `./gaussian_output`).
- `PRUNED_GAUSSIAN_PATH`: pruned Gaussian root
  (default: `./gaussian_output_pruned_policy_30_55`).
- `BG_IMG_PATH`: background image path (default: `./data/bg.png`).

Output location: not config-rooted; case outputs go under
`results/batched_render/<run>/<case>/batch_<N>/<render-mode-dir>/` and logs under
`results/batched_render/logs/`.

### `benchmarks/run_batched_full_runtime_batch_scaling.sh`

Purpose: measures full-runtime throughput over a fixed list of batch sizes.

Canonical/default command:

```bash
bash benchmarks/run_batched_full_runtime_batch_scaling.sh --batch_sizes 1 2 4 8
```

Command-line options:

- `--batch_sizes N...`: required positive batch sizes.
- `--render_mode instance|batch_images`: render mode (default: `batch_images`).
- `--gaussian_render_mode shared_template|duplicated`: Gaussian render mode
  (default: `shared_template`).
- `--batched_render_variant batch_original|batch_optimized|batch_prune`:
  batch-images render preset (default: unset). Deprecated aliases:
  `baseline`, `optimized`, `optimized_pruned`.
- `--instance_id I`: required for `--render_mode instance` (default: unset).
- `--num_views N`: camera views, valid values `1`, `2`, `3` (default: `1`).
- `--save_video`: save videos (default: off).
- `--save_batch_images`: save per-instance batch images (default: off).
- `--save_batch_grid`: save tiled batch previews (default: off).
- `--display_batch_grid`: show tiled batch previews in a screen-sized live
  window (default: off).
- `--batch_image_resolution native|640x480`: resolution (default: `native`).
- `--batch_grid_cols N`: grid columns (default: `ceil(sqrt(batch_size))`).
- `case_name ...`: optional case names; otherwise read from `CASES_FILE`
  (default: `data_config.csv`).
- `--help`, `-h`: print usage.

Environment overrides:

- `NUM_RUNS`: runs per case/batch (default: `3`).
- `CASES_FILE`: case CSV (default: `data_config.csv`).
- `BASE_PATH`: case data root (default: `./data/different_types`).
- `GAUSSIAN_PATH`: baseline Gaussian root (default: `./gaussian_output`).
- `PRUNED_GAUSSIAN_PATH`: pruned Gaussian root
  (default: `./gaussian_output_pruned_policy_30_55`).
- `BG_IMG_PATH`: background image path (default: `./data/bg.png`).

Output location: not config-rooted; writes under `results/batched_render/` and
aggregates with `aggregate_batched_full_runtime_batch_scaling.py`.

### `benchmarks/run_batched_full_runtime_case.py`

Purpose: runs one full-runtime batched case and writes one output directory.

Canonical/default command:

```bash
python benchmarks/run_batched_full_runtime_case.py \
  --case_name single_lift_rope --batch_size 4
```

Command-line options:

- `--base_path PATH`: case data root (default: `./data/different_types`).
- `--gaussian_path PATH`: Gaussian root (default: `./gaussian_output`).
- `--pruned_gaussian_path PATH`: pruned Gaussian root
  (default: `./gaussian_output_pruned_policy_30_55`).
- `--pruning_source_gaussian_path PATH`: source Gaussian root for pruning
  (default: `./gaussian_output`).
- `--case_keep_ratio_csv PATH`: case keep-ratio CSV
  (default: `benchmarks/pruning_ratio_policy_30_55.csv`).
- `--default_prune_keep_ratio R`: fallback pruning ratio (default: `0.3`).
- `--prune_mode opacity|opacity_area|opacity_volume`: pruning score
  (default: `opacity_area`).
- `--force_prune`: recreate pruned assets (default: off).
- `--bg_img_path PATH`: background image path (default: `./data/bg.png`).
- `--case_name NAME`: required case name.
- `--batch_size N`: required positive batch size.
- `--batched_render_variant batch_original|batch_optimized|batch_prune`:
  batch-images render preset (default: unset). Deprecated aliases:
  `baseline`, `optimized`, `optimized_pruned`.
- `--render_mode instance|batch_images`: render mode (default: `batch_images`).
- `--gaussian_render_mode shared_template|duplicated`: Gaussian render mode
  (default: `shared_template`).
- `--sim_force_mode gather|template_state_batched_atomic`: simulation force
  path (default: `gather`).
- `--instance_id I`: required for instance mode (default: unset).
- `--save_video`: save videos (default: off).
- `--save_batch_images`: save per-instance batch images (default: off).
- `--save_batch_grid`: save tiled batch previews (default: off).
- `--display_batch_grid`: show tiled batch previews in a screen-sized live
  window (default: off).
- `--batch_image_resolution native|640x480`: resolution (default: `native`).
- `--batch_grid_cols N`: grid columns (default: `ceil(sqrt(batch_size))`).
- `--profile_render_components`: write render timing details (default: off).
- `--num_views N`: camera views (default: `1`).
- `--output_dir PATH`: output directory. If unset, writes under
  `results/batched_render/<case>/batch_<N>/<render-mode-dir>/`.

Environment overrides: none read directly by this helper.

Output location: not config-rooted unless called through a config-rooted runner;
`--output_dir` controls the exact path.

## Single-Instance Full Runtime

### `benchmarks/Boba_Local_single_inst_perf.sh`

Purpose: measures Boba Local single-instance full-runtime performance.

Canonical/default command:

```bash
bash benchmarks/Boba_Local_single_inst_perf.sh
```

Command-line options:

- `case_name ...`: optional case names; otherwise read from `CASES_FILE`
  (default: `data_config.csv`).
- `--help`, `-h`: print usage.

Environment overrides:

- `NUM_RUNS`: runs per case (default: `3`).
- `CASES_FILE`: case CSV (default: `data_config.csv`).

Output location: `results/perf/`, including `logs/` and
`performance_table.csv`. Not config-rooted.

### `benchmarks/Boba_Local_single_inst_quality.sh`

Purpose: measures Boba Local single-instance quality metrics.

Canonical/default command:

```bash
bash benchmarks/Boba_Local_single_inst_quality.sh
```

Command-line options:

- `--gaussian_variant baseline|pruned`: Gaussian asset variant
  (default: `baseline`).
- `--num_views N`: camera views, valid values `1`, `2`, `3` (default: `1`).
- `--overall_mode scene_mean|phystwin`: render metric aggregation
  (default: `phystwin`).
- `--gaussian_path PATH`: custom Gaussian root. Defaults by variant:
  baseline `./gaussian_output`, pruned
  `./gaussian_output_pruned_policy_30_55`.
- `--result_root PATH`: custom result root. Defaults by variant:
  baseline `results/quality`, pruned `results/quality_pruned_policy_30_55`.
- `case_name ...`: optional case names; otherwise read from `CASES_FILE`
  (default: `data_config.csv`).
- `--help`, `-h`: print usage.

Environment overrides:

- `CASES_FILE`: case CSV (default: `data_config.csv`).

Output location: `results/quality/` by default, or the variant/custom
`--result_root`. Not config-rooted.

### `benchmarks/run_full_runtime_quality_pruning_ablation.sh`

Purpose: runs baseline quality or, with pruning enabled, compares baseline and
pruned quality.

Canonical/default command:

```bash
bash benchmarks/run_full_runtime_quality_pruning_ablation.sh
```

Command-line options:

- `--gaussian_path PATH`: baseline Gaussian root
  (default: `./gaussian_output`).
- `--result_root PATH`: result root; non-pruning mode delegates to
  `Boba_Local_single_inst_quality.sh`, pruning mode defaults to
  `results/quality_pruning_ablation`.
- `--num_views N`: views for evaluation (default: `1`).
- `--overall_mode scene_mean|phystwin`: render aggregation
  (default: `phystwin`).
- `--enable_pruning`: generate/reuse pruned PLY assets and compare quality
  (default: off).
- `--keep_ratio R`: fallback Gaussian keep ratio (default: `0.3`).
- `--keep_count N`: fixed Gaussian keep count instead of ratio (default: unset).
- `--prune_mode opacity|opacity_area|opacity_volume`: pruning score
  (default: `opacity_area`).
- `--pruned_gaussian_path PATH`: pruned Gaussian root
  (default: derived from policy/ratio/count).
- `--case_keep_ratio_csv PATH`: case-specific keep ratios
  (default: `benchmarks/pruning_ratio_policy_30_55.csv`).
- `--no_case_keep_ratio_policy`: disable the default mixed-ratio policy
  (default: off).
- `--policy_name NAME`: policy label for output paths (default: `30_55`).
- `--force_prune`: recreate pruned PLYs (default: off).
- `case_name ...`: optional case names; otherwise read from `CASES_FILE`
  (default: `data_config.csv`).
- `--help`, `-h`: print usage.

Environment overrides:

- `CASES_FILE`: case CSV (default: `data_config.csv`).

Output location: `results/quality_pruning_ablation/` in pruning mode unless
`--result_root` is set. Not config-rooted.

## Additional Cases

### `benchmarks/run_additional_runtime_perf.sh`

Purpose: measures single-instance performance on the additional-case CSV.

Canonical/default command:

```bash
bash benchmarks/run_additional_runtime_perf.sh
```

Command-line options:

- `case_name ...`: optional case names; otherwise read from `CASES_FILE`
  (default: `benchmarks/additional_data_config.csv`).
- `--help`, `-h`: print usage.

Environment overrides:

- `NUM_RUNS`: runs per case (default: `3`).
- `CASES_FILE`: case CSV (default: `benchmarks/additional_data_config.csv`).

Output location: `results/additional_perf/`. Not config-rooted.

### `benchmarks/run_additional_runtime_quality.sh`

Purpose: measures quality on the additional-case CSV.

Canonical/default command:

```bash
bash benchmarks/run_additional_runtime_quality.sh
```

Command-line options:

- `--num_views N`: camera views, valid values `1`, `2`, `3` (default: `1`).
- `--overall_mode scene_mean|phystwin`: render aggregation
  (default: `phystwin`).
- `case_name ...`: optional case names; otherwise read from `CASES_FILE`
  (default: `benchmarks/additional_data_config.csv`).
- `--help`, `-h`: print usage.

Environment overrides:

- `CASES_FILE`: case CSV (default: `benchmarks/additional_data_config.csv`).

Output location: `results/additional_quality/`. Not config-rooted.

## Sim+LBS Benchmarks

### `benchmarks/run_sim_lbs_batch_scaling.sh`

Purpose: measures spring-mass plus LBS scaling over batch sizes, optionally with
NCU profiling and summary plots.

Canonical/default command:

```bash
bash benchmarks/run_sim_lbs_batch_scaling.sh --batch_sizes 1 2 4 8
```

Scaling-analysis command:

```bash
bash benchmarks/run_sim_lbs_batch_scaling.sh --scaling-analysis single_lift_rope
```

Command-line options:

- `--batch_sizes N...`, `--batch-sizes "N ..."`: batch sizes. Required in
  legacy mode; in scaling-analysis mode defaults to `1 32 64 128 256 512`.
- `--scaling-analysis`: enable timing, NCU, and summary workflow
  (default: off).
- `--timing-only`: run only timing pass; requires `--scaling-analysis`
  (default: off).
- `--ncu-only`: run only NCU pass; requires `--scaling-analysis`
  (default: off).
- `--summarize-only`: run only summary pass; requires `--scaling-analysis`
  (default: off).
- `--output-dir DIR`: output root. Default is `results/batch_scaling` in
  legacy mode and `results/sim_lbs_batch_scaling` in scaling-analysis mode.
- `--ncu-bin PATH`: Nsight Compute executable; defaults to
  `/home/yihanp2/cuda-12.1/bin/ncu` when executable, otherwise `ncu` on `PATH`.
- `case_name ...`: optional cases; legacy mode can run multiple, scaling-analysis
  mode supports one case. Defaults from `CASES_FILE`.
- `--help`, `-h`: print usage.

Environment overrides:

- `NUM_RUNS`: runs per batch (default: `3`).
- `FORCE_RERUN`: reuse existing timing/NCU metrics unless `1` (default: `0`).
- `CASES_FILE`: case CSV (default: `data_config.csv`).
- `BASE_PATH`: case data root (default: `./data/different_types`).
- `GAUSSIAN_PATH`: Gaussian root (default: `./gaussian_output`).
- `BG_IMG_PATH`: background image path (default: `./data/bg.png`).
- `NCU_PROFILE_FRAME_STRIDE`: frame stride for NCU profiling (default: unset).
- `NCU_PROFILE_MAX_FRAMES`: maximum profiled frames (default: `3`).
- `NCU_PROFILE_NVTX_NAME`: NVTX range name (default: `sim_lbs_profile_frame`).
- `NCU_TARGET_PROCESSES`: NCU target process setting
  (default: `application-only`).

Output location: `results/batch_scaling/` in legacy mode or
`results/sim_lbs_batch_scaling/` in scaling-analysis mode unless
`--output-dir` is set. Not config-rooted.

### `benchmarks/run_sim_lbs_batch_scaling_all_cases.sh`

Purpose: runs the Sim+LBS scaling-analysis workflow once per case and merges the
case CSVs.

Canonical/default command:

```bash
bash benchmarks/run_sim_lbs_batch_scaling_all_cases.sh
```

Command-line options:

- `--batch_sizes N...`, `--batch-sizes "N ..."`: batch sizes passed to each case
  (default: the per-case scaling-analysis default `1 32 64 128 256 512`).
- `--output-dir DIR`: all-case output root (default:
  `results/sim_lbs_batch_scaling_all_cases`).
- `--ncu-bin PATH`: Nsight Compute executable (default: per-case script default).
- `--timing-only`: timing pass only (default: off).
- `--ncu-only`: NCU pass only (default: off).
- `--summarize-only`: summary pass only (default: off).
- `case_name ...`: optional cases; otherwise read from `CASES_FILE`
  (default: `data_config.csv`).
- `--help`, `-h`: print usage.

Environment overrides:

- `NUM_RUNS`: passed to per-case runner (default: `3` there).
- `FORCE_RERUN`: passed to per-case runner (default: `0` there).
- `CASES_FILE`: case CSV (default: `data_config.csv`).
- `BASE_PATH`: case data root passed through (default: `./data/different_types`).
- `GAUSSIAN_PATH`: Gaussian root passed through (default: `./gaussian_output`).
- `BG_IMG_PATH`: background image path passed through (default: `./data/bg.png`).
- `RESULTS_ROOT`: all-case output root
  (default: `results/sim_lbs_batch_scaling_all_cases`).
- `NCU_PROFILE_FRAME_STRIDE`: passed through (default: unset).
- `NCU_PROFILE_MAX_FRAMES`: passed through (default: `3`).
- `NCU_PROFILE_NVTX_NAME`: passed through
  (default: `sim_lbs_profile_frame`).
- `NCU_TARGET_PROCESSES`: passed through (default: `application-only`).

Output location: `results/sim_lbs_batch_scaling_all_cases/` unless
`RESULTS_ROOT` or `--output-dir` is set. Not config-rooted.

### `benchmarks/run_sim_lbs_batch_scaling_case.py`

Purpose: runs one spring-mass plus LBS case for one batch size.

Canonical/default command:

```bash
python benchmarks/run_sim_lbs_batch_scaling_case.py \
  --case_name single_lift_rope --batch_size 4
```

Command-line options:

- `--base_path PATH`: case data root (default: `./data/different_types`).
- `--gaussian_path PATH`: Gaussian root (default: `./gaussian_output`).
- `--bg_img_path PATH`: background image path (default: `./data/bg.png`).
- `--case_name NAME`: required case name.
- `--batch_size N`: required positive batch size.
- `--output_dir PATH`: output directory
  (default: `results/batch_scaling/batch_<N>/<case_name>`).
- `--scaling-analysis`: write scaling metrics JSON (default: off).
- `--ncu-profile-loop`: wrap measured frames for NCU capture (default: off).
- `--ncu-profile-frame-stride N`: NCU frame stride
  (default: `NCU_PROFILE_FRAME_STRIDE`, otherwise unset).
- `--ncu-profile-max-frames N`: max NCU frames
  (default: `NCU_PROFILE_MAX_FRAMES`, otherwise `3`).
- `--ncu-profile-nvtx-name NAME`: NCU NVTX range
  (default: `sim_lbs_profile_frame`).

Environment overrides:

- `NCU_PROFILE_FRAME_STRIDE`: default for `--ncu-profile-frame-stride`
  (default: unset).
- `NCU_PROFILE_MAX_FRAMES`: default for `--ncu-profile-max-frames`
  (default: `3`).

Output location: controlled by `--output_dir` or the default
`results/batch_scaling/batch_<N>/<case_name>`. Not config-rooted.

### `benchmarks/run_sim_lbs_best_throughput.sh`

Purpose: autotunes spring-mass plus LBS batch size for best throughput.

Canonical/default command:

```bash
bash benchmarks/run_sim_lbs_best_throughput.sh
```

Command-line options:

- `case_name ...`: optional cases; otherwise read from `CASES_FILE`
  (default: `data_config.csv`).
- `--help`, `-h`: print usage.

Environment overrides:

- `NUM_RUNS`: runs per candidate (default: `1`).
- `CASES_FILE`: case CSV (default: `data_config.csv`).
- `BASE_PATH`: case data root (default: `./data/different_types`).
- `GAUSSIAN_PATH`: Gaussian root (default: `./gaussian_output`).
- `BG_IMG_PATH`: background image path (default: `./data/bg.png`).
- `MIN_BATCH_SIZE`: lower search bound (default: `1`).
- `MAX_BATCH_SIZE`: upper search bound (default: `256`).
- `REFINE_SAMPLES`: sampled candidates per refine round (default: `9`).
- `REFINE_ROUNDS`: zoom refinement rounds (default: `2`).
- `FINAL_DENSE_WINDOW`: dense search radius around best sample (default: `8`).
- `RESULTS_ROOT`: output root (default: `results/batch_autotune`).

Output location: `results/batch_autotune/` by default, including
`best_throughput_table.csv`, `candidate_table.csv`, `attempted_candidates.csv`,
`logs/`, and `run_01/`. Not config-rooted.

## Post-Processing

### `benchmarks/post-processing/aggregate_batched_full_runtime_batch_scaling.py`

Purpose: aggregates fixed-batch full-runtime scaling runs.

Canonical/default command:

```bash
python benchmarks/post-processing/aggregate_batched_full_runtime_batch_scaling.py
```

Options:

- `--results_root PATH`: input root (default: `results/batched_render`).
- `--cases_file PATH`: case CSV (default: `data_config.csv`).
- `--render_mode instance|batch_images`: render mode filter
  (default: `batch_images`).
- `--gaussian_render_mode shared_template|duplicated`: Gaussian render filter
  (default: `shared_template`).
- `--instance_id I`: instance filter for instance mode (default: unset).
- `--batch_image_resolution native|640x480`: resolution filter
  (default: `native`).
- `--batched_render_variant batch_original|batch_optimized|batch_prune`:
  render variant filter (default: unset). Deprecated aliases are accepted.
- `--output_table PATH`: per-case output CSV (default: inside `--results_root`).
- `--output_overall PATH`: overall output CSV (default: inside `--results_root`).
- `cases ...`: optional case names.

Environment overrides: none.

Output location: paths from `--output_table` and `--output_overall`; not
config-rooted unless the paths are set that way.

### `benchmarks/post-processing/aggregate_batched_full_runtime_best_throughput.py`

Purpose: aggregates full-runtime autotune candidates into best and candidate
tables.

Canonical/default command:

```bash
python benchmarks/post-processing/aggregate_batched_full_runtime_best_throughput.py
```

Options:

- `--results_root PATH`: input root
  (default: `results/batched_full_runtime_autotune`). For the new layout, pass
  `results/batched_full_runtime_autotune/<config>`.
- `--cases_file PATH`: case CSV (default: `data_config.csv`).
- `--render_mode instance|batch_images`: render mode filter
  (default: `batch_images`).
- `--gaussian_render_mode shared_template|duplicated`: Gaussian render filter
  (default: `shared_template`).
- `--instance_id I`: instance filter for instance mode (default: unset).
- `--batch_image_resolution native|640x480`: resolution filter
  (default: `640x480`).
- `--batched_render_variant batch_original|batch_optimized|batch_prune`:
  render variant filter (default: unset). Deprecated aliases are accepted.
- `--sim_force_mode gather|template_state_batched_atomic`: simulation filter
  (default: `gather`).
- `--output_best PATH`: best table path
  (default: `results/batched_full_runtime_autotune/best_throughput_table.csv`).
- `--output_candidates PATH`: candidate table path
  (default: `results/batched_full_runtime_autotune/candidate_table.csv`).
- `--attempted_manifest PATH`: candidate manifest (default: unset).
- `--num_runs N`: number of run directories to consider (default: all).
- `--min_successes N`: minimum successful runs for eligibility (default: `1`).
- `cases ...`: optional case names.

Environment overrides: none.

Output location: paths from `--output_best` and `--output_candidates`; the
runner sets these inside the config folder.

### `benchmarks/post-processing/aggregate_full_runtime_perf_runs.py`

Purpose: aggregates single-instance full-runtime performance runs.

Canonical/default command:

```bash
python benchmarks/post-processing/aggregate_full_runtime_perf_runs.py
```

Options:

- `--results_root PATH`: input root (default: `results/perf`).
- `--cases_file PATH`: case CSV (default: `data_config.csv`).
- `--output_file PATH`: output CSV
  (default: `results/perf/performance_table.csv`).
- `cases ...`: optional case names.

Environment overrides: none.

Output location: `--output_file`. Not config-rooted.

### `benchmarks/post-processing/aggregate_sim_lbs_batch_scaling.py`

Purpose: aggregates legacy Sim+LBS fixed-batch scaling runs.

Canonical/default command:

```bash
python benchmarks/post-processing/aggregate_sim_lbs_batch_scaling.py
```

Options:

- `--results_root PATH`: input root (default: `results/batch_scaling`).
- `--cases_file PATH`: case CSV (default: `data_config.csv`).
- `--output_table PATH`: per-case table
  (default: `results/batch_scaling/batch_scaling_table.csv`).
- `--output_overall PATH`: overall table
  (default: `results/batch_scaling/batch_scaling_overall.csv`).
- `cases ...`: optional case names.

Environment overrides: none.

Output location: `--output_table` and `--output_overall`. Not config-rooted.

### `benchmarks/post-processing/aggregate_sim_lbs_best_throughput.py`

Purpose: aggregates Sim+LBS autotune candidates into best and candidate tables.

Canonical/default command:

```bash
python benchmarks/post-processing/aggregate_sim_lbs_best_throughput.py
```

Options:

- `--results_root PATH`: input root (default: `results/batch_autotune`).
- `--cases_file PATH`: case CSV (default: `data_config.csv`).
- `--output_best PATH`: best table
  (default: `results/batch_autotune/best_throughput_table.csv`).
- `--output_candidates PATH`: candidate table
  (default: `results/batch_autotune/candidate_table.csv`).
- `--attempted_manifest PATH`: candidate manifest (default: unset).
- `--num_runs N`: number of run directories to consider (default: all).
- `cases ...`: optional case names.

Environment overrides: none.

Output location: `--output_best` and `--output_candidates`. Not config-rooted.

### `benchmarks/post-processing/compare_batched_render_boundaries.py`

Purpose: compares duplicated and shared-template boundary CSVs.

Canonical/default command:

```bash
python benchmarks/post-processing/compare_batched_render_boundaries.py
```

Options:

- `--duplicated_csv PATH`: duplicated-mode boundary CSV
  (default: `results/batched_render/boundary_duplicated.csv`).
- `--shared_template_csv PATH`: shared-template boundary CSV
  (default: `results/batched_render/boundary_shared_template.csv`).
- `--output PATH`: comparison CSV
  (default: `results/batched_render/boundary_comparison.csv`).

Environment overrides: none.

Output location: `--output`. Not config-rooted.

### `benchmarks/post-processing/summarize_sim_lbs_scaling_analysis.py`

Purpose: summarizes Sim+LBS scaling-analysis timing and NCU outputs.

Canonical/default command:

```bash
python benchmarks/post-processing/summarize_sim_lbs_scaling_analysis.py \
  --results_root results/sim_lbs_batch_scaling \
  --case_name single_lift_rope \
  --base_path ./data/different_types \
  --gaussian_path ./gaussian_output \
  --bg_img_path ./data/bg.png \
  --batch_sizes 1 32 64 128 256 512
```

Options:

- `--results_root PATH`: required input/output root.
- `--case_name NAME`: required case name.
- `--base_path PATH`: required case data root.
- `--gaussian_path PATH`: required Gaussian root.
- `--bg_img_path PATH`: required background image path.
- `--ncu_bin PATH`: Nsight Compute executable (default: empty).
- `--ncu_profile_frame_stride N`: NCU frame stride (default: unset).
- `--ncu_profile_max_frames N`: NCU max frames (default: `3`).
- `--ncu_profile_nvtx_name NAME`: NCU NVTX range
  (default: `sim_lbs_profile_frame`).
- `--ncu_target_processes VALUE`: NCU target process setting
  (default: `application-only`).
- `--ncu_metrics CSV`: NCU metric list (default: script metric list).
- `--script_path PATH`: runner used in generated commands
  (default: `benchmarks/run_sim_lbs_batch_scaling.sh`).
- `--batch_sizes N...`: required batch sizes.

Environment overrides: none.

Output location: writes CSVs, figures, summary text, generated profile commands,
metadata, and NCU JSONs under `--results_root`. Not config-rooted.

### `benchmarks/post-processing/plot_sim_lbs_rebuttal_scaling.py`

Purpose: creates the compact Sim+LBS scaling rebuttal figure.

Canonical/default command:

```bash
python benchmarks/post-processing/plot_sim_lbs_rebuttal_scaling.py
```

Options:

- `--scaling-csv PATH`: merged fixed-batch scaling CSV
  (default:
  `results/sim_lbs_batch_scaling_rebuttal/all_cases_batch_scaling_sim_lbs.csv`).
- `--best-csv PATH`: best-throughput CSV
  (default: `results/batch_autotune/best_throughput_table.csv`).
- `--output-dir PATH`: output directory
  (default: `results/sim_lbs_batch_scaling_rebuttal`).
- `--output-stem NAME`: PDF/PNG filename stem
  (default: `sim_lbs_rebuttal_scaling_figure`).
- `--throughput-stat median|mean`: throughput aggregation statistic
  (default: `median`).

Environment overrides: none.

Output location: PDF and PNG under `--output-dir`. Not config-rooted.

### `benchmarks/post-processing/plot_collision_pruning_overlay.py`

Purpose: creates collision/pruning overlay figures for selected spring-mass
patches.

Canonical/default command:

```bash
python benchmarks/post-processing/plot_collision_pruning_overlay.py
```

Options:

- `--case_name NAME`: case to plot (default: `double_stretch_sloth`).
- `--export_npz PATH`: exported data NPZ (default: script constant).
- `--config PATH`: config YAML (default: `configs/real.yaml`).
- `--output_dir PATH`: figure output directory
  (default: `results/figures/collision_pruning`).
- `--output_stem NAME`: output filename stem
  (default: `double_stretch_sloth_rest_motion_adjacency`).
- `--layout zoom_inset|full_gaussian_highlight|leg_focus_highlight|rest_map_anchor`:
  figure layout (default: `zoom_inset`).
- `--rest_frame N`: rest frame index (default: `0`).
- `--contact_frame N`: contact frame index (default: `41`).
- `--camera_index N`: camera index (default: `0`).
- `--selected_nodes CSV`: highlighted node IDs (default: curated node set).
- `--search_top_quantile R`: y-quantile for leg search (default: `0.35`).
- `--right_leg_margin_px R`: x-offset filter (default: `35.0`).
- `--torso_exclusion_y_quantile R`: torso exclusion quantile
  (default: `0.42`).
- `--preferred_patch_span_px R`: preferred patch span (default: `8.0`).
- `--min_patch_span_px R`: minimum patch span (default: `8.0`).
- `--max_patch_span_px R`: maximum patch span (default: `38.0`).
- `--visual_pad_px R`: visual padding (default: `0.50`).
- `--crop_size_px R`: crop size (default: `15.0`).
- `--local_radius_px R`: local patch radius (default: `19.0`).
- `--max_local_nodes N`: maximum local nodes (default: `5`).
- `--min_local_nodes N`: minimum local nodes (default: `3`).
- `--candidate_neighbor_limit N`: neighbor search limit (default: `9`).
- `--leg_crop_size_px R`: leg-focus crop size (default: `30.0`).
- `--selected_patch_max_nodes N`: highlighted connected nodes
  (default: `16`).
- `--anchor_node N`: anchor node for rest-map layout (default: `2351`).
- `--anchor_crop_size_px R`: anchor crop size (default: `45.0`).
- `--structural_radius_px R`: rest-map pruning radius (default: `7.5`).
- `--render_zoom_crop_size_px R`: rendered zoom crop size
  (default: `150.0`).
- `--max_gaussian_points_per_panel N`: point cap per panel (default: `36000`).
- `--gaussian_color_strength R`: Gaussian color blend strength
  (default: `0.58`).
- `--seed N`: random seed (default: `7`).
- `--dpi N`: output DPI (default: `300`).

Environment overrides: none.

Output location: figure files under `--output_dir`. Not config-rooted.

## Offline Helpers

### `benchmarks/scripts/prune_gaussians.py`

Purpose: prunes 3D Gaussian PLY files or directories with a top-k importance
score.

Canonical/default command:

```bash
python benchmarks/scripts/prune_gaussians.py \
  --input input.ply --output output.ply --keep-ratio 0.3
```

Options:

- `--input PATH`: required input PLY file or directory.
- `--output PATH`: required output PLY file or directory.
- `--keep-ratio R`: fraction of points to keep; required unless
  `--keep-count` is used.
- `--keep-count N`: number of points to keep per file; required unless
  `--keep-ratio` is used.
- `--mode opacity|opacity_area|opacity_volume`: importance score
  (default: `opacity_area`).

Environment overrides: none.

Output location: exactly `--output`, preserving relative paths for directory
inputs. Not config-rooted.
