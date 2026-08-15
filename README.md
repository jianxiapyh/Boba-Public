# PhysTwin Open Source

This README documents the `PhysTwin` branch layout inside `Boba.git`.

`PhysTwin` keeps the original PhysTwin simulation pipeline for spring-mass simulation, motion interpolation, Gaussian rendering, and frame compositing, but wraps it in the same public runtime and benchmark surface used by `Boba_OpenSource`.

## Branch Scope

This branch is intentionally focused on runtime replay and benchmarking.

- Public runtime script: `interactive_playground.py`
- Public runtime modes: `--mode perf` and `--mode quality`
- Perf outputs: `results/perf/...`
- Quality outputs: `results/quality/...`
- Public compatibility option: `--inv_ctrl`

This branch does not ship the alternate experiment variants from the original PhysTwin repo. Only the canonical trainer, simulator, and dynamic-utils path are included.

`quality` automatically enables evaluation output and runs `gaussian_splatting/img2video.py`.

`perf` does not generate videos because that would contaminate timing measurements.

## Setup

```bash
export PATH={YOUR_DIR}/cuda/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH={YOUR_DIR}/cuda/cuda-12.1/lib64:$LD_LIBRARY_PATH

conda create -y -n phystwin python=3.10
conda activate phystwin

bash ./env_install/env_install.sh
```

## Required External Assets

Large runtime assets are intentionally distributed separately from this branch.

Download these folders and place them at the repository root:

- [data](https://drive.google.com/file/d/1A6X7X6yZFYJ8oo6Bd5LLn-RldeCKJw5Z/view?usp=sharing)
- [experiments_optimization](https://drive.google.com/file/d/1xKlk3WumFp1Qz31NB4DQxos8jMD_pBAt/view?usp=sharing)
- [experiments](https://drive.google.com/file/d/1hCGzdGlzL4qvZV3GzOCGiaVBshDgFKjq/view?usp=sharing)
- [gaussian_output](https://drive.google.com/file/d/12EoxhEhE90NMAqLlQoj_zM_C63BOftNW/view?usp=sharing)

The expected layout is:

```text
data/different_types/<case>/final_data.pkl
data/different_types/<case>/calibrate.pkl
data/different_types/<case>/metadata.json
data/different_types_human_mask/<case>/...
experiments/<case>/train/best_*.pth
experiments_optimization/<case>/optimal_params.pkl
gaussian_output/<case>/.../point_cloud.ply
```

`data/render_eval_data` is prepared automatically by the evaluation flow and does not need to be downloaded separately.

## Run One Case

Performance mode:

```bash
python interactive_playground.py --mode perf --case_name double_lift_cloth_3
```

Quality mode:

```bash
python interactive_playground.py --mode quality --case_name double_lift_cloth_3
```

Useful arguments:

- `--base_path`: defaults to `./data/different_types`
- `--gaussian_path`: defaults to `./gaussian_output`
- `--bg_img_path`: defaults to `./data/bg.png`
- `--case_name`: case to replay
- `--mode`: `perf` or `quality`
- `--output_dir`: defaults to `./results/<mode>/<case_name>`
- `--num_views`: number of calibrated views to export during quality mode
- `--inv_ctrl`: invert the manual horizontal control direction, default `false`

The branch infers controller-part count internally:

- case names containing `single` use one control part
- case names containing `double` and `weird_package` use two control parts

## Benchmark All Cases

Run performance for all cases:

```bash
bash benchmarks/run_all_perf.sh
```

This runs each case `NUM_RUNS` times. The default is `3`, and you can override it:

```bash
NUM_RUNS=5 bash benchmarks/run_all_perf.sh
```

Run quality for all cases:

```bash
bash benchmarks/run_all_quality.sh
```

That script refreshes `data/render_eval_data` automatically before chamfer, track, and render metrics are computed, so you do not need to run `python export_render_eval_data.py` manually for the standard workflow.

You can choose how many calibrated render views to generate and evaluate:

```bash
bash benchmarks/run_all_quality.sh --num_views 3
```

Valid values are `1`, `2`, or `3`.

The canonical case list lives in `benchmarks/cases.txt`.

## Outputs

Perf mode writes one case directory per run:

```text
results/perf/run_01/<case>/performance_summary.txt
results/perf/logs/run_01/<case>.log
```

After aggregation, the consolidated table is:

```text
results/perf/performance_table.csv
```

That CSV contains one row per case, one column for each metric/run combination, one average column per metric, and `successful_runs`.

Quality mode writes:

```text
results/quality/<case>/0/*.png
results/quality/<case>/1/*.png
results/quality/<case>/2/*.png
results/quality/<case>/0.mp4
results/quality/<case>/1.mp4
results/quality/<case>/2.mp4
results/quality/<case>/0_realtime.mp4
results/quality/<case>/1_realtime.mp4
results/quality/<case>/2_realtime.mp4
results/quality/<case>/output/*.png
results/quality/<case>/output.mp4
results/quality/<case>/output_realtime.mp4
results/quality/<case>/inference.pkl
results/quality/<case>/performance_summary.txt
results/quality/<case>/performance_summary.json
```

Quality metrics are written to:

```text
results/quality/metrics/chamfer.csv
results/quality/metrics/track.csv
results/quality/metrics/render_metrics.csv
results/quality/metrics/render_metrics.txt
```

`chamfer.csv` and `track.csv` include a final `OVERALL` row. `render_metrics.csv` keeps its `OVERALL` row and writes those aggregate values with 3 decimal places.

## Compare to the Paper

After a quality benchmark run, compare the branch metrics to the paper table with:

```bash
python benchmarks/compare_to_paper.py
```

This reads the aggregated `OVERALL` train/test rows from:

- `results/quality/metrics/chamfer.csv`
- `results/quality/metrics/track.csv`
- `results/quality/metrics/render_metrics.csv`

and writes:

```text
results/quality/metrics/paper_comparison.csv
results/quality/metrics/paper_comparison.md
```

The comparison maps:

- `train` -> `Reconstruction & Resimulation`
- `test` -> `Future Prediction`

If the reported metrics are not very similar to the paper numbers, the generated markdown report includes a concrete investigation checklist focused on splits, frame alignment, render baseline preparation, view-count semantics, and representative-case comparison against `PhysTwin_original`.

## Notes

- `quality` automatically implies evaluation output.
- `quality` automatically runs `gaussian_splatting/img2video.py`.
- `quality` evaluation automatically refreshes `data/render_eval_data` before metrics are computed.
- `--num_views` controls both how many raw render folders are generated and how many views `evaluate_render.py` scores.
- `0.mp4` and `output.mp4` use the dataset capture FPS from `metadata.json`.
- `0_realtime.mp4` and `output_realtime.mp4` use the measured average FPS from the quality run.
- `perf` keeps video generation off on purpose so timing numbers stay meaningful.
- This branch is runtime and benchmarking only; it intentionally omits the broader training/data-processing and experiment-variant surface from `PhysTwin_original`.

## Citation

If this branch is useful for your research, please cite the original PhysTwin work:

```bibtex
@article{jiang2025phystwin,
    title={PhysTwin: Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos},
    author={Jiang, Hanxiao and Hsu, Hao-Yu and Zhang, Kaifeng and Yu, Hsin-Ni and Wang, Shenlong and Li, Yunzhu},
    journal={arXiv preprint arXiv:2503.17973},
    year={2025}
}
```
