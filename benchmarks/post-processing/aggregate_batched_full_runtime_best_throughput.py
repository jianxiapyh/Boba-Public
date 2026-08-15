import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict


SUMMARY_PATTERN = re.compile(r"=== Final Summary \(averaged over (\d+) frames\) ===")
METRIC_MS_SHARE_PATTERN = re.compile(
    r"^(.+): ([+-]?\d+(?:\.\d+)?) ms \(([+-]?\d+(?:\.\d+)?)%\)$"
)
METRIC_MS_PATTERN = re.compile(r"^(.+): ([+-]?\d+(?:\.\d+)?) ms$")
METRIC_VALUE_PATTERN = re.compile(r"^(.+): ([+-]?\d+(?:\.\d+)?)$")
LABEL_NORMALIZATION = {
    "Full motion interpolation": "Linear Blend Skinning",
    "Full Motion Interpolation": "Linear Blend Skinning",
}
TRACKED_METRICS = OrderedDict(
    [
        ("Frames Used", "frames_used_for_stats"),
        ("Average FPS", "average_fps"),
        ("Average Throughput (instances/s)", "average_throughput"),
        ("Average Total Frame Time (ms)", "average_total_frame_time_ms"),
        ("Simulator (ms)", "average_simulator_ms"),
        ("Simulator Share (%)", "average_simulator_share_pct"),
        ("Linear Blend Skinning (ms)", "average_full_motion_interpolation_ms"),
        (
            "Linear Blend Skinning Share (%)",
            "average_full_motion_interpolation_share_pct",
        ),
        ("Rendering (ms)", "average_rendering_ms"),
        ("Rendering Share (%)", "average_rendering_share_pct"),
        ("Frame compositing (ms)", "average_frame_compositing_ms"),
        ("Frame compositing Share (%)", "average_frame_compositing_share_pct"),
    ]
)
GAUSSIAN_RENDER_MODES = ("shared_template", "duplicated")
BATCHED_RENDER_VARIANTS = ("batch_original", "batch_optimized", "batch_prune")
BATCHED_RENDER_VARIANT_ALIASES = {
    "baseline": "batch_original",
    "optimized": "batch_optimized",
    "optimized_pruned": "batch_prune",
}
SIM_FORCE_MODE_GATHER = "gather"
SIM_FORCE_MODE_TEMPLATE_STATE_BATCHED_ATOMIC = "template_state_batched_atomic"
SIM_FORCE_MODES = (
    SIM_FORCE_MODE_GATHER,
    SIM_FORCE_MODE_TEMPLATE_STATE_BATCHED_ATOMIC,
)


def read_cases(cases_file, explicit_cases):
    if explicit_cases:
        return list(explicit_cases)

    with open(cases_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        return [row[0].strip() for row in reader if row and row[0].strip()]


def normalize_metric_label(label):
    return LABEL_NORMALIZATION.get(label, label)


def parse_summary_text(summary_path):
    metrics = {}
    with open(summary_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            match = SUMMARY_PATTERN.match(line)
            if match:
                metrics["Frames Used"] = float(match.group(1))
                continue

            match = METRIC_MS_SHARE_PATTERN.match(line)
            if match:
                label = normalize_metric_label(match.group(1))
                metrics[f"{label} (ms)"] = float(match.group(2))
                metrics[f"{label} Share (%)"] = float(match.group(3))
                continue

            match = METRIC_MS_PATTERN.match(line)
            if match:
                label = normalize_metric_label(match.group(1))
                metrics[f"{label} (ms)"] = float(match.group(2))
                continue

            match = METRIC_VALUE_PATTERN.match(line)
            if match:
                label = normalize_metric_label(match.group(1))
                metrics[label] = float(match.group(2))

    metrics.pop("Batch Size", None)
    metrics.pop("Render Mode", None)
    metrics.pop("Gaussian Render Mode", None)
    metrics.pop("Instance ID", None)
    return {metric_name: metrics.get(metric_name) for metric_name in TRACKED_METRICS}


def parse_summary_json(summary_path):
    with open(summary_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    return {
        metric_name: (
            float(metrics[json_key]) if metrics.get(json_key) is not None else None
        )
        for metric_name, json_key in TRACKED_METRICS.items()
    }


def load_summary_metrics(summary_dir):
    json_path = os.path.join(summary_dir, "performance_summary.json")
    if os.path.isfile(json_path):
        metrics = parse_summary_json(json_path)
        if (metrics.get("Frames Used") or 0) <= 0:
            return None
        return metrics

    text_path = os.path.join(summary_dir, "performance_summary.txt")
    if os.path.isfile(text_path):
        metrics = parse_summary_text(text_path)
        if (metrics.get("Frames Used") or 0) <= 0:
            return None
        return metrics

    return None


def sorted_run_dirs(results_root):
    run_dirs = []
    if not os.path.isdir(results_root):
        return run_dirs

    for name in os.listdir(results_root):
        match = re.fullmatch(r"run_(\d+)", name)
        if match:
            run_dirs.append((int(match.group(1)), name))
    run_dirs.sort()
    return run_dirs


def selected_run_dirs(results_root, num_runs):
    if num_runs is None:
        return sorted_run_dirs(results_root)
    if num_runs < 1:
        raise ValueError("--num_runs must be a positive integer.")
    return [(run_idx, f"run_{run_idx:02d}") for run_idx in range(1, num_runs + 1)]


def normalize_batched_render_variant(variant):
    if variant in BATCHED_RENDER_VARIANT_ALIASES:
        normalized = BATCHED_RENDER_VARIANT_ALIASES[variant]
        print(
            "[WARN] --batched_render_variant "
            f"{variant!r} is deprecated; use {normalized!r}.",
            file=sys.stderr,
        )
        return normalized
    return variant


def mode_dir(
    render_mode,
    instance_id,
    gaussian_render_mode,
    batch_image_resolution,
    batched_render_variant=None,
    sim_force_mode=SIM_FORCE_MODE_GATHER,
):
    if gaussian_render_mode not in GAUSSIAN_RENDER_MODES:
        raise ValueError(
            "--gaussian_render_mode must be 'shared_template' or 'duplicated'. "
            f"Received: {gaussian_render_mode}"
        )

    if render_mode == "instance":
        if instance_id is None:
            raise ValueError("--instance_id is required when --render_mode=instance.")
        selected = f"instance_{instance_id}_{gaussian_render_mode}"
        if batched_render_variant:
            selected = f"{selected}_{batched_render_variant}"
        if sim_force_mode != SIM_FORCE_MODE_GATHER:
            selected = f"{selected}_sim_{sim_force_mode}"
        return selected
    if render_mode == "batch_images":
        if instance_id is not None:
            raise ValueError("--instance_id can only be used with --render_mode=instance.")
        if batch_image_resolution not in ("native", "640x480"):
            raise ValueError(
                "--batch_image_resolution must be 'native' or '640x480'. "
                f"Received: {batch_image_resolution}"
            )
        mode_prefix = (
            "batch_images"
            if batch_image_resolution == "native"
            else f"batch_images_{batch_image_resolution}"
        )
        selected = f"{mode_prefix}_{gaussian_render_mode}"
        if batched_render_variant:
            selected = f"{selected}_{batched_render_variant}"
        if sim_force_mode != SIM_FORCE_MODE_GATHER:
            selected = f"{selected}_sim_{sim_force_mode}"
        return selected

    raise ValueError(
        "--render_mode must be 'instance' or 'batch_images'. "
        f"Received: {render_mode}"
    )


def sorted_attempted_batches(results_root, run_dirs, cases, selected_mode_dir):
    attempted = {case_name: set() for case_name in cases}

    for _, run_name in run_dirs:
        run_dir = os.path.join(results_root, run_name)
        if not os.path.isdir(run_dir):
            continue

        for case_name in cases:
            case_dir = os.path.join(run_dir, case_name)
            if not os.path.isdir(case_dir):
                continue

            for name in os.listdir(case_dir):
                match = re.fullmatch(r"batch_(\d+)", name)
                if not match:
                    continue

                batch_size = int(match.group(1))
                summary_dir = os.path.join(case_dir, name, selected_mode_dir)
                if os.path.isdir(summary_dir):
                    attempted[case_name].add(batch_size)

    return {
        case_name: sorted(batch_sizes)
        for case_name, batch_sizes in attempted.items()
    }


def read_attempted_manifest(manifest_path, cases):
    attempted = {case_name: set() for case_name in cases}
    if not manifest_path or not os.path.isfile(manifest_path):
        return None

    with open(manifest_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            case_name = row.get("Case Name", "").strip()
            batch_size = row.get("Batch Size", "").strip()
            if case_name not in attempted or not batch_size:
                continue
            attempted[case_name].add(int(batch_size))

    return {
        case_name: sorted(batch_sizes)
        for case_name, batch_sizes in attempted.items()
    }


def summary_dir(results_root, run_name, batch_size, case_name, selected_mode_dir):
    return os.path.join(
        results_root,
        run_name,
        case_name,
        f"batch_{batch_size}",
        selected_mode_dir,
    )


def format_value(value):
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_batch_list(batch_sizes):
    return " ".join(str(batch_size) for batch_size in batch_sizes)


def average(values):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def build_candidate_rows(results_root, run_dirs, cases, attempted_batches, selected_mode_dir):
    rows = []
    for case_name in cases:
        for batch_size in attempted_batches.get(case_name, []):
            metrics_by_run = []
            for _, run_name in run_dirs:
                metrics = load_summary_metrics(
                    summary_dir(
                        results_root,
                        run_name,
                        batch_size,
                        case_name,
                        selected_mode_dir,
                    )
                )
                if metrics is not None:
                    metrics_by_run.append(metrics)

            row = {
                "Case Name": case_name,
                "Batch Size": batch_size,
                "successful_runs": len(metrics_by_run),
            }
            for metric_name in TRACKED_METRICS:
                row[f"{metric_name} average"] = average(
                    [metrics.get(metric_name) for metrics in metrics_by_run]
                )
            rows.append(row)

    return rows


def choose_best_row(candidate_rows, min_successes):
    successful_rows = [
        row
        for row in candidate_rows
        if row["successful_runs"] >= min_successes
        and row.get("Average Throughput (instances/s) average") is not None
    ]
    if not successful_rows:
        return None

    return max(
        successful_rows,
        key=lambda row: (
            row["Average Throughput (instances/s) average"],
            -int(row["Batch Size"]),
        ),
    )


def write_candidate_table(output_path, rows):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    headers = ["Case Name", "Batch Size", "successful_runs"]
    headers.extend(f"{metric_name} average" for metric_name in TRACKED_METRICS)

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for metric_name in TRACKED_METRICS:
                key = f"{metric_name} average"
                formatted[key] = format_value(formatted.get(key))
            writer.writerow(formatted)


def write_best_table(output_path, cases, attempted_batches, candidate_rows, min_successes):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rows_by_case = {case_name: [] for case_name in cases}
    for row in candidate_rows:
        rows_by_case[row["Case Name"]].append(row)

    headers = [
        "Case Name",
        "Best Batch Size",
        "Best Average FPS",
        "Best Average Throughput (instances/s)",
        "Best Average Total Frame Time (ms)",
        "successful_runs",
        "searched_batch_sizes",
        "failed_batch_sizes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for case_name in cases:
            case_rows = rows_by_case.get(case_name, [])
            best_row = choose_best_row(case_rows, min_successes)
            failed_batches = [
                int(row["Batch Size"])
                for row in case_rows
                if row["successful_runs"] < min_successes
            ]

            row = {
                "Case Name": case_name,
                "searched_batch_sizes": format_batch_list(
                    attempted_batches.get(case_name, [])
                ),
                "failed_batch_sizes": format_batch_list(sorted(failed_batches)),
            }
            if best_row is None:
                row.update(
                    {
                        "Best Batch Size": "",
                        "Best Average FPS": "",
                        "Best Average Throughput (instances/s)": "",
                        "Best Average Total Frame Time (ms)": "",
                        "successful_runs": 0,
                    }
                )
            else:
                row.update(
                    {
                        "Best Batch Size": int(best_row["Batch Size"]),
                        "Best Average FPS": format_value(
                            best_row.get("Average FPS average")
                        ),
                        "Best Average Throughput (instances/s)": format_value(
                            best_row.get("Average Throughput (instances/s) average")
                        ),
                        "Best Average Total Frame Time (ms)": format_value(
                            best_row.get("Average Total Frame Time (ms) average")
                        ),
                        "successful_runs": int(best_row["successful_runs"]),
                    }
                )

            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root",
        default="results/batched_full_runtime_autotune",
    )
    parser.add_argument("--cases_file", default="data_config.csv")
    parser.add_argument(
        "--render_mode",
        choices=("instance", "batch_images"),
        default="batch_images",
    )
    parser.add_argument(
        "--gaussian_render_mode",
        choices=GAUSSIAN_RENDER_MODES,
        default="shared_template",
    )
    parser.add_argument("--instance_id", type=int, default=None)
    parser.add_argument(
        "--batch_image_resolution",
        choices=("native", "640x480"),
        default="640x480",
    )
    parser.add_argument(
        "--batched_render_variant",
        choices=BATCHED_RENDER_VARIANTS
        + tuple(BATCHED_RENDER_VARIANT_ALIASES.keys()),
        default=None,
    )
    parser.add_argument(
        "--sim_force_mode",
        choices=SIM_FORCE_MODES,
        default=SIM_FORCE_MODE_GATHER,
    )
    parser.add_argument(
        "--output_best",
        default="results/batched_full_runtime_autotune/best_throughput_table.csv",
    )
    parser.add_argument(
        "--output_candidates",
        default="results/batched_full_runtime_autotune/candidate_table.csv",
    )
    parser.add_argument("--attempted_manifest", default=None)
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--min_successes", type=int, default=1)
    parser.add_argument("cases", nargs="*")
    args = parser.parse_args()
    args.batched_render_variant = normalize_batched_render_variant(
        args.batched_render_variant
    )

    cases = read_cases(args.cases_file, args.cases)
    if args.min_successes < 1:
        raise ValueError("--min_successes must be a positive integer.")
    if args.num_runs is not None and args.min_successes > args.num_runs:
        raise ValueError("--min_successes cannot exceed --num_runs.")

    run_dirs = selected_run_dirs(args.results_root, args.num_runs)
    selected_mode_dir = mode_dir(
        args.render_mode,
        args.instance_id,
        args.gaussian_render_mode,
        args.batch_image_resolution,
        args.batched_render_variant,
        args.sim_force_mode,
    )
    attempted_batches = read_attempted_manifest(args.attempted_manifest, cases)
    if attempted_batches is None:
        attempted_batches = sorted_attempted_batches(
            args.results_root,
            run_dirs,
            cases,
            selected_mode_dir,
        )
    candidate_rows = build_candidate_rows(
        args.results_root,
        run_dirs,
        cases,
        attempted_batches,
        selected_mode_dir,
    )

    write_candidate_table(args.output_candidates, candidate_rows)
    write_best_table(
        args.output_best,
        cases,
        attempted_batches,
        candidate_rows,
        args.min_successes,
    )


if __name__ == "__main__":
    main()
