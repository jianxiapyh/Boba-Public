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
        ("Average FPS", "average_fps"),
        ("Average Throughput (instances/s)", "average_throughput"),
        ("Average Total Frame Time (ms)", "average_total_frame_time_ms"),
        ("Simulator (ms)", "average_simulator_ms"),
        ("Linear Blend Skinning (ms)", "average_full_motion_interpolation_ms"),
        ("Rendering (ms)", "average_rendering_ms"),
        ("Frame compositing (ms)", "average_frame_compositing_ms"),
        ("Simulator Share (%)", "average_simulator_share_pct"),
        (
            "Linear Blend Skinning Share (%)",
            "average_full_motion_interpolation_share_pct",
        ),
        ("Rendering Share (%)", "average_rendering_share_pct"),
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
        return parse_summary_json(json_path)

    text_path = os.path.join(summary_dir, "performance_summary.txt")
    if os.path.isfile(text_path):
        return parse_summary_text(text_path)

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


def format_value(value):
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def average(values):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def compute_delta(current_value, baseline_value):
    if current_value is None or baseline_value is None:
        return None
    return current_value - baseline_value


def compute_ratio(current_value, baseline_value):
    if current_value is None or baseline_value in (None, 0):
        return None
    return current_value / baseline_value


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
        return selected

    raise ValueError(
        "--render_mode must be 'instance' or 'batch_images'. "
        f"Received: {render_mode}"
    )


def sorted_batch_sizes(results_root, run_dirs, cases, selected_mode_dir):
    batch_sizes = set()
    for _, run_name in run_dirs:
        for case_name in cases:
            case_dir = os.path.join(results_root, run_name, case_name)
            if not os.path.isdir(case_dir):
                continue

            for name in os.listdir(case_dir):
                match = re.fullmatch(r"batch_(\d+)", name)
                if not match:
                    continue

                summary_dir = os.path.join(case_dir, name, selected_mode_dir)
                if os.path.isdir(summary_dir):
                    batch_sizes.add(int(match.group(1)))

    return sorted(batch_sizes)


def build_case_average_lookup(parsed, run_dirs, batch_sizes, cases):
    case_averages = {}
    for batch_size in batch_sizes:
        for case_name in cases:
            for metric_name in TRACKED_METRICS:
                run_values = [
                    parsed.get((run_name, batch_size, case_name), {}).get(metric_name)
                    for _, run_name in run_dirs
                ]
                case_averages[(batch_size, case_name, metric_name)] = average(run_values)
    return case_averages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/batched_render")
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
        default="native",
    )
    parser.add_argument(
        "--batched_render_variant",
        choices=BATCHED_RENDER_VARIANTS
        + tuple(BATCHED_RENDER_VARIANT_ALIASES.keys()),
        default=None,
    )
    parser.add_argument(
        "--output_table",
        default=None,
    )
    parser.add_argument(
        "--output_overall",
        default=None,
    )
    parser.add_argument("cases", nargs="*")
    args = parser.parse_args()
    args.batched_render_variant = normalize_batched_render_variant(
        args.batched_render_variant
    )

    resolution_suffix = (
        "" if args.batch_image_resolution == "native" else f"_{args.batch_image_resolution}"
    )
    variant_suffix = (
        "" if args.batched_render_variant is None else f"_{args.batched_render_variant}"
    )
    if args.output_table is None:
        args.output_table = (
            "results/batched_render/"
            f"batch_scaling_{args.gaussian_render_mode}{resolution_suffix}{variant_suffix}_table.csv"
        )
    if args.output_overall is None:
        args.output_overall = (
            "results/batched_render/"
            f"batch_scaling_{args.gaussian_render_mode}{resolution_suffix}{variant_suffix}_overall.csv"
        )

    selected_mode_dir = mode_dir(
        args.render_mode,
        args.instance_id,
        args.gaussian_render_mode,
        args.batch_image_resolution,
        args.batched_render_variant,
    )
    cases = read_cases(args.cases_file, args.cases)
    run_dirs = sorted_run_dirs(args.results_root)
    batch_sizes = sorted_batch_sizes(args.results_root, run_dirs, cases, selected_mode_dir)

    parsed = {}
    for _, run_name in run_dirs:
        for batch_size in batch_sizes:
            for case_name in cases:
                summary_dir = os.path.join(
                    args.results_root,
                    run_name,
                    case_name,
                    f"batch_{batch_size}",
                    selected_mode_dir,
                )
                metrics = load_summary_metrics(summary_dir)
                if metrics is None:
                    continue
                parsed[(run_name, batch_size, case_name)] = metrics

    for output_file in (args.output_table, args.output_overall):
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    case_averages = build_case_average_lookup(parsed, run_dirs, batch_sizes, cases)

    table_headers = ["Case Name", "Batch Size", "successful_runs"]
    for metric_name in TRACKED_METRICS:
        table_headers.append(f"{metric_name} average")
    for metric_name in TRACKED_METRICS:
        table_headers.append(f"{metric_name} delta_vs_batch_1")
    for metric_name in TRACKED_METRICS:
        table_headers.append(f"{metric_name} ratio_vs_batch_1")
    for metric_name in TRACKED_METRICS:
        for _, run_name in run_dirs:
            table_headers.append(f"{metric_name} {run_name}")

    with open(args.output_table, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=table_headers)
        writer.writeheader()

        for batch_size in batch_sizes:
            for case_name in cases:
                row = {"Case Name": case_name, "Batch Size": batch_size}
                row["successful_runs"] = sum(
                    1
                    for _, run_name in run_dirs
                    if (run_name, batch_size, case_name) in parsed
                )

                for metric_name in TRACKED_METRICS:
                    current_average = case_averages[(batch_size, case_name, metric_name)]
                    baseline_average = case_averages.get((1, case_name, metric_name))
                    row[f"{metric_name} average"] = format_value(current_average)
                    row[f"{metric_name} delta_vs_batch_1"] = format_value(
                        compute_delta(current_average, baseline_average)
                    )
                    row[f"{metric_name} ratio_vs_batch_1"] = format_value(
                        compute_ratio(current_average, baseline_average)
                    )

                    for _, run_name in run_dirs:
                        run_value = parsed.get((run_name, batch_size, case_name), {}).get(
                            metric_name
                        )
                        row[f"{metric_name} {run_name}"] = format_value(run_value)

                writer.writerow(row)

    overall_headers = ["Batch Size", "successful_cases"]
    for metric_name in TRACKED_METRICS:
        overall_headers.append(metric_name)
    for metric_name in TRACKED_METRICS:
        overall_headers.append(f"{metric_name} delta_vs_batch_1")
    for metric_name in TRACKED_METRICS:
        overall_headers.append(f"{metric_name} ratio_vs_batch_1")

    overall_baselines = {}
    for metric_name in TRACKED_METRICS:
        overall_baselines[metric_name] = average(
            [case_averages.get((1, case_name, metric_name)) for case_name in cases]
        )

    with open(args.output_overall, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=overall_headers)
        writer.writeheader()

        for batch_size in batch_sizes:
            row = {"Batch Size": batch_size}
            row["successful_cases"] = sum(
                1
                for case_name in cases
                if any((run_name, batch_size, case_name) in parsed for _, run_name in run_dirs)
            )

            for metric_name in TRACKED_METRICS:
                current_average = average(
                    [
                        case_averages[(batch_size, case_name, metric_name)]
                        for case_name in cases
                    ]
                )
                baseline_average = overall_baselines[metric_name]
                row[metric_name] = format_value(current_average)
                row[f"{metric_name} delta_vs_batch_1"] = format_value(
                    compute_delta(current_average, baseline_average)
                )
                row[f"{metric_name} ratio_vs_batch_1"] = format_value(
                    compute_ratio(current_average, baseline_average)
                )

            writer.writerow(row)


if __name__ == "__main__":
    main()
