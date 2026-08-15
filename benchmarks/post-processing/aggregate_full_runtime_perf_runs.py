import argparse
import csv
import os
import re
from collections import OrderedDict


SUMMARY_PATTERN = re.compile(r"=== Final Summary \(averaged over (\d+) frames\) ===")
METRIC_MS_SHARE_PATTERN = re.compile(r"^(.+): ([+-]?\d+(?:\.\d+)?) ms \(([+-]?\d+(?:\.\d+)?)%\)$")
METRIC_MS_PATTERN = re.compile(r"^(.+): ([+-]?\d+(?:\.\d+)?) ms$")
METRIC_VALUE_PATTERN = re.compile(r"^(.+): ([+-]?\d+(?:\.\d+)?)$")
SUMMARY_AVERAGE_METRICS = [
    "Average FPS",
    "Average Total Frame Time (ms)",
    "Simulator (ms)",
    "Linear Blend Skinning (ms)",
    "Rendering (ms)",
    "Frame compositing (ms)",
]
LABEL_NORMALIZATION = {
    "Full motion interpolation": "Linear Blend Skinning",
    "Full Motion Interpolation": "Linear Blend Skinning",
}
EXCLUDED_DETAIL_METRICS = {"Frames Used"}


def read_cases(cases_file, explicit_cases):
    if explicit_cases:
        return list(explicit_cases)

    with open(cases_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        return [row[0].strip() for row in reader if row and row[0].strip()]


def normalize_metric_label(label):
    return LABEL_NORMALIZATION.get(label, label)


def parse_summary(summary_path):
    metrics = OrderedDict()
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

    return metrics


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


def build_case_row(case_name, run_dirs, metrics_for_averages, metrics_for_details, parsed):
    row = {"Case Name": case_name}
    successful_runs = sum(
        1 for _, run_name in run_dirs if (run_name, case_name) in parsed
    )

    metric_values = {}
    tracked_metrics = list(
        OrderedDict.fromkeys(metrics_for_averages + metrics_for_details).keys()
    )
    for metric_name in tracked_metrics:
        values = []
        for _, run_name in run_dirs:
            value = parsed.get((run_name, case_name), {}).get(metric_name)
            if metric_name in metrics_for_details:
                row[f"{metric_name} {run_name}"] = format_value(value)
            if value is not None:
                values.append(value)
        average_value = sum(values) / len(values) if values else None
        if metric_name in metrics_for_averages:
            row[f"{metric_name} average"] = format_value(average_value)
        metric_values[metric_name] = {
            "per_run": {
                run_name: parsed.get((run_name, case_name), {}).get(metric_name)
                for _, run_name in run_dirs
            },
            "average": average_value,
        }

    row["successful_runs"] = successful_runs
    return row, metric_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/perf")
    parser.add_argument("--cases_file", default="data_config.csv")
    parser.add_argument(
        "--output_file",
        default="results/perf/performance_table.csv",
    )
    parser.add_argument("cases", nargs="*")
    args = parser.parse_args()

    cases = read_cases(args.cases_file, args.cases)
    run_dirs = sorted_run_dirs(args.results_root)

    parsed = {}
    metric_order = []
    metric_seen = set()

    for run_idx, run_name in run_dirs:
        for case_name in cases:
            summary_path = os.path.join(
                args.results_root, run_name, case_name, "performance_summary.txt"
            )
            if not os.path.isfile(summary_path):
                continue

            metrics = parse_summary(summary_path)
            parsed[(run_name, case_name)] = metrics
            for metric_name in metrics:
                if metric_name not in metric_seen:
                    metric_seen.add(metric_name)
                    metric_order.append(metric_name)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    metrics_for_averages = list(SUMMARY_AVERAGE_METRICS)
    detail_priority = list(SUMMARY_AVERAGE_METRICS)
    metrics_for_details = [
        metric_name
        for metric_name in detail_priority
        if metric_name in metric_seen and metric_name not in EXCLUDED_DETAIL_METRICS
    ]
    metrics_for_details.extend(
        metric_name
        for metric_name in metric_order
        if metric_name not in metrics_for_details
        and metric_name not in EXCLUDED_DETAIL_METRICS
    )

    headers = ["Case Name", "successful_runs"]
    headers.extend(f"{metric_name} average" for metric_name in metrics_for_averages)
    for metric_name in metrics_for_details:
        for _, run_name in run_dirs:
            headers.append(f"{metric_name} {run_name}")

    with open(args.output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        overall_metric_values = {
            metric_name: {"per_run": {run_name: [] for _, run_name in run_dirs}, "average": []}
            for metric_name in OrderedDict.fromkeys(metrics_for_averages + metrics_for_details)
        }
        successful_cases = 0

        for case_name in cases:
            row, metric_values = build_case_row(
                case_name,
                run_dirs,
                metrics_for_averages,
                metrics_for_details,
                parsed,
            )
            writer.writerow(row)

            if int(row["successful_runs"]) > 0:
                successful_cases += 1

            for metric_name in overall_metric_values:
                average_value = metric_values[metric_name]["average"]
                if average_value is not None:
                    overall_metric_values[metric_name]["average"].append(average_value)

                for _, run_name in run_dirs:
                    run_value = metric_values[metric_name]["per_run"][run_name]
                    if run_value is not None:
                        overall_metric_values[metric_name]["per_run"][run_name].append(
                            run_value
                        )

        overall_row = {"Case Name": "OVERALL", "successful_runs": successful_cases}
        for metric_name in metrics_for_averages:
            average_values = overall_metric_values[metric_name]["average"]
            overall_row[f"{metric_name} average"] = format_value(
                sum(average_values) / len(average_values) if average_values else None
            )

        for metric_name in metrics_for_details:
            for _, run_name in run_dirs:
                values = overall_metric_values[metric_name]["per_run"][run_name]
                overall_row[f"{metric_name} {run_name}"] = format_value(
                    sum(values) / len(values) if values else None
                )

        writer.writerow(overall_row)


if __name__ == "__main__":
    main()
