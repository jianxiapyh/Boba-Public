import argparse
import csv
import os
import re
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


def sorted_batch_sizes(results_root, run_dirs):
    batch_sizes = set()
    for _, run_name in run_dirs:
        run_dir = os.path.join(results_root, run_name)
        if not os.path.isdir(run_dir):
            continue
        for name in os.listdir(run_dir):
            match = re.fullmatch(r"batch_(\d+)", name)
            if match:
                batch_sizes.add(int(match.group(1)))
    return sorted(batch_sizes)


def format_value(value):
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/batch_scaling")
    parser.add_argument("--cases_file", default="data_config.csv")
    parser.add_argument(
        "--output_table",
        default="results/batch_scaling/batch_scaling_table.csv",
    )
    parser.add_argument(
        "--output_overall",
        default="results/batch_scaling/batch_scaling_overall.csv",
    )
    parser.add_argument("cases", nargs="*")
    args = parser.parse_args()

    cases = read_cases(args.cases_file, args.cases)
    run_dirs = sorted_run_dirs(args.results_root)
    batch_sizes = sorted_batch_sizes(args.results_root, run_dirs)

    parsed = {}
    metric_order = []
    metric_seen = set()

    for _, run_name in run_dirs:
        for batch_size in batch_sizes:
            for case_name in cases:
                summary_path = os.path.join(
                    args.results_root,
                    run_name,
                    f"batch_{batch_size}",
                    case_name,
                    "performance_summary.txt",
                )
                if not os.path.isfile(summary_path):
                    continue

                metrics = parse_summary(summary_path)
                parsed[(run_name, batch_size, case_name)] = metrics
                for metric_name in metrics:
                    if metric_name not in metric_seen:
                        metric_seen.add(metric_name)
                        metric_order.append(metric_name)

    for output_file in (args.output_table, args.output_overall):
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    table_headers = ["Case Name", "Batch Size", "successful_runs"]
    for metric_name in metric_order:
        for _, run_name in run_dirs:
            table_headers.append(f"{metric_name} {run_name}")
        table_headers.append(f"{metric_name} average")

    with open(args.output_table, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=table_headers)
        writer.writeheader()

        for batch_size in batch_sizes:
            for case_name in cases:
                row = {"Case Name": case_name, "Batch Size": batch_size}
                successful_runs = sum(
                    1
                    for _, run_name in run_dirs
                    if (run_name, batch_size, case_name) in parsed
                )

                for metric_name in metric_order:
                    values = []
                    for _, run_name in run_dirs:
                        value = parsed.get((run_name, batch_size, case_name), {}).get(
                            metric_name
                        )
                        row[f"{metric_name} {run_name}"] = format_value(value)
                        if value is not None:
                            values.append(value)
                    row[f"{metric_name} average"] = format_value(
                        sum(values) / len(values) if values else None
                    )

                row["successful_runs"] = successful_runs
                writer.writerow(row)

    overall_headers = ["Batch Size", "successful_cases"]
    for metric_name in metric_order:
        overall_headers.append(metric_name)

    with open(args.output_overall, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=overall_headers)
        writer.writeheader()

        for batch_size in batch_sizes:
            row = {"Batch Size": batch_size}
            successful_cases = 0

            for metric_name in metric_order:
                case_averages = []
                for case_name in cases:
                    run_values = []
                    for _, run_name in run_dirs:
                        value = parsed.get((run_name, batch_size, case_name), {}).get(
                            metric_name
                        )
                        if value is not None:
                            run_values.append(value)

                    if run_values:
                        case_averages.append(sum(run_values) / len(run_values))

                if metric_name == metric_order[0]:
                    successful_cases = len(case_averages)
                row[metric_name] = format_value(
                    sum(case_averages) / len(case_averages) if case_averages else None
                )

            row["successful_cases"] = successful_cases
            writer.writerow(row)


if __name__ == "__main__":
    main()
