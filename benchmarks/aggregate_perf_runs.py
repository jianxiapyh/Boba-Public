import argparse
import csv
import os
import re
from collections import OrderedDict


SUMMARY_PATTERN = re.compile(r"=== Final Summary \(averaged over (\d+) frames\) ===")
METRIC_MS_SHARE_PATTERN = re.compile(r"^(.+): ([+-]?\d+(?:\.\d+)?) ms \(([+-]?\d+(?:\.\d+)?)%\)$")
METRIC_MS_PATTERN = re.compile(r"^(.+): ([+-]?\d+(?:\.\d+)?) ms$")
METRIC_VALUE_PATTERN = re.compile(r"^(.+): ([+-]?\d+(?:\.\d+)?)$")


def read_cases(cases_file, explicit_cases):
    if explicit_cases:
        return list(explicit_cases)

    with open(cases_file, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


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
                label = match.group(1)
                metrics[f"{label} (ms)"] = float(match.group(2))
                metrics[f"{label} Share (%)"] = float(match.group(3))
                continue

            match = METRIC_MS_PATTERN.match(line)
            if match:
                metrics[f"{match.group(1)} (ms)"] = float(match.group(2))
                continue

            match = METRIC_VALUE_PATTERN.match(line)
            if match:
                metrics[match.group(1)] = float(match.group(2))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/perf")
    parser.add_argument("--cases_file", default="benchmarks/cases.txt")
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

    headers = ["Case Name", "successful_runs"]
    for metric_name in metric_order:
        for _, run_name in run_dirs:
            headers.append(f"{metric_name} {run_name}")
        headers.append(f"{metric_name} average")

    with open(args.output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for case_name in cases:
            row = {"Case Name": case_name}
            successful_runs = sum(
                1 for _, run_name in run_dirs if (run_name, case_name) in parsed
            )

            for metric_name in metric_order:
                values = []
                for _, run_name in run_dirs:
                    value = parsed.get((run_name, case_name), {}).get(metric_name)
                    row[f"{metric_name} {run_name}"] = format_value(value)
                    if value is not None:
                        values.append(value)
                row[f"{metric_name} average"] = format_value(
                    sum(values) / len(values) if values else None
                )

            row["successful_runs"] = successful_runs
            writer.writerow(row)


if __name__ == "__main__":
    main()
