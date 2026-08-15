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
TRACKED_METRICS = OrderedDict(
    [
        ("Frames Used", "frames_used"),
        ("Average Batch FPS", "average_batch_fps"),
        ("Average Throughput (instances/s)", "average_throughput"),
        ("Average Sim+LBS Total (ms)", "average_sim_lbs_total_ms"),
        ("Simulator (ms)", "average_simulator_ms"),
        ("Simulator Share (%)", "average_simulator_share_pct"),
        ("Linear Blend Skinning (ms)", "average_lbs_ms"),
        ("Linear Blend Skinning Share (%)", "average_lbs_share_pct"),
    ]
)


def read_cases(cases_file, explicit_cases):
    if explicit_cases:
        return list(explicit_cases)

    with open(cases_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        return [row[0].strip() for row in reader if row and row[0].strip()]


def normalize_metric_label(label):
    return LABEL_NORMALIZATION.get(label, label)


def parse_summary(summary_path):
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
    return {
        metric_name: metrics.get(metric_name) for metric_name in TRACKED_METRICS
    }


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


def sorted_attempted_batches(results_root, run_dirs, cases):
    attempted = {case_name: set() for case_name in cases}

    for _, run_name in run_dirs:
        run_dir = os.path.join(results_root, run_name)
        if not os.path.isdir(run_dir):
            continue

        for name in os.listdir(run_dir):
            match = re.fullmatch(r"batch_(\d+)", name)
            if not match:
                continue

            batch_size = int(match.group(1))
            batch_dir = os.path.join(run_dir, name)
            for case_name in cases:
                if os.path.isdir(os.path.join(batch_dir, case_name)):
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


def summary_path(results_root, run_name, batch_size, case_name):
    return os.path.join(
        results_root,
        run_name,
        f"batch_{batch_size}",
        case_name,
        "performance_summary.txt",
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


def build_candidate_rows(results_root, run_dirs, cases, attempted_batches):
    rows = []
    for case_name in cases:
        for batch_size in attempted_batches.get(case_name, []):
            metrics_by_run = []
            for _, run_name in run_dirs:
                path = summary_path(results_root, run_name, batch_size, case_name)
                if os.path.isfile(path):
                    metrics_by_run.append(parse_summary(path))

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


def choose_best_row(candidate_rows):
    successful_rows = [
        row
        for row in candidate_rows
        if row["successful_runs"] > 0
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


def write_best_table(output_path, cases, attempted_batches, candidate_rows):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rows_by_case = {case_name: [] for case_name in cases}
    for row in candidate_rows:
        rows_by_case[row["Case Name"]].append(row)

    headers = [
        "Case Name",
        "Best Batch Size",
        "Best Average Batch FPS",
        "Best Average Throughput (instances/s)",
        "successful_runs",
        "searched_batch_sizes",
        "failed_batch_sizes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for case_name in cases:
            case_rows = rows_by_case.get(case_name, [])
            best_row = choose_best_row(case_rows)
            failed_batches = [
                int(row["Batch Size"]) for row in case_rows if row["successful_runs"] == 0
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
                        "Best Average Batch FPS": "",
                        "Best Average Throughput (instances/s)": "",
                        "successful_runs": 0,
                    }
                )
            else:
                row.update(
                    {
                        "Best Batch Size": int(best_row["Batch Size"]),
                        "Best Average Batch FPS": format_value(
                            best_row.get("Average Batch FPS average")
                        ),
                        "Best Average Throughput (instances/s)": format_value(
                            best_row.get("Average Throughput (instances/s) average")
                        ),
                        "successful_runs": int(best_row["successful_runs"]),
                    }
                )

            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results/batch_autotune")
    parser.add_argument("--cases_file", default="data_config.csv")
    parser.add_argument(
        "--output_best",
        default="results/batch_autotune/best_throughput_table.csv",
    )
    parser.add_argument(
        "--output_candidates",
        default="results/batch_autotune/candidate_table.csv",
    )
    parser.add_argument("--attempted_manifest", default=None)
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("cases", nargs="*")
    args = parser.parse_args()

    cases = read_cases(args.cases_file, args.cases)
    run_dirs = selected_run_dirs(args.results_root, args.num_runs)
    attempted_batches = read_attempted_manifest(args.attempted_manifest, cases)
    if attempted_batches is None:
        attempted_batches = sorted_attempted_batches(args.results_root, run_dirs, cases)
    candidate_rows = build_candidate_rows(
        args.results_root,
        run_dirs,
        cases,
        attempted_batches,
    )

    write_candidate_table(args.output_candidates, candidate_rows)
    write_best_table(args.output_best, cases, attempted_batches, candidate_rows)


if __name__ == "__main__":
    main()
