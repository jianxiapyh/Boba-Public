import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


CSV_COLUMNS = [
    "batch_size",
    "total_sim_lbs_ms",
    "simulation_ms",
    "lbs_ms",
    "throughput_instances_per_sec",
    "per_instance_ms",
    "scaling_efficiency_vs_b1",
    "peak_allocated_gb",
    "peak_reserved_gb",
    "gpu_total_memory_gb",
    "loop_peak_allocated_gb",
    "loop_peak_reserved_gb",
    "dram_bw_gb_s",
    "sm_util_pct",
    "dram_bw_pct_peak",
    "achieved_occupancy_pct",
    "bottleneck_class",
    "bottleneck_reason",
    "ncu_status",
    "status",
    "notes",
]

NCU_PERCENT_METRIC_TO_COLUMN = {
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_util_pct",
    "dram__bytes.sum.pct_of_peak_sustained_elapsed": "dram_bw_pct_peak",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
}
NCU_DURATION_METRIC = "gpu__time_duration.sum"
NCU_DRAM_BYTES_METRIC = "dram__bytes.sum"
DEFAULT_NCU_METRICS = [
    NCU_DURATION_METRIC,
    NCU_DRAM_BYTES_METRIC,
    "dram__bytes.sum.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]
SUCCESSFUL_ATTEMPT_STATUSES = {"ok", "skipped_existing"}
BOTTLENECK_COLORS = {
    "bandwidth-pressure": "#4C78A8",
    "compute-pressure": "#F58518",
    "memory-capacity-pressure": "#E45756",
    "under-utilized": "#72B7B2",
    "unknown": "#9D9D9D",
}
BOTTLENECK_LABELS = {
    "bandwidth-pressure": "Bandwidth pressure",
    "compute-pressure": "Compute pressure",
    "memory-capacity-pressure": "Memory capacity pressure",
    "under-utilized": "Under-utilized",
    "unknown": "Unknown",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--case_name", required=True)
    parser.add_argument("--base_path", required=True)
    parser.add_argument("--gaussian_path", required=True)
    parser.add_argument("--bg_img_path", required=True)
    parser.add_argument("--ncu_bin", default="")
    parser.add_argument("--ncu_profile_frame_stride", type=int, default=None)
    parser.add_argument("--ncu_profile_max_frames", type=int, default=3)
    parser.add_argument("--ncu_profile_nvtx_name", default="sim_lbs_profile_frame")
    parser.add_argument("--ncu_target_processes", default="application-only")
    parser.add_argument("--ncu_metrics", default=",".join(DEFAULT_NCU_METRICS))
    parser.add_argument("--script_path", default="benchmarks/run_sim_lbs_batch_scaling.sh")
    parser.add_argument("--batch_sizes", nargs="+", type=int, required=True)
    return parser.parse_args()


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "n/a", "na", "--"}:
        return None
    text = text.replace("%", "").replace(",", "")
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def normalize_duration_seconds(value, unit):
    if value is None:
        return None
    text = (unit or "").strip().lower()
    text = text.replace(" ", "").replace("_", "")
    if text in {"s", "sec", "secs", "second", "seconds"}:
        return value
    if text in {"ms", "msec", "msecs", "msecond", "mseconds", "millisecond", "milliseconds"}:
        return value * 1e-3
    if text in {"us", "usec", "usecs", "usecond", "useconds", "microsecond", "microseconds"}:
        return value * 1e-6
    if text in {"ns", "nsec", "nsecs", "nsecond", "nseconds", "nanosecond", "nanoseconds"}:
        return value * 1e-9
    if text:
        return None
    return None


def normalize_bytes(value, unit):
    if value is None:
        return None
    text = (unit or "").strip().lower()
    text = text.replace(" ", "").replace("_", "")
    if text in {"", "b", "byte", "bytes"}:
        return value
    if text in {"kb", "kbyte", "kbytes", "kilobyte", "kilobytes"}:
        return value * 1e3
    if text in {"mb", "mbyte", "mbytes", "megabyte", "megabytes"}:
        return value * 1e6
    if text in {"gb", "gbyte", "gbytes", "gigabyte", "gigabytes"}:
        return value * 1e9
    return None


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_attempts(results_root):
    attempts_path = Path(results_root) / "batch_scaling_sim_lbs_attempts.csv"
    if not attempts_path.is_file():
        return []
    with open(attempts_path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def is_oom_log(path):
    if not path or not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read().lower()
    except OSError:
        return False
    oom_markers = [
        "out of memory",
        "cuda error: out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cusparse_status_alloc_failed",
        "std::bad_alloc",
    ]
    return any(marker in text for marker in oom_markers)


def collect_timing_metrics(results_root, case_name, batch_size):
    pattern = (
        Path(results_root)
        / "run_*"
        / f"batch_{batch_size}"
        / case_name
        / "scaling_metrics.json"
    )
    return [read_json(path) for path in sorted(Path(results_root).glob(str(pattern.relative_to(results_root))))]


def collect_ncu_profile_metrics(results_root, case_name, batch_size):
    path = (
        Path(results_root)
        / "ncu_runs"
        / f"batch_{batch_size}"
        / case_name
        / "ncu_profile_metrics.json"
    )
    if not path.is_file():
        return None
    return read_json(path)


def split_ncu_metrics(metrics_text):
    return [metric.strip() for metric in metrics_text.split(",") if metric.strip()]


def mean(values):
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def classify_bottleneck(row, near_larger_capacity_failure):
    status = row.get("status")
    if status != "success":
        if status == "oom":
            return "memory-capacity-pressure", "batch failed with OOM"
        return "unknown", f"batch status={status}"

    sm = row.get("sm_util_pct")
    dram = row.get("dram_bw_pct_peak")
    efficiency = row.get("scaling_efficiency_vs_b1")
    peak_reserved = row.get("peak_reserved_gb")
    gpu_total_memory = row.get("gpu_total_memory_gb")
    ncu_status = row.get("ncu_status")

    if near_larger_capacity_failure:
        return (
            "memory-capacity-pressure",
            "next larger measured batch failed/OOM; memory capacity pressure may be possible near this region",
        )

    if peak_reserved is not None and gpu_total_memory is not None and gpu_total_memory > 0:
        reserved_fraction = peak_reserved / gpu_total_memory
        if reserved_fraction >= 0.85:
            return (
                "memory-capacity-pressure",
                f"peak reserved memory {peak_reserved:.2f}/{gpu_total_memory:.2f} GB ({reserved_fraction * 100:.1f}%)",
            )

    if sm is None or dram is None or ncu_status != "ok":
        reason = "insufficient NCU evidence"
        if ncu_status:
            reason += f" (ncu_status={ncu_status})"
        return "unknown", reason

    sublinear = efficiency is not None and efficiency < 0.70
    high_dram = dram >= 60.0
    high_sm = sm >= 60.0
    low_dram = dram < 35.0
    low_sm = sm < 35.0

    if high_dram and (sublinear or dram >= sm + 10.0):
        reason = f"DRAM {dram:.1f}% with SM {sm:.1f}%"
        if sublinear:
            reason += f" and scaling efficiency {efficiency:.2f}"
        return "bandwidth-pressure", reason

    if high_sm and dram < 60.0:
        return "compute-pressure", f"SM {sm:.1f}% with DRAM {dram:.1f}%"

    if low_sm and low_dram:
        return "under-utilized", f"SM {sm:.1f}% and DRAM {dram:.1f}% are both low"

    if peak_reserved is not None and peak_reserved > 0:
        return (
            "unknown",
            f"mixed utilization evidence; peak reserved memory {peak_reserved:.2f} GB",
        )
    return "unknown", "mixed utilization evidence"


def annotate_bottlenecks(rows):
    capacity_failure_batches = [
        row["batch_size"]
        for row in rows
        if row.get("status") in {"oom", "failed"} and row.get("batch_size") is not None
    ]
    successful_batches = [
        row["batch_size"]
        for row in rows
        if row.get("status") == "success" and row.get("batch_size") is not None
    ]
    for row in rows:
        batch_size = row.get("batch_size")
        near_larger_capacity_failure = False
        if row.get("status") == "success" and batch_size is not None:
            for failure_batch in capacity_failure_batches:
                if failure_batch <= batch_size:
                    continue
                has_success_between = any(
                    batch_size < successful_batch < failure_batch
                    for successful_batch in successful_batches
                )
                if not has_success_between:
                    near_larger_capacity_failure = True
                    break
        bottleneck_class, bottleneck_reason = classify_bottleneck(
            row, near_larger_capacity_failure
        )
        row["bottleneck_class"] = bottleneck_class
        row["bottleneck_reason"] = bottleneck_reason


def average_timing_metrics(metrics_list):
    fields = [
        "total_sim_lbs_ms",
        "simulation_ms",
        "lbs_ms",
        "throughput_instances_per_sec",
        "per_instance_ms",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "gpu_total_memory_gb",
        "measured_iterations",
        "warmup_iterations",
    ]
    averaged = {}
    for field in fields:
        averaged[field] = mean([parse_float(item.get(field)) for item in metrics_list])
    return averaged


def header_lookup(header):
    return {name.strip().lower(): idx for idx, name in enumerate(header)}


def get_index(lookup, *names):
    for name in names:
        idx = lookup.get(name)
        if idx is not None:
            return idx
    return None


def parse_ncu_csv(path):
    if not path or not os.path.isfile(path):
        return {}, "missing", ["ncu_csv_missing"]

    groups = {}
    header = None
    metric_idx = value_idx = unit_idx = None
    group_indices = []
    wide_metric_indices = {}
    wide_group_indices = []
    wide_units = {}
    expect_wide_units = False
    wanted_metrics = set(DEFAULT_NCU_METRICS)
    no_kernels_profiled = False
    child_process_warning = False

    with open(path, "r", newline="", encoding="utf-8", errors="ignore") as file:
        reader = csv.reader(file)
        for row_number, row in enumerate(reader):
            if not row:
                continue

            lowered = [cell.strip().lower() for cell in row]
            row_text = " ".join(lowered)
            if "no kernels were profiled" in row_text:
                no_kernels_profiled = True
            if "target-processes all" in row_text:
                child_process_warning = True

            wide_lookup = header_lookup(row)
            row_wide_metric_indices = {
                metric_name: wide_lookup[metric_name.lower()]
                for metric_name in wanted_metrics
                if metric_name.lower() in wide_lookup
            }
            if row_wide_metric_indices:
                wide_metric_indices = row_wide_metric_indices
                wide_group_indices = [
                    idx
                    for idx in [
                        get_index(wide_lookup, "id"),
                        get_index(wide_lookup, "kernel name"),
                        get_index(wide_lookup, "context"),
                        get_index(wide_lookup, "stream"),
                        get_index(wide_lookup, "process id"),
                    ]
                    if idx is not None
                ]
                wide_units = {}
                expect_wide_units = True
                continue

            if "metric name" in lowered and (
                "metric value" in lowered or "value" in lowered
            ):
                header = row
                lookup = header_lookup(header)
                metric_idx = get_index(lookup, "metric name", "name")
                value_idx = get_index(lookup, "metric value", "value")
                unit_idx = get_index(lookup, "metric unit", "unit")
                group_indices = [
                    idx
                    for idx in [
                        get_index(lookup, "id"),
                        get_index(lookup, "kernel name"),
                        get_index(lookup, "context"),
                        get_index(lookup, "stream"),
                        get_index(lookup, "process id"),
                    ]
                    if idx is not None
                ]
                continue

            if wide_metric_indices:
                if expect_wide_units:
                    contains_metric_values = any(
                        idx < len(row) and parse_float(row[idx]) is not None
                        for idx in wide_metric_indices.values()
                    )
                    if not contains_metric_values:
                        wide_units = {
                            metric_name: row[idx].strip() if idx < len(row) else ""
                            for metric_name, idx in wide_metric_indices.items()
                        }
                        expect_wide_units = False
                        continue
                    expect_wide_units = False

                if len(row) <= max(wide_metric_indices.values()):
                    continue
                if wide_group_indices:
                    key = tuple(
                        row[idx].strip() if idx < len(row) else ""
                        for idx in wide_group_indices
                    )
                else:
                    key = (str(row_number),)
                for metric_name, idx in wide_metric_indices.items():
                    value = parse_float(row[idx])
                    if value is None:
                        continue
                    groups.setdefault(key, {})[metric_name] = {
                        "value": value,
                        "unit": wide_units.get(metric_name, ""),
                    }
                continue

            if header is None or metric_idx is None or value_idx is None:
                continue
            if len(row) <= max(metric_idx, value_idx):
                continue

            metric_name = row[metric_idx].strip()
            if metric_name not in wanted_metrics:
                continue

            value = parse_float(row[value_idx])
            if value is None:
                continue
            unit = row[unit_idx].strip() if unit_idx is not None and unit_idx < len(row) else ""

            if group_indices:
                key = tuple(row[idx].strip() if idx < len(row) else "" for idx in group_indices)
            else:
                key = (str(row_number),)
            groups.setdefault(key, {})[metric_name] = {"value": value, "unit": unit}

    if not groups:
        if no_kernels_profiled:
            notes = [
                "ncu_no_kernels_profiled",
                "check_nvtx_filter_and_ncu_target_processes",
            ]
            if child_process_warning:
                notes.append("ncu_child_process_warning")
            return {}, "no_kernels_profiled", notes
        return {}, "no_matching_ranges", ["ncu_no_matching_ranges"]

    parsed = {}
    notes = []
    duration_seconds_by_group = {}
    for key, group_values in groups.items():
        duration = group_values.get(NCU_DURATION_METRIC)
        if duration is None:
            continue
        duration_seconds = normalize_duration_seconds(
            duration.get("value"), duration.get("unit")
        )
        if duration_seconds is None:
            notes.append(f"ncu_duration_unit_unrecognized={duration.get('unit') or 'blank'}")
            continue
        if duration_seconds > 0:
            duration_seconds_by_group[key] = duration_seconds

    total_dram_bytes = 0.0
    total_duration_seconds = 0.0
    saw_dram_bytes = False
    for key, group_values in groups.items():
        dram_bytes = group_values.get(NCU_DRAM_BYTES_METRIC)
        duration_seconds = duration_seconds_by_group.get(key)
        if dram_bytes is None:
            continue
        saw_dram_bytes = True
        bytes_value = normalize_bytes(dram_bytes.get("value"), dram_bytes.get("unit"))
        if bytes_value is None:
            notes.append(f"ncu_dram_unit_unrecognized={dram_bytes.get('unit') or 'blank'}")
            continue
        if duration_seconds is None:
            continue
        total_dram_bytes += bytes_value
        total_duration_seconds += duration_seconds
    if total_duration_seconds > 0:
        parsed["dram_bw_gb_s"] = (total_dram_bytes / 1e9) / total_duration_seconds
    elif saw_dram_bytes:
        notes.append("ncu_duration_missing_for_dram_bw")

    for metric_name, column_name in NCU_PERCENT_METRIC_TO_COLUMN.items():
        weighted_sum = 0.0
        weight_sum = 0.0
        simple_values = []
        for key, group_values in groups.items():
            value = group_values.get(metric_name)
            if value is None:
                continue
            metric_value = value.get("value")
            duration_seconds = duration_seconds_by_group.get(key)
            if duration_seconds is not None and duration_seconds > 0:
                weighted_sum += metric_value * duration_seconds
                weight_sum += duration_seconds
            else:
                simple_values.append(metric_value)
        if weight_sum > 0:
            parsed[column_name] = weighted_sum / weight_sum
            if simple_values:
                notes.append(f"{column_name}_partial_duration_weighting")
        elif simple_values:
            parsed[column_name] = sum(simple_values) / len(simple_values)
            notes.append(f"{column_name}_simple_average_no_duration")

    if not parsed:
        return {}, "no_matching_ranges", notes + ["ncu_no_supported_metrics"]
    return parsed, "ok", sorted(set(notes))


def build_rows(args, attempts):
    rows = []
    timing_by_batch = {}
    ncu_profile_by_batch = {}

    for batch_size in args.batch_sizes:
        metrics_list = collect_timing_metrics(args.results_root, args.case_name, batch_size)
        timing_attempts = [
            attempt
            for attempt in attempts
            if attempt.get("batch_size") == str(batch_size)
            and attempt.get("run_type") == "timing"
        ]
        ncu_attempts = [
            attempt
            for attempt in attempts
            if attempt.get("batch_size") == str(batch_size)
            and attempt.get("run_type") == "ncu"
        ]

        row = {column: None for column in CSV_COLUMNS}
        row["batch_size"] = batch_size
        row["ncu_status"] = "skipped"
        notes = []

        if metrics_list:
            averaged = average_timing_metrics(metrics_list)
            row.update(
                {
                    "total_sim_lbs_ms": averaged.get("total_sim_lbs_ms"),
                    "simulation_ms": averaged.get("simulation_ms"),
                    "lbs_ms": averaged.get("lbs_ms"),
                    "throughput_instances_per_sec": averaged.get(
                        "throughput_instances_per_sec"
                    ),
                    "per_instance_ms": averaged.get("per_instance_ms"),
                    "peak_allocated_gb": averaged.get("peak_allocated_gb"),
                    "peak_reserved_gb": averaged.get("peak_reserved_gb"),
                    "gpu_total_memory_gb": averaged.get("gpu_total_memory_gb"),
                    "status": "success",
                }
            )
            timing_by_batch[batch_size] = {
                **averaged,
                "successful_runs": len(metrics_list),
            }
            failed_timing = sum(
                1
                for attempt in timing_attempts
                if attempt.get("status") not in SUCCESSFUL_ATTEMPT_STATUSES
            )
            if failed_timing:
                notes.append(f"timing_failures={failed_timing}")
        else:
            oom = any(is_oom_log(attempt.get("log_path")) for attempt in timing_attempts)
            row["status"] = "oom" if oom else "failed"
            notes.append("no_successful_timing_run")

        profile_metrics = collect_ncu_profile_metrics(
            args.results_root, args.case_name, batch_size
        )
        if profile_metrics is not None:
            ncu_profile_by_batch[batch_size] = profile_metrics
            row["loop_peak_allocated_gb"] = parse_float(
                profile_metrics.get("loop_peak_allocated_gb")
            )
            row["loop_peak_reserved_gb"] = parse_float(
                profile_metrics.get("loop_peak_reserved_gb")
            )

        if row["status"] == "success":
            ncu_csv = Path(args.results_root) / "ncu" / f"batch_{batch_size}.csv"
            if not ncu_attempts:
                if ncu_csv.is_file():
                    ncu_metrics, ncu_status, ncu_notes = parse_ncu_csv(str(ncu_csv))
                    if ncu_status == "ok":
                        row.update(ncu_metrics)
                        row["ncu_status"] = "ok"
                    else:
                        row["ncu_status"] = ncu_status
                    notes.append("ncu_attempt_manifest_missing")
                    notes.extend(ncu_notes)
                else:
                    row["ncu_status"] = "not_run"
                    notes.append("ncu_not_run")
            else:
                latest_ncu_attempt = ncu_attempts[-1]
                attempt_status = latest_ncu_attempt.get("status") or "failed"
                if attempt_status in SUCCESSFUL_ATTEMPT_STATUSES:
                    ncu_metrics, ncu_status, ncu_notes = parse_ncu_csv(str(ncu_csv))
                    if ncu_status == "ok":
                        row.update(ncu_metrics)
                        row["ncu_status"] = "ok"
                    else:
                        row["ncu_status"] = ncu_status
                    notes.extend(ncu_notes)
                elif attempt_status == "unavailable":
                    row["ncu_status"] = "unavailable"
                    notes.append("ncu_unavailable")
                else:
                    row["ncu_status"] = "failed"
                    notes.append(f"ncu_attempt_status={attempt_status}")

            if profile_metrics is None and row["ncu_status"] in {
                "ok",
                "failed",
                "no_matching_ranges",
                "no_kernels_profiled",
            }:
                notes.append("ncu_profile_metrics_missing")

        row["notes"] = "; ".join(notes)
        rows.append(row)

    b1_throughput = None
    for row in rows:
        if row["batch_size"] == 1 and row["status"] == "success":
            b1_throughput = row.get("throughput_instances_per_sec")
            break

    if b1_throughput and b1_throughput > 0:
        for row in rows:
            throughput = row.get("throughput_instances_per_sec")
            batch_size = row["batch_size"]
            if row["status"] == "success" and throughput is not None:
                row["scaling_efficiency_vs_b1"] = throughput / (
                    batch_size * b1_throughput
                )

    annotate_bottlenecks(rows)
    return rows, timing_by_batch, ncu_profile_by_batch


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in CSV_COLUMNS})


def run_text_command(command):
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError:
        return None
    output = completed.stdout.strip()
    return output or None


def write_metadata(args, rows, timing_by_batch, ncu_profile_by_batch):
    successful_rows = [row for row in rows if row["status"] == "success"]
    first_batch = successful_rows[0]["batch_size"] if successful_rows else None
    first_metrics = None
    if first_batch is not None:
        metrics = collect_timing_metrics(args.results_root, args.case_name, first_batch)
        first_metrics = metrics[0] if metrics else None

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    case_data_path = os.path.join(args.base_path, args.case_name, "final_data.pkl")
    gaussian_point_cloud_path = os.path.join(
        args.gaussian_path,
        args.case_name,
        exp_name,
        "point_cloud",
        "iteration_10000",
        "point_cloud.ply",
    )

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_name": args.case_name,
        "case_paths": {
            "base_path": args.base_path,
            "case_data_path": case_data_path,
            "gaussian_path": args.gaussian_path,
            "gaussian_point_cloud_path": gaussian_point_cloud_path,
            "bg_img_path": args.bg_img_path,
            "best_model_path": first_metrics.get("best_model_path")
            if first_metrics
            else None,
        },
        "gpu_name": first_metrics.get("gpu_name") if first_metrics else None,
        "gpu_total_memory_gb": first_metrics.get("gpu_total_memory_gb")
        if first_metrics
        else None,
        "cuda_version": first_metrics.get("cuda_version") if first_metrics else None,
        "pytorch_version": first_metrics.get("pytorch_version") if first_metrics else None,
        "ncu_bin": args.ncu_bin or None,
        "ncu_version": run_text_command([args.ncu_bin, "--version"])
        if args.ncu_bin
        else None,
        "ncu_profile_frame_stride": args.ncu_profile_frame_stride,
        "ncu_profile_max_frames": args.ncu_profile_max_frames,
        "ncu_profile_frame_selection": "stride"
        if args.ncu_profile_frame_stride is not None
        else "evenly_spaced",
        "ncu_profile_nvtx_name": args.ncu_profile_nvtx_name,
        "ncu_profiled_frame_indices": {
            str(batch_size): ncu_profile_by_batch.get(batch_size, {}).get(
                "ncu_profiled_frame_indices"
            )
            for batch_size in args.batch_sizes
        },
        "ncu_num_profiled_frames": {
            str(batch_size): ncu_profile_by_batch.get(batch_size, {}).get(
                "ncu_num_profiled_frames"
            )
            for batch_size in args.batch_sizes
        },
        "ncu_target_processes": args.ncu_target_processes,
        "ncu_metrics": split_ncu_metrics(args.ncu_metrics),
        "warmup_iterations": 2,
        "measured_iterations": {
            str(batch_size): timing_by_batch.get(batch_size, {}).get(
                "measured_iterations"
            )
            for batch_size in args.batch_sizes
        },
        "batch_sizes": args.batch_sizes,
    }

    metadata_path = Path(args.results_root) / "batch_scaling_sim_lbs_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def write_profile_commands(args):
    commands_path = Path(args.results_root) / "batch_scaling_sim_lbs_profile_commands.sh"
    ncu_bin = args.ncu_bin or "ncu"
    ncu_metrics = args.ncu_metrics or ",".join(DEFAULT_NCU_METRICS)
    ncu_target_processes = args.ncu_target_processes or "application-only"
    ncu_nvtx_name = args.ncu_profile_nvtx_name or "sim_lbs_profile_frame"
    with open(commands_path, "w", encoding="utf-8") as file:
        file.write("#!/usr/bin/env bash\n")
        file.write("set -u\n\n")
        for batch_size in args.batch_sizes:
            output_dir = (
                f"{args.results_root}/manual_ncu_runs/batch_{batch_size}/{args.case_name}"
            )
            command = [
                ncu_bin,
                "--target-processes",
                ncu_target_processes,
                "--profile-from-start",
                "off",
                "--nvtx",
                "--nvtx-include",
                ncu_nvtx_name,
                "--csv",
                "--page",
                "raw",
                "--print-units",
                "base",
                "--print-fp",
                "--metrics",
                ncu_metrics,
                "--log-file",
                f"{args.results_root}/ncu/manual_batch_{batch_size}.csv",
                "python",
                "benchmarks/run_sim_lbs_batch_scaling_case.py",
                "--base_path",
                args.base_path,
                "--gaussian_path",
                args.gaussian_path,
                "--bg_img_path",
                args.bg_img_path,
                "--case_name",
                args.case_name,
                "--batch_size",
                str(batch_size),
                "--output_dir",
                output_dir,
                "--scaling-analysis",
                "--ncu-profile-loop",
                "--ncu-profile-max-frames",
                str(args.ncu_profile_max_frames),
                "--ncu-profile-nvtx-name",
                ncu_nvtx_name,
            ]
            if args.ncu_profile_frame_stride is not None:
                command.extend(
                    [
                        "--ncu-profile-frame-stride",
                        str(args.ncu_profile_frame_stride),
                    ]
                )
            file.write(" ".join(shlex.quote(part) for part in command) + "\n")
    os.chmod(commands_path, 0o755)


def write_figure(args, rows):
    successful_rows = [row for row in rows if row["status"] == "success"]
    if not successful_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping scaling figure generation: {exc}")
        return

    labels = [str(row["batch_size"]) for row in successful_rows]
    simulation = [row.get("simulation_ms") or 0.0 for row in successful_rows]
    lbs = [row.get("lbs_ms") or 0.0 for row in successful_rows]
    throughput = [
        row.get("throughput_instances_per_sec") or 0.0 for row in successful_rows
    ]
    x_positions = list(range(len(successful_rows)))

    fig, ax_latency = plt.subplots(figsize=(5.4, 2.8))
    ax_latency.bar(x_positions, simulation, label="Simulation", color="#4C78A8")
    ax_latency.bar(
        x_positions,
        lbs,
        bottom=simulation,
        label="LBS",
        color="#F58518",
    )
    ax_latency.set_xlabel("Batch size")
    ax_latency.set_ylabel("Latency per step (ms)")
    ax_latency.set_xticks(x_positions)
    ax_latency.set_xticklabels(labels)
    ax_latency.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)

    ax_throughput = ax_latency.twinx()
    ax_throughput.plot(
        x_positions,
        throughput,
        color="#54A24B",
        marker="o",
        linewidth=1.8,
        label="Throughput",
    )
    ax_throughput.set_ylabel("Throughput (instances/s)")

    handles_a, labels_a = ax_latency.get_legend_handles_labels()
    handles_b, labels_b = ax_throughput.get_legend_handles_labels()
    ax_latency.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()

    output_base = Path(args.results_root) / "batch_scaling_sim_lbs"
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def bottleneck_evidence_label(row):
    status = row.get("status")
    sm = row.get("sm_util_pct")
    dram = row.get("dram_bw_pct_peak")
    loop_mem = row.get("loop_peak_reserved_gb")
    peak_mem = row.get("peak_reserved_gb")
    total_mem = row.get("gpu_total_memory_gb")
    ncu_status = row.get("ncu_status")

    memory_value = loop_mem if loop_mem is not None else peak_mem
    parts = []
    if status and status != "success":
        parts.append(status)
    if sm is not None:
        parts.append(f"SM {sm:.0f}%")
    if dram is not None:
        parts.append(f"DRAM {dram:.0f}%")
    if memory_value is not None and total_mem is not None and total_mem > 0:
        parts.append(f"mem {memory_value:.1f}/{total_mem:.0f}G")
    elif memory_value is not None:
        parts.append(f"mem {memory_value:.1f}G")
    if ncu_status:
        parts.append(f"NCU {ncu_status}")
    return "\n".join(parts[:5])


def write_bottleneck_figure(args, rows):
    plot_rows = [row for row in rows if row.get("batch_size") is not None]
    if not plot_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable; skipping bottleneck figure generation: {exc}")
        return

    def finite_or_nan(value):
        return float(value) if value is not None else math.nan

    labels = [str(row["batch_size"]) for row in plot_rows]
    x_positions = list(range(len(plot_rows)))
    throughput = [
        finite_or_nan(row.get("throughput_instances_per_sec"))
        if row.get("status") == "success"
        else math.nan
        for row in plot_rows
    ]
    latency = [
        finite_or_nan(row.get("per_instance_ms"))
        if row.get("status") == "success"
        else math.nan
        for row in plot_rows
    ]

    if not any(math.isfinite(value) for value in throughput + latency):
        return

    fig, ax_throughput = plt.subplots(figsize=(5.8, 2.55))
    ax_throughput.plot(
        x_positions,
        throughput,
        color="#2F6F73",
        linewidth=1.8,
        zorder=1,
        label="Throughput",
    )

    for x_value, y_value, row in zip(x_positions, throughput, plot_rows):
        bottleneck_class = row.get("bottleneck_class") or "unknown"
        color = BOTTLENECK_COLORS.get(bottleneck_class, BOTTLENECK_COLORS["unknown"])
        label = bottleneck_evidence_label(row)
        if math.isfinite(y_value):
            ax_throughput.scatter(
                [x_value],
                [y_value],
                color=color,
                edgecolor="white",
                linewidth=0.8,
                s=42,
                zorder=3,
            )
            annotation_y = y_value
        else:
            ax_throughput.scatter(
                [x_value],
                [0.0],
                color=color,
                marker="x",
                linewidth=1.2,
                s=44,
                zorder=3,
            )
            annotation_y = 0.0
        if label:
            ax_throughput.annotate(
                label,
                (x_value, annotation_y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                va="bottom",
                fontsize=5.6,
                color="#333333",
            )

    ax_throughput.set_xlabel("Batch size")
    ax_throughput.set_ylabel("Throughput (instances/s)")
    ax_throughput.set_xticks(x_positions)
    ax_throughput.set_xticklabels(labels)
    ax_throughput.set_ylim(bottom=0)
    ax_throughput.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)

    ax_latency = ax_throughput.twinx()
    ax_latency.plot(
        x_positions,
        latency,
        color="#5F5F5F",
        linestyle="--",
        linewidth=1.5,
        zorder=2,
        label="Per-instance latency",
    )
    ax_latency.set_ylabel("Per-instance latency (ms)")

    present_classes = [
        class_name
        for class_name in BOTTLENECK_COLORS
        if any((row.get("bottleneck_class") or "unknown") == class_name for row in plot_rows)
    ]
    legend_handles = [
        Line2D([0], [0], color="#2F6F73", linewidth=1.8, label="Throughput"),
        Line2D(
            [0],
            [0],
            color="#5F5F5F",
            linestyle="--",
            linewidth=1.5,
            label="Latency/instance",
        ),
    ]
    legend_handles.extend(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=BOTTLENECK_COLORS[class_name],
            markeredgecolor="white",
            markersize=6,
            label=BOTTLENECK_LABELS[class_name],
        )
        for class_name in present_classes
    )
    ax_throughput.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    fig.tight_layout()

    output_base = Path(args.results_root) / "batch_scaling_sim_lbs_bottleneck"
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_base}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def classify_scaling(rows):
    successful = [row for row in rows if row["status"] == "success"]
    if not successful:
        return "No successful batch sizes were measured."
    efficiencies = [
        row.get("scaling_efficiency_vs_b1")
        for row in successful
        if row.get("scaling_efficiency_vs_b1") is not None
    ]
    if not efficiencies:
        return "B=1 was not successfully measured, so efficiency versus ideal B=1 scaling is unavailable."
    return (
        "Scaling efficiency versus ideal B=1 scaling ranged from "
        f"{min(efficiencies):.3f} to {max(efficiencies):.3f}."
    )


def write_summary(args, rows):
    successful = [row for row in rows if row["status"] == "success"]
    lines = [
        "Sim+LBS batch scaling summary",
        f"Case: {args.case_name}",
        "Latency and throughput come from the normal timing run, not the NCU run.",
    ]

    if not successful:
        lines.append("No batch size completed successfully.")
    else:
        best = max(successful, key=lambda row: row.get("throughput_instances_per_sec") or 0.0)
        lines.append(
            "Best throughput: "
            f"{best.get('throughput_instances_per_sec'):.3f} instances/s at B={best['batch_size']}."
        )
        lines.append(classify_scaling(rows))

        first = successful[0]
        last = successful[-1]
        lines.append(
            "Per-instance latency changed from "
            f"{first.get('per_instance_ms'):.6f} ms at B={first['batch_size']} to "
            f"{last.get('per_instance_ms'):.6f} ms at B={last['batch_size']}."
        )

        mem_first = first.get("peak_reserved_gb")
        mem_last = last.get("peak_reserved_gb")
        if mem_first is not None and mem_last is not None:
            lines.append(
                "PyTorch peak reserved memory changed from "
                f"{mem_first:.3f} GB to {mem_last:.3f} GB across successful batch sizes."
            )

        largest = last
        sm = largest.get("sm_util_pct")
        dram = largest.get("dram_bw_pct_peak")
        dram_gb_s = largest.get("dram_bw_gb_s")
        occ = largest.get("achieved_occupancy_pct")
        if sm is not None or dram is not None or dram_gb_s is not None or occ is not None:
            counter_bits = []
            if sm is not None:
                counter_bits.append(f"SM throughput {sm:.1f}%")
            if dram_gb_s is not None:
                counter_bits.append(f"DRAM bandwidth {dram_gb_s:.1f} GB/s")
            if dram is not None:
                counter_bits.append(f"DRAM bandwidth {dram:.1f}% of peak")
            if occ is not None:
                counter_bits.append(f"achieved occupancy {occ:.1f}%")
            lines.append(
                f"At the largest successful batch size (B={largest['batch_size']}), "
                + ", ".join(counter_bits)
                + "."
            )
            lines.append(
                "These counters are reported descriptively and are not used as standalone proof of a bottleneck."
            )
        else:
            lines.append("NCU hardware counters were not available in the final CSV.")

        known_bottlenecks = [
            row
            for row in successful
            if row.get("bottleneck_class")
            and row.get("bottleneck_class") != "unknown"
        ]
        if known_bottlenecks:
            class_counts = Counter(row["bottleneck_class"] for row in known_bottlenecks)
            dominant_class, dominant_count = class_counts.most_common(1)[0]
            lines.append(
                "The heuristic labels suggest "
                f"{dominant_class} in {dominant_count}/{len(successful)} successful batch sizes."
            )
            largest_class = largest.get("bottleneck_class")
            largest_reason = largest.get("bottleneck_reason")
            if largest_class:
                lines.append(
                    "At the largest successful batch size, the evidence likely suggests "
                    f"{largest_class}: {largest_reason}."
                )
        else:
            lines.append(
                "The heuristic pressure label is unknown for successful batch sizes because "
                "NCU evidence is missing or mixed."
            )

    failed = [row for row in rows if row["status"] != "success"]
    if failed:
        failed_desc = ", ".join(f"B={row['batch_size']}:{row['status']}" for row in failed)
        lines.append(f"Failed batch sizes: {failed_desc}.")

    summary_path = Path(args.results_root) / "batch_scaling_sim_lbs_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.results_root, exist_ok=True)
    attempts = read_attempts(args.results_root)
    rows, timing_by_batch, ncu_profile_by_batch = build_rows(args, attempts)
    write_csv(Path(args.results_root) / "batch_scaling_sim_lbs.csv", rows)
    write_metadata(args, rows, timing_by_batch, ncu_profile_by_batch)
    write_profile_commands(args)
    write_figure(args, rows)
    write_bottleneck_figure(args, rows)
    write_summary(args, rows)


if __name__ == "__main__":
    main()
