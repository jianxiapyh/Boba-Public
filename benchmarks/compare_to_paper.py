import argparse
import csv
from pathlib import Path


PAPER_VALUES = {
    "train": {
        "CD": 0.005,
        "Track Error": 0.009,
        "IoU %": 84.4,
        "PSNR": 28.214,
        "SSIM": 0.945,
        "LPIPS": 0.034,
    },
    "test": {
        "CD": 0.012,
        "Track Error": 0.022,
        "IoU %": 72.5,
        "PSNR": 25.617,
        "SSIM": 0.941,
        "LPIPS": 0.055,
    },
}

MISMATCH_THRESHOLDS = {
    "CD": 0.003,
    "Track Error": 0.005,
    "IoU %": 3.0,
    "PSNR": 1.0,
    "SSIM": 0.01,
    "LPIPS": 0.02,
}


def read_overall_row(csv_path, key_name):
    with open(csv_path, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        if row.get(key_name) == "OVERALL":
            return row

    raise ValueError(f"Missing OVERALL row in {csv_path}")


def load_actual_values(metrics_dir):
    chamfer = read_overall_row(metrics_dir / "chamfer.csv", "Case Name")
    track = read_overall_row(metrics_dir / "track.csv", "Case Name")
    render = read_overall_row(metrics_dir / "render_metrics.csv", "scene")

    return {
        "train": {
            "CD": float(chamfer["Train Chamfer Error"]),
            "Track Error": float(track["Train Track Error"]),
            "IoU %": float(render["iou_train"]) * 100.0,
            "PSNR": float(render["psnr_train"]),
            "SSIM": float(render["ssim_train"]),
            "LPIPS": float(render["lpips_train"]),
        },
        "test": {
            "CD": float(chamfer["Test Chamfer Error"]),
            "Track Error": float(track["Test Track Error"]),
            "IoU %": float(render["iou_test"]) * 100.0,
            "PSNR": float(render["psnr_test"]),
            "SSIM": float(render["ssim_test"]),
            "LPIPS": float(render["lpips_test"]),
        },
    }


def build_rows(actual_values):
    rows = []
    mismatch_found = False
    for task in ("train", "test"):
        for metric_name, paper_value in PAPER_VALUES[task].items():
            measured_value = actual_values[task][metric_name]
            delta = measured_value - paper_value
            abs_delta = abs(delta)
            threshold = MISMATCH_THRESHOLDS[metric_name]
            close = abs_delta <= threshold
            mismatch_found = mismatch_found or not close
            rows.append(
                {
                    "task": task,
                    "paper_task_name": (
                        "Reconstruction & Resimulation"
                        if task == "train"
                        else "Future Prediction"
                    ),
                    "metric": metric_name,
                    "paper_value": paper_value,
                    "measured_value": measured_value,
                    "delta": delta,
                    "abs_delta": abs_delta,
                    "threshold": threshold,
                    "close": "yes" if close else "no",
                }
            )
    return rows, mismatch_found


def write_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task",
                "paper_task_name",
                "metric",
                "paper_value",
                "measured_value",
                "delta",
                "abs_delta",
                "threshold",
                "close",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "paper_value": f"{row['paper_value']:.3f}",
                    "measured_value": f"{row['measured_value']:.3f}",
                    "delta": f"{row['delta']:.3f}",
                    "abs_delta": f"{row['abs_delta']:.3f}",
                    "threshold": f"{row['threshold']:.3f}",
                }
            )


def write_report(rows, output_path, mismatch_found):
    grouped = {"train": [], "test": []}
    for row in rows:
        grouped[row["task"]].append(row)

    lines = ["# PhysTwin Paper Comparison", ""]
    for task in ("train", "test"):
        lines.append(
            f"## {'Reconstruction & Resimulation' if task == 'train' else 'Future Prediction'}"
        )
        lines.append("")
        lines.append("| Metric | Paper | Measured | Delta | Abs Delta | Close |")
        lines.append("| --- | ---: | ---: | ---: | ---: | :---: |")
        for row in grouped[task]:
            lines.append(
                "| {metric} | {paper:.3f} | {measured:.3f} | {delta:.3f} | {abs_delta:.3f} | {close} |".format(
                    metric=row["metric"],
                    paper=row["paper_value"],
                    measured=row["measured_value"],
                    delta=row["delta"],
                    abs_delta=row["abs_delta"],
                    close=row["close"],
                )
            )
        lines.append("")

    if mismatch_found:
        lines.extend(
            [
                "## Investigation Checklist",
                "",
                "- Confirm the train/test split semantics match original PhysTwin and the paper.",
                "- Confirm frame indexing and render export alignment, especially the first frame.",
                "- Confirm `export_render_eval_data.py` and human-mask usage match the original PhysTwin evaluation flow.",
                "- Confirm the comparison is using the intended view count and aggregation semantics.",
                "- Compare one representative case against `PhysTwin_original` outputs to isolate wrapper/output drift from deeper logic issues.",
                "- Do not change spring-mass simulation, LBS, rendering, or frame compositing to chase the paper numbers.",
            ]
        )
    else:
        lines.extend(["## Status", "", "All metrics are within the configured similarity thresholds."])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_dir",
        default="results/quality/metrics",
        help="Directory containing chamfer.csv, track.csv, and render_metrics.csv.",
    )
    parser.add_argument(
        "--csv_output",
        default="results/quality/metrics/paper_comparison.csv",
    )
    parser.add_argument(
        "--report_output",
        default="results/quality/metrics/paper_comparison.md",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    actual_values = load_actual_values(metrics_dir)
    rows, mismatch_found = build_rows(actual_values)
    write_csv(rows, Path(args.csv_output))
    write_report(rows, Path(args.report_output), mismatch_found)
    print(f"Saved paper comparison to {args.csv_output} and {args.report_output}")
    if mismatch_found:
        print("Paper comparison found metrics outside the configured similarity thresholds.")


if __name__ == "__main__":
    main()
