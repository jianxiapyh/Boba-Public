import argparse
import csv
import math
from pathlib import Path


FIXED_BATCHES = [1, 32, 64, 128, 256]
X_LABELS = [str(batch) for batch in FIXED_BATCHES] + ["Max\nthroughput"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the compact Sim+LBS scaling rebuttal figure."
    )
    parser.add_argument(
        "--scaling-csv",
        default="results/sim_lbs_batch_scaling_rebuttal/all_cases_batch_scaling_sim_lbs.csv",
        help="Merged fixed-batch Sim+LBS scaling CSV.",
    )
    parser.add_argument(
        "--best-csv",
        default="results/batch_autotune/best_throughput_table.csv",
        help="Autotune best-throughput CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/sim_lbs_batch_scaling_rebuttal",
        help="Directory where the rebuttal figure is written.",
    )
    parser.add_argument(
        "--output-stem",
        default="sim_lbs_rebuttal_scaling_figure",
        help="Output filename stem for PDF and PNG.",
    )
    parser.add_argument(
        "--throughput-stat",
        choices=("median", "mean"),
        default="median",
        help="Aggregate statistic for throughput. GPU counter lines remain medians.",
    )
    return parser.parse_args()


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "n/a", "na", "--"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def percentile(values, q):
    valid = sorted(value for value in values if value is not None and math.isfinite(value))
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    position = (len(valid) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return valid[int(position)]
    weight = position - lower
    return valid[lower] * (1.0 - weight) + valid[upper] * weight


def summarize(values):
    median = percentile(values, 0.5)
    if median is None:
        return {"median": None, "q1": None, "q3": None, "count": 0}
    return {
        "median": median,
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "count": len([value for value in values if value is not None and math.isfinite(value)]),
    }


def summarize_mean(values):
    valid = [value for value in values if value is not None and math.isfinite(value)]
    if not valid:
        return {"median": None, "q1": None, "q3": None, "count": 0}
    mean_value = sum(valid) / len(valid)
    return {
        "median": mean_value,
        "q1": mean_value,
        "q3": mean_value,
        "count": len(valid),
    }


def yerr_from_stats(stats):
    medians = []
    lower = []
    upper = []
    for item in stats:
        median = item["median"]
        medians.append(math.nan if median is None else median)
        if median is None:
            lower.append(0.0)
            upper.append(0.0)
        else:
            lower.append(max(0.0, median - item["q1"]))
            upper.append(max(0.0, item["q3"] - median))
    return medians, [lower, upper]


def build_fixed_batch_stats(rows, throughput_stat):
    all_cases = {row.get("case_name") for row in rows if row.get("case_name")}
    throughput_stats = []
    counter_stats = {
        "dram_bw_pct_peak": [],
        "sm_util_pct": [],
        "achieved_occupancy_pct": [],
    }

    fixed_success_cases_by_batch = {}
    for batch_size in FIXED_BATCHES:
        batch_rows = [
            row
            for row in rows
            if parse_float(row.get("batch_size")) == float(batch_size)
        ]
        throughput_by_case = {case_name: 0.0 for case_name in all_cases}
        for row in batch_rows:
            case_name = row.get("case_name")
            if not case_name or row.get("status") != "success":
                continue
            throughput = parse_float(row.get("throughput_instances_per_sec"))
            if throughput is not None:
                throughput_by_case[case_name] = throughput
        successful_rows = [row for row in batch_rows if row.get("status") == "success"]
        fixed_success_cases_by_batch[batch_size] = set(throughput_by_case)
        throughput_values = list(throughput_by_case.values())
        if throughput_stat == "mean":
            throughput_stats.append(summarize_mean(throughput_values))
        else:
            throughput_stats.append(summarize(throughput_values))

        ncu_rows = [
            row
            for row in successful_rows
            if row.get("ncu_status") == "ok"
        ]
        for column in counter_stats:
            counter_stats[column].append(
                summarize([parse_float(row.get(column)) for row in ncu_rows])
            )

    max_reference_cases = fixed_success_cases_by_batch.get(FIXED_BATCHES[-1], set())
    return throughput_stats, counter_stats, max_reference_cases


def build_max_stats(rows, reference_cases, throughput_stat):
    if reference_cases:
        rows = [row for row in rows if row.get("Case Name") in reference_cases]
    values = [
        parse_float(row.get("Best Average Throughput (instances/s)"))
        for row in rows
    ]
    if throughput_stat == "mean":
        return summarize_mean(values)
    return summarize(values)


def write_figure(args):
    scaling_rows = read_csv(args.scaling_csv)
    best_rows = read_csv(args.best_csv)
    fixed_throughput_stats, counter_stats, max_reference_cases = build_fixed_batch_stats(
        scaling_rows, args.throughput_stat
    )
    max_throughput_stats = build_max_stats(
        best_rows, max_reference_cases, args.throughput_stat
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required to write the figure: {exc}") from exc

    x_fixed = list(range(len(FIXED_BATCHES)))
    x_all = list(range(len(X_LABELS)))
    x_max = [len(X_LABELS) - 1]

    fixed_medians, _ = yerr_from_stats(fixed_throughput_stats)
    max_medians, _ = yerr_from_stats([max_throughput_stats])

    fig, ax_throughput = plt.subplots(figsize=(5.4, 2.08))

    throughput_color = "#2F6F73"
    ax_throughput.plot(
        x_fixed,
        fixed_medians,
        color=throughput_color,
        marker="o",
        markersize=4.8,
        linewidth=1.8,
        label=f"{args.throughput_stat.capitalize()}\nThroughput",
    )
    ax_throughput.plot(
        x_max,
        max_medians,
        color=throughput_color,
        marker="s",
        markersize=5.0,
        linewidth=0,
    )
    ax_throughput.set_ylabel("Throughput (inst./s)")
    ax_throughput.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax_throughput.set_axisbelow(True)
    ax_throughput.margins(y=0.16)

    metric_specs = [
        ("dram_bw_pct_peak", "DRAM BW\n(% peak)", "#B45F37"),
        ("sm_util_pct", "SM util.\n(% peak)", "#7B61A8"),
        ("achieved_occupancy_pct", "Occupancy\n(% active)", "#54A24B"),
    ]
    ax_counters = ax_throughput.twinx()
    for column, label, color in metric_specs:
        medians, _ = yerr_from_stats(counter_stats[column])
        ax_counters.plot(
            x_fixed,
            medians,
            color=color,
            marker="o",
            markersize=4.2,
            linewidth=1.6,
            label=label,
        )

    ax_counters.set_ylabel("GPU metrics (%)")
    ax_counters.set_ylim(bottom=0)

    handles_a, labels_a = ax_throughput.get_legend_handles_labels()
    handles_b, labels_b = ax_counters.get_legend_handles_labels()
    ax_throughput.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        loc="center",
        bbox_to_anchor=(0.90, 0.39),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.86,
        fontsize=6.8,
        handlelength=1.15,
        handletextpad=0.28,
        labelspacing=0.33,
    )

    ax_throughput.set_xticks(x_all)
    ax_throughput.set_xticklabels(X_LABELS)
    ax_throughput.set_xlabel("Batch size", labelpad=-6.0)
    ax_throughput.set_xlim(-0.35, len(X_LABELS) - 0.08)
    for axis in (ax_throughput, ax_counters):
        axis.spines["top"].set_visible(True)
        axis.tick_params(axis="both", labelsize=7.6)
    ax_throughput.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.45)
    fig.subplots_adjust(bottom=0.21)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / args.output_stem
    fig.savefig(f"{output_base}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[DONE] Wrote {output_base}.pdf")
    print(f"[DONE] Wrote {output_base}.png")


def main():
    args = parse_args()
    write_figure(args)


if __name__ == "__main__":
    main()
