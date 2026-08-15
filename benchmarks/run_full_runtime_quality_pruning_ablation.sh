#!/usr/bin/env bash
set -u

CASES_FILE="${CASES_FILE:-data_config.csv}"
NUM_VIEWS=1
OVERALL_MODE=phystwin
KEEP_RATIO=0.3
KEEP_COUNT=""
PRUNE_MODE=opacity_area
GAUSSIAN_PATH="./gaussian_output"
PRUNED_GAUSSIAN_PATH=""
PRUNED_GAUSSIAN_PATH_EXPLICIT=0
RESULT_ROOT=""
RESULT_ROOT_EXPLICIT=0
EXP_NAME="init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
FORCE_PRUNE=0
ENABLE_PRUNING=0
PRUNING_ARG_USED=0
CASE_KEEP_RATIO_CSV="benchmarks/pruning_ratio_policy_30_55.csv"
CASE_KEEP_RATIO_CSV_EXPLICIT=0
NO_CASE_KEEP_RATIO_POLICY=0
POLICY_NAME="30_55"

read_cases_file() {
  mapfile -t cases < <(awk -F, 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); if ($1 != "") print $1}' "$1")
}

usage() {
  cat <<'EOF'
Usage: bash benchmarks/run_full_runtime_quality_pruning_ablation.sh [options] [case ...]

By default this delegates to the Boba Local single-instance quality evaluator.
Pass --enable_pruning to generate/reuse pruned PLY assets and run the
baseline-vs-pruned comparison workflow.

Common options:
  --gaussian_path P          Baseline Gaussian root (default: ./gaussian_output)
  --result_root P            Result root. Non-pruning default is inherited from
                             Boba_Local_single_inst_quality.sh; pruning default is
                             results/quality_pruning_ablation
  --num_views N              Number of views for quality evaluation (default: 1)
  --overall_mode M           scene_mean or phystwin (default: phystwin)

Pruning options:
  --enable_pruning           Generate/reuse pruned PLY assets and compare
                             baseline vs pruned quality
  --keep_ratio R             Fallback fraction of Gaussians to keep (default: 0.3)
  --keep_count N             Number of Gaussians to keep per case instead of ratio
  --prune_mode M             opacity, opacity_area, or opacity_volume (default: opacity_area)
  --pruned_gaussian_path P   Pruned Gaussian root (default: derived from policy/ratio/count)
  --case_keep_ratio_csv P    CSV with case_name,keep_ratio overrides
                             (default: benchmarks/pruning_ratio_policy_30_55.csv)
  --no_case_keep_ratio_policy
                             Disable the default mixed-ratio policy
  --policy_name NAME         Name used for mixed-policy output paths (default: 30_55)
  --force_prune              Recreate pruned PLYs even if they already exist
EOF
}

mark_pruning_arg() {
  PRUNING_ARG_USED=1
}

validate_ratio() {
  python - "$1" <<'PY'
import sys
ratio = float(sys.argv[1])
if not (0.0 < ratio <= 1.0):
    raise SystemExit("--keep_ratio values must be in (0, 1]")
PY
}

ratio_label_for() {
  python - "$1" <<'PY'
import sys
ratio = float(sys.argv[1])
if abs(ratio * 100 - round(ratio * 100)) < 1e-6:
    print(f"{int(round(ratio * 100))}pct")
else:
    print(str(ratio).replace(".", "p"))
PY
}

cases=()
while (($# > 0)); do
  case "$1" in
    --enable_pruning|--enable-pruning)
      ENABLE_PRUNING=1
      ;;
    --keep_ratio|--keep-ratio)
      mark_pruning_arg
      shift
      if (($# == 0)); then
        echo "Error: --keep_ratio requires a value." >&2
        usage >&2
        exit 1
      fi
      KEEP_RATIO="$1"
      ;;
    --keep_ratio=*|--keep-ratio=*)
      mark_pruning_arg
      KEEP_RATIO="${1#*=}"
      ;;
    --keep_count|--keep-count)
      mark_pruning_arg
      shift
      if (($# == 0)); then
        echo "Error: --keep_count requires a value." >&2
        usage >&2
        exit 1
      fi
      KEEP_COUNT="$1"
      ;;
    --keep_count=*|--keep-count=*)
      mark_pruning_arg
      KEEP_COUNT="${1#*=}"
      ;;
    --prune_mode|--prune-mode)
      mark_pruning_arg
      shift
      if (($# == 0)); then
        echo "Error: --prune_mode requires a value." >&2
        usage >&2
        exit 1
      fi
      PRUNE_MODE="$1"
      ;;
    --prune_mode=*|--prune-mode=*)
      mark_pruning_arg
      PRUNE_MODE="${1#*=}"
      ;;
    --gaussian_path|--gaussian-path)
      shift
      if (($# == 0)); then
        echo "Error: --gaussian_path requires a value." >&2
        usage >&2
        exit 1
      fi
      GAUSSIAN_PATH="$1"
      ;;
    --gaussian_path=*|--gaussian-path=*)
      GAUSSIAN_PATH="${1#*=}"
      ;;
    --pruned_gaussian_path|--pruned-gaussian-path)
      mark_pruning_arg
      shift
      if (($# == 0)); then
        echo "Error: --pruned_gaussian_path requires a value." >&2
        usage >&2
        exit 1
      fi
      PRUNED_GAUSSIAN_PATH="$1"
      PRUNED_GAUSSIAN_PATH_EXPLICIT=1
      ;;
    --pruned_gaussian_path=*|--pruned-gaussian-path=*)
      mark_pruning_arg
      PRUNED_GAUSSIAN_PATH="${1#*=}"
      PRUNED_GAUSSIAN_PATH_EXPLICIT=1
      ;;
    --case_keep_ratio_csv|--case-keep-ratio-csv)
      mark_pruning_arg
      shift
      if (($# == 0)); then
        echo "Error: --case_keep_ratio_csv requires a value." >&2
        usage >&2
        exit 1
      fi
      CASE_KEEP_RATIO_CSV="$1"
      CASE_KEEP_RATIO_CSV_EXPLICIT=1
      ;;
    --case_keep_ratio_csv=*|--case-keep-ratio-csv=*)
      mark_pruning_arg
      CASE_KEEP_RATIO_CSV="${1#*=}"
      CASE_KEEP_RATIO_CSV_EXPLICIT=1
      ;;
    --no_case_keep_ratio_policy|--no-case-keep-ratio-policy)
      mark_pruning_arg
      NO_CASE_KEEP_RATIO_POLICY=1
      ;;
    --policy_name|--policy-name)
      mark_pruning_arg
      shift
      if (($# == 0)); then
        echo "Error: --policy_name requires a value." >&2
        usage >&2
        exit 1
      fi
      POLICY_NAME="$1"
      ;;
    --policy_name=*|--policy-name=*)
      mark_pruning_arg
      POLICY_NAME="${1#*=}"
      ;;
    --result_root|--result-root)
      shift
      if (($# == 0)); then
        echo "Error: --result_root requires a value." >&2
        usage >&2
        exit 1
      fi
      RESULT_ROOT="$1"
      RESULT_ROOT_EXPLICIT=1
      ;;
    --result_root=*|--result-root=*)
      RESULT_ROOT="${1#*=}"
      RESULT_ROOT_EXPLICIT=1
      ;;
    --num_views|--num-views)
      shift
      if (($# == 0)); then
        echo "Error: --num_views requires a value." >&2
        usage >&2
        exit 1
      fi
      NUM_VIEWS="$1"
      ;;
    --num_views=*|--num-views=*)
      NUM_VIEWS="${1#*=}"
      ;;
    --overall_mode|--overall-mode)
      shift
      if (($# == 0)); then
        echo "Error: --overall_mode requires a value." >&2
        usage >&2
        exit 1
      fi
      OVERALL_MODE="$1"
      ;;
    --overall_mode=*|--overall-mode=*)
      OVERALL_MODE="${1#*=}"
      ;;
    --force_prune|--force-prune)
      mark_pruning_arg
      FORCE_PRUNE=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Error: unknown option '$1'." >&2
      usage >&2
      exit 1
      ;;
    *)
      cases+=("$1")
      ;;
  esac
  shift
done

if ((${#cases[@]} == 0)); then
  read_cases_file "$CASES_FILE"
fi

if ((ENABLE_PRUNING == 0)); then
  if ((PRUNING_ARG_USED == 1)); then
    echo "Error: pruning options require --enable_pruning." >&2
    echo "       Without --enable_pruning, this script delegates to benchmarks/Boba_Local_single_inst_quality.sh." >&2
    exit 1
  fi

  echo "=== [quality] pruning disabled; delegating to Boba_Local_single_inst_quality.sh ==="
  quality_cmd=(bash benchmarks/Boba_Local_single_inst_quality.sh
    --gaussian_variant baseline
    --gaussian_path "$GAUSSIAN_PATH"
    --num_views "$NUM_VIEWS"
    --overall_mode "$OVERALL_MODE")
  if ((RESULT_ROOT_EXPLICIT == 1)); then
    quality_cmd+=(--result_root "$RESULT_ROOT")
  fi
  quality_cmd+=("${cases[@]}")
  "${quality_cmd[@]}"
  exit $?
fi

if [[ -z "$RESULT_ROOT" ]]; then
  RESULT_ROOT="results/quality_pruning_ablation"
fi

if ((NO_CASE_KEEP_RATIO_POLICY == 1 && CASE_KEEP_RATIO_CSV_EXPLICIT == 1)); then
  echo "Error: --case_keep_ratio_csv cannot be combined with --no_case_keep_ratio_policy." >&2
  exit 1
fi

if [[ -n "$KEEP_COUNT" && $NO_CASE_KEEP_RATIO_POLICY -eq 0 ]]; then
  echo "Error: --keep_count cannot be combined with the case keep-ratio policy." >&2
  echo "       Use --no_case_keep_ratio_policy for one global keep_count." >&2
  exit 1
fi

if ((NO_CASE_KEEP_RATIO_POLICY == 1)); then
  CASE_KEEP_RATIO_CSV=""
fi

if [[ "$PRUNE_MODE" != "opacity" && "$PRUNE_MODE" != "opacity_area" && "$PRUNE_MODE" != "opacity_volume" ]]; then
  echo "Error: --prune_mode must be opacity, opacity_area, or opacity_volume. Received: ${PRUNE_MODE}" >&2
  exit 1
fi

if [[ -n "$KEEP_COUNT" ]]; then
  if ! [[ "$KEEP_COUNT" =~ ^[0-9]+$ ]] || ((KEEP_COUNT < 1)); then
    echo "Error: --keep_count must be a positive integer. Received: ${KEEP_COUNT}" >&2
    exit 1
  fi
else
  validate_ratio "$KEEP_RATIO"
fi

declare -A CASE_KEEP_RATIOS=()
if [[ -n "$CASE_KEEP_RATIO_CSV" ]]; then
  if [[ ! -f "$CASE_KEEP_RATIO_CSV" ]]; then
    echo "Error: missing case keep-ratio policy CSV: ${CASE_KEEP_RATIO_CSV}" >&2
    exit 1
  fi
  while IFS=$'\t' read -r policy_case policy_ratio; do
    CASE_KEEP_RATIOS["$policy_case"]="$policy_ratio"
  done < <(python - "$CASE_KEEP_RATIO_CSV" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    required = {"case_name", "keep_ratio"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise SystemExit(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    for row in reader:
        case_name = (row.get("case_name") or "").strip()
        ratio_text = (row.get("keep_ratio") or "").strip()
        if not case_name:
            continue
        ratio = float(ratio_text)
        if not (0.0 < ratio <= 1.0):
            raise SystemExit(f"{path}: keep_ratio for {case_name} must be in (0, 1]")
        print(f"{case_name}\t{ratio_text}")
PY
)
fi

if [[ -n "$KEEP_COUNT" ]]; then
  ablation_name="keep_${KEEP_COUNT}_${PRUNE_MODE}"
else
  ratio_label="$(ratio_label_for "$KEEP_RATIO")"
  if [[ -n "$CASE_KEEP_RATIO_CSV" ]]; then
    ablation_name="keep_policy_${POLICY_NAME}_${PRUNE_MODE}"
  else
    ablation_name="keep_${ratio_label}_${PRUNE_MODE}"
  fi
fi

if [[ -z "$PRUNED_GAUSSIAN_PATH" ]]; then
  if [[ -n "$KEEP_COUNT" ]]; then
    PRUNED_GAUSSIAN_PATH="./gaussian_output_pruned_keep_${KEEP_COUNT}"
  elif [[ -n "$CASE_KEEP_RATIO_CSV" ]]; then
    PRUNED_GAUSSIAN_PATH="./gaussian_output_pruned_policy_${POLICY_NAME}"
  else
    PRUNED_GAUSSIAN_PATH="./gaussian_output_pruned_${ratio_label}"
  fi
fi

if ((PRUNED_GAUSSIAN_PATH_EXPLICIT == 1)); then
  pruned_path_source="user-provided"
else
  pruned_path_source="auto-derived"
fi

baseline_root="${RESULT_ROOT}/baseline"
pruned_root="${RESULT_ROOT}/${ablation_name}"
comparison_csv="${RESULT_ROOT}/comparison_${ablation_name}.csv"

echo "=== [ablation config] ==="
echo "pruning=enabled"
echo "ablation_name=${ablation_name}"
if [[ -n "$KEEP_COUNT" ]]; then
  echo "keep_count=${KEEP_COUNT}"
else
  echo "fallback_keep_ratio=${KEEP_RATIO}"
fi
if [[ -n "$CASE_KEEP_RATIO_CSV" ]]; then
  echo "case_keep_ratio_csv=${CASE_KEEP_RATIO_CSV}"
  echo "policy_name=${POLICY_NAME}"
else
  echo "case_keep_ratio_csv=disabled"
fi
echo "prune_mode=${PRUNE_MODE}"
echo "PRUNED_GAUSSIAN_PATH=${PRUNED_GAUSSIAN_PATH} (${pruned_path_source})"
echo "comparison_csv=${comparison_csv}"

for case_name in "${cases[@]}"; do
  effective_keep_ratio="$KEEP_RATIO"
  if [[ -n "$CASE_KEEP_RATIO_CSV" && -n "${CASE_KEEP_RATIOS[$case_name]+set}" ]]; then
    effective_keep_ratio="${CASE_KEEP_RATIOS[$case_name]}"
  fi

  src="${GAUSSIAN_PATH}/${case_name}/${EXP_NAME}/point_cloud/iteration_10000/point_cloud.ply"
  dst="${PRUNED_GAUSSIAN_PATH}/${case_name}/${EXP_NAME}/point_cloud/iteration_10000/point_cloud.ply"
  if [[ ! -f "$src" ]]; then
    echo "Error: missing source PLY: ${src}" >&2
    exit 1
  fi
  count_info="$(python - "$src" "$dst" "$effective_keep_ratio" "$KEEP_COUNT" <<'PY'
import sys
from pathlib import Path

from plyfile import PlyData

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
keep_ratio = sys.argv[3]
keep_count_arg = sys.argv[4]

baseline_count = int(PlyData.read(src)["vertex"].count)
if keep_count_arg:
    keep_count = int(keep_count_arg)
else:
    keep_count = int(round(baseline_count * float(keep_ratio)))
expected_count = min(max(1, keep_count), baseline_count)
existing_count = ""
if dst.exists():
    existing_count = str(int(PlyData.read(dst)["vertex"].count))
print(baseline_count, expected_count, existing_count)
PY
)"
  read -r baseline_count expected_count existing_pruned_count <<< "$count_info"

  should_prune=0
  if ((FORCE_PRUNE == 1)) || [[ ! -f "$dst" ]]; then
    should_prune=1
  elif [[ "$existing_pruned_count" != "$expected_count" ]]; then
    if [[ -n "$KEEP_COUNT" ]]; then
      requested_label="keep_count=${KEEP_COUNT}"
    else
      requested_label="keep_ratio=${effective_keep_ratio}"
    fi
    echo "[WARN] Existing pruned PLY count mismatch for ${case_name}:" >&2
    echo "       requested ${requested_label} expects ${expected_count} / ${baseline_count}," >&2
    echo "       found ${existing_pruned_count} in ${dst}" >&2
    echo "       Re-pruning this case." >&2
    should_prune=1
  fi

  if ((should_prune == 1)); then
    prune_cmd=(python benchmarks/scripts/prune_gaussians.py --input "$src" --output "$dst" --mode "$PRUNE_MODE")
    if [[ -n "$KEEP_COUNT" ]]; then
      prune_cmd+=(--keep-count "$KEEP_COUNT")
    else
      prune_cmd+=(--keep-ratio "$effective_keep_ratio")
    fi
    echo "=== [prune] ${case_name} ==="
    if [[ -z "$KEEP_COUNT" ]]; then
      echo "requested_keep_ratio=${effective_keep_ratio}"
    fi
    "${prune_cmd[@]}"
  else
    echo "=== [prune] ${case_name}: using existing ${dst} ==="
    if [[ -z "$KEEP_COUNT" ]]; then
      echo "requested_keep_ratio=${effective_keep_ratio}"
    fi
  fi
done

echo "=== [quality] baseline ==="
bash benchmarks/Boba_Local_single_inst_quality.sh \
  --gaussian_variant baseline \
  --gaussian_path "$GAUSSIAN_PATH" \
  --result_root "$baseline_root" \
  --num_views "$NUM_VIEWS" \
  --overall_mode "$OVERALL_MODE" \
  "${cases[@]}"

echo "=== [quality] ${ablation_name} ==="
bash benchmarks/Boba_Local_single_inst_quality.sh \
  --gaussian_variant pruned \
  --gaussian_path "$PRUNED_GAUSSIAN_PATH" \
  --result_root "$pruned_root" \
  --num_views "$NUM_VIEWS" \
  --overall_mode "$OVERALL_MODE" \
  "${cases[@]}"

python - "$comparison_csv" "$baseline_root" "$pruned_root" "$GAUSSIAN_PATH" "$PRUNED_GAUSSIAN_PATH" "$EXP_NAME" "$KEEP_RATIO" "$KEEP_COUNT" "$CASE_KEEP_RATIO_CSV" "${cases[@]}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

from plyfile import PlyData


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_table(path, key_field):
    path = Path(path)
    rows = {}
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = row.get(key_field)
            if key:
                rows[key] = row
    return rows


def read_keep_ratio_policy(path):
    if not path:
        return {}
    path = Path(path)
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            case_name = (row.get("case_name") or "").strip()
            ratio = (row.get("keep_ratio") or "").strip()
            if case_name:
                rows[case_name] = ratio
    return rows


def parse_float(value):
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def format_value(value):
    if value is None:
        return ""
    return f"{value:.6f}"


def safe_div(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def ply_count(path):
    path = Path(path)
    if not path.exists():
        return None
    return int(PlyData.read(path)["vertex"].count)


comparison_csv = Path(sys.argv[1])
baseline_root = Path(sys.argv[2])
pruned_root = Path(sys.argv[3])
baseline_gaussian_root = Path(sys.argv[4])
pruned_gaussian_root = Path(sys.argv[5])
exp_name = sys.argv[6]
fallback_keep_ratio = sys.argv[7]
keep_count = sys.argv[8]
case_keep_ratio_csv = sys.argv[9]
cases = sys.argv[10:]
case_keep_ratios = read_keep_ratio_policy(case_keep_ratio_csv)

baseline_render = read_table(baseline_root / "metrics" / "render_metrics.csv", "scene")
pruned_render = read_table(pruned_root / "metrics" / "render_metrics.csv", "scene")
baseline_chamfer = read_table(baseline_root / "metrics" / "chamfer.csv", "Case Name")
pruned_chamfer = read_table(pruned_root / "metrics" / "chamfer.csv", "Case Name")
baseline_track = read_table(baseline_root / "metrics" / "track.csv", "Case Name")
pruned_track = read_table(pruned_root / "metrics" / "track.csv", "Case Name")

render_fields = [
    "psnr_train",
    "ssim_train",
    "lpips_train",
    "iou_train",
    "psnr_test",
    "ssim_test",
    "lpips_test",
    "iou_test",
]
chamfer_fields = ["Train Chamfer Error", "Test Chamfer Error"]
track_fields = ["Train Track Error", "Test Track Error"]

fieldnames = [
    "Case Name",
    "Requested Keep Ratio",
    "Baseline Raw Gaussians",
    "Pruned Raw Gaussians",
    "Raw Keep Ratio",
    "Baseline Rendered Gaussians",
    "Pruned Rendered Gaussians",
    "Rendered Keep Ratio",
    "Baseline FPS",
    "Pruned FPS",
    "FPS Speedup",
    "Baseline Total Frame ms",
    "Pruned Total Frame ms",
    "Total Frame Speedup",
    "Baseline Rendering ms",
    "Pruned Rendering ms",
    "Rendering Speedup",
]
for field in render_fields:
    label = field.replace("_", " ").title()
    fieldnames.extend([f"Baseline {label}", f"Pruned {label}", f"Delta {label}"])
for field in chamfer_fields:
    fieldnames.extend([f"Baseline {field}", f"Pruned {field}", f"Delta {field}"])
for field in track_fields:
    fieldnames.extend([f"Baseline {field}", f"Pruned {field}", f"Delta {field}"])

rows = []
for case_name in cases:
    rel_ply = Path(case_name) / exp_name / "point_cloud" / "iteration_10000" / "point_cloud.ply"
    baseline_raw = ply_count(baseline_gaussian_root / rel_ply)
    pruned_raw = ply_count(pruned_gaussian_root / rel_ply)
    baseline_perf = read_json(baseline_root / case_name / "performance_summary.json")
    pruned_perf = read_json(pruned_root / case_name / "performance_summary.json")

    baseline_fps = parse_float(baseline_perf.get("average_fps"))
    pruned_fps = parse_float(pruned_perf.get("average_fps"))
    baseline_total = parse_float(baseline_perf.get("average_total_frame_time_ms"))
    pruned_total = parse_float(pruned_perf.get("average_total_frame_time_ms"))
    baseline_rendering = parse_float(baseline_perf.get("average_rendering_ms"))
    pruned_rendering = parse_float(pruned_perf.get("average_rendering_ms"))
    baseline_rendered = parse_float(baseline_perf.get("rendered_gaussian_count"))
    pruned_rendered = parse_float(pruned_perf.get("rendered_gaussian_count"))
    requested_keep_ratio = "" if keep_count else case_keep_ratios.get(case_name, fallback_keep_ratio)

    row = {
        "Case Name": case_name,
        "Requested Keep Ratio": requested_keep_ratio,
        "Baseline Raw Gaussians": baseline_raw,
        "Pruned Raw Gaussians": pruned_raw,
        "Raw Keep Ratio": format_value(safe_div(pruned_raw, baseline_raw)),
        "Baseline Rendered Gaussians": format_value(baseline_rendered),
        "Pruned Rendered Gaussians": format_value(pruned_rendered),
        "Rendered Keep Ratio": format_value(safe_div(pruned_rendered, baseline_rendered)),
        "Baseline FPS": format_value(baseline_fps),
        "Pruned FPS": format_value(pruned_fps),
        "FPS Speedup": format_value(safe_div(pruned_fps, baseline_fps)),
        "Baseline Total Frame ms": format_value(baseline_total),
        "Pruned Total Frame ms": format_value(pruned_total),
        "Total Frame Speedup": format_value(safe_div(baseline_total, pruned_total)),
        "Baseline Rendering ms": format_value(baseline_rendering),
        "Pruned Rendering ms": format_value(pruned_rendering),
        "Rendering Speedup": format_value(safe_div(baseline_rendering, pruned_rendering)),
    }

    base_render_row = baseline_render.get(case_name, {})
    pruned_render_row = pruned_render.get(case_name, {})
    for field in render_fields:
        label = field.replace("_", " ").title()
        base_value = parse_float(base_render_row.get(field))
        pruned_value = parse_float(pruned_render_row.get(field))
        row[f"Baseline {label}"] = format_value(base_value)
        row[f"Pruned {label}"] = format_value(pruned_value)
        row[f"Delta {label}"] = format_value(
            None if base_value is None or pruned_value is None else pruned_value - base_value
        )

    base_chamfer_row = baseline_chamfer.get(case_name, {})
    pruned_chamfer_row = pruned_chamfer.get(case_name, {})
    for field in chamfer_fields:
        base_value = parse_float(base_chamfer_row.get(field))
        pruned_value = parse_float(pruned_chamfer_row.get(field))
        row[f"Baseline {field}"] = format_value(base_value)
        row[f"Pruned {field}"] = format_value(pruned_value)
        row[f"Delta {field}"] = format_value(
            None if base_value is None or pruned_value is None else pruned_value - base_value
        )

    base_track_row = baseline_track.get(case_name, {})
    pruned_track_row = pruned_track.get(case_name, {})
    for field in track_fields:
        base_value = parse_float(base_track_row.get(field))
        pruned_value = parse_float(pruned_track_row.get(field))
        row[f"Baseline {field}"] = format_value(base_value)
        row[f"Pruned {field}"] = format_value(pruned_value)
        row[f"Delta {field}"] = format_value(
            None if base_value is None or pruned_value is None else pruned_value - base_value
        )

    rows.append(row)

comparison_csv.parent.mkdir(parents=True, exist_ok=True)
with comparison_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[comparison] wrote {comparison_csv}")
PY
