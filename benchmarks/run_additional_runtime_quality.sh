#!/usr/bin/env bash
set -u

CASES_FILE="${CASES_FILE:-benchmarks/additional_data_config.csv}"
NUM_VIEWS=1
OVERALL_MODE=phystwin
RESULTS_ROOT="results/additional_quality"
RENDER_PATH="./data/render_eval_data_additional"
HUMAN_MASK_PATH="./data/different_types_human_mask_additional"

read_cases_file() {
  mapfile -t cases < <(awk -F, 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); if ($1 != "") print $1}' "$1")
}

usage() {
  cat <<'EOF'
Usage: bash benchmarks/run_additional_runtime_quality.sh [--num_views N] [--overall_mode scene_mean|phystwin] [case ...]

This benchmark measures the single-instance full runtime quality path on the additional cases:
  spring-mass + LBS + rendering + frame compositing

  --num_views N      Number of camera views to generate and evaluate (valid: 1, 2, 3)
  --overall_mode M   Render OVERALL aggregation: scene_mean or phystwin (default: phystwin)
EOF
}

cases=()
while (($# > 0)); do
  case "$1" in
    --num_views)
      shift
      if (($# == 0)); then
        echo "Error: --num_views requires an integer value." >&2
        usage >&2
        exit 1
      fi
      NUM_VIEWS="$1"
      ;;
    --num_views=*)
      NUM_VIEWS="${1#*=}"
      ;;
    --overall_mode)
      shift
      if (($# == 0)); then
        echo "Error: --overall_mode requires scene_mean or phystwin." >&2
        usage >&2
        exit 1
      fi
      OVERALL_MODE="$1"
      ;;
    --overall_mode=*)
      OVERALL_MODE="${1#*=}"
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

if [[ ! "$NUM_VIEWS" =~ ^[0-9]+$ ]]; then
  echo "Error: --num_views must be an integer between 1 and 3." >&2
  exit 1
fi

if ((NUM_VIEWS < 1 || NUM_VIEWS > 3)); then
  echo "Error: --num_views must be between 1 and 3. Received: ${NUM_VIEWS}" >&2
  exit 1
fi

if [[ "$OVERALL_MODE" != "scene_mean" && "$OVERALL_MODE" != "phystwin" ]]; then
  echo "Error: --overall_mode must be scene_mean or phystwin. Received: ${OVERALL_MODE}" >&2
  exit 1
fi

if ((${#cases[@]} == 0)); then
  read_cases_file "$CASES_FILE"
fi

mkdir -p "${RESULTS_ROOT}/logs" "${RESULTS_ROOT}/metrics"

for case_name in "${cases[@]}"; do
  output_dir="${RESULTS_ROOT}/${case_name}"
  log_path="${RESULTS_ROOT}/logs/${case_name}.log"

  echo "=== [additional quality] ${case_name} ==="
  if python interactive_playground.py \
    --mode quality \
    --case_name "$case_name" \
    --num_views "$NUM_VIEWS" \
    --output_dir "$output_dir" \
    >"$log_path" 2>&1; then
    echo "[OK] ${case_name}"
  else
    echo "[FAIL] ${case_name}"
  fi
done

python evaluate_chamfer.py \
  --prediction_path "$RESULTS_ROOT" \
  --output_file "${RESULTS_ROOT}/metrics/chamfer.csv"

python gaussian_splatting/evaluate_render.py \
  --render_path "$RENDER_PATH" \
  --human_mask_path "$HUMAN_MASK_PATH" \
  --output_dir "$RESULTS_ROOT" \
  --num_views "$NUM_VIEWS" \
  --overall_mode "$OVERALL_MODE" \
  --text_output "${RESULTS_ROOT}/metrics/render_metrics.txt" \
  --csv_output "${RESULTS_ROOT}/metrics/render_metrics.csv"
