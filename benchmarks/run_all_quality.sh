#!/usr/bin/env bash
set -u

CASES_FILE="${CASES_FILE:-benchmarks/cases.txt}"
NUM_VIEWS=1

usage() {
  cat <<'EOF'
Usage: bash benchmarks/run_all_quality.sh [--num_views N] [case ...]

  --num_views N   Number of camera views to generate and evaluate (valid: 1, 2, 3)
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

if ((${#cases[@]} == 0)); then
  mapfile -t cases < "$CASES_FILE"
fi

mkdir -p results/quality/logs results/quality/metrics

for case_name in "${cases[@]}"; do
  output_dir="results/quality/${case_name}"
  log_path="results/quality/logs/${case_name}.log"

  echo "=== [quality] ${case_name} ==="
  if python interactive_playground.py \
    --mode quality \
    --case_name "$case_name" \
    --num_views "$NUM_VIEWS" \
    --output_dir "$output_dir" \
    >"$log_path" 2>&1; then
    echo "[OK] ${case_name}"
    if [[ -d "$output_dir/output" ]]; then
      rm -rf "$output_dir/output"
    fi
  else
    echo "[FAIL] ${case_name}"
  fi
done

python export_render_eval_data.py

python evaluate_chamfer.py \
  --prediction_path results/quality \
  --output_file results/quality/metrics/chamfer.csv

python evaluate_track.py \
  --prediction_path results/quality \
  --output_file results/quality/metrics/track.csv

python gaussian_splatting/evaluate_render.py \
  --output_dir results/quality \
  --num_views "$NUM_VIEWS" \
  --text_output results/quality/metrics/render_metrics.txt \
  --csv_output results/quality/metrics/render_metrics.csv
