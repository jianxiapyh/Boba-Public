#!/usr/bin/env bash
set -u

CASES_FILE="${CASES_FILE:-data_config.csv}"
NUM_VIEWS=1
OVERALL_MODE=phystwin
GAUSSIAN_VARIANT="baseline"
GAUSSIAN_PATH=""
RESULT_ROOT=""

read_cases_file() {
  mapfile -t cases < <(awk -F, 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); if ($1 != "") print $1}' "$1")
}

usage() {
  cat <<'EOF'
Usage: bash benchmarks/Boba_Local_single_inst_quality.sh [--gaussian_variant baseline|pruned] [--num_views N] [--overall_mode scene_mean|phystwin] [--gaussian_path PATH] [--result_root PATH] [case ...]

This benchmark measures the Boba Local single-instance full runtime quality path:
  spring-mass + LBS + rendering + frame compositing

  --gaussian_variant V
                      Gaussian asset variant to evaluate: baseline or pruned
                      (default: baseline). This only selects the Gaussian root;
                      evaluation metrics are identical for both variants.
  --num_views N      Number of camera views to generate and evaluate (valid: 1, 2, 3)
  --overall_mode M   Render OVERALL aggregation: scene_mean or phystwin (default: phystwin)
  --gaussian_path P  Custom Gaussian root passed to interactive_playground.py.
                     Overrides --gaussian_variant defaults.
                     Defaults: baseline=./gaussian_output,
                     pruned=./gaussian_output_pruned_policy_30_55
  --result_root P    Result root for per-case outputs and metrics.
                     Overrides --gaussian_variant defaults.
                     Defaults: baseline=results/quality,
                     pruned=results/quality_pruned_policy_30_55
EOF
}

cases=()
while (($# > 0)); do
  case "$1" in
    --gaussian_variant|--gaussian-variant)
      shift
      if (($# == 0)); then
        echo "Error: --gaussian_variant requires baseline or pruned." >&2
        usage >&2
        exit 1
      fi
      GAUSSIAN_VARIANT="$1"
      ;;
    --gaussian_variant=*|--gaussian-variant=*)
      GAUSSIAN_VARIANT="${1#*=}"
      ;;
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
    --gaussian_path)
      shift
      if (($# == 0)); then
        echo "Error: --gaussian_path requires a value." >&2
        usage >&2
        exit 1
      fi
      GAUSSIAN_PATH="$1"
      ;;
    --gaussian_path=*)
      GAUSSIAN_PATH="${1#*=}"
      ;;
    --result_root)
      shift
      if (($# == 0)); then
        echo "Error: --result_root requires a value." >&2
        usage >&2
        exit 1
      fi
      RESULT_ROOT="$1"
      ;;
    --result_root=*)
      RESULT_ROOT="${1#*=}"
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

if [[ "$GAUSSIAN_VARIANT" != "baseline" && "$GAUSSIAN_VARIANT" != "pruned" ]]; then
  echo "Error: --gaussian_variant must be baseline or pruned. Received: ${GAUSSIAN_VARIANT}" >&2
  exit 1
fi

if [[ -z "$GAUSSIAN_PATH" ]]; then
  if [[ "$GAUSSIAN_VARIANT" == "pruned" ]]; then
    GAUSSIAN_PATH="./gaussian_output_pruned_policy_30_55"
  else
    GAUSSIAN_PATH="./gaussian_output"
  fi
fi

if [[ -z "$RESULT_ROOT" ]]; then
  if [[ "$GAUSSIAN_VARIANT" == "pruned" ]]; then
    RESULT_ROOT="results/quality_pruned_policy_30_55"
  else
    RESULT_ROOT="results/quality"
  fi
fi

if ((${#cases[@]} == 0)); then
  read_cases_file "$CASES_FILE"
fi

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/metrics"

for case_name in "${cases[@]}"; do
  output_dir="${RESULT_ROOT}/${case_name}"
  log_path="${RESULT_ROOT}/logs/${case_name}.log"

  echo "=== [quality] variant=${GAUSSIAN_VARIANT} :: ${case_name} ==="
  if python interactive_playground.py \
    --mode quality \
    --case_name "$case_name" \
    --num_views "$NUM_VIEWS" \
    --gaussian_path "$GAUSSIAN_PATH" \
    --output_dir "$output_dir" \
    >"$log_path" 2>&1; then
    echo "[OK] ${case_name}"
  else
    echo "[FAIL] ${case_name}"
  fi
done

python export_render_eval_data.py

python evaluate_chamfer.py \
  --prediction_path "$RESULT_ROOT" \
  --output_file "${RESULT_ROOT}/metrics/chamfer.csv"

python evaluate_track.py \
  --prediction_path "$RESULT_ROOT" \
  --output_file "${RESULT_ROOT}/metrics/track.csv"

python gaussian_splatting/evaluate_render.py \
  --output_dir "$RESULT_ROOT" \
  --num_views "$NUM_VIEWS" \
  --overall_mode "$OVERALL_MODE" \
  --text_output "${RESULT_ROOT}/metrics/render_metrics.txt" \
  --csv_output "${RESULT_ROOT}/metrics/render_metrics.csv"
