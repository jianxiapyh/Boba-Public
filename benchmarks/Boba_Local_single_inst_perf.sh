#!/usr/bin/env bash
set -u

NUM_RUNS="${NUM_RUNS:-3}"
CASES_FILE="${CASES_FILE:-data_config.csv}"

read_cases_file() {
  mapfile -t cases < <(awk -F, 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); if ($1 != "") print $1}' "$1")
}

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/Boba_Local_single_inst_perf.sh [case_name ...]

This benchmark measures the Boba Local single-instance full runtime path:
  spring-mass + LBS + rendering + frame compositing

Environment overrides:
  NUM_RUNS
  CASES_FILE
EOF
}

cases=()
while (($# > 0)); do
  case "$1" in
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

mkdir -p results/perf

for ((run_idx=1; run_idx<=NUM_RUNS; run_idx++)); do
  run_name=$(printf "run_%02d" "$run_idx")
  mkdir -p "results/perf/logs/${run_name}"

  for case_name in "${cases[@]}"; do
    output_dir="results/perf/${run_name}/${case_name}"
    log_path="results/perf/logs/${run_name}/${case_name}.log"

    echo "=== [perf] ${run_name} :: ${case_name} ==="
    if python interactive_playground.py \
      --mode perf \
      --case_name "$case_name" \
      --output_dir "$output_dir" \
      >"$log_path" 2>&1; then
      echo "[OK] ${case_name} (${run_name})"
    else
      echo "[FAIL] ${case_name} (${run_name})"
    fi
  done
done

python benchmarks/post-processing/aggregate_full_runtime_perf_runs.py "${cases[@]}"
