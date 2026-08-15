#!/usr/bin/env bash
set -u

NUM_RUNS="${NUM_RUNS:-3}"
CASES_FILE="${CASES_FILE:-benchmarks/cases.txt}"

if (($# > 0)); then
  cases=("$@")
else
  mapfile -t cases < "$CASES_FILE"
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

python benchmarks/aggregate_perf_runs.py "${cases[@]}"
