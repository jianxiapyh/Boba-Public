#!/usr/bin/env bash
set -u

NUM_RUNS="${NUM_RUNS:-1}"
CASES_FILE="${CASES_FILE:-data_config.csv}"
BASE_PATH="${BASE_PATH:-./data/different_types}"
GAUSSIAN_PATH="${GAUSSIAN_PATH:-./gaussian_output}"
BG_IMG_PATH="${BG_IMG_PATH:-./data/bg.png}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-256}"
REFINE_SAMPLES="${REFINE_SAMPLES:-9}"
REFINE_ROUNDS="${REFINE_ROUNDS:-2}"
FINAL_DENSE_WINDOW="${FINAL_DENSE_WINDOW:-8}"
RESULTS_ROOT="${RESULTS_ROOT:-results/batch_autotune}"
ATTEMPTED_MANIFEST="${RESULTS_ROOT}/attempted_candidates.csv"

read_cases_file() {
  mapfile -t cases < <(awk -F, 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); if ($1 != "") print $1}' "$1")
}

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_sim_lbs_best_throughput.sh [case_name ...]

This benchmark automatically searches for the best Sim+LBS throughput:
  spring-mass + LBS

Search policy:
  1. Try powers of two from MIN_BATCH_SIZE to MAX_BATCH_SIZE.
  2. Stop power-of-two expansion for a case after a batch size has zero successful runs.
  3. Sample and zoom around the best successful power-of-two batch.
  4. Densely refine only a small final window around the best sampled batch.

Environment overrides:
  NUM_RUNS            default: 1
  CASES_FILE         default: data_config.csv
  BASE_PATH          default: ./data/different_types
  GAUSSIAN_PATH      default: ./gaussian_output
  BG_IMG_PATH        default: ./data/bg.png
  MIN_BATCH_SIZE     default: 1
  MAX_BATCH_SIZE     default: 256
  REFINE_SAMPLES     default: 9
  REFINE_ROUNDS      default: 2
  FINAL_DENSE_WINDOW default: 8
  RESULTS_ROOT       default: results/batch_autotune
EOF
}

validate_positive_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]] || ((value < 1)); then
    echo "[ERROR] ${name} must be a positive integer. Received: ${value}" >&2
    exit 1
  fi
}

find_best_batch() {
  local case_name="$1"
  shift
  if (($# == 0)); then
    return 0
  fi

  python - "$RESULTS_ROOT" "$case_name" "$NUM_RUNS" "$@" <<'PY'
import os
import sys


def parse_throughput(summary_path):
    with open(summary_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line.startswith("Average Throughput (instances/s):"):
                return float(line.split(":", 1)[1].strip())
    return None


results_root = sys.argv[1]
case_name = sys.argv[2]
num_runs = int(sys.argv[3])
batch_sizes = [int(value) for value in sys.argv[4:]]

best_batch = None
best_throughput = None
for batch_size in batch_sizes:
    values = []
    for run_idx in range(1, num_runs + 1):
        run_name = f"run_{run_idx:02d}"
        summary_path = os.path.join(
            results_root,
            run_name,
            f"batch_{batch_size}",
            case_name,
            "performance_summary.txt",
        )
        if not os.path.isfile(summary_path):
            continue
        throughput = parse_throughput(summary_path)
        if throughput is not None:
            values.append(throughput)

    if not values:
        continue

    average_throughput = sum(values) / len(values)
    if (
        best_throughput is None
        or average_throughput > best_throughput
        or (
            average_throughput == best_throughput
            and batch_size < best_batch
        )
    ):
        best_batch = batch_size
        best_throughput = average_throughput

if best_batch is not None:
    print(best_batch)
PY
}

generate_sampled_batches() {
  local start="$1"
  local end="$2"
  local samples="$3"

  if ((end < start)); then
    return 0
  fi

  if ((samples <= 1 || start == end)); then
    echo $(((start + end) / 2))
    return 0
  fi

  local span=$((end - start))
  local denom=$((samples - 1))
  local i
  declare -A emitted=()

  for ((i=0; i<samples; i++)); do
    local batch_size=$((start + (i * span + denom / 2) / denom))
    if ((batch_size < start)); then
      batch_size="$start"
    fi
    if ((batch_size > end)); then
      batch_size="$end"
    fi
    if [[ -z "${emitted[$batch_size]+x}" ]]; then
      emitted["$batch_size"]=1
      echo "$batch_size"
    fi
  done
}

was_tested() {
  local batch_size="$1"
  [[ -n "${tested_batches[$batch_size]+x}" ]]
}

run_candidate() {
  local case_name="$1"
  local batch_size="$2"
  local successful_runs=0

  tested_batches["$batch_size"]=1
  printf "%s,%s\n" "$case_name" "$batch_size" >> "$ATTEMPTED_MANIFEST"

  for ((run_idx=1; run_idx<=NUM_RUNS; run_idx++)); do
    local run_name
    run_name=$(printf "run_%02d" "$run_idx")
    local output_dir="${RESULTS_ROOT}/${run_name}/batch_${batch_size}/${case_name}"
    local log_dir="${RESULTS_ROOT}/logs/${run_name}/batch_${batch_size}"
    local log_path="${log_dir}/${case_name}.log"

    mkdir -p "$output_dir" "$log_dir"
    rm -f "${output_dir}/performance_summary.txt"

    echo "=== [best_throughput] ${run_name} :: batch=${batch_size} :: ${case_name} ==="
    if python benchmarks/run_sim_lbs_batch_scaling_case.py \
      --base_path "$BASE_PATH" \
      --gaussian_path "$GAUSSIAN_PATH" \
      --bg_img_path "$BG_IMG_PATH" \
      --case_name "$case_name" \
      --batch_size "$batch_size" \
      --output_dir "$output_dir" \
      >"$log_path" 2>&1; then
      successful_runs=$((successful_runs + 1))
      echo "[OK] ${case_name} (${run_name}, batch=${batch_size})"
    else
      echo "[FAIL] ${case_name} (${run_name}, batch=${batch_size})"
    fi
  done

  LAST_SUCCESSFUL_RUNS="$successful_runs"
}

run_candidate_if_needed() {
  local case_name="$1"
  local batch_size="$2"
  if was_tested "$batch_size"; then
    return 0
  fi
  run_candidate "$case_name" "$batch_size"
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

validate_positive_integer "NUM_RUNS" "$NUM_RUNS"
validate_positive_integer "MIN_BATCH_SIZE" "$MIN_BATCH_SIZE"
validate_positive_integer "MAX_BATCH_SIZE" "$MAX_BATCH_SIZE"
validate_positive_integer "REFINE_SAMPLES" "$REFINE_SAMPLES"
validate_positive_integer "REFINE_ROUNDS" "$REFINE_ROUNDS"
validate_positive_integer "FINAL_DENSE_WINDOW" "$FINAL_DENSE_WINDOW"

if ((MIN_BATCH_SIZE > MAX_BATCH_SIZE)); then
  echo "[ERROR] MIN_BATCH_SIZE cannot exceed MAX_BATCH_SIZE." >&2
  exit 1
fi

if ((${#cases[@]} == 0)); then
  read_cases_file "$CASES_FILE"
fi

mkdir -p "$RESULTS_ROOT"
printf "Case Name,Batch Size\n" > "$ATTEMPTED_MANIFEST"

for case_name in "${cases[@]}"; do
  echo "=== [best_throughput] searching case=${case_name} ==="

  declare -A tested_batches=()
  power_batches=()
  LAST_SUCCESSFUL_RUNS=0

  batch_size=1
  while ((batch_size < MIN_BATCH_SIZE)); do
    batch_size=$((batch_size * 2))
  done

  while ((batch_size <= MAX_BATCH_SIZE)); do
    run_candidate_if_needed "$case_name" "$batch_size"
    power_batches+=("$batch_size")

    if ((LAST_SUCCESSFUL_RUNS == 0)); then
      echo "[STOP] ${case_name}: batch=${batch_size} had zero successful runs; stopping power-of-two expansion."
      break
    fi

    batch_size=$((batch_size * 2))
  done

  best_power_batch=$(find_best_batch "$case_name" "${power_batches[@]}")
  if [[ -z "$best_power_batch" ]]; then
    echo "[WARN] ${case_name}: no successful power-of-two candidate; skipping dense refinement."
    continue
  fi

  refine_start="$MIN_BATCH_SIZE"
  refine_end="$MAX_BATCH_SIZE"
  for power_index in "${!power_batches[@]}"; do
    if ((power_batches[power_index] == best_power_batch)); then
      if ((power_index > 0)); then
        refine_start="${power_batches[power_index - 1]}"
      fi
      if ((power_index + 1 < ${#power_batches[@]})); then
        refine_end="${power_batches[power_index + 1]}"
      fi
      break
    fi
  done
  if ((refine_start < MIN_BATCH_SIZE)); then
    refine_start="$MIN_BATCH_SIZE"
  fi
  if ((refine_end > MAX_BATCH_SIZE)); then
    refine_end="$MAX_BATCH_SIZE"
  fi

  current_best_batch="$best_power_batch"
  echo "[REFINE] ${case_name}: best power batch=${best_power_batch}; initial bracket=${refine_start}..${refine_end}."

  for ((refine_round=1; refine_round<=REFINE_ROUNDS; refine_round++)); do
    mapfile -t sampled_batches < <(generate_sampled_batches "$refine_start" "$refine_end" "$REFINE_SAMPLES")
    round_candidates=("${sampled_batches[@]}" "$current_best_batch")

    echo "[REFINE] ${case_name}: round=${refine_round}, bracket=${refine_start}..${refine_end}, testing samples: ${sampled_batches[*]}"
    for batch_size in "${sampled_batches[@]}"; do
      run_candidate_if_needed "$case_name" "$batch_size"
    done

    round_best_batch=$(find_best_batch "$case_name" "${round_candidates[@]}")
    if [[ -z "$round_best_batch" ]]; then
      echo "[WARN] ${case_name}: no successful sampled candidate in round ${refine_round}; stopping refinement."
      break
    fi

    current_best_batch="$round_best_batch"
    mapfile -t ordered_round_candidates < <(printf "%s\n" "${round_candidates[@]}" | sort -n -u)

    next_refine_start="$refine_start"
    next_refine_end="$refine_end"
    for ordered_index in "${!ordered_round_candidates[@]}"; do
      if ((ordered_round_candidates[ordered_index] == current_best_batch)); then
        if ((ordered_index > 0)); then
          next_refine_start="${ordered_round_candidates[ordered_index - 1]}"
        fi
        if ((ordered_index + 1 < ${#ordered_round_candidates[@]})); then
          next_refine_end="${ordered_round_candidates[ordered_index + 1]}"
        fi
        break
      fi
    done

    refine_start="$next_refine_start"
    refine_end="$next_refine_end"
  done

  final_best_batch=$(find_best_batch "$case_name" "${!tested_batches[@]}")
  if [[ -z "$final_best_batch" ]]; then
    echo "[WARN] ${case_name}: no successful candidate after sampled refinement; skipping final dense window."
    continue
  fi

  final_start=$((final_best_batch - FINAL_DENSE_WINDOW))
  final_end=$((final_best_batch + FINAL_DENSE_WINDOW))
  if ((final_start < MIN_BATCH_SIZE)); then
    final_start="$MIN_BATCH_SIZE"
  fi
  if ((final_end > MAX_BATCH_SIZE)); then
    final_end="$MAX_BATCH_SIZE"
  fi

  echo "[FINAL] ${case_name}: best sampled batch=${final_best_batch}; dense testing ${final_start}..${final_end}."
  for ((batch_size=final_start; batch_size<=final_end; batch_size++)); do
    run_candidate_if_needed "$case_name" "$batch_size"
  done
done

python benchmarks/post-processing/aggregate_sim_lbs_best_throughput.py \
  --results_root "$RESULTS_ROOT" \
  --cases_file "$CASES_FILE" \
  --output_best "${RESULTS_ROOT}/best_throughput_table.csv" \
  --output_candidates "${RESULTS_ROOT}/candidate_table.csv" \
  --attempted_manifest "$ATTEMPTED_MANIFEST" \
  --num_runs "$NUM_RUNS" \
  "${cases[@]}"

echo "[DONE] Best-throughput table: ${RESULTS_ROOT}/best_throughput_table.csv"
echo "[DONE] Candidate table: ${RESULTS_ROOT}/candidate_table.csv"
