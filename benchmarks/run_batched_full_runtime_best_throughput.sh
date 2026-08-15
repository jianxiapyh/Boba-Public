#!/usr/bin/env bash
set -u

NUM_RUNS="${NUM_RUNS:-1}"
MIN_SUCCESSES="${MIN_SUCCESSES:-1}"
CASES_FILE="${CASES_FILE:-data_config.csv}"
BASE_PATH="${BASE_PATH:-./data/different_types}"
GAUSSIAN_PATH="${GAUSSIAN_PATH:-./gaussian_output}"
PRUNED_GAUSSIAN_PATH="${PRUNED_GAUSSIAN_PATH:-./gaussian_output_pruned_policy_30_55}"
BG_IMG_PATH="${BG_IMG_PATH:-./data/bg.png}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-1}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-2048}"
REFINE_SAMPLES="${REFINE_SAMPLES:-9}"
REFINE_ROUNDS="${REFINE_ROUNDS:-2}"
FINAL_DENSE_WINDOW="${FINAL_DENSE_WINDOW:-8}"
RESULTS_ROOT_WAS_SET=0
if [[ -n "${RESULTS_ROOT+x}" ]]; then
  RESULTS_ROOT_WAS_SET=1
fi
RESULTS_ROOT="${RESULTS_ROOT:-results/batched_full_runtime_autotune}"
BATCH_IMAGE_RESOLUTION="${BATCH_IMAGE_RESOLUTION:-640x480}"
RENDER_MODE="${RENDER_MODE:-batch_images}"
GAUSSIAN_RENDER_MODE="${GAUSSIAN_RENDER_MODE:-shared_template}"
BATCHED_RENDER_VARIANT="${BATCHED_RENDER_VARIANT-batch_prune}"
SIM_FORCE_MODE="${SIM_FORCE_MODE:-gather}"
NUM_VIEWS="${NUM_VIEWS:-1}"

read_cases_file() {
  mapfile -t cases < <(awk -F, 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); if ($1 != "") print $1}' "$1")
}

normalize_batched_render_variant() {
  case "$1" in
    batch_original|batch_optimized|batch_prune)
      printf "%s\n" "$1"
      ;;
    baseline)
      echo "[WARN] --batched_render_variant baseline is deprecated; use batch_original." >&2
      printf "%s\n" "batch_original"
      ;;
    optimized)
      echo "[WARN] --batched_render_variant optimized is deprecated; use batch_optimized." >&2
      printf "%s\n" "batch_optimized"
      ;;
    optimized_pruned)
      echo "[WARN] --batched_render_variant optimized_pruned is deprecated; use batch_prune." >&2
      printf "%s\n" "batch_prune"
      ;;
    *)
      echo "[ERROR] BATCHED_RENDER_VARIANT must be batch_original, batch_optimized, or batch_prune. Received: $1" >&2
      return 1
      ;;
  esac
}

validate_sim_force_mode() {
  case "$1" in
    gather|template_state_batched_atomic)
      ;;
    *)
      echo "[ERROR] SIM_FORCE_MODE must be gather or template_state_batched_atomic. Received: $1" >&2
      return 1
      ;;
  esac
}

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_batched_full_runtime_best_throughput.sh [--batched_render_variant batch_original|batch_optimized|batch_prune] [--sim_force_mode gather|template_state_batched_atomic] [case_name ...]

This benchmark automatically searches for the best full-runtime throughput:
  spring-mass + LBS + batched rendering + frame compositing

Default benchmark path:
  --batched_render_variant batch_prune
  --render_mode batch_images
  --gaussian_render_mode shared_template
  --batch_image_resolution 640x480
  GAUSSIAN_PATH=./gaussian_output
  PRUNED_GAUSSIAN_PATH=./gaussian_output_pruned_policy_30_55

Deprecated aliases are accepted with a warning:
  baseline -> batch_original
  optimized -> batch_optimized
  optimized_pruned -> batch_prune

Search policy:
  1. Try powers of two from MIN_BATCH_SIZE to MAX_BATCH_SIZE.
  2. Stop power-of-two expansion for a case after a batch size has zero successful runs.
  3. Sample and zoom around the best successful power-of-two batch.
  4. Densely refine only a small final window around the best sampled batch.

Environment overrides:
  NUM_RUNS               default: 1
  MIN_SUCCESSES          default: 1. Candidate must have at least this many
                         successful runs to count as supported/eligible.
  CASES_FILE            default: data_config.csv
  BASE_PATH             default: ./data/different_types
  GAUSSIAN_PATH         default: ./gaussian_output
  PRUNED_GAUSSIAN_PATH  default: ./gaussian_output_pruned_policy_30_55
  BG_IMG_PATH           default: ./data/bg.png
  MIN_BATCH_SIZE        default: 1
  MAX_BATCH_SIZE        default: 2048
  REFINE_SAMPLES        default: 9
  REFINE_ROUNDS         default: 2
  FINAL_DENSE_WINDOW    default: 8
  RESULTS_ROOT          default: results/batched_full_runtime_autotune/<config>
                        If set, used exactly as provided.
  RENDER_MODE           default: batch_images
                        This autotune script currently supports batch_images.
  GAUSSIAN_RENDER_MODE  default: shared_template
  BATCH_IMAGE_RESOLUTION default: 640x480
  BATCHED_RENDER_VARIANT default: batch_prune
  SIM_FORCE_MODE        default: gather

Output layout:
  Default outputs are grouped by simulation/render config, e.g.:
    results/batched_full_runtime_autotune/sim_gather_render_batch_images_640x480_shared_template_batch_prune/
  Use --sim_force_mode template_state_batched_atomic to select the template-state
  batched atomic simulation path.
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

mode_dir_for_current_config() {
  local mode_dir="batch_images"
  if [[ "$BATCH_IMAGE_RESOLUTION" != "native" ]]; then
    mode_dir="batch_images_${BATCH_IMAGE_RESOLUTION}"
  fi
  mode_dir="${mode_dir}_${GAUSSIAN_RENDER_MODE}"
  if [[ -n "$BATCHED_RENDER_VARIANT" ]]; then
    mode_dir="${mode_dir}_${BATCHED_RENDER_VARIANT}"
  fi
  if [[ "$SIM_FORCE_MODE" != "gather" ]]; then
    mode_dir="${mode_dir}_sim_${SIM_FORCE_MODE}"
  fi
  printf "%s\n" "$mode_dir"
}

config_dir_for_current_config() {
  local render_config=""
  if [[ "$RENDER_MODE" == "batch_images" ]]; then
    render_config="batch_images_${BATCH_IMAGE_RESOLUTION}_${GAUSSIAN_RENDER_MODE}"
    if [[ -n "$BATCHED_RENDER_VARIANT" ]]; then
      render_config="${render_config}_${BATCHED_RENDER_VARIANT}"
    fi
  else
    render_config="${RENDER_MODE}_${GAUSSIAN_RENDER_MODE}"
  fi
  printf "sim_%s_render_%s\n" "$SIM_FORCE_MODE" "$render_config"
}

find_best_batch() {
  local case_name="$1"
  shift
  if (($# == 0)); then
    return 0
  fi

  python - "$RESULTS_ROOT" "$case_name" "$NUM_RUNS" "$MIN_SUCCESSES" "$MODE_DIR" "$@" <<'PY'
import json
import os
import sys


def parse_throughput(summary_dir):
    json_path = os.path.join(summary_dir, "performance_summary.json")
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)
        frames = metrics.get("frames_used_for_stats")
        if frames is None or int(frames) <= 0:
            return None
        value = metrics.get("average_throughput")
        return float(value) if value is not None else None

    text_path = os.path.join(summary_dir, "performance_summary.txt")
    if os.path.isfile(text_path):
        frames = None
        throughput = None
        with open(text_path, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if line.startswith("=== Final Summary (averaged over "):
                    prefix = "=== Final Summary (averaged over "
                    suffix = " frames) ==="
                    if line.endswith(suffix):
                        frames = int(line[len(prefix) : -len(suffix)])
                if line.startswith("Average Throughput (instances/s):"):
                    throughput = float(line.split(":", 1)[1].strip())
        if frames is None or frames <= 0:
            return None
        return throughput
    return None


results_root = sys.argv[1]
case_name = sys.argv[2]
num_runs = int(sys.argv[3])
min_successes = int(sys.argv[4])
mode_dir = sys.argv[5]
batch_sizes = [int(value) for value in sys.argv[6:]]

best_batch = None
best_throughput = None
for batch_size in batch_sizes:
    values = []
    for run_idx in range(1, num_runs + 1):
        run_name = f"run_{run_idx:02d}"
        summary_dir = os.path.join(
            results_root,
            run_name,
            case_name,
            f"batch_{batch_size}",
            mode_dir,
        )
        throughput = parse_throughput(summary_dir)
        if throughput is not None:
            values.append(throughput)

    if len(values) < min_successes:
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
    local output_dir="${RESULTS_ROOT}/${run_name}/${case_name}/batch_${batch_size}/${MODE_DIR}"
    local log_dir="${RESULTS_ROOT}/logs/${run_name}/batch_${batch_size}"
    local log_path="${log_dir}/${case_name}.log"

    mkdir -p "$output_dir" "$log_dir"
    rm -f "${output_dir}/performance_summary.txt" "${output_dir}/performance_summary.json"

    local cmd=(
      python benchmarks/run_batched_full_runtime_case.py
      --base_path "$BASE_PATH"
      --gaussian_path "$GAUSSIAN_PATH"
      --bg_img_path "$BG_IMG_PATH"
      --case_name "$case_name"
      --batch_size "$batch_size"
      --render_mode "$RENDER_MODE"
      --gaussian_render_mode "$GAUSSIAN_RENDER_MODE"
      --num_views "$NUM_VIEWS"
      --batch_image_resolution "$BATCH_IMAGE_RESOLUTION"
      --sim_force_mode "$SIM_FORCE_MODE"
      --output_dir "$output_dir"
      --pruned_gaussian_path "$PRUNED_GAUSSIAN_PATH"
    )
    if [[ -n "$BATCHED_RENDER_VARIANT" ]]; then
      cmd+=(--batched_render_variant "$BATCHED_RENDER_VARIANT")
    fi

    echo "=== [full_runtime_best_throughput] ${run_name} :: batch=${batch_size} :: ${case_name} ==="
    if "${cmd[@]}" >"$log_path" 2>&1; then
      if candidate_has_valid_summary "$output_dir"; then
        successful_runs=$((successful_runs + 1))
        echo "[OK] ${case_name} (${run_name}, batch=${batch_size})"
      else
        rm -f "${output_dir}/performance_summary.txt" "${output_dir}/performance_summary.json"
        echo "[FAIL] ${case_name} (${run_name}, batch=${batch_size}, no measured frames)"
      fi
    else
      rm -f "${output_dir}/performance_summary.txt" "${output_dir}/performance_summary.json"
      echo "[FAIL] ${case_name} (${run_name}, batch=${batch_size})"
    fi
  done

  LAST_SUCCESSFUL_RUNS="$successful_runs"
}

candidate_has_valid_summary() {
  local output_dir="$1"
  python - "$output_dir" <<'PY'
import json
import os
import re
import sys


def frames_from_text(path):
    pattern = re.compile(r"=== Final Summary \(averaged over (\d+) frames\) ===")
    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            match = pattern.match(raw_line.strip())
            if match:
                return int(match.group(1))
    return 0


output_dir = sys.argv[1]
json_path = os.path.join(output_dir, "performance_summary.json")
if os.path.isfile(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)
    sys.exit(0 if int(metrics.get("frames_used_for_stats") or 0) > 0 else 1)

text_path = os.path.join(output_dir, "performance_summary.txt")
if os.path.isfile(text_path):
    sys.exit(0 if frames_from_text(text_path) > 0 else 1)

sys.exit(1)
PY
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
    --batched_render_variant|--batched-render-variant)
      shift
      if (($# == 0)); then
        echo "Error: --batched_render_variant requires a value." >&2
        usage >&2
        exit 1
      fi
      BATCHED_RENDER_VARIANT="$1"
      ;;
    --batched_render_variant=*|--batched-render-variant=*)
      BATCHED_RENDER_VARIANT="${1#*=}"
      ;;
    --sim_force_mode|--sim-force-mode)
      shift
      if (($# == 0)); then
        echo "Error: --sim_force_mode requires a value." >&2
        usage >&2
        exit 1
      fi
      SIM_FORCE_MODE="$1"
      ;;
    --sim_force_mode=*|--sim-force-mode=*)
      SIM_FORCE_MODE="${1#*=}"
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

validate_positive_integer "NUM_RUNS" "$NUM_RUNS"
validate_positive_integer "MIN_SUCCESSES" "$MIN_SUCCESSES"
validate_positive_integer "MIN_BATCH_SIZE" "$MIN_BATCH_SIZE"
validate_positive_integer "MAX_BATCH_SIZE" "$MAX_BATCH_SIZE"
validate_positive_integer "REFINE_SAMPLES" "$REFINE_SAMPLES"
validate_positive_integer "REFINE_ROUNDS" "$REFINE_ROUNDS"
validate_positive_integer "FINAL_DENSE_WINDOW" "$FINAL_DENSE_WINDOW"
validate_positive_integer "NUM_VIEWS" "$NUM_VIEWS"
if ! validate_sim_force_mode "$SIM_FORCE_MODE"; then
  exit 1
fi

if ((MIN_BATCH_SIZE > MAX_BATCH_SIZE)); then
  echo "[ERROR] MIN_BATCH_SIZE cannot exceed MAX_BATCH_SIZE." >&2
  exit 1
fi
if ((MIN_SUCCESSES > NUM_RUNS)); then
  echo "[ERROR] MIN_SUCCESSES cannot exceed NUM_RUNS (${NUM_RUNS}). Received: ${MIN_SUCCESSES}" >&2
  exit 1
fi

if [[ -n "$BATCHED_RENDER_VARIANT" ]]; then
  if ! BATCHED_RENDER_VARIANT="$(normalize_batched_render_variant "$BATCHED_RENDER_VARIANT")"; then
    exit 1
  fi
  case "$BATCHED_RENDER_VARIANT" in
    batch_original)
      RENDER_MODE="batch_images"
      GAUSSIAN_RENDER_MODE="duplicated"
      GAUSSIAN_PATH="./gaussian_output"
      NUM_VIEWS=1
      ;;
    batch_optimized)
      RENDER_MODE="batch_images"
      GAUSSIAN_RENDER_MODE="shared_template"
      GAUSSIAN_PATH="./gaussian_output"
      NUM_VIEWS=1
      ;;
    batch_prune)
      RENDER_MODE="batch_images"
      GAUSSIAN_RENDER_MODE="shared_template"
      GAUSSIAN_PATH="$PRUNED_GAUSSIAN_PATH"
      NUM_VIEWS=1
      ;;
  esac
fi

if [[ "$RENDER_MODE" != "batch_images" ]]; then
  echo "[ERROR] This autotune script supports only RENDER_MODE=batch_images." >&2
  exit 1
fi
if [[ "$GAUSSIAN_RENDER_MODE" != "shared_template" && "$GAUSSIAN_RENDER_MODE" != "duplicated" ]]; then
  echo "[ERROR] GAUSSIAN_RENDER_MODE must be shared_template or duplicated. Received: ${GAUSSIAN_RENDER_MODE}" >&2
  exit 1
fi
if [[ "$BATCH_IMAGE_RESOLUTION" != "native" && "$BATCH_IMAGE_RESOLUTION" != "640x480" ]]; then
  echo "[ERROR] BATCH_IMAGE_RESOLUTION must be native or 640x480. Received: ${BATCH_IMAGE_RESOLUTION}" >&2
  exit 1
fi
if ((NUM_VIEWS != 1)); then
  echo "[ERROR] batch_images full-runtime autotune supports only NUM_VIEWS=1." >&2
  exit 1
fi

SEARCH_MIN_BATCH_SIZE="$MIN_BATCH_SIZE"
if ((SEARCH_MIN_BATCH_SIZE > MAX_BATCH_SIZE)); then
  echo "[ERROR] Effective minimum batch size (${SEARCH_MIN_BATCH_SIZE}) exceeds MAX_BATCH_SIZE (${MAX_BATCH_SIZE})." >&2
  exit 1
fi

if ((${#cases[@]} == 0)); then
  read_cases_file "$CASES_FILE"
fi

MODE_DIR="$(mode_dir_for_current_config)"
CONFIG_DIR="$(config_dir_for_current_config)"
if ((RESULTS_ROOT_WAS_SET == 0)); then
  RESULTS_ROOT="${RESULTS_ROOT}/${CONFIG_DIR}"
fi
ATTEMPTED_MANIFEST="${RESULTS_ROOT}/attempted_candidates.csv"

mkdir -p "$RESULTS_ROOT"
printf "Case Name,Batch Size\n" > "$ATTEMPTED_MANIFEST"

for case_name in "${cases[@]}"; do
  echo "=== [full_runtime_best_throughput] searching case=${case_name} ==="

  declare -A tested_batches=()
  power_batches=()
  LAST_SUCCESSFUL_RUNS=0

  batch_size=1
  while ((batch_size < SEARCH_MIN_BATCH_SIZE)); do
    batch_size=$((batch_size * 2))
  done

  while ((batch_size <= MAX_BATCH_SIZE)); do
    run_candidate_if_needed "$case_name" "$batch_size"
    power_batches+=("$batch_size")

    if ((LAST_SUCCESSFUL_RUNS < MIN_SUCCESSES)); then
      echo "[STOP] ${case_name}: batch=${batch_size} had ${LAST_SUCCESSFUL_RUNS}/${NUM_RUNS} successful runs, below MIN_SUCCESSES=${MIN_SUCCESSES}; stopping power-of-two expansion."
      break
    fi

    batch_size=$((batch_size * 2))
  done

  best_power_batch=$(find_best_batch "$case_name" "${power_batches[@]}")
  if [[ -z "$best_power_batch" ]]; then
    echo "[WARN] ${case_name}: no successful power-of-two candidate; skipping dense refinement."
    continue
  fi

  refine_start="$SEARCH_MIN_BATCH_SIZE"
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
  if ((refine_start < SEARCH_MIN_BATCH_SIZE)); then
    refine_start="$SEARCH_MIN_BATCH_SIZE"
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
  if ((final_start < SEARCH_MIN_BATCH_SIZE)); then
    final_start="$SEARCH_MIN_BATCH_SIZE"
  fi
  if ((final_end > MAX_BATCH_SIZE)); then
    final_end="$MAX_BATCH_SIZE"
  fi

  echo "[FINAL] ${case_name}: best sampled batch=${final_best_batch}; dense testing ${final_start}..${final_end}."
  for ((batch_size=final_start; batch_size<=final_end; batch_size++)); do
    run_candidate_if_needed "$case_name" "$batch_size"
  done
done

aggregate_cmd=(
  python benchmarks/post-processing/aggregate_batched_full_runtime_best_throughput.py
  --results_root "$RESULTS_ROOT"
  --cases_file "$CASES_FILE"
  --render_mode "$RENDER_MODE"
  --gaussian_render_mode "$GAUSSIAN_RENDER_MODE"
  --batch_image_resolution "$BATCH_IMAGE_RESOLUTION"
  --sim_force_mode "$SIM_FORCE_MODE"
  --output_best "${RESULTS_ROOT}/best_throughput_table.csv"
  --output_candidates "${RESULTS_ROOT}/candidate_table.csv"
  --attempted_manifest "$ATTEMPTED_MANIFEST"
  --num_runs "$NUM_RUNS"
  --min_successes "$MIN_SUCCESSES"
)
if [[ -n "$BATCHED_RENDER_VARIANT" ]]; then
  aggregate_cmd+=(--batched_render_variant "$BATCHED_RENDER_VARIANT")
fi

"${aggregate_cmd[@]}" "${cases[@]}"

echo "[DONE] Best-throughput table: ${RESULTS_ROOT}/best_throughput_table.csv"
echo "[DONE] Candidate table: ${RESULTS_ROOT}/candidate_table.csv"
