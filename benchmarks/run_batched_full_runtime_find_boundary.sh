#!/usr/bin/env bash
set -u -o pipefail

CASES_FILE="${CASES_FILE:-data_config.csv}"
BASE_PATH="${BASE_PATH:-./data/different_types}"
GAUSSIAN_PATH="${GAUSSIAN_PATH:-./gaussian_output}"
PRUNED_GAUSSIAN_PATH="${PRUNED_GAUSSIAN_PATH:-./gaussian_output_pruned_policy_30_55}"
BG_IMG_PATH="${BG_IMG_PATH:-./data/bg.png}"

BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-64}"
GROWTH_NUM="${GROWTH_NUM:-3}"
GROWTH_DEN="${GROWTH_DEN:-2}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-16384}"
RETRIES="${RETRIES:-${NUM_RUNS:-1}}"
MIN_SUCCESSES="${MIN_SUCCESSES:-$RETRIES}"
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"
SIM_FORCE_MODE="${SIM_FORCE_MODE:-gather}"
BATCHED_RENDER_VARIANT_DEFAULT="${BATCHED_RENDER_VARIANT-batch_prune}"
BATCH_IMAGE_RESOLUTION_DEFAULT="${BATCH_IMAGE_RESOLUTION-640x480}"
OUT_CSV_WAS_SET=0
LOG_DIR_WAS_SET=0
RUN_ROOT_WAS_SET=0
if [[ -n "${OUT_CSV+x}" ]]; then
  OUT_CSV_WAS_SET=1
fi
if [[ -n "${LOG_DIR+x}" ]]; then
  LOG_DIR_WAS_SET=1
fi
if [[ -n "${RUN_ROOT+x}" ]]; then
  RUN_ROOT_WAS_SET=1
fi

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
      echo "Error: --batched_render_variant must be batch_original, batch_optimized, or batch_prune. Received: $1" >&2
      return 1
      ;;
  esac
}

validate_sim_force_mode() {
  case "$1" in
    gather|template_state_batched_atomic)
      ;;
    *)
      echo "Error: SIM_FORCE_MODE must be gather or template_state_batched_atomic. Received: $1" >&2
      return 1
      ;;
  esac
}

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_batched_full_runtime_find_boundary.sh [--batched_render_variant batch_original|batch_optimized|batch_prune] [--sim_force_mode gather|template_state_batched_atomic] [--render_mode instance|batch_images] [--gaussian_render_mode shared_template|duplicated] [--instance_id I] [--num_views N] [--batch_image_resolution native|640x480] [--save_video] [--save_batch_images] [--save_batch_grid] [case_name ...]

This benchmark finds the maximum full-runtime batch size each case can run:
  spring-mass + LBS + rendering + frame compositing

Defaults:
  render_mode=batch_images
  gaussian_render_mode=shared_template
  num_views=1
  BASE_BATCH_SIZE=64
  GROWTH_NUM=3
  GROWTH_DEN=2
  MAX_BATCH_SIZE=16384
  RETRIES=1
  MIN_SUCCESSES=RETRIES
  TIMEOUT_SEC=0
  batched_render_variant=batch_prune
  batch_image_resolution=640x480

Options:
  --render_mode MODE           instance or batch_images (default: batch_images).
  --gaussian_render_mode MODE  shared_template or duplicated (default: shared_template).
  --batched_render_variant VARIANT
                               Preset for batch_images rendering:
                               batch_original, batch_optimized, or batch_prune.
                               Deprecated aliases are accepted: baseline,
                               optimized, optimized_pruned.
  --instance_id I              Required when --render_mode instance.
  --num_views N                Number of camera views to generate (valid: 1, 2, 3; default: 1).
  --save_video                 Save PNG folders and MP4 videos for successful candidates.
  --save_batch_images          Save per-instance composited images for batch_images mode.
  --save_batch_grid            Save tiled batch preview images for batch_images mode.
  --batch_image_resolution SIZE
                               native or 640x480 (default: 640x480; batch_images only).
  --batch_grid_cols N          Number of columns in tiled batch previews (default: ceil(sqrt(batch_size))).

Environment overrides:
  CASES_FILE
  BASE_PATH
  GAUSSIAN_PATH
  PRUNED_GAUSSIAN_PATH
  BG_IMG_PATH
  BASE_BATCH_SIZE
  GROWTH_NUM
  GROWTH_DEN
  MAX_BATCH_SIZE
  RETRIES
  NUM_RUNS                     Used as RETRIES only when RETRIES is unset.
  MIN_SUCCESSES                Number of successful attempts required for a batch
                               to count as supported. Defaults to RETRIES.
  TIMEOUT_SEC                  0 disables timeout.
  SIM_FORCE_MODE               default: gather.
  BATCHED_RENDER_VARIANT       default: batch_prune.
  BATCH_IMAGE_RESOLUTION       default: 640x480.
  OUT_CSV                      default: results/batched_full_runtime_boundary/<config>/boundary.csv.
                                If set, used exactly as provided.
  LOG_DIR                      default: results/batched_full_runtime_boundary/<config>/logs.
                                If set, used exactly as provided.
  RUN_ROOT                     default: results/batched_full_runtime_boundary/<config>/runs.
                                If set, used exactly as provided.

Output layout:
  Default outputs are grouped by simulation/render config, e.g.:
    results/batched_full_runtime_boundary/sim_gather_render_batch_images_640x480_shared_template_batch_prune/
  Use --sim_force_mode template_state_batched_atomic to select the template-state
  batched atomic simulation path.
EOF
}

is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && (("$1" > 0))
}

is_nonnegative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

csv_escape() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

csv_write_row() {
  local first=1
  for value in "$@"; do
    if ((first == 0)); then
      printf ','
    fi
    csv_escape "$value"
    first=0
  done
  printf '\n'
}

join_by_space() {
  local IFS=" "
  echo "$*"
}

assoc_get() {
  local array_name="$1"
  local key="${2-}"
  local fallback="${3-}"
  if [[ -z "$key" ]]; then
    printf "%s" "$fallback"
    return 0
  fi

  local -n array_ref="$array_name"
  if [[ -v "array_ref[$key]" ]]; then
    printf "%s" "${array_ref[$key]}"
  else
    printf "%s" "$fallback"
  fi
}

next_hi() {
  local x="$1"
  local num="$2"
  local den="$3"
  echo $(((x * num + den - 1) / den))
}

config_dir_for_current_config() {
  local render_config=""
  if [[ "$render_mode" == "instance" ]]; then
    render_config="instance_${instance_id}_${gaussian_render_mode}"
  else
    render_config="batch_images_${batch_image_resolution}_${gaussian_render_mode}"
    if [[ -n "$batched_render_variant" ]]; then
      render_config="${render_config}_${batched_render_variant}"
    fi
  fi
  printf "sim_%s_render_%s\n" "$SIM_FORCE_MODE" "$render_config"
}

render_mode="batch_images"
gaussian_render_mode="shared_template"
batched_render_variant=""
instance_id=""
num_views=1
save_video=0
save_batch_images=0
save_batch_grid=0
batch_image_resolution=""
batch_grid_cols=""
cases=()
explicit_render_mode=0
explicit_gaussian_render_mode=0
explicit_batched_render_variant=0
explicit_batch_image_resolution=0

while (($# > 0)); do
  case "$1" in
    --render_mode)
      shift
      if (($# == 0)); then
        echo "Error: --render_mode requires a value." >&2
        usage >&2
        exit 1
      fi
      render_mode="$1"
      explicit_render_mode=1
      ;;
    --render_mode=*)
      render_mode="${1#*=}"
      explicit_render_mode=1
      ;;
    --gaussian_render_mode)
      shift
      if (($# == 0)); then
        echo "Error: --gaussian_render_mode requires a value." >&2
        usage >&2
        exit 1
      fi
      gaussian_render_mode="$1"
      explicit_gaussian_render_mode=1
      ;;
    --gaussian_render_mode=*)
      gaussian_render_mode="${1#*=}"
      explicit_gaussian_render_mode=1
      ;;
    --batched_render_variant|--batched-render-variant)
      shift
      if (($# == 0)); then
        echo "Error: --batched_render_variant requires a value." >&2
        usage >&2
        exit 1
      fi
      batched_render_variant="$1"
      explicit_batched_render_variant=1
      ;;
    --batched_render_variant=*|--batched-render-variant=*)
      batched_render_variant="${1#*=}"
      explicit_batched_render_variant=1
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
    --instance_id)
      shift
      if (($# == 0)); then
        echo "Error: --instance_id requires an integer value." >&2
        usage >&2
        exit 1
      fi
      instance_id="$1"
      ;;
    --instance_id=*)
      instance_id="${1#*=}"
      ;;
    --num_views)
      shift
      if (($# == 0)); then
        echo "Error: --num_views requires an integer value." >&2
        usage >&2
        exit 1
      fi
      num_views="$1"
      ;;
    --num_views=*)
      num_views="${1#*=}"
      ;;
    --save_video)
      save_video=1
      ;;
    --save_batch_images)
      save_batch_images=1
      ;;
    --save_batch_grid)
      save_batch_grid=1
      ;;
    --batch_image_resolution)
      shift
      if (($# == 0)); then
        echo "Error: --batch_image_resolution requires a value." >&2
        usage >&2
        exit 1
      fi
      batch_image_resolution="$1"
      explicit_batch_image_resolution=1
      ;;
    --batch_image_resolution=*)
      batch_image_resolution="${1#*=}"
      explicit_batch_image_resolution=1
      ;;
    --batch_grid_cols)
      shift
      if (($# == 0)); then
        echo "Error: --batch_grid_cols requires an integer value." >&2
        usage >&2
        exit 1
      fi
      batch_grid_cols="$1"
      ;;
    --batch_grid_cols=*)
      batch_grid_cols="${1#*=}"
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

if ((explicit_batched_render_variant == 0)); then
  if [[ -n "${BATCHED_RENDER_VARIANT+x}" ]] || ((explicit_render_mode == 0 && explicit_gaussian_render_mode == 0)); then
    batched_render_variant="$BATCHED_RENDER_VARIANT_DEFAULT"
  fi
fi

if ((explicit_batch_image_resolution == 0)); then
  if [[ -n "${BATCH_IMAGE_RESOLUTION+x}" || "$render_mode" == "batch_images" ]]; then
    batch_image_resolution="$BATCH_IMAGE_RESOLUTION_DEFAULT"
  else
    batch_image_resolution="native"
  fi
fi

if [[ "$render_mode" != "instance" && "$render_mode" != "batch_images" ]]; then
  echo "Error: --render_mode must be 'instance' or 'batch_images'. Received: ${render_mode}" >&2
  exit 1
fi

if [[ "$gaussian_render_mode" != "shared_template" && "$gaussian_render_mode" != "duplicated" ]]; then
  echo "Error: --gaussian_render_mode must be 'shared_template' or 'duplicated'. Received: ${gaussian_render_mode}" >&2
  exit 1
fi

if ! validate_sim_force_mode "$SIM_FORCE_MODE"; then
  exit 1
fi

if [[ -n "$batched_render_variant" ]]; then
  if ! batched_render_variant="$(normalize_batched_render_variant "$batched_render_variant")"; then
    exit 1
  fi
  case "$batched_render_variant" in
    batch_original)
      render_mode="batch_images"
      gaussian_render_mode="duplicated"
      GAUSSIAN_PATH="./gaussian_output"
      instance_id=""
      num_views=1
      ;;
    batch_optimized)
      render_mode="batch_images"
      gaussian_render_mode="shared_template"
      GAUSSIAN_PATH="./gaussian_output"
      instance_id=""
      num_views=1
      ;;
    batch_prune)
      render_mode="batch_images"
      gaussian_render_mode="shared_template"
      GAUSSIAN_PATH="$PRUNED_GAUSSIAN_PATH"
      instance_id=""
      num_views=1
      ;;
  esac
fi

if ! is_positive_int "$num_views" || ((num_views > 3)); then
  echo "Error: --num_views must be an integer between 1 and 3. Received: ${num_views}" >&2
  exit 1
fi

if [[ "$batch_image_resolution" != "native" && "$batch_image_resolution" != "640x480" ]]; then
  echo "Error: --batch_image_resolution must be 'native' or '640x480'. Received: ${batch_image_resolution}" >&2
  exit 1
fi

if [[ -n "$batch_grid_cols" ]] && ! is_positive_int "$batch_grid_cols"; then
  echo "Error: --batch_grid_cols must be a positive integer. Received: ${batch_grid_cols}" >&2
  exit 1
fi

if [[ "$render_mode" != "batch_images" && ( "$save_batch_images" -eq 1 || "$save_batch_grid" -eq 1 ) ]]; then
  echo "Error: --save_batch_images and --save_batch_grid can only be used with --render_mode batch_images." >&2
  exit 1
fi

if [[ "$render_mode" != "batch_images" && "$batch_image_resolution" != "native" ]]; then
  echo "Error: --batch_image_resolution 640x480 can only be used with --render_mode batch_images." >&2
  exit 1
fi

if [[ "$render_mode" == "batch_images" ]]; then
  if ((num_views != 1)); then
    echo "Error: --render_mode batch_images currently supports only --num_views 1." >&2
    exit 1
  fi
  if [[ -n "$instance_id" ]]; then
    echo "Error: --instance_id can only be used with --render_mode instance." >&2
    exit 1
  fi
fi

for value_name in BASE_BATCH_SIZE GROWTH_NUM GROWTH_DEN MAX_BATCH_SIZE RETRIES MIN_SUCCESSES TIMEOUT_SEC; do
  value="${!value_name}"
  if ! is_nonnegative_int "$value"; then
    echo "Error: ${value_name} must be a nonnegative integer. Received: ${value}" >&2
    exit 1
  fi
done

if ((BASE_BATCH_SIZE < 1)); then
  echo "Error: BASE_BATCH_SIZE must be positive. Received: ${BASE_BATCH_SIZE}" >&2
  exit 1
fi
if ((GROWTH_NUM < 1 || GROWTH_DEN < 1)); then
  echo "Error: GROWTH_NUM and GROWTH_DEN must be positive." >&2
  exit 1
fi
if ((MAX_BATCH_SIZE < BASE_BATCH_SIZE)); then
  echo "Error: MAX_BATCH_SIZE must be >= BASE_BATCH_SIZE." >&2
  exit 1
fi
if ((RETRIES < 1)); then
  echo "Error: RETRIES must be positive. Received: ${RETRIES}" >&2
  exit 1
fi
if ((MIN_SUCCESSES < 1 || MIN_SUCCESSES > RETRIES)); then
  echo "Error: MIN_SUCCESSES must be between 1 and RETRIES (${RETRIES}). Received: ${MIN_SUCCESSES}" >&2
  exit 1
fi

if [[ "$render_mode" == "instance" ]]; then
  if [[ -z "$instance_id" ]]; then
    echo "Error: --instance_id is required when --render_mode instance." >&2
    exit 1
  fi
  if ! is_nonnegative_int "$instance_id"; then
    echo "Error: --instance_id must be a nonnegative integer. Received: ${instance_id}" >&2
    exit 1
  fi
else
  if [[ -n "$instance_id" ]]; then
    echo "Error: --instance_id can only be used with --render_mode instance." >&2
    exit 1
  fi
fi

if ((${#cases[@]} == 0)); then
  read_cases_file "$CASES_FILE"
fi

if [[ "$render_mode" == "instance" ]]; then
  mode_dir="instance_${instance_id}_${gaussian_render_mode}"
elif [[ "$render_mode" == "batch_images" ]]; then
  if [[ "$batch_image_resolution" == "native" ]]; then
    mode_dir="batch_images_${gaussian_render_mode}"
  else
    mode_dir="batch_images_${batch_image_resolution}_${gaussian_render_mode}"
  fi
fi
if [[ -n "$batched_render_variant" ]]; then
  mode_dir="${mode_dir}_${batched_render_variant}"
fi
if [[ "$SIM_FORCE_MODE" != "gather" ]]; then
  mode_dir="${mode_dir}_sim_${SIM_FORCE_MODE}"
fi

CONFIG_DIR="$(config_dir_for_current_config)"
DEFAULT_OUTPUT_ROOT="results/batched_full_runtime_boundary/${CONFIG_DIR}"
if ((OUT_CSV_WAS_SET == 0)); then
  OUT_CSV="${DEFAULT_OUTPUT_ROOT}/boundary.csv"
fi
if ((LOG_DIR_WAS_SET == 0)); then
  LOG_DIR="${DEFAULT_OUTPUT_ROOT}/logs"
fi
if ((RUN_ROOT_WAS_SET == 0)); then
  RUN_ROOT="${DEFAULT_OUTPUT_ROOT}/runs"
fi

mkdir -p "$(dirname "$OUT_CSV")" "$LOG_DIR" "$RUN_ROOT"

csv_write_row \
  "Case Name" \
  "Gaussian Render Mode" \
  "Sim Force Mode" \
  "Render Mode" \
  "Max OK Batch Size" \
  "Min Failed Batch Size" \
  "Status" \
  "searched_batch_sizes" \
  "failed_batch_sizes" \
  "max_ok_average_fps" \
  "max_ok_average_throughput" \
  "failure_reason" >"$OUT_CSV"

validate_summary() {
  local summary_path="$1"
  python - "$summary_path" <<'PY'
import json
import sys

summary_path = sys.argv[1]
try:
    with open(summary_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)
except Exception as exc:
    print(f"missing_or_invalid_summary:{exc}")
    sys.exit(1)

frames = int(metrics.get("frames_used_for_stats") or 0)
fps = float(metrics.get("average_fps") or 0.0)
throughput = float(metrics.get("average_throughput") or 0.0)
if frames > 0 and fps > 0.0 and throughput > 0.0:
    print(f"{fps:.6f} {throughput:.6f}")
    sys.exit(0)

print(
    "zero_or_invalid_metrics:"
    f"frames_used_for_stats={frames},average_fps={fps},average_throughput={throughput}"
)
sys.exit(1)
PY
}

extract_failure_reason() {
  local log_path="$1"
  local fallback="$2"
  local reason=""
  if [[ -f "$log_path" ]]; then
    reason="$(grep -E 'OutOfMemoryError|RuntimeError: Failed to allocate|CUDA out of memory|Traceback|Error:|Exception|RuntimeError:' "$log_path" | tail -n 1 || true)"
  fi
  if [[ -z "$reason" ]]; then
    reason="$fallback"
  fi
  reason="${reason//$'\r'/ }"
  reason="${reason//$'\n'/ }"
  echo "$reason"
}

run_candidate_uncached() {
  local case_name="$1"
  local batch_size="$2"
  local fps_values=()
  local throughput_values=()
  local failure_reasons=()
  local success_count=0

  if [[ "$render_mode" == "instance" ]] && ((instance_id >= batch_size)); then
    LAST_FAILURE_REASON="instance_id ${instance_id} is outside batch size ${batch_size}"
    return 1
  fi

  for ((try_idx = 1; try_idx <= RETRIES; try_idx++)); do
    local try_name
    try_name="$(printf "try_%02d" "$try_idx")"
    local log_path="${LOG_DIR}/${case_name}__batch_${batch_size}__${try_name}.log"
    local output_dir="${RUN_ROOT}/${case_name}/batch_${batch_size}/${mode_dir}"
    if ((RETRIES > 1)); then
      output_dir="${output_dir}/${try_name}"
    fi

    mkdir -p "$output_dir" "$(dirname "$log_path")"
    rm -f "${output_dir}/performance_summary.json" "${output_dir}/performance_summary.txt"

    local cmd=(
      python benchmarks/run_batched_full_runtime_case.py
      --base_path "$BASE_PATH"
      --gaussian_path "$GAUSSIAN_PATH"
      --bg_img_path "$BG_IMG_PATH"
      --case_name "$case_name"
      --batch_size "$batch_size"
      --render_mode "$render_mode"
      --gaussian_render_mode "$gaussian_render_mode"
      --sim_force_mode "$SIM_FORCE_MODE"
      --num_views "$num_views"
      --batch_image_resolution "$batch_image_resolution"
      --output_dir "$output_dir"
    )

    if [[ "$render_mode" == "instance" ]]; then
      cmd+=(--instance_id "$instance_id")
    fi
    if ((save_video == 1)); then
      cmd+=(--save_video)
    fi
    if ((save_batch_images == 1)); then
      cmd+=(--save_batch_images)
    fi
    if ((save_batch_grid == 1)); then
      cmd+=(--save_batch_grid)
    fi
    if [[ -n "$batch_grid_cols" ]]; then
      cmd+=(--batch_grid_cols "$batch_grid_cols")
    fi
    if [[ -n "$batched_render_variant" ]]; then
      cmd+=(--batched_render_variant "$batched_render_variant")
      cmd+=(--pruned_gaussian_path "$PRUNED_GAUSSIAN_PATH")
    fi

    local rc
    if ((TIMEOUT_SEC > 0)); then
      timeout "${TIMEOUT_SEC}s" "${cmd[@]}" >"$log_path" 2>&1
      rc=$?
    else
      "${cmd[@]}" >"$log_path" 2>&1
      rc=$?
    fi

    if ((rc != 0)); then
      failure_reasons+=("try_${try_idx}: $(extract_failure_reason "$log_path" "runner_exit_code=${rc}")")
      continue
    fi

    local metrics_output
    metrics_output="$(validate_summary "${output_dir}/performance_summary.json" 2>&1)"
    rc=$?
    if ((rc != 0)); then
      failure_reasons+=("try_${try_idx}: $(extract_failure_reason "$log_path" "$metrics_output")")
      continue
    fi

    success_count=$((success_count + 1))
    fps_values+=("$(awk '{print $1}' <<<"$metrics_output")")
    throughput_values+=("$(awk '{print $2}' <<<"$metrics_output")")
  done

  LAST_SUCCESSFUL_ATTEMPTS="$success_count"
  if ((success_count < MIN_SUCCESSES)); then
    local failure_count="${#failure_reasons[@]}"
    local last_failure_reason=""
    if ((failure_count > 0)); then
      last_failure_reason="${failure_reasons[$((failure_count - 1))]}"
    fi
    LAST_FAILURE_REASON="successes=${success_count}/${RETRIES} below MIN_SUCCESSES=${MIN_SUCCESSES}"
    if [[ -n "$last_failure_reason" ]]; then
      LAST_FAILURE_REASON="${LAST_FAILURE_REASON}; ${last_failure_reason}"
    fi
    return 1
  fi

  read -r LAST_AVERAGE_FPS LAST_AVERAGE_THROUGHPUT < <(
    python - "${fps_values[@]}" -- "${throughput_values[@]}" <<'PY'
import sys

separator = sys.argv.index("--")
fps_values = [float(value) for value in sys.argv[1:separator]]
throughput_values = [float(value) for value in sys.argv[separator + 1:]]
print(
    f"{sum(fps_values) / len(fps_values):.6f} "
    f"{sum(throughput_values) / len(throughput_values):.6f}"
)
PY
  )
  LAST_FAILURE_REASON=""
  return 0
}

find_boundary_for_case() {
  local case_name="$1"
  local searched=()
  local failed=()
  local -A result_cache=()
  local -A fps_cache=()
  local -A throughput_cache=()
  local -A reason_cache=()

  run_candidate() {
    local batch_size="$1"
    if [[ -n "${result_cache[$batch_size]+x}" ]]; then
      LAST_AVERAGE_FPS="$(assoc_get fps_cache "$batch_size")"
      LAST_AVERAGE_THROUGHPUT="$(assoc_get throughput_cache "$batch_size")"
      LAST_FAILURE_REASON="$(assoc_get reason_cache "$batch_size")"
      [[ "${result_cache[$batch_size]}" == "ok" ]]
      return $?
    fi

    searched+=("$batch_size")
    echo "  try batch=${batch_size} ..."
    if run_candidate_uncached "$case_name" "$batch_size"; then
      result_cache[$batch_size]="ok"
      fps_cache[$batch_size]="$LAST_AVERAGE_FPS"
      throughput_cache[$batch_size]="$LAST_AVERAGE_THROUGHPUT"
      reason_cache[$batch_size]=""
      echo "    OK successes=${LAST_SUCCESSFUL_ATTEMPTS}/${RETRIES} fps=${LAST_AVERAGE_FPS} throughput=${LAST_AVERAGE_THROUGHPUT}"
      return 0
    fi

    result_cache[$batch_size]="fail"
    fps_cache[$batch_size]=""
    throughput_cache[$batch_size]=""
    reason_cache[$batch_size]="$LAST_FAILURE_REASON"
    failed+=("$batch_size")
    echo "    FAIL ${LAST_FAILURE_REASON}"
    return 1
  }

  bisect_boundary() {
    local lo="$1"
    local hi="$2"
    while ((lo + 1 < hi)); do
      local mid=$(((lo + hi) / 2))
      if run_candidate "$mid"; then
        lo="$mid"
      else
        hi="$mid"
      fi
    done
    BISECT_MAX_OK="$lo"
    BISECT_MIN_FAIL="$hi"
  }

  echo "=== Boundary search: ${case_name} (${gaussian_render_mode}, ${render_mode}) ==="

  local max_ok=""
  local min_fail=""
  local status=""
  local failure_reason=""

  if run_candidate "$BASE_BATCH_SIZE"; then
    local lo="$BASE_BATCH_SIZE"
    local hi="$BASE_BATCH_SIZE"

    while true; do
      lo="$hi"
      hi="$(next_hi "$hi" "$GROWTH_NUM" "$GROWTH_DEN")"
      if ((hi <= lo)); then
        hi=$((lo + 1))
      fi

      if ((hi > MAX_BATCH_SIZE)); then
        if [[ "$lo" != "$MAX_BATCH_SIZE" ]]; then
          if run_candidate "$MAX_BATCH_SIZE"; then
            max_ok="$MAX_BATCH_SIZE"
            min_fail=""
            status="reached_max_batch_size"
            failure_reason=""
          else
            bisect_boundary "$lo" "$MAX_BATCH_SIZE"
            max_ok="$BISECT_MAX_OK"
            min_fail="$BISECT_MIN_FAIL"
            status="boundary_found"
            failure_reason="$(assoc_get reason_cache "$min_fail")"
          fi
        else
          max_ok="$MAX_BATCH_SIZE"
          min_fail=""
          status="reached_max_batch_size"
          failure_reason=""
        fi
        break
      fi

      if ! run_candidate "$hi"; then
        bisect_boundary "$lo" "$hi"
        max_ok="$BISECT_MAX_OK"
        min_fail="$BISECT_MIN_FAIL"
        status="boundary_found"
        failure_reason="$(assoc_get reason_cache "$min_fail")"
        break
      fi
    done
  else
    local base_failure_reason="$LAST_FAILURE_REASON"
    if run_candidate 1; then
      bisect_boundary 1 "$BASE_BATCH_SIZE"
      max_ok="$BISECT_MAX_OK"
      min_fail="$BISECT_MIN_FAIL"
      status="boundary_found_below_base"
      failure_reason="$(assoc_get reason_cache "$min_fail" "$base_failure_reason")"
    else
      max_ok="0"
      min_fail="1"
      status="batch_1_failed"
      failure_reason="$LAST_FAILURE_REASON"
    fi
  fi

  local max_ok_fps
  local max_ok_throughput
  max_ok_fps="$(assoc_get fps_cache "$max_ok")"
  max_ok_throughput="$(assoc_get throughput_cache "$max_ok")"
  local searched_text failed_text
  searched_text="$(join_by_space "${searched[@]}")"
  failed_text="$(join_by_space "${failed[@]}")"

  csv_write_row \
    "$case_name" \
    "$gaussian_render_mode" \
    "$SIM_FORCE_MODE" \
    "$render_mode" \
    "$max_ok" \
    "$min_fail" \
    "$status" \
    "$searched_text" \
    "$failed_text" \
    "$max_ok_fps" \
    "$max_ok_throughput" \
    "$failure_reason" >>"$OUT_CSV"

  echo "=== Result: ${case_name} max_ok=${max_ok} min_fail=${min_fail:-none} status=${status} ==="
}

for case_name in "${cases[@]}"; do
  find_boundary_for_case "$case_name"
done

echo "Done. Results written to: ${OUT_CSV}"
echo "Logs in: ${LOG_DIR}"
echo "Candidate summaries in: ${RUN_ROOT}"
