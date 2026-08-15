#!/usr/bin/env bash
set -u

NUM_RUNS="${NUM_RUNS:-3}"
FORCE_RERUN="${FORCE_RERUN:-0}"
CASES_FILE="${CASES_FILE:-data_config.csv}"
BASE_PATH="${BASE_PATH:-./data/different_types}"
GAUSSIAN_PATH="${GAUSSIAN_PATH:-./gaussian_output}"
BG_IMG_PATH="${BG_IMG_PATH:-./data/bg.png}"

DEFAULT_SCALING_BATCH_SIZES=(1 32 64 128 256 512)
DEFAULT_NCU_BIN="/home/yihanp2/cuda-12.1/bin/ncu"
NCU_PROFILE_FRAME_STRIDE="${NCU_PROFILE_FRAME_STRIDE:-}"
NCU_PROFILE_MAX_FRAMES="${NCU_PROFILE_MAX_FRAMES:-3}"
NCU_PROFILE_NVTX_NAME="${NCU_PROFILE_NVTX_NAME:-sim_lbs_profile_frame}"
NCU_TARGET_PROCESSES="${NCU_TARGET_PROCESSES:-application-only}"
NCU_METRICS="gpu__time_duration.sum,dram__bytes.sum,dram__bytes.sum.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

read_cases_file() {
  mapfile -t cases < <(awk -F, 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); if ($1 != "") print $1}' "$1")
}

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_sim_lbs_batch_scaling.sh --batch_sizes 1 2 4 8 [case_name ...]
  bash benchmarks/run_sim_lbs_batch_scaling.sh --scaling-analysis [--timing-only|--ncu-only|--summarize-only] [--batch-sizes "1 32 64 128 256 512"] [--output-dir DIR] [--ncu-bin PATH] [case_name]

This benchmark measures sim_lbs only:
  spring-mass + LBS

Scaling-analysis outputs:
  batch_scaling_sim_lbs.csv
  batch_scaling_sim_lbs.pdf
  batch_scaling_sim_lbs.png
  batch_scaling_sim_lbs_bottleneck.pdf
  batch_scaling_sim_lbs_bottleneck.png
  batch_scaling_sim_lbs_summary.txt
  batch_scaling_sim_lbs_profile_commands.sh
  batch_scaling_sim_lbs_metadata.json
  ncu/batch_<B>.csv

Environment overrides:
  NUM_RUNS
  FORCE_RERUN
  CASES_FILE
  BASE_PATH
  GAUSSIAN_PATH
  BG_IMG_PATH
  NCU_PROFILE_FRAME_STRIDE
  NCU_PROFILE_MAX_FRAMES
  NCU_PROFILE_NVTX_NAME
  NCU_TARGET_PROCESSES
EOF
}

add_batch_values() {
  local raw_value="$1"
  local token
  for token in $raw_value; do
    batch_sizes+=("$token")
  done
}

validate_batch_sizes() {
  local batch_size
  for batch_size in "${batch_sizes[@]}"; do
    if ! [[ "$batch_size" =~ ^[0-9]+$ ]] || ((batch_size < 1)); then
      echo "[ERROR] Invalid batch size: ${batch_size}" >&2
      exit 1
    fi
  done
}

set_pass_mode() {
  local requested="$1"
  if ((pass_mode_set)); then
    echo "[ERROR] Use only one of --timing-only, --ncu-only, or --summarize-only." >&2
    exit 1
  fi
  pass_mode="$requested"
  pass_mode_set=1
}

resolve_ncu_bin() {
  local override="$1"
  if [[ -n "$override" ]]; then
    if [[ ! -x "$override" ]]; then
      echo "[ERROR] --ncu-bin is not executable: ${override}" >&2
      exit 1
    fi
    printf "%s\n" "$override"
    return 0
  fi

  if [[ -x "$DEFAULT_NCU_BIN" ]]; then
    printf "%s\n" "$DEFAULT_NCU_BIN"
    return 0
  fi

  command -v ncu || true
}

classify_failure_status() {
  local log_path="$1"
  if [[ -f "$log_path" ]] && grep -qiE "out of memory|cuda error: out of memory|cuda out of memory|cublas_status_alloc_failed|cusparse_status_alloc_failed|std::bad_alloc" "$log_path"; then
    echo "oom"
  else
    echo "failed"
  fi
}

append_attempt() {
  local manifest="$1"
  local run_type="$2"
  local run_name="$3"
  local batch_size="$4"
  local case_name="$5"
  local status="$6"
  local output_dir="$7"
  local log_path="$8"
  printf "%s,%s,%s,%s,%s,%s,%s\n" \
    "$run_type" "$run_name" "$batch_size" "$case_name" "$status" "$output_dir" "$log_path" \
    >> "$manifest"
}

init_attempt_manifest() {
  local manifest="$1"
  if [[ "$pass_mode" == "full" || ! -f "$manifest" ]]; then
    printf "run_type,run_name,batch_size,case_name,status,output_dir,log_path\n" > "$manifest"
  fi
}

count_existing_timing_metrics() {
  local results_root="$1"
  local batch_size="$2"
  local case_name="$3"
  find "$results_root" -path "*/batch_${batch_size}/${case_name}/scaling_metrics.json" -type f 2>/dev/null | wc -l | tr -d ' '
}

run_summarizer() {
  local results_root="$1"
  local ncu_bin="$2"
  local case_name="$3"
  local summarize_args

  summarize_args=(
    python benchmarks/post-processing/summarize_sim_lbs_scaling_analysis.py
    --results_root "$results_root"
    --case_name "$case_name"
    --base_path "$BASE_PATH"
    --gaussian_path "$GAUSSIAN_PATH"
    --bg_img_path "$BG_IMG_PATH"
    --ncu_bin "$ncu_bin"
    --ncu_profile_max_frames "$NCU_PROFILE_MAX_FRAMES"
    --ncu_profile_nvtx_name "$NCU_PROFILE_NVTX_NAME"
    --ncu_target_processes "$NCU_TARGET_PROCESSES"
    --ncu_metrics "$NCU_METRICS"
    --script_path "benchmarks/run_sim_lbs_batch_scaling.sh"
    --batch_sizes "${batch_sizes[@]}"
  )
  if [[ -n "$NCU_PROFILE_FRAME_STRIDE" ]]; then
    summarize_args+=(--ncu_profile_frame_stride "$NCU_PROFILE_FRAME_STRIDE")
  fi
  "${summarize_args[@]}"
}

run_legacy_batch_scaling() {
  local results_root="$1"
  local run_idx run_name batch_size case_name output_dir log_path

  mkdir -p "$results_root"

  for ((run_idx=1; run_idx<=NUM_RUNS; run_idx++)); do
    run_name=$(printf "run_%02d" "$run_idx")
    mkdir -p "${results_root}/logs/${run_name}"

    for batch_size in "${batch_sizes[@]}"; do
      mkdir -p "${results_root}/logs/${run_name}/batch_${batch_size}"

      for case_name in "${cases[@]}"; do
        output_dir="${results_root}/${run_name}/batch_${batch_size}/${case_name}"
        log_path="${results_root}/logs/${run_name}/batch_${batch_size}/${case_name}.log"

        echo "=== [batch_scaling] ${run_name} :: batch=${batch_size} :: ${case_name} ==="
        if python benchmarks/run_sim_lbs_batch_scaling_case.py \
          --base_path "$BASE_PATH" \
          --gaussian_path "$GAUSSIAN_PATH" \
          --bg_img_path "$BG_IMG_PATH" \
          --case_name "$case_name" \
          --batch_size "$batch_size" \
          --output_dir "$output_dir" \
          >"$log_path" 2>&1; then
          echo "[OK] ${case_name} (${run_name}, batch=${batch_size})"
        else
          echo "[FAIL] ${case_name} (${run_name}, batch=${batch_size})"
        fi
      done
    done
  done

  python benchmarks/post-processing/aggregate_sim_lbs_batch_scaling.py \
    --results_root "$results_root" \
    --cases_file "$CASES_FILE" \
    --output_table "${results_root}/batch_scaling_table.csv" \
    --output_overall "${results_root}/batch_scaling_overall.csv" \
    "${cases[@]}"
}

run_scaling_analysis() {
  local results_root="$1"
  local ncu_bin="$2"
  local case_name="${cases[0]}"
  local attempts_manifest="${results_root}/batch_scaling_sim_lbs_attempts.csv"
  local run_idx run_name batch_size output_dir log_path status successful_runs
  local ncu_output_dir ncu_log_path ncu_csv_path
  local ncu_profile_args existing_timing_count

  mkdir -p "$results_root" "${results_root}/logs" "${results_root}/ncu"
  init_attempt_manifest "$attempts_manifest"

  ncu_profile_args=(
    --ncu-profile-loop
    --ncu-profile-max-frames "$NCU_PROFILE_MAX_FRAMES"
    --ncu-profile-nvtx-name "$NCU_PROFILE_NVTX_NAME"
  )
  if [[ -n "$NCU_PROFILE_FRAME_STRIDE" ]]; then
    ncu_profile_args+=(--ncu-profile-frame-stride "$NCU_PROFILE_FRAME_STRIDE")
  fi

  for batch_size in "${batch_sizes[@]}"; do
    successful_runs=$(count_existing_timing_metrics "$results_root" "$batch_size" "$case_name")

    if [[ "$pass_mode" == "full" || "$pass_mode" == "timing" ]]; then
      successful_runs=0
      for ((run_idx=1; run_idx<=NUM_RUNS; run_idx++)); do
        run_name=$(printf "run_%02d" "$run_idx")
        output_dir="${results_root}/${run_name}/batch_${batch_size}/${case_name}"
        log_path="${results_root}/logs/${run_name}/batch_${batch_size}/${case_name}.log"
        mkdir -p "$output_dir" "$(dirname "$log_path")"

        if [[ "$FORCE_RERUN" != "1" && -f "${output_dir}/scaling_metrics.json" ]]; then
          successful_runs=$((successful_runs + 1))
          append_attempt "$attempts_manifest" "timing" "$run_name" "$batch_size" "$case_name" "skipped_existing" "$output_dir" "$log_path"
          echo "[SKIP] Existing timing metrics (${case_name}, ${run_name}, batch=${batch_size})"
          continue
        fi

        rm -f "${output_dir}/scaling_metrics.json" "${output_dir}/performance_summary.txt"
        echo "=== [scaling_analysis:timing] ${run_name} :: batch=${batch_size} :: ${case_name} ==="
        if python benchmarks/run_sim_lbs_batch_scaling_case.py \
          --base_path "$BASE_PATH" \
          --gaussian_path "$GAUSSIAN_PATH" \
          --bg_img_path "$BG_IMG_PATH" \
          --case_name "$case_name" \
          --batch_size "$batch_size" \
          --output_dir "$output_dir" \
          --scaling-analysis \
          >"$log_path" 2>&1; then
          successful_runs=$((successful_runs + 1))
          append_attempt "$attempts_manifest" "timing" "$run_name" "$batch_size" "$case_name" "ok" "$output_dir" "$log_path"
          echo "[OK] ${case_name} (${run_name}, batch=${batch_size})"
        else
          status=$(classify_failure_status "$log_path")
          append_attempt "$attempts_manifest" "timing" "$run_name" "$batch_size" "$case_name" "$status" "$output_dir" "$log_path"
          echo "[FAIL] ${case_name} (${run_name}, batch=${batch_size}, status=${status})"
        fi
      done
    fi

    if [[ "$pass_mode" == "timing" || "$pass_mode" == "summarize" ]]; then
      continue
    fi

    existing_timing_count=$(count_existing_timing_metrics "$results_root" "$batch_size" "$case_name")
    if ((existing_timing_count < 1)); then
      echo "[SKIP] NCU skipped for batch=${batch_size}; no successful timing run."
      continue
    fi

    ncu_output_dir="${results_root}/ncu_runs/batch_${batch_size}/${case_name}"
    ncu_log_path="${results_root}/logs/ncu/batch_${batch_size}.log"
    ncu_csv_path="${results_root}/ncu/batch_${batch_size}.csv"
    mkdir -p "$ncu_output_dir" "$(dirname "$ncu_log_path")" "$(dirname "$ncu_csv_path")"

    if [[ "$FORCE_RERUN" != "1" && -f "$ncu_csv_path" && -f "${ncu_output_dir}/ncu_profile_metrics.json" ]]; then
      append_attempt "$attempts_manifest" "ncu" "ncu" "$batch_size" "$case_name" "skipped_existing" "$ncu_output_dir" "$ncu_log_path"
      echo "[SKIP] Existing NCU metrics (${case_name}, batch=${batch_size})"
      continue
    fi

    rm -f "$ncu_csv_path" "${ncu_output_dir}/scaling_metrics.json" "${ncu_output_dir}/ncu_profile_metrics.json"

    if [[ -z "$ncu_bin" ]]; then
      append_attempt "$attempts_manifest" "ncu" "ncu" "$batch_size" "$case_name" "unavailable" "$ncu_output_dir" "$ncu_log_path"
      echo "[WARN] NCU unavailable; hardware counters will be blank for batch=${batch_size}."
      continue
    fi

    echo "=== [scaling_analysis:ncu] batch=${batch_size} :: ${case_name} ==="
    if "$ncu_bin" \
      --target-processes "$NCU_TARGET_PROCESSES" \
      --profile-from-start off \
      --nvtx \
      --nvtx-include "$NCU_PROFILE_NVTX_NAME" \
      --csv \
      --page raw \
      --print-units base \
      --print-fp \
      --metrics "$NCU_METRICS" \
      --log-file "$ncu_csv_path" \
      python benchmarks/run_sim_lbs_batch_scaling_case.py \
        --base_path "$BASE_PATH" \
        --gaussian_path "$GAUSSIAN_PATH" \
        --bg_img_path "$BG_IMG_PATH" \
        --case_name "$case_name" \
        --batch_size "$batch_size" \
        --output_dir "$ncu_output_dir" \
        --scaling-analysis \
        "${ncu_profile_args[@]}" \
        >"$ncu_log_path" 2>&1; then
      append_attempt "$attempts_manifest" "ncu" "ncu" "$batch_size" "$case_name" "ok" "$ncu_output_dir" "$ncu_log_path"
      echo "[OK] NCU (${case_name}, batch=${batch_size})"
    else
      status=$(classify_failure_status "$ncu_log_path")
      append_attempt "$attempts_manifest" "ncu" "ncu" "$batch_size" "$case_name" "$status" "$ncu_output_dir" "$ncu_log_path"
      echo "[FAIL] NCU (${case_name}, batch=${batch_size}, status=${status})"
    fi
  done

  run_summarizer "$results_root" "$ncu_bin" "$case_name"

  echo "[DONE] Scaling CSV: ${results_root}/batch_scaling_sim_lbs.csv"
  echo "[DONE] Scaling summary: ${results_root}/batch_scaling_sim_lbs_summary.txt"
}

batch_sizes=()
cases=()
explicit_case_count=0
scaling_analysis=0
pass_mode="full"
pass_mode_set=0
output_root=""
ncu_bin_override=""

while (($# > 0)); do
  case "$1" in
    --batch_sizes|--batch-sizes)
      shift
      while (($# > 0)); do
        case "$1" in
          --*)
            break
            ;;
          *)
            if [[ "$1" =~ ^[0-9]+$ || "$1" =~ [[:space:]] ]]; then
              add_batch_values "$1"
              shift
            else
              break
            fi
            ;;
        esac
      done
      ;;
    --scaling-analysis)
      scaling_analysis=1
      shift
      ;;
    --timing-only)
      set_pass_mode "timing"
      shift
      ;;
    --ncu-only)
      set_pass_mode "ncu"
      shift
      ;;
    --summarize-only)
      set_pass_mode "summarize"
      shift
      ;;
    --output-dir)
      shift
      if (($# < 1)); then
        echo "[ERROR] --output-dir requires a value." >&2
        exit 1
      fi
      output_root="$1"
      shift
      ;;
    --ncu-bin)
      shift
      if (($# < 1)); then
        echo "[ERROR] --ncu-bin requires a value." >&2
        exit 1
      fi
      ncu_bin_override="$1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      cases+=("$1")
      explicit_case_count=$((explicit_case_count + 1))
      shift
      ;;
  esac
done

if ((pass_mode_set)) && ((scaling_analysis == 0)); then
  echo "[ERROR] --timing-only, --ncu-only, and --summarize-only require --scaling-analysis." >&2
  exit 1
fi

if ((scaling_analysis)); then
  if ((${#batch_sizes[@]} == 0)); then
    batch_sizes=("${DEFAULT_SCALING_BATCH_SIZES[@]}")
  fi
else
  if ((${#batch_sizes[@]} == 0)); then
    echo "[ERROR] --batch_sizes is required." >&2
    usage >&2
    exit 1
  fi
fi

validate_batch_sizes

if ((${#cases[@]} == 0)); then
  read_cases_file "$CASES_FILE"
fi

if ((${#cases[@]} == 0)); then
  echo "[ERROR] No cases provided and ${CASES_FILE} did not contain any cases." >&2
  exit 1
fi

if ((scaling_analysis)); then
  if ((explicit_case_count > 1)); then
    echo "[ERROR] --scaling-analysis supports exactly one case/template." >&2
    exit 1
  fi
  cases=("${cases[0]}")
  RESULTS_ROOT="${output_root:-results/sim_lbs_batch_scaling}"
  NCU_BIN="$(resolve_ncu_bin "$ncu_bin_override")"
  run_scaling_analysis "$RESULTS_ROOT" "$NCU_BIN"
else
  RESULTS_ROOT="${output_root:-results/batch_scaling}"
  run_legacy_batch_scaling "$RESULTS_ROOT"
fi
