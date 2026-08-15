#!/usr/bin/env bash
set -u -o pipefail

CASES_FILE="${CASES_FILE:-data_config.csv}"
RESULTS_ROOT="${RESULTS_ROOT:-results/sim_lbs_batch_scaling_all_cases}"

read_cases_file() {
  mapfile -t cases < <(awk -F, 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); if ($1 != "") print $1}' "$1")
}

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_sim_lbs_batch_scaling_all_cases.sh [--timing-only|--ncu-only|--summarize-only] [--batch-sizes "1 32 64 128 256 512"] [--output-dir DIR] [--ncu-bin PATH] [case_name ...]

Runs Sim+LBS scaling-analysis profiling once per case. If no case names are
provided, cases are read from CASES_FILE (default: data_config.csv).

Environment overrides:
  NUM_RUNS
  FORCE_RERUN
  CASES_FILE
  BASE_PATH
  GAUSSIAN_PATH
  BG_IMG_PATH
  RESULTS_ROOT
  NCU_PROFILE_FRAME_STRIDE
  NCU_PROFILE_MAX_FRAMES
  NCU_PROFILE_NVTX_NAME
  NCU_TARGET_PROCESSES
EOF
}

set_pass_mode() {
  local requested="$1"
  if ((pass_mode_set)); then
    echo "[ERROR] Use only one of --timing-only, --ncu-only, or --summarize-only." >&2
    exit 1
  fi
  pass_mode_args=("$requested")
  pass_mode_set=1
}

csv_escape() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

csv_write_row() {
  local first=1
  for value in "$@"; do
    if ((first)); then
      first=0
    else
      printf ','
    fi
    csv_escape "$value"
  done
  printf '\n'
}

merge_case_csvs() {
  local output_csv="$1"
  local case_name case_csv header
  local wrote_header=0

  : > "$output_csv"
  for case_name in "${cases[@]}"; do
    case_csv="${RESULTS_ROOT}/${case_name}/batch_scaling_sim_lbs.csv"
    if [[ ! -f "$case_csv" ]]; then
      continue
    fi

    if ((wrote_header == 0)); then
      IFS= read -r header < "$case_csv" || true
      if [[ -n "$header" ]]; then
        printf "case_name,%s\n" "$header" >> "$output_csv"
        wrote_header=1
      fi
    fi

    awk -v case_name="$case_name" 'NR > 1 { print "\"" case_name "\"," $0 }' "$case_csv" >> "$output_csv"
  done
}

batch_sizes=()
cases=()
output_root=""
ncu_bin_override=""
pass_mode_args=()
pass_mode_set=0

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
              for token in $1; do
                batch_sizes+=("$token")
              done
              shift
            else
              break
            fi
            ;;
        esac
      done
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
    --timing-only)
      set_pass_mode "--timing-only"
      shift
      ;;
    --ncu-only)
      set_pass_mode "--ncu-only"
      shift
      ;;
    --summarize-only)
      set_pass_mode "--summarize-only"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      cases+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$output_root" ]]; then
  RESULTS_ROOT="$output_root"
fi

if ((${#cases[@]} == 0)); then
  read_cases_file "$CASES_FILE"
fi

if ((${#cases[@]} == 0)); then
  echo "[ERROR] No cases provided and ${CASES_FILE} did not contain any cases." >&2
  exit 1
fi

mkdir -p "$RESULTS_ROOT" "${RESULTS_ROOT}/logs"
attempts_manifest="${RESULTS_ROOT}/all_cases_attempts.csv"
merged_csv="${RESULTS_ROOT}/all_cases_batch_scaling_sim_lbs.csv"
csv_write_row "case_name" "status" "output_dir" "log_path" > "$attempts_manifest"

for case_name in "${cases[@]}"; do
  case_output_dir="${RESULTS_ROOT}/${case_name}"
  log_path="${RESULTS_ROOT}/logs/${case_name}.log"
  cmd=(
    bash benchmarks/run_sim_lbs_batch_scaling.sh
    --scaling-analysis
    "${pass_mode_args[@]}"
    --output-dir "$case_output_dir"
  )

  if ((${#batch_sizes[@]} > 0)); then
    cmd+=(--batch-sizes "${batch_sizes[@]}")
  fi
  if [[ -n "$ncu_bin_override" ]]; then
    cmd+=(--ncu-bin "$ncu_bin_override")
  fi
  cmd+=("$case_name")

  echo "=== [all_cases:scaling_analysis] case=${case_name} ==="
  if "${cmd[@]}" > "$log_path" 2>&1; then
    csv_write_row "$case_name" "ok" "$case_output_dir" "$log_path" >> "$attempts_manifest"
    echo "[OK] ${case_name}"
  else
    csv_write_row "$case_name" "failed" "$case_output_dir" "$log_path" >> "$attempts_manifest"
    echo "[FAIL] ${case_name}; see ${log_path}"
  fi
done

merge_case_csvs "$merged_csv"

echo "[DONE] Attempts manifest: ${attempts_manifest}"
echo "[DONE] Merged CSV: ${merged_csv}"
