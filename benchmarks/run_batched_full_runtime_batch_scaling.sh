#!/usr/bin/env bash
set -u

NUM_RUNS="${NUM_RUNS:-3}"
CASES_FILE="${CASES_FILE:-data_config.csv}"
BASE_PATH="${BASE_PATH:-./data/different_types}"
GAUSSIAN_PATH="${GAUSSIAN_PATH:-./gaussian_output}"
PRUNED_GAUSSIAN_PATH="${PRUNED_GAUSSIAN_PATH:-./gaussian_output_pruned_policy_30_55}"
BG_IMG_PATH="${BG_IMG_PATH:-./data/bg.png}"

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

usage() {
  cat <<'EOF'
Usage:
  bash benchmarks/run_batched_full_runtime_batch_scaling.sh --batch_sizes 1 2 4 8 [--batched_render_variant batch_original|batch_optimized|batch_prune] [--render_mode instance|batch_images] [--gaussian_render_mode shared_template|duplicated] [--instance_id I] [--num_views N] [--batch_image_resolution native|640x480] [--save_video] [--save_batch_images] [--save_batch_grid] [--display_batch_grid] [case_name ...]

This benchmark measures batched full runtime scaling:
  spring-mass + LBS + rendering + frame compositing

Defaults:
  render_mode=batch_images
  gaussian_render_mode=shared_template
  num_views=1
  save_video=false
  NUM_RUNS=3 (via environment default)

Options:
  --batch_sizes N...   Required list of positive batch sizes.
  --render_mode MODE   instance or batch_images (default: batch_images).
  --gaussian_render_mode MODE
                       shared_template or duplicated (default: shared_template).
  --batched_render_variant VARIANT
                       Preset for batch_images rendering:
                       batch_original, batch_optimized, or batch_prune.
                       Deprecated aliases are accepted: baseline, optimized,
                       optimized_pruned.
  --instance_id I      Required when --render_mode instance.
  --num_views N        Number of camera views to generate (valid: 1, 2, 3; default: 1).
  --save_video         Save PNG folders and MP4 videos.
  --save_batch_images  Save per-instance composited images for batch_images mode.
  --save_batch_grid    Save tiled batch preview images for batch_images mode.
  --display_batch_grid Display tiled batch preview in a screen-sized live window for batch_images mode.
  --batch_image_resolution SIZE
                       native or 640x480 (default: native; batch_images only).
  --batch_grid_cols N  Number of columns in tiled batch previews (default: ceil(sqrt(batch_size))).

Environment overrides:
  NUM_RUNS
  CASES_FILE
  BASE_PATH
  GAUSSIAN_PATH
  PRUNED_GAUSSIAN_PATH
  BG_IMG_PATH
EOF
}

batch_sizes=()
render_mode="batch_images"
gaussian_render_mode="shared_template"
batched_render_variant=""
instance_id=""
num_views=1
save_video=0
save_batch_images=0
save_batch_grid=0
display_batch_grid=0
batch_image_resolution="native"
batch_grid_cols=""
cases=()

while (($# > 0)); do
  case "$1" in
    --batch_sizes)
      shift
      while (($# > 0)); do
        case "$1" in
          --*)
            break
            ;;
          *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
              batch_sizes+=("$1")
              shift
            else
              break
            fi
            ;;
        esac
      done
      continue
      ;;
    --render_mode)
      shift
      if (($# == 0)); then
        echo "Error: --render_mode requires a value." >&2
        usage >&2
        exit 1
      fi
      render_mode="$1"
      ;;
    --render_mode=*)
      render_mode="${1#*=}"
      ;;
    --gaussian_render_mode)
      shift
      if (($# == 0)); then
        echo "Error: --gaussian_render_mode requires a value." >&2
        usage >&2
        exit 1
      fi
      gaussian_render_mode="$1"
      ;;
    --gaussian_render_mode=*)
      gaussian_render_mode="${1#*=}"
      ;;
    --batched_render_variant|--batched-render-variant)
      shift
      if (($# == 0)); then
        echo "Error: --batched_render_variant requires a value." >&2
        usage >&2
        exit 1
      fi
      batched_render_variant="$1"
      ;;
    --batched_render_variant=*|--batched-render-variant=*)
      batched_render_variant="${1#*=}"
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
    --display_batch_grid)
      display_batch_grid=1
      ;;
    --batch_image_resolution)
      shift
      if (($# == 0)); then
        echo "Error: --batch_image_resolution requires a value." >&2
        usage >&2
        exit 1
      fi
      batch_image_resolution="$1"
      ;;
    --batch_image_resolution=*)
      batch_image_resolution="${1#*=}"
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

if ((${#batch_sizes[@]} == 0)); then
  echo "[ERROR] --batch_sizes is required." >&2
  usage >&2
  exit 1
fi

for batch_size in "${batch_sizes[@]}"; do
  if ! [[ "$batch_size" =~ ^[0-9]+$ ]] || ((batch_size < 1)); then
    echo "[ERROR] Invalid batch size: ${batch_size}" >&2
    exit 1
  fi
done

if [[ ! "$num_views" =~ ^[0-9]+$ ]] || ((num_views < 1 || num_views > 3)); then
  echo "Error: --num_views must be an integer between 1 and 3. Received: ${num_views}" >&2
  exit 1
fi

if [[ "$render_mode" != "instance" && "$render_mode" != "batch_images" ]]; then
  echo "Error: --render_mode must be 'instance' or 'batch_images'. Received: ${render_mode}" >&2
  exit 1
fi

if [[ "$gaussian_render_mode" != "shared_template" && "$gaussian_render_mode" != "duplicated" ]]; then
  echo "Error: --gaussian_render_mode must be 'shared_template' or 'duplicated'. Received: ${gaussian_render_mode}" >&2
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

if [[ "$batch_image_resolution" != "native" && "$batch_image_resolution" != "640x480" ]]; then
  echo "Error: --batch_image_resolution must be 'native' or '640x480'. Received: ${batch_image_resolution}" >&2
  exit 1
fi

if [[ -n "$batch_grid_cols" ]]; then
  if ! [[ "$batch_grid_cols" =~ ^[0-9]+$ ]] || ((batch_grid_cols < 1)); then
    echo "Error: --batch_grid_cols must be a positive integer. Received: ${batch_grid_cols}" >&2
    exit 1
  fi
fi

if [[ "$render_mode" != "batch_images" && ( "$save_batch_images" -eq 1 || "$save_batch_grid" -eq 1 || "$display_batch_grid" -eq 1 ) ]]; then
  echo "Error: --save_batch_images, --save_batch_grid, and --display_batch_grid can only be used with --render_mode batch_images." >&2
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
elif [[ "$render_mode" == "instance" ]]; then
  if [[ -z "$instance_id" ]]; then
    echo "Error: --instance_id is required when --render_mode instance." >&2
    exit 1
  fi
  for batch_size in "${batch_sizes[@]}"; do
    if ! [[ "$instance_id" =~ ^[0-9]+$ ]] || ((instance_id < 0 || instance_id >= batch_size)); then
      echo "Error: --instance_id must be in [0, $((batch_size - 1))] for batch size ${batch_size}. Received: ${instance_id}" >&2
      exit 1
    fi
  done
fi

if ((${#cases[@]} == 0)); then
  read_cases_file "$CASES_FILE"
fi

if [[ "$render_mode" == "instance" ]]; then
  mode_dir="instance_${instance_id}"
elif [[ "$render_mode" == "batch_images" ]]; then
  if [[ "$batch_image_resolution" == "native" ]]; then
    mode_dir="batch_images"
  else
    mode_dir="batch_images_${batch_image_resolution}"
  fi
fi
mode_dir="${mode_dir}_${gaussian_render_mode}"
if [[ -n "$batched_render_variant" ]]; then
  mode_dir="${mode_dir}_${batched_render_variant}"
fi

mkdir -p results/batched_render

resolution_suffix=""
if [[ "$batch_image_resolution" != "native" ]]; then
  resolution_suffix="_${batch_image_resolution}"
fi

for ((run_idx=1; run_idx<=NUM_RUNS; run_idx++)); do
  run_name=$(printf "run_%02d" "$run_idx")

  for batch_size in "${batch_sizes[@]}"; do
    mkdir -p "results/batched_render/logs/${run_name}/batch_${batch_size}"

    for case_name in "${cases[@]}"; do
      output_dir="results/batched_render/${run_name}/${case_name}/batch_${batch_size}/${mode_dir}"
      log_path="results/batched_render/logs/${run_name}/batch_${batch_size}/${case_name}.log"

      cmd=(
        python benchmarks/run_batched_full_runtime_case.py
        --base_path "$BASE_PATH"
        --gaussian_path "$GAUSSIAN_PATH"
        --bg_img_path "$BG_IMG_PATH"
        --case_name "$case_name"
        --batch_size "$batch_size"
        --render_mode "$render_mode"
        --gaussian_render_mode "$gaussian_render_mode"
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
      if ((display_batch_grid == 1)); then
        cmd+=(--display_batch_grid)
      fi
      if [[ -n "$batch_grid_cols" ]]; then
        cmd+=(--batch_grid_cols "$batch_grid_cols")
      fi
      if [[ -n "$batched_render_variant" ]]; then
        cmd+=(--batched_render_variant "$batched_render_variant")
        cmd+=(--pruned_gaussian_path "$PRUNED_GAUSSIAN_PATH")
      fi

      echo "=== [batched_render_scaling] ${run_name} :: batch=${batch_size} :: mode=${mode_dir} :: ${case_name} ==="
      if "${cmd[@]}" >"$log_path" 2>&1; then
        echo "[OK] ${case_name} (${run_name}, batch=${batch_size}, mode=${mode_dir})"
      else
        echo "[FAIL] ${case_name} (${run_name}, batch=${batch_size}, mode=${mode_dir})"
      fi
    done
  done
done

variant_suffix=""
if [[ -n "$batched_render_variant" ]]; then
  variant_suffix="_${batched_render_variant}"
fi

aggregate_cmd=(
  python benchmarks/post-processing/aggregate_batched_full_runtime_batch_scaling.py
  --render_mode "$render_mode"
  --gaussian_render_mode "$gaussian_render_mode"
  --batch_image_resolution "$batch_image_resolution"
  --output_table "results/batched_render/batch_scaling_${gaussian_render_mode}${resolution_suffix}${variant_suffix}_table.csv"
  --output_overall "results/batched_render/batch_scaling_${gaussian_render_mode}${resolution_suffix}${variant_suffix}_overall.csv"
)

if [[ -n "$batched_render_variant" ]]; then
  aggregate_cmd+=(--batched_render_variant "$batched_render_variant")
fi

if [[ "$render_mode" == "instance" ]]; then
  aggregate_cmd+=(--instance_id "$instance_id")
fi

"${aggregate_cmd[@]}" "${cases[@]}"
