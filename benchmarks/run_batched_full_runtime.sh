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
  bash benchmarks/run_batched_full_runtime.sh --batch_size N [--batched_render_variant batch_original|batch_optimized|batch_prune] [--render_mode instance|batch_images] [--gaussian_render_mode shared_template|duplicated] [--instance_id I] [--num_views N] [--batch_image_resolution native|640x480] [--save_video] [--save_batch_images] [--save_batch_grid] [--display_batch_grid] [--profile_render_components] [case_name ...]

This benchmark measures the batched full runtime path:
  spring-mass + LBS + rendering + frame compositing

Options:
  --batch_size N      Required positive batch size.
  --render_mode MODE  instance or batch_images (default: batch_images).
  --gaussian_render_mode MODE
                     shared_template or duplicated (default: shared_template).
  --batched_render_variant VARIANT
                     Preset for batch_images rendering:
                     batch_original, batch_optimized, or batch_prune.
                     Deprecated aliases are accepted: baseline, optimized,
                     optimized_pruned.
  --instance_id I     Required when --render_mode instance.
  --num_views N       Number of camera views to generate (valid: 1, 2, 3; default: 1).
  --save_video        Save PNG folders and MP4 videos.
  --save_batch_images Save per-instance composited images for batch_images mode.
  --save_batch_grid   Save tiled batch preview images for batch_images mode.
  --display_batch_grid Display tiled batch preview in a screen-sized live window for batch_images mode.
  --batch_image_resolution SIZE
                     native or 640x480 (default: native; batch_images only).
  --batch_grid_cols N Number of columns in tiled batch previews (default: ceil(sqrt(batch_size))).
  --profile_render_components
                     Write render component timing JSONs and a per-case CSV summary.

Environment overrides:
  NUM_RUNS
  CASES_FILE
  BASE_PATH
  GAUSSIAN_PATH
  PRUNED_GAUSSIAN_PATH
  BG_IMG_PATH
EOF
}

batch_size=""
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
profile_render_components=0
cases=()

while (($# > 0)); do
  case "$1" in
    --batch_size)
      shift
      if (($# == 0)); then
        echo "Error: --batch_size requires an integer value." >&2
        usage >&2
        exit 1
      fi
      batch_size="$1"
      ;;
    --batch_size=*)
      batch_size="${1#*=}"
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
    --profile_render_components)
      profile_render_components=1
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

if [[ -z "$batch_size" ]]; then
  echo "Error: --batch_size is required." >&2
  usage >&2
  exit 1
fi

if ! [[ "$batch_size" =~ ^[0-9]+$ ]] || ((batch_size < 1)); then
  echo "Error: --batch_size must be a positive integer. Received: ${batch_size}" >&2
  exit 1
fi

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
  if ! [[ "$instance_id" =~ ^[0-9]+$ ]] || ((instance_id < 0 || instance_id >= batch_size)); then
    echo "Error: --instance_id must be in [0, $((batch_size - 1))]. Received: ${instance_id}" >&2
    exit 1
  fi
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

for ((run_idx=1; run_idx<=NUM_RUNS; run_idx++)); do
  run_name=$(printf "run_%02d" "$run_idx")
  mkdir -p "results/batched_render/logs/${run_name}"

  for case_name in "${cases[@]}"; do
    output_dir="results/batched_render/${run_name}/${case_name}/batch_${batch_size}/${mode_dir}"
    log_path="results/batched_render/logs/${run_name}/${case_name}.log"

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
    if ((profile_render_components == 1)); then
      cmd+=(--profile_render_components)
    fi
    if [[ -n "$batched_render_variant" ]]; then
      cmd+=(--batched_render_variant "$batched_render_variant")
      cmd+=(--pruned_gaussian_path "$PRUNED_GAUSSIAN_PATH")
    fi

    echo "=== [batched_render] ${run_name} :: batch=${batch_size} :: mode=${mode_dir} :: ${case_name} ==="
    if "${cmd[@]}" >"$log_path" 2>&1; then
      echo "[OK] ${case_name} (${run_name}, batch=${batch_size}, mode=${mode_dir})"
    else
      echo "[FAIL] ${case_name} (${run_name}, batch=${batch_size}, mode=${mode_dir})"
    fi
  done
done

if ((profile_render_components == 1)); then
  python - "$NUM_RUNS" "$batch_size" "$gaussian_render_mode" "$render_mode" "$mode_dir" "$batch_image_resolution" "$batched_render_variant" "${cases[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path


COMMON_COLUMNS = [
    "Case Name",
    "Batch Size",
    "Gaussian Render Mode",
    "Batched Render Variant",
    "Batch Image Resolution",
    "Render Width",
    "Render Height",
    "Gaussians Per Instance",
    "Time Unit",
]

COMMON_TIMING_COLUMNS = [
    ("prepare_inputs_ms", "Prepare Inputs"),
    ("fully_fused_projection_ms", "Fully Fused Projection"),
    ("spherical_harmonics_ms", "Spherical Harmonics"),
    ("isect_tiles_ms", "Intersect Tiles"),
    ("isect_tiles_count_kernel_ms", "Intersect Tiles Count Kernel"),
    ("isect_tiles_cumsum_ms", "Intersect Tiles Prefix Sum"),
    ("isect_tiles_emit_kernel_ms", "Intersect Tiles Emit Kernel"),
    ("isect_tiles_sort_ms", "Intersect Tiles Radix Sort"),
    ("isect_tiles_cuda_total_ms", "Intersect Tiles CUDA Total"),
    ("isect_visible_gaussians", "Visible Gaussians"),
    ("isect_total_tile_intersections", "Total Tile Intersections"),
    ("isect_avg_tiles_per_gaussian", "Avg Tiles/Gaussian"),
    ("isect_max_tiles_per_gaussian", "Max Tiles/Gaussian"),
    ("isect_offset_encode_ms", "Intersect Offset Encode"),
    ("rasterize_to_pixels_ms", "Rasterize To Pixels"),
    ("background_depth_finalize_ms", "Background/Depth Finalize"),
    ("format_output_ms", "Format Output"),
    ("render_gsplat_total_ms", "Render Total"),
]

SHARED_TEMPLATE_TIMING_COLUMNS = [
    ("prepare_inputs_ms", "Prepare Inputs"),
    ("fully_fused_projection_ms", "Fully Fused Projection"),
    ("shared_template_gather_ms", "Shared Template Gather"),
    ("spherical_harmonics_ms", "Spherical Harmonics"),
    ("isect_tiles_ms", "Intersect Tiles"),
    ("isect_tiles_count_kernel_ms", "Intersect Tiles Count Kernel"),
    ("isect_tiles_cumsum_ms", "Intersect Tiles Prefix Sum"),
    ("isect_tiles_emit_kernel_ms", "Intersect Tiles Emit Kernel"),
    ("isect_tiles_sort_ms", "Intersect Tiles Radix Sort"),
    ("isect_tiles_cuda_total_ms", "Intersect Tiles CUDA Total"),
    ("isect_visible_gaussians", "Visible Gaussians"),
    ("isect_total_tile_intersections", "Total Tile Intersections"),
    ("isect_avg_tiles_per_gaussian", "Avg Tiles/Gaussian"),
    ("isect_max_tiles_per_gaussian", "Max Tiles/Gaussian"),
    ("isect_offset_encode_ms", "Intersect Offset Encode"),
    ("rasterize_to_pixels_ms", "Rasterize To Pixels"),
    ("background_depth_finalize_ms", "Background/Depth Finalize"),
    ("densify_projection_metadata_ms", "Densify Projection Metadata"),
    ("format_output_ms", "Format Output"),
    ("render_gsplat_total_ms", "Render Total"),
]


def mean_or_blank(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return ""
    return f"{sum(values) / len(values):.6f}"


num_runs = int(sys.argv[1])
batch_size = int(sys.argv[2])
gaussian_render_mode = sys.argv[3]
render_mode = sys.argv[4]
mode_dir = sys.argv[5]
batch_image_resolution = sys.argv[6]
batched_render_variant = sys.argv[7]
cases = sys.argv[8:]
timing_columns = (
    SHARED_TEMPLATE_TIMING_COLUMNS
    if gaussian_render_mode == "shared_template"
    else COMMON_TIMING_COLUMNS
)
profile_columns = COMMON_COLUMNS + [label for _, label in timing_columns]

root = Path("results") / "batched_render"
rows = []
for case_name in cases:
    profiles = []
    for run_idx in range(1, num_runs + 1):
        run_name = f"run_{run_idx:02d}"
        profile_path = (
            root
            / run_name
            / case_name
            / f"batch_{batch_size}"
            / mode_dir
            / "render_component_profile.json"
        )
        if not profile_path.exists():
            continue
        with profile_path.open("r", encoding="utf-8") as f:
            profiles.append(json.load(f))

    row = {
        "Case Name": case_name,
        "Batch Size": batch_size,
        "Gaussian Render Mode": gaussian_render_mode,
        "Batched Render Variant": batched_render_variant,
        "Batch Image Resolution": batch_image_resolution,
        "Render Width": mean_or_blank(profile.get("render_width") for profile in profiles),
        "Render Height": mean_or_blank(profile.get("render_height") for profile in profiles),
        "Gaussians Per Instance": mean_or_blank(
            profile.get("gaussians_per_instance") for profile in profiles
        ),
        "Time Unit": "ms",
    }
    for field, label in timing_columns:
        row[label] = mean_or_blank(
            profile.get(field) for profile in profiles
        )
    rows.append(row)

resolution_suffix = (
    "" if batch_image_resolution == "native" else f"_{batch_image_resolution}"
)
csv_path = root / (
    f"render_component_profile_{gaussian_render_mode}_batch_{batch_size}"
    f"{resolution_suffix}"
    f"{'' if not batched_render_variant else '_' + batched_render_variant}.csv"
)
with csv_path.open("w", newline="", encoding="utf-8") as f:
    if gaussian_render_mode == "shared_template":
        f.write(
            "# Fully Fused Projection is one shared-template packed projection over the whole batch.\n"
        )
    writer = csv.DictWriter(f, fieldnames=profile_columns)
    writer.writeheader()
    writer.writerows(rows)
print(f"[profile] wrote {csv_path}")
PY
fi
