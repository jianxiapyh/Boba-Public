import csv
import glob
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

DEFAULT_EXP_NAME = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
DEFAULT_PRUNED_GAUSSIAN_PATH = "./gaussian_output_pruned_policy_30_55"
DEFAULT_PRUNING_RATIO_CSV = "benchmarks/pruning_ratio_policy_30_55.csv"
BATCHED_RENDER_VARIANTS = ("batch_original", "batch_optimized", "batch_prune")
BATCHED_RENDER_VARIANT_ALIASES = {
    "baseline": "batch_original",
    "optimized": "batch_optimized",
    "optimized_pruned": "batch_prune",
}
SIM_FORCE_MODE_GATHER = "gather"
SIM_FORCE_MODE_TEMPLATE_STATE_BATCHED_ATOMIC = "template_state_batched_atomic"
SIM_FORCE_MODES = (
    SIM_FORCE_MODE_GATHER,
    SIM_FORCE_MODE_TEMPLATE_STATE_BATCHED_ATOMIC,
)


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--gaussian_path", type=str, default="./gaussian_output")
    parser.add_argument("--pruned_gaussian_path", type=str, default=DEFAULT_PRUNED_GAUSSIAN_PATH)
    parser.add_argument("--pruning_source_gaussian_path", type=str, default="./gaussian_output")
    parser.add_argument("--case_keep_ratio_csv", type=str, default=DEFAULT_PRUNING_RATIO_CSV)
    parser.add_argument("--default_prune_keep_ratio", type=float, default=0.3)
    parser.add_argument(
        "--prune_mode",
        choices=("opacity", "opacity_area", "opacity_volume"),
        default="opacity_area",
    )
    parser.add_argument("--force_prune", action="store_true")
    parser.add_argument("--bg_img_path", type=str, default="./data/bg.png")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument(
        "--batched_render_variant",
        choices=BATCHED_RENDER_VARIANTS
        + tuple(BATCHED_RENDER_VARIANT_ALIASES.keys()),
        default=None,
    )
    parser.add_argument(
        "--render_mode",
        choices=("instance", "batch_images"),
        default="batch_images",
    )
    parser.add_argument(
        "--gaussian_render_mode",
        choices=("shared_template", "duplicated"),
        default="shared_template",
    )
    parser.add_argument(
        "--sim_force_mode",
        choices=SIM_FORCE_MODES,
        default=SIM_FORCE_MODE_GATHER,
    )
    parser.add_argument("--instance_id", type=int, default=None)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--save_batch_images", action="store_true")
    parser.add_argument("--save_batch_grid", action="store_true")
    parser.add_argument("--display_batch_grid", action="store_true")
    parser.add_argument(
        "--batch_image_resolution",
        choices=("native", "640x480"),
        default="native",
    )
    parser.add_argument("--batch_grid_cols", type=int, default=None)
    parser.add_argument("--profile_render_components", action="store_true")
    parser.add_argument("--num_views", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser


def normalize_batched_render_variant(variant):
    if variant in BATCHED_RENDER_VARIANT_ALIASES:
        normalized = BATCHED_RENDER_VARIANT_ALIASES[variant]
        print(
            "[WARN] --batched_render_variant "
            f"{variant!r} is deprecated; use {normalized!r}.",
            file=sys.stderr,
        )
        return normalized
    return variant


def read_case_keep_ratio(case_keep_ratio_csv, case_name, default_keep_ratio):
    ratio = float(default_keep_ratio)
    csv_path = Path(case_keep_ratio_csv)
    if not csv_path.is_file():
        return ratio

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("case_name", "").strip() == case_name:
                return float(row["keep_ratio"])
    return ratio


def point_cloud_path(gaussian_root, case_name, exp_name=DEFAULT_EXP_NAME):
    return (
        Path(gaussian_root)
        / case_name
        / exp_name
        / "point_cloud"
        / "iteration_10000"
        / "point_cloud.ply"
    )


def ensure_pruned_gaussian(args):
    src = point_cloud_path(args.pruning_source_gaussian_path, args.case_name)
    dst = point_cloud_path(args.pruned_gaussian_path, args.case_name)
    if dst.is_file() and not args.force_prune:
        return
    if not src.is_file():
        raise FileNotFoundError(f"Missing source Gaussian PLY for pruning: {src}")

    keep_ratio = read_case_keep_ratio(
        args.case_keep_ratio_csv,
        args.case_name,
        args.default_prune_keep_ratio,
    )
    cmd = [
        sys.executable,
        "benchmarks/scripts/prune_gaussians.py",
        "--input",
        str(src),
        "--output",
        str(dst),
        "--keep-ratio",
        str(keep_ratio),
        "--mode",
        args.prune_mode,
    ]
    print(
        "[BatchedRender] generating pruned Gaussian asset "
        f"case={args.case_name} keep_ratio={keep_ratio:g} output={dst}"
    )
    subprocess.run(cmd, check=True)


def apply_batched_render_variant(args):
    if args.batched_render_variant is None:
        return

    args.batched_render_variant = normalize_batched_render_variant(
        args.batched_render_variant
    )
    args.render_mode = "batch_images"
    args.instance_id = None
    args.num_views = 1
    if args.batched_render_variant == "batch_original":
        args.gaussian_render_mode = "duplicated"
        args.gaussian_path = args.pruning_source_gaussian_path
    elif args.batched_render_variant == "batch_optimized":
        args.gaussian_render_mode = "shared_template"
        args.gaussian_path = args.pruning_source_gaussian_path
    elif args.batched_render_variant == "batch_prune":
        args.gaussian_render_mode = "shared_template"
        args.gaussian_path = args.pruned_gaussian_path
        ensure_pruned_gaussian(args)


def render_output_mode_dir(
    render_mode,
    instance_id,
    gaussian_render_mode,
    batch_image_resolution="native",
    batched_render_variant=None,
    sim_force_mode=SIM_FORCE_MODE_GATHER,
):
    if render_mode == "instance":
        mode_dir = f"instance_{instance_id}"
    elif render_mode == "batch_images":
        mode_dir = (
            "batch_images"
            if batch_image_resolution == "native"
            else f"batch_images_{batch_image_resolution}"
        )
    else:
        raise ValueError(
            "--render_mode must be 'instance' or 'batch_images'. "
            f"Received: {render_mode}"
        )
    mode_dir = f"{mode_dir}_{gaussian_render_mode}"
    if batched_render_variant:
        mode_dir = f"{mode_dir}_{batched_render_variant}"
    if sim_force_mode != SIM_FORCE_MODE_GATHER:
        mode_dir = f"{mode_dir}_sim_{sim_force_mode}"
    return mode_dir


def default_output_dir(
    case_name,
    batch_size,
    render_mode,
    instance_id,
    gaussian_render_mode,
    batch_image_resolution="native",
    batched_render_variant=None,
    sim_force_mode=SIM_FORCE_MODE_GATHER,
):
    return os.path.join(
        "results",
        "batched_render",
        case_name,
        f"batch_{batch_size}",
        render_output_mode_dir(
            render_mode,
            instance_id,
            gaussian_render_mode,
            batch_image_resolution=batch_image_resolution,
            batched_render_variant=batched_render_variant,
            sim_force_mode=sim_force_mode,
        ),
    )


def validate_args(args):
    if args.batch_size < 1:
        raise ValueError(
            f"--batch_size must be a positive integer. Received: {args.batch_size}"
        )
    if args.num_views < 1 or args.num_views > 3:
        raise ValueError(
            f"--num_views must be between 1 and 3. Received: {args.num_views}"
        )
    if args.render_mode == "batch_images":
        if args.num_views != 1:
            raise ValueError("--render_mode=batch_images currently supports --num_views=1 only.")
        if args.instance_id is not None:
            raise ValueError("--instance_id can only be provided with --render_mode=instance.")
    elif args.render_mode == "instance":
        if args.instance_id is None:
            raise ValueError("--instance_id is required when --render_mode=instance.")
        if args.instance_id < 0 or args.instance_id >= args.batch_size:
            raise ValueError(
                f"--instance_id must be in [0, {args.batch_size - 1}], got {args.instance_id}"
            )
    else:
        raise ValueError(
            "--render_mode must be 'instance' or 'batch_images'. "
            f"Received: {args.render_mode}"
        )
    if args.render_mode != "batch_images" and args.batch_image_resolution != "native":
        raise ValueError(
            "--batch_image_resolution=640x480 can only be used with "
            "--render_mode=batch_images."
        )
    if args.render_mode != "batch_images" and (
        args.save_batch_images or args.save_batch_grid or args.display_batch_grid
    ):
        raise ValueError(
            "--save_batch_images, --save_batch_grid, and --display_batch_grid "
            "can only be used with --render_mode=batch_images."
        )
    if args.batch_grid_cols is not None and args.batch_grid_cols < 1:
        raise ValueError("--batch_grid_cols must be a positive integer.")


def main():
    args = build_parser().parse_args()
    apply_batched_render_variant(args)
    validate_args(args)

    output_dir = args.output_dir or default_output_dir(
        case_name=args.case_name,
        batch_size=args.batch_size,
        render_mode=args.render_mode,
        instance_id=args.instance_id,
        gaussian_render_mode=args.gaussian_render_mode,
        batch_image_resolution=args.batch_image_resolution,
        batched_render_variant=args.batched_render_variant,
        sim_force_mode=args.sim_force_mode,
    )
    os.makedirs(output_dir, exist_ok=True)
    profile_json_path = os.path.join(output_dir, "render_component_profile.json")
    if os.path.exists(profile_json_path):
        os.remove(profile_json_path)

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    from gaussian_splatting import dynamic_utils as dynamic_utils_backend
    from interactive_playground import (
        create_gl_window,
        load_case_config,
        load_quality_metrics,
        run_img2video,
        set_all_seeds,
    )

    print(
        "[Boba] dynamic_utils backend: "
        f"{dynamic_utils_backend.SELECTED_DYNAMIC_UTIL_VARIANT} "
        f"(device={dynamic_utils_backend.DETECTED_DEVICE_NAME}, "
        f"BOBA_DEVICE={dynamic_utils_backend.BOBA_DEVICE})"
    )

    if args.display_batch_grid:
        window = create_gl_window(640, 480, use_screen_resolution=True)
    elif args.render_mode == "batch_images" and args.batch_image_resolution == "640x480":
        window = create_gl_window(640, 480)
    else:
        window = create_gl_window(848, 400)
    _ = torch.empty(1, device="cuda")
    set_all_seeds(42)

    import pycuda.driver as cuda_driver

    cuda_driver.init()
    cuda_ctx = cuda_driver.Context.attach()

    try:
        from qqtt import InvPhyTrainerWarp
        from qqtt.utils import cfg, logger

        metadata = load_case_config(args, cfg, logger)

        gaussians_path = (
            f"{args.gaussian_path}/{args.case_name}/{DEFAULT_EXP_NAME}/point_cloud/"
            "iteration_10000/point_cloud.ply"
        )
        best_model_matches = glob.glob(f"experiments/{args.case_name}/train/best_*.pth")
        if not best_model_matches:
            raise FileNotFoundError(
                f"Unable to find best checkpoint for case: {args.case_name}"
            )
        best_model_path = best_model_matches[0]

        logger.set_log_file(path=output_dir, name="inference_log")
        trainer = InvPhyTrainerWarp(
            data_path=f"{args.base_path}/{args.case_name}/final_data.pkl",
            base_dir=output_dir,
        )
        trainer.run_batched_full_runtime(
            model_path=best_model_path,
            gs_path=gaussians_path,
            output_dir=output_dir,
            window=window,
            cuda_ctx=cuda_ctx,
            batch_size=args.batch_size,
            num_views=args.num_views,
            render_mode=args.render_mode,
            instance_id=args.instance_id,
            gaussian_render_mode=args.gaussian_render_mode,
            save_video=args.save_video,
            save_batch_images=args.save_batch_images,
            save_batch_grid=args.save_batch_grid,
            display_batch_grid=args.display_batch_grid,
            batch_image_resolution=args.batch_image_resolution,
            batch_grid_cols=args.batch_grid_cols,
            profile_render_components=args.profile_render_components,
            sim_force_mode=args.sim_force_mode,
        )

        if args.save_video:
            measured_fps = load_quality_metrics(output_dir)["average_fps"]
            metadata_fps = float(metadata["fps"])
            if args.render_mode == "batch_images":
                run_img2video(
                    os.path.join(output_dir, "output"),
                    os.path.join(output_dir, "output.mp4"),
                    metadata_fps,
                )
                run_img2video(
                    os.path.join(output_dir, "output"),
                    os.path.join(output_dir, "output_realtime.mp4"),
                    measured_fps,
                )
                if args.save_batch_images:
                    batch_image_root = os.path.join(output_dir, "batch_images")
                    for batch_idx in range(args.batch_size):
                        image_dir = os.path.join(
                            batch_image_root, f"instance_{batch_idx:03d}"
                        )
                        video_base = os.path.join(
                            batch_image_root, f"instance_{batch_idx:03d}"
                        )
                        run_img2video(
                            image_dir,
                            f"{video_base}.mp4",
                            metadata_fps,
                        )
                        run_img2video(
                            image_dir,
                            f"{video_base}_realtime.mp4",
                            measured_fps,
                        )
                if args.save_batch_grid:
                    run_img2video(
                        os.path.join(output_dir, "output_grid"),
                        os.path.join(output_dir, "output_grid.mp4"),
                        metadata_fps,
                    )
                    run_img2video(
                        os.path.join(output_dir, "output_grid"),
                        os.path.join(output_dir, "output_grid_realtime.mp4"),
                        measured_fps,
                    )
            else:
                for view_idx in range(args.num_views):
                    run_img2video(
                        os.path.join(output_dir, str(view_idx)),
                        os.path.join(output_dir, f"{view_idx}.mp4"),
                        metadata_fps,
                    )
                    run_img2video(
                        os.path.join(output_dir, str(view_idx)),
                        os.path.join(output_dir, f"{view_idx}_realtime.mp4"),
                        measured_fps,
                    )
                run_img2video(
                    os.path.join(output_dir, "output"),
                    os.path.join(output_dir, "output.mp4"),
                    metadata_fps,
                )
                run_img2video(
                    os.path.join(output_dir, "output"),
                    os.path.join(output_dir, "output_realtime.mp4"),
                    measured_fps,
                )
    finally:
        import glfw

        try:
            glfw.make_context_current(window)
        except Exception:
            pass
        try:
            glfw.destroy_window(window)
            glfw.terminate()
        except Exception:
            pass
        try:
            cuda_ctx.detach()
        except Exception:
            pass


if __name__ == "__main__":
    main()
