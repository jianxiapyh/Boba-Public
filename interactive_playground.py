import glob
import json
import os
import pickle
import random
import subprocess
from argparse import ArgumentParser

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_gl_window(width, height, use_screen_resolution=False):
    import glfw
    from OpenGL import GL as gl

    assert glfw.init(), "GLFW init failed"
    if use_screen_resolution:
        primary_monitor = glfw.get_primary_monitor()
        video_mode = glfw.get_video_mode(primary_monitor) if primary_monitor else None
        if video_mode is not None:
            mode_size = getattr(video_mode, "size", None)
            if mode_size is not None:
                width, height = int(mode_size.width), int(mode_size.height)
            else:
                width = int(getattr(video_mode, "width", width))
                height = int(getattr(video_mode, "height", height))

    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 6)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(width, height, "Boba_Batched Playground", None, None)
    assert window, "create_window failed (need X11 desktop GL)"
    if use_screen_resolution:
        glfw.set_window_pos(window, 0, 0)

    glfw.make_context_current(window)
    _ = gl.glGetString(gl.GL_VERSION)
    glfw.swap_interval(0)
    return window


def resolve_trainer(mode):
    if mode == "perf":
        from qqtt import InvPhyTrainerWarp

        return InvPhyTrainerWarp

    from qqtt.engine.trainer_warp_quality import (
        InvPhyTrainerWarp as QualityTrainerWarp,
    )

    return QualityTrainerWarp


def run_img2video(image_folder, video_path, fps):
    if not os.path.isdir(image_folder):
        return

    images = [
        name
        for name in os.listdir(image_folder)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not images:
        return

    subprocess.run(
        [
            "python",
            "gaussian_splatting/img2video.py",
            "--image_folder",
            image_folder,
            "--video_path",
            video_path,
            "--fps",
            str(fps),
        ],
        check=True,
    )


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--gaussian_path", type=str, default="./gaussian_output")
    parser.add_argument("--bg_img_path", type=str, default="./data/bg.png")
    parser.add_argument("--case_name", type=str, default="double_lift_cloth_3")
    parser.add_argument(
        "--mode",
        choices=("perf", "quality"),
        default="perf",
        help="Use the fast performance path or the single-instance quality/eval path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to ./results/<mode>/<case_name>.",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=1,
        help="Number of calibrated views to generate during single-instance quality mode (valid: 1, 2, 3).",
    )
    parser.add_argument(
        "--collision_pruning_export_path",
        type=str,
        default=None,
        help="Optional .npz path for exporting real collision-pruning figure data in quality mode.",
    )
    return parser


def load_case_config(args, cfg, logger):
    case_name = args.case_name
    base_path = args.base_path

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as file:
        optimal_params = pickle.load(file)
    cfg.set_optimal_params(optimal_params)

    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as file:
        c2ws = pickle.load(file)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)

    with open(f"{base_path}/{case_name}/metadata.json", "r") as file:
        metadata = json.load(file)
    cfg.intrinsics = np.array(metadata["intrinsics"])
    cfg.WH = metadata["WH"]
    cfg.bg_img_path = args.bg_img_path
    return metadata


def load_quality_metrics(output_dir):
    metrics_path = os.path.join(output_dir, "performance_summary.json")
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Missing performance summary JSON: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    args = build_parser().parse_args()
    if args.num_views < 1 or args.num_views > 3:
        raise ValueError(f"--num_views must be between 1 and 3. Received: {args.num_views}")
    if args.collision_pruning_export_path is not None and args.mode != "quality":
        raise ValueError("--collision_pruning_export_path is only supported in quality mode")

    output_dir = args.output_dir or os.path.join(
        "results", args.mode, args.case_name
    )
    os.makedirs(output_dir, exist_ok=True)

    from gaussian_splatting import dynamic_utils as dynamic_utils_backend

    print(
        "[Boba] dynamic_utils backend: "
        f"{dynamic_utils_backend.SELECTED_DYNAMIC_UTIL_VARIANT} "
        f"(device={dynamic_utils_backend.DETECTED_DEVICE_NAME}, "
        f"BOBA_DEVICE={dynamic_utils_backend.BOBA_DEVICE})"
    )

    export_only = args.collision_pruning_export_path is not None
    window = None
    cuda_ctx = None
    if not export_only:
        window = create_gl_window(848, 400)

    _ = torch.empty(1, device="cuda")
    set_all_seeds(42)

    if not export_only:
        import pycuda.driver as cuda_driver

        cuda_driver.init()
        cuda_ctx = cuda_driver.Context.attach()

    try:
        TrainerWarp = resolve_trainer(args.mode)
        from qqtt.utils import cfg, logger

        metadata = load_case_config(args, cfg, logger)

        exp_name = (
            "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
        )
        gaussians_path = (
            f"{args.gaussian_path}/{args.case_name}/{exp_name}/point_cloud/"
            "iteration_10000/point_cloud.ply"
        )
        best_model_path = glob.glob(f"experiments/{args.case_name}/train/best_*.pth")[0]

        logger.set_log_file(path=output_dir, name="inference_log")
        trainer = TrainerWarp(
            data_path=f"{args.base_path}/{args.case_name}/final_data.pkl",
            base_dir=output_dir,
        )

        if args.mode == "perf":
            trainer.interactive_playground(
                best_model_path,
                gaussians_path,
                output_dir=output_dir,
                window=window,
                cuda_ctx=cuda_ctx,
                save_eval_artifacts=False,
            )
        else:
            trainer.interactive_playground(
                best_model_path,
                gaussians_path,
                output_dir=output_dir,
                window=window,
                cuda_ctx=cuda_ctx,
                num_views=args.num_views,
                collision_pruning_export_path=args.collision_pruning_export_path,
            )
            if args.collision_pruning_export_path is not None:
                return
            measured_fps = load_quality_metrics(output_dir)["average_fps"]
            metadata_fps = float(metadata["fps"])
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
        if window is not None:
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
        if cuda_ctx is not None:
            try:
                cuda_ctx.detach()
            except Exception:
                pass


if __name__ == "__main__":
    main()
