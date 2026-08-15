import glob
import json
import os
import pickle
import random
import subprocess
from argparse import ArgumentParser

import numpy as np
import torch


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_n_ctrl_parts(case_name):
    if "single" in case_name:
        return 1
    if "double" in case_name or case_name == "weird_package" or case_name.startswith("cloth_"):
        return 2
    raise ValueError(f"Cannot infer controller count for case '{case_name}'")


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
        help="Use the fast performance path or the quality/eval path.",
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
        help="Number of calibrated views to generate during quality mode (valid: 1, 2, 3).",
    )
    parser.add_argument(
        "--n_ctrl_parts",
        type=int,
        default=None,
        help="Override inferred controller-part count. If omitted, infer from case name.",
    )
    parser.add_argument(
        "--inv_ctrl",
        action="store_true",
        help="Invert the manual horizontal control direction.",
    )
    return parser


def load_case_config(args, cfg, logger):
    case_name = args.case_name

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

    with open(f"{args.base_path}/{case_name}/calibrate.pkl", "rb") as file:
        c2ws = pickle.load(file)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)

    with open(f"{args.base_path}/{case_name}/metadata.json", "r") as file:
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

    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import cfg, logger

    output_dir = args.output_dir or os.path.join("results", args.mode, args.case_name)
    os.makedirs(output_dir, exist_ok=True)

    set_all_seeds(42)
    metadata = load_case_config(args, cfg, logger)

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = (
        f"{args.gaussian_path}/{args.case_name}/{exp_name}/point_cloud/"
        "iteration_10000/point_cloud.ply"
    )
    best_model_path = glob.glob(f"experiments/{args.case_name}/train/best_*.pth")[0]

    logger.set_log_file(path=output_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{args.base_path}/{args.case_name}/final_data.pkl",
        base_dir=output_dir,
        pure_inference_mode=True,
    )

    trainer.interactive_playground(
        best_model_path,
        gaussians_path,
        n_ctrl_parts=(
            args.n_ctrl_parts
            if args.n_ctrl_parts is not None
            else infer_n_ctrl_parts(args.case_name)
        ),
        inv_ctrl=args.inv_ctrl,
        replay_experiment_trace=True,
        output_dir=output_dir,
        save_eval_artifacts=(args.mode == "quality"),
        num_views=args.num_views if args.mode == "quality" else 1,
    )

    if args.mode == "quality":
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


if __name__ == "__main__":
    main()
