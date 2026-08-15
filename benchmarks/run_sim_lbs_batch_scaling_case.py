import glob
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path


# This helper lives under benchmarks/ but imports modules from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_str = str(REPO_ROOT)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)


def default_ncu_profile_frame_stride():
    raw_value = os.environ.get("NCU_PROFILE_FRAME_STRIDE")
    if raw_value is None or raw_value == "":
        return None
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "NCU_PROFILE_FRAME_STRIDE must be a positive integer. "
            f"Received: {raw_value}"
        ) from exc


def default_ncu_profile_max_frames():
    raw_value = os.environ.get("NCU_PROFILE_MAX_FRAMES", "3")
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "NCU_PROFILE_MAX_FRAMES must be a positive integer. "
            f"Received: {raw_value}"
        ) from exc


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--gaussian_path", type=str, default="./gaussian_output")
    parser.add_argument("--bg_img_path", type=str, default="./data/bg.png")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--scaling-analysis", action="store_true")
    parser.add_argument("--ncu-profile-loop", action="store_true")
    parser.add_argument(
        "--ncu-profile-frame-stride",
        type=int,
        default=default_ncu_profile_frame_stride(),
    )
    parser.add_argument(
        "--ncu-profile-max-frames",
        type=int,
        default=default_ncu_profile_max_frames(),
    )
    parser.add_argument(
        "--ncu-profile-nvtx-name",
        type=str,
        default="sim_lbs_profile_frame",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError(
            f"--batch_size must be a positive integer. Received: {args.batch_size}"
        )
    if args.ncu_profile_frame_stride is not None and args.ncu_profile_frame_stride < 1:
        raise ValueError(
            "--ncu-profile-frame-stride must be a positive integer. "
            f"Received: {args.ncu_profile_frame_stride}"
        )
    if args.ncu_profile_max_frames < 1:
        raise ValueError(
            "--ncu-profile-max-frames must be a positive integer. "
            f"Received: {args.ncu_profile_max_frames}"
        )

    output_dir = args.output_dir or os.path.join(
        "results", "batch_scaling", f"batch_{args.batch_size}", args.case_name
    )
    os.makedirs(output_dir, exist_ok=True)

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    from gaussian_splatting import dynamic_utils as dynamic_utils_backend
    from interactive_playground import load_case_config, set_all_seeds
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import cfg, logger

    print(
        "[Boba] dynamic_utils backend: "
        f"{dynamic_utils_backend.SELECTED_DYNAMIC_UTIL_VARIANT} "
        f"(device={dynamic_utils_backend.DETECTED_DEVICE_NAME}, "
        f"BOBA_DEVICE={dynamic_utils_backend.BOBA_DEVICE})"
    )

    _ = torch.empty(1, device="cuda")
    set_all_seeds(42)
    load_case_config(args, cfg, logger)

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = (
        f"{args.gaussian_path}/{args.case_name}/{exp_name}/point_cloud/"
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

    if args.scaling_analysis and not args.ncu_profile_loop and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    summary = trainer.run_headless_sim_lbs(
        model_path=best_model_path,
        gs_path=gaussians_path,
        output_dir=output_dir,
        n_dup=args.batch_size - 1,
        ncu_profile_loop=args.ncu_profile_loop,
        ncu_profile_frame_stride=args.ncu_profile_frame_stride,
        ncu_profile_max_frames=args.ncu_profile_max_frames,
        ncu_profile_nvtx_name=args.ncu_profile_nvtx_name,
    )

    if args.scaling_analysis and not args.ncu_profile_loop:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
            peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
            gpu_total_memory_gb = (
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
            )
            gpu_name = torch.cuda.get_device_name(0)
        else:
            peak_allocated_gb = None
            peak_reserved_gb = None
            gpu_total_memory_gb = None
            gpu_name = None

        simulation_ms = summary.get("average_simulator_ms")
        lbs_ms = summary.get("average_full_motion_interpolation_ms")
        if simulation_ms is not None and lbs_ms is not None:
            total_sim_lbs_ms = simulation_ms + lbs_ms
        else:
            total_sim_lbs_ms = summary.get("average_sim_lbs_total_ms")

        if total_sim_lbs_ms and total_sim_lbs_ms > 0:
            throughput = args.batch_size / (total_sim_lbs_ms / 1000.0)
            per_instance_ms = total_sim_lbs_ms / args.batch_size
        else:
            throughput = None
            per_instance_ms = None

        metrics = {
            "status": "success",
            "case_name": args.case_name,
            "batch_size": int(args.batch_size),
            "frames_used_for_stats": summary.get("frames_used_for_stats"),
            "warmup_iterations": 2,
            "measured_iterations": summary.get("frames_used_for_stats"),
            "total_sim_lbs_ms": total_sim_lbs_ms,
            "simulation_ms": simulation_ms,
            "lbs_ms": lbs_ms,
            "throughput_instances_per_sec": throughput,
            "per_instance_ms": per_instance_ms,
            "peak_allocated_gb": peak_allocated_gb,
            "peak_reserved_gb": peak_reserved_gb,
            "gpu_total_memory_gb": gpu_total_memory_gb,
            "gpu_name": gpu_name,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "case_data_path": f"{args.base_path}/{args.case_name}/final_data.pkl",
            "best_model_path": best_model_path,
            "gaussian_point_cloud_path": gaussians_path,
        }
        metrics_path = os.path.join(output_dir, "scaling_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)


if __name__ == "__main__":
    main()
