import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image

from lpipsPyTorch.modules.lpips import LPIPS
from utils.image_utils import psnr
from utils.loss_utils import ssim


def img2tensor(img, device):
    img = np.array(img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0).to(device)


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 1.0


def frame_ranges(split_info):
    train_frames = list(range(split_info["train"][0] + 1, split_info["train"][1]))
    test_frames = list(range(split_info["test"][0], split_info["test"][1]))
    return {"train": train_frames, "test": test_frames}


@torch.no_grad()
def evaluate_frame(render_file, gt_file, gt_mask_file, human_mask_file, lpips_model, device):
    gt = np.array(Image.open(gt_file))
    gt_mask = np.array(Image.open(gt_mask_file)).astype(np.float32) / 255.0

    render = np.array(Image.open(render_file))
    render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])

    human_mask = np.array(Image.open(human_mask_file))
    inv_human_mask = (1.0 - human_mask / 255.0).astype(np.float32)

    gt = gt.astype(np.float32) * gt_mask[..., None]
    bg_mask = gt_mask == 0
    gt[bg_mask] = [255, 255, 255]
    render = render[:, :, :3].astype(np.float32)

    gt = gt * inv_human_mask[..., None]
    render = render * inv_human_mask[..., None]
    render_mask = render_mask * inv_human_mask

    gt_tensor = img2tensor(gt, device)
    render_tensor = img2tensor(render, device)

    return {
        "psnr": psnr(render_tensor, gt_tensor).item(),
        "ssim": ssim(render_tensor, gt_tensor).item(),
        "lpips": lpips_model(render_tensor, gt_tensor).item(),
        "iou": compute_iou(gt_mask > 0, render_mask > 0),
    }


def mean_metric(values):
    if not values:
        return float("nan")
    return float(np.mean(values))


def empty_metric_lists():
    return {
        "train": {"psnr": [], "ssim": [], "lpips": [], "iou": []},
        "test": {"psnr": [], "ssim": [], "lpips": [], "iou": []},
    }


def scene_row(scene, scene_metrics):
    return {
        "scene": scene,
        "psnr_train": mean_metric(scene_metrics["train"]["psnr"]),
        "ssim_train": mean_metric(scene_metrics["train"]["ssim"]),
        "lpips_train": mean_metric(scene_metrics["train"]["lpips"]),
        "iou_train": mean_metric(scene_metrics["train"]["iou"]),
        "psnr_test": mean_metric(scene_metrics["test"]["psnr"]),
        "ssim_test": mean_metric(scene_metrics["test"]["ssim"]),
        "lpips_test": mean_metric(scene_metrics["test"]["lpips"]),
        "iou_test": mean_metric(scene_metrics["test"]["iou"]),
    }


def evaluate_scene(scene, args, lpips_model, device):
    render_eval_dir = os.path.join(args.render_path, scene)
    output_scene_dir = os.path.join(args.output_dir, scene)
    human_mask_dir = os.path.join(args.human_mask_path, scene)

    if not os.path.isdir(render_eval_dir) or not os.path.isdir(output_scene_dir):
        return None

    with open(os.path.join(render_eval_dir, "split.json"), "r") as file:
        split_info = json.load(file)

    ranges = frame_ranges(split_info)
    scene_metrics = empty_metric_lists()

    for view_idx in range(args.num_views):
        render_view_dir = os.path.join(output_scene_dir, str(view_idx))
        if not os.path.isdir(render_view_dir):
            continue

        for split_name, frame_indices in ranges.items():
            for frame_idx in frame_indices:
                render_file = os.path.join(render_view_dir, f"{frame_idx:05d}.png")
                gt_file = os.path.join(render_eval_dir, "color", str(view_idx), f"{frame_idx}.png")
                gt_mask_file = os.path.join(render_eval_dir, "mask", str(view_idx), f"{frame_idx}.png")
                human_mask_file = os.path.join(
                    human_mask_dir, "mask", str(view_idx), "0", f"{frame_idx}.png"
                )

                if not (
                    os.path.isfile(render_file)
                    and os.path.isfile(gt_file)
                    and os.path.isfile(gt_mask_file)
                    and os.path.isfile(human_mask_file)
                ):
                    continue

                metrics = evaluate_frame(
                    render_file=render_file,
                    gt_file=gt_file,
                    gt_mask_file=gt_mask_file,
                    human_mask_file=human_mask_file,
                    lpips_model=lpips_model,
                    device=device,
                )
                for metric_name, value in metrics.items():
                    scene_metrics[split_name][metric_name].append(value)

    return scene_row(scene, scene_metrics), scene_metrics


def extend_metrics(destination, source):
    for split_name, metric_lists in source.items():
        for metric_name, values in metric_lists.items():
            destination[split_name][metric_name].extend(values)


def overall_metrics(scene_rows, sample_metrics, overall_mode):
    if overall_mode == "phystwin":
        return {
            "scene": "OVERALL",
            "psnr_train": mean_metric(sample_metrics["train"]["psnr"]),
            "ssim_train": mean_metric(sample_metrics["train"]["ssim"]),
            "lpips_train": mean_metric(sample_metrics["train"]["lpips"]),
            "iou_train": mean_metric(sample_metrics["train"]["iou"]),
            "psnr_test": mean_metric(sample_metrics["test"]["psnr"]),
            "ssim_test": mean_metric(sample_metrics["test"]["ssim"]),
            "lpips_test": mean_metric(sample_metrics["test"]["lpips"]),
            "iou_test": mean_metric(sample_metrics["test"]["iou"]),
        }

    return {
        "scene": "OVERALL",
        "psnr_train": mean_metric([row["psnr_train"] for row in scene_rows]),
        "ssim_train": mean_metric([row["ssim_train"] for row in scene_rows]),
        "lpips_train": mean_metric([row["lpips_train"] for row in scene_rows]),
        "iou_train": mean_metric([row["iou_train"] for row in scene_rows]),
        "psnr_test": mean_metric([row["psnr_test"] for row in scene_rows]),
        "ssim_test": mean_metric([row["ssim_test"] for row in scene_rows]),
        "lpips_test": mean_metric([row["lpips_test"] for row in scene_rows]),
        "iou_test": mean_metric([row["iou_test"] for row in scene_rows]),
    }


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_outputs(rows, text_path, csv_path):
    ensure_parent(text_path)
    ensure_parent(csv_path)

    headers = [
        "scene",
        "psnr_train",
        "ssim_train",
        "lpips_train",
        "iou_train",
        "psnr_test",
        "ssim_test",
        "lpips_test",
        "iou_test",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            if csv_row["scene"] == "OVERALL":
                for field in headers[1:]:
                    csv_row[field] = f"{csv_row[field]:.3f}"
            writer.writerow(csv_row)

    with open(text_path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(f"{row['scene']}\n")
            file.write(f"  PSNR train: {row['psnr_train']:.6f}\n")
            file.write(f"  SSIM train: {row['ssim_train']:.6f}\n")
            file.write(f"  LPIPS train: {row['lpips_train']:.6f}\n")
            file.write(f"  IoU train: {row['iou_train']:.6f}\n")
            file.write(f"  PSNR test: {row['psnr_test']:.6f}\n")
            file.write(f"  SSIM test: {row['ssim_test']:.6f}\n")
            file.write(f"  LPIPS test: {row['lpips_test']:.6f}\n")
            file.write(f"  IoU test: {row['iou_test']:.6f}\n\n")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_path", default="./data/render_eval_data")
    parser.add_argument("--human_mask_path", default="./data/different_types_human_mask")
    parser.add_argument("--root_data_dir", default="./data/gaussian_data")
    parser.add_argument("--output_dir", default="./results/quality")
    parser.add_argument(
        "--text_output",
        default="./results/quality/metrics/render_metrics.txt",
    )
    parser.add_argument(
        "--csv_output",
        default="./results/quality/metrics/render_metrics.csv",
    )
    parser.add_argument("--num_views", type=int, default=1)
    parser.add_argument(
        "--overall_mode",
        choices=("scene_mean", "phystwin"),
        default="phystwin",
        help=(
            "scene_mean averages case-level rows equally; phystwin averages all "
            "evaluated frame/view samples directly."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    _ = args.root_data_dir

    if not os.path.isdir(args.render_path):
        raise FileNotFoundError(f"Render eval data not found: {args.render_path}")
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Prediction output dir not found: {args.output_dir}")

    device = torch.device("cuda")
    lpips_model = LPIPS().to(device)
    lpips_model.eval()

    rows = []
    sample_metrics = empty_metric_lists()
    for scene in sorted(os.listdir(args.render_path)):
        result = evaluate_scene(scene, args, lpips_model, device)
        if result is not None:
            row, scene_metrics = result
            rows.append(row)
            extend_metrics(sample_metrics, scene_metrics)

    if not rows:
        raise RuntimeError("No render metrics could be computed.")

    rows.append(overall_metrics(rows, sample_metrics, args.overall_mode))
    write_outputs(rows, args.text_output, args.csv_output)
    print(f"Saved render metrics to {args.text_output} and {args.csv_output}")


if __name__ == "__main__":
    main()
