import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_BASE_PATH = "./data/different_types"
DEFAULT_PREDICTION_DIR = "./results/quality"
DEFAULT_HUMAN_MASK_PATH = "./data/different_types_human_mask"
EXPECTED_VIEWS = ("0", "1", "2")
ALPHA = 0.7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay raw render frames on whitened RGB frames for quality comparison."
    )
    parser.add_argument("--base_path", default=DEFAULT_BASE_PATH)
    parser.add_argument("--prediction_dir", default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--human_mask_path", default=DEFAULT_HUMAN_MASK_PATH)
    parser.add_argument("--case_name", default=None)
    return parser.parse_args()


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def safe_imread(path: Path, flags=cv2.IMREAD_UNCHANGED):
    return cv2.imread(str(path), flags)


def ensure_size(img, width, height, interpolation):
    if img is None:
        return None
    if img.shape[0] == height and img.shape[1] == width:
        return img
    return cv2.resize(img, (width, height), interpolation=interpolation)


def to_bool_mask(mask_img, width, height):
    if mask_img is None:
        return np.zeros((height, width), dtype=bool)
    if mask_img.ndim == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    mask_img = ensure_size(mask_img, width, height, cv2.INTER_NEAREST)
    return mask_img > 0


def whiten(origin_img):
    origin_f = origin_img.astype(np.float32)
    return ALPHA * origin_f + (1.0 - ALPHA) * 255.0


def load_render_rgba(render_path: Path, width, height):
    render_img = safe_imread(render_path, cv2.IMREAD_UNCHANGED)
    if render_img is None:
        return None
    render_img = ensure_size(render_img, width, height, cv2.INTER_LINEAR)
    if render_img.ndim == 2:
        render_rgb = cv2.cvtColor(render_img, cv2.COLOR_GRAY2BGR).astype(np.float32)
        render_alpha = (render_img > 5).astype(np.float32)
        return render_rgb, render_alpha
    if render_img.shape[2] >= 4:
        render_rgb = render_img[:, :, :3].astype(np.float32)
        render_alpha = render_img[:, :, 3].astype(np.float32) / 255.0
        valid_rgb = (render_rgb > 0).any(axis=2)
        valid_alpha = render_img[:, :, 3] > 100
        render_alpha = np.where(valid_rgb & valid_alpha, render_alpha, 0.0)
        return render_rgb, render_alpha
    render_rgb = render_img[:, :, :3].astype(np.float32)
    render_alpha = (render_rgb > 5).any(axis=2).astype(np.float32)
    return render_rgb, render_alpha


def iter_cases(base_path: Path, prediction_dir: Path, case_name):
    if case_name is not None:
        yield case_name
        return
    for case_dir in sorted(base_path.iterdir()):
        if not case_dir.is_dir():
            continue
        if (prediction_dir / case_dir.name).is_dir():
            yield case_dir.name


def process_case(case_name: str, base_path: Path, prediction_dir: Path, human_mask_path: Path):
    case_base = base_path / case_name
    case_prediction = prediction_dir / case_name

    if not case_base.is_dir():
        print(f"Skip case {case_name}: missing dataset folder {case_base}")
        return
    if not case_prediction.is_dir():
        print(f"Skip case {case_name}: missing prediction folder {case_prediction}")
        return

    metadata = load_json(case_base / "metadata.json")
    split = load_json(case_base / "split.json")
    width, height = metadata.get("WH", [848, 480])
    fps = float(metadata.get("fps", 30))
    frame_len = int(split["frame_len"])

    print(f"Processing {case_name} with fps={fps} and frame_len={frame_len}")

    for view in EXPECTED_VIEWS:
        render_dir = case_prediction / view
        if not render_dir.is_dir():
            print(f"Skip case {case_name} view {view}: missing render folder {render_dir}")
            continue

        output_frames_dir = case_prediction / f"{view}_integrate_frames"
        output_frames_dir.mkdir(parents=True, exist_ok=True)
        video_path = case_prediction / f"{view}_integrate.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {video_path}")

        missing_frames = []

        for frame_idx in range(frame_len):
            origin_path = case_base / "color" / view / f"{frame_idx}.png"
            human_mask_path_frame = (
                human_mask_path / case_name / "mask" / view / "0" / f"{frame_idx}.png"
            )
            render_path = render_dir / f"{frame_idx:05d}.png"

            origin_img = safe_imread(origin_path, cv2.IMREAD_COLOR)
            if origin_img is None:
                video_writer.release()
                raise FileNotFoundError(
                    f"Missing RGB frame for case={case_name} view={view} frame={frame_idx}: {origin_path}"
                )
            origin_img = ensure_size(origin_img, width, height, cv2.INTER_LINEAR)
            base_white = whiten(origin_img)

            human_mask = to_bool_mask(
                safe_imread(human_mask_path_frame, cv2.IMREAD_COLOR),
                width,
                height,
            )

            render_rgba = load_render_rgba(render_path, width, height)
            if render_rgba is None:
                print(
                    f"ERROR missing render frame: case={case_name} view={view} frame={frame_idx} path={render_path}"
                )
                missing_frames.append(frame_idx)
                final_image = base_white.copy()
            else:
                render_rgb, render_alpha = render_rgba
                final_image = (
                    render_rgb * render_alpha[:, :, None]
                    + base_white * (1.0 - render_alpha[:, :, None])
                )

            final_image[human_mask] = base_white[human_mask]
            final_u8 = np.clip(final_image, 0, 255).astype(np.uint8)

            cv2.imwrite(str(output_frames_dir / f"{frame_idx:05d}.png"), final_u8)
            video_writer.write(final_u8)

        video_writer.release()

        if missing_frames:
            joined = ", ".join(str(frame) for frame in missing_frames)
            print(
                f"Missing render summary for case={case_name} view={view}: {len(missing_frames)} frame(s): {joined}"
            )
        else:
            print(f"Completed case {case_name} view {view} with no missing render frames")


def main():
    args = parse_args()
    base_path = Path(args.base_path)
    prediction_dir = Path(args.prediction_dir)
    human_mask_path = Path(args.human_mask_path)

    for case_name in iter_cases(base_path, prediction_dir, args.case_name):
        process_case(case_name, base_path, prediction_dir, human_mask_path)


if __name__ == "__main__":
    main()
