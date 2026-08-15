#!/usr/bin/env python
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


CONTROLLER_NAME = "hand"
DEFAULT_VALIDATION_CASES = ("single_lift_rope", "double_lift_cloth_3")
METRIC_FIELDS = (
    "psnr_train",
    "ssim_train",
    "lpips_train",
    "iou_train",
    "psnr_test",
    "ssim_test",
    "lpips_test",
    "iou_test",
)
cv2 = None
np = None


def load_image_dependencies():
    global cv2, np
    if cv2 is not None and np is not None:
        return

    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ImportError as exc:
        raise RuntimeError(
            "prepare_additional_render_eval_data.py requires cv2 and numpy. "
            "Run it inside the PhysTwin environment."
        ) from exc

    cv2 = cv2_module
    np = np_module


def read_cases(cases_file, explicit_cases):
    if explicit_cases:
        return list(explicit_cases)

    cases = []
    with open(cases_file, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            case_name = row[0].strip()
            if case_name:
                cases.append(case_name)
    return cases


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def copytree_merge(src, dst):
    if not src.is_dir():
        raise FileNotFoundError(f"Missing directory: {src}")
    shutil.copytree(src, dst, dirs_exist_ok=True)


def read_mask_info(case_path, view_idx):
    info_path = case_path / "mask" / f"mask_info_{view_idx}.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing mask info: {info_path}")
    with open(info_path, "r", encoding="utf-8") as file:
        return json.load(file)


def object_mask_id(mask_info, case_name, view_idx):
    object_ids = [
        int(mask_id)
        for mask_id, label in mask_info.items()
        if label != CONTROLLER_NAME
    ]
    if len(object_ids) != 1:
        raise ValueError(
            f"{case_name} view {view_idx}: expected exactly one object mask, "
            f"found {object_ids}"
        )
    return object_ids[0]


def hand_mask_ids(mask_info, case_name, view_idx):
    hand_ids = [
        int(mask_id)
        for mask_id, label in mask_info.items()
        if label == CONTROLLER_NAME
    ]
    if not hand_ids:
        raise ValueError(f"{case_name} view {view_idx}: no hand masks found")
    return hand_ids


def mask_files_for_ids(case_path, view_idx, mask_ids):
    frame_names = set()
    for mask_id in mask_ids:
        mask_dir = case_path / "mask" / str(view_idx) / str(mask_id)
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing mask directory: {mask_dir}")
        frame_names.update(path.name for path in mask_dir.glob("*.png"))
    return sorted(frame_names, key=lambda name: int(Path(name).stem))


def read_binary_mask(mask_path):
    load_image_dependencies()
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask image: {mask_path}")
    return mask > 0


def write_binary_mask(mask, output_path):
    load_image_dependencies()
    ensure_dir(output_path.parent)
    image = (mask.astype(np.uint8) * 255)
    if not cv2.imwrite(str(output_path), image):
        raise ValueError(f"Could not write mask image: {output_path}")


def build_human_masks(case_path, output_case_path):
    for view_idx in range(3):
        mask_info = read_mask_info(case_path, view_idx)
        hand_ids = hand_mask_ids(mask_info, case_path.name, view_idx)
        frame_names = mask_files_for_ids(case_path, view_idx, hand_ids)

        for frame_name in frame_names:
            union_mask = None
            for hand_id in hand_ids:
                mask_path = case_path / "mask" / str(view_idx) / str(hand_id) / frame_name
                if not mask_path.is_file():
                    continue
                current_mask = read_binary_mask(mask_path)
                if union_mask is None:
                    union_mask = current_mask
                elif union_mask.shape != current_mask.shape:
                    raise ValueError(
                        f"Shape mismatch while unioning masks for {case_path.name} "
                        f"view {view_idx} frame {frame_name}"
                    )
                else:
                    union_mask = np.logical_or(union_mask, current_mask)

            if union_mask is None:
                raise FileNotFoundError(
                    f"No hand mask files found for {case_path.name} "
                    f"view {view_idx} frame {frame_name}"
                )

            write_binary_mask(
                union_mask,
                output_case_path / "mask" / str(view_idx) / "0" / frame_name,
            )


def build_render_eval_case(case_path, output_case_path):
    ensure_dir(output_case_path)
    ensure_dir(output_case_path / "mask")

    copytree_merge(case_path / "color", output_case_path / "color")

    for view_idx in range(3):
        mask_info = read_mask_info(case_path, view_idx)
        obj_idx = object_mask_id(mask_info, case_path.name, view_idx)
        copytree_merge(
            case_path / "mask" / str(view_idx) / str(obj_idx),
            output_case_path / "mask" / str(view_idx),
        )

    split_path = case_path / "split.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    shutil.copy2(split_path, output_case_path / "split.json")


def remove_case_outputs(case_name, render_output_path, human_mask_output_path):
    for root in (render_output_path, human_mask_output_path):
        case_output = root / case_name
        if case_output.exists():
            shutil.rmtree(case_output)


def prepare_cases(
    case_names,
    base_path,
    render_output_path,
    human_mask_output_path,
    clean=True,
):
    ensure_dir(render_output_path)
    ensure_dir(human_mask_output_path)

    for case_name in case_names:
        case_path = base_path / case_name
        if not case_path.is_dir():
            raise FileNotFoundError(f"Missing case directory: {case_path}")

        if clean:
            remove_case_outputs(case_name, render_output_path, human_mask_output_path)

        print(f"Preparing render-eval data for {case_name}")
        build_render_eval_case(case_path, render_output_path / case_name)
        build_human_masks(case_path, human_mask_output_path / case_name)


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_files(root):
    return sorted(path.relative_to(root) for path in root.rglob("*") if path.is_file())


def compare_file_trees_bytewise(generated_root, provided_root, label):
    generated_files = relative_files(generated_root)
    provided_files = relative_files(provided_root)
    if generated_files != provided_files:
        raise AssertionError(
            f"{label}: file set mismatch for {generated_root} vs {provided_root}"
        )

    for rel_path in generated_files:
        generated_file = generated_root / rel_path
        provided_file = provided_root / rel_path
        if file_digest(generated_file) != file_digest(provided_file):
            raise AssertionError(
                f"{label}: byte mismatch for {generated_file} vs {provided_file}"
            )


def compare_binary_mask_files(generated_file, provided_file, label):
    if not provided_file.is_file():
        raise AssertionError(f"{label}: missing provided mask {provided_file}")

    generated_mask = read_binary_mask(generated_file)
    provided_mask = read_binary_mask(provided_file)
    if generated_mask.shape != provided_mask.shape:
        raise AssertionError(
            f"{label}: shape mismatch for {generated_file} vs {provided_file}"
        )
    if not np.array_equal(generated_mask, provided_mask):
        intersection = np.logical_and(generated_mask, provided_mask)
        generated_only = np.logical_and(generated_mask, np.logical_not(provided_mask))
        provided_only = np.logical_and(provided_mask, np.logical_not(generated_mask))
        raise AssertionError(
            f"{label}: pixel mismatch for {generated_file} vs {provided_file}; "
            f"generated_pixels={int(generated_mask.sum())}, "
            f"provided_pixels={int(provided_mask.sum())}, "
            f"intersection={int(intersection.sum())}, "
            f"generated_only={int(generated_only.sum())}, "
            f"provided_only={int(provided_only.sum())}"
        )


def compare_mask_trees_binary(generated_root, provided_root, label):
    generated_files = relative_files(generated_root)
    provided_files = relative_files(provided_root)
    if generated_files != provided_files:
        raise AssertionError(
            f"{label}: file set mismatch for {generated_root} vs {provided_root}"
        )

    for rel_path in generated_files:
        compare_binary_mask_files(
            generated_root / rel_path,
            provided_root / rel_path,
            label,
        )


def compare_eval_human_masks(generated_root, provided_root, label):
    generated_files = relative_files(generated_root)
    provided_files = [
        rel_path
        for rel_path in relative_files(provided_root)
        if len(rel_path.parts) >= 3 and rel_path.parts[1] == "0"
    ]
    if generated_files != provided_files:
        raise AssertionError(
            f"{label}: eval human-mask file set mismatch for "
            f"{generated_root} vs {provided_root}"
        )

    for rel_path in generated_files:
        compare_binary_mask_files(
            generated_root / rel_path,
            provided_root / rel_path,
            label,
        )


def validate_existing(args):
    validation_cases = list(args.validation_cases)
    with tempfile.TemporaryDirectory(prefix="phystwin_render_eval_validation_") as tmp:
        tmp_path = Path(tmp)
        generated_render_path = tmp_path / "render_eval"
        generated_human_path = tmp_path / "human_mask"

        prepare_cases(
            validation_cases,
            args.base_path,
            generated_render_path,
            generated_human_path,
            clean=True,
        )

        for case_name in validation_cases:
            generated_render_case = generated_render_path / case_name
            provided_render_case = args.provided_render_eval_path / case_name
            generated_human_case = generated_human_path / case_name
            provided_human_case = args.provided_human_mask_path / case_name

            compare_file_trees_bytewise(
                generated_render_case / "color",
                provided_render_case / "color",
                f"{case_name} color",
            )
            if file_digest(generated_render_case / "split.json") != file_digest(
                provided_render_case / "split.json"
            ):
                raise AssertionError(f"{case_name} split.json byte mismatch")
            compare_mask_trees_binary(
                generated_render_case / "mask",
                provided_render_case / "mask",
                f"{case_name} object mask",
            )
            compare_eval_human_masks(
                generated_human_case / "mask",
                provided_human_case / "mask",
                f"{case_name} human mask",
            )

    print(
        "Validation passed: generated render-eval data matches provided data for "
        + ", ".join(validation_cases)
    )


def has_prediction(case_name, prediction_root):
    case_root = prediction_root / case_name
    return case_root.is_dir() and any((case_root / "0").glob("*.png"))


def copy_existing_predictions(validation_cases, source_root, destination_root):
    copied_cases = []
    if not source_root.is_dir():
        return copied_cases

    ensure_dir(destination_root)
    for case_name in validation_cases:
        if not has_prediction(case_name, source_root):
            continue
        shutil.copytree(
            source_root / case_name,
            destination_root / case_name,
            dirs_exist_ok=True,
        )
        copied_cases.append(case_name)
    return copied_cases


def run_quality_prediction(case_name, args, output_dir):
    command = [
        sys.executable,
        "interactive_playground.py",
        "--mode",
        "quality",
        "--base_path",
        str(args.base_path),
        "--gaussian_path",
        str(args.gaussian_path),
        "--bg_img_path",
        str(args.bg_img_path),
        "--case_name",
        case_name,
        "--num_views",
        str(args.num_views),
        "--output_dir",
        str(output_dir),
    ]
    print(f"Generating temporary quality prediction for {case_name}")
    subprocess.run(command, check=True, env=subprocess_env())


def subprocess_env():
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return env


def ensure_validation_predictions(validation_cases, args, prediction_root):
    copied_cases = copy_existing_predictions(
        validation_cases,
        args.prediction_path,
        prediction_root,
    )
    copied_set = set(copied_cases)
    if copied_cases:
        print(
            "Using existing quality predictions for "
            + ", ".join(sorted(copied_cases))
        )

    for case_name in validation_cases:
        if case_name in copied_set:
            continue
        run_quality_prediction(case_name, args, prediction_root / case_name)


def run_render_metrics(render_path, human_mask_path, output_dir, metrics_dir, args):
    text_output = metrics_dir / "render_metrics.txt"
    csv_output = metrics_dir / "render_metrics.csv"
    command = [
        sys.executable,
        "gaussian_splatting/evaluate_render.py",
        "--render_path",
        str(render_path),
        "--human_mask_path",
        str(human_mask_path),
        "--output_dir",
        str(output_dir),
        "--num_views",
        str(args.num_views),
        "--overall_mode",
        args.overall_mode,
        "--text_output",
        str(text_output),
        "--csv_output",
        str(csv_output),
    ]
    subprocess.run(command, check=True, env=subprocess_env())
    return csv_output


def read_metric_csv(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = {}
        for row in reader:
            scene = row["scene"]
            rows[scene] = row
    return rows


def parse_metric(value, field, scene):
    try:
        return float(value)
    except ValueError as exc:
        raise AssertionError(
            f"Could not parse metric {field} for scene {scene}: {value!r}"
        ) from exc


def compare_metric_csvs(provided_csv, generated_csv, tolerance):
    provided_rows = read_metric_csv(provided_csv)
    generated_rows = read_metric_csv(generated_csv)

    if set(provided_rows) != set(generated_rows):
        raise AssertionError(
            "Render metric scene mismatch: "
            f"provided={sorted(provided_rows)}, generated={sorted(generated_rows)}"
        )

    mismatches = []
    for scene in sorted(provided_rows):
        for field in METRIC_FIELDS:
            provided_value = parse_metric(provided_rows[scene][field], field, scene)
            generated_value = parse_metric(generated_rows[scene][field], field, scene)
            if math.isnan(provided_value) and math.isnan(generated_value):
                continue
            delta = abs(provided_value - generated_value)
            if delta > tolerance:
                mismatches.append(
                    (
                        scene,
                        field,
                        provided_value,
                        generated_value,
                        delta,
                    )
                )

    if mismatches:
        preview = "\n".join(
            (
                f"{scene} {field}: provided={provided_value:.9g}, "
                f"generated={generated_value:.9g}, delta={delta:.9g}"
            )
            for scene, field, provided_value, generated_value, delta in mismatches[:20]
        )
        raise AssertionError(
            f"Render metrics differ beyond tolerance {tolerance}.\n{preview}"
        )


def validate_metric_equivalence(args):
    validation_cases = list(args.validation_cases)
    with tempfile.TemporaryDirectory(prefix="phystwin_render_metric_validation_") as tmp:
        tmp_path = Path(tmp)
        generated_render_path = tmp_path / "generated_render_eval"
        generated_human_path = tmp_path / "generated_human_mask"
        prediction_root = tmp_path / "predictions"
        provided_metrics_dir = tmp_path / "provided_metrics"
        generated_metrics_dir = tmp_path / "generated_metrics"

        prepare_cases(
            validation_cases,
            args.base_path,
            generated_render_path,
            generated_human_path,
            clean=True,
        )
        ensure_validation_predictions(validation_cases, args, prediction_root)

        provided_csv = run_render_metrics(
            args.provided_render_eval_path,
            args.provided_human_mask_path,
            prediction_root,
            provided_metrics_dir,
            args,
        )
        generated_csv = run_render_metrics(
            generated_render_path,
            generated_human_path,
            prediction_root,
            generated_metrics_dir,
            args,
        )

        compare_metric_csvs(
            provided_csv,
            generated_csv,
            args.metric_tolerance,
        )

    print(
        "Validation passed: generated render-eval inputs produce matching render "
        "metrics for " + ", ".join(validation_cases)
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases_file", default="benchmarks/additional_data_config.csv")
    parser.add_argument("--base_path", type=Path, default=Path("./data/different_types"))
    parser.add_argument(
        "--render_output_path",
        type=Path,
        default=Path("./data/render_eval_data_additional"),
    )
    parser.add_argument(
        "--human_mask_output_path",
        type=Path,
        default=Path("./data/different_types_human_mask_additional"),
    )
    parser.add_argument("--no_clean", action="store_true")
    parser.add_argument("--validate_existing", action="store_true")
    parser.add_argument("--validate_metrics", action="store_true")
    parser.add_argument(
        "--validation_cases",
        nargs="+",
        default=list(DEFAULT_VALIDATION_CASES),
    )
    parser.add_argument(
        "--prediction_path",
        type=Path,
        default=Path("./results/quality"),
        help=(
            "Existing prediction root to reuse for --validate_metrics. Missing "
            "validation cases are generated in a temporary directory."
        ),
    )
    parser.add_argument("--gaussian_path", type=Path, default=Path("./gaussian_output"))
    parser.add_argument("--bg_img_path", type=Path, default=Path("./data/bg.png"))
    parser.add_argument("--num_views", type=int, default=1)
    parser.add_argument(
        "--overall_mode",
        choices=("scene_mean", "phystwin"),
        default="phystwin",
    )
    parser.add_argument("--metric_tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--provided_render_eval_path",
        type=Path,
        default=Path("./data/render_eval_data"),
    )
    parser.add_argument(
        "--provided_human_mask_path",
        type=Path,
        default=Path("./data/different_types_human_mask"),
    )
    parser.add_argument("cases", nargs="*")
    return parser


def main():
    args = build_parser().parse_args()

    if args.validate_existing:
        validate_existing(args)
        return

    if args.validate_metrics:
        validate_metric_equivalence(args)
        return

    cases = read_cases(args.cases_file, args.cases)
    if not cases:
        raise ValueError("No cases selected.")

    prepare_cases(
        cases,
        args.base_path,
        args.render_output_path,
        args.human_mask_output_path,
        clean=not args.no_clean,
    )


if __name__ == "__main__":
    main()
