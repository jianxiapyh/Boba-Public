import glob
import pickle
import json
import torch
import csv
import numpy as np
import os
from pytorch3d.loss import chamfer_distance
from argparse import ArgumentParser


def format_overall(value):
    return f"{value:.3f}"

def evaluate_prediction(
    start_frame,
    end_frame,
    vertices,
    object_points,
    object_visibilities,
    object_motions_valid,
    num_original_points,
    num_surface_points,
):
    chamfer_errors = []

    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(object_points, torch.Tensor):
        object_points = torch.tensor(object_points, dtype=torch.float32)
    if not isinstance(object_visibilities, torch.Tensor):
        object_visibilities = torch.tensor(object_visibilities, dtype=torch.bool)
    if not isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = torch.tensor(object_motions_valid, dtype=torch.bool)

    for frame_idx in range(start_frame, end_frame):
        x = vertices[frame_idx]
        current_object_points = object_points[frame_idx]
        current_object_visibilities = object_visibilities[frame_idx]
        # The motion valid indicates if the tracking is valid from prev_frame
        current_object_motions_valid = object_motions_valid[frame_idx - 1]

        # Compute the single-direction chamfer loss for the object points
        chamfer_object_points = current_object_points[current_object_visibilities]
        chamfer_x = x[:num_surface_points]
        # The GT chamfer_object_points can be partial,first find the nearest in second
        chamfer_error = chamfer_distance(
            chamfer_object_points.unsqueeze(0),
            chamfer_x.unsqueeze(0),
            single_directional=True,
            norm=1,  # Get the L1 distance
        )[0]

        chamfer_errors.append(chamfer_error.item())

    chamfer_errors = np.array(chamfer_errors)

    results = {
        "frame_len": len(chamfer_errors),
        "chamfer_error": np.mean(chamfer_errors),
    }

    return results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./data/different_types",
                        help="Path to ground truth data directory")
    parser.add_argument("--prediction_path", type=str,  default="./gaussian_output_dynamic",
                        help="Path to predicted results directory")
    parser.add_argument("--output_file", type=str, default="results/final_results.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    base_path = args.base_path
    prediction_dir = args.prediction_path
    output_file = args.output_file

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    file = open(output_file, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(file)

    writer.writerow(
        [
            "Case Name",
            "Train Frame Num",
            "Train Chamfer Error",
            "Test Frame Num",
            "Test Chamfer Error",
        ]
    )

    train_frame_counts = []
    test_frame_counts = []
    train_errors = []
    test_errors = []

    dir_names = glob.glob(f"{prediction_dir}/*")
    for dir_name in dir_names:
        case_name = dir_name.split("/")[-1]
        inference_path = f"{dir_name}/inference.pkl"
        if not os.path.isfile(inference_path):
            continue
        print(f"Processing {case_name}")

        # Read the trajectory data
        with open(inference_path, "rb") as f:
        #pyh use below for simulated cdense nodes
        #with open(f"{prediction_dir}/{case_name}/predicted_fine_nodes.pkl", "rb") as f:
            vertices = pickle.load(f)

        # Read the GT object points and masks
        with open(f"{base_path}/{case_name}/final_data.pkl", "rb") as f:
            data = pickle.load(f)

        object_points = data["object_points"]
        object_visibilities = data["object_visibilities"]
        object_motions_valid = data["object_motions_valid"]
        num_original_points = object_points.shape[1]
        num_surface_points = num_original_points + data["surface_points"].shape[0]

        # read the train/test split
        with open(f"{base_path}/{case_name}/split.json", "r") as f:
            split = json.load(f)
        train_frame = split["train"][1]
        test_frame = split["test"][1]

        assert (
            test_frame == vertices.shape[0]
        ), f"Test frame {test_frame} != {vertices.shape[0]}"

        # Do the statistics on train split, only evalaute from the 2nd frame
        results_train = evaluate_prediction(
            1,
            train_frame,
            vertices,
            object_points,
            object_visibilities,
            object_motions_valid,
            num_original_points,
            num_surface_points,
        )
        results_test = evaluate_prediction(
            train_frame,
            test_frame,
            vertices,
            object_points,
            object_visibilities,
            object_motions_valid,
            num_original_points,
            num_surface_points,
        )

        writer.writerow(
            [
                case_name,
                results_train["frame_len"],
                results_train["chamfer_error"],
                results_test["frame_len"],
                results_test["chamfer_error"],
            ]
        )

        train_frame_counts.append(results_train["frame_len"])
        test_frame_counts.append(results_test["frame_len"])
        train_errors.append(results_train["chamfer_error"])
        test_errors.append(results_test["chamfer_error"])

    if train_errors and test_errors:
        writer.writerow(
            [
                "OVERALL",
                int(sum(train_frame_counts)),
                format_overall(float(np.mean(train_errors))),
                int(sum(test_frame_counts)),
                format_overall(float(np.mean(test_errors))),
            ]
        )
    file.close()
