import pickle
import glob
import csv
import json
import numpy as np
import os
from scipy.spatial import KDTree
from argparse import ArgumentParser


def format_overall(value):
    return f"{value:.3f}"


def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        # Get the new mask and see
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0
        else:
            track_error = np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1))
        
        track_errors.append(track_error)
    return np.mean(track_errors)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="./data/different_types",
                        help="Path to ground truth data directory")
    parser.add_argument("--prediction_path", type=str,  default="./gaussian_output_dynamic",
                        help="Path to predicted results directory")
    parser.add_argument("--output_file", type=str, default="results/final_track.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    base_path = args.base_path
    prediction_path = args.prediction_path
    output_file = args.output_file

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    file = open(output_file, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(
        [
            "Case Name",
            "Train Track Error",
            "Test Track Error",
        ]
    )

    train_errors = []
    test_errors = []

    dir_names = glob.glob(f"{prediction_path}/*")
    for dir_name in dir_names:
        case_name = dir_name.split("/")[-1]
        inference_path = f"{prediction_path}/{case_name}/inference.pkl"
        if not os.path.isfile(inference_path):
            continue
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")

        with open(f"{base_path}/{case_name}/split.json", "r") as f:
            split = json.load(f)
        frame_len = split["frame_len"]
        train_frame = split["train"][1]
        test_frame = split["test"][1]

        with open(inference_path, "rb") as f:
        #pyh use below for simulated dense nodes
        #with open(f"{prediction_path}/{case_name}/predicted_fine_nodes.pkl", "rb") as f:
            vertices = pickle.load(f)

        with open(f"{base_path}/{case_name}/gt_track_3d.pkl", "rb") as f:
            gt_track_3d = pickle.load(f)

        # Locate the index of corresponding point index in the vertices, if nan, then ignore the points
        mask = ~np.isnan(gt_track_3d[0]).any(axis=1)

        kdtree = KDTree(vertices[0])
        dis, idx = kdtree.query(gt_track_3d[0][mask])

        train_track_error = evaluate_prediction(
            1, train_frame, vertices, gt_track_3d, idx, mask
        )
        test_track_error = evaluate_prediction(
            train_frame, test_frame, vertices, gt_track_3d, idx, mask
        )
        writer.writerow([case_name, train_track_error, test_track_error])

        train_errors.append(train_track_error)
        test_errors.append(test_track_error)

    if train_errors and test_errors:
        writer.writerow(
            [
                "OVERALL",
                format_overall(float(np.mean(train_errors))),
                format_overall(float(np.mean(test_errors))),
            ]
        )
    file.close()
