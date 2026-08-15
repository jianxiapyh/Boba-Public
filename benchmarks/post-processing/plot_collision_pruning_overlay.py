#!/usr/bin/env python3
"""Create a real-data collision-pruning adjacency figure for double_stretch_sloth."""

import argparse
import ast
from itertools import combinations
import json
import math
from pathlib import Path
import struct

import numpy as np


DEFAULT_EXPORT_NPZ = (
    "results/figures/collision_pruning/"
    "double_stretch_sloth_rest_motion_adjacency_export.npz"
)
MIN_REST_LENGTH = 1e-4
GRAPH_SPRING_COLOR = "#D6A100"
GRAPH_NODE_COLOR = "#4E7D68"
ZOOM_RED_RADIUS_FRACTION_TARGET = 0.70
CURATED_SELECTED_NODES = (1280, 1312, 1328)
ZOOM_GAUSSIAN_KERNEL_MAX_ELLIPSES = 800
ZOOM_GAUSSIAN_KERNEL_WIDTH_PX = 0.48
ZOOM_GAUSSIAN_KERNEL_HEIGHT_PX = 0.20
ZOOM_GAUSSIAN_KERNEL_FACE_ALPHA = 0.13
ZOOM_GAUSSIAN_KERNEL_EDGE_ALPHA = 0.18
ZOOM_GAUSSIAN_KERNEL_EDGE_LINEWIDTH = 0.045
ZOOM_GAUSSIAN_KERNEL_SIGMA_LEVEL = 2.0


def repo_root():
    return Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Draw frame-41 and frame-0 Gaussian point clouds with the same selected "
            "rest-neighbor spring-mass patch overlaid."
        )
    )
    parser.add_argument("--case_name", default="double_stretch_sloth")
    parser.add_argument("--export_npz", default=DEFAULT_EXPORT_NPZ)
    parser.add_argument("--config", default="configs/real.yaml")
    parser.add_argument("--output_dir", default="results/figures/collision_pruning")
    parser.add_argument(
        "--output_stem",
        default="double_stretch_sloth_rest_motion_adjacency",
    )
    parser.add_argument(
        "--layout",
        choices=(
            "zoom_inset",
            "full_gaussian_highlight",
            "leg_focus_highlight",
            "rest_map_anchor",
        ),
        default="zoom_inset",
        help=(
            "Figure layout. 'zoom_inset' preserves the original circular zoom view; "
            "'full_gaussian_highlight' shows full-object Gaussian panels with the "
            "selected spring patch highlighted directly; 'leg_focus_highlight' "
            "shows a larger raw-camera crop around the selected leg patch with "
            "uniform mass-node and spring sizes; 'rest_map_anchor' shows one "
            "anchor node, its generated spring-star, and rest-map skipped neighbors."
        ),
    )
    parser.add_argument("--rest_frame", type=int, default=0)
    parser.add_argument("--contact_frame", type=int, default=41)
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument(
        "--selected_nodes",
        default=",".join(str(node) for node in CURATED_SELECTED_NODES),
        help=(
            "Optional comma-separated 3-5 node IDs for the highlighted spring patch. "
            "Defaults to the curated compact right-leg patch."
        ),
    )
    parser.add_argument(
        "--search_top_quantile",
        type=float,
        default=0.35,
        help="Projected y quantile used to isolate the raised leg search band.",
    )
    parser.add_argument(
        "--right_leg_margin_px",
        type=float,
        default=35.0,
        help="Minimum projected x offset from the raised-leg midline for automatic right-leg selection.",
    )
    parser.add_argument(
        "--torso_exclusion_y_quantile",
        type=float,
        default=0.42,
        help="Candidate centers must be above this projected y quantile to avoid torso/body-junction picks.",
    )
    parser.add_argument(
        "--preferred_patch_span_px",
        type=float,
        default=8.0,
        help="Preferred projected span for a readable local spring patch.",
    )
    parser.add_argument("--min_patch_span_px", type=float, default=8.0)
    parser.add_argument("--max_patch_span_px", type=float, default=38.0)
    parser.add_argument("--visual_pad_px", type=float, default=0.50)
    parser.add_argument("--crop_size_px", type=float, default=15.0)
    parser.add_argument(
        "--local_radius_px",
        type=float,
        default=19.0,
        help="Maximum projected radius from a seed node when proposing patch members.",
    )
    parser.add_argument(
        "--max_local_nodes",
        type=int,
        default=5,
        help="Maximum selected spring-patch nodes; the renderer draws no extra context nodes.",
    )
    parser.add_argument("--min_local_nodes", type=int, default=3)
    parser.add_argument("--candidate_neighbor_limit", type=int, default=9)
    parser.add_argument(
        "--leg_crop_size_px",
        type=float,
        default=30.0,
        help="Square raw-camera crop size for --layout leg_focus_highlight.",
    )
    parser.add_argument(
        "--selected_patch_max_nodes",
        type=int,
        default=16,
        help=(
            "Maximum connected visible spring-graph nodes to highlight for "
            "--layout leg_focus_highlight."
        ),
    )
    parser.add_argument(
        "--anchor_node",
        type=int,
        default=2351,
        help="Anchor mass-node ID for --layout rest_map_anchor.",
    )
    parser.add_argument(
        "--anchor_crop_size_px",
        type=float,
        default=45.0,
        help="Square raw-camera crop size for --layout rest_map_anchor.",
    )
    parser.add_argument(
        "--structural_radius_px",
        type=float,
        default=7.5,
        help=(
            "Anchor-local visual radius in pixels for selecting the "
            "rest-map pruning region in --layout rest_map_anchor."
        ),
    )
    parser.add_argument(
        "--render_zoom_crop_size_px",
        type=float,
        default=150.0,
        help=(
            "Square raw-camera crop size for the transparent rendered zoom "
            "inset in --layout rest_map_anchor."
        ),
    )
    parser.add_argument("--max_gaussian_points_per_panel", type=int, default=36000)
    parser.add_argument("--gaussian_color_strength", type=float, default=0.58)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def resolve_path(path, root):
    path = Path(path)
    if path.is_absolute():
        return path
    return root / path


def boba_quality_render_path(root, case_name, camera_index, frame):
    return (
        root
        / "results"
        / "quality"
        / str(case_name)
        / str(int(camera_index))
        / f"{int(frame):05d}.png"
    )


def png_dimensions(path):
    with Path(path).open("rb") as file:
        header = file.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Expected a PNG image with an IHDR chunk: {path}")
    return struct.unpack(">II", header[16:24])


def parse_scalar(text):
    text = text.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if text.startswith("[") and text.endswith("]"):
        return ast.literal_eval(text)
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text.strip("\"'")


def read_config(path):
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except Exception:
        config = {}
        with open(path, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                config[key.strip()] = parse_scalar(value)
        return config


def parse_selected_nodes(text):
    if text is None:
        return None
    nodes = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not 3 <= len(nodes) <= 5 or len(set(nodes)) != len(nodes):
        raise ValueError("--selected_nodes must contain 3-5 distinct node IDs")
    return tuple(nodes)


def import_ckdtree():
    try:
        from scipy.spatial import cKDTree

        return cKDTree
    except Exception:
        return None


def build_spring_graph(points, radius, max_neighbours):
    cKDTree = import_ckdtree()
    edges = []
    seen = set()
    n_points = points.shape[0]

    if cKDTree is not None:
        tree = cKDTree(points)
        dists, indices = tree.query(
            points,
            k=max_neighbours,
            distance_upper_bound=radius,
            workers=-1,
        )
        if max_neighbours == 1:
            dists = dists[:, None]
            indices = indices[:, None]
        for i in range(n_points):
            for dist, j in zip(dists[i], indices[i]):
                if j >= n_points or j == i or not math.isfinite(float(dist)):
                    continue
                if dist <= MIN_REST_LENGTH or dist > radius:
                    continue
                edge = (i, int(j)) if i < int(j) else (int(j), i)
                if edge not in seen:
                    seen.add(edge)
                    edges.append(edge)
    else:
        for i in range(n_points):
            diffs = points - points[i]
            dist_sq = np.einsum("ij,ij->i", diffs, diffs)
            nearest = np.argsort(dist_sq)[:max_neighbours]
            for j in nearest:
                if j == i:
                    continue
                dist = math.sqrt(float(dist_sq[j]))
                if dist <= MIN_REST_LENGTH or dist > radius:
                    continue
                edge = (i, int(j)) if i < int(j) else (int(j), i)
                if edge not in seen:
                    seen.add(edge)
                    edges.append(edge)

    adjacency = [set() for _ in range(n_points)]
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    return np.asarray(edges, dtype=np.int64), adjacency


def frame_index(frames, frame):
    matches = np.flatnonzero(frames == frame)
    if matches.size == 0:
        raise ValueError(f"Frame {frame} was not exported. Available frames: {frames.tolist()}")
    return int(matches[0])


def project_points(points, c2w, intrinsic):
    w2c = np.linalg.inv(c2w)
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (z > 1e-6)
    xy = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    xy[valid, 0] = intrinsic[0, 0] * camera[valid, 0] / z[valid] + intrinsic[0, 2]
    xy[valid, 1] = intrinsic[1, 1] * camera[valid, 1] / z[valid] + intrinsic[1, 2]
    return xy, valid


def count_gaussians_near(gaussian_xy, center, radius):
    if gaussian_xy is None or gaussian_xy.size == 0:
        return 0
    deltas = gaussian_xy - center
    dists = np.einsum("ij,ij->i", deltas, deltas)
    return int(np.count_nonzero(dists <= radius * radius))


def induced_edges(nodes, adjacency):
    node_set = set(int(node) for node in nodes)
    edges = []
    for i in sorted(node_set):
        for j in sorted(adjacency[i]):
            if j in node_set and i < j:
                edges.append((i, j))
    return np.asarray(edges, dtype=np.int64)


def is_connected_patch(nodes, edges):
    nodes = [int(node) for node in nodes]
    if not nodes:
        return False
    graph = {node: set() for node in nodes}
    for i, j in edges:
        graph[int(i)].add(int(j))
        graph[int(j)].add(int(i))
    seen = {nodes[0]}
    stack = [nodes[0]]
    while stack:
        node = stack.pop()
        for neighbour in graph[node]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    return len(seen) == len(nodes)


def spring_length_metrics(edges, rest_nodes, contact_nodes):
    rows = []
    rest_lengths = []
    contact_lengths = []
    ratios = []
    rel_changes = []
    for i, j in edges:
        rest_length = float(np.linalg.norm(rest_nodes[int(i)] - rest_nodes[int(j)]))
        contact_length = float(np.linalg.norm(contact_nodes[int(i)] - contact_nodes[int(j)]))
        if rest_length <= MIN_REST_LENGTH:
            continue
        ratio = contact_length / rest_length
        rel_change = abs(ratio - 1.0)
        rows.append(
            {
                "edge": [int(i), int(j)],
                "rest_length": rest_length,
                "contact_length": contact_length,
                "length_ratio_contact_over_rest": float(ratio),
                "relative_length_change": float(rel_change),
            }
        )
        rest_lengths.append(rest_length)
        contact_lengths.append(contact_length)
        ratios.append(ratio)
        rel_changes.append(rel_change)

    if rel_changes:
        summary = {
            "edge_count": len(rel_changes),
            "median_relative_length_change": float(np.median(rel_changes)),
            "max_relative_length_change": float(np.max(rel_changes)),
            "median_length_ratio_contact_over_rest": float(np.median(ratios)),
            "min_length_ratio_contact_over_rest": float(np.min(ratios)),
            "max_length_ratio_contact_over_rest": float(np.max(ratios)),
        }
    else:
        summary = {
            "edge_count": 0,
            "median_relative_length_change": None,
            "max_relative_length_change": None,
            "median_length_ratio_contact_over_rest": None,
            "min_length_ratio_contact_over_rest": None,
            "max_length_ratio_contact_over_rest": None,
        }
    return rows, summary


def patch_spans(rest_xy, contact_xy, nodes):
    pts_contact = contact_xy[list(nodes)]
    pts_rest = rest_xy[list(nodes)]
    contact_span = float(np.max(np.ptp(pts_contact, axis=0)))
    rest_span = float(np.max(np.ptp(pts_rest, axis=0)))
    return rest_span, contact_span


def iter_seed_patches(pool, adjacency, contact_xy, args):
    pool_set = set(int(item) for item in pool)
    seen = set()
    min_nodes = max(3, int(args.min_local_nodes))
    max_nodes = min(5, max(min_nodes, int(args.max_local_nodes)))
    for seed in sorted(pool_set):
        neighbours = [
            int(node)
            for node in adjacency[seed]
            if int(node) in pool_set
            and np.isfinite(contact_xy[int(node)]).all()
            and np.linalg.norm(contact_xy[int(node)] - contact_xy[seed]) <= args.local_radius_px
        ]
        neighbours.sort(key=lambda node: float(np.linalg.norm(contact_xy[node] - contact_xy[seed])))
        neighbours = neighbours[: int(args.candidate_neighbor_limit)]
        for size in range(max_nodes, min_nodes - 1, -1):
            if len(neighbours) < size - 1:
                continue
            for combo in combinations(neighbours, size - 1):
                patch = tuple(sorted((seed, *combo)))
                key = frozenset(patch)
                if key in seen:
                    continue
                seen.add(key)
                yield patch


def choose_leg_patch(
    rest_xy,
    contact_xy,
    rest_valid,
    contact_valid,
    adjacency,
    gaussian_contact_xy,
    rest_nodes,
    contact_nodes,
    args,
):
    override = parse_selected_nodes(args.selected_nodes)
    if override is not None:
        edges = induced_edges(override, adjacency)
        if edges.shape[0] < len(override) - 1 or not is_connected_patch(override, edges):
            raise ValueError("--selected_nodes must form a connected rest-spring patch")
        edge_lengths, length_summary = spring_length_metrics(edges, rest_nodes, contact_nodes)
        return override, edges, {
            "mode": "manual",
            "selected_nodes": list(override),
            "selected_rest_edges": edges.astype(int).tolist(),
            "spring_length_summary": length_summary,
            "spring_lengths": edge_lengths,
        }

    visible = rest_valid & contact_valid
    visible_indices = np.flatnonzero(visible)
    if visible_indices.size < 3:
        raise RuntimeError("Not enough projected visible mass nodes for patch search")

    torso_y_cut = float(
        np.quantile(contact_xy[visible, 1], float(args.torso_exclusion_y_quantile))
    )
    quantiles = [
        float(args.search_top_quantile),
        0.65,
        0.75,
        0.85,
        1.0,
    ]
    candidates = []
    chosen_quantile = None
    margin_candidates = [
        float(args.right_leg_margin_px),
        float(args.right_leg_margin_px) * 0.5,
        0.0,
    ]
    for quantile in quantiles:
        y_cut = float(np.quantile(contact_xy[visible, 1], quantile))
        top = visible & (contact_xy[:, 1] <= y_cut)
        top_indices = np.flatnonzero(top)
        if top_indices.size < 6:
            continue
        x_split = float(np.median(contact_xy[top_indices, 0]))
        for x_margin in margin_candidates:
            right_leg = top & (contact_xy[:, 0] >= x_split + x_margin)
            left_leg = top & (contact_xy[:, 0] <= x_split - x_margin)
            pool = np.flatnonzero(right_leg)
            opposite_xy = contact_xy[np.flatnonzero(left_leg)]
            if pool.size < 3 or opposite_xy.size == 0:
                continue

            for patch in iter_seed_patches(pool, adjacency, contact_xy, args):
                patch_nodes = np.asarray(patch, dtype=np.int64)
                pts_contact = contact_xy[patch_nodes]
                pts_rest = rest_xy[patch_nodes]
                if not np.isfinite(pts_contact).all() or not np.isfinite(pts_rest).all():
                    continue

                contact_span = float(np.max(np.ptp(pts_contact, axis=0)))
                rest_span = float(np.max(np.ptp(pts_rest, axis=0)))
                if contact_span < args.min_patch_span_px or rest_span < args.min_patch_span_px:
                    continue
                if contact_span > args.max_patch_span_px or rest_span > args.max_patch_span_px:
                    continue

                patch_edges = induced_edges(patch_nodes, adjacency)
                if patch_edges.shape[0] < len(patch_nodes) - 1:
                    continue
                if not is_connected_patch(patch_nodes, patch_edges):
                    continue
                spring_lengths, length_summary = spring_length_metrics(
                    patch_edges, rest_nodes, contact_nodes
                )
                if length_summary["edge_count"] < len(patch_nodes) - 1:
                    continue

                center = pts_contact.mean(axis=0)
                if center[1] > torso_y_cut:
                    continue
                opposite_distance = float(np.min(np.linalg.norm(opposite_xy - center, axis=1)))
                gaussian_count = count_gaussians_near(
                    gaussian_contact_xy,
                    center,
                    max(args.crop_size_px * 0.28, args.local_radius_px),
                )
                preferred = float(args.preferred_patch_span_px)
                median_rel = float(length_summary["median_relative_length_change"])
                max_rel = float(length_summary["max_relative_length_change"])
                score = (
                    92.0 * median_rel
                    + 38.0 * max_rel
                    + 0.16 * opposite_distance
                    + 0.32 * abs(contact_span - preferred)
                    + 0.18 * abs(rest_span - preferred)
                    - 2.7 * (len(patch_nodes) - 3)
                    - 0.75 * max(0, int(patch_edges.shape[0]) - (len(patch_nodes) - 1))
                    - 0.018 * min(gaussian_count, 220)
                    - 0.12 * x_margin
                )
                candidates.append(
                    {
                        "patch_nodes": tuple(int(item) for item in patch_nodes.tolist()),
                        "selected_rest_edges": patch_edges.astype(int).tolist(),
                        "score": float(score),
                        "search_top_quantile": float(quantile),
                        "search_y_cut_px": y_cut,
                        "torso_exclusion_y_cut_px": torso_y_cut,
                        "search_x_split_px": x_split,
                        "right_leg_margin_px": float(x_margin),
                        "contact_span_px": contact_span,
                        "rest_span_px": rest_span,
                        "opposite_leg_distance_px": opposite_distance,
                        "nearby_gaussian_count": gaussian_count,
                        "spring_length_summary": length_summary,
                        "spring_lengths": spring_lengths,
                    }
                )
            if candidates:
                chosen_quantile = quantile
                break
        if candidates:
            break

    if not candidates:
        raise RuntimeError(
            "Could not find a readable right-leg rest-neighbor spring patch. "
            "Try --selected_nodes or relax the search thresholds."
        )

    candidates.sort(key=lambda item: item["score"])
    selected = candidates[0]
    selected["mode"] = "automatic"
    selected["candidate_count"] = len(candidates)
    selected["chosen_search_top_quantile"] = float(chosen_quantile)
    selected["selected_nodes"] = list(selected["patch_nodes"])
    return (
        tuple(selected["patch_nodes"]),
        np.asarray(selected["selected_rest_edges"], dtype=np.int64),
        selected,
    )


def local_edges(local_nodes, adjacency):
    local_set = set(int(item) for item in local_nodes)
    edges = []
    for i in sorted(local_set):
        for j in sorted(adjacency[i]):
            if j in local_set and i < j:
                edges.append((i, j))
    return np.asarray(edges, dtype=np.int64)


def expand_connected_visible_patch(
    seed_nodes,
    adjacency,
    rest_xy,
    contact_xy,
    rest_visible_mask,
    contact_visible_mask,
    rest_center,
    contact_center,
    max_nodes,
):
    seed_nodes = [int(node) for node in seed_nodes]
    max_nodes = int(max_nodes)
    if max_nodes < len(seed_nodes):
        raise ValueError("--selected_patch_max_nodes must be at least the seed node count")

    common_visible = set(np.flatnonzero(rest_visible_mask & contact_visible_mask).tolist())
    missing = [node for node in seed_nodes if node not in common_visible]
    if missing:
        raise RuntimeError(
            "Seed selected nodes are not visible in both leg crops: "
            f"{missing}"
        )

    selected = set(seed_nodes)
    ordered = list(seed_nodes)

    def score(node):
        return (
            float(np.linalg.norm(rest_xy[int(node)] - rest_center))
            + float(np.linalg.norm(contact_xy[int(node)] - contact_center)),
            int(node),
        )

    while len(ordered) < max_nodes:
        candidates = sorted(
            {
                int(neighbour)
                for node in selected
                for neighbour in adjacency[int(node)]
                if int(neighbour) in common_visible and int(neighbour) not in selected
            },
            key=score,
        )
        if not candidates:
            break
        node = candidates[0]
        selected.add(node)
        ordered.append(node)

    return np.asarray(ordered, dtype=np.int64)


def incident_edges(anchor_node, adjacency):
    anchor_node = int(anchor_node)
    edges = [
        (min(anchor_node, int(neighbour)), max(anchor_node, int(neighbour)))
        for neighbour in sorted(adjacency[anchor_node])
    ]
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def rest_map_neighbors_for_anchor(rest_nodes, anchor_node, rest_map_radius):
    anchor_node = int(anchor_node)
    distances = np.linalg.norm(rest_nodes - rest_nodes[anchor_node], axis=1)
    mask = (distances <= float(rest_map_radius)) & (distances > 0.0)
    return np.flatnonzero(mask).astype(np.int64), distances


def projected_world_radius_px(point_world, radius_world, c2w, intrinsic):
    radius_world = float(radius_world)
    if radius_world <= 0.0:
        raise ValueError("Projected radius requires a positive world-space radius")

    point_world = np.asarray(point_world, dtype=np.float64).reshape(3)
    c2w = np.asarray(c2w, dtype=np.float64)
    camera_x = c2w[:3, 0]
    camera_y = c2w[:3, 1]
    camera_x = camera_x / np.linalg.norm(camera_x)
    camera_y = camera_y / np.linalg.norm(camera_y)
    samples = np.vstack(
        [
            point_world,
            point_world + camera_x * radius_world,
            point_world - camera_x * radius_world,
            point_world + camera_y * radius_world,
            point_world - camera_y * radius_world,
        ]
    )
    projected, valid = project_points(samples, c2w, intrinsic)
    if not valid[0] or not np.isfinite(projected[0]).all():
        return float("nan")
    distances = np.linalg.norm(projected[1:] - projected[0], axis=1)
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        return float("nan")
    return float(np.median(distances))


def rotation_only_alignment_2d(source_offsets, target_offsets):
    source_offsets = np.asarray(source_offsets, dtype=np.float64).reshape(-1, 2)
    target_offsets = np.asarray(target_offsets, dtype=np.float64).reshape(-1, 2)
    if source_offsets.shape != target_offsets.shape:
        raise ValueError("2D rotation alignment inputs must have matching shapes")
    if source_offsets.shape[0] == 0:
        rotation = np.eye(2, dtype=np.float64)
        return {
            "rotation": rotation,
            "rotation_deg": 0.0,
            "residual_rms_px": 0.0,
            "residual_max_px": 0.0,
            "residual_rms_normalized": 0.0,
        }

    u, _, vt = np.linalg.svd(source_offsets.T @ target_offsets)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    aligned = source_offsets @ rotation
    residuals = np.linalg.norm(aligned - target_offsets, axis=1)
    target_radius = float(np.max(np.linalg.norm(target_offsets, axis=1)))
    rms = float(np.sqrt(np.mean(residuals**2)))
    angle_deg = float(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])))
    return {
        "rotation": rotation,
        "rotation_deg": angle_deg,
        "residual_rms_px": rms,
        "residual_max_px": float(np.max(residuals)),
        "residual_rms_normalized": float(rms / target_radius)
        if target_radius > 1e-8
        else 0.0,
    }


def rotation_only_alignment_nd(source_offsets, target_offsets):
    source_offsets = np.asarray(source_offsets, dtype=np.float64)
    target_offsets = np.asarray(target_offsets, dtype=np.float64)
    if source_offsets.shape != target_offsets.shape:
        raise ValueError("Rotation alignment inputs must have matching shapes")
    if source_offsets.ndim != 2:
        raise ValueError("Rotation alignment inputs must be 2D arrays")
    dim = source_offsets.shape[1]
    if source_offsets.shape[0] == 0:
        rotation = np.eye(dim, dtype=np.float64)
        return {
            "rotation": rotation,
            "residual_rms": 0.0,
            "residual_max": 0.0,
            "residual_rms_normalized": 0.0,
        }

    u, _, vt = np.linalg.svd(source_offsets.T @ target_offsets)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    aligned = source_offsets @ rotation
    residuals = np.linalg.norm(aligned - target_offsets, axis=1)
    target_radius = float(np.max(np.linalg.norm(target_offsets, axis=1)))
    rms = float(np.sqrt(np.mean(residuals**2)))
    return {
        "rotation": rotation,
        "residual_rms": rms,
        "residual_max": float(np.max(residuals)),
        "residual_rms_normalized": float(rms / target_radius)
        if target_radius > 1e-8
        else 0.0,
    }


def anchor_local_project_world(
    points_world,
    anchor_world,
    x_axis,
    y_axis,
    scale_px_per_world,
    rotation=None,
):
    offsets = np.asarray(points_world, dtype=np.float64) - np.asarray(
        anchor_world,
        dtype=np.float64,
    )
    if rotation is not None:
        offsets = offsets @ np.asarray(rotation, dtype=np.float64)
    x = offsets @ np.asarray(x_axis, dtype=np.float64)
    y = offsets @ np.asarray(y_axis, dtype=np.float64)
    return np.column_stack([x, y]) * float(scale_px_per_world)


def anchor_local_xy(xy, anchor_xy, rotation=None):
    local_xy = np.asarray(xy, dtype=np.float64) - np.asarray(anchor_xy, dtype=np.float64)
    if rotation is not None:
        local_xy = local_xy @ np.asarray(rotation, dtype=np.float64)
    return local_xy


def nodes_visible_in_crop(nodes, node_mask):
    nodes = np.asarray(nodes, dtype=np.int64)
    if nodes.size == 0:
        return nodes
    return nodes[node_mask[nodes]]


def visual_circle(points_xy, pad_px):
    mins = points_xy.min(axis=0)
    maxs = points_xy.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(np.linalg.norm(points_xy - center, axis=1))) + float(pad_px)
    return center, radius


def crop_limits(center, crop_size_px):
    half = 0.5 * float(crop_size_px)
    return (center[0] - half, center[0] + half), (center[1] - half, center[1] + half)


def zoom_crop_size(circle, min_crop_size_px):
    _, radius = circle
    target = float(ZOOM_RED_RADIUS_FRACTION_TARGET)
    if target <= 0.0:
        return float(min_crop_size_px)
    return max(float(min_crop_size_px), 2.0 * float(radius) / target)


def rigid_alignment_2d(source_points, target_points):
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    aligned = source_centered @ rotation + target_center
    residual = float(np.sqrt(np.mean(np.sum((aligned - target_points) ** 2, axis=1))))
    angle_deg = float(math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])))
    return {
        "rotation": rotation,
        "source_center": source_center,
        "target_center": target_center,
        "rotation_deg": angle_deg,
        "residual_rms_px": residual,
    }


def apply_rigid_alignment_2d(points, alignment):
    points = np.asarray(points, dtype=np.float64)
    return (
        (points - alignment["source_center"]) @ alignment["rotation"]
        + alignment["target_center"]
    )


def rigid_alignment_3d(source_points, target_points):
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    aligned = source_centered @ rotation + target_center
    residual = float(np.sqrt(np.mean(np.sum((aligned - target_points) ** 2, axis=1))))
    trace_value = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle_deg = float(math.degrees(math.acos(trace_value)))
    return {
        "rotation": rotation,
        "source_center": source_center,
        "target_center": target_center,
        "rotation_angle_deg": angle_deg,
        "residual_rms": residual,
    }


def apply_rigid_alignment_3d(points, alignment):
    points = np.asarray(points, dtype=np.float64)
    return (
        (points - alignment["source_center"]) @ alignment["rotation"]
        + alignment["target_center"]
    )


def tangent_plane_basis(rest_nodes, selected_nodes, fit_nodes, camera_position, visual_span_px):
    selected_points = np.asarray(rest_nodes[np.asarray(selected_nodes, dtype=np.int64)], dtype=np.float64)
    fit_points = np.asarray(rest_nodes[np.asarray(fit_nodes, dtype=np.int64)], dtype=np.float64)
    origin = selected_points.mean(axis=0)
    centered = fit_points - fit_points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    to_camera = np.asarray(camera_position, dtype=np.float64) - origin
    if np.dot(normal, to_camera) < 0.0:
        normal = -normal

    pair_indices = list(combinations(range(fit_points.shape[0]), 2))
    i_long, j_long = max(
        pair_indices,
        key=lambda pair: float(np.linalg.norm(fit_points[pair[1]] - fit_points[pair[0]])),
    )
    x_axis = fit_points[j_long] - fit_points[i_long]
    x_axis = x_axis - np.dot(x_axis, normal) * normal
    if np.linalg.norm(x_axis) <= 1e-12:
        x_axis = vt[0] - np.dot(vt[0], normal) * normal
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    rest_local = project_to_tangent_plane(rest_nodes, origin, x_axis, y_axis, scale=1.0)
    span = float(np.max(np.ptp(rest_local[np.asarray(selected_nodes, dtype=np.int64)], axis=0)))
    scale = float(visual_span_px) / max(span, 1e-12)
    return {
        "origin": origin,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "normal": normal,
        "scale": scale,
        "fit_longest_pair_local_indices": [int(i_long), int(j_long)],
    }


def project_to_tangent_plane(points, origin, x_axis, y_axis, scale):
    points = np.asarray(points, dtype=np.float64)
    rel = points - np.asarray(origin, dtype=np.float64)
    xy = np.stack((rel @ x_axis, rel @ y_axis), axis=1)
    return xy * float(scale)


def quaternion_to_rotation_matrices(quaternions):
    quaternions = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions / np.maximum(norms, 1e-12)
    r, x, y, z = quaternions.T
    rotations = np.empty((quaternions.shape[0], 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - r * z)
    rotations[:, 0, 2] = 2.0 * (x * z + r * y)
    rotations[:, 1, 0] = 2.0 * (x * y + r * z)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - r * x)
    rotations[:, 2, 0] = 2.0 * (x * z - r * y)
    rotations[:, 2, 1] = 2.0 * (y * z + r * x)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotations


def gaussian_covariances_3d(scales, quaternions):
    scales = np.asarray(scales, dtype=np.float64).reshape(-1, 3)
    rotations = quaternion_to_rotation_matrices(quaternions)
    if scales.shape[0] != rotations.shape[0]:
        raise ValueError("Gaussian scale and quaternion counts do not match")
    scale_cov = scales * scales
    return np.einsum("nij,nj,nkj->nik", rotations, scale_cov, rotations)


def transform_covariances_by_alignment_3d(covariances, alignment):
    covariances = np.asarray(covariances, dtype=np.float64).reshape(-1, 3, 3)
    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    return np.einsum("ab,nbc,cd->nad", rotation.T, covariances, rotation)


def project_covariances_to_tangent_plane(covariances, zoom_basis):
    covariances = np.asarray(covariances, dtype=np.float64).reshape(-1, 3, 3)
    projection = float(zoom_basis["scale"]) * np.stack(
        (zoom_basis["x_axis"], zoom_basis["y_axis"]),
        axis=0,
    )
    return np.einsum("ia,nab,jb->nij", projection, covariances, projection)


def full_object_limits(xy, pad_fraction=0.055):
    valid = np.isfinite(xy).all(axis=1)
    if not np.any(valid):
        raise ValueError("Cannot compute full-object limits from empty projected points")
    pts = xy[valid]
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = np.maximum(maxs - mins, 1.0)
    pad = float(np.max(span)) * float(pad_fraction)
    return (mins[0] - pad, maxs[0] + pad), (mins[1] - pad, maxs[1] + pad)


def downsample_indices(count, max_count, seed):
    if max_count <= 0 or count <= max_count:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, size=max_count, replace=False))


def zoom_gaussian_kernel_indices(count):
    return downsample_indices(
        int(count), int(ZOOM_GAUSSIAN_KERNEL_MAX_ELLIPSES), seed=311
    )


def zoom_gaussian_kernel_ellipse_count(gaussian_xy):
    gaussian_xy = np.asarray(gaussian_xy)
    return int(min(gaussian_xy.shape[0], ZOOM_GAUSSIAN_KERNEL_MAX_ELLIPSES))


def draw_zoom_gaussian_kernel_ellipses(
    ax,
    gaussian_xy,
    gaussian_rgb,
    gaussian_cov2=None,
    gaussian_opacity=None,
    zorder=1,
    clip_circle=None,
):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle, Ellipse

    gaussian_xy = np.asarray(gaussian_xy, dtype=np.float64).reshape(-1, 2)
    gaussian_rgb = np.asarray(gaussian_rgb, dtype=np.float64).reshape(-1, 3)
    if gaussian_xy.size == 0 or gaussian_rgb.shape[0] != gaussian_xy.shape[0]:
        return 0

    selected = zoom_gaussian_kernel_indices(gaussian_xy.shape[0])
    centers = gaussian_xy[selected]
    colors = np.clip(gaussian_rgb[selected] * 0.98 + 0.02, 0.0, 1.0)
    alpha_scale = np.ones(colors.shape[0], dtype=np.float64)
    if gaussian_opacity is not None:
        opacity = np.asarray(gaussian_opacity, dtype=np.float64).reshape(-1)
        if opacity.shape[0] == gaussian_xy.shape[0]:
            alpha_scale = np.clip(opacity[selected], 0.25, 1.0)
    facecolors = np.column_stack(
        (
            colors,
            ZOOM_GAUSSIAN_KERNEL_FACE_ALPHA * alpha_scale,
        )
    )
    edgecolors = np.column_stack(
        (
            colors,
            ZOOM_GAUSSIAN_KERNEL_EDGE_ALPHA * alpha_scale,
        )
    )
    if gaussian_cov2 is not None:
        covariances = np.asarray(gaussian_cov2, dtype=np.float64).reshape(-1, 2, 2)
    else:
        covariances = np.empty((0, 2, 2), dtype=np.float64)
    if covariances.shape[0] == gaussian_xy.shape[0]:
        selected_covariances = covariances[selected]
        finite_covariances = np.isfinite(selected_covariances).all(axis=(1, 2))
        centers = centers[finite_covariances]
        facecolors = facecolors[finite_covariances]
        edgecolors = edgecolors[finite_covariances]
        selected_covariances = selected_covariances[finite_covariances]
        eigvals, eigvecs = np.linalg.eigh(selected_covariances)
        eigvals = np.clip(eigvals, 0.0, None)
        major = eigvals[:, 1]
        minor = eigvals[:, 0]
        major_vectors = eigvecs[:, :, 1]
        widths = 2.0 * ZOOM_GAUSSIAN_KERNEL_SIGMA_LEVEL * np.sqrt(major)
        heights = 2.0 * ZOOM_GAUSSIAN_KERNEL_SIGMA_LEVEL * np.sqrt(minor)
        angles = np.degrees(np.arctan2(major_vectors[:, 1], major_vectors[:, 0]))
    else:
        widths = np.full(centers.shape[0], ZOOM_GAUSSIAN_KERNEL_WIDTH_PX)
        heights = np.full(centers.shape[0], ZOOM_GAUSSIAN_KERNEL_HEIGHT_PX)
        angles = ((selected * 37) % 120) - 60

    patches = [
        Ellipse(
            (float(center[0]), float(center[1])),
            width=float(width),
            height=float(height),
            angle=float(angle),
        )
        for center, width, height, angle in zip(centers, widths, heights, angles)
    ]
    if not patches:
        return 0
    collection = PatchCollection(
        patches,
        facecolors=facecolors,
        edgecolors=edgecolors,
        linewidths=ZOOM_GAUSSIAN_KERNEL_EDGE_LINEWIDTH,
        zorder=zorder,
    )
    collection.set_rasterized(True)
    ax.add_collection(collection)
    if clip_circle is not None:
        clip_center, clip_radius = clip_circle
        collection.set_clip_path(
            Circle(clip_center, clip_radius, transform=ax.transData)
        )
    return int(len(patches))


def crop_gaussians(xy, rgb, xlim, ylim, args, frame_seed, return_indices=False):
    valid = (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= xlim[0])
        & (xy[:, 0] <= xlim[1])
        & (xy[:, 1] >= ylim[0])
        & (xy[:, 1] <= ylim[1])
    )
    selected = np.flatnonzero(valid)
    selected = selected[
        downsample_indices(
            selected.size,
            int(args.max_gaussian_points_per_panel),
            int(args.seed) + int(frame_seed),
        )
    ]
    colors = rgb[selected]
    strength = float(np.clip(args.gaussian_color_strength, 0.0, 1.0))
    colors = colors * strength + (1.0 - strength)
    colors = np.clip(colors, 0.0, 1.0)
    if return_indices:
        return xy[selected], colors, selected
    return xy[selected], colors


def projected_node_mask(mass_xy, xlim=None, ylim=None):
    mask = np.isfinite(mass_xy).all(axis=1)
    if xlim is not None:
        mask &= (mass_xy[:, 0] >= xlim[0]) & (mass_xy[:, 0] <= xlim[1])
    if ylim is not None:
        mask &= (mass_xy[:, 1] >= ylim[0]) & (mass_xy[:, 1] <= ylim[1])
    return mask


def projected_edges_in_mask(edges, mask):
    if edges.size == 0:
        return edges.reshape(0, 2)
    return edges[mask[edges[:, 0]] & mask[edges[:, 1]]]


def downsample_edges(edges, max_edges, seed):
    if edges.size == 0 or int(max_edges) <= 0 or edges.shape[0] <= int(max_edges):
        return edges.astype(np.int64).reshape(-1, 2)
    selected = downsample_indices(edges.shape[0], int(max_edges), int(seed))
    return edges[selected].astype(np.int64).reshape(-1, 2)


def remove_edges(edges, edges_to_remove):
    if edges.size == 0 or edges_to_remove.size == 0:
        return edges
    remove = {
        (min(int(i), int(j)), max(int(i), int(j)))
        for i, j in np.asarray(edges_to_remove, dtype=np.int64)
    }
    keep = [
        (int(i), int(j))
        for i, j in np.asarray(edges, dtype=np.int64)
        if (min(int(i), int(j)), max(int(i), int(j))) not in remove
    ]
    return np.asarray(keep, dtype=np.int64).reshape(-1, 2)


def zoom_graph_context(mass_xy, edges, xlim, ylim, selected_nodes=None, selected_edges=None):
    visible = projected_node_mask(mass_xy, xlim, ylim)
    context_nodes = np.flatnonzero(visible)
    context_edges = projected_edges_in_mask(edges, visible)
    return context_nodes.astype(np.int64), context_edges.astype(np.int64)


def segment_distance_to_point(points_a, points_b, point):
    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    segment = points_b - points_a
    denom = np.einsum("ij,ij->i", segment, segment)
    t = np.zeros(points_a.shape[0], dtype=np.float64)
    valid = denom > 0.0
    if np.any(valid):
        t[valid] = np.einsum(
            "ij,ij->i", point - points_a[valid], segment[valid]
        ) / denom[valid]
    t = np.clip(t, 0.0, 1.0)
    closest = points_a + t[:, None] * segment
    return np.linalg.norm(closest - point, axis=1)


def visible_zoom_context_graph(
    mass_xy, context_nodes, context_edges, circle, selected_nodes=None
):
    context_nodes = np.asarray(context_nodes, dtype=np.int64)
    context_edges = np.asarray(context_edges, dtype=np.int64).reshape(-1, 2)
    selected_nodes = (
        np.asarray(selected_nodes, dtype=np.int64)
        if selected_nodes is not None
        else np.empty(0, dtype=np.int64)
    )
    selected_node_set = {int(node) for node in selected_nodes.tolist()}
    center, radius = circle
    outside_node_mask = np.zeros(mass_xy.shape[0], dtype=bool)
    if context_nodes.size:
        distances = np.linalg.norm(mass_xy[context_nodes] - np.asarray(center), axis=1)
        outside_node_mask[context_nodes] = distances > float(radius)

    outside_candidate_nodes = context_nodes[outside_node_mask[context_nodes]]
    outside_candidate_edges = np.empty((0, 2), dtype=np.int64)
    boundary_context_edges = np.empty((0, 2), dtype=np.int64)
    selected_boundary_edges = np.empty((0, 2), dtype=np.int64)
    nonselected_boundary_edges = np.empty((0, 2), dtype=np.int64)
    visible_context_edges = np.empty((0, 2), dtype=np.int64)
    crossing_context_edges_dropped = 0

    if context_edges.size:
        outside_edge_endpoints = outside_node_mask[context_edges]
        outside_candidate_edges = context_edges[
            outside_edge_endpoints[:, 0] & outside_edge_endpoints[:, 1]
        ]
        boundary_context_edges = context_edges[
            outside_edge_endpoints[:, 0] ^ outside_edge_endpoints[:, 1]
        ]
        if boundary_context_edges.size:
            selected_boundary_mask = np.asarray(
                [
                    int(edge[0]) in selected_node_set or int(edge[1]) in selected_node_set
                    for edge in boundary_context_edges
                ],
                dtype=bool,
            )
            selected_boundary_edges = boundary_context_edges[selected_boundary_mask]
            nonselected_boundary_edges = boundary_context_edges[~selected_boundary_mask]
        if outside_candidate_edges.size:
            segment_distances = segment_distance_to_point(
                mass_xy[outside_candidate_edges[:, 0]],
                mass_xy[outside_candidate_edges[:, 1]],
                center,
            )
            outside_red_circle = segment_distances > float(radius)
            visible_context_edges = outside_candidate_edges[outside_red_circle]
            crossing_context_edges_dropped = int(
                outside_candidate_edges.shape[0] - visible_context_edges.shape[0]
            )

    rendered_context_edges = (
        np.vstack([visible_context_edges, selected_boundary_edges])
        if visible_context_edges.size and selected_boundary_edges.size
        else visible_context_edges
        if visible_context_edges.size
        else selected_boundary_edges
    )

    if rendered_context_edges.size:
        rendered_context_nodes = np.unique(rendered_context_edges.reshape(-1))
        visible_context_nodes = np.asarray(
            [node for node in rendered_context_nodes if outside_node_mask[int(node)]],
            dtype=np.int64,
        )
    else:
        visible_context_nodes = np.empty(0, dtype=np.int64)

    stats = {
        "context_nodes_available": int(context_nodes.shape[0]),
        "context_edges_available": int(context_edges.shape[0]),
        "outside_context_nodes_available": int(outside_candidate_nodes.shape[0]),
        "outside_context_edges_available": int(outside_candidate_edges.shape[0]),
        "outside_context_edges_rendered": int(visible_context_edges.shape[0]),
        "selected_boundary_edges_rendered": int(selected_boundary_edges.shape[0]),
        "nonselected_boundary_edges_dropped": int(nonselected_boundary_edges.shape[0]),
        "boundary_context_edges_dropped": int(nonselected_boundary_edges.shape[0]),
        "crossing_context_edges_dropped": int(crossing_context_edges_dropped),
        "isolated_context_nodes_dropped": int(
            outside_candidate_nodes.shape[0] - visible_context_nodes.shape[0]
        ),
    }
    return (
        visible_context_nodes.astype(np.int64),
        visible_context_edges.astype(np.int64),
        selected_boundary_edges.astype(np.int64),
        stats,
    )


def context_aware_zoom_crop_size(
    mass_xy,
    edges,
    circle,
    min_crop_size_px,
    selected_nodes=None,
    max_crop_size_px=16.5,
    min_context_nodes=3,
    min_context_edges=2,
):
    base_crop_size_px = zoom_crop_size(circle, min_crop_size_px)
    max_crop_size_px = max(float(max_crop_size_px), float(base_crop_size_px))
    candidates = np.unique(
        np.concatenate(
            [
                np.asarray([base_crop_size_px, max_crop_size_px], dtype=np.float64),
                np.linspace(base_crop_size_px, max_crop_size_px, 16),
            ]
        )
    )
    best = None
    best_stats = None
    for crop_size_px in candidates:
        xlim, ylim = crop_limits(circle[0], crop_size_px)
        context_nodes, context_edges = zoom_graph_context(mass_xy, edges, xlim, ylim)
        visible_nodes, visible_edges, selected_boundary_edges, stats = (
            visible_zoom_context_graph(
                mass_xy, context_nodes, context_edges, circle, selected_nodes
            )
        )
        stats = dict(stats)
        stats["visible_context_nodes_rendered"] = int(visible_nodes.shape[0])
        stats["visible_context_edges_rendered"] = int(
            visible_edges.shape[0] + selected_boundary_edges.shape[0]
        )
        stats["context_aware_crop_size_px"] = float(crop_size_px)
        best = float(crop_size_px)
        best_stats = stats
        if (
            visible_nodes.shape[0] >= int(min_context_nodes)
            and stats["visible_context_edges_rendered"] >= int(min_context_edges)
        ):
            break
    return best, best_stats


def set_clean_image_axis(ax, xlim, ylim):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(ylim[1], ylim[0])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_panel(
    ax,
    title,
    gaussian_xy,
    gaussian_rgb,
    mass_xy,
    selected_nodes,
    edges,
    circle,
    xlim,
    ylim,
    label=None,
):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    spring_color = GRAPH_SPRING_COLOR
    selected_color = GRAPH_NODE_COLOR
    red = "#D62728"

    if gaussian_xy.size:
        ax.scatter(
            gaussian_xy[:, 0],
            gaussian_xy[:, 1],
            s=0.72,
            c=gaussian_rgb,
            alpha=0.48,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    if edges.size:
        ax.add_collection(
            LineCollection(
                mass_xy[edges],
                colors=spring_color,
                linewidths=1.55,
                alpha=0.94,
                zorder=3,
            )
        )

    selected_nodes = [int(node) for node in selected_nodes]
    ax.scatter(
        mass_xy[selected_nodes, 0],
        mass_xy[selected_nodes, 1],
        s=34,
        c=selected_color,
        edgecolors="none",
        linewidths=0,
        zorder=6,
    )

    center, radius = circle
    ax.add_patch(
        Circle(
            center,
            radius,
            fill=False,
            edgecolor=red,
            linewidth=1.6,
            zorder=8,
        )
    )

    if label:
        text_x = min(center[0] + radius * 0.9, xlim[1] - 0.18 * (xlim[1] - xlim[0]))
        text_y = max(center[1] - radius * 0.9, ylim[0] + 0.12 * (ylim[1] - ylim[0]))
        ax.annotate(
            label,
            xy=(center[0] + radius * 0.45, center[1] - radius * 0.15),
            xytext=(text_x, text_y),
            fontsize=8.0,
            color=red,
            weight="bold",
            arrowprops={
                "arrowstyle": "->",
                "color": red,
                "lw": 1.0,
                "connectionstyle": "arc3,rad=0.18",
            },
            annotation_clip=False,
            zorder=10,
        )

    ax.set_title(title, fontsize=9.0, weight="bold", pad=2)
    set_clean_image_axis(ax, xlim, ylim)


def node_label_positions(points_xy, circle_center, circle_radius, crop_size_px):
    n_points = points_xy.shape[0]
    if n_points == 0:
        return []
    node_angles = np.arctan2(points_xy[:, 1] - circle_center[1], points_xy[:, 0] - circle_center[0])
    order = np.argsort(node_angles)
    ring_radius = min(float(circle_radius) + 5.0, 0.38 * float(crop_size_px))
    label_angles = np.linspace(-math.pi, math.pi, n_points, endpoint=False) - 0.18
    positions = [None] * n_points
    for rank, node_index in enumerate(order):
        angle = label_angles[rank]
        positions[int(node_index)] = np.asarray(
            [
                circle_center[0] + ring_radius * math.cos(angle),
                circle_center[1] + ring_radius * math.sin(angle),
            ],
            dtype=np.float64,
        )
    return positions


def render_zoom_patch(
    ax,
    gaussian_xy,
    gaussian_rgb,
    gaussian_cov2,
    gaussian_opacity,
    mass_xy,
    selected_nodes,
    selected_edges,
    circle,
    xlim,
    ylim,
    node_labels,
    context_nodes=None,
    context_edges=None,
    label=None,
    title=None,
    circular=False,
    border_color="#2CA9E1",
    node_label_offsets=None,
    render_node_labels=False,
):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    spring_color = GRAPH_SPRING_COLOR
    node_color = GRAPH_NODE_COLOR
    red = "#D62728"
    center, radius = circle

    if gaussian_xy.size and not circular:
        ax.scatter(
            gaussian_xy[:, 0],
            gaussian_xy[:, 1],
            s=0.62,
            c=gaussian_rgb,
            alpha=0.43,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    context_nodes = (
        np.asarray(context_nodes, dtype=np.int64)
        if context_nodes is not None
        else np.empty(0, dtype=np.int64)
    )
    context_edges = (
        np.asarray(context_edges, dtype=np.int64).reshape(-1, 2)
        if context_edges is not None
        else np.empty((0, 2), dtype=np.int64)
    )
    selected_nodes = np.asarray(selected_nodes, dtype=np.int64)
    selected_edges = np.asarray(selected_edges, dtype=np.int64).reshape(-1, 2)

    (
        visible_context_nodes,
        visible_context_edges,
        selected_boundary_edges,
        _,
    ) = visible_zoom_context_graph(
        mass_xy, context_nodes, context_edges, circle, selected_nodes
    )

    if visible_context_edges.size:
        context_collection = LineCollection(
            mass_xy[visible_context_edges],
            colors=spring_color,
            linewidths=0.36,
            alpha=0.30,
            zorder=2,
        )
        context_collection.set_rasterized(True)
        ax.add_collection(context_collection)

    if visible_context_nodes.size:
        ax.scatter(
            mass_xy[visible_context_nodes, 0],
            mass_xy[visible_context_nodes, 1],
            s=6.0,
            c=node_color,
            alpha=0.95,
            linewidths=0,
            rasterized=True,
            zorder=2.5,
        )

    if circular:
        ax.add_patch(
            Circle(center, radius, fill=True, facecolor="white", edgecolor="none", zorder=3)
        )

    if selected_edges.size:
        ax.add_collection(
            LineCollection(
                mass_xy[selected_edges],
                colors=spring_color,
                linewidths=0.36,
                alpha=0.82,
                zorder=4,
            )
        )

    if selected_boundary_edges.size:
        ax.add_collection(
            LineCollection(
                mass_xy[selected_boundary_edges],
                colors=spring_color,
                linewidths=0.36,
                alpha=0.82,
                zorder=4,
            )
        )

    if selected_nodes.size:
        ax.scatter(
            mass_xy[selected_nodes, 0],
            mass_xy[selected_nodes, 1],
            s=6.0,
            c=node_color,
            alpha=0.95,
            linewidths=0,
            rasterized=True,
            zorder=5,
        )

    ax.add_patch(
        Circle(center, radius, fill=False, edgecolor=red, linewidth=1.18, zorder=8)
    )

    if render_node_labels:
        selected_points = mass_xy[selected_nodes]
        if node_label_offsets is None:
            label_positions = node_label_positions(
                selected_points, center, radius, xlim[1] - xlim[0]
            )
        else:
            label_positions = [
                point + np.asarray(node_label_offsets[int(node)], dtype=np.float64)
                for node, point in zip(selected_nodes, selected_points)
            ]
        for node, point, label_pos in zip(selected_nodes, selected_points, label_positions):
            ax.plot(
                [point[0], label_pos[0]],
                [point[1], label_pos[1]],
                color="#12382F",
                linewidth=0.32,
                alpha=0.78,
                zorder=8,
            )
            ax.text(
                label_pos[0],
                label_pos[1],
                node_labels[int(node)],
                fontsize=4.8,
                color="#12382F",
                weight="bold",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.06",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.86,
                },
                zorder=9,
            )

    if title:
        ax.set_title(title, fontsize=6.4, weight="bold", pad=1.0)
    set_clean_image_axis(ax, xlim, ylim)
    if circular:
        clip_circle = Circle((0.5, 0.5), 0.5, transform=ax.transAxes)
        for artist in [*ax.collections, *ax.lines, *ax.patches, *ax.texts]:
            artist.set_clip_path(clip_circle)
            artist.set_clip_on(True)
        ax.patch.set_alpha(0.0)
        ax.add_patch(
            Circle(
                (0.5, 0.5),
                0.5,
                transform=ax.transAxes,
                fill=False,
                edgecolor=border_color,
                linewidth=1.35,
                clip_on=False,
                zorder=20,
            )
        )


def render_overview(
    ax,
    overview_xy,
    overview_rgb,
    overview_mass_xy,
    overview_edges,
    focus_center,
    focus_radius,
    overview_xlim,
    overview_ylim,
    callout_blue,
):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    if overview_xy.size:
        ax.scatter(
            overview_xy[:, 0],
            overview_xy[:, 1],
            s=0.34,
            c=overview_rgb,
            alpha=0.44,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
    if overview_edges is not None and np.asarray(overview_edges).size:
        edge_collection = LineCollection(
            overview_mass_xy[np.asarray(overview_edges, dtype=np.int64)],
            colors=GRAPH_SPRING_COLOR,
            linewidths=0.10,
            alpha=0.08,
            zorder=2,
        )
        edge_collection.set_rasterized(True)
        ax.add_collection(edge_collection)
    if overview_mass_xy is not None:
        visible_nodes = np.flatnonzero(
            projected_node_mask(overview_mass_xy, overview_xlim, overview_ylim)
        )
        if visible_nodes.size:
            ax.scatter(
                overview_mass_xy[visible_nodes, 0],
                overview_mass_xy[visible_nodes, 1],
                s=0.48,
                c=GRAPH_NODE_COLOR,
                alpha=0.16,
                linewidths=0,
                rasterized=True,
                zorder=2.5,
            )
    ax.add_patch(
        Circle(
            focus_center,
            focus_radius,
            fill=False,
            edgecolor=callout_blue,
            linewidth=0.8,
            zorder=5,
        )
    )
    ax.scatter(
        [focus_center[0]],
        [focus_center[1]],
        s=7,
        c=callout_blue,
        linewidths=0,
        zorder=6,
    )
    set_clean_image_axis(ax, overview_xlim, overview_ylim)


def render_full_gaussian_panel(
    ax,
    title,
    gaussian_xy,
    gaussian_rgb,
    mass_xy,
    context_edges,
    selected_nodes,
    selected_edges,
    xlim,
    ylim,
):
    from matplotlib.collections import LineCollection

    if gaussian_xy.size:
        ax.scatter(
            gaussian_xy[:, 0],
            gaussian_xy[:, 1],
            s=1.05,
            c=gaussian_rgb,
            alpha=0.52,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    if context_edges is not None and np.asarray(context_edges).size:
        context_collection = LineCollection(
            mass_xy[np.asarray(context_edges, dtype=np.int64)],
            colors=GRAPH_SPRING_COLOR,
            linewidths=0.22,
            alpha=0.10,
            zorder=2,
        )
        context_collection.set_rasterized(True)
        ax.add_collection(context_collection)

    visible_nodes = np.flatnonzero(projected_node_mask(mass_xy, xlim, ylim))
    if visible_nodes.size:
        ax.scatter(
            mass_xy[visible_nodes, 0],
            mass_xy[visible_nodes, 1],
            s=2.6,
            c=GRAPH_NODE_COLOR,
            alpha=0.18,
            linewidths=0,
            rasterized=True,
            zorder=2.5,
        )

    selected_edges = np.asarray(selected_edges, dtype=np.int64).reshape(-1, 2)
    if selected_edges.size:
        ax.add_collection(
            LineCollection(
                mass_xy[selected_edges],
                colors="white",
                linewidths=4.2,
                alpha=0.95,
                zorder=4,
            )
        )

    selected_nodes = np.asarray(selected_nodes, dtype=np.int64)
    if selected_nodes.size:
        ax.scatter(
            mass_xy[selected_nodes, 0],
            mass_xy[selected_nodes, 1],
            s=92,
            c=GRAPH_NODE_COLOR,
            edgecolors="white",
            linewidths=0.85,
            zorder=6,
        )

    if selected_edges.size:
        ax.add_collection(
            LineCollection(
                mass_xy[selected_edges],
                colors=GRAPH_SPRING_COLOR,
                linewidths=3.2,
                alpha=0.98,
                zorder=9,
            )
        )

    if selected_nodes.size:
        ax.scatter(
            mass_xy[selected_nodes, 0],
            mass_xy[selected_nodes, 1],
            s=28,
            c=GRAPH_NODE_COLOR,
            edgecolors="none",
            linewidths=0,
            zorder=8,
        )

    ax.set_title(title, fontsize=10.5, weight="bold", pad=4)
    set_clean_image_axis(ax, xlim, ylim)


def render_leg_focus_panel(
    ax,
    title,
    gaussian_xy,
    gaussian_rgb,
    mass_xy,
    context_edges,
    selected_nodes,
    selected_edges,
    xlim,
    ylim,
    show_labels=False,
):
    from matplotlib.collections import LineCollection

    context_alpha = 0.18
    selected_alpha = 0.95
    node_size = 14.0
    edge_width = 0.72
    context_edge_color = "#7E827B"
    context_node_color = GRAPH_NODE_COLOR
    selected_edge_color = GRAPH_SPRING_COLOR
    selected_node_color = "#C99700"

    if gaussian_xy.size:
        ax.scatter(
            gaussian_xy[:, 0],
            gaussian_xy[:, 1],
            s=3.4,
            c=gaussian_rgb,
            alpha=0.34,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    context_edges = np.asarray(context_edges, dtype=np.int64).reshape(-1, 2)
    selected_edges = np.asarray(selected_edges, dtype=np.int64).reshape(-1, 2)
    selected_nodes = np.asarray(selected_nodes, dtype=np.int64)
    selected_node_set = {int(node) for node in selected_nodes.tolist()}
    selected_edge_set = {
        (min(int(i), int(j)), max(int(i), int(j))) for i, j in selected_edges
    }

    if context_edges.size:
        nonselected_edges = np.asarray(
            [
                (int(i), int(j))
                for i, j in context_edges
                if (min(int(i), int(j)), max(int(i), int(j))) not in selected_edge_set
            ],
            dtype=np.int64,
        ).reshape(-1, 2)
        if nonselected_edges.size:
            collection = LineCollection(
                mass_xy[nonselected_edges],
                colors=context_edge_color,
                linewidths=edge_width,
                alpha=context_alpha,
                zorder=2,
            )
            collection.set_rasterized(True)
            ax.add_collection(collection)

    if selected_edges.size:
        ax.add_collection(
            LineCollection(
                mass_xy[selected_edges],
                colors=selected_edge_color,
                linewidths=edge_width,
                alpha=selected_alpha,
                zorder=3,
            )
        )

    visible_nodes = np.flatnonzero(projected_node_mask(mass_xy, xlim, ylim))
    if visible_nodes.size:
        nonselected_nodes = np.asarray(
            [node for node in visible_nodes if int(node) not in selected_node_set],
            dtype=np.int64,
        )
        if nonselected_nodes.size:
            ax.scatter(
                mass_xy[nonselected_nodes, 0],
                mass_xy[nonselected_nodes, 1],
                s=node_size,
                c=context_node_color,
                alpha=context_alpha,
                linewidths=0,
                rasterized=True,
                zorder=4,
            )

    if selected_nodes.size:
        ax.scatter(
            mass_xy[selected_nodes, 0],
            mass_xy[selected_nodes, 1],
            s=node_size,
            c=selected_node_color,
            alpha=selected_alpha,
            linewidths=0,
            zorder=5,
        )

    ax.set_title(title, fontsize=10.5, weight="bold", pad=4)
    set_clean_image_axis(ax, xlim, ylim)
    if show_labels:
        draw_leg_focus_labels(ax)


def draw_leg_focus_labels(ax):
    label_color = "#26302D"
    text_kwargs = {
        "transform": ax.transAxes,
        "fontsize": 6.1,
        "color": label_color,
        "ha": "left",
        "va": "center",
        "bbox": {
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        },
        "zorder": 20,
    }
    rows = [
        (0.925, "Real Gaussians"),
        (0.850, "Generated spring-mass graph"),
        (0.775, "Structurally adjacent patch"),
    ]
    ax.scatter(
        [0.037, 0.050, 0.063],
        [rows[0][0]] * 3,
        transform=ax.transAxes,
        s=10,
        c=["#B8B0A6", "#9C9E92", "#D4CEC4"],
        alpha=0.75,
        linewidths=0,
        clip_on=False,
        zorder=21,
    )
    ax.plot(
        [0.032, 0.068],
        [rows[1][0], rows[1][0]],
        transform=ax.transAxes,
        color="#7E827B",
        lw=0.72,
        alpha=0.36,
        clip_on=False,
        zorder=21,
    )
    ax.scatter(
        [0.050],
        [rows[1][0]],
        transform=ax.transAxes,
        s=14,
        c=GRAPH_NODE_COLOR,
        alpha=0.36,
        linewidths=0,
        clip_on=False,
        zorder=22,
    )
    ax.plot(
        [0.032, 0.068],
        [rows[2][0], rows[2][0]],
        transform=ax.transAxes,
        color=GRAPH_SPRING_COLOR,
        lw=0.72,
        alpha=0.95,
        clip_on=False,
        zorder=21,
    )
    ax.scatter(
        [0.050],
        [rows[2][0]],
        transform=ax.transAxes,
        s=14,
        c="#C99700",
        alpha=0.95,
        linewidths=0,
        clip_on=False,
        zorder=22,
    )
    for y, text in rows:
        ax.text(0.083, y, text, **text_kwargs)


def render_rest_map_anchor_panel(
    ax,
    title,
    gaussian_xy,
    gaussian_rgb,
    mass_xy,
    anchor_node,
    context_edges,
    selected_nodes,
    selected_edges,
    visual_radius_px,
    xlim,
    ylim,
    inset_image=None,
    inset_anchor_xy=None,
    show_labels=False,
):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    selected_alpha = 0.95
    node_size = 18.0
    edge_width = 0.90
    circle_color = "#C83B32"
    spring_edge_color = GRAPH_SPRING_COLOR
    spring_node_color = GRAPH_NODE_COLOR
    anchor_node_color = "#C83B32"
    context_edge_color = spring_edge_color
    context_node_color = spring_node_color
    context_edge_alpha = 0.14
    context_node_alpha = 0.32

    if gaussian_xy.size:
        ax.scatter(
            gaussian_xy[:, 0],
            gaussian_xy[:, 1],
            s=3.2,
            c=gaussian_rgb,
            alpha=0.30,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

    context_edges = np.asarray(context_edges, dtype=np.int64).reshape(-1, 2)
    selected_nodes = np.asarray(selected_nodes, dtype=np.int64)
    selected_edges = np.asarray(selected_edges, dtype=np.int64).reshape(-1, 2)
    anchor_node = int(anchor_node)

    if context_edges.size:
        ax.add_collection(
            LineCollection(
                mass_xy[context_edges],
                colors=context_edge_color,
                linewidths=edge_width,
                alpha=context_edge_alpha,
                zorder=2,
            )
        )
        context_nodes = np.unique(context_edges.reshape(-1))
        ax.scatter(
            mass_xy[context_nodes, 0],
            mass_xy[context_nodes, 1],
            s=node_size,
            c=context_node_color,
            alpha=context_node_alpha,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )

    if np.isfinite(float(visual_radius_px)) and float(visual_radius_px) > 0.0:
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                float(visual_radius_px),
                facecolor="none",
                edgecolor=circle_color,
                linewidth=1.25,
                alpha=0.95,
                zorder=4,
            )
        )

    if selected_edges.size:
        ax.add_collection(
            LineCollection(
                mass_xy[selected_edges],
                colors=spring_edge_color,
                linewidths=edge_width,
                alpha=selected_alpha,
                zorder=5,
            )
        )

    visible_mask = projected_node_mask(mass_xy, xlim, ylim)
    visible_selected_nodes = np.asarray(
        sorted(int(node) for node in selected_nodes if visible_mask[int(node)]),
        dtype=np.int64,
    )
    if visible_selected_nodes.size:
        ax.scatter(
            mass_xy[visible_selected_nodes, 0],
            mass_xy[visible_selected_nodes, 1],
            s=node_size,
            c=spring_node_color,
            alpha=selected_alpha,
            linewidths=0,
            zorder=6,
        )

    if visible_mask[anchor_node]:
        ax.scatter(
            [0.0],
            [0.0],
            s=node_size,
            c=anchor_node_color,
            alpha=selected_alpha,
            linewidths=0,
            zorder=7,
        )

    ax.set_title(title, fontsize=10.5, weight="bold", pad=4)
    set_clean_image_axis(ax, xlim, ylim)
    if inset_image is not None:
        draw_rest_map_full_object_inset(
            ax,
            inset_image,
            inset_anchor_xy,
        )
    if show_labels:
        draw_rest_map_anchor_labels(ax)


def draw_rest_map_full_object_inset(
    ax,
    image,
    anchor_xy,
):
    inset = ax.inset_axes([0.030, 0.060, 0.430, 0.255])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.0))
    inset.patch.set_alpha(0.0)

    image = normalize_render_image_alpha(image)
    image, crop_extent = crop_render_image_to_alpha_bbox(image)
    x0, x1, y0, y1 = crop_extent
    inset.imshow(
        image,
        origin="upper",
        extent=(float(x0), float(x1), float(y1), float(y0)),
        interpolation="nearest",
        zorder=1,
    )

    if anchor_xy is not None and np.isfinite(anchor_xy).all():
        inset.scatter(
            [float(anchor_xy[0])],
            [float(anchor_xy[1])],
            s=16,
            facecolors="none",
            edgecolors="#C83B32",
            linewidths=0.70,
            zorder=3,
        )
        inset.scatter(
            [float(anchor_xy[0])],
            [float(anchor_xy[1])],
            s=3.5,
            c="#C83B32",
            linewidths=0,
            zorder=4,
        )

    inset.set_xlim(float(x0), float(x1))
    inset.set_ylim(float(y1), float(y0))
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)


def normalize_render_image_alpha(image):
    image = np.asarray(image)
    if image.dtype.kind in {"u", "i"}:
        image = image.astype(np.float64) / float(np.iinfo(image.dtype).max)
    else:
        image = image.astype(np.float64)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 3:
        alpha = np.ones((*image.shape[:2], 1), dtype=image.dtype)
        image = np.concatenate([image, alpha], axis=2)
    elif image.shape[2] > 4:
        image = image[..., :4]
    return np.clip(image, 0.0, 1.0)


def render_image_alpha_bbox(image, pad_px=10, alpha_threshold=1.0 / 255.0):
    image = normalize_render_image_alpha(image)
    alpha = image[..., 3]
    mask = alpha > float(alpha_threshold)
    height, width = image.shape[:2]
    if not np.any(mask):
        return (0, int(width), 0, int(height))

    ys, xs = np.nonzero(mask)
    pad_px = int(max(0, pad_px))
    x0 = max(0, int(xs.min()) - pad_px)
    x1 = min(int(width), int(xs.max()) + pad_px + 1)
    y0 = max(0, int(ys.min()) - pad_px)
    y1 = min(int(height), int(ys.max()) + pad_px + 1)
    return (x0, x1, y0, y1)


def crop_render_image_to_alpha_bbox(image, pad_px=10):
    image = normalize_render_image_alpha(image)
    x0, x1, y0, y1 = render_image_alpha_bbox(image, pad_px=pad_px)
    image = image[y0:y1, x0:x1]
    return image, (x0, x1, y0, y1)


def alpha_bbox_metadata(image, pad_px=10):
    x0, x1, y0, y1 = render_image_alpha_bbox(image, pad_px=pad_px)
    return {
        "x0": int(x0),
        "x1": int(x1),
        "y0": int(y0),
        "y1": int(y1),
        "width": int(x1 - x0),
        "height": int(y1 - y0),
        "pad_px": int(max(0, pad_px)),
    }


def load_render_image(path):
    import matplotlib.image as mpimg

    return mpimg.imread(str(path))


def clipped_crop_limits(center, crop_size_px, width, height):
    xlim, ylim = crop_limits(np.asarray(center, dtype=np.float64), crop_size_px)
    x0 = max(0.0, float(xlim[0]))
    x1 = min(float(width), float(xlim[1]))
    y0 = max(0.0, float(ylim[0]))
    y1 = min(float(height), float(ylim[1]))
    return (x0, x1), (y0, y1)


def raw_anchor_pruning_circle(mass_xy, selected_nodes, anchor_node, pad_px=2.0):
    mass_xy = np.asarray(mass_xy, dtype=np.float64)
    selected_nodes = np.asarray(selected_nodes, dtype=np.int64)
    center = mass_xy[int(anchor_node)].astype(np.float64)
    if not selected_nodes.size:
        return center, float(pad_px)
    points = mass_xy[selected_nodes]
    valid = np.isfinite(points).all(axis=1)
    if not np.any(valid) or not np.isfinite(center).all():
        return center, float(pad_px)
    radius = float(np.max(np.linalg.norm(points[valid] - center, axis=1)))
    return center, radius + float(pad_px)


def circle_metadata(circle):
    center, radius = circle
    return {
        "center": [float(value) for value in np.asarray(center).tolist()],
        "radius": float(radius),
    }


def limits_metadata(xlim, ylim):
    return {
        "xlim": [float(value) for value in xlim],
        "ylim": [float(value) for value in ylim],
    }


def crop_edges_for_raw_view(edges, mass_xy, xlim, ylim):
    return projected_edges_in_mask(
        np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        projected_node_mask(mass_xy, xlim, ylim),
    )


def render_boba_image_background(ax, image, zorder=0):
    image = normalize_render_image_alpha(image)
    height, width = image.shape[:2]
    ax.imshow(
        image,
        origin="upper",
        extent=(0.0, float(width), float(height), 0.0),
        interpolation="bilinear",
        zorder=zorder,
    )
    return int(width), int(height)


def render_raw_spring_mass_overlay(
    ax,
    mass_xy,
    context_edges,
    selected_nodes,
    selected_edges,
    anchor_node,
    pruning_circle,
    xlim,
    ylim,
    node_size=9.0,
    edge_width=0.65,
    context_edge_alpha=0.12,
    context_node_alpha=0.24,
    selected_alpha=0.96,
):
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    mass_xy = np.asarray(mass_xy, dtype=np.float64)
    context_edges = np.asarray(context_edges, dtype=np.int64).reshape(-1, 2)
    selected_nodes = np.asarray(selected_nodes, dtype=np.int64)
    selected_edges = np.asarray(selected_edges, dtype=np.int64).reshape(-1, 2)
    anchor_node = int(anchor_node)

    if context_edges.size:
        ax.add_collection(
            LineCollection(
                mass_xy[context_edges],
                colors=GRAPH_SPRING_COLOR,
                linewidths=edge_width,
                alpha=context_edge_alpha,
                zorder=2,
            )
        )
        context_nodes = np.unique(context_edges.reshape(-1))
        visible_context = projected_node_mask(mass_xy, xlim, ylim)[context_nodes]
        context_nodes = context_nodes[visible_context]
        if context_nodes.size:
            ax.scatter(
                mass_xy[context_nodes, 0],
                mass_xy[context_nodes, 1],
                s=node_size,
                c=GRAPH_NODE_COLOR,
                alpha=context_node_alpha,
                linewidths=0,
                rasterized=True,
                zorder=3,
            )

    center, radius = pruning_circle
    if np.isfinite(center).all() and np.isfinite(float(radius)) and float(radius) > 0:
        ax.add_patch(
            Circle(
                (float(center[0]), float(center[1])),
                float(radius),
                facecolor="none",
                edgecolor="#C83B32",
                linewidth=1.25,
                alpha=0.96,
                zorder=4,
            )
        )

    if selected_edges.size:
        ax.add_collection(
            LineCollection(
                mass_xy[selected_edges],
                colors=GRAPH_SPRING_COLOR,
                linewidths=edge_width,
                alpha=selected_alpha,
                zorder=5,
            )
        )

    visible_mask = projected_node_mask(mass_xy, xlim, ylim)
    visible_selected_nodes = np.asarray(
        sorted(int(node) for node in selected_nodes if visible_mask[int(node)]),
        dtype=np.int64,
    )
    if visible_selected_nodes.size:
        ax.scatter(
            mass_xy[visible_selected_nodes, 0],
            mass_xy[visible_selected_nodes, 1],
            s=node_size,
            c=GRAPH_NODE_COLOR,
            alpha=selected_alpha,
            linewidths=0,
            zorder=6,
        )

    if visible_mask[anchor_node]:
        ax.scatter(
            [mass_xy[anchor_node, 0]],
            [mass_xy[anchor_node, 1]],
            s=node_size,
            c="#C83B32",
            alpha=selected_alpha,
            linewidths=0,
            zorder=7,
        )


def draw_rest_map_full_context_inset(
    ax,
    image,
    anchor_xy,
):
    inset = ax.inset_axes([0.045, 0.055, 0.270, 0.270])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.0))
    inset.patch.set_alpha(0.0)
    width, height = render_boba_image_background(inset, image, zorder=1)
    if anchor_xy is not None and np.isfinite(anchor_xy).all():
        inset.scatter(
            [float(anchor_xy[0])],
            [float(anchor_xy[1])],
            s=18,
            facecolors="none",
            edgecolors="#C83B32",
            linewidths=0.85,
            zorder=3,
        )
        inset.scatter(
            [float(anchor_xy[0])],
            [float(anchor_xy[1])],
            s=4.5,
            c="#C83B32",
            linewidths=0,
            zorder=4,
        )
    set_clean_image_axis(inset, (0.0, float(width)), (0.0, float(height)))


def render_rest_map_anchor_raw_panel(
    ax,
    title,
    image,
    mass_xy,
    context_edges,
    selected_nodes,
    selected_edges,
    anchor_node,
    pruning_circle,
    zoom_crop_xlim,
    zoom_crop_ylim,
    show_labels=False,
):
    render_boba_image_background(ax, image, zorder=1)
    render_raw_spring_mass_overlay(
        ax,
        mass_xy,
        crop_edges_for_raw_view(context_edges, mass_xy, zoom_crop_xlim, zoom_crop_ylim),
        selected_nodes,
        crop_edges_for_raw_view(selected_edges, mass_xy, zoom_crop_xlim, zoom_crop_ylim),
        anchor_node,
        pruning_circle,
        zoom_crop_xlim,
        zoom_crop_ylim,
        node_size=18.0,
        edge_width=0.90,
        context_edge_alpha=0.18,
        context_node_alpha=0.34,
    )
    draw_rest_map_full_context_inset(
        ax,
        image,
        mass_xy[int(anchor_node)],
    )
    ax.set_title(title, fontsize=10.5, weight="bold", pad=4)
    set_clean_image_axis(ax, zoom_crop_xlim, zoom_crop_ylim)
    if show_labels:
        draw_rest_map_anchor_labels(ax)


def draw_rest_map_anchor_labels(ax):
    label_color = "#26302D"
    text_kwargs = {
        "transform": ax.transAxes,
        "fontsize": 6.1,
        "color": label_color,
        "ha": "left",
        "va": "center",
        "bbox": {
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        },
        "zorder": 20,
    }
    rows = [
        (0.925, "Real Gaussians"),
        (0.850, "Mass nodes"),
        (0.775, "Springs"),
        (0.700, "Pruning region"),
    ]
    ax.scatter(
        [0.037, 0.050, 0.063],
        [rows[0][0]] * 3,
        transform=ax.transAxes,
        s=10,
        c=["#B8B0A6", "#9C9E92", "#D4CEC4"],
        alpha=0.75,
        linewidths=0,
        clip_on=False,
        zorder=21,
    )
    ax.scatter(
        [0.050],
        [rows[1][0]],
        transform=ax.transAxes,
        s=14,
        c=GRAPH_NODE_COLOR,
        alpha=0.95,
        linewidths=0,
        clip_on=False,
        zorder=22,
    )
    ax.plot(
        [0.034, 0.066],
        [rows[2][0], rows[2][0]],
        transform=ax.transAxes,
        color=GRAPH_SPRING_COLOR,
        lw=0.90,
        alpha=0.95,
        clip_on=False,
        zorder=21,
    )
    ax.scatter(
        [0.050],
        [rows[3][0]],
        transform=ax.transAxes,
        s=36,
        facecolors="none",
        edgecolors="#C83B32",
        linewidths=1.25,
        alpha=0.95,
        clip_on=False,
        zorder=21,
    )
    for y, text in rows:
        ax.text(0.083, y, text, **text_kwargs)


def draw_rest_map_anchor_center_legend(fig):
    ax = fig.add_axes([0.415, 0.345, 0.170, 0.340])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_facecolor("white")
    ax.patch.set_alpha(0.84)

    rows = [
        (0.82, "Real Gaussians"),
        (0.58, "Mass nodes"),
        (0.34, "Springs"),
        (0.10, "Pruning region"),
    ]
    ax.scatter(
        [0.10, 0.16, 0.22],
        [rows[0][0]] * 3,
        s=18,
        c=["#B8B0A6", "#9C9E92", "#D4CEC4"],
        alpha=0.75,
        linewidths=0,
        clip_on=False,
    )
    ax.scatter(
        [0.16],
        [rows[1][0]],
        s=26,
        c=GRAPH_NODE_COLOR,
        alpha=0.95,
        linewidths=0,
        clip_on=False,
    )
    ax.plot(
        [0.08, 0.24],
        [rows[2][0], rows[2][0]],
        color=GRAPH_SPRING_COLOR,
        lw=1.35,
        alpha=0.95,
        clip_on=False,
    )
    ax.scatter(
        [0.16],
        [rows[3][0]],
        s=58,
        facecolors="none",
        edgecolors="#C83B32",
        linewidths=1.55,
        alpha=0.95,
        clip_on=False,
    )
    for y, text in rows:
        ax.text(0.34, y, text, fontsize=8.0, color="#26302D", va="center")


def outside_anchor_context_graph(mass_xy, context_edges, selected_nodes, visual_radius_px):
    context_edges = np.asarray(context_edges, dtype=np.int64).reshape(-1, 2)
    selected_nodes = np.asarray(selected_nodes, dtype=np.int64)
    selected_mask = np.zeros(mass_xy.shape[0], dtype=bool)
    if selected_nodes.size:
        selected_mask[selected_nodes] = True

    if not context_edges.size:
        return context_edges, np.empty(0, dtype=np.int64), {
            "context_edges_available": 0,
            "context_edges_rendered": 0,
            "context_nodes_rendered": 0,
            "inside_context_edges_dropped": 0,
            "boundary_context_edges_dropped": 0,
            "selected_touch_context_edges_dropped": 0,
            "inside_context_nodes_dropped": 0,
        }

    radius = float(visual_radius_px)
    distances = np.linalg.norm(mass_xy, axis=1)
    inside_mask = distances <= radius
    edge_inside = inside_mask[context_edges]
    edge_touches_selected = selected_mask[context_edges].any(axis=1)
    both_outside = ~edge_inside[:, 0] & ~edge_inside[:, 1]
    both_inside = edge_inside[:, 0] & edge_inside[:, 1]
    boundary = edge_inside[:, 0] ^ edge_inside[:, 1]
    render_mask = both_outside & ~edge_touches_selected
    rendered_edges = context_edges[render_mask]
    rendered_nodes = (
        np.unique(rendered_edges.reshape(-1))
        if rendered_edges.size
        else np.empty(0, dtype=np.int64)
    )
    nonselected_context_nodes = np.unique(context_edges[~edge_touches_selected].reshape(-1))
    inside_nonselected_nodes = nonselected_context_nodes[
        inside_mask[nonselected_context_nodes] & ~selected_mask[nonselected_context_nodes]
    ]
    stats = {
        "context_edges_available": int(context_edges.shape[0]),
        "context_edges_rendered": int(rendered_edges.shape[0]),
        "context_nodes_rendered": int(rendered_nodes.shape[0]),
        "inside_context_edges_dropped": int(
            np.count_nonzero(both_inside & ~edge_touches_selected)
        ),
        "boundary_context_edges_dropped": int(
            np.count_nonzero(boundary & ~edge_touches_selected)
        ),
        "selected_touch_context_edges_dropped": int(
            np.count_nonzero(edge_touches_selected)
        ),
        "inside_context_nodes_dropped": int(inside_nonselected_nodes.shape[0]),
    }
    return rendered_edges, rendered_nodes, stats


def draw_shared_encoding_legend(
    fig, gaussian_xy=None, gaussian_rgb=None, gaussian_xlim=None, gaussian_ylim=None
):
    ax = fig.add_axes([0.368, 0.405, 0.185, 0.32])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_alpha(0.0)

    gaussian_xy = (
        np.asarray(gaussian_xy, dtype=np.float64).reshape(-1, 2)
        if gaussian_xy is not None
        else np.empty((0, 2), dtype=np.float64)
    )
    gaussian_rgb = (
        np.asarray(gaussian_rgb, dtype=np.float64).reshape(-1, 3)
        if gaussian_rgb is not None
        else np.empty((0, 3), dtype=np.float64)
    )
    if gaussian_xy.size and gaussian_rgb.shape[0] == gaussian_xy.shape[0]:
        legend_xy = gaussian_xy
        legend_rgb = gaussian_rgb
        max_legend_points = 12000
        if legend_xy.shape[0] > max_legend_points:
            selected = downsample_indices(legend_xy.shape[0], max_legend_points, 7013)
            legend_xy = legend_xy[selected]
            legend_rgb = legend_rgb[selected]
        glyph_ax = ax.inset_axes([0.02, 0.64, 0.25, 0.34])
        glyph_ax.scatter(
            legend_xy[:, 0],
            legend_xy[:, 1],
            s=0.34,
            c=np.clip(legend_rgb, 0.0, 1.0),
            alpha=0.44,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        if gaussian_xlim is None or gaussian_ylim is None:
            gaussian_xlim, gaussian_ylim = full_object_limits(legend_xy)
        set_clean_image_axis(glyph_ax, gaussian_xlim, gaussian_ylim)
    else:
        ax.scatter(
            [0.12, 0.19, 0.25],
            [0.84, 0.91, 0.86],
            s=[8, 13, 7],
            c=[(0.50, 0.58, 0.50), (0.73, 0.64, 0.52), (0.63, 0.69, 0.70)],
            alpha=0.65,
            linewidths=0,
            zorder=2,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.255,
        0.86,
        "Gaussians",
        fontsize=5.2,
        color="#2E3633",
        va="center",
        zorder=3,
    )

    mass_y = 0.46
    ax.scatter(
        [0.145],
        [mass_y],
        s=17,
        c=GRAPH_NODE_COLOR,
        alpha=0.85,
        linewidths=0,
        zorder=3,
    )
    ax.text(
        0.255,
        mass_y,
        "Mass nodes",
        fontsize=5.1,
        color="#2E3633",
        va="center",
        zorder=3,
    )

    spring_y = 0.17
    ax.plot(
        [0.075, 0.215],
        [spring_y, spring_y],
        color=GRAPH_SPRING_COLOR,
        lw=0.78,
        alpha=0.88,
        zorder=2,
    )
    ax.text(
        0.255,
        spring_y,
        "Springs",
        fontsize=5.1,
        color="#2E3633",
        va="center",
        zorder=3,
    )


def left_circle_border_point(y_frac):
    dy = float(y_frac) - 0.5
    radius = 0.5
    x_frac = 0.5 - math.sqrt(max(0.0, radius * radius - dy * dy))
    return x_frac, float(y_frac)


def data_to_inset_axes(point, xlim, ylim):
    x_frac = (float(point[0]) - float(xlim[0])) / (float(xlim[1]) - float(xlim[0]))
    y_frac = (float(point[1]) - float(ylim[1])) / (float(ylim[0]) - float(ylim[1]))
    return np.asarray([x_frac, y_frac], dtype=np.float64)


def inset_axes_to_parent(point, inset_bounds):
    return (
        float(inset_bounds[0]) + float(inset_bounds[2]) * float(point[0]),
        float(inset_bounds[1]) + float(inset_bounds[3]) * float(point[1]),
    )


def annotate_red_circle_callout(
    ax,
    inset,
    zoom_circle,
    text,
    text_parent,
    target_angle_deg,
    arrow_line_index=-1,
    arrow_line_side="right",
    fontsize=5.0,
    linespacing=0.88,
):
    from matplotlib.patches import ConnectionPatch
    from matplotlib.transforms import ScaledTranslation

    red = "#D62728"
    center, radius = zoom_circle
    target_angle = math.radians(float(target_angle_deg))
    target_data = np.asarray(
        [
            center[0] + radius * 0.97 * math.cos(target_angle),
            center[1] + radius * 0.97 * math.sin(target_angle),
        ],
        dtype=np.float64,
    )
    lines = str(text).splitlines() or [str(text)]
    line_gap_pt = float(fontsize) * float(linespacing)
    center_index = 0.5 * (len(lines) - 1)
    line_artists = []
    for index, line in enumerate(lines):
        offset_pt = (center_index - index) * line_gap_pt
        transform = ax.transAxes + ScaledTranslation(
            0.0, offset_pt / 72.0, ax.figure.dpi_scale_trans
        )
        line_artists.append(
            ax.text(
                text_parent[0],
                text_parent[1],
                line,
                transform=transform,
                fontsize=fontsize,
                color=red,
                weight="bold",
                ha="left",
                va="center",
                zorder=30,
                clip_on=False,
            )
        )

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    line_index = int(np.clip(int(arrow_line_index), -len(lines), len(lines) - 1))
    line_bbox = line_artists[line_index].get_window_extent(renderer)
    arrow_y_px = 0.5 * (line_bbox.y0 + line_bbox.y1)
    if arrow_line_side == "left":
        arrow_x_px = line_bbox.x0 - 1.0
    else:
        arrow_x_px = line_bbox.x1 + 1.0
    arrow_start = ax.transAxes.inverted().transform((arrow_x_px, arrow_y_px))
    arrow = ConnectionPatch(
        xyA=arrow_start,
        coordsA=ax.transAxes,
        axesA=ax,
        xyB=target_data,
        coordsB=inset.transData,
        axesB=inset,
        arrowstyle="->",
        color=red,
        linewidth=0.82,
        mutation_scale=6.2,
        connectionstyle="arc3,rad=0.0",
        clip_on=False,
        zorder=30,
    )
    ax.add_artist(arrow)


def render_overview_with_zoom(
    ax,
    title,
    overview_xy,
    overview_rgb,
    overview_mass_xy,
    overview_edges,
    mass_xy,
    selected_nodes,
    selected_edges,
    zoom_context_nodes,
    zoom_context_edges,
    zoom_gaussian_xy,
    zoom_gaussian_rgb,
    zoom_gaussian_cov2,
    zoom_gaussian_opacity,
    zoom_circle,
    zoom_xlim,
    zoom_ylim,
    overview_xlim,
    overview_ylim,
    node_labels,
    label=None,
    overview_circle=None,
    node_label_offsets=None,
    inset_bounds=None,
    title_x=0.5,
    callout_text=None,
    callout_position=None,
    callout_target_angle_deg=30.0,
    callout_arrow_line_index=-1,
    callout_arrow_line_side="right",
    callout_fontsize=5.0,
):
    from matplotlib.patches import Circle, ConnectionPatch

    callout_blue = "#2CA9E1"
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    zoom_center, zoom_radius = zoom_circle
    overview_circle = zoom_circle if overview_circle is None else overview_circle
    overview_center, overview_zoom_radius = overview_circle
    full_span = max(overview_xlim[1] - overview_xlim[0], overview_ylim[1] - overview_ylim[0])
    overview_radius = max(float(overview_zoom_radius) * 1.18, 0.022 * full_span)

    overview_ax = ax.inset_axes([-0.018, 0.30, 0.35, 0.35])
    render_overview(
        overview_ax,
        overview_xy,
        overview_rgb,
        overview_mass_xy,
        overview_edges,
        overview_center,
        overview_radius,
        overview_xlim,
        overview_ylim,
        callout_blue,
    )

    if inset_bounds is None:
        inset_bounds = [0.185, 0.315, 0.43, 0.675]
    inset = ax.inset_axes(inset_bounds)
    render_zoom_patch(
        inset,
        zoom_gaussian_xy,
        zoom_gaussian_rgb,
        zoom_gaussian_cov2,
        zoom_gaussian_opacity,
        mass_xy,
        selected_nodes,
        selected_edges,
        zoom_circle,
        zoom_xlim,
        zoom_ylim,
        node_labels,
        context_nodes=zoom_context_nodes,
        context_edges=zoom_context_edges,
        circular=True,
        border_color=callout_blue,
        node_label_offsets=node_label_offsets,
    )
    for spine in inset.spines.values():
        spine.set_visible(False)

    for angle, border_point in (
        (-0.58, left_circle_border_point(0.34)),
        (0.58, left_circle_border_point(0.66)),
    ):
        overview_anchor = (
            overview_center[0] + overview_radius * math.cos(angle),
            overview_center[1] + overview_radius * math.sin(angle),
        )
        connection = ConnectionPatch(
            xyA=overview_anchor,
            coordsA=overview_ax.transData,
            xyB=border_point,
            coordsB=inset.transAxes,
            axesA=overview_ax,
            axesB=inset,
            color=callout_blue,
            linewidth=0.86,
            linestyle=(0, (3.0, 2.2)),
            zorder=6,
        )
        ax.add_artist(connection)

    if callout_text is None:
        callout_text = label
    if callout_text:
        if callout_position is None:
            callout_position = (
                min(0.88, float(inset_bounds[0]) + float(inset_bounds[2]) - 0.13),
                max(0.07, float(inset_bounds[1]) - 0.035),
            )
        annotate_red_circle_callout(
            ax,
            inset,
            zoom_circle,
            callout_text,
            callout_position,
            callout_target_angle_deg,
            arrow_line_index=callout_arrow_line_index,
            arrow_line_side=callout_arrow_line_side,
            fontsize=callout_fontsize,
        )

    ax.text(
        title_x,
        0.15,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.4,
        weight="bold",
        color="black",
        zorder=35,
    )


def render_figures(data, output_dir, output_stem, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox

    def save_combined_with_custom_crop(fig, path, dpi):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight_bbox = fig.get_tightbbox(renderer)
        protected_text = {
            "Structurally",
            "adjacent pair",
            "Collision",
            "response",
            "pruned for",
            "structural",
            "neighbors",
            "Resting State",
            "Simulation Stage",
        }
        bottom_label_text = {"Resting State", "Simulation Stage"}
        text_artists = [
            text
            for text in fig.findobj(match=lambda artist: hasattr(artist, "get_text"))
        ]
        protected_right_px = [
            text.get_window_extent(renderer).x1
            for text in text_artists
            if text.get_text() in protected_text
        ]
        bottom_label_bottom_px = [
            text.get_window_extent(renderer).y0
            for text in text_artists
            if text.get_text() in bottom_label_text
        ]
        if protected_right_px:
            cropped_right = max(protected_right_px) / float(fig.dpi) + 0.006
        else:
            cropped_right = tight_bbox.x1
        cropped_right = min(max(cropped_right, tight_bbox.x0), tight_bbox.x1)
        cropped_left = min(tight_bbox.x0 + 0.08, cropped_right - 0.1)
        if bottom_label_bottom_px:
            cropped_bottom = min(bottom_label_bottom_px) / float(fig.dpi) - 0.006
        else:
            cropped_bottom = tight_bbox.y0
        cropped_bottom = min(max(cropped_bottom, tight_bbox.y0), tight_bbox.y1 - 0.1)
        crop_bbox = Bbox.from_extents(
            cropped_left,
            cropped_bottom,
            cropped_right,
            tight_bbox.y1,
        )
        fig.savefig(path, dpi=dpi, bbox_inches=crop_bbox, pad_inches=0.001)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_png = output_dir / f"{output_stem}.png"
    combined_pdf = output_dir / f"{output_stem}.pdf"
    contact_png = output_dir / f"{output_stem}_frame_{args.contact_frame:03d}_contact.png"
    rest_png = output_dir / f"{output_stem}_frame_{args.rest_frame:03d}_rest.png"

    if args.layout == "full_gaussian_highlight":
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8.8, 3.55),
            gridspec_kw={"wspace": 0.04},
            constrained_layout=True,
        )
        render_full_gaussian_panel(
            axes[0],
            "Resting State",
            data["rest_overview_xy"],
            data["rest_overview_rgb"],
            data["rest_overview_mass_xy"],
            data["rest_overview_edges"],
            data["selected_nodes"],
            data["selected_edges"],
            data["rest_overview_xlim"],
            data["rest_overview_ylim"],
        )
        render_full_gaussian_panel(
            axes[1],
            "Simulation Stage",
            data["contact_overview_xy"],
            data["contact_overview_rgb"],
            data["contact_overview_mass_xy"],
            data["contact_overview_edges"],
            data["selected_nodes"],
            data["selected_edges"],
            data["contact_overview_xlim"],
            data["contact_overview_ylim"],
        )
        fig.savefig(combined_png, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.035)
        fig.savefig(combined_pdf, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.035)
        plt.close(fig)

        for path, key, title in (
            (contact_png, "contact", "Simulation Stage"),
            (rest_png, "rest", "Resting State"),
        ):
            fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.7), constrained_layout=True)
            render_full_gaussian_panel(
                ax,
                title,
                data[f"{key}_overview_xy"],
                data[f"{key}_overview_rgb"],
                data[f"{key}_overview_mass_xy"],
                data[f"{key}_overview_edges"],
                data["selected_nodes"],
                data["selected_edges"],
                data[f"{key}_overview_xlim"],
                data[f"{key}_overview_ylim"],
            )
            fig.savefig(path, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.035)
            plt.close(fig)

        return {
            "combined_png": combined_png,
            "combined_pdf": combined_pdf,
            "contact_png": contact_png,
            "rest_png": rest_png,
        }

    if args.layout == "rest_map_anchor":
        rest_render_image = load_render_image(data["rest_render_image_path"])
        contact_render_image = load_render_image(data["contact_render_image_path"])
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(9.35, 3.35),
            gridspec_kw={"wspace": 0.16},
            constrained_layout=True,
        )
        render_rest_map_anchor_raw_panel(
            axes[0],
            "Resting State",
            rest_render_image,
            data["rest_overview_mass_xy"],
            data["rest_raw_graph_edges"],
            data["anchor_structural_nodes"],
            data["anchor_structural_edges"],
            data["anchor_node"],
            data["rest_raw_pruning_circle"],
            data["rest_render_zoom_xlim"],
            data["rest_render_zoom_ylim"],
        )
        render_rest_map_anchor_raw_panel(
            axes[1],
            "Simulation Stage",
            contact_render_image,
            data["contact_overview_mass_xy"],
            data["contact_raw_graph_edges"],
            data["anchor_structural_nodes"],
            data["anchor_structural_edges"],
            data["anchor_node"],
            data["contact_raw_pruning_circle"],
            data["contact_render_zoom_xlim"],
            data["contact_render_zoom_ylim"],
        )
        draw_rest_map_anchor_center_legend(fig)
        fig.savefig(combined_png, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.012)
        fig.savefig(combined_pdf, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.012)
        plt.close(fig)

        for path, key, title in (
            (contact_png, "contact", "Simulation Stage"),
            (rest_png, "rest", "Resting State"),
        ):
            render_image = contact_render_image if key == "contact" else rest_render_image
            fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.35), constrained_layout=True)
            render_rest_map_anchor_raw_panel(
                ax,
                title,
                render_image,
                data[f"{key}_overview_mass_xy"],
                data[f"{key}_raw_graph_edges"],
                data["anchor_structural_nodes"],
                data["anchor_structural_edges"],
                data["anchor_node"],
                data[f"{key}_raw_pruning_circle"],
                data[f"{key}_render_zoom_xlim"],
                data[f"{key}_render_zoom_ylim"],
                show_labels=True,
            )
            fig.savefig(path, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.035)
            plt.close(fig)

        return {
            "combined_png": combined_png,
            "combined_pdf": combined_pdf,
            "contact_png": contact_png,
            "rest_png": rest_png,
        }

    if args.layout == "leg_focus_highlight":
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7.2, 3.35),
            gridspec_kw={"wspace": 0.08},
            constrained_layout=True,
        )
        render_leg_focus_panel(
            axes[0],
            "Resting State",
            data["rest_leg_gaussian_xy"],
            data["rest_leg_gaussian_rgb"],
            data["rest_overview_mass_xy"],
            data["rest_leg_edges"],
            data["leg_selected_nodes"],
            data["leg_selected_edges"],
            data["rest_leg_xlim"],
            data["rest_leg_ylim"],
            show_labels=True,
        )
        render_leg_focus_panel(
            axes[1],
            "Simulation Stage",
            data["contact_leg_gaussian_xy"],
            data["contact_leg_gaussian_rgb"],
            data["contact_overview_mass_xy"],
            data["contact_leg_edges"],
            data["leg_selected_nodes"],
            data["leg_selected_edges"],
            data["contact_leg_xlim"],
            data["contact_leg_ylim"],
        )
        fig.savefig(combined_png, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.035)
        fig.savefig(combined_pdf, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.035)
        plt.close(fig)

        for path, key, title in (
            (contact_png, "contact", "Simulation Stage"),
            (rest_png, "rest", "Resting State"),
        ):
            fig, ax = plt.subplots(1, 1, figsize=(3.7, 3.55), constrained_layout=True)
            render_leg_focus_panel(
                ax,
                title,
                data[f"{key}_leg_gaussian_xy"],
                data[f"{key}_leg_gaussian_rgb"],
                data[f"{key}_overview_mass_xy"],
                data[f"{key}_leg_edges"],
                data["leg_selected_nodes"],
                data["leg_selected_edges"],
                data[f"{key}_leg_xlim"],
                data[f"{key}_leg_ylim"],
                show_labels=True,
            )
            fig.savefig(path, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.035)
            plt.close(fig)

        return {
            "combined_png": combined_png,
            "combined_pdf": combined_pdf,
            "contact_png": contact_png,
            "rest_png": rest_png,
        }

    fig, axes = plt.subplots(1, 2, figsize=(6.25, 1.35), gridspec_kw={"wspace": -0.35})
    draw_shared_encoding_legend(
        fig,
        data["rest_overview_xy"],
        data["rest_overview_rgb"],
        data["rest_overview_xlim"],
        data["rest_overview_ylim"],
    )
    render_overview_with_zoom(
        axes[0],
        "Resting State",
        data["rest_overview_xy"],
        data["rest_overview_rgb"],
        data["rest_overview_mass_xy"],
        data["rest_overview_edges"],
        data["rest_zoom_xy"],
        data["selected_nodes"],
        data["selected_edges"],
        data["rest_zoom_context_nodes"],
        data["rest_zoom_context_edges"],
        data["rest_gaussian_crop_xy"],
        data["rest_gaussian_crop_rgb"],
        data["rest_gaussian_crop_cov2"],
        data["rest_gaussian_crop_opacity"],
        data["rest_zoom_circle"],
        data["rest_xlim"],
        data["rest_ylim"],
        data["rest_overview_xlim"],
        data["rest_overview_ylim"],
        data["node_labels"],
        overview_circle=data["rest_overview_circle"],
        title_x=0.28,
        callout_text="Structurally\nadjacent pair",
        callout_position=(0.075, 0.80),
        callout_target_angle_deg=200.0,
        callout_arrow_line_index=1,
        callout_arrow_line_side="right",
    )
    render_overview_with_zoom(
        axes[1],
        "Simulation Stage",
        data["contact_overview_xy"],
        data["contact_overview_rgb"],
        data["contact_overview_mass_xy"],
        data["contact_overview_edges"],
        data["contact_zoom_xy"],
        data["selected_nodes"],
        data["selected_edges"],
        data["contact_zoom_context_nodes"],
        data["contact_zoom_context_edges"],
        data["contact_gaussian_crop_xy"],
        data["contact_gaussian_crop_rgb"],
        data["contact_gaussian_crop_cov2"],
        data["contact_gaussian_crop_opacity"],
        data["contact_zoom_circle"],
        data["contact_xlim"],
        data["contact_ylim"],
        data["contact_overview_xlim"],
        data["contact_overview_ylim"],
        data["node_labels"],
        label="Collision\nresponse\npruned for\nstructural\nneighbors",
        overview_circle=data["contact_overview_circle"],
        inset_bounds=[0.145, 0.315, 0.43, 0.675],
        title_x=0.26,
        callout_position=(0.485, 0.405),
        callout_target_angle_deg=45.0,
        callout_arrow_line_index=2,
        callout_arrow_line_side="left",
    )
    save_combined_with_custom_crop(fig, combined_png, int(args.dpi))
    save_combined_with_custom_crop(fig, combined_pdf, int(args.dpi))
    plt.close(fig)

    for path, key, title, label in (
        (
            contact_png,
            "contact",
            "Simulation Stage",
            "Collision\nresponse\npruned for\nstructural\nneighbors",
        ),
        (rest_png, "rest", "Resting State", "Structurally\nadjacent pair"),
    ):
        fig, ax = plt.subplots(1, 1, figsize=(3.3, 1.58))
        render_overview_with_zoom(
            ax,
            title,
            data[f"{key}_overview_xy"],
            data[f"{key}_overview_rgb"],
            data[f"{key}_overview_mass_xy"],
            data[f"{key}_overview_edges"],
            data[f"{key}_zoom_xy"],
            data["selected_nodes"],
            data["selected_edges"],
            data[f"{key}_zoom_context_nodes"],
            data[f"{key}_zoom_context_edges"],
            data[f"{key}_gaussian_crop_xy"],
            data[f"{key}_gaussian_crop_rgb"],
            data[f"{key}_gaussian_crop_cov2"],
            data[f"{key}_gaussian_crop_opacity"],
            data[f"{key}_zoom_circle"],
            data[f"{key}_xlim"],
            data[f"{key}_ylim"],
            data[f"{key}_overview_xlim"],
            data[f"{key}_overview_ylim"],
            data["node_labels"],
            label=label,
            overview_circle=data[f"{key}_overview_circle"],
            title_x=0.26 if key == "contact" else 0.28,
            callout_position=(0.485, 0.405) if key == "contact" else (0.075, 0.80),
            callout_target_angle_deg=45.0 if key == "contact" else 200.0,
            callout_arrow_line_index=2 if key == "contact" else 1,
            callout_arrow_line_side="left" if key == "contact" else "right",
        )
        fig.savefig(path, dpi=int(args.dpi), bbox_inches="tight", pad_inches=0.035)
        plt.close(fig)

    return {
        "combined_png": combined_png,
        "combined_pdf": combined_pdf,
        "contact_png": contact_png,
        "rest_png": rest_png,
    }


def relative_to_root(path, root):
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def main():
    args = parse_args()
    root = repo_root()
    export_npz = resolve_path(args.export_npz, root)
    config_path = resolve_path(args.config, root)
    output_dir = resolve_path(args.output_dir, root)
    output_json = output_dir / f"{args.output_stem}.json"

    if not export_npz.exists():
        raise FileNotFoundError(
            f"Missing export data: {export_npz}. Run interactive_playground.py "
            "with --mode quality --collision_pruning_export_path first."
        )

    config = read_config(config_path)
    object_radius = float(config.get("object_radius", 0.02))
    object_max_neighbours = int(config.get("object_max_neighbours", 30))
    collision_dist = float(config.get("collision_dist", 0.02))

    with np.load(export_npz) as export:
        export_keys = set(export.files)
        frames = np.asarray(export["frames"], dtype=np.int32)
        mass_nodes = np.asarray(export["mass_nodes"], dtype=np.float64)
        gaussian_xyz = np.asarray(export["gaussian_xyz"], dtype=np.float64)
        gaussian_rgb = np.asarray(export["gaussian_rgb"], dtype=np.float64)
        gaussian_quat = (
            np.asarray(export["gaussian_quat"], dtype=np.float64)
            if "gaussian_quat" in export_keys
            else None
        )
        gaussian_scales_rest = (
            np.asarray(export["gaussian_scales_rest"], dtype=np.float64)
            if "gaussian_scales_rest" in export_keys
            else None
        )
        gaussian_opacity = (
            np.asarray(export["gaussian_opacity"], dtype=np.float64).reshape(-1)
            if "gaussian_opacity" in export_keys
            else None
        )
        export_schema_version = int(
            np.asarray(export["export_schema_version"])
            if "export_schema_version" in export_keys
            else 1
        )
        gaussian_kernel_parameterization = str(
            np.asarray(export["gaussian_kernel_parameterization"]).item()
            if "gaussian_kernel_parameterization" in export_keys
            else "centers_only"
        )
        c2ws = np.asarray(export["c2ws"], dtype=np.float64)
        intrinsics = np.asarray(export["intrinsics"], dtype=np.float64)
        wh = np.asarray(export["WH"], dtype=np.int32)
        object_mass_node_count = int(np.asarray(export["object_mass_node_count"]))
        gaussian_count = int(np.asarray(export["gaussian_count"]))

    rest_idx = frame_index(frames, int(args.rest_frame))
    contact_idx = frame_index(frames, int(args.contact_frame))
    camera_index = int(args.camera_index)
    if camera_index < 0 or camera_index >= c2ws.shape[0]:
        raise ValueError(f"camera_index must be in [0, {c2ws.shape[0] - 1}]")
    true_covariance_available = (
        gaussian_quat is not None
        and gaussian_scales_rest is not None
        and gaussian_quat.ndim == 3
        and gaussian_quat.shape[0] == frames.shape[0]
        and gaussian_quat.shape[1] == gaussian_count
        and gaussian_quat.shape[2] == 4
        and gaussian_scales_rest.shape[0] == gaussian_count
    )
    if true_covariance_available:
        gaussian_scales_rest = np.asarray(gaussian_scales_rest, dtype=np.float64).reshape(
            gaussian_count, -1
        )
        if gaussian_scales_rest.shape[1] == 1:
            gaussian_scales_rest = np.repeat(gaussian_scales_rest, 3, axis=1)
        elif gaussian_scales_rest.shape[1] != 3:
            raise ValueError(
                "gaussian_scales_rest must have shape (N, 1) or (N, 3) for "
                "true covariance rendering"
            )
    if gaussian_opacity is not None and gaussian_opacity.shape[0] != gaussian_count:
        gaussian_opacity = None

    rest_nodes = mass_nodes[rest_idx]
    contact_nodes = mass_nodes[contact_idx]
    rest_gaussians = gaussian_xyz[rest_idx]
    contact_gaussians = gaussian_xyz[contact_idx]
    if gaussian_rgb.ndim == 3:
        rest_rgb = gaussian_rgb[rest_idx]
        contact_rgb = gaussian_rgb[contact_idx]
    else:
        rest_rgb = gaussian_rgb
        contact_rgb = gaussian_rgb

    rest_xy, rest_valid = project_points(rest_nodes, c2ws[camera_index], intrinsics[camera_index])
    contact_xy, contact_valid = project_points(
        contact_nodes, c2ws[camera_index], intrinsics[camera_index]
    )
    rest_gaussian_xy, rest_gaussian_valid = project_points(
        rest_gaussians, c2ws[camera_index], intrinsics[camera_index]
    )
    contact_gaussian_xy, contact_gaussian_valid = project_points(
        contact_gaussians, c2ws[camera_index], intrinsics[camera_index]
    )
    rest_gaussians_visible = rest_gaussians[rest_gaussian_valid]
    contact_gaussians_visible = contact_gaussians[contact_gaussian_valid]
    rest_gaussian_xy = rest_gaussian_xy[rest_gaussian_valid]
    contact_gaussian_xy = contact_gaussian_xy[contact_gaussian_valid]
    rest_rgb = rest_rgb[rest_gaussian_valid]
    contact_rgb = contact_rgb[contact_gaussian_valid]
    rest_gaussian_cov3 = None
    contact_gaussian_cov3 = None
    rest_gaussian_opacity_visible = None
    contact_gaussian_opacity_visible = None
    if true_covariance_available:
        rest_gaussian_cov3 = gaussian_covariances_3d(
            gaussian_scales_rest[rest_gaussian_valid],
            gaussian_quat[rest_idx][rest_gaussian_valid],
        )
        contact_gaussian_cov3 = gaussian_covariances_3d(
            gaussian_scales_rest[contact_gaussian_valid],
            gaussian_quat[contact_idx][contact_gaussian_valid],
        )
        if gaussian_opacity is not None:
            rest_gaussian_opacity_visible = gaussian_opacity[rest_gaussian_valid]
            contact_gaussian_opacity_visible = gaussian_opacity[contact_gaussian_valid]

    edges, adjacency = build_spring_graph(rest_nodes, object_radius, object_max_neighbours)
    selected_nodes, selected_edges, selection = choose_leg_patch(
        rest_xy,
        contact_xy,
        rest_valid,
        contact_valid,
        adjacency,
        contact_gaussian_xy,
        rest_nodes,
        contact_nodes,
        args,
    )
    if len(selected_nodes) > int(args.max_local_nodes):
        raise RuntimeError("Selected patch exceeds --max_local_nodes")
    for edge in selected_edges:
        if int(edge[1]) not in adjacency[int(edge[0])]:
            raise RuntimeError(f"Selected spring edge {edge.tolist()} is missing from the rest graph")

    selected_nodes_array = np.asarray(selected_nodes, dtype=np.int64)
    node_labels = {
        int(node): f"M{idx}" for idx, node in enumerate(selected_nodes_array.tolist())
    }
    if selected_nodes_array.size == 3:
        alignment_fit_nodes = selected_nodes_array
        alignment_excluded_nodes = selected_nodes_array[:0]
    else:
        alignment_fit_nodes = selected_nodes_array[1:]
        alignment_excluded_nodes = selected_nodes_array[:1]
    alignment_fit_labels = [node_labels[int(node)] for node in alignment_fit_nodes.tolist()]
    alignment_excluded_labels = [
        node_labels[int(node)] for node in alignment_excluded_nodes.tolist()
    ]

    contact_zoom_alignment_3d = rigid_alignment_3d(
        contact_nodes[alignment_fit_nodes],
        rest_nodes[alignment_fit_nodes],
    )
    contact_zoom_nodes_3d = apply_rigid_alignment_3d(
        contact_nodes,
        contact_zoom_alignment_3d,
    )
    contact_zoom_gaussians_3d = apply_rigid_alignment_3d(
        contact_gaussians_visible,
        contact_zoom_alignment_3d,
    )
    zoom_basis = tangent_plane_basis(
        rest_nodes,
        selected_nodes_array,
        alignment_fit_nodes,
        c2ws[camera_index][:3, 3],
        args.preferred_patch_span_px,
    )
    rest_zoom_gaussian_cov2 = None
    contact_zoom_gaussian_cov2 = None
    if true_covariance_available:
        rest_zoom_gaussian_cov2 = project_covariances_to_tangent_plane(
            rest_gaussian_cov3,
            zoom_basis,
        )
        contact_zoom_gaussian_cov2 = project_covariances_to_tangent_plane(
            transform_covariances_by_alignment_3d(
                contact_gaussian_cov3,
                contact_zoom_alignment_3d,
            ),
            zoom_basis,
        )
    rest_zoom_xy = project_to_tangent_plane(
        rest_nodes,
        zoom_basis["origin"],
        zoom_basis["x_axis"],
        zoom_basis["y_axis"],
        zoom_basis["scale"],
    )
    contact_zoom_xy = project_to_tangent_plane(
        contact_zoom_nodes_3d,
        zoom_basis["origin"],
        zoom_basis["x_axis"],
        zoom_basis["y_axis"],
        zoom_basis["scale"],
    )
    rest_zoom_gaussian_xy = project_to_tangent_plane(
        rest_gaussians_visible,
        zoom_basis["origin"],
        zoom_basis["x_axis"],
        zoom_basis["y_axis"],
        zoom_basis["scale"],
    )
    contact_zoom_gaussian_xy = project_to_tangent_plane(
        contact_zoom_gaussians_3d,
        zoom_basis["origin"],
        zoom_basis["x_axis"],
        zoom_basis["y_axis"],
        zoom_basis["scale"],
    )

    contact_overview_circle = visual_circle(
        contact_xy[selected_nodes_array], args.visual_pad_px
    )
    rest_overview_circle = visual_circle(rest_xy[selected_nodes_array], args.visual_pad_px)
    contact_zoom_circle = visual_circle(
        contact_zoom_xy[selected_nodes_array], args.visual_pad_px
    )
    rest_zoom_circle = visual_circle(rest_zoom_xy[selected_nodes_array], args.visual_pad_px)
    contact_zoom_crop_size_px, contact_zoom_crop_stats = context_aware_zoom_crop_size(
        contact_zoom_xy,
        edges,
        contact_zoom_circle,
        args.crop_size_px,
        selected_nodes_array,
        max_crop_size_px=16.5,
    )
    rest_zoom_crop_size_px, rest_zoom_crop_stats = context_aware_zoom_crop_size(
        rest_zoom_xy,
        edges,
        rest_zoom_circle,
        args.crop_size_px,
        selected_nodes_array,
        max_crop_size_px=16.5,
    )
    contact_xlim, contact_ylim = crop_limits(
        contact_zoom_circle[0], contact_zoom_crop_size_px
    )
    rest_xlim, rest_ylim = crop_limits(rest_zoom_circle[0], rest_zoom_crop_size_px)
    contact_overview_xlim, contact_overview_ylim = full_object_limits(contact_gaussian_xy)
    rest_overview_xlim, rest_overview_ylim = full_object_limits(rest_gaussian_xy)

    (
        contact_gaussian_crop_xy,
        contact_gaussian_crop_rgb,
        contact_gaussian_crop_indices,
    ) = crop_gaussians(
        contact_zoom_gaussian_xy,
        contact_rgb,
        contact_xlim,
        contact_ylim,
        args,
        args.contact_frame,
        return_indices=True,
    )
    (
        rest_gaussian_crop_xy,
        rest_gaussian_crop_rgb,
        rest_gaussian_crop_indices,
    ) = crop_gaussians(
        rest_zoom_gaussian_xy,
        rest_rgb,
        rest_xlim,
        rest_ylim,
        args,
        args.rest_frame,
        return_indices=True,
    )
    contact_gaussian_crop_rgb = np.clip(contact_rgb[contact_gaussian_crop_indices], 0.0, 1.0)
    rest_gaussian_crop_rgb = np.clip(rest_rgb[rest_gaussian_crop_indices], 0.0, 1.0)
    if true_covariance_available:
        contact_gaussian_crop_cov2 = contact_zoom_gaussian_cov2[
            contact_gaussian_crop_indices
        ]
        rest_gaussian_crop_cov2 = rest_zoom_gaussian_cov2[rest_gaussian_crop_indices]
    else:
        contact_gaussian_crop_cov2 = None
        rest_gaussian_crop_cov2 = None
    contact_gaussian_crop_opacity = (
        contact_gaussian_opacity_visible[contact_gaussian_crop_indices]
        if contact_gaussian_opacity_visible is not None
        else None
    )
    rest_gaussian_crop_opacity = (
        rest_gaussian_opacity_visible[rest_gaussian_crop_indices]
        if rest_gaussian_opacity_visible is not None
        else None
    )
    contact_overview_xy, contact_overview_rgb = crop_gaussians(
        contact_gaussian_xy,
        contact_rgb,
        contact_overview_xlim,
        contact_overview_ylim,
        args,
        args.contact_frame + 1000,
    )
    rest_overview_xy, rest_overview_rgb = crop_gaussians(
        rest_gaussian_xy,
        rest_rgb,
        rest_overview_xlim,
        rest_overview_ylim,
        args,
        args.rest_frame + 1000,
    )
    contact_overview_edges_available = projected_edges_in_mask(
        edges,
        projected_node_mask(contact_xy, contact_overview_xlim, contact_overview_ylim),
    )
    rest_overview_edges_available = projected_edges_in_mask(
        edges,
        projected_node_mask(rest_xy, rest_overview_xlim, rest_overview_ylim),
    )
    contact_overview_edges = downsample_edges(
        contact_overview_edges_available,
        2200,
        int(args.seed) + 2001,
    )
    rest_overview_edges = downsample_edges(
        rest_overview_edges_available,
        2200,
        int(args.seed) + 2002,
    )
    if float(args.leg_crop_size_px) <= 0.0:
        raise ValueError("--leg_crop_size_px must be positive")
    rest_leg_center = rest_xy[selected_nodes_array].mean(axis=0)
    contact_leg_center = contact_xy[selected_nodes_array].mean(axis=0)
    rest_leg_xlim, rest_leg_ylim = crop_limits(rest_leg_center, args.leg_crop_size_px)
    contact_leg_xlim, contact_leg_ylim = crop_limits(
        contact_leg_center,
        args.leg_crop_size_px,
    )
    rest_leg_gaussian_xy, rest_leg_gaussian_rgb = crop_gaussians(
        rest_gaussian_xy,
        rest_rgb,
        rest_leg_xlim,
        rest_leg_ylim,
        args,
        args.rest_frame + 3000,
    )
    contact_leg_gaussian_xy, contact_leg_gaussian_rgb = crop_gaussians(
        contact_gaussian_xy,
        contact_rgb,
        contact_leg_xlim,
        contact_leg_ylim,
        args,
        args.contact_frame + 3000,
    )
    rest_leg_node_mask = projected_node_mask(rest_xy, rest_leg_xlim, rest_leg_ylim)
    contact_leg_node_mask = projected_node_mask(
        contact_xy,
        contact_leg_xlim,
        contact_leg_ylim,
    )
    rest_leg_edges = projected_edges_in_mask(edges, rest_leg_node_mask)
    contact_leg_edges = projected_edges_in_mask(edges, contact_leg_node_mask)
    leg_selected_nodes = expand_connected_visible_patch(
        selected_nodes_array,
        adjacency,
        rest_xy,
        contact_xy,
        rest_leg_node_mask,
        contact_leg_node_mask,
        rest_leg_center,
        contact_leg_center,
        args.selected_patch_max_nodes,
    )
    leg_selected_edges = induced_edges(leg_selected_nodes, adjacency)
    leg_spring_lengths, leg_spring_length_summary = spring_length_metrics(
        leg_selected_edges,
        rest_nodes,
        contact_nodes,
    )
    anchor_node = int(args.anchor_node)
    if anchor_node < 0 or anchor_node >= rest_nodes.shape[0]:
        raise ValueError(
            f"--anchor_node {anchor_node} is outside the valid mass-node range "
            f"[0, {rest_nodes.shape[0] - 1}]"
        )
    if float(args.anchor_crop_size_px) <= 0.0:
        raise ValueError("--anchor_crop_size_px must be positive")
    if float(args.structural_radius_px) <= 0.0:
        raise ValueError("--structural_radius_px must be positive")
    if float(args.render_zoom_crop_size_px) <= 0.0:
        raise ValueError("--render_zoom_crop_size_px must be positive")
    if not np.isfinite(rest_xy[anchor_node]).all():
        raise RuntimeError(f"Anchor node {anchor_node} is not visible in the rest frame")
    if not np.isfinite(contact_xy[anchor_node]).all():
        raise RuntimeError(
            f"Anchor node {anchor_node} is not visible in the contact frame"
        )

    anchor_structural_radius_px = float(args.structural_radius_px)
    rest_map_radius = collision_dist * 5.0
    anchor_spring_edges = incident_edges(anchor_node, adjacency)
    anchor_spring_nodes = (
        np.unique(anchor_spring_edges.reshape(-1))
        if anchor_spring_edges.size
        else np.asarray([anchor_node], dtype=np.int64)
    )
    anchor_spring_lengths, anchor_spring_length_summary = spring_length_metrics(
        anchor_spring_edges,
        rest_nodes,
        contact_nodes,
    )
    anchor_rest_map_nodes, anchor_rest_map_distances = rest_map_neighbors_for_anchor(
        rest_nodes,
        anchor_node,
        rest_map_radius,
    )
    anchor_endpoint_nodes = np.asarray(
        [node for node in anchor_spring_nodes.tolist() if int(node) != anchor_node],
        dtype=np.int64,
    )
    rest_anchor_center = rest_xy[anchor_node]
    contact_anchor_center = contact_xy[anchor_node]
    rest_anchor_world = rest_nodes[anchor_node]
    contact_anchor_world = contact_nodes[anchor_node]
    if anchor_endpoint_nodes.size:
        rest_anchor_endpoint_offsets_3d = (
            rest_nodes[anchor_endpoint_nodes] - rest_anchor_world
        )
        contact_anchor_endpoint_offsets_3d = (
            contact_nodes[anchor_endpoint_nodes] - contact_anchor_world
        )
        contact_anchor_3d_alignment = rotation_only_alignment_nd(
            contact_anchor_endpoint_offsets_3d,
            rest_anchor_endpoint_offsets_3d,
        )
    else:
        contact_anchor_3d_alignment = rotation_only_alignment_nd(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
    camera_x_axis = np.asarray(c2ws[camera_index][:3, 0], dtype=np.float64)
    camera_y_axis = np.asarray(c2ws[camera_index][:3, 1], dtype=np.float64)
    camera_x_axis = camera_x_axis / np.linalg.norm(camera_x_axis)
    camera_y_axis = camera_y_axis / np.linalg.norm(camera_y_axis)
    anchor_local_scale_px_per_world = projected_world_radius_px(
        rest_anchor_world,
        collision_dist,
        c2ws[camera_index],
        intrinsics[camera_index],
    ) / max(float(collision_dist), MIN_REST_LENGTH)
    if not np.isfinite(anchor_local_scale_px_per_world):
        anchor_local_scale_px_per_world = 1.0
    rest_anchor_mass_xy = anchor_local_project_world(
        rest_nodes,
        rest_anchor_world,
        camera_x_axis,
        camera_y_axis,
        anchor_local_scale_px_per_world,
    )
    contact_anchor_mass_xy = anchor_local_project_world(
        contact_nodes,
        contact_anchor_world,
        camera_x_axis,
        camera_y_axis,
        anchor_local_scale_px_per_world,
        contact_anchor_3d_alignment["rotation"],
    )
    if anchor_endpoint_nodes.size:
        rest_anchor_endpoint_distances_px = np.linalg.norm(
            rest_anchor_mass_xy[anchor_endpoint_nodes],
            axis=1,
        )
        contact_anchor_endpoint_distances_px = np.linalg.norm(
            contact_anchor_mass_xy[anchor_endpoint_nodes],
            axis=1,
        )
        anchor_visual_radius_px = anchor_structural_radius_px
        rest_anchor_visual_radius_endpoint_coverage = float(
            np.mean(rest_anchor_endpoint_distances_px <= anchor_visual_radius_px)
        )
        contact_anchor_visual_radius_endpoint_coverage = float(
            np.mean(contact_anchor_endpoint_distances_px <= anchor_visual_radius_px)
        )
    else:
        rest_anchor_endpoint_distances_px = np.empty((0,), dtype=np.float64)
        contact_anchor_endpoint_distances_px = np.empty((0,), dtype=np.float64)
        anchor_visual_radius_px = anchor_structural_radius_px
        rest_anchor_visual_radius_endpoint_coverage = 0.0
        contact_anchor_visual_radius_endpoint_coverage = 0.0
    rest_anchor_local_xlim, rest_anchor_local_ylim = crop_limits(
        np.asarray([0.0, 0.0], dtype=np.float64),
        args.anchor_crop_size_px,
    )
    contact_anchor_local_xlim, contact_anchor_local_ylim = crop_limits(
        np.asarray([0.0, 0.0], dtype=np.float64),
        args.anchor_crop_size_px,
    )
    rest_anchor_xlim, rest_anchor_ylim = crop_limits(
        rest_anchor_center,
        args.anchor_crop_size_px,
    )
    contact_anchor_xlim, contact_anchor_ylim = crop_limits(
        contact_anchor_center,
        args.anchor_crop_size_px,
    )
    rest_anchor_gaussian_all_xy = anchor_local_project_world(
        rest_gaussians_visible,
        rest_anchor_world,
        camera_x_axis,
        camera_y_axis,
        anchor_local_scale_px_per_world,
    )
    contact_anchor_gaussian_all_xy = anchor_local_project_world(
        contact_gaussians_visible,
        contact_anchor_world,
        camera_x_axis,
        camera_y_axis,
        anchor_local_scale_px_per_world,
        contact_anchor_3d_alignment["rotation"],
    )
    rest_anchor_gaussian_xy, rest_anchor_gaussian_rgb = crop_gaussians(
        rest_anchor_gaussian_all_xy,
        rest_rgb,
        rest_anchor_local_xlim,
        rest_anchor_local_ylim,
        args,
        args.rest_frame + 4000,
    )
    contact_anchor_gaussian_xy, contact_anchor_gaussian_rgb = crop_gaussians(
        contact_anchor_gaussian_all_xy,
        contact_rgb,
        contact_anchor_local_xlim,
        contact_anchor_local_ylim,
        args,
        args.contact_frame + 4000,
    )
    if anchor_endpoint_nodes.size:
        anchor_local_residuals_px = np.linalg.norm(
            contact_anchor_mass_xy[anchor_endpoint_nodes]
            - rest_anchor_mass_xy[anchor_endpoint_nodes],
            axis=1,
        )
        anchor_local_alignment_residual_rms_px = float(
            np.sqrt(np.mean(anchor_local_residuals_px**2))
        )
        anchor_local_alignment_residual_max_px = float(np.max(anchor_local_residuals_px))
        anchor_local_alignment_residual_rms_normalized = float(
            anchor_local_alignment_residual_rms_px
            / max(float(np.max(rest_anchor_endpoint_distances_px)), 1e-8)
        )
    else:
        anchor_local_alignment_residual_rms_px = 0.0
        anchor_local_alignment_residual_max_px = 0.0
        anchor_local_alignment_residual_rms_normalized = 0.0
    rest_anchor_node_mask = projected_node_mask(
        rest_anchor_mass_xy,
        rest_anchor_local_xlim,
        rest_anchor_local_ylim,
    )
    contact_anchor_node_mask = projected_node_mask(
        contact_anchor_mass_xy,
        contact_anchor_local_xlim,
        contact_anchor_local_ylim,
    )
    rest_anchor_edges = projected_edges_in_mask(edges, rest_anchor_node_mask)
    contact_anchor_edges = projected_edges_in_mask(edges, contact_anchor_node_mask)
    rest_anchor_distance_from_anchor_px = np.linalg.norm(rest_anchor_mass_xy, axis=1)
    contact_anchor_distance_from_anchor_px = np.linalg.norm(
        contact_anchor_mass_xy,
        axis=1,
    )
    anchor_structural_node_mask = (
        rest_anchor_node_mask
        & (rest_anchor_distance_from_anchor_px <= anchor_visual_radius_px)
    )
    anchor_structural_nodes = np.flatnonzero(anchor_structural_node_mask).astype(
        np.int64
    )
    anchor_structural_edges = projected_edges_in_mask(
        edges,
        anchor_structural_node_mask,
    )
    (
        anchor_structural_spring_lengths,
        anchor_structural_spring_length_summary,
    ) = spring_length_metrics(anchor_structural_edges, rest_nodes, contact_nodes)
    rest_anchor_structural_nodes = nodes_visible_in_crop(
        anchor_structural_nodes,
        rest_anchor_node_mask,
    )
    contact_anchor_structural_nodes = nodes_visible_in_crop(
        anchor_structural_nodes,
        contact_anchor_node_mask,
    )
    rest_anchor_structural_edges = projected_edges_in_mask(
        anchor_structural_edges,
        rest_anchor_node_mask,
    )
    contact_anchor_structural_edges = projected_edges_in_mask(
        anchor_structural_edges,
        contact_anchor_node_mask,
    )
    (
        rest_anchor_context_edges,
        rest_anchor_context_nodes,
        rest_anchor_context_stats,
    ) = outside_anchor_context_graph(
        rest_anchor_mass_xy,
        rest_anchor_edges,
        anchor_structural_nodes,
        anchor_visual_radius_px,
    )
    (
        contact_anchor_context_edges,
        contact_anchor_context_nodes,
        contact_anchor_context_stats,
    ) = outside_anchor_context_graph(
        contact_anchor_mass_xy,
        contact_anchor_edges,
        anchor_structural_nodes,
        anchor_visual_radius_px,
    )
    rest_anchor_structural_node_coverage = (
        float(
            np.mean(
                rest_anchor_distance_from_anchor_px[anchor_structural_nodes]
                <= anchor_visual_radius_px
            )
        )
        if anchor_structural_nodes.size
        else 0.0
    )
    contact_anchor_structural_node_coverage = (
        float(
            np.mean(
                contact_anchor_distance_from_anchor_px[anchor_structural_nodes]
                <= anchor_visual_radius_px
            )
        )
        if anchor_structural_nodes.size
        else 0.0
    )
    anchor_structural_nodes_without_anchor = anchor_structural_nodes[
        anchor_structural_nodes != anchor_node
    ]
    if anchor_structural_nodes_without_anchor.size:
        structural_node_residuals_px = np.linalg.norm(
            contact_anchor_mass_xy[anchor_structural_nodes_without_anchor]
            - rest_anchor_mass_xy[anchor_structural_nodes_without_anchor],
            axis=1,
        )
        structural_node_residual_summary = {
            "node_count_excluding_anchor": int(
                anchor_structural_nodes_without_anchor.shape[0]
            ),
            "rms_px": float(np.sqrt(np.mean(structural_node_residuals_px**2))),
            "median_px": float(np.median(structural_node_residuals_px)),
            "max_px": float(np.max(structural_node_residuals_px)),
            "rms_over_structural_radius": float(
                np.sqrt(np.mean(structural_node_residuals_px**2))
                / max(anchor_visual_radius_px, 1e-8)
            ),
        }
    else:
        structural_node_residuals_px = np.empty((0,), dtype=np.float64)
        structural_node_residual_summary = {
            "node_count_excluding_anchor": 0,
            "rms_px": 0.0,
            "median_px": 0.0,
            "max_px": 0.0,
            "rms_over_structural_radius": 0.0,
        }
    rest_anchor_rest_map_nodes = nodes_visible_in_crop(
        anchor_rest_map_nodes,
        rest_anchor_node_mask,
    )
    contact_anchor_rest_map_nodes = nodes_visible_in_crop(
        anchor_rest_map_nodes,
        contact_anchor_node_mask,
    )
    rest_anchor_spring_nodes = nodes_visible_in_crop(
        anchor_spring_nodes,
        rest_anchor_node_mask,
    )
    contact_anchor_spring_nodes = nodes_visible_in_crop(
        anchor_spring_nodes,
        contact_anchor_node_mask,
    )
    rest_anchor_spring_edges = projected_edges_in_mask(
        anchor_spring_edges,
        rest_anchor_node_mask,
    )
    contact_anchor_spring_edges = projected_edges_in_mask(
        anchor_spring_edges,
        contact_anchor_node_mask,
    )
    rest_inset_render_path = None
    contact_inset_render_path = None
    rest_inset_render_size = None
    contact_inset_render_size = None
    rest_inset_alpha_bbox = None
    contact_inset_alpha_bbox = None
    if args.layout == "rest_map_anchor":
        expected_render_size = (int(wh[0]), int(wh[1]))
        rest_inset_render_path = boba_quality_render_path(
            root,
            args.case_name,
            camera_index,
            args.rest_frame,
        )
        contact_inset_render_path = boba_quality_render_path(
            root,
            args.case_name,
            camera_index,
            args.contact_frame,
        )
        for label, path in (
            ("rest", rest_inset_render_path),
            ("contact", contact_inset_render_path),
        ):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing Boba quality render PNG for {label} inset: {path}"
                )
        rest_inset_render_size = png_dimensions(rest_inset_render_path)
        contact_inset_render_size = png_dimensions(contact_inset_render_path)
        for label, size, path in (
            ("rest", rest_inset_render_size, rest_inset_render_path),
            ("contact", contact_inset_render_size, contact_inset_render_path),
        ):
            if tuple(size) != expected_render_size:
                raise ValueError(
                    f"Boba quality render PNG for {label} inset has size {size}, "
                    f"but the exported camera frame is {expected_render_size}: {path}"
                )
        rest_inset_alpha_bbox = alpha_bbox_metadata(
            load_render_image(rest_inset_render_path)
        )
        contact_inset_alpha_bbox = alpha_bbox_metadata(
            load_render_image(contact_inset_render_path)
        )
    image_xlim = (0.0, float(wh[0]))
    image_ylim = (0.0, float(wh[1]))
    rest_raw_pruning_circle = raw_anchor_pruning_circle(
        rest_xy,
        anchor_structural_nodes,
        anchor_node,
    )
    contact_raw_pruning_circle = raw_anchor_pruning_circle(
        contact_xy,
        anchor_structural_nodes,
        anchor_node,
    )
    rest_render_zoom_xlim, rest_render_zoom_ylim = clipped_crop_limits(
        rest_xy[anchor_node],
        args.render_zoom_crop_size_px,
        wh[0],
        wh[1],
    )
    contact_render_zoom_xlim, contact_render_zoom_ylim = clipped_crop_limits(
        contact_xy[anchor_node],
        args.render_zoom_crop_size_px,
        wh[0],
        wh[1],
    )
    rest_render_zoom_edges = crop_edges_for_raw_view(
        rest_overview_edges_available,
        rest_xy,
        rest_render_zoom_xlim,
        rest_render_zoom_ylim,
    )
    contact_render_zoom_edges = crop_edges_for_raw_view(
        contact_overview_edges_available,
        contact_xy,
        contact_render_zoom_xlim,
        contact_render_zoom_ylim,
    )
    contact_zoom_context_nodes, contact_zoom_context_edges = zoom_graph_context(
        contact_zoom_xy,
        edges,
        contact_xlim,
        contact_ylim,
        selected_nodes_array,
        selected_edges,
    )
    rest_zoom_context_nodes, rest_zoom_context_edges = zoom_graph_context(
        rest_zoom_xy,
        edges,
        rest_xlim,
        rest_ylim,
        selected_nodes_array,
        selected_edges,
    )
    _, _, _, contact_zoom_context_render_stats = visible_zoom_context_graph(
        contact_zoom_xy,
        contact_zoom_context_nodes,
        contact_zoom_context_edges,
        contact_zoom_circle,
        selected_nodes_array,
    )
    _, _, _, rest_zoom_context_render_stats = visible_zoom_context_graph(
        rest_zoom_xy,
        rest_zoom_context_nodes,
        rest_zoom_context_edges,
        rest_zoom_circle,
        selected_nodes_array,
    )
    render_data = {
        "selected_nodes": selected_nodes_array,
        "selected_edges": selected_edges,
        "node_labels": node_labels,
        "contact_zoom_xy": contact_zoom_xy,
        "rest_zoom_xy": rest_zoom_xy,
        "contact_overview_mass_xy": contact_xy,
        "rest_overview_mass_xy": rest_xy,
        "contact_zoom_circle": contact_zoom_circle,
        "rest_zoom_circle": rest_zoom_circle,
        "contact_overview_circle": contact_overview_circle,
        "rest_overview_circle": rest_overview_circle,
        "contact_overview_edges": contact_overview_edges,
        "rest_overview_edges": rest_overview_edges,
        "contact_raw_graph_edges": contact_overview_edges_available,
        "rest_raw_graph_edges": rest_overview_edges_available,
        "contact_raw_pruning_circle": contact_raw_pruning_circle,
        "rest_raw_pruning_circle": rest_raw_pruning_circle,
        "contact_render_zoom_edges": contact_render_zoom_edges,
        "rest_render_zoom_edges": rest_render_zoom_edges,
        "contact_render_zoom_xlim": contact_render_zoom_xlim,
        "contact_render_zoom_ylim": contact_render_zoom_ylim,
        "rest_render_zoom_xlim": rest_render_zoom_xlim,
        "rest_render_zoom_ylim": rest_render_zoom_ylim,
        "image_xlim": image_xlim,
        "image_ylim": image_ylim,
        "contact_zoom_context_nodes": contact_zoom_context_nodes,
        "contact_zoom_context_edges": contact_zoom_context_edges,
        "rest_zoom_context_nodes": rest_zoom_context_nodes,
        "rest_zoom_context_edges": rest_zoom_context_edges,
        "contact_zoom_context_render_stats": contact_zoom_context_render_stats,
        "rest_zoom_context_render_stats": rest_zoom_context_render_stats,
        "contact_zoom_crop_stats": contact_zoom_crop_stats,
        "rest_zoom_crop_stats": rest_zoom_crop_stats,
        "contact_xlim": contact_xlim,
        "contact_ylim": contact_ylim,
        "rest_xlim": rest_xlim,
        "rest_ylim": rest_ylim,
        "contact_overview_xlim": contact_overview_xlim,
        "contact_overview_ylim": contact_overview_ylim,
        "rest_overview_xlim": rest_overview_xlim,
        "rest_overview_ylim": rest_overview_ylim,
        "contact_gaussian_crop_xy": contact_gaussian_crop_xy,
        "contact_gaussian_crop_rgb": contact_gaussian_crop_rgb,
        "contact_gaussian_crop_cov2": contact_gaussian_crop_cov2,
        "contact_gaussian_crop_opacity": contact_gaussian_crop_opacity,
        "rest_gaussian_crop_xy": rest_gaussian_crop_xy,
        "rest_gaussian_crop_rgb": rest_gaussian_crop_rgb,
        "rest_gaussian_crop_cov2": rest_gaussian_crop_cov2,
        "rest_gaussian_crop_opacity": rest_gaussian_crop_opacity,
        "contact_overview_xy": contact_overview_xy,
        "contact_overview_rgb": contact_overview_rgb,
        "rest_overview_xy": rest_overview_xy,
        "rest_overview_rgb": rest_overview_rgb,
        "contact_leg_gaussian_xy": contact_leg_gaussian_xy,
        "contact_leg_gaussian_rgb": contact_leg_gaussian_rgb,
        "rest_leg_gaussian_xy": rest_leg_gaussian_xy,
        "rest_leg_gaussian_rgb": rest_leg_gaussian_rgb,
        "contact_leg_edges": contact_leg_edges,
        "rest_leg_edges": rest_leg_edges,
        "leg_selected_nodes": leg_selected_nodes,
        "leg_selected_edges": leg_selected_edges,
        "contact_leg_xlim": contact_leg_xlim,
        "contact_leg_ylim": contact_leg_ylim,
        "rest_leg_xlim": rest_leg_xlim,
        "rest_leg_ylim": rest_leg_ylim,
        "anchor_node": anchor_node,
        "anchor_spring_edges": anchor_spring_edges,
        "anchor_structural_nodes": anchor_structural_nodes,
        "anchor_structural_edges": anchor_structural_edges,
        "anchor_rest_map_nodes": anchor_rest_map_nodes,
        "contact_anchor_gaussian_xy": contact_anchor_gaussian_xy,
        "contact_anchor_gaussian_rgb": contact_anchor_gaussian_rgb,
        "rest_anchor_gaussian_xy": rest_anchor_gaussian_xy,
        "rest_anchor_gaussian_rgb": rest_anchor_gaussian_rgb,
        "contact_anchor_mass_xy": contact_anchor_mass_xy,
        "rest_anchor_mass_xy": rest_anchor_mass_xy,
        "contact_anchor_overview_xy": contact_xy[anchor_node],
        "rest_anchor_overview_xy": rest_xy[anchor_node],
        "contact_anchor_edges": contact_anchor_context_edges,
        "rest_anchor_edges": rest_anchor_context_edges,
        "contact_anchor_xlim": contact_anchor_local_xlim,
        "contact_anchor_ylim": contact_anchor_local_ylim,
        "rest_anchor_xlim": rest_anchor_local_xlim,
        "rest_anchor_ylim": rest_anchor_local_ylim,
        "contact_anchor_visual_radius_px": anchor_visual_radius_px,
        "rest_anchor_visual_radius_px": anchor_visual_radius_px,
        "contact_inset_image_path": contact_inset_render_path,
        "rest_inset_image_path": rest_inset_render_path,
        "contact_render_image_path": contact_inset_render_path,
        "rest_render_image_path": rest_inset_render_path,
    }
    outputs = render_figures(render_data, output_dir, args.output_stem, args)

    metadata = {
        "case_name": args.case_name,
        "export_npz": relative_to_root(export_npz, root),
        "config": relative_to_root(config_path, root),
        "rest_frame": int(args.rest_frame),
        "contact_frame": int(args.contact_frame),
        "camera_index": camera_index,
        "image_width": int(wh[0]),
        "image_height": int(wh[1]),
        "object_mass_node_count": object_mass_node_count,
        "gaussian_count": gaussian_count,
        "export_schema_version": int(export_schema_version),
        "gaussian_kernel_parameterization": gaussian_kernel_parameterization,
        "object_radius": object_radius,
        "object_max_neighbours": object_max_neighbours,
        "rest_spring_count": int(edges.shape[0]),
        "layout": "lower_left_overview_with_circular_zoom",
        "algorithmic_collision_dist": collision_dist,
        "algorithmic_pruning_radius_collision_dist_times_5": collision_dist * 5.0,
        "visual_circle_uses_collision_distance": False,
        "overview_callout_uses_collision_distance": False,
        "zoom_callout_uses_collision_distance": False,
        "zoom_projection_mode": "rest_tangent_plane_shape_aligned",
        "zoom_orientation_mode": "contact_3d_shape_aligned_to_rest_tangent_plane",
        "zoom_alignment_fit_labels": alignment_fit_labels,
        "zoom_alignment_fit_nodes": [int(node) for node in alignment_fit_nodes.tolist()],
        "zoom_alignment_excluded_labels": alignment_excluded_labels,
        "zoom_alignment_excluded_nodes": [
            int(node) for node in alignment_excluded_nodes.tolist()
        ],
        "zoom_tangent_plane_scale_px_per_world_unit": float(zoom_basis["scale"]),
        "zoom_tangent_plane_origin_world": [
            float(value) for value in zoom_basis["origin"].tolist()
        ],
        "zoom_tangent_plane_x_axis_world": [
            float(value) for value in zoom_basis["x_axis"].tolist()
        ],
        "zoom_tangent_plane_y_axis_world": [
            float(value) for value in zoom_basis["y_axis"].tolist()
        ],
        "zoom_tangent_plane_normal_world": [
            float(value) for value in zoom_basis["normal"].tolist()
        ],
        "node_labels_rendered": False,
        "zoom_label_layout_mode": "not_rendered",
        "zoom_label_offsets_px": None,
        "contact_zoom_alignment_rotation_deg": float(
            contact_zoom_alignment_3d["rotation_angle_deg"]
        ),
        "contact_zoom_alignment_3d_rotation_deg": float(
            contact_zoom_alignment_3d["rotation_angle_deg"]
        ),
        "contact_zoom_alignment_3d_residual_rms_world": float(
            contact_zoom_alignment_3d["residual_rms"]
        ),
        "visual_pad_px": float(args.visual_pad_px),
        "visual_highlight_radius_px": {
            "contact": float(contact_zoom_circle[1]),
            "rest": float(rest_zoom_circle[1]),
        },
        "red_circle_blue_radius_fraction_target": float(
            ZOOM_RED_RADIUS_FRACTION_TARGET
        ),
        "selected_nodes": [int(item) for item in selected_nodes_array.tolist()],
        "node_label_map": {
            label: int(node)
            for node, label in sorted(node_labels.items(), key=lambda item: item[1])
        },
        "selected_rest_edges": selected_edges.astype(int).tolist(),
        "local_nodes": [int(item) for item in selected_nodes_array.tolist()],
        "local_edges": selected_edges.astype(int).tolist(),
        "spring_lengths": selection.get("spring_lengths", []),
        "spring_length_summary": selection.get("spring_length_summary", {}),
        "selection": selection,
        "crop_size_px": float(args.crop_size_px),
        "zoom_crop_size_px": {
            "contact": float(contact_zoom_crop_size_px),
            "rest": float(rest_zoom_crop_size_px),
        },
        "zoom_context_crop_policy": {
            "min_context_nodes": 3,
            "min_context_edges": 2,
            "max_crop_size_px": 16.5,
            "contact_satisfied": bool(
                (
                    contact_zoom_context_render_stats["outside_context_edges_rendered"]
                    + contact_zoom_context_render_stats["selected_boundary_edges_rendered"]
                )
                >= 2
                and (
                    contact_zoom_context_render_stats["outside_context_nodes_available"]
                    - contact_zoom_context_render_stats["isolated_context_nodes_dropped"]
                )
                >= 3
            ),
            "rest_satisfied": bool(
                (
                    rest_zoom_context_render_stats["outside_context_edges_rendered"]
                    + rest_zoom_context_render_stats["selected_boundary_edges_rendered"]
                )
                >= 2
                and (
                    rest_zoom_context_render_stats["outside_context_nodes_available"]
                    - rest_zoom_context_render_stats["isolated_context_nodes_dropped"]
                )
                >= 3
            ),
        },
        "contact_gaussian_points_rendered": 0,
        "rest_gaussian_points_rendered": 0,
        "contact_zoom_gaussian_points_available": int(contact_gaussian_crop_xy.shape[0]),
        "rest_zoom_gaussian_points_available": int(rest_gaussian_crop_xy.shape[0]),
        "zoom_gaussian_kernel_ellipses_rendered": False,
        "zoom_gaussian_kernel_mode": "not_rendered",
        "zoom_gaussian_kernel_legacy_fallback_used": bool(
            not true_covariance_available
        ),
        "zoom_gaussian_kernel_sigma_level": float(
            ZOOM_GAUSSIAN_KERNEL_SIGMA_LEVEL
        ),
        "zoom_gaussian_kernel_visible_inside_red_circle": False,
        "zoom_gaussian_kernel_color_source": "not_rendered",
        "zoom_gaussian_kernel_ellipses_note": "Gaussian kernels are not rendered in the circular zoom.",
        "zoom_gaussian_kernel_max_ellipses": int(
            ZOOM_GAUSSIAN_KERNEL_MAX_ELLIPSES
        ),
        "zoom_gaussian_kernel_width_px": float(ZOOM_GAUSSIAN_KERNEL_WIDTH_PX),
        "zoom_gaussian_kernel_height_px": float(ZOOM_GAUSSIAN_KERNEL_HEIGHT_PX),
        "zoom_gaussian_kernel_face_alpha": float(
            ZOOM_GAUSSIAN_KERNEL_FACE_ALPHA
        ),
        "zoom_gaussian_kernel_edge_alpha": float(
            ZOOM_GAUSSIAN_KERNEL_EDGE_ALPHA
        ),
        "contact_zoom_gaussian_kernel_ellipses_rendered": 0,
        "rest_zoom_gaussian_kernel_ellipses_rendered": 0,
        "contact_overview_gaussian_points_rendered": int(contact_overview_xy.shape[0]),
        "rest_overview_gaussian_points_rendered": int(rest_overview_xy.shape[0]),
        "graph_context_rendering": {
            "contact_overview_nodes_rendered": int(
                np.count_nonzero(
                    projected_node_mask(contact_xy, contact_overview_xlim, contact_overview_ylim)
                )
            ),
            "contact_overview_edges_rendered": int(contact_overview_edges.shape[0]),
            "contact_overview_edges_available": int(contact_overview_edges_available.shape[0]),
            "contact_overview_focus_edges_rendered": 0,
            "rest_overview_nodes_rendered": int(
                np.count_nonzero(
                    projected_node_mask(rest_xy, rest_overview_xlim, rest_overview_ylim)
                )
            ),
            "rest_overview_edges_rendered": int(rest_overview_edges.shape[0]),
            "rest_overview_edges_available": int(rest_overview_edges_available.shape[0]),
            "rest_overview_focus_edges_rendered": 0,
            "contact_zoom_context_nodes_available": int(
                contact_zoom_context_nodes.shape[0]
            ),
            "contact_zoom_context_edges_available": int(
                contact_zoom_context_edges.shape[0]
            ),
            "contact_zoom_context_nodes_rendered": int(
                contact_zoom_context_render_stats["outside_context_nodes_available"]
                - contact_zoom_context_render_stats["isolated_context_nodes_dropped"]
            ),
            "contact_zoom_context_edges_rendered": int(
                contact_zoom_context_render_stats["outside_context_edges_rendered"]
                + contact_zoom_context_render_stats["selected_boundary_edges_rendered"]
            ),
            "contact_zoom_outside_context_edges_rendered": int(
                contact_zoom_context_render_stats["outside_context_edges_rendered"]
            ),
            "contact_zoom_selected_boundary_edges_rendered": int(
                contact_zoom_context_render_stats["selected_boundary_edges_rendered"]
            ),
            "contact_zoom_boundary_context_edges_dropped": int(
                contact_zoom_context_render_stats["boundary_context_edges_dropped"]
            ),
            "contact_zoom_nonselected_boundary_edges_dropped": int(
                contact_zoom_context_render_stats["nonselected_boundary_edges_dropped"]
            ),
            "contact_zoom_crossing_context_edges_dropped": int(
                contact_zoom_context_render_stats["crossing_context_edges_dropped"]
            ),
            "contact_zoom_isolated_context_nodes_dropped": int(
                contact_zoom_context_render_stats["isolated_context_nodes_dropped"]
            ),
            "rest_zoom_context_nodes_available": int(rest_zoom_context_nodes.shape[0]),
            "rest_zoom_context_edges_available": int(rest_zoom_context_edges.shape[0]),
            "rest_zoom_context_nodes_rendered": int(
                rest_zoom_context_render_stats["outside_context_nodes_available"]
                - rest_zoom_context_render_stats["isolated_context_nodes_dropped"]
            ),
            "rest_zoom_context_edges_rendered": int(
                rest_zoom_context_render_stats["outside_context_edges_rendered"]
                + rest_zoom_context_render_stats["selected_boundary_edges_rendered"]
            ),
            "rest_zoom_outside_context_edges_rendered": int(
                rest_zoom_context_render_stats["outside_context_edges_rendered"]
            ),
            "rest_zoom_selected_boundary_edges_rendered": int(
                rest_zoom_context_render_stats["selected_boundary_edges_rendered"]
            ),
            "rest_zoom_boundary_context_edges_dropped": int(
                rest_zoom_context_render_stats["boundary_context_edges_dropped"]
            ),
            "rest_zoom_nonselected_boundary_edges_dropped": int(
                rest_zoom_context_render_stats["nonselected_boundary_edges_dropped"]
            ),
            "rest_zoom_crossing_context_edges_dropped": int(
                rest_zoom_context_render_stats["crossing_context_edges_dropped"]
            ),
            "rest_zoom_isolated_context_nodes_dropped": int(
                rest_zoom_context_render_stats["isolated_context_nodes_dropped"]
            ),
        },
        "outputs": {key: relative_to_root(value, root) for key, value in outputs.items()},
        "notes": [
            "The rest spring graph is built once from frame-0 mass nodes.",
            "The frame-41 panel reuses the selected frame-0 rest adjacency on simulated frame-41 mass-node positions.",
            "The frame-41 zoom inset is 3D-aligned to the frame-0 selected patch using the outer shape nodes and excluding M0, then projected into the rest tangent plane; full-view panels remain raw camera projections.",
            "The overview circles and leader lines are camera-projected visual callouts; the zoom circles are visual callouts in the rest tangent-plane view.",
            "The circular zoom insets omit Gaussian dots so the spring-mass graph connectivity remains readable.",
            "The visual callouts are not the collision pruning radius.",
            "Controller nodes and controller-object springs are hidden.",
        ],
    }
    if args.layout == "full_gaussian_highlight":
        metadata.update(
            {
                "layout": "full_gaussian_highlight",
                "visual_circle_uses_collision_distance": False,
                "overview_callout_uses_collision_distance": False,
                "zoom_callout_uses_collision_distance": False,
                "zoom_projection_mode": "raw_camera_projection",
                "zoom_orientation_mode": "raw_camera_projection_no_alignment",
                "zoom_alignment_fit_labels": [],
                "zoom_alignment_fit_nodes": [],
                "zoom_alignment_excluded_labels": [],
                "zoom_alignment_excluded_nodes": [],
                "zoom_tangent_plane_scale_px_per_world_unit": None,
                "zoom_tangent_plane_origin_world": None,
                "zoom_tangent_plane_x_axis_world": None,
                "zoom_tangent_plane_y_axis_world": None,
                "zoom_tangent_plane_normal_world": None,
                "contact_zoom_alignment_rotation_deg": None,
                "contact_zoom_alignment_3d_rotation_deg": None,
                "contact_zoom_alignment_3d_residual_rms_world": None,
                "visual_highlight_radius_px": None,
                "highlight_marker_size_pt2": 92,
                "highlight_edge_linewidth_pt": 3.2,
                "highlight_edge_halo_linewidth_pt": 4.2,
                "red_circle_blue_radius_fraction_target": None,
                "crop_size_px": None,
                "zoom_crop_size_px": {
                    "contact": float(args.render_zoom_crop_size_px),
                    "rest": float(args.render_zoom_crop_size_px),
                },
                "zoom_context_crop_policy": None,
                "contact_gaussian_points_rendered": int(contact_overview_xy.shape[0]),
                "rest_gaussian_points_rendered": int(rest_overview_xy.shape[0]),
                "contact_zoom_gaussian_points_available": 0,
                "rest_zoom_gaussian_points_available": 0,
                "zoom_gaussian_kernel_ellipses_rendered": False,
                "zoom_gaussian_kernel_mode": "not_used",
                "zoom_gaussian_kernel_legacy_fallback_used": False,
                "zoom_gaussian_kernel_visible_inside_red_circle": False,
                "zoom_gaussian_kernel_color_source": "not_used",
                "zoom_gaussian_kernel_ellipses_note": "No zoom inset or Gaussian kernel ellipses are rendered in the full-Gaussian highlight layout.",
                "contact_zoom_gaussian_kernel_ellipses_rendered": 0,
                "rest_zoom_gaussian_kernel_ellipses_rendered": 0,
                "notes": [
                    "The rest spring graph is built once from frame-0 mass nodes.",
                    "The frame-41 panel reuses the selected frame-0 rest adjacency on simulated frame-41 mass-node positions.",
                    "The full projected Gaussian object is shown directly in each panel with no circular zoom inset, blue focus circle, leader lines, or callout text.",
                    "Selected spring edges and mass nodes are highlighted directly on the full-object camera projection.",
                    "Controller nodes and controller-object springs are hidden.",
                ],
            }
        )
        metadata["graph_context_rendering"].update(
            {
                "contact_overview_focus_edges_rendered": int(selected_edges.shape[0]),
                "rest_overview_focus_edges_rendered": int(selected_edges.shape[0]),
                "contact_full_gaussian_nodes_rendered": int(
                    np.count_nonzero(
                        projected_node_mask(
                            contact_xy,
                            contact_overview_xlim,
                            contact_overview_ylim,
                        )
                    )
                ),
                "rest_full_gaussian_nodes_rendered": int(
                    np.count_nonzero(
                        projected_node_mask(
                            rest_xy,
                            rest_overview_xlim,
                            rest_overview_ylim,
                        )
                    )
                ),
                "contact_full_gaussian_edges_rendered": int(contact_overview_edges.shape[0]),
                "rest_full_gaussian_edges_rendered": int(rest_overview_edges.shape[0]),
                "selected_edges_rendered": int(selected_edges.shape[0]),
                "contact_zoom_context_nodes_available": 0,
                "contact_zoom_context_edges_available": 0,
                "contact_zoom_context_nodes_rendered": 0,
                "contact_zoom_context_edges_rendered": 0,
                "contact_zoom_outside_context_edges_rendered": 0,
                "contact_zoom_selected_boundary_edges_rendered": 0,
                "contact_zoom_boundary_context_edges_dropped": 0,
                "contact_zoom_nonselected_boundary_edges_dropped": 0,
                "contact_zoom_crossing_context_edges_dropped": 0,
                "contact_zoom_isolated_context_nodes_dropped": 0,
                "rest_zoom_context_nodes_available": 0,
                "rest_zoom_context_edges_available": 0,
                "rest_zoom_context_nodes_rendered": 0,
                "rest_zoom_context_edges_rendered": 0,
                "rest_zoom_outside_context_edges_rendered": 0,
                "rest_zoom_selected_boundary_edges_rendered": 0,
                "rest_zoom_boundary_context_edges_dropped": 0,
                "rest_zoom_nonselected_boundary_edges_dropped": 0,
                "rest_zoom_crossing_context_edges_dropped": 0,
                "rest_zoom_isolated_context_nodes_dropped": 0,
            }
        )
    if args.layout == "rest_map_anchor":
        rest_anchor_context_nodes_available = (
            np.unique(rest_anchor_edges.reshape(-1))
            if rest_anchor_edges.size
            else np.empty(0, dtype=np.int64)
        )
        contact_anchor_context_nodes_available = (
            np.unique(contact_anchor_edges.reshape(-1))
            if contact_anchor_edges.size
            else np.empty(0, dtype=np.int64)
        )
        anchor_rest_map_distance_records = [
            {
                "node": int(node),
                "rest_distance": float(anchor_rest_map_distances[int(node)]),
            }
            for node in anchor_rest_map_nodes.tolist()
        ]
        metadata.update(
            {
                "layout": "rest_map_anchor",
                "render_coordinate_mode": "raw_camera_projection",
                "main_view": "rendered_zoom_crop",
                "context_inset": "full_boba_render_with_anchor_marker",
                "main_background_source": "boba_quality_render_png",
                "main_background_crop_mode": "raw_camera_anchor_crop",
                "main_background_image_paths": {
                    "contact": relative_to_root(contact_inset_render_path, root),
                    "rest": relative_to_root(rest_inset_render_path, root),
                },
                "main_background_image_size_px": {
                    "contact": [int(value) for value in contact_inset_render_size],
                    "rest": [int(value) for value in rest_inset_render_size],
                },
                "full_graph_overlay_rendered": True,
                "zoom_background_source": "boba_quality_render_png_alpha_crop",
                "zoom_crop_insets_rendered": True,
                "render_zoom_crop_size_px": float(args.render_zoom_crop_size_px),
                "render_zoom_crop_limits_px": {
                    "contact": limits_metadata(
                        contact_render_zoom_xlim,
                        contact_render_zoom_ylim,
                    ),
                    "rest": limits_metadata(
                        rest_render_zoom_xlim,
                        rest_render_zoom_ylim,
                    ),
                },
                "anchor_node": int(anchor_node),
                "anchor_selection_reason": (
                    f"Node {anchor_node} is near the current right-leg selected "
                    "patch and was chosen because the selected pruning "
                    "region has very small rest/contact relative motion in "
                    "anchor-local selection metrics."
                ),
                "anchor_crop_size_px": float(args.anchor_crop_size_px),
                "anchor_crop_center_px": {
                    "contact": [float(value) for value in contact_anchor_center.tolist()],
                    "rest": [float(value) for value in rest_anchor_center.tolist()],
                },
                "anchor_crop_xlim": {
                    "contact": [float(value) for value in contact_anchor_xlim],
                    "rest": [float(value) for value in rest_anchor_xlim],
                },
                "anchor_crop_ylim": {
                    "contact": [float(value) for value in contact_anchor_ylim],
                    "rest": [float(value) for value in rest_anchor_ylim],
                },
                "anchor_local_frame": {
                    "origin": "anchor_node_world_position",
                    "projection_basis": "camera_x_y_axes_at_rest",
                    "scale_px_per_world_unit": float(
                        anchor_local_scale_px_per_world
                    ),
                    "contact_3d_alignment_residual_rms_world": float(
                        contact_anchor_3d_alignment["residual_rms"]
                    ),
                    "contact_3d_alignment_residual_max_world": float(
                        contact_anchor_3d_alignment["residual_max"]
                    ),
                    "contact_3d_alignment_residual_rms_normalized": float(
                        contact_anchor_3d_alignment["residual_rms_normalized"]
                    ),
                    "contact_projected_alignment_residual_rms_px": float(
                        anchor_local_alignment_residual_rms_px
                    ),
                    "contact_projected_alignment_residual_max_px": float(
                        anchor_local_alignment_residual_max_px
                    ),
                    "contact_projected_alignment_residual_rms_normalized": float(
                        anchor_local_alignment_residual_rms_normalized
                    ),
                },
                "anchor_local_xlim": {
                    "contact": [float(value) for value in contact_anchor_local_xlim],
                    "rest": [float(value) for value in rest_anchor_local_xlim],
                },
                "anchor_local_ylim": {
                    "contact": [float(value) for value in contact_anchor_local_ylim],
                    "rest": [float(value) for value in rest_anchor_local_ylim],
                },
                "selection_radius_mode": "fixed_anchor_local_radius_px",
                "visual_radius_mode": "raw_camera_selected_node_envelope",
                "structural_radius_px": float(anchor_structural_radius_px),
                "visual_radius_world": None,
                "visual_radius_definition": (
                    "raw camera-space red circle centered on the anchor and "
                    "expanded to enclose the selected pruning-region nodes"
                ),
                "visual_radius_semantics": "rest_map_pruning_region_visualization",
                "raw_pruning_circle_px": {
                    "contact": circle_metadata(contact_raw_pruning_circle),
                    "rest": circle_metadata(rest_raw_pruning_circle),
                },
                "visual_radius_endpoint_quantile": None,
                "visual_radius_padding_px": 0.0,
                "visual_radius_px": {
                    "contact": float(contact_raw_pruning_circle[1]),
                    "rest": float(rest_raw_pruning_circle[1]),
                },
                "visual_radius_endpoint_coverage": {
                    "contact": float(
                        contact_anchor_visual_radius_endpoint_coverage
                    ),
                    "rest": float(rest_anchor_visual_radius_endpoint_coverage),
                },
                "visual_highlight_radius_px": {
                    "contact": float(contact_raw_pruning_circle[1]),
                    "rest": float(rest_raw_pruning_circle[1]),
                },
                "incident_spring_endpoint_distance_px": {
                    "contact_max_after_alignment": float(
                        np.max(contact_anchor_endpoint_distances_px)
                    )
                    if contact_anchor_endpoint_distances_px.size
                    else 0.0,
                    "contact_median_after_alignment": float(
                        np.median(contact_anchor_endpoint_distances_px)
                    )
                    if contact_anchor_endpoint_distances_px.size
                    else 0.0,
                    "rest_max": float(np.max(rest_anchor_endpoint_distances_px))
                    if rest_anchor_endpoint_distances_px.size
                    else 0.0,
                    "rest_median": float(np.median(rest_anchor_endpoint_distances_px))
                    if rest_anchor_endpoint_distances_px.size
                    else 0.0,
                },
                "collision_dist": float(collision_dist),
                "rest_map_radius": float(rest_map_radius),
                "rest_map_exact_radius": float(rest_map_radius),
                "rest_map_radius_definition": "collision_dist * 5.0",
                "anchor_rest_map_skipped_neighbor_count": int(
                    anchor_rest_map_nodes.shape[0]
                ),
                "anchor_rest_map_skipped_nodes": [
                    int(node) for node in anchor_rest_map_nodes.tolist()
                ],
                "anchor_rest_map_skipped_neighbor_distances": (
                    anchor_rest_map_distance_records
                ),
                "rest_anchor_crop_skipped_neighbor_count": int(
                    rest_anchor_rest_map_nodes.shape[0]
                ),
                "contact_anchor_crop_skipped_neighbor_count": int(
                    contact_anchor_rest_map_nodes.shape[0]
                ),
                "anchor_incident_spring_count": int(anchor_spring_edges.shape[0]),
                "anchor_incident_spring_nodes": [
                    int(node) for node in anchor_spring_nodes.tolist()
                ],
                "anchor_incident_spring_edges": (
                    anchor_spring_edges.astype(int).tolist()
                ),
                "selection_mode": "rest_radius_structural_region",
                "selection_basis": "same_node_ids_from_rest_frame",
                "structural_region_node_count": int(anchor_structural_nodes.shape[0]),
                "structural_region_edge_count": int(anchor_structural_edges.shape[0]),
                "structural_region_selected_node_coverage": {
                    "contact": float(contact_anchor_structural_node_coverage),
                    "rest": float(rest_anchor_structural_node_coverage),
                },
                "structural_region_projected_node_residual_px": (
                    structural_node_residual_summary
                ),
                "structural_region_nodes": [
                    int(node) for node in anchor_structural_nodes.tolist()
                ],
                "structural_region_edges": anchor_structural_edges.astype(int).tolist(),
                "selected_nodes": [
                    int(node) for node in anchor_structural_nodes.tolist()
                ],
                "node_label_map": {},
                "selected_rest_edges": anchor_structural_edges.astype(int).tolist(),
                "local_nodes": [
                    int(node) for node in anchor_structural_nodes.tolist()
                ],
                "local_edges": anchor_structural_edges.astype(int).tolist(),
                "spring_lengths": anchor_structural_spring_lengths,
                "spring_length_summary": anchor_structural_spring_length_summary,
                "anchor_incident_spring_lengths": anchor_spring_lengths,
                "anchor_incident_spring_length_summary": anchor_spring_length_summary,
                "selection": {
                    "mode": "rest_radius_structural_region",
                    "basis": "same_node_ids_from_rest_frame",
                    "anchor_node": int(anchor_node),
                    "selected_nodes": [
                        int(node) for node in anchor_structural_nodes.tolist()
                    ],
                    "selected_rest_edges": anchor_structural_edges.astype(int).tolist(),
                    "spring_length_summary": anchor_structural_spring_length_summary,
                    "spring_lengths": anchor_structural_spring_lengths,
                    "rest_map_skipped_neighbor_count": int(
                        anchor_rest_map_nodes.shape[0]
                    ),
                },
                "uniform_graph_style": True,
                "selected_style_difference": "alpha_only_with_same_element_colors",
                "spring_color_role": "generated_springs_gold",
                "mass_node_color_role": "generated_mass_nodes_green",
                "legend_placement": "center_between_rest_and_simulation_panels",
                "legend_scale": "large",
                "legend_entries": [
                    "Real Gaussians",
                    "Mass nodes",
                    "Springs",
                    "Pruning region",
                ],
                "export_padding_inches": 0.012,
                "bottom_gap_reduction": "tight_combined_bbox",
                "mass_node_marker_size_pt2": 18.0,
                "spring_linewidth_pt": 0.90,
                "structural_neighbor_circle_linewidth_pt": 1.25,
                "pruning_region_circle_linewidth_pt": 1.25,
                "context_graph_style": "faint_full_raw_camera_generated_graph",
                "context_graph_scope": (
                    "all generated spring edges with both endpoints projected "
                    "inside the raw camera frame"
                ),
                "context_graph_alpha": 0.14,
                "context_graph_edge_alpha": 0.14,
                "context_graph_node_alpha": 0.32,
                "selected_graph_alpha": 0.95,
                "gaussian_alpha": 0.30,
                "visual_circle_uses_collision_distance": False,
                "overview_callout_uses_collision_distance": False,
                "zoom_callout_uses_collision_distance": False,
                "zoom_projection_mode": "not_used",
                "zoom_orientation_mode": "not_used",
                "zoom_alignment_fit_labels": [],
                "zoom_alignment_fit_nodes": [],
                "zoom_alignment_excluded_labels": [],
                "zoom_alignment_excluded_nodes": [],
                "zoom_tangent_plane_scale_px_per_world_unit": None,
                "zoom_tangent_plane_origin_world": None,
                "zoom_tangent_plane_x_axis_world": None,
                "zoom_tangent_plane_y_axis_world": None,
                "zoom_tangent_plane_normal_world": None,
                "contact_zoom_alignment_rotation_deg": None,
                "contact_zoom_alignment_3d_rotation_deg": None,
                "contact_zoom_alignment_3d_residual_rms_world": None,
                "red_circle_blue_radius_fraction_target": None,
                "crop_size_px": None,
                "zoom_crop_size_px": None,
                "zoom_context_crop_policy": None,
                "contact_gaussian_points_rendered": 0,
                "rest_gaussian_points_rendered": 0,
                "contact_anchor_gaussian_points_rendered": int(
                    contact_anchor_gaussian_xy.shape[0]
                ),
                "rest_anchor_gaussian_points_rendered": int(
                    rest_anchor_gaussian_xy.shape[0]
                ),
                "full_object_insets_rendered": True,
                "full_object_inset_source": "boba_quality_render_png",
                "full_object_inset_background": "transparent_outside_rendered_sloth",
                "full_object_inset_crop_mode": "full_render_context",
                "full_object_inset_position": "lower_left_context",
                "inset_anchor_marker": "red_anchor_location_in_full_context_inset",
                "full_object_inset_image_paths": {
                    "contact": relative_to_root(contact_inset_render_path, root),
                    "rest": relative_to_root(rest_inset_render_path, root),
                },
                "full_object_inset_image_size_px": {
                    "contact": [int(value) for value in contact_inset_render_size],
                    "rest": [int(value) for value in rest_inset_render_size],
                },
                "full_object_inset_alpha_bbox_px": {
                    "contact": contact_inset_alpha_bbox,
                    "rest": rest_inset_alpha_bbox,
                },
                "full_object_inset_expected_image_size_px": [
                    int(wh[0]),
                    int(wh[1]),
                ],
                "full_object_inset_gaussian_points_rendered": None,
                "full_object_inset_anchor_xy_px": {
                    "contact": [float(value) for value in contact_xy[anchor_node].tolist()],
                    "rest": [float(value) for value in rest_xy[anchor_node].tolist()],
                },
                "contact_overview_gaussian_points_rendered": int(
                    contact_overview_xy.shape[0]
                ),
                "rest_overview_gaussian_points_rendered": int(
                    rest_overview_xy.shape[0]
                ),
                "contact_zoom_gaussian_points_available": 0,
                "rest_zoom_gaussian_points_available": 0,
                "zoom_gaussian_kernel_ellipses_rendered": False,
                "zoom_gaussian_kernel_mode": "not_used",
                "zoom_gaussian_kernel_legacy_fallback_used": False,
                "zoom_gaussian_kernel_visible_inside_red_circle": False,
                "zoom_gaussian_kernel_color_source": "not_used",
                "zoom_gaussian_kernel_ellipses_note": "No zoom inset or Gaussian kernel ellipses are rendered in the rest-map anchor layout.",
                "contact_zoom_gaussian_kernel_ellipses_rendered": 0,
                "rest_zoom_gaussian_kernel_ellipses_rendered": 0,
                "notes": [
                    "The rest spring graph is built once from frame-0 mass nodes.",
                    "The frame-41 panel reuses the selected frame-0 rest adjacency on simulated frame-41 mass-node positions.",
                    "The rest-map anchor panels use a raw camera crop from the Boba-rendered quality PNG frames as the main view.",
                    "Each main view includes a small full-object Boba render inset with a red anchor marker for global pose context.",
                    "The red circle is a raw camera-space visual pruning region enclosing the selected pruning nodes, not a literal collision-distance radius.",
                    "The simulator collision distance is recorded as collision_dist, and the exact rest-map skip table radius is collision_dist * 5.0, recorded as rest_map_exact_radius.",
                    "The highlighted pruning region contains every rest-frame generated mass node inside the red circle and every generated spring induced by those selected nodes.",
                    "Faint graph context is drawn across the full Boba-rendered sloth so the highlighted pruning region reads as a subset of the generated graph.",
                    "Dense rest-map skipped nodes are intentionally hidden; the red pruning region communicates the structural-neighbor pruning intuition.",
                    "The figure overlays the generated spring-mass graph on real Boba-rendered Gaussian frames; no separate collision-edge records are exported or drawn.",
                    "No blue focus circle, leader lines, or dense callout text is rendered.",
                    "Controller nodes and controller-object springs are hidden.",
                ],
            }
        )
        metadata["graph_context_rendering"].update(
            {
                "contact_overview_nodes_rendered": int(
                    np.count_nonzero(projected_node_mask(contact_xy, image_xlim, image_ylim))
                ),
                "contact_overview_edges_rendered": int(
                    contact_overview_edges_available.shape[0]
                ),
                "contact_overview_edges_available": int(
                    contact_overview_edges_available.shape[0]
                ),
                "contact_overview_focus_edges_rendered": int(
                    anchor_structural_edges.shape[0]
                ),
                "rest_overview_nodes_rendered": int(
                    np.count_nonzero(projected_node_mask(rest_xy, image_xlim, image_ylim))
                ),
                "rest_overview_edges_rendered": int(
                    rest_overview_edges_available.shape[0]
                ),
                "rest_overview_edges_available": int(rest_overview_edges_available.shape[0]),
                "rest_overview_focus_edges_rendered": int(
                    anchor_structural_edges.shape[0]
                ),
                "contact_raw_full_graph_nodes_rendered": int(
                    np.count_nonzero(projected_node_mask(contact_xy, image_xlim, image_ylim))
                ),
                "contact_raw_full_graph_edges_rendered": int(
                    contact_overview_edges_available.shape[0]
                ),
                "rest_raw_full_graph_nodes_rendered": int(
                    np.count_nonzero(projected_node_mask(rest_xy, image_xlim, image_ylim))
                ),
                "rest_raw_full_graph_edges_rendered": int(
                    rest_overview_edges_available.shape[0]
                ),
                "contact_render_zoom_context_edges_rendered": int(
                    contact_render_zoom_edges.shape[0]
                ),
                "rest_render_zoom_context_edges_rendered": int(
                    rest_render_zoom_edges.shape[0]
                ),
                "selected_pruning_edges_rendered": int(
                    anchor_structural_edges.shape[0]
                ),
                "contact_anchor_nodes_rendered": int(
                    contact_anchor_context_stats["context_nodes_rendered"]
                    + contact_anchor_structural_nodes.shape[0]
                ),
                "contact_anchor_edges_available": int(contact_anchor_edges.shape[0]),
                "contact_anchor_edges_rendered": int(
                    contact_anchor_context_stats["context_edges_rendered"]
                ),
                "contact_anchor_context_nodes_available": int(
                    contact_anchor_context_nodes_available.shape[0]
                ),
                "contact_anchor_context_nodes_rendered": int(
                    contact_anchor_context_stats["context_nodes_rendered"]
                ),
                "contact_anchor_context_edges_rendered": int(
                    contact_anchor_context_stats["context_edges_rendered"]
                ),
                "contact_anchor_inside_context_edges_dropped": int(
                    contact_anchor_context_stats["inside_context_edges_dropped"]
                ),
                "contact_anchor_boundary_context_edges_dropped": int(
                    contact_anchor_context_stats["boundary_context_edges_dropped"]
                ),
                "contact_anchor_selected_touch_context_edges_dropped": int(
                    contact_anchor_context_stats["selected_touch_context_edges_dropped"]
                ),
                "contact_anchor_inside_context_nodes_dropped": int(
                    contact_anchor_context_stats["inside_context_nodes_dropped"]
                ),
                "contact_anchor_spring_nodes_rendered": int(
                    contact_anchor_structural_nodes.shape[0]
                ),
                "contact_anchor_spring_edges_rendered": int(
                    contact_anchor_structural_edges.shape[0]
                ),
                "contact_anchor_structural_nodes_rendered": int(
                    contact_anchor_structural_nodes.shape[0]
                ),
                "contact_anchor_structural_edges_rendered": int(
                    contact_anchor_structural_edges.shape[0]
                ),
                "contact_anchor_rest_map_skipped_nodes_rendered": int(
                    0
                ),
                "contact_anchor_rest_map_skipped_nodes_available": int(
                    contact_anchor_rest_map_nodes.shape[0]
                ),
                "rest_anchor_nodes_rendered": int(
                    rest_anchor_context_stats["context_nodes_rendered"]
                    + rest_anchor_structural_nodes.shape[0]
                ),
                "rest_anchor_edges_available": int(rest_anchor_edges.shape[0]),
                "rest_anchor_edges_rendered": int(
                    rest_anchor_context_stats["context_edges_rendered"]
                ),
                "rest_anchor_context_nodes_available": int(
                    rest_anchor_context_nodes_available.shape[0]
                ),
                "rest_anchor_context_nodes_rendered": int(
                    rest_anchor_context_stats["context_nodes_rendered"]
                ),
                "rest_anchor_context_edges_rendered": int(
                    rest_anchor_context_stats["context_edges_rendered"]
                ),
                "rest_anchor_inside_context_edges_dropped": int(
                    rest_anchor_context_stats["inside_context_edges_dropped"]
                ),
                "rest_anchor_boundary_context_edges_dropped": int(
                    rest_anchor_context_stats["boundary_context_edges_dropped"]
                ),
                "rest_anchor_selected_touch_context_edges_dropped": int(
                    rest_anchor_context_stats["selected_touch_context_edges_dropped"]
                ),
                "rest_anchor_inside_context_nodes_dropped": int(
                    rest_anchor_context_stats["inside_context_nodes_dropped"]
                ),
                "rest_anchor_spring_nodes_rendered": int(
                    rest_anchor_structural_nodes.shape[0]
                ),
                "rest_anchor_spring_edges_rendered": int(
                    rest_anchor_structural_edges.shape[0]
                ),
                "rest_anchor_structural_nodes_rendered": int(
                    rest_anchor_structural_nodes.shape[0]
                ),
                "rest_anchor_structural_edges_rendered": int(
                    rest_anchor_structural_edges.shape[0]
                ),
                "rest_anchor_rest_map_skipped_nodes_rendered": int(
                    0
                ),
                "rest_anchor_rest_map_skipped_nodes_available": int(
                    rest_anchor_rest_map_nodes.shape[0]
                ),
                "contact_zoom_context_nodes_available": 0,
                "contact_zoom_context_edges_available": 0,
                "contact_zoom_context_nodes_rendered": 0,
                "contact_zoom_context_edges_rendered": 0,
                "contact_zoom_outside_context_edges_rendered": 0,
                "contact_zoom_selected_boundary_edges_rendered": 0,
                "contact_zoom_boundary_context_edges_dropped": 0,
                "contact_zoom_nonselected_boundary_edges_dropped": 0,
                "contact_zoom_crossing_context_edges_dropped": 0,
                "contact_zoom_isolated_context_nodes_dropped": 0,
                "rest_zoom_context_nodes_available": 0,
                "rest_zoom_context_edges_available": 0,
                "rest_zoom_context_nodes_rendered": 0,
                "rest_zoom_context_edges_rendered": 0,
                "rest_zoom_outside_context_edges_rendered": 0,
                "rest_zoom_selected_boundary_edges_rendered": 0,
                "rest_zoom_boundary_context_edges_dropped": 0,
                "rest_zoom_nonselected_boundary_edges_dropped": 0,
                "rest_zoom_crossing_context_edges_dropped": 0,
                "rest_zoom_isolated_context_nodes_dropped": 0,
            }
        )
    if args.layout == "leg_focus_highlight":
        metadata.update(
            {
                "layout": "leg_focus_highlight",
                "leg_crop_size_px": float(args.leg_crop_size_px),
                "leg_crop_center_px": {
                    "contact": [float(value) for value in contact_leg_center.tolist()],
                    "rest": [float(value) for value in rest_leg_center.tolist()],
                },
                "leg_crop_xlim": {
                    "contact": [float(value) for value in contact_leg_xlim],
                    "rest": [float(value) for value in rest_leg_xlim],
                },
                "leg_crop_ylim": {
                    "contact": [float(value) for value in contact_leg_ylim],
                    "rest": [float(value) for value in rest_leg_ylim],
                },
                "seed_selected_nodes": [
                    int(item) for item in selected_nodes_array.tolist()
                ],
                "seed_selected_rest_edges": selected_edges.astype(int).tolist(),
                "seed_node_label_map": {
                    label: int(node)
                    for node, label in sorted(node_labels.items(), key=lambda item: item[1])
                },
                "selected_nodes": [int(item) for item in leg_selected_nodes.tolist()],
                "node_label_map": {},
                "selected_rest_edges": leg_selected_edges.astype(int).tolist(),
                "local_nodes": [int(item) for item in leg_selected_nodes.tolist()],
                "local_edges": leg_selected_edges.astype(int).tolist(),
                "spring_lengths": leg_spring_lengths,
                "spring_length_summary": leg_spring_length_summary,
                "seed_selection": selection,
                "selection": {
                    "mode": "connected_visible_component",
                    "seed_nodes": [int(item) for item in selected_nodes_array.tolist()],
                    "selected_nodes": [
                        int(item) for item in leg_selected_nodes.tolist()
                    ],
                    "selected_rest_edges": leg_selected_edges.astype(int).tolist(),
                    "spring_length_summary": leg_spring_length_summary,
                    "spring_lengths": leg_spring_lengths,
                },
                "selected_patch_mode": "connected_visible_component",
                "selected_patch_max_nodes": int(args.selected_patch_max_nodes),
                "selected_patch_seed_nodes": [
                    int(item) for item in selected_nodes_array.tolist()
                ],
                "selected_patch_visible_in_both_frames": True,
                "uniform_graph_style": True,
                "selected_style_difference": "color_and_alpha_only",
                "mass_node_marker_size_pt2": 14.0,
                "spring_linewidth_pt": 0.72,
                "context_graph_alpha": 0.18,
                "selected_graph_alpha": 0.95,
                "gaussian_alpha": 0.34,
                "visual_circle_uses_collision_distance": False,
                "overview_callout_uses_collision_distance": False,
                "zoom_callout_uses_collision_distance": False,
                "zoom_projection_mode": "not_used",
                "zoom_orientation_mode": "not_used",
                "zoom_alignment_fit_labels": [],
                "zoom_alignment_fit_nodes": [],
                "zoom_alignment_excluded_labels": [],
                "zoom_alignment_excluded_nodes": [],
                "zoom_tangent_plane_scale_px_per_world_unit": None,
                "zoom_tangent_plane_origin_world": None,
                "zoom_tangent_plane_x_axis_world": None,
                "zoom_tangent_plane_y_axis_world": None,
                "zoom_tangent_plane_normal_world": None,
                "contact_zoom_alignment_rotation_deg": None,
                "contact_zoom_alignment_3d_rotation_deg": None,
                "contact_zoom_alignment_3d_residual_rms_world": None,
                "visual_highlight_radius_px": None,
                "red_circle_blue_radius_fraction_target": None,
                "crop_size_px": None,
                "zoom_crop_size_px": None,
                "zoom_context_crop_policy": None,
                "contact_gaussian_points_rendered": int(contact_leg_gaussian_xy.shape[0]),
                "rest_gaussian_points_rendered": int(rest_leg_gaussian_xy.shape[0]),
                "contact_leg_gaussian_points_rendered": int(
                    contact_leg_gaussian_xy.shape[0]
                ),
                "rest_leg_gaussian_points_rendered": int(rest_leg_gaussian_xy.shape[0]),
                "contact_overview_gaussian_points_rendered": 0,
                "rest_overview_gaussian_points_rendered": 0,
                "contact_zoom_gaussian_points_available": 0,
                "rest_zoom_gaussian_points_available": 0,
                "zoom_gaussian_kernel_ellipses_rendered": False,
                "zoom_gaussian_kernel_mode": "not_used",
                "zoom_gaussian_kernel_legacy_fallback_used": False,
                "zoom_gaussian_kernel_visible_inside_red_circle": False,
                "zoom_gaussian_kernel_color_source": "not_used",
                "zoom_gaussian_kernel_ellipses_note": "No zoom inset or Gaussian kernel ellipses are rendered in the leg-focus highlight layout.",
                "contact_zoom_gaussian_kernel_ellipses_rendered": 0,
                "rest_zoom_gaussian_kernel_ellipses_rendered": 0,
                "notes": [
                    "The rest spring graph is built once from frame-0 mass nodes.",
                    "The frame-41 panel reuses the selected frame-0 rest adjacency on simulated frame-41 mass-node positions.",
                    "Each panel shows a raw-camera leg crop centered on the seed selected spring patch.",
                    "The highlighted patch is expanded from the seed nodes through the generated real rest spring graph, constrained to nodes visible in both rest and contact crops.",
                    "Selected mass nodes and selected springs use the same sizes as neighboring graph elements and are distinguished by color and alpha only.",
                    "The figure overlays the generated spring-mass graph on real Gaussian projections; no separate collision-edge records are exported or drawn.",
                    "No circular zoom inset, blue focus circle, leader lines, or dense callout text is rendered.",
                    "Controller nodes and controller-object springs are hidden.",
                ],
            }
        )
        metadata["graph_context_rendering"].update(
            {
                "contact_overview_nodes_rendered": 0,
                "contact_overview_edges_rendered": 0,
                "contact_overview_edges_available": int(
                    contact_overview_edges_available.shape[0]
                ),
                "contact_overview_focus_edges_rendered": 0,
                "rest_overview_nodes_rendered": 0,
                "rest_overview_edges_rendered": 0,
                "rest_overview_edges_available": int(rest_overview_edges_available.shape[0]),
                "rest_overview_focus_edges_rendered": 0,
                "contact_leg_nodes_rendered": int(np.count_nonzero(contact_leg_node_mask)),
                "contact_leg_edges_rendered": int(contact_leg_edges.shape[0]),
                "contact_leg_selected_nodes_rendered": int(leg_selected_nodes.shape[0]),
                "contact_leg_selected_edges_rendered": int(leg_selected_edges.shape[0]),
                "rest_leg_nodes_rendered": int(np.count_nonzero(rest_leg_node_mask)),
                "rest_leg_edges_rendered": int(rest_leg_edges.shape[0]),
                "rest_leg_selected_nodes_rendered": int(leg_selected_nodes.shape[0]),
                "rest_leg_selected_edges_rendered": int(leg_selected_edges.shape[0]),
                "contact_zoom_context_nodes_available": 0,
                "contact_zoom_context_edges_available": 0,
                "contact_zoom_context_nodes_rendered": 0,
                "contact_zoom_context_edges_rendered": 0,
                "contact_zoom_outside_context_edges_rendered": 0,
                "contact_zoom_selected_boundary_edges_rendered": 0,
                "contact_zoom_boundary_context_edges_dropped": 0,
                "contact_zoom_nonselected_boundary_edges_dropped": 0,
                "contact_zoom_crossing_context_edges_dropped": 0,
                "contact_zoom_isolated_context_nodes_dropped": 0,
                "rest_zoom_context_nodes_available": 0,
                "rest_zoom_context_edges_available": 0,
                "rest_zoom_context_nodes_rendered": 0,
                "rest_zoom_context_edges_rendered": 0,
                "rest_zoom_outside_context_edges_rendered": 0,
                "rest_zoom_selected_boundary_edges_rendered": 0,
                "rest_zoom_boundary_context_edges_dropped": 0,
                "rest_zoom_nonselected_boundary_edges_dropped": 0,
                "rest_zoom_crossing_context_edges_dropped": 0,
                "rest_zoom_isolated_context_nodes_dropped": 0,
            }
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)
        file.write("\n")

    print(f"Wrote {outputs['combined_png']}")
    print(f"Wrote {outputs['combined_pdf']}")
    print(f"Wrote {outputs['contact_png']}")
    print(f"Wrote {outputs['rest_png']}")
    print(f"Wrote {output_json}")
    if args.layout == "rest_map_anchor":
        print(
            "Summary: "
            f"{object_mass_node_count} mass nodes, "
            f"{edges.shape[0]} rest springs, "
            f"anchor node {metadata['anchor_node']}, "
            f"{metadata['anchor_incident_spring_count']} incident springs, "
            f"{metadata['anchor_rest_map_skipped_neighbor_count']} rest-map skipped neighbors."
        )
    else:
        print(
            "Summary: "
            f"{object_mass_node_count} mass nodes, "
            f"{edges.shape[0]} rest springs, "
            f"selected spring patch {metadata['selected_nodes']}, "
            f"visual radii px {metadata['visual_highlight_radius_px']}."
        )


if __name__ == "__main__":
    main()
