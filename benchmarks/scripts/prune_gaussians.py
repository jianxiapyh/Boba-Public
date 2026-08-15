#!/usr/bin/env python3
"""Offline pruning for Boba 3D Gaussian PLY assets.

This is a minimal port of the external pruning script: it keeps the
highest-scoring Gaussians and writes standard 3DGS PLY files that Boba can
load normally. The pruning algorithm is intentionally unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def gaussian_importance(vertex_data, mode: str) -> np.ndarray:
    opacity = np.asarray(vertex_data["opacity"]).reshape(-1)

    if mode == "opacity":
        return opacity

    names = vertex_data.dtype.names or ()
    if "scale_0" not in names:
        raise ValueError("input PLY must contain at least scale_0")
    scale_0 = np.asarray(vertex_data["scale_0"])
    scale = np.stack(
        [
            scale_0,
            np.asarray(vertex_data["scale_1"]) if "scale_1" in names else scale_0,
            np.asarray(vertex_data["scale_2"]) if "scale_2" in names else scale_0,
        ],
        axis=1,
    )

    if mode == "opacity_area":
        sorted_scale = np.sort(np.maximum(scale, 1e-8), axis=1)
        area = sorted_scale[:, 1] * sorted_scale[:, 2]
        return opacity * area

    if mode == "opacity_volume":
        volume = np.prod(np.maximum(scale, 1e-8), axis=1)
        return opacity * volume

    raise ValueError(f"unknown mode: {mode}")


def prune_file(
    src: Path,
    dst: Path,
    keep_ratio: float | None,
    keep_count: int | None,
    mode: str,
) -> tuple[int, int]:
    ply = PlyData.read(src)
    vertex = ply["vertex"]
    before = int(vertex.count)

    if keep_count is None:
        assert keep_ratio is not None
        keep_count = int(round(before * keep_ratio))
    keep_count = min(max(1, keep_count), before)

    score = gaussian_importance(vertex.data, mode)
    keep_idx = np.argpartition(score, -keep_count)[-keep_count:]
    keep_idx = keep_idx[np.argsort(score[keep_idx])[::-1]]

    dst.parent.mkdir(parents=True, exist_ok=True)
    pruned_vertex = PlyElement.describe(vertex.data[keep_idx], "vertex")
    PlyData(
        [pruned_vertex],
        text=ply.text,
        byte_order=ply.byte_order,
        comments=ply.comments,
        obj_info=ply.obj_info,
    ).write(dst)
    return before, keep_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input PLY file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output PLY file or directory")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keep-ratio", type=float, help="Fraction of points to keep, e.g. 0.1")
    group.add_argument("--keep-count", type=int, help="Number of points to keep per file")

    parser.add_argument(
        "--mode",
        choices=["opacity", "opacity_area", "opacity_volume"],
        default="opacity_area",
        help="Importance score used for top-k pruning",
    )
    args = parser.parse_args()

    if args.keep_ratio is not None and not (0.0 < args.keep_ratio <= 1.0):
        raise ValueError("--keep-ratio must be in (0, 1]")

    if args.input.is_file():
        output = args.output
        if output.suffix.lower() != ".ply":
            output = output / args.input.name
        before, after = prune_file(args.input, output, args.keep_ratio, args.keep_count, args.mode)
        print(f"{args.input} -> {output}: {before} -> {after} ({after / before:.1%})")
        return

    if not args.input.is_dir():
        raise FileNotFoundError(args.input)

    total_before = 0
    total_after = 0
    for src in sorted(args.input.rglob("*.ply")):
        rel = src.relative_to(args.input)
        dst = args.output / rel
        before, after = prune_file(src, dst, args.keep_ratio, args.keep_count, args.mode)
        total_before += before
        total_after += after
        print(f"{rel}: {before} -> {after} ({after / before:.1%})")

    print(f"total: {total_before} -> {total_after} ({total_after / total_before:.1%})")


if __name__ == "__main__":
    main()
