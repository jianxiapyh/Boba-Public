"""Compare automatic release LBS against a recorded Git revision on real states."""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REFERENCE_FILE = "gaussian_splatting/dynamic_utils_fp16_no_profiling_orin.py"
DYNAMIC = ("R_cache", "F_prev", "rotation_computed", "Q_cache_bm", "motions_bm_fp32")


def revision_function(revision, namespace):
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{REFERENCE_FILE}"], cwd=ROOT, text=True,
    )
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "lbs_with_rotation_reuse")
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    code = "\n".join(source.splitlines()[first - 1:node.end_lineno])
    globals_copy = dict(namespace)
    exec(compile(code, f"{revision}:{REFERENCE_FILE}", "exec"), globals_copy)
    return globals_copy[node.name], hashlib.sha256(source.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=("single_lift_rope", "double_stretch_sloth"), required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--reference-revision", default="77a3aa309d325fe3ae022d47c91d9b3037cdeb00")
    p.add_argument("--frames", type=int, default=64, help="Maximum recorded frames; 0 uses the whole episode")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--render-check", action="store_true")
    args = p.parse_args()
    if args.batch < 1 or args.frames < 0 or args.rounds < 1 or args.warmup < 0:
        p.error("Use positive batch/round counts and nonnegative frames/warmup")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import warp as wp
    from gaussian_splatting import dynamic_utils as du
    from interactive_playground import load_case_config, set_all_seeds
    from qqtt import InvPhyTrainerWarp
    from qqtt.utils import cfg, logger

    torch.set_grad_enabled(False)
    set_all_seeds(42)
    load_case_config(SimpleNamespace(case_name=args.case,
        base_path=str(ROOT / "data/different_types"), bg_img_path=str(ROOT / "data/bg.png")), cfg, logger)
    logger.set_log_file(path=str(args.output_dir), name="profile")
    trainer = InvPhyTrainerWarp(
        data_path=str(ROOT / "data/different_types" / args.case / "final_data.pkl"),
        base_dir=str(args.output_dir),
    )
    gs = next((ROOT / "gaussian_output" / args.case).glob("*/point_cloud/iteration_10000/point_cloud.ply"))
    checkpoint = sorted((ROOT / "experiments" / args.case / "train").glob("best_*.pth"))[0]
    runtime = trainer._build_runtime_core(
        str(checkpoint), str(gs), n_dup=args.batch - 1,
        cycle_controller_trajectories=True, force_shared_batched_gaussians=True,
    )
    reference, reference_hash = revision_function(
        args.reference_revision, inspect.unwrap(du.lbs_with_rotation_reuse).__globals__,
    )
    recorded = []
    count = min(args.frames or runtime.frame_len, runtime.frame_len)
    if args.warmup >= count:
        p.error("warmup must leave at least one measured frame")
    for index in range(count):
        previous = trainer.batch_controller_points[max(0, index - 1)]
        trainer.simulator.set_controller_interactive(previous, trainer.batch_controller_points[index])
        if trainer.simulator.object_collision_flag:
            trainer.simulator.update_collision_graph()
        wp.capture_launch(trainer.simulator.forward_graph)
        wp.synchronize()
        nodes = wp.to_torch(trainer.simulator.wp_states[-1].wp_x, requires_grad=False)
        trainer.simulator.set_init_state(trainer.simulator.wp_states[-1].wp_x, trainer.simulator.wp_states[-1].wp_v)
        assert bool(torch.isfinite(nodes).all()), index
        recorded.append(nodes.clone())

    def duplicate(cache):
        return {key: (value.clone() if key in DYNAMIC else value)
                for key, value in cache.items() if not key.startswith("_lbs_")}

    caches = {name: duplicate(runtime.rotation_cache) for name in ("original", "release")}
    reset_values = {key: runtime.rotation_cache[key].clone() for key in DYNAMIC}
    def reset(cache):
        for key, value in reset_values.items():
            cache[key].copy_(value)

    functions = {
        "original": lambda nodes: reference(nodes, caches["original"]),
        "release": lambda nodes: du.lbs_with_rotation_reuse(nodes, caches["release"], copy_outputs=False),
    }
    # First-use costs include graph capture, lazy libraries and compilation.
    setup = {}
    for name, function in functions.items():
        torch.cuda.synchronize()
        start = time.perf_counter()
        function(recorded[0])
        torch.cuda.synchronize()
        setup[name] = (time.perf_counter() - start) * 1000
        reset(caches[name])

    draw = None
    if args.render_check:
        import cv2
        from gaussian_splatting.gaussian_renderer import render
        from qqtt.utils.gaussian import build_batch_images_render_view
        width, height = 640, 480
        intrinsic = trainer._scale_intrinsic_for_render_size(cfg.intrinsics[0],
            native_width=int(cfg.WH[0]), native_height=int(cfg.WH[1]),
            render_width=width, render_height=height)
        view, _ = trainer._create_gs_view(cfg.w2cs[0], intrinsic, height, width)
        rendered = build_batch_images_render_view(runtime.gaussians, gaussian_render_mode="shared_template")
        rendered.uses_separate_render_alpha = True
        background = torch.ones(3, device="cuda")
        overlay = cv2.cvtColor(cv2.imread(cfg.bg_img_path), cv2.COLOR_BGR2RGB)
        overlay = torch.as_tensor(cv2.resize(overlay, (width, height)), device="cuda", dtype=torch.float32)
        def draw(outputs):
            runtime.gaussians._xyz, runtime.gaussians._rotation = outputs
            result = render(view, rendered, None, background)
            pixels, _ = trainer._composite_batch_images_without_shadows(
                result["render"], overlay, batch_alpha=result.get("alpha"),
            )
            return pixels.clamp(0, 255).to(torch.uint8).cpu().numpy()

    max_errors = [0.0, 0.0]
    image_rows = []
    selected = {0, count // 2, count - 1}
    for index, nodes in enumerate(recorded):
        expected = functions["original"](nodes)
        actual = functions["release"](nodes)
        for k, (left, right) in enumerate(zip(actual, expected)):
            assert bool(torch.isfinite(left).all()), (index, k)
            max_errors[k] = max(max_errors[k], float((left - right).abs().max()))
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        for key in DYNAMIC:
            torch.testing.assert_close(caches["release"][key], caches["original"][key], atol=0, rtol=0)
        if draw is not None and index in selected:
            left, right = draw(expected), draw(actual)
            equal = np.array_equal(left, right)
            image_rows.append(dict(frame=index, images=args.batch, exact=equal))
            assert equal, (args.case, args.batch, index)

    graphs = caches["release"].get("_lbs_cuda_graphs", {})
    uses_graph = args.batch <= 64
    assert len(graphs) == int(uses_graph)
    borrowed_storage_verified = None
    if uses_graph:
        entry = next(iter(graphs.values()))
        pointers = tuple(value.data_ptr() for value in entry.output)
        assert pointers == tuple(value.data_ptr() for value in functions["release"](recorded[-1]))
        borrowed_storage_verified = True
    rounds = []
    for repeat in range(args.rounds):
        row = {}
        for name in (("original", "release") if repeat % 2 == 0 else ("release", "original")):
            reset(caches[name])
            samples = []
            function = functions[name]
            for index, nodes in enumerate(recorded):
                torch.cuda.synchronize()
                start = time.perf_counter()
                function(nodes)
                torch.cuda.synchronize()
                if index >= args.warmup:
                    samples.append((time.perf_counter() - start) * 1000)
            row[name] = dict(mean_ms=statistics.mean(samples), median_ms=statistics.median(samples), samples_ms=samples)
        rounds.append(row)
    latency = {name: statistics.median(row[name]["mean_ms"] for row in rounds) for name in functions}
    sources = [REFERENCE_FILE, "gaussian_splatting/lbs_cuda_graph.py", "qqtt/engine/trainer_warp.py", "benchmarks/profile_lbs_cuda_graph.py"]
    report = dict(case=args.case, batch_size=args.batch, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        reference_revision=args.reference_revision, reference_lbs_source_sha256=reference_hash,
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        mass_nodes_per_instance=runtime.object_nodes_per_instance, gaussians_per_instance=runtime.gaussians_per_instance,
        episode_frames=runtime.frame_len, recorded_frames=count, warmup_frames=args.warmup, rounds=rounds,
        median_round_mean_ms=latency, latency_reduction_percent=100 * (latency["original"] - latency["release"]) / latency["original"],
        first_call_ms=setup, max_abs_position_difference=max_errors[0], max_abs_rotation_difference=max_errors[1],
        exact_dynamic_cache_parity=True, borrowed_output_storage_verified=borrowed_storage_verified,
        lbs_execution="cuda_graph" if uses_graph else "cuda_kernels", rendered_image_checks=image_rows,
        source_state='GitHub revision reference with shared unchanged quaternion implementation; local runtime edits preserved.',
        scope='LBS only, including automatic dispatch and input staging; borrowed outputs when graphs are used. Physics records the common input sequence but is outside the LBS timing. Rendering checks also run outside timing.',
        tf32=torch.backends.cuda.matmul.allow_tf32,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key:report[key] for key in ("case", "batch_size", "median_round_mean_ms", "latency_reduction_percent", "max_abs_position_difference", "max_abs_rotation_difference", "rendered_image_checks")}), flush=True)


if __name__ == "__main__":
    main()
