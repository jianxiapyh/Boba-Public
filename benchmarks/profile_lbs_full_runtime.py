"""Run the existing rendered benchmark with the automatic release LBS policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.batch < 1:
        p.error("Use a positive batch size")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    import warp as wp
    from qqtt.engine import trainer_warp
    from benchmarks import run_batched_full_runtime_case

    production = trainer_warp.lbs_with_rotation_reuse
    def audited(current_mass_nodes, cache, tau_F=5e-5, *, copy_outputs=True):
        assert copy_outputs is False, "Runtime must avoid graph output clones"
        return production(current_mass_nodes, cache, tau_F, copy_outputs=False)
    trainer_warp.lbs_with_rotation_reuse = audited

    collected = {}
    build = trainer_warp.InvPhyTrainerWarp._build_runtime_core
    def track_runtime(self, *arguments, **kwargs):
        runtime = build(self, *arguments, **kwargs)
        collected.update(trainer=self, runtime=runtime)
        return runtime
    trainer_warp.InvPhyTrainerWarp._build_runtime_core = track_runtime
    sys.argv = [str(ROOT / "benchmarks/run_batched_full_runtime_case.py"),
        "--case_name", args.case, "--batch_size", str(args.batch),
        "--batched_render_variant", "batch_optimized", "--batch_image_resolution", "640x480",
        "--cycle_controller_trajectories", "--output_dir", str(args.output_dir)]
    run_batched_full_runtime_case.main()
    trainer, runtime = collected["trainer"], collected["runtime"]
    tensors = {
        "positions": wp.to_torch(trainer.simulator.wp_states[0].wp_x),
        "velocities": wp.to_torch(trainer.simulator.wp_states[0].wp_v),
        "gaussian_positions": runtime.gaussians._xyz,
        "gaussian_rotations": runtime.gaussians._rotation,
    }
    validation = {
        "runtime_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "runtime_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "runtime_end_allocated_bytes": torch.cuda.memory_allocated(),
    }
    for name, tensor in tensors.items():
        assert bool(torch.isfinite(tensor).all()), name
        array = tensor.detach().cpu().numpy()
        validation[name] = dict(shape=list(array.shape), sha256=hashlib.sha256(array.tobytes()).hexdigest(), finite=True)
    validation["graph_entries"] = len(runtime.rotation_cache.get("_lbs_cuda_graphs", {}))
    assert validation["graph_entries"] == int(args.batch <= 64)
    validation["lbs_execution"] = "cuda_graph" if args.batch <= 64 else "cuda_kernels"
    validation["completed_episode_frames"] = runtime.frame_len
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(dict(case=args.case, batch=args.batch, lbs_execution=validation["lbs_execution"], completed_episode_frames=runtime.frame_len, all_finite=True)), flush=True)


if __name__ == "__main__":
    main()
