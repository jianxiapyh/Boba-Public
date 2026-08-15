# Boba runtime path for single-instance and batched execution.
# instance == 1 uses the single-instance path.
# instance > 1 enables batched optimizations.
import json
import math
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

import open3d as o3d

from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.dynamic_utils import (
    lbs_with_rotation_reuse,
    build_rotation_reuse_cache,
    knn_weights_sparse,
    get_topk_indices,
)
from gaussian_splatting.utils.graphics_utils import focal2fov
from qqtt.data import RealData
from qqtt.model.diff_simulator import (
    SpringMassSystemWarp,
)
from qqtt.utils import logger, cfg
from qqtt.utils.gaussian import (
    build_batch_images_render_view,
    build_instance_selective_render_view,
    remove_gaussians_with_low_opacity,
    load_shared_batched_gaussians,
    normalize_gaussian_render_mode,
)

SIM_FORCE_MODE_GATHER = "gather"
SIM_FORCE_MODE_TEMPLATE_STATE_BATCHED_ATOMIC = "template_state_batched_atomic"
SIM_FORCE_MODES = (
    SIM_FORCE_MODE_GATHER,
    SIM_FORCE_MODE_TEMPLATE_STATE_BATCHED_ATOMIC,
)

#pyh moving timer class outside 
class Timer:
    def __init__(self, name):
        self.name = name
        self.elapsed = 0
        self.start_time = None
        self.cuda_start_event = None
        self.cuda_end_event = None
        self.use_cuda = torch.cuda.is_available()

    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.cuda_start_event = torch.cuda.Event(enable_timing=True)
            self.cuda_end_event = torch.cuda.Event(enable_timing=True)
            self.cuda_start_event.record()
        self.start_time = time.time()

    def stop(self):
        if self.use_cuda:
            self.cuda_end_event.record()
            torch.cuda.synchronize()
            self.elapsed = (
                self.cuda_start_event.elapsed_time(self.cuda_end_event) / 1000
            )  # convert ms to seconds
        else:
            self.elapsed = time.time() - self.start_time
        return self.elapsed

    def reset(self):
        self.elapsed = 0
        self.start_time = None
        self.cuda_start_event = None
        self.cuda_end_event = None


class RenderComponentProfiler:
    COMPONENT_FIELDS = [
        "render_gsplat_total_ms",
        "prepare_inputs_ms",
        "fully_fused_projection_ms",
        "spherical_harmonics_ms",
        "isect_tiles_ms",
        "isect_tiles_count_kernel_ms",
        "isect_tiles_cumsum_ms",
        "isect_tiles_emit_kernel_ms",
        "isect_tiles_sort_ms",
        "isect_tiles_cuda_total_ms",
        "isect_visible_gaussians",
        "isect_total_tile_intersections",
        "isect_avg_tiles_per_gaussian",
        "isect_max_tiles_per_gaussian",
        "isect_offset_encode_ms",
        "rasterize_to_pixels_ms",
        "background_depth_finalize_ms",
        "format_output_ms",
        "shared_projection_loop_ms",
        "shared_template_gather_ms",
        "shared_projection_cat_ms",
        "densify_projection_metadata_ms",
    ]

    def __init__(self):
        self.frames = []
        self.current_frame = None

    def begin_frame(self, total_gaussians=None, gaussians_per_instance=None):
        self.current_frame = {
            "total_gaussians": total_gaussians,
            "gaussians_per_instance": gaussians_per_instance,
            "components": {},
        }

    def end_frame(self):
        if self.current_frame is not None:
            self.frames.append(self.current_frame)
        self.current_frame = None

    def discard_frame(self):
        self.current_frame = None

    class _RecordScope:
        def __init__(self, profiler, name):
            self.profiler = profiler
            self.name = name
            self.start_time = None

        def __enter__(self):
            if self.profiler.current_frame is None:
                return self
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.profiler.current_frame is None or self.start_time is None:
                return False
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
            components = self.profiler.current_frame["components"]
            components[self.name] = components.get(self.name, 0.0) + elapsed_ms
            return False

    def record(self, name):
        return self._RecordScope(self, name)

    def record_value(self, name, value):
        if self.current_frame is None:
            return
        components = self.current_frame["components"]
        components[name] = components.get(name, 0.0) + float(value)

    def _mean_frame_value(self, field):
        values = [
            frame[field]
            for frame in self.frames
            if frame.get(field) is not None
        ]
        if not values:
            return None
        return float(np.mean(values))

    def _mean_component_value(self, component_name):
        values = [
            frame["components"].get(component_name)
            for frame in self.frames
            if frame["components"].get(component_name) is not None
        ]
        if not values:
            return None
        return float(np.mean(values))

    def summary(self, metadata):
        profile = dict(metadata)
        profile["frames_profiled"] = len(self.frames)
        profile["total_gaussians"] = self._mean_frame_value("total_gaussians")
        profile["gaussians_per_instance"] = self._mean_frame_value(
            "gaussians_per_instance"
        )
        for component_name in self.COMPONENT_FIELDS:
            profile[component_name] = self._mean_component_value(component_name)
        return profile

    def write_json(self, output_dir, metadata):
        if output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        profile_path = os.path.join(output_dir, "render_component_profile.json")
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(self.summary(metadata), f, indent=2)


def perf_component_label(component_name):
    labels = {
        "simulator": "Simulator",
        "full_motion_interpolation": "Linear Blend Skinning",
        "rendering": "Rendering",
        "frame_compositing": "Frame compositing",
    }
    return labels.get(component_name, component_name.replace("_", " ").capitalize())


class InvPhyTrainerWarp:
    #getting called automatically right after you create an instance of the class
    def __init__(
        self,
        data_path,
        base_dir,
    ):
        cfg.data_path= data_path
        cfg.base_dir = base_dir
        cfg.device = "cuda:0"

        self.init_masks = None
        self.init_velocities = None
        # Load the data
        if cfg.data_type == "real":
            self.dataset = RealData(visualize=False, save_gt=False)
            # Get the object points and controller points
            self.controller_points = self.dataset.controller_points
            self.structure_points = self.dataset.structure_points
            self.num_all_points = self.dataset.num_all_points
        elif cfg.data_type == "synthetic":
            print(f"synthetic data detected")
            import pdb
            pdb.set_trace()
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")
        
        first_frame_controller_points = self.controller_points[0]
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            self.structure_points,
            first_frame_controller_points,
            object_radius=cfg.object_radius,
            object_max_neighbours=cfg.object_max_neighbours,
            controller_radius=cfg.controller_radius,
            controller_max_neighbours=cfg.controller_max_neighbours,
            mask=self.init_masks,
        )

        #pyh move gaussian to a class variable
        self.gaussians =  None
        self.controller_points_group = None
        self.multi_frame_len = None
        self.num_input_trajectories = None

    #init_start with morton reordering for both mass node and spring (current last working version)
    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
        mask=None,
    ):
        object_points = object_points.cpu().numpy()
        if controller_points is not None:
            controller_points = controller_points.cpu().numpy()
        if mask is None:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

            # Connect the springs of the objects first
            points = np.asarray(object_pcd.points)
            spring_flags = np.zeros((len(points), len(points)))
            springs = []
            rest_lengths = []
            for i in range(len(points)):
                [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    points[i], object_radius, object_max_neighbours
                )
                idx = idx[1:]
                for j in idx:
                    rest_length = np.linalg.norm(points[i] - points[j])
                    if (
                        spring_flags[i, j] == 0
                        and spring_flags[j, i] == 0
                        and rest_length > 1e-4
                    ):
                        spring_flags[i, j] = 1
                        spring_flags[j, i] = 1
                        springs.append([i, j])
                        rest_lengths.append(np.linalg.norm(points[i] - points[j]))

            num_object_springs = len(springs)
            
            # ============================================================
            # 🆕 NEW: Save num_object_points BEFORE adding controllers
            # ============================================================
            num_object_points = len(points)
            print(f" num object springs {num_object_springs}")
            if controller_points is not None:
                # Connect the springs between the controller points and the object points
                # ============================================================
                # ⚠️  REMOVED: num_object_points = len(points)
                # (moved above, before controller springs are added)
                # ============================================================

                points = np.concatenate([points, controller_points], axis=0)
                for i in range(len(controller_points)):
                    [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                        controller_points[i],
                        controller_radius,
                        controller_max_neighbours,
                    )
                    for j in idx:
                        springs.append([num_object_points + i, j])
                        rest_lengths.append(
                            np.linalg.norm(controller_points[i] - points[j])
                        )

            springs = np.array(springs)
            print(f" total springs {len(springs)}")

            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(points))
            
            # ============================================================
            # 🆕 NEW: Morton reordering section (entire block is new)
            # ============================================================
            # Convert to torch tensors
            points_torch = torch.tensor(points, dtype=torch.float32, device=cfg.device)
            springs_torch = torch.tensor(springs, dtype=torch.int32, device=cfg.device)
            rest_lengths_torch = torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device)
            masses_torch = torch.tensor(masses, dtype=torch.float32, device=cfg.device)
            
            # Apply Morton reordering
            logger.info("Applying Morton ordering to improve cache performance...")
            points_torch, springs_torch, rest_lengths_torch, spring_perm = self._apply_morton_reordering(
                points_torch, springs_torch, rest_lengths_torch, num_object_points
            )
            self.spring_permutation = spring_perm

            return (
                points_torch,
                springs_torch,
                rest_lengths_torch,
                masses_torch,
                num_object_springs,
            )
            # ============================================================            
        else:
            mask = mask.cpu().numpy()
            # Get the unique value in masks
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
            # Loop different objects to connect the springs separately
            for value in unique_values:
                temp_points = object_points[mask == value]
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(temp_points)
                temp_tree = o3d.geometry.KDTreeFlann(temp_pcd)
                temp_spring_flags = np.zeros((len(temp_points), len(temp_points)))
                temp_springs = []
                temp_rest_lengths = []
                for i in range(len(temp_points)):
                    [k, idx, _] = temp_tree.search_hybrid_vector_3d(
                        temp_points[i], object_radius, object_max_neighbours
                    )
                    idx = idx[1:]
                    for j in idx:
                        rest_length = np.linalg.norm(temp_points[i] - temp_points[j])
                        if (
                            temp_spring_flags[i, j] == 0
                            and temp_spring_flags[j, i] == 0
                            and rest_length > 1e-4
                        ):
                            temp_spring_flags[i, j] = 1
                            temp_spring_flags[j, i] = 1
                            temp_springs.append([i + index, j + index])
                            temp_rest_lengths.append(rest_length)
                vertices += temp_points.tolist()
                springs += temp_springs
                rest_lengths += temp_rest_lengths
                index += len(temp_points)

            num_object_springs = len(springs)
            
            # ============================================================
            # 🆕 NEW: Save num_object_points (for multi-object case)
            # ============================================================
            num_object_points = len(vertices)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(vertices))
            
            # ============================================================
            # 🆕 NEW: Morton reordering section (entire block is new)
            # ============================================================
            # Convert to torch tensors
            vertices_torch = torch.tensor(vertices, dtype=torch.float32, device=cfg.device)
            springs_torch = torch.tensor(springs, dtype=torch.int32, device=cfg.device)
            rest_lengths_torch = torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device)
            masses_torch = torch.tensor(masses, dtype=torch.float32, device=cfg.device)
            
            # Apply Morton reordering
            logger.info("Applying Morton ordering to improve cache performance (multi-object)...")
            vertices_torch, springs_torch, rest_lengths_torch, spring_perm = self._apply_morton_reordering(
                vertices_torch, springs_torch, rest_lengths_torch, num_object_points
            )
            self.spring_permutation = spring_perm

            # NEW:
            return (
                vertices_torch,
                springs_torch,
                rest_lengths_torch,
                masses_torch,
                num_object_springs,
            )
            # ============================================================

    def _find_closest_point(self, target_points):
        """Find the closest structure point to any of the target points."""
        dist_matrix = torch.sum(
            (target_points.unsqueeze(1) - self.structure_points.unsqueeze(0)) ** 2,
            dim=2,
        )
        min_dist_per_ctrl_pts, min_indices = torch.min(dist_matrix, dim=1)
        min_idx = min_indices[torch.argmin(min_dist_per_ctrl_pts)]
        return self.structure_points[min_idx].unsqueeze(0)

    def stable_lexsort(self,keys):
        """
        keys: list of 1D tensors, all same length.
            Order is MOST-significant -> LEAST-significant.
        Returns: permutation idx such that keys are sorted lexicographically.
        """
        assert len(keys) > 0
        n = keys[0].numel()
        idx = torch.arange(n, device=keys[0].device)

        # stable sort from least-significant to most-significant
        for k in reversed(keys):
            idx = idx[torch.argsort(k[idx], stable=True)]
        return idx

    def _reorder_springs_spatial_blocking(self, springs, rest_lengths, num_object_points, block_size=32):
        """
        Cluster springs so consecutive springs mostly touch vertices in the same index-block
        (after Morton vertex reordering, index-blocks correspond to spatial regions).
        
        Preserves: object-object springs first, then controller-object springs.
        Returns: springs2, rest2, perm
        """
        N = int(num_object_points)
        s = springs.long()
        i = s[:, 0]
        j = s[:, 1]
        
        # Split types (preserve your num_object_springs prefix assumption)
        obj_obj_mask = (i < N) & (j < N)
        ctrl_mask = ~obj_obj_mask

        if ctrl_mask.any():
            has_obj_ep = (i[ctrl_mask] < N) | (j[ctrl_mask] < N)
            assert has_obj_ep.all(), "Found controller-controller springs!"
                
        # ---- object-object springs ----
        obj_ids = torch.nonzero(obj_obj_mask, as_tuple=False).squeeze(1)
        if obj_ids.numel() > 0:
            io = i[obj_ids]
            jo = j[obj_ids]
            a = torch.minimum(io, jo)
            b = torch.maximum(io, jo)
            
            ablk = a // block_size
            bblk = b // block_size
            
            # Sort by (ablk, bblk, a, b) lexicographically
            perm_obj_local = self.stable_lexsort([ablk, bblk, a, b])
            obj_order = obj_ids[perm_obj_local]
        else:
            obj_order = obj_ids
        
        # ---- controller-object springs ----
        ctrl_ids = torch.nonzero(ctrl_mask, as_tuple=False).squeeze(1)
        if ctrl_ids.numel() > 0:
            ic = i[ctrl_ids]
            jc = j[ctrl_ids]
            
            # object endpoint = the one < N
            obj_ep  = torch.where(ic < N, ic, jc)
            ctrl_ep = torch.where(ic >= N, ic, jc)
            ctrl_id = (ctrl_ep - N).clamp_min(0)
            
            obj_blk = obj_ep // block_size
            
            perm_ctrl_local = self.stable_lexsort([obj_blk, obj_ep, ctrl_id])
            ctrl_order = ctrl_ids[perm_ctrl_local]

        else:
            ctrl_order = ctrl_ids
        
        perm = torch.cat([obj_order, ctrl_order], dim=0)
        
        # Sanity checks
        assert perm.shape[0] == springs.shape[0], "Permutation size mismatch"
        assert torch.unique(perm).shape[0] == springs.shape[0], "Permutation has duplicates"
        
        print(f"Reordered springs with spatial blocking (block_size={block_size}):")
        print(f"  Object-object springs: {obj_ids.numel()}")
        print(f"  Controller-object springs: {ctrl_ids.numel()}")
        
        return springs[perm], rest_lengths[perm], perm

    def _apply_morton_reordering(self, vertices, springs, rest_lengths, num_object_points):
        """
        Reorder object vertices using Morton (Z-order) curve AND reorder springs for coalescing.
        
        Returns:
            new_vertices: reordered vertices
            new_springs: springs with remapped indices AND reordered for coalescing
            new_rest_lengths: rest_lengths reordered to match springs
            spring_permutation: permutation applied to springs (for reordering other per-spring arrays)
        """
        device = vertices.device
        
        # Device assertions
        assert springs.device == device
        assert rest_lengths.device == device
        
        obj_end = num_object_points
        has_controllers = (num_object_points < len(vertices))
        
        # === Step 1: Morton reorder vertices ===
        obj_verts = vertices[:obj_end].detach().cpu().numpy()
        
        # Normalize to [0,1] for Morton encoding
        mins = obj_verts.min(axis=0)
        maxs = obj_verts.max(axis=0)
        range_vals = maxs - mins
        range_vals[range_vals < 1e-8] = 1.0
        normalized = (obj_verts - mins) / range_vals
        
        # Convert to 21-bit integer coordinates
        BITS = 21
        MAX_VAL = (1 << BITS) - 1
        int_coords = (normalized * MAX_VAL).astype(np.uint64)
        
        # Compute Morton codes
        def part1by2(n):
            n = np.uint64(n)
            n = (n | (n << 32)) & np.uint64(0x1f00000000ffff)
            n = (n | (n << 16)) & np.uint64(0x1f0000ff0000ff)
            n = (n | (n << 8))  & np.uint64(0x100f00f00f00f00f)
            n = (n | (n << 4))  & np.uint64(0x10c30c30c30c30c3)
            n = (n | (n << 2))  & np.uint64(0x1249249249249249)
            return n
        
        x = part1by2(int_coords[:, 0])
        y = part1by2(int_coords[:, 1])
        z = part1by2(int_coords[:, 2])
        morton_codes = x | (y << 1) | (z << 2)
        
        # Stable sort
        perm = np.argsort(morton_codes, kind="stable")
        inv_perm = np.empty(obj_end, dtype=np.int64)
        inv_perm[perm] = np.arange(obj_end)

        sorted_morton_codes = morton_codes[perm]
        morton_codes_torch = torch.from_numpy(sorted_morton_codes).to(device).long()  # ✅ Convert to int64

        
        # Convert to torch
        perm_torch = torch.from_numpy(perm).to(device).long()
        inv_perm_torch = torch.from_numpy(inv_perm).to(device).long()
        
        # Reorder vertices
        reordered_obj_verts = torch.index_select(vertices[:obj_end], 0, perm_torch)
        if has_controllers:
            new_vertices = torch.cat([reordered_obj_verts, vertices[obj_end:]], dim=0)
        else:
            new_vertices = reordered_obj_verts
        
        # === Step 2: Remap spring indices ===
        springs_dtype = springs.dtype
        new_springs = springs.clone()
        
        for col in [0, 1]:
            obj_mask = springs[:, col] < obj_end
            obj_indices = springs[obj_mask, col].long()
            remapped_indices = inv_perm_torch[obj_indices].to(springs_dtype)
            new_springs[obj_mask, col] = remapped_indices
        
        logger.info(f"Applied Morton reordering to {obj_end} object vertices")
        
        # === Step 3: Reorder springs for coalescing ===
        new_springs, new_rest_lengths, spring_permutation = self._reorder_springs_spatial_blocking(
            new_springs, rest_lengths, num_object_points, block_size=32
        )
        return new_vertices, new_springs, new_rest_lengths, spring_permutation

    def build_signed_incidence_map(self, base_springs, object_massnode_single, device="cuda:0"):
        """
        Build a flattened signed incidence map for the gather-based spring assembly.

        Each row corresponds to one object mass node. Positive entries mean the node
        adds the corresponding spring force; negative entries mean it subtracts it.
        """
        springs_cpu = base_springs.detach().to(dtype=torch.int64, device="cpu")
        per_node_incidence = [[] for _ in range(int(object_massnode_single))]

        for spring_idx, endpoints in enumerate(springs_cpu.tolist()):
            endpoint_a, endpoint_b = endpoints
            signed_spring_id = spring_idx + 1

            if endpoint_a < object_massnode_single:
                per_node_incidence[endpoint_a].append(signed_spring_id)
            if endpoint_b < object_massnode_single:
                per_node_incidence[endpoint_b].append(-signed_spring_id)

        max_incident_springs = max((len(entries) for entries in per_node_incidence), default=0)
        incidence_map = torch.zeros(
            (int(object_massnode_single), max_incident_springs),
            dtype=torch.int32,
        )

        for node_idx, entries in enumerate(per_node_incidence):
            if entries:
                incidence_map[node_idx, : len(entries)] = torch.tensor(entries, dtype=torch.int32)

        logger.info(
            f"Built signed incidence map for gather assembly with max incident spring count {max_incident_springs}"
        )
        return incidence_map.flatten().to(device=device), max_incident_springs

    def check_controller_group_same_start(
        self,
        controller_points_group,
        atol=1e-6,
        rtol=0.0,
        allow_global_translation=False,
        translation_mode="mean",
        verbose=True,
    ):
        assert controller_points_group.ndim == 4 and controller_points_group.shape[-1] == 3, (
            f"Expected (N,T,C,3), got {tuple(controller_points_group.shape)}"
        )

        num_instances, num_frames, num_controllers, _ = controller_points_group.shape
        start = controller_points_group[:, 0]
        ref = start[0:1]

        if allow_global_translation:
            if translation_mode == "mean":
                translation = start.mean(dim=1, keepdim=True) - ref.mean(dim=1, keepdim=True)
            elif translation_mode == "first":
                translation = start[:, 0:1, :] - ref[:, 0:1, :]
            else:
                raise ValueError("translation_mode must be 'mean' or 'first'")
            diff = (start - translation) - ref
        else:
            diff = start - ref

        per_instance_max = diff.abs().amax(dim=(1, 2))
        tol = atol + rtol * ref.abs().amax(dim=(1, 2))
        ok = (per_instance_max <= tol).tolist()
        all_ok = all(ok)

        if verbose:
            print(
                f"[Test] N={num_instances}, T={num_frames}, C={num_controllers}, "
                f"allow_translation={allow_global_translation} ({translation_mode})"
            )
            print(f"[Test] all_ok={all_ok}")
            if not all_ok:
                bad = [i for i, passed in enumerate(ok) if not passed]
                print(f"[Test] mismatching instances: {bad}")
            print(f"[Test] per-instance max abs diff: {per_instance_max.detach().cpu().numpy()}")

        return all_ok, per_instance_max, ok

    def load_controller_points_group_pkl(self, pkl_path, device="cuda"):
        with open(pkl_path, "rb") as handle:
            root = pickle.load(handle)

        if not isinstance(root, dict):
            raise TypeError(f"PKL root must be dict, got {type(root)}")
        if "controller_points_group" not in root:
            raise KeyError(
                f"Missing key 'controller_points_group' in {pkl_path}. Keys={list(root.keys())}"
            )

        group = root["controller_points_group"]
        if not isinstance(group, list) or len(group) == 0:
            raise TypeError(
                f"'controller_points_group' must be a non-empty list, got {type(group)} "
                f"len={len(group) if hasattr(group, '__len__') else '??'}"
            )

        frame_count = controller_count = None
        tensors = []
        for idx, arr in enumerate(group):
            arr = np.asarray(arr)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(
                    f"controller_points_group[{idx}] shape {arr.shape}, expected (T, C, 3)"
                )
            cur_frames, cur_controllers, _ = arr.shape
            if frame_count is None:
                frame_count, controller_count = cur_frames, cur_controllers
            elif (cur_frames, cur_controllers) != (frame_count, controller_count):
                raise ValueError(
                    f"controller_points_group[{idx}] has (T,C)=({cur_frames},{cur_controllers}) "
                    f"but expected ({frame_count},{controller_count})."
                )
            tensors.append(torch.from_numpy(arr.astype(np.float32, copy=False)).to(device=device))

        return torch.stack(tensors, dim=0)

    def _ensure_controller_points_group_loaded(self):
        if self.controller_points_group is not None:
            return self.controller_points_group

        controller_group_path = Path(cfg.data_path).parent / "multi_ctrls.pkl"
        if not controller_group_path.exists():
            raise FileNotFoundError(
                f"Missing multi-instance controller trajectories: {controller_group_path}"
            )

        controller_points_group = self.load_controller_points_group_pkl(
            controller_group_path, device=cfg.device
        )
        self.check_controller_group_same_start(controller_points_group, atol=1e-5)

        if controller_points_group.shape[2] != self.controller_points.shape[1]:
            raise ValueError(
                "multi_ctrls.pkl controller count does not match single-instance controller count: "
                f"{controller_points_group.shape[2]} vs {self.controller_points.shape[1]}"
            )

        reference_start = self.controller_points[0].to(
            device=controller_points_group.device,
            dtype=controller_points_group.dtype,
        )
        if not torch.allclose(
            controller_points_group[0, 0], reference_start, atol=1e-5, rtol=0.0
        ):
            max_diff = (
                controller_points_group[0, 0] - reference_start
            ).abs().max().item()
            raise ValueError(
                "multi_ctrls.pkl frame-0 controller pose does not match self.controller_points[0]. "
                f"max abs diff: {max_diff:.6e}"
            )

        self.controller_points_group = controller_points_group
        self.multi_frame_len = int(controller_points_group.shape[1])
        self.num_input_trajectories = int(controller_points_group.shape[0])
        return self.controller_points_group

    def _load_render_backend(self):
        import cv2
        import glfw
        from OpenGL import GL as gl
        import pycuda.driver as cuda_driver
        from pycuda.gl import RegisteredBuffer, graphics_map_flags
        from gaussian_splatting.gaussian_renderer import render as render_gaussian

        return SimpleNamespace(
            cv2=cv2,
            glfw=glfw,
            gl=gl,
            cuda_driver=cuda_driver,
            RegisteredBuffer=RegisteredBuffer,
            graphics_map_flags=graphics_map_flags,
            render_gaussian=render_gaussian,
        )

    def _default_diagonal_instance_offsets(self, batch_size, dtype):
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if batch_size == 1:
            return torch.zeros((1, 3), dtype=dtype, device=cfg.device)

        offset_step = torch.tensor([10, 10, 0], dtype=dtype, device=cfg.device)
        instance_ids = torch.arange(
            batch_size, device=cfg.device, dtype=dtype
        ).unsqueeze(1)
        return (instance_ids * offset_step.view(1, 3)).contiguous()

    def _build_runtime_core(
        self,
        model_path,
        gs_path,
        n_dup=0,
        instance_offsets=None,
        gaussian_render_mode="shared_template",
        force_shared_batched_gaussians=False,
        sim_force_mode=SIM_FORCE_MODE_GATHER,
    ):
        gaussian_render_mode = normalize_gaussian_render_mode(gaussian_render_mode)
        if sim_force_mode not in SIM_FORCE_MODES:
            raise ValueError(
                f"sim_force_mode must be one of {SIM_FORCE_MODES}. "
                f"Received: {sim_force_mode}"
            )
        if cfg.self_collision:
            print(f"collision dist {cfg.collision_dist}")
        else:
            print("no collision flag set")

        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        trained_spring_Y = checkpoint["spring_Y"]
        trained_collide_elas = checkpoint["collide_elas"]
        trained_collide_fric = checkpoint["collide_fric"]
        trained_collide_object_elas = checkpoint["collide_object_elas"]
        trained_collide_object_fric = checkpoint["collide_object_fric"]

        trained_spring_Y = trained_spring_Y[self.spring_permutation]

        obj_init_vertices = self.init_vertices[: self.num_all_points]
        ctrl_init_vertices = self.init_vertices[self.num_all_points :]
        n_vert_single_obj = obj_init_vertices.shape[0]
        n_vert_single_ctrl = ctrl_init_vertices.shape[0]
        frame_len = int(self.dataset.frame_len)
        controller_points_group = None
        batch_size = n_dup + 1
        if n_dup > 0:
            controller_points_group = self._ensure_controller_points_group_loaded()
            required_instances = batch_size
            if self.num_input_trajectories < required_instances:
                raise ValueError(
                    "multi_ctrls.pkl does not contain enough trajectories for the requested "
                    f"batch size {required_instances}. Found {self.num_input_trajectories}."
                )
            frame_len = self.multi_frame_len

        if instance_offsets is None:
            resolved_instance_offsets = self._default_diagonal_instance_offsets(
                batch_size=batch_size,
                dtype=obj_init_vertices.dtype,
            )
        else:
            resolved_instance_offsets = instance_offsets.to(
                device=cfg.device,
                dtype=obj_init_vertices.dtype,
            ).contiguous()
            expected_shape = (batch_size, 3)
            if tuple(resolved_instance_offsets.shape) != expected_shape:
                raise ValueError(
                    "instance_offsets must have shape "
                    f"{expected_shape}, got {tuple(resolved_instance_offsets.shape)}"
                )

        out_init_vertices = []
        out_init_velocities = []
        out_controller_points = []

        for dup_i in range(batch_size):
            shift = resolved_instance_offsets[dup_i]
            obj_v = obj_init_vertices + shift

            if self.init_velocities is not None:
                out_init_velocities.append(self.init_velocities)

            out_init_vertices.append(obj_v)

        base_ctrl_vert_offset = batch_size * n_vert_single_obj

        for dup_i in range(batch_size):
            shift = resolved_instance_offsets[dup_i]
            ctrl_v = ctrl_init_vertices + shift
            if n_dup > 0:
                new_controller_points = controller_points_group[dup_i] + shift
            else:
                new_controller_points = self.controller_points
            out_init_vertices.append(ctrl_v)
            out_controller_points.append(new_controller_points)

        self.batch_init_vertices = torch.cat(out_init_vertices, dim=0)
        self.batch_init_velocities = None
        if self.init_velocities is not None:
            self.batch_init_velocities = torch.cat(out_init_velocities, dim=0)

        self.batch_controller_points = torch.cat(out_controller_points, dim=1)

        expected_total = base_ctrl_vert_offset + batch_size * n_vert_single_ctrl
        print(
            f"[Check] single instance object mass node {n_vert_single_obj}, controller mass node {n_vert_single_ctrl}"
        )
        print(
            f"[CHECK] total mass nodes {self.batch_init_vertices.shape[0]}, expected {expected_total}"
        )
        print(
            "batch_init_vertices:",
            type(self.batch_init_vertices),
            self.batch_init_vertices.shape,
            self.batch_init_vertices.dtype,
            self.batch_init_vertices.device,
        )
        print(
            "batch_controller_points:",
            type(self.batch_controller_points),
            self.batch_controller_points.shape,
            self.batch_controller_points.dtype,
            self.batch_controller_points.device,
        )

        signed_incidence_map_flat = None
        max_incident_springs = 0
        if n_dup > 0 and sim_force_mode == SIM_FORCE_MODE_GATHER:
            signed_incidence_map_flat, max_incident_springs = self.build_signed_incidence_map(
                self.init_springs,
                n_vert_single_obj,
                device=cfg.device,
            )

        self.simulator = SpringMassSystemWarp(
            base_springs=self.init_springs,
            base_rest_lengths=self.init_rest_lengths,
            init_masses=self.init_masses,
            init_masks=self.init_masks,
            signed_incidence_map=signed_incidence_map_flat,
            max_incident_springs=max_incident_springs,
            init_vertices=self.batch_init_vertices,
            init_velocities=self.batch_init_velocities,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collision_dist=cfg.collision_dist,
            reverse_z=cfg.reverse_z,
            spring_Y_max=cfg.spring_Y_max,
            spring_Y_min=cfg.spring_Y_min,
            self_collision=cfg.self_collision,
            collide_elas=trained_collide_elas,
            collide_fric=trained_collide_fric,
            collide_object_elas=trained_collide_object_elas,
            collide_object_fric=trained_collide_object_fric,
            spring_Y=trained_spring_Y,
            object_massnodes_total=base_ctrl_vert_offset,
            object_massnodes_single=n_vert_single_obj,
            controller_massnodes_single=n_vert_single_ctrl,
            controller_rest_location=self.batch_controller_points[0],
            number_of_instance=batch_size,
            sim_force_mode=sim_force_mode,
        )

        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )

        if self.simulator.object_collision_flag:
            self.simulator.create_resting_case()

        self.simulator.create_cuda_graph()

        if n_dup == 0 and not force_shared_batched_gaussians:
            gaussians = GaussianModel(sh_degree=3)
            gaussians.load_ply(gs_path)
            gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
            gaussians.isotropic = True
            render_gaussians = gaussians
            n_gaussians_single_obj = gaussians._xyz.shape[0]
            xyz_rest_single = gaussians.get_xyz[:n_gaussians_single_obj]
            rot_rest_single = gaussians.get_rotation[:n_gaussians_single_obj]
        else:
            gaussians, render_gaussians = load_shared_batched_gaussians(
                gs_path=gs_path,
                number_of_instance=batch_size,
                instance_offsets=resolved_instance_offsets,
                sh_degree=3,
                opacity_threshold=0.1,
                gaussian_render_mode=gaussian_render_mode,
            )
            n_gaussians_single_obj = gaussians.gaussians_per_instance
            xyz_rest_single = gaussians.xyz_rest_single
            rot_rest_single = gaussians.rotation_rest_single

        torch.cuda.empty_cache()

        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        current_pos = gaussians.get_xyz

        rest_mass_node_single = prev_x[:n_vert_single_obj]
        relations_single = get_topk_indices(rest_mass_node_single, K=3)
        weights_single, weights_indices_single = knn_weights_sparse(
            rest_mass_node_single, current_pos[:n_gaussians_single_obj], K=3
        )

        rotation_cache = build_rotation_reuse_cache(
            weights_indices=weights_indices_single,
            weights=weights_single,
            relations=relations_single,
            mass_nodes_rest=rest_mass_node_single,
            gaussians_xyz_rest=xyz_rest_single,
            gaussians_quat_rest=rot_rest_single,
            device=cfg.device,
            mass_node_per_instance=n_vert_single_obj,
            gaussians_per_instance=n_gaussians_single_obj,
            number_of_instance=batch_size,
        )

        instance_offsets = resolved_instance_offsets

        return SimpleNamespace(
            gaussians=gaussians,
            render_gaussians=render_gaussians,
            rotation_cache=rotation_cache,
            prev_x=prev_x,
            prev_target=self.batch_controller_points[0],
            current_target=self.batch_controller_points[0],
            frame_count=0,
            frame_len=frame_len,
            batch_size=batch_size,
            object_nodes_per_instance=n_vert_single_obj,
            controller_nodes_per_instance=n_vert_single_ctrl,
            gaussians_per_instance=n_gaussians_single_obj,
            instance_offsets=instance_offsets,
            gaussian_render_mode=gaussian_render_mode,
            sim_force_mode=sim_force_mode,
        )

    def _validate_batched_render_request(self, render_mode, instance_id, batch_size):
        if render_mode not in ("instance", "batch_images"):
            raise ValueError(
                "render_mode must be 'instance' or 'batch_images'. "
                f"Received: {render_mode}"
            )

        if render_mode == "instance":
            if instance_id is None:
                raise ValueError(
                    "instance_id is required when render_mode='instance'."
                )
            instance_id = int(instance_id)
            if instance_id < 0 or instance_id >= int(batch_size):
                raise ValueError(
                    f"instance_id must be in [0, {int(batch_size) - 1}], got {instance_id}"
                )
            return instance_id

        if instance_id is not None:
            raise ValueError(
                "instance_id can only be provided when render_mode='instance'."
            )
        return None

    def _select_runtime_points(self, points, runtime, render_mode, instance_id):
        if render_mode != "instance":
            raise ValueError(
                "runtime point selection is only valid for render_mode='instance'."
            )

        object_start = instance_id * runtime.object_nodes_per_instance
        object_end = object_start + runtime.object_nodes_per_instance
        controller_block_start = runtime.batch_size * runtime.object_nodes_per_instance
        controller_start = (
            controller_block_start
            + instance_id * runtime.controller_nodes_per_instance
        )
        controller_end = controller_start + runtime.controller_nodes_per_instance

        point_chunks = [points[object_start:object_end]]
        if runtime.controller_nodes_per_instance > 0:
            point_chunks.append(points[controller_start:controller_end])

        selected_points = torch.cat(point_chunks, dim=0)
        return selected_points - runtime.instance_offsets[instance_id]

    def _composite_batch_images_without_shadows(self, batch_rendering, overlay):
        if batch_rendering.ndim != 4 or batch_rendering.shape[1] != 4:
            raise ValueError(
                "batch image rendering expects [B, 4, H, W], got "
                f"{tuple(batch_rendering.shape)}"
            )

        images = batch_rendering.permute(0, 2, 3, 1).detach().clamp(0, 1)
        image_mask = torch.logical_and(
            (images[..., :3] != 1.0).any(dim=3),
            images[..., 3] > 100 / 255,
        )
        alpha = torch.where(
            image_mask[..., None],
            images[..., 3:4],
            torch.zeros_like(images[..., 3:4]),
        )
        frames = overlay.unsqueeze(0) * (1.0 - alpha) + images[..., :3] * alpha * 255.0
        return frames, image_mask

    def _resolve_batch_image_render_size(
        self,
        batch_image_resolution,
        native_width,
        native_height,
    ):
        if batch_image_resolution == "native":
            return int(native_width), int(native_height)
        if batch_image_resolution == "640x480":
            return 640, 480
        raise ValueError(
            "batch_image_resolution must be 'native' or '640x480'. "
            f"Received: {batch_image_resolution}"
        )

    def _scale_intrinsic_for_render_size(
        self,
        intrinsic,
        native_width,
        native_height,
        render_width,
        render_height,
    ):
        scaled = np.asarray(intrinsic, dtype=np.float32).copy()
        x_scale = float(render_width) / float(native_width)
        y_scale = float(render_height) / float(native_height)
        scaled[0, 0] *= x_scale
        scaled[0, 2] *= x_scale
        scaled[1, 1] *= y_scale
        scaled[1, 2] *= y_scale
        return scaled

    def _make_batch_image_grid(self, batch_frames, batch_grid_cols):
        batch_size = int(batch_frames.shape[0])
        if batch_size < 1:
            raise ValueError("batch image grid requires at least one frame.")
        if batch_frames.ndim != 4 or batch_frames.shape[-1] != 3:
            raise ValueError(
                "batch image grid expects [B, H, W, 3], got "
                f"{tuple(batch_frames.shape)}"
            )

        cols = (
            int(batch_grid_cols)
            if batch_grid_cols is not None
            else int(math.ceil(math.sqrt(batch_size)))
        )
        if cols < 1:
            raise ValueError("--batch_grid_cols must be a positive integer.")

        _, tile_height, tile_width, channels = batch_frames.shape
        rows = int(math.ceil(batch_size / float(cols)))
        grid = torch.full(
            (rows * tile_height, cols * tile_width, channels),
            255.0,
            device=batch_frames.device,
            dtype=batch_frames.dtype,
        )
        for batch_idx in range(batch_size):
            row = batch_idx // cols
            col = batch_idx % cols
            grid[
                row * tile_height : (row + 1) * tile_height,
                col * tile_width : (col + 1) * tile_width,
                :,
            ] = batch_frames[batch_idx]
        return grid

    def _write_headless_summary(self, output_dir, batch_size, component_times):
        frames_used_for_stats = len(component_times["total"])
        if frames_used_for_stats == 0:
            return {}

        total_frame_times = component_times["total"]
        total_time_seconds = sum(total_frame_times)
        average_batch_fps = frames_used_for_stats / total_time_seconds
        average_frame_time = float(np.mean(total_frame_times))
        average_throughput = batch_size * average_batch_fps

        print(f"\n=== Final Summary (averaged over {frames_used_for_stats} frames) ===")
        log_lines = [
            f"=== Final Summary (averaged over {frames_used_for_stats} frames) ===",
            f"Batch Size: {batch_size}",
            f"Average Batch FPS: {average_batch_fps:.2f}",
            f"Average Throughput (instances/s): {average_throughput:.2f}",
            f"Average Sim+LBS Total: {average_frame_time * 1000:.2f} ms",
        ]

        print(f"Batch Size: {batch_size}")
        print(f"Average Batch FPS: {average_batch_fps:.2f}")
        print(f"Average Throughput (instances/s): {average_throughput:.2f}")
        print(f"Average Sim+LBS Total: {average_frame_time * 1000:.2f} ms")

        for component_name in ["simulator", "full_motion_interpolation"]:
            component_times_list = component_times.get(component_name, [])
            if component_times_list:
                average_component_time = float(np.mean(component_times_list))
                time_share_percentage = (
                    average_component_time / average_frame_time
                ) * 100.0
                readable_name = perf_component_label(component_name)
                print(
                    f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)"
                )
                log_lines.append(
                    f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)"
                )

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            log_file_path = os.path.join(output_dir, "performance_summary.txt")
            with open(log_file_path, "w", encoding="utf-8") as log_file:
                log_file.write("\n".join(log_lines))

        return {
            "frames_used_for_stats": frames_used_for_stats,
            "batch_size": batch_size,
            "average_batch_fps": average_batch_fps,
            "average_throughput": average_throughput,
            "average_sim_lbs_total_ms": average_frame_time * 1000.0,
            "average_simulator_ms": float(np.mean(component_times["simulator"])) * 1000.0
            if component_times["simulator"]
            else None,
            "average_full_motion_interpolation_ms": float(
                np.mean(component_times["full_motion_interpolation"])
            )
            * 1000.0
            if component_times["full_motion_interpolation"]
            else None,
        }

    def _write_headless_ncu_profile_metrics(self, output_dir, metrics):
        if output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "ncu_profile_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as metrics_file:
            json.dump(metrics, metrics_file, indent=2)

    def _write_full_runtime_summary(
        self,
        output_dir,
        batch_size,
        render_mode,
        gaussian_render_mode,
        instance_id,
        component_times,
        batch_image_resolution="native",
        render_width=None,
        render_height=None,
        sim_force_mode=SIM_FORCE_MODE_GATHER,
    ):
        frames_used_for_stats = len(component_times["total"])
        total_frame_times = component_times["total"]
        total_time_seconds = sum(total_frame_times) if total_frame_times else 0.0
        average_fps = (
            frames_used_for_stats / total_time_seconds
            if frames_used_for_stats > 0 and total_time_seconds > 0.0
            else 0.0
        )
        average_frame_time = (
            float(np.mean(total_frame_times)) if total_frame_times else 0.0
        )
        average_throughput = batch_size * average_fps

        print(
            f"\n=== Final Summary (averaged over {frames_used_for_stats} frames) ==="
        )
        print(f"Batch Size: {batch_size}")
        print(f"Render Mode: {render_mode}")
        print(f"Gaussian Render Mode: {gaussian_render_mode}")
        print(f"Sim Force Mode: {sim_force_mode}")
        if instance_id is not None:
            print(f"Instance ID: {instance_id}")
        if render_width is not None and render_height is not None:
            print(f"Batch Image Resolution: {batch_image_resolution}")
            print(f"Render Size: {int(render_width)}x{int(render_height)}")
        print(f"Average FPS: {average_fps:.2f}")
        print(f"Average Throughput (instances/s): {average_throughput:.2f}")
        print(f"Average Total Frame Time: {average_frame_time * 1000:.2f} ms")

        log_lines = [
            f"=== Final Summary (averaged over {frames_used_for_stats} frames) ===",
            f"Batch Size: {batch_size}",
            f"Render Mode: {render_mode}",
            f"Gaussian Render Mode: {gaussian_render_mode}",
            f"Sim Force Mode: {sim_force_mode}",
        ]
        if instance_id is not None:
            log_lines.append(f"Instance ID: {instance_id}")
        if render_width is not None and render_height is not None:
            log_lines.append(f"Batch Image Resolution: {batch_image_resolution}")
            log_lines.append(f"Render Size: {int(render_width)}x{int(render_height)}")
        log_lines.extend(
            [
                f"Average FPS: {average_fps:.2f}",
                f"Average Throughput (instances/s): {average_throughput:.2f}",
                f"Average Total Frame Time: {average_frame_time * 1000:.2f} ms",
            ]
        )

        metrics = {
            "frames_used_for_stats": int(frames_used_for_stats),
            "batch_size": int(batch_size),
            "render_mode": render_mode,
            "gaussian_render_mode": gaussian_render_mode,
            "sim_force_mode": sim_force_mode,
            "instance_id": int(instance_id) if instance_id is not None else None,
            "batch_image_resolution": batch_image_resolution,
            "render_width": int(render_width) if render_width is not None else None,
            "render_height": int(render_height) if render_height is not None else None,
            "average_fps": float(average_fps),
            "average_throughput": float(average_throughput),
            "average_total_frame_time_ms": float(average_frame_time * 1000.0),
        }

        for component_name in [
            "simulator",
            "full_motion_interpolation",
            "rendering",
            "frame_compositing",
        ]:
            component_times_list = component_times.get(component_name, [])
            average_component_time = (
                float(np.mean(component_times_list)) if component_times_list else None
            )
            time_share_percentage = None
            if average_component_time is not None and average_frame_time > 0.0:
                time_share_percentage = (
                    average_component_time / average_frame_time
                ) * 100.0
                readable_name = perf_component_label(component_name)
                print(
                    f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)"
                )
                log_lines.append(
                    f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)"
                )
            metrics[f"average_{component_name}_ms"] = (
                float(average_component_time * 1000.0)
                if average_component_time is not None
                else None
            )
            metrics[f"average_{component_name}_share_pct"] = (
                float(time_share_percentage)
                if time_share_percentage is not None
                else None
            )

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            log_file_path = os.path.join(output_dir, "performance_summary.txt")
            with open(log_file_path, "w", encoding="utf-8") as log_file:
                log_file.write("\n".join(log_lines))

            metrics_path = os.path.join(output_dir, "performance_summary.json")
            with open(metrics_path, "w", encoding="utf-8") as metrics_file:
                json.dump(metrics, metrics_file, indent=2)

        return metrics

    def run_headless_sim_lbs(
        self,
        model_path,
        gs_path,
        output_dir=None,
        n_dup=0,
        ncu_profile_loop=False,
        ncu_profile_frame_stride=None,
        ncu_profile_max_frames=3,
        ncu_profile_nvtx_name="sim_lbs_profile_frame",
    ):
        if ncu_profile_loop:
            if ncu_profile_frame_stride is not None:
                ncu_profile_frame_stride = int(ncu_profile_frame_stride)
                if ncu_profile_frame_stride < 1:
                    raise ValueError(
                        "ncu_profile_frame_stride must be a positive integer. "
                        f"Received: {ncu_profile_frame_stride}"
                    )
            ncu_profile_max_frames = int(ncu_profile_max_frames)
            if ncu_profile_max_frames < 1:
                raise ValueError(
                    "ncu_profile_max_frames must be a positive integer. "
                    f"Received: {ncu_profile_max_frames}"
                )

        runtime = self._build_runtime_core(model_path, gs_path, n_dup=n_dup)
        measured_profile_frames = list(range(2, runtime.frame_len))
        if ncu_profile_loop and ncu_profile_frame_stride is not None:
            ncu_profile_frame_selection = "stride"
            ncu_profile_frame_indices_to_capture = measured_profile_frames[
                ::ncu_profile_frame_stride
            ]
        elif ncu_profile_loop:
            ncu_profile_frame_selection = "evenly_spaced"
            if len(measured_profile_frames) <= ncu_profile_max_frames:
                ncu_profile_frame_indices_to_capture = measured_profile_frames
            elif ncu_profile_max_frames == 1:
                middle_idx = round((len(measured_profile_frames) - 1) / 2)
                ncu_profile_frame_indices_to_capture = [
                    measured_profile_frames[middle_idx]
                ]
            else:
                last_idx = len(measured_profile_frames) - 1
                ncu_profile_frame_indices_to_capture = sorted(
                    {
                        measured_profile_frames[
                            round(i * last_idx / (ncu_profile_max_frames - 1))
                        ]
                        for i in range(ncu_profile_max_frames)
                    }
                )
        else:
            ncu_profile_frame_selection = None
            ncu_profile_frame_indices_to_capture = []
        ncu_profile_frame_indices_to_capture = set(
            int(frame_idx) for frame_idx in ncu_profile_frame_indices_to_capture
        )

        sim_timer = Timer("Simulator")
        interp_timer = Timer("Linear Blend Skinning")
        total_timer = Timer("Sim+LBS Total")
        component_times = {
            "simulator": [],
            "full_motion_interpolation": [],
            "total": [],
        }

        summary = {}
        profile_cuda = bool(ncu_profile_loop and torch.cuda.is_available())
        cuda_profiler = torch.cuda.cudart() if profile_cuda else None
        loop_memory_reset = False
        loop_peak_allocated_gb = None
        loop_peak_reserved_gb = None
        ncu_profiled_frame_indices = []
        try:
            while True:
                if profile_cuda and runtime.frame_count == 2 and not loop_memory_reset:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    loop_memory_reset = True

                profile_this_frame = (
                    profile_cuda
                    and runtime.frame_count in ncu_profile_frame_indices_to_capture
                )
                profiler_started = False
                nvtx_range_id = None

                try:
                    if profile_this_frame:
                        torch.cuda.synchronize()
                        nvtx_range_id = torch.cuda.nvtx.range_start(
                            ncu_profile_nvtx_name
                        )
                        cuda_profiler.cudaProfilerStart()
                        profiler_started = True
                        ncu_profiled_frame_indices.append(int(runtime.frame_count))

                    total_timer.start()

                    sim_timer.start()
                    self.simulator.set_controller_interactive(
                        runtime.prev_target, runtime.current_target
                    )

                    if self.simulator.object_collision_flag:
                        self.simulator.update_collision_graph()
                    wp.capture_launch(self.simulator.forward_graph)
                    wp.synchronize()

                    x = wp.to_torch(
                        self.simulator.wp_states[-1].wp_x, requires_grad=False
                    )
                    self.simulator.set_init_state(
                        self.simulator.wp_states[-1].wp_x,
                        self.simulator.wp_states[-1].wp_v,
                    )
                    sim_time = sim_timer.stop()

                    with torch.no_grad():
                        interp_timer.start()
                        current_pos, current_rot = lbs_with_rotation_reuse(
                            current_mass_nodes=x,
                            cache=runtime.rotation_cache,
                        )
                        interp_time = interp_timer.stop()
                        runtime.gaussians._xyz = current_pos
                        runtime.gaussians._rotation = current_rot

                    total_time = total_timer.stop()
                finally:
                    if profiler_started:
                        try:
                            torch.cuda.synchronize()
                            cuda_profiler.cudaProfilerStop()
                        except Exception as exc:
                            print(f"[WARN] Failed to stop CUDA profiler: {exc}")
                    if nvtx_range_id is not None:
                        try:
                            torch.cuda.nvtx.range_end(nvtx_range_id)
                        except Exception as exc:
                            print(f"[WARN] Failed to end NVTX range: {exc}")

                if runtime.frame_count > 1:
                    component_times["simulator"].append(sim_time)
                    component_times["full_motion_interpolation"].append(interp_time)
                    component_times["total"].append(total_time)

                runtime.prev_x = x.clone()
                runtime.frame_count += 1
                runtime.prev_target = runtime.current_target

                if runtime.frame_count < runtime.frame_len:
                    runtime.current_target = self.batch_controller_points[
                        runtime.frame_count
                    ]
                else:
                    print("Reached end of recorded control sequence")
                    break
        finally:
            summary = self._write_headless_summary(
                output_dir=output_dir,
                batch_size=runtime.batch_size,
                component_times=component_times,
            )
            if ncu_profile_loop:
                if profile_cuda and loop_memory_reset:
                    torch.cuda.synchronize()
                    loop_peak_allocated_gb = torch.cuda.max_memory_allocated() / (
                        1024**3
                    )
                    loop_peak_reserved_gb = torch.cuda.max_memory_reserved() / (
                        1024**3
                    )
                self._write_headless_ncu_profile_metrics(
                    output_dir=output_dir,
                    metrics={
                        "ncu_profile_loop": True,
                        "ncu_profile_frame_stride": int(ncu_profile_frame_stride)
                        if ncu_profile_frame_stride is not None
                        else None,
                        "ncu_profile_max_frames": int(ncu_profile_max_frames),
                        "ncu_profile_frame_selection": ncu_profile_frame_selection,
                        "ncu_profile_nvtx_name": ncu_profile_nvtx_name,
                        "ncu_profiled_frame_indices": ncu_profiled_frame_indices,
                        "ncu_num_profiled_frames": len(ncu_profiled_frame_indices),
                        "loop_peak_allocated_gb": loop_peak_allocated_gb,
                        "loop_peak_reserved_gb": loop_peak_reserved_gb,
                    },
                )
        return summary

    def run_batched_full_runtime(
        self,
        model_path,
        gs_path,
        output_dir,
        window,
        cuda_ctx,
        batch_size=1,
        num_views=1,
        render_mode="batch_images",
        instance_id=None,
        gaussian_render_mode="shared_template",
        save_video=False,
        save_batch_images=False,
        save_batch_grid=False,
        display_batch_grid=False,
        batch_image_resolution="native",
        batch_grid_cols=None,
        profile_render_components=False,
        sim_force_mode=SIM_FORCE_MODE_GATHER,
    ):
        gaussian_render_mode = normalize_gaussian_render_mode(gaussian_render_mode)
        if sim_force_mode not in SIM_FORCE_MODES:
            raise ValueError(
                f"sim_force_mode must be one of {SIM_FORCE_MODES}. "
                f"Received: {sim_force_mode}"
            )
        if output_dir is None:
            raise ValueError("output_dir is required for batched full-runtime runs.")
        if batch_size < 1:
            raise ValueError(
                f"batch_size must be a positive integer. Received: {batch_size}"
            )
        instance_id = self._validate_batched_render_request(
            render_mode=render_mode,
            instance_id=instance_id,
            batch_size=batch_size,
        )
        if render_mode == "batch_images":
            if num_views != 1:
                raise ValueError(
                    "render_mode='batch_images' currently supports num_views=1 only."
                )
            if batch_image_resolution not in ("native", "640x480"):
                raise ValueError(
                    "batch_image_resolution must be 'native' or '640x480'. "
                    f"Received: {batch_image_resolution}"
                )
            if batch_grid_cols is not None and int(batch_grid_cols) < 1:
                raise ValueError("batch_grid_cols must be a positive integer.")
        elif batch_image_resolution != "native":
            raise ValueError(
                "batch_image_resolution='640x480' can only be used with "
                "render_mode='batch_images'."
            )
        elif save_batch_images or save_batch_grid or display_batch_grid:
            raise ValueError(
                "--save_batch_images, --save_batch_grid, and --display_batch_grid "
                "can only be used with render_mode='batch_images'."
            )

        print(f"[BatchedRender] gaussian_render_mode={gaussian_render_mode}")
        print(f"[BatchedRender] sim_force_mode={sim_force_mode}")
        if batch_size == 1:
            print("[BatchedRender] single-instance/no-offset/no-camera-change")
        elif render_mode == "batch_images":
            print(
                "[BatchedRender] batch-images-mode/per-instance-calibrated-output/"
                f"{gaussian_render_mode}/no-shadows"
            )
        else:
            print(
                "[BatchedRender] instance-mode/diagonal-sim-offset/"
                "output-recenter/no-camera-change"
            )
        profile_path = os.path.join(output_dir, "render_component_profile.json")
        if not profile_render_components and os.path.exists(profile_path):
            os.remove(profile_path)

        runtime = self._build_runtime_core(
            model_path,
            gs_path,
            n_dup=batch_size - 1,
            gaussian_render_mode=gaussian_render_mode,
            force_shared_batched_gaussians=(render_mode == "batch_images"),
            sim_force_mode=sim_force_mode,
        )
        print("[BatchedRender] Runtime core built")

        render_backend = self._load_render_backend()
        render_backend.glfw.make_context_current(window)

        native_width, native_height = [int(value) for value in cfg.WH]
        width, height = self._resolve_batch_image_render_size(
            batch_image_resolution=batch_image_resolution,
            native_width=native_width,
            native_height=native_height,
        )
        display_width = width
        display_height = height
        if display_batch_grid:
            framebuffer_width, framebuffer_height = render_backend.glfw.get_framebuffer_size(
                window
            )
            if framebuffer_width > 0 and framebuffer_height > 0:
                display_width = int(framebuffer_width)
                display_height = int(framebuffer_height)
        available_views = min(len(cfg.intrinsics), len(cfg.w2cs))
        if num_views < 1 or num_views > available_views:
            raise ValueError(
                f"num_views must be between 1 and {available_views}. Received: {num_views}"
            )

        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        intrinsic = self._scale_intrinsic_for_render_size(
            cfg.intrinsics[0],
            native_width=native_width,
            native_height=native_height,
            render_width=width,
            render_height=height,
        )
        w2c = cfg.w2cs[0]
        view, K_cuda = self._create_gs_view(w2c, intrinsic, height, width)
        render_views = [(0, view)]
        for view_idx in range(1, num_views):
            extra_intrinsic = self._scale_intrinsic_for_render_size(
                cfg.intrinsics[view_idx],
                native_width=native_width,
                native_height=native_height,
                render_width=width,
                render_height=height,
            )
            extra_view, _ = self._create_gs_view(
                cfg.w2cs[view_idx], extra_intrinsic, height, width
            )
            render_views.append((view_idx, extra_view))

        overlay = render_backend.cv2.imread(cfg.bg_img_path)
        overlay = render_backend.cv2.cvtColor(
            overlay, render_backend.cv2.COLOR_BGR2RGB
        )
        if overlay.shape[1] != width or overlay.shape[0] != height:
            overlay = render_backend.cv2.resize(
                overlay,
                (width, height),
                interpolation=render_backend.cv2.INTER_LINEAR,
            )
        overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)
        assert overlay.shape[0] == height and overlay.shape[1] == width, (
            f"overlay {tuple(overlay.shape)} != (H,W,3)=({height},{width},3)"
        )

        lights = torch.tensor(
            [[0, 0, -3], [1, 0.5, -2], [-3, -0.5, -5]],
            device=cfg.device,
            dtype=torch.float32,
        )
        coeffs = torch.tensor(
            [0.95, 0.97, 0.98], device=cfg.device, dtype=torch.float32
        )
        w2c_cuda = torch.tensor(w2c, dtype=torch.float32, device=cfg.device)
        coeffs_b = coeffs.view(-1, 1, 1)
        w2c_T = w2c_cuda.T.contiguous()
        intrinsic_T = K_cuda.T.contiguous()
        inv_Lz = 1.0 / lights[:, 2]
        bytes_per_pixel = 4
        pbo_size = display_width * display_height * bytes_per_pixel
        row_pitch = display_width * bytes_per_pixel

        tex = None
        pbo = None
        reg = None
        prog = None
        vao = None
        summary = {}

        try:
            print("[BatchedRender] GL interop setup start")
            tex = render_backend.gl.glGenTextures(1)
            render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
            render_backend.gl.glTexParameteri(
                render_backend.gl.GL_TEXTURE_2D,
                render_backend.gl.GL_TEXTURE_MIN_FILTER,
                render_backend.gl.GL_NEAREST,
            )
            render_backend.gl.glTexParameteri(
                render_backend.gl.GL_TEXTURE_2D,
                render_backend.gl.GL_TEXTURE_MAG_FILTER,
                render_backend.gl.GL_NEAREST,
            )
            render_backend.gl.glTexImage2D(
                render_backend.gl.GL_TEXTURE_2D,
                0,
                render_backend.gl.GL_RGBA8,
                display_width,
                display_height,
                0,
                render_backend.gl.GL_RGBA,
                render_backend.gl.GL_UNSIGNED_BYTE,
                None,
            )
            render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, 0)

            pbo = render_backend.gl.glGenBuffers(1)
            render_backend.gl.glBindBuffer(
                render_backend.gl.GL_PIXEL_UNPACK_BUFFER, pbo
            )
            render_backend.gl.glBufferData(
                render_backend.gl.GL_PIXEL_UNPACK_BUFFER,
                pbo_size,
                None,
                render_backend.gl.GL_STREAM_DRAW,
            )
            render_backend.gl.glBindBuffer(
                render_backend.gl.GL_PIXEL_UNPACK_BUFFER, 0
            )
            cuda_logic_error = getattr(
                render_backend.cuda_driver, "LogicError", RuntimeError
            )
            try:
                reg = render_backend.RegisteredBuffer(
                    int(pbo), render_backend.graphics_map_flags.WRITE_DISCARD
                )
            except cuda_logic_error as exc:
                if pbo is not None:
                    render_backend.gl.glDeleteBuffers(1, [pbo])
                    pbo = None
                raise RuntimeError(
                    "Failed to register the OpenGL pixel buffer with CUDA in "
                    "batched full-runtime rendering. This usually indicates a "
                    "CUDA/OpenGL interop initialization mismatch in the batched "
                    "render path."
                ) from exc

            vertex_shader_source = """
            #version 330 core
            out vec2 uv; const vec2 V[4]=vec2[4](vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(1,1));
            const vec2 T[4]=vec2[4](vec2(0,0),vec2(1,0),vec2(0,1),vec2(1,1));
            void main(){ gl_Position=vec4(V[gl_VertexID],0,1); uv=T[gl_VertexID]; }
            """
            fragment_shader_source = """
            #version 330 core
            in vec2 uv; out vec4 frag; uniform sampler2D uTex;
            void main(){ frag = texture(uTex, vec2(uv.x,1.0 - uv.y)); }
            """

            def _compile(kind, src):
                shader_id = render_backend.gl.glCreateShader(kind)
                render_backend.gl.glShaderSource(shader_id, src)
                render_backend.gl.glCompileShader(shader_id)
                if not render_backend.gl.glGetShaderiv(
                    shader_id, render_backend.gl.GL_COMPILE_STATUS
                ):
                    raise RuntimeError(
                        render_backend.gl.glGetShaderInfoLog(shader_id).decode()
                    )
                return shader_id

            prog = render_backend.gl.glCreateProgram()
            render_backend.gl.glAttachShader(
                prog,
                _compile(render_backend.gl.GL_VERTEX_SHADER, vertex_shader_source),
            )
            render_backend.gl.glAttachShader(
                prog,
                _compile(render_backend.gl.GL_FRAGMENT_SHADER, fragment_shader_source),
            )
            render_backend.gl.glLinkProgram(prog)
            if not render_backend.gl.glGetProgramiv(
                prog, render_backend.gl.GL_LINK_STATUS
            ):
                raise RuntimeError(
                    render_backend.gl.glGetProgramInfoLog(prog).decode()
                )
            render_backend.gl.glUseProgram(prog)
            render_backend.gl.glUniform1i(
                render_backend.gl.glGetUniformLocation(prog, "uTex"), 0
            )
            render_backend.gl.glUseProgram(0)
            vao = render_backend.gl.glGenVertexArrays(1)
            render_backend.gl.glBindVertexArray(vao)

            pbo_stream = render_backend.cuda_driver.Stream()
            cpy2d = render_backend.cuda_driver.Memcpy2D()
            cpy2d.src_pitch = row_pitch
            cpy2d.dst_pitch = row_pitch
            cpy2d.width_in_bytes = row_pitch
            cpy2d.height = display_height
            print("[BatchedRender] GL interop setup complete")

            frame_rgba = torch.empty(
                (display_height, display_width, 4),
                dtype=torch.uint8,
                device=cfg.device,
            )
            frame = torch.empty(
                (display_height, display_width, 3),
                dtype=overlay.dtype,
                device=cfg.device,
            )
            rgb_temp = torch.empty(
                (height, width, 3), dtype=overlay.dtype, device=cfg.device
            )

            save_frame_artifacts = save_video or save_batch_images or save_batch_grid
            eval_render_paths = {}
            view_render_path = None
            batch_image_paths = []
            batch_grid_path = None
            if save_frame_artifacts:
                import torchvision

                if render_mode == "batch_images":
                    if save_video:
                        view_render_path = os.path.join(output_dir, "output")
                        os.makedirs(view_render_path, exist_ok=True)
                    if save_batch_images:
                        batch_image_root = os.path.join(output_dir, "batch_images")
                        batch_image_paths = [
                            os.path.join(batch_image_root, f"instance_{idx:03d}")
                            for idx in range(runtime.batch_size)
                        ]
                        for render_path in batch_image_paths:
                            os.makedirs(render_path, exist_ok=True)
                    if save_batch_grid:
                        batch_grid_path = os.path.join(output_dir, "output_grid")
                        os.makedirs(batch_grid_path, exist_ok=True)
                else:
                    eval_render_paths = {
                        view_idx: os.path.join(output_dir, str(view_idx))
                        for view_idx in range(num_views)
                    }
                    view_render_path = os.path.join(output_dir, "output")
                    os.makedirs(view_render_path, exist_ok=True)
                    for render_path in eval_render_paths.values():
                        os.makedirs(render_path, exist_ok=True)

            sim_timer = Timer("Simulator")
            interp_timer = Timer("Linear Blend Skinning")
            render_timer = Timer("Rendering")
            frame_timer = Timer("Frame Compositing")
            total_timer = Timer("Total Loop")

            component_times = {
                "simulator": [],
                "full_motion_interpolation": [],
                "rendering": [],
                "frame_compositing": [],
                "total": [],
            }
            render_component_profiler = (
                RenderComponentProfiler()
                if profile_render_components
                else None
            )

            gaussians = runtime.gaussians
            render_gaussians = runtime.render_gaussians
            if render_mode == "instance" and runtime.batch_size > 1:
                render_gaussians = build_instance_selective_render_view(
                    runtime.gaussians,
                    instance_id,
                    gaussian_render_mode=gaussian_render_mode,
                )
            elif render_mode == "batch_images":
                render_gaussians = build_batch_images_render_view(
                    runtime.gaussians,
                    gaussian_render_mode=gaussian_render_mode,
                )

            frame_count = runtime.frame_count
            prev_target = runtime.prev_target
            current_target = runtime.current_target
            prev_x = runtime.prev_x

            while True:
                total_timer.start()

                sim_timer.start()
                self.simulator.set_controller_interactive(prev_target, current_target)

                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()
                wp.capture_launch(self.simulator.forward_graph)
                wp.synchronize()

                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )
                sim_time = sim_timer.stop()

                if frame_count > 1:
                    component_times["simulator"].append(sim_time)

                render_timer.start()
                profile_this_frame = bool(profile_render_components and frame_count > 1)
                frame_profiler = (
                    render_component_profiler
                    if render_component_profiler is not None
                    and profile_this_frame
                    else None
                )
                if frame_profiler is not None:
                    if render_mode == "instance":
                        total_render_gaussians = int(runtime.gaussians_per_instance)
                    else:
                        total_render_gaussians = int(
                            runtime.batch_size * runtime.gaussians_per_instance
                        )
                    frame_profiler.begin_frame(
                        total_gaussians=total_render_gaussians,
                        gaussians_per_instance=int(runtime.gaussians_per_instance),
                    )
                render_profile_success = False
                try:
                    results = render_backend.render_gaussian(
                        view,
                        render_gaussians,
                        None,
                        background,
                        profiler=frame_profiler,
                    )
                    render_profile_success = True
                finally:
                    if frame_profiler is not None:
                        if render_profile_success and profile_this_frame:
                            frame_profiler.end_frame()
                        else:
                            frame_profiler.discard_frame()
                rendering = results["render"]
                additional_renderings = {}
                if render_mode != "batch_images" and save_video and num_views > 1:
                    for view_idx, extra_view in render_views[1:]:
                        additional_renderings[view_idx] = render_backend.render_gaussian(
                            extra_view, render_gaussians, None, background
                        )["render"]
                if render_mode != "batch_images":
                    image = rendering.permute(1, 2, 0).detach()
                render_time = render_timer.stop()

                if frame_count > 1:
                    component_times["rendering"].append(render_time)

                frame_timer.start()
                if render_mode == "batch_images":
                    batch_frames, _ = self._composite_batch_images_without_shadows(
                        rendering,
                        overlay,
                    )
                    batch_grid = None
                    if display_batch_grid:
                        batch_grid = self._make_batch_image_grid(
                            batch_frames,
                            batch_grid_cols=batch_grid_cols,
                        )
                        display_grid = batch_grid
                        if (
                            display_grid.shape[0] != display_height
                            or display_grid.shape[1] != display_width
                        ):
                            display_grid = F.interpolate(
                                display_grid.permute(2, 0, 1).unsqueeze(0),
                                size=(display_height, display_width),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0).permute(1, 2, 0)
                        frame.copy_(display_grid)
                    else:
                        frame.copy_(batch_frames[0])
                else:
                    frame.copy_(overlay)

                    image.clamp_(0, 1)
                    image_mask = torch.logical_and(
                        (image != 1.0).any(dim=2), image[:, :, 3] > 100 / 255
                    )
                    image[..., 3].masked_fill_(~image_mask, 0.0)

                    alpha = image[..., 3:4]
                    torch.mul(image[..., :3], alpha, out=rgb_temp)
                    rgb_temp.mul_(255.0)
                    frame.mul_(1.0 - alpha).add_(rgb_temp)

                    composite_points = self._select_runtime_points(
                        points=x,
                        runtime=runtime,
                        render_mode=render_mode,
                        instance_id=instance_id,
                    )
                    masks = get_shadow_masks_batched_downsampled(
                        points=composite_points,
                        intrinsic_T=intrinsic_T,
                        w2c_T=w2c_T,
                        W=width,
                        H=height,
                        image_mask=image_mask,
                        lights=lights,
                        inv_Lz=inv_Lz,
                        kernel_size=7,
                        scale=2,
                        use_half=False,
                        upsample_mode="bilinear",
                        post_blur=False,
                    )

                    attenuation = torch.prod(
                        1.0 - masks.to(frame.dtype) + masks.to(frame.dtype) * coeffs_b,
                        dim=0,
                    )
                    frame.mul_(attenuation.unsqueeze(-1))

                frame_u8 = frame.clamp(0, 255).to(torch.uint8)
                frame_rgba[:, :, :3] = frame_u8
                frame_rgba[:, :, 3] = 255
                torch.cuda.current_stream().synchronize()

                mapping = reg.map()
                try:
                    ptr, _ = mapping.device_ptr_and_size()
                    cpy2d.set_src_device(frame_rgba.data_ptr())
                    cpy2d.set_dst_device(ptr)
                    cpy2d(pbo_stream)
                    pbo_stream.synchronize()
                finally:
                    mapping.unmap()

                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
                render_backend.gl.glBindBuffer(
                    render_backend.gl.GL_PIXEL_UNPACK_BUFFER, pbo
                )
                render_backend.gl.glTexSubImage2D(
                    render_backend.gl.GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    display_width,
                    display_height,
                    render_backend.gl.GL_RGBA,
                    render_backend.gl.GL_UNSIGNED_BYTE,
                    None,
                )
                render_backend.gl.glBindBuffer(
                    render_backend.gl.GL_PIXEL_UNPACK_BUFFER, 0
                )

                render_backend.gl.glViewport(0, 0, display_width, display_height)
                render_backend.gl.glDisable(render_backend.gl.GL_DEPTH_TEST)
                render_backend.gl.glClear(render_backend.gl.GL_COLOR_BUFFER_BIT)
                render_backend.gl.glUseProgram(prog)
                render_backend.gl.glActiveTexture(render_backend.gl.GL_TEXTURE0)
                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
                render_backend.gl.glDrawArrays(
                    render_backend.gl.GL_TRIANGLE_STRIP, 0, 4
                )
                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, 0)
                render_backend.gl.glUseProgram(0)

                render_backend.glfw.swap_buffers(window)
                render_backend.glfw.poll_events()

                frame_comp_time = frame_timer.stop()
                if frame_count > 1:
                    component_times["frame_compositing"].append(frame_comp_time)

                should_prepare_next_frame = frame_count + 1 < runtime.frame_len
                if prev_x is not None and should_prepare_next_frame:
                    with torch.no_grad():
                        interp_timer.start()
                        current_pos, current_rot = lbs_with_rotation_reuse(
                            current_mass_nodes=x,
                            cache=runtime.rotation_cache,
                        )
                        interp_time = interp_timer.stop()
                        gaussians._xyz = current_pos
                        gaussians._rotation = current_rot

                    if frame_count > 1:
                        component_times["full_motion_interpolation"].append(
                            interp_time
                        )

                prev_x = x.clone()

                total_time = total_timer.stop()
                if frame_count > 1:
                    component_times["total"].append(total_time)

                if save_frame_artifacts:
                    if render_mode == "batch_images":
                        if save_video:
                            torchvision.utils.save_image(
                                frame.permute(2, 0, 1).float() / 255.0,
                                os.path.join(view_render_path, f"{frame_count:05d}.png"),
                            )
                        if save_batch_images:
                            for batch_idx, render_path in enumerate(batch_image_paths):
                                torchvision.utils.save_image(
                                    batch_frames[batch_idx]
                                    .permute(2, 0, 1)
                                    .float()
                                    / 255.0,
                                    os.path.join(
                                        render_path, f"{frame_count:05d}.png"
                                    ),
                                )
                        if save_batch_grid:
                            if batch_grid is None:
                                batch_grid = self._make_batch_image_grid(
                                    batch_frames,
                                    batch_grid_cols=batch_grid_cols,
                                )
                            torchvision.utils.save_image(
                                batch_grid.permute(2, 0, 1).float() / 255.0,
                                os.path.join(
                                    batch_grid_path, f"{frame_count:05d}.png"
                                ),
                            )
                    else:
                        torchvision.utils.save_image(
                            rendering,
                            os.path.join(
                                eval_render_paths[0], f"{frame_count:05d}.png"
                            ),
                        )
                        for view_idx, extra_rendering in additional_renderings.items():
                            torchvision.utils.save_image(
                                extra_rendering,
                                os.path.join(
                                    eval_render_paths[view_idx],
                                    f"{frame_count:05d}.png",
                                ),
                            )
                        composite_rgb = frame_rgba[:, :, :3]
                        composite_rgb = composite_rgb.permute(2, 0, 1).float() / 255.0
                        torchvision.utils.save_image(
                            composite_rgb,
                            os.path.join(view_render_path, f"{frame_count:05d}.png"),
                        )

                frame_count += 1
                prev_target = current_target
                if frame_count < runtime.frame_len:
                    current_target = self.batch_controller_points[frame_count]
                else:
                    print("Reached end of recorded control sequence")
                    break
        finally:
            summary = self._write_full_runtime_summary(
                output_dir=output_dir,
                batch_size=runtime.batch_size if "runtime" in locals() else batch_size,
                render_mode=render_mode,
                gaussian_render_mode=gaussian_render_mode,
                instance_id=instance_id,
                component_times=component_times
                if "component_times" in locals()
                else {
                    "simulator": [],
                    "full_motion_interpolation": [],
                    "rendering": [],
                    "frame_compositing": [],
                    "total": [],
                },
                batch_image_resolution=batch_image_resolution,
                render_width=width if "width" in locals() else None,
                render_height=height if "height" in locals() else None,
                sim_force_mode=runtime.sim_force_mode
                if "runtime" in locals()
                else sim_force_mode,
            )
            if profile_render_components and "render_component_profiler" in locals():
                render_component_profiler.write_json(
                    output_dir=output_dir,
                    metadata={
                        "batch_size": int(
                            runtime.batch_size if "runtime" in locals() else batch_size
                        ),
                        "gaussian_render_mode": gaussian_render_mode,
                        "sim_force_mode": runtime.sim_force_mode
                        if "runtime" in locals()
                        else sim_force_mode,
                        "render_mode": render_mode,
                        "instance_id": int(instance_id)
                        if instance_id is not None
                        else None,
                        "batch_image_resolution": batch_image_resolution,
                        "render_width": int(width) if "width" in locals() else None,
                        "render_height": int(height) if "height" in locals() else None,
                    },
                )

            if reg is not None:
                reg.unregister()
            if prog is not None:
                render_backend.gl.glDeleteProgram(prog)
            if tex is not None:
                render_backend.gl.glDeleteTextures([tex])
            if pbo is not None:
                render_backend.gl.glDeleteBuffers(1, [pbo])
            if vao is not None:
                render_backend.gl.glDeleteVertexArrays(1, [vao])
            cuda_ctx.pop()

        return summary


    #this is basically baseline + rendering (to verify correctness)        
    def interactive_playground(
        self,
        model_path,
        gs_path,
        output_dir=None,
        window=None,
        cuda_ctx=None,
        n_dup=0,
        save_eval_artifacts=False,
    ):
        runtime = self._build_runtime_core(model_path, gs_path, n_dup=n_dup)
        gaussians = runtime.gaussians
        render_gaussians = runtime.render_gaussians
        rotation_cache = runtime.rotation_cache
        prev_x = runtime.prev_x
        prev_target = runtime.prev_target
        current_target = runtime.current_target
        frame_count = runtime.frame_count
        render_backend = self._load_render_backend()

        ########add visualization initialization
        render_backend.glfw.make_context_current(window)
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[0]
        w2c = cfg.w2cs[0]

        use_white_background = True  # set to True for white background
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        view, K_cuda = self._create_gs_view(w2c, intrinsic, height, width)
        image_path = cfg.bg_img_path
        overlay = render_backend.cv2.imread(image_path)
        overlay = render_backend.cv2.cvtColor(overlay, render_backend.cv2.COLOR_BGR2RGB)
        overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)
        assert overlay.shape[0] == height and overlay.shape[1] == width, \
            f"overlay {tuple(overlay.shape)} != (H,W,3)=({height},{width},3)"

        lights = torch.tensor([[0, 0, -3],
                    [1, 0.5, -2],
                    [-3, -0.5, -5]], device=cfg.device, dtype=torch.float32)
        coeffs = torch.tensor([0.95, 0.97, 0.98], device=cfg.device, dtype=torch.float32)
        #K_cuda   = torch.tensor(intrinsic, dtype=torch.float32, device=cfg.device)
        w2c_cuda = torch.tensor(w2c, dtype=torch.float32, device=cfg.device)
        coeffs_b = coeffs.view(-1, 1, 1)
        w2c_T = w2c_cuda.T.contiguous()
        intrinsic_T = K_cuda.T.contiguous()
        inv_Lz = 1.0 / lights[:, 2] 
        BYTES_PER_PIXEL = 4
        pbo_size = width * height * BYTES_PER_PIXEL
        row_pitch = width * BYTES_PER_PIXEL  # <-- define this before using Memcpy2D

        # Texture
        tex = render_backend.gl.glGenTextures(1)
        render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
        render_backend.gl.glTexParameteri(render_backend.gl.GL_TEXTURE_2D, render_backend.gl.GL_TEXTURE_MIN_FILTER, render_backend.gl.GL_NEAREST)
        render_backend.gl.glTexParameteri(render_backend.gl.GL_TEXTURE_2D, render_backend.gl.GL_TEXTURE_MAG_FILTER, render_backend.gl.GL_NEAREST)
        render_backend.gl.glTexImage2D(render_backend.gl.GL_TEXTURE_2D, 0, render_backend.gl.GL_RGBA8, width, height, 0, render_backend.gl.GL_RGBA, render_backend.gl.GL_UNSIGNED_BYTE, None)
        render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, 0)

        # PBO
        pbo = render_backend.gl.glGenBuffers(1)
        render_backend.gl.glBindBuffer(render_backend.gl.GL_PIXEL_UNPACK_BUFFER, pbo)
        render_backend.gl.glBufferData(render_backend.gl.GL_PIXEL_UNPACK_BUFFER, pbo_size, None, render_backend.gl.GL_STREAM_DRAW)
        render_backend.gl.glBindBuffer(render_backend.gl.GL_PIXEL_UNPACK_BUFFER, 0)
        reg = render_backend.RegisteredBuffer(
            int(pbo), render_backend.graphics_map_flags.WRITE_DISCARD
        )

        # 4) Tiny fullscreen shader (one quad, no VAO needed)
        VS = """
        #version 330 core
        out vec2 uv; const vec2 V[4]=vec2[4](vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(1,1));
        const vec2 T[4]=vec2[4](vec2(0,0),vec2(1,0),vec2(0,1),vec2(1,1));
        void main(){ gl_Position=vec4(V[gl_VertexID],0,1); uv=T[gl_VertexID]; }
        """
        FS = """
        #version 330 core
        in vec2 uv; out vec4 frag; uniform sampler2D uTex;
        void main(){ frag = texture(uTex, vec2(uv.x,1.0 - uv.y)); }
        """
        def _compile(kind, src):
            sid = render_backend.gl.glCreateShader(kind); render_backend.gl.glShaderSource(sid, src); render_backend.gl.glCompileShader(sid)
            if not render_backend.gl.glGetShaderiv(sid, render_backend.gl.GL_COMPILE_STATUS):
                raise RuntimeError(render_backend.gl.glGetShaderInfoLog(sid).decode())
            return sid
        
        prog = render_backend.gl.glCreateProgram()
        render_backend.gl.glAttachShader(prog, _compile(render_backend.gl.GL_VERTEX_SHADER, VS))
        render_backend.gl.glAttachShader(prog, _compile(render_backend.gl.GL_FRAGMENT_SHADER, FS))
        render_backend.gl.glLinkProgram(prog)
        if not render_backend.gl.glGetProgramiv(prog, render_backend.gl.GL_LINK_STATUS):
            raise RuntimeError(render_backend.gl.glGetProgramInfoLog(prog).decode())
        render_backend.gl.glUseProgram(prog); render_backend.gl.glUniform1i(render_backend.gl.glGetUniformLocation(prog, "uTex"), 0); render_backend.gl.glUseProgram(0)
        vao = render_backend.gl.glGenVertexArrays(1)
        render_backend.gl.glBindVertexArray(vao)


        # Reuse Stream
        pbo_stream = render_backend.cuda_driver.Stream()

        cpy2d = render_backend.cuda_driver.Memcpy2D()
        cpy2d.src_pitch = row_pitch
        cpy2d.dst_pitch = row_pitch
        cpy2d.width_in_bytes = row_pitch
        cpy2d.height = height

        # These could be pre-allocated once:
        frame_rgba = torch.empty((height, width, 4), dtype=torch.uint8, device=cfg.device)
        frame = torch.empty_like(overlay)
        rgb_temp = torch.empty((height, width, 3), dtype=overlay.dtype, device=cfg.device)  # Add this

        #for saving output videos
        if output_dir and save_eval_artifacts:
            eval_render_path = os.path.join(output_dir, "0")
            view_render_path = os.path.join(output_dir, "output")
            os.makedirs(view_render_path, exist_ok=True)
            os.makedirs(eval_render_path, exist_ok=True)
            #traj_save_path = os.path.join(eval_image_path, "inference.pkl")


        #############end of visualization initialization

        #timer initialization code
        sim_timer = Timer("Simulator")
        interp_timer = Timer("Linear Blend Skinning")
        render_timer = Timer("Rendering")
        frame_timer = Timer("Frame Compositing")
        total_timer = Timer("Total Loop")

        # Performance stats
        fps_history = []
        component_times = {
            "simulator": [],
            "full_motion_interpolation": [],
            "rendering": [],
            "frame_compositing": [],
            "total": [],
        }
        #uncomment this for generating qualtiy comparision data
        # traj_frames_cpu = []  # list of (N,3) CPU tensors
        # x0 = wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
        # traj_frames_cpu.append(x0.detach().cpu())
        try:
            while True:

                total_timer.start()

                # 1. Simulator step
                #uncomment for correct trajectory generation
                # if frame_count == 0:
                #     x = wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
                # else:
                #     sim_timer.start()

                #     self.simulator.set_controller_interactive(prev_target, current_target)

                #     if self.simulator.object_collision_flag:
                #         self.simulator.update_collision_graph()
                #     wp.capture_launch(self.simulator.forward_graph)
                #     wp.synchronize()

                #     x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                #     #uncomment to generate quality comparison data
                #     traj_frames_cpu.append(x.detach().cpu())
                
                #     # Set the intial state for the next step
                #     self.simulator.set_init_state(
                #         self.simulator.wp_states[-1].wp_x,
                #         self.simulator.wp_states[-1].wp_v,
                #     )

                #     sim_time = sim_timer.stop()

                #     #pyh ignore first two frame since 1st frame is skewed during to rendering initialization and 2nd frame is skewed toward getting neighboruing weights
                #     if frame_count > 1:
                #         component_times["simulator"].append(sim_time)

                #uncomment for performance                
                sim_timer.start()

                self.simulator.set_controller_interactive(prev_target, current_target)

                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()
                wp.capture_launch(self.simulator.forward_graph)
                wp.synchronize()

                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                # if frame_count == 0:
                #     max_err = (x.detach() - wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)).abs().max().item()
                #     print(f"[traj] frame0 post-step max |Δx| = {max_err:e}")
                # #uncomment to generate quality comparison data
                # if frame_count > 0:
                #     traj_frames_cpu.append(x.detach().cpu())
            
                # Set the intial state for the next step
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )

                sim_time = sim_timer.stop()

                #pyh ignore first two frame since 1st frame is skewed during to rendering initialization and 2nd frame is skewed toward getting neighboruing weights
                if frame_count > 1:
                    component_times["simulator"].append(sim_time)

                # 3. Rendering
                render_timer.start()

                # render with gaussians and paste the image on top of the frame
                results = render_backend.render_gaussian(
                    view, render_gaussians, None, background
                )
                rendering = results["render"]  # (4, H, W)
                image = rendering.permute(1, 2, 0).detach()

                render_time = render_timer.stop()
                if frame_count > 1:
                    component_times["rendering"].append(render_time)

                #frame compositing
                frame_timer.start()
                
                frame.copy_(overlay)
                
                image.clamp_(0, 1)
                image_mask = torch.logical_and(
                    (image != 1.0).any(dim=2), image[:, :, 3] > 100 / 255
                )
                image[..., 3].masked_fill_(~image_mask, 0.0)

                alpha = image[..., 3:4]
                torch.mul(image[..., :3], alpha, out=rgb_temp)  # rgb_temp = image_rgb * alpha
                rgb_temp.mul_(255.0)                            # rgb_temp *= 255
                frame.mul_(1.0 - alpha).add_(rgb_temp)          # frame = frame * (1 - alpha) + rgb_temp        

                masks = get_shadow_masks_batched_downsampled(
                    points=x,
                    intrinsic_T=intrinsic_T,
                    w2c_T=w2c_T,
                    W=width, H=height,
                    image_mask=image_mask,
                    lights=lights,
                    inv_Lz=inv_Lz,
                    kernel_size=7,   # your original full-res k
                    scale=2,         # try 2 first; 4 if you need more speed
                    use_half=False,  # try True later if acceptable
                    upsample_mode="bilinear",   # "nearest" is sharper/aliasy, faster
                    post_blur=False  # set True if you see stair-steps
                )   # (L,H,W) bool
              
                # Turn masks + coeffs into one attenuation map A[H,W]
                M = masks.to(frame.dtype)                                     # float
                A = torch.prod(1.0 - M + M * coeffs_b, dim=0) 
                frame.mul_(A.unsqueeze(-1)) 

                frame_u8 = frame.clamp_(0, 255).to(torch.uint8)   # RGB uint8

                ####################pyh new rendering direclty render on GPU n(o GPU to CPU copying)
                # device→device copy into the mapped PBO, then update the texture and draw
                #convert to rgba
                frame_rgba[:, :, :3] = frame_u8
                frame_rgba[:, :, 3] = 255
                torch.cuda.current_stream().synchronize()  # ensures frame_rgba is ready to be read
                
                mapping = reg.map()
                try:
                    ptr, _ = mapping.device_ptr_and_size()
                    cpy2d.set_src_device(frame_rgba.data_ptr())
                    cpy2d.set_dst_device(ptr)
                    cpy2d(pbo_stream)

                    pbo_stream.synchronize()
                finally:
                    mapping.unmap()

                # Upload from PBO to texture (still on GPU)
                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
                render_backend.gl.glBindBuffer(render_backend.gl.GL_PIXEL_UNPACK_BUFFER, pbo)
                render_backend.gl.glTexSubImage2D(render_backend.gl.GL_TEXTURE_2D, 0, 0, 0, width, height, render_backend.gl.GL_RGBA, render_backend.gl.GL_UNSIGNED_BYTE, None)
                render_backend.gl.glBindBuffer(render_backend.gl.GL_PIXEL_UNPACK_BUFFER, 0)
                
                # Draw
                render_backend.gl.glViewport(0, 0, width, height)
                render_backend.gl.glDisable(render_backend.gl.GL_DEPTH_TEST)
                render_backend.gl.glClear(render_backend.gl.GL_COLOR_BUFFER_BIT)
                render_backend.gl.glUseProgram(prog)
                render_backend.gl.glActiveTexture(render_backend.gl.GL_TEXTURE0)
                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, tex)
                render_backend.gl.glDrawArrays(render_backend.gl.GL_TRIANGLE_STRIP, 0, 4)
                render_backend.gl.glBindTexture(render_backend.gl.GL_TEXTURE_2D, 0)
                render_backend.gl.glUseProgram(0)
                
                render_backend.glfw.swap_buffers(window)
                render_backend.glfw.poll_events()

                frame_comp_time = frame_timer.stop() 
                if frame_count > 1:
                    # Total frame compositing time
                    component_times["frame_compositing"].append(frame_comp_time)


                #LBs code
                if prev_x is not None:
                    with torch.no_grad():
                        interp_timer.start()
                        
                        #R reuse with dropping weight
                        current_pos, current_rot= lbs_with_rotation_reuse(
                            current_mass_nodes = x,
                            cache = rotation_cache,
                        )

                        interp_time = interp_timer.stop() 
                        gaussians._xyz = current_pos
                        gaussians._rotation = current_rot              

                    if frame_count > 1:
                        component_times["full_motion_interpolation"].append(interp_time)
                

                prev_x = x.clone()


                ############### Temporary timer ###############
                # Total loop time
                total_time = total_timer.stop()
                if frame_count > 1:
                    component_times["total"].append(total_time)

                # Calculate FPS
                fps = 1.0 / total_time
                if frame_count > 1:
                    fps_history.append(fps)

                # Save images before incrementing so frame 0 is written as 00000.png.
                if output_dir and save_eval_artifacts:
                   import torchvision

                   torchvision.utils.save_image(
                           rendering,
                           os.path.join(eval_render_path, "{0:05d}".format(frame_count) + ".png"),
                   )
                   save_path = os.path.join(view_render_path, f"{frame_count:05d}.png")
                   img_rgb = frame_rgba[:, :, :3]
                   img_rgb = img_rgb.permute(2, 0, 1).float() / 255.0
                   torchvision.utils.save_image(img_rgb, save_path)

                frame_count += 1

                prev_target = current_target
                #New updated to use the multi-input length
                if frame_count < runtime.frame_len:
                    current_target = self.batch_controller_points[frame_count]
                else:
                    print("Reached end of recorded control sequence")
                    break


        finally:

            #pyh add overall stat printing
            # --- Final Summary Statistics ---
            if frame_count > 1:

                frames_used_for_stats = len(component_times["total"])

                print(f"\n=== Final Summary (averaged over {frames_used_for_stats} frames) ===")
                #pyh save output for file as well
                log_lines = []
                log_lines.append(f"=== Final Summary (averaged over {frames_used_for_stats} frames) ===")

                total_frame_times = component_times["total"]
                total_time_seconds = sum(total_frame_times)
                average_fps = frames_used_for_stats / total_time_seconds
                average_frame_time = np.mean(total_frame_times)

                print(f"Average FPS: {average_fps:.2f}")
                print(f"Average Total Frame Time: {average_frame_time * 1000:.2f} ms")
                log_lines.append(f"Average FPS: {average_fps:.2f}")
                log_lines.append(f"Average Total Frame Time: {average_frame_time * 1000:.2f} ms")
                
                # Detailed breakdown by components
                components_to_report = [
                    "simulator",
                    "full_motion_interpolation",
                    "rendering",
                    "frame_compositing",
                ]

                for component_name in components_to_report:
                    component_times_list = component_times.get(component_name, [])
                    if component_times_list:
                        average_component_time = np.mean(component_times_list)
                        time_share_percentage = (average_component_time / average_frame_time) * 100
                        readable_name = perf_component_label(component_name)
                        print(f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)")
                        log_lines.append(f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)")

                #pyh save the performance log to a file
                if output_dir is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    log_file_path = os.path.join(output_dir, "performance_summary.txt")
                    with open(log_file_path, "w") as log_file:
                        log_file.write("\n".join(log_lines))

            #uncomment to save trajectory data for quality comparison
            # vertices = torch.stack(traj_frames_cpu, dim=0)  # (T, N, 3) on CPU
            # with open(traj_save_path, "wb") as f:
            #     pickle.dump(vertices.numpy(), f)
            # print(f"[Saved] inference.pkl -> {traj_save_path}, shape={tuple(vertices.shape)}")
            
            reg.unregister()
            render_backend.gl.glDeleteProgram(prog)
            render_backend.gl.glDeleteTextures([tex])
            render_backend.gl.glDeleteBuffers(1, [pbo])
            render_backend.gl.glDeleteVertexArrays(1, [vao])
            cuda_ctx.pop()

    
    def _create_gs_view(self, w2c, intrinsic, height, width):
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        K = torch.tensor(intrinsic, dtype=torch.float32, device="cuda")
        zoom_out = 1  # >1 means zoom out (wider angle)
        K[0,0] /= zoom_out
        K[1,1] /= zoom_out
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        view = Camera(
            (width, height),
            colmap_id="0000",
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            depth_params=None,
            image=None,
            invdepthmap=None,
            image_name="0000",
            uid="0000",
            data_device="cuda",
            train_test_exp=None,
            is_test_dataset=None,
            is_test_view=None,
            K=K,
            normal=None,
            depth=None,
            occ_mask=None,
        )
        return view, K

@torch.no_grad()
def get_shadow_masks_batched_downsampled(
    points,          # (N,3) float CUDA
    intrinsic_T,     # (3,3) float CUDA
    w2c_T,           # (4,4) float CUDA
    W: int, H: int,
    image_mask,      # (H,W) bool CUDA
    lights,          # (L,3) float CUDA
    inv_Lz,          # (L,)  float CUDA
    kernel_size: int = 7,     # original full-res k
    scale: int = 2,           # 2 or 4 are typical
    use_half: bool = False,
    upsample_mode: str = "bilinear",   # or "nearest"
    post_blur: bool = False,           # optional small blur after upsample
):
    """
    Returns: (L,H,W) bool CUDA  — masks computed at low-res and upsampled.
    """
    assert scale >= 1
    if scale == 1:
        # fall back to your existing high-res function if you like
        raise NotImplementedError("Use non-downsampled path when scale=1.")

    device = points.device
    dtype  = torch.float16 if use_half else points.dtype

    points      = points.to(dtype)
    intrinsic_T = intrinsic_T.to(dtype)
    w2c_T       = w2c_T.to(dtype)
    lights      = lights.to(dtype)
    inv_Lz      = inv_Lz.to(dtype)

    Hs, Ws = H // scale, W // scale
    N = points.shape[0]
    L = lights.shape[0]

    # ---- 1) Factor projection once: P = w2c[:,:3] @ K
    P = w2c_T[:, :3] @ intrinsic_T                      # (4,3)

    ones  = torch.ones((N,1), device=device, dtype=dtype)
    base4 = torch.cat([points, ones], dim=1)            # (N,4)
    pix3_base = base4 @ P                               # (N,3)

    zeros1 = torch.zeros((L,1), device=device, dtype=dtype)
    light4 = torch.cat([lights, zeros1], dim=1)         # (L,4)
    pix3_dir = light4 @ P                               # (L,3)

    # ---- 2) Project all lights; then scale pixel coords by 1/scale
    t_base = -points[:, 2].to(dtype)                    # (N,)
    t = inv_Lz.view(L, 1) * t_base.view(1, N)           # (L,N)
    pix3 = pix3_base.unsqueeze(0) + t.unsqueeze(-1) * pix3_dir.unsqueeze(1)  # (L,N,3)
    z = torch.clamp(pix3[..., 2:3], min=1e-12)
    pix = pix3[..., :2] / z                             # (L,N,2)

    # Downsampled pixels (divide by scale before flooring)
    x = torch.floor(pix[..., 0] / scale).to(torch.int64)
    y = torch.floor(pix[..., 1] / scale).to(torch.int64)
    valid = (x >= 0) & (x < Ws) & (y >= 0) & (y < Hs)

    idx = (y * Ws + x)
    #idx = torch.where(valid, idx, torch.zeros_like(idx))
    idx = idx.masked_fill_(~valid, 0)

    # ---- 3) Rasterize in low-res space (L, Hs*Ws)
    shadow_flat = torch.zeros((L, Hs * Ws), device=device, dtype=torch.float32)
    if hasattr(shadow_flat, "scatter_reduce_"):
        src = valid.to(torch.float32)
        shadow_flat.scatter_reduce_(dim=1, index=idx, src=src, reduce="amax", include_self=False)
    else:
        vmask = valid.view(-1)
        rows = torch.arange(L, device=device).view(L, 1).expand_as(idx).view(-1)[vmask]
        cols = idx.view(-1)[vmask]
        shadow_flat.index_put_((rows, cols), torch.ones_like(cols, dtype=torch.float32), accumulate=True)
        shadow_flat.clamp_(0, 1.0)

    shadow_lo = shadow_flat.view(1, L, Hs, Ws)   # N=1,C=L, Hs,Ws
    shadow_lo = shadow_lo.contiguous(memory_format=torch.channels_last)

    # ---- 4) Morphology at low-res; scale kernel to keep physical size similar
    k_lo = max(1, int(round(kernel_size / scale)) | 1)
    # separable k×1 then 1×k (flat structuring element)
    shadow_lo = F.max_pool2d(shadow_lo, kernel_size=(k_lo, 1), stride=1, padding=(k_lo // 2, 0))
    shadow_lo = F.max_pool2d(shadow_lo, kernel_size=(1, k_lo), stride=1, padding=(0, k_lo // 2))
    #pyh removed erosion step to get softer shadows
    shadow_lo = 1.0 - F.max_pool2d(1.0 - shadow_lo, kernel_size=(k_lo, 1), stride=1, padding=(k_lo // 2, 0))
    shadow_lo = 1.0 - F.max_pool2d(1.0 - shadow_lo, kernel_size=(1, k_lo), stride=1, padding=(0, k_lo // 2))

    # ---- 5) Apply occlusion at low-res (downsample occ by avg-pooling)
    # (avg-pool then threshold to get a conservative occlusion)
    occ = image_mask.view(1, 1, H, W).float()
    occ_lo = F.avg_pool2d(occ, kernel_size=scale, stride=scale)     # (1,1,Hs,Ws)
    occ_lo = (occ_lo > 0.5).to(shadow_lo.dtype)                      # conservative block
    shadow_lo.mul_(1.0 - occ_lo)

    # ---- 6) Upsample back to (H,W) and re-threshold
    shadow_hi = F.interpolate(shadow_lo, size=(H, W), mode=upsample_mode, align_corners=False if upsample_mode=="bilinear" else None)
    if post_blur:
        # simple 3x3 box blur per-channel (approximate soft edges)
        shadow_hi = F.avg_pool2d(shadow_hi, kernel_size=3, stride=1, padding=1)
    masks = (shadow_hi > 0.5).squeeze(0)   # (L,H,W) bool

    return masks
