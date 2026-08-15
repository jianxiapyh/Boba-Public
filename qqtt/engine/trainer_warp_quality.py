"""Single-instance quality/evaluation runtime for calibrated PhysTwin comparisons."""

import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
import warp as wp

import open3d as o3d

import torchvision
import cv2
import glfw
import pycuda.driver as cuda_driver
from OpenGL import GL as gl
from pycuda.gl import RegisteredBuffer, graphics_map_flags

from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.gaussian_renderer import render as render_gaussian
from gaussian_splatting.dynamic_utils import (
    lbs_with_rotation_reuse,
    build_rotation_reuse_cache,
    knn_weights_sparse,
    get_topk_indices,
)
from gaussian_splatting.utils.graphics_utils import focal2fov
from gaussian_splatting.utils.sh_utils import SH2RGB
from qqtt.data import RealData
from qqtt.model.diff_simulator import (
    SpringMassSystemWarp,
)
from qqtt.utils import logger, cfg
from qqtt.utils.gaussian import (
    remove_gaussians_with_low_opacity,
)


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
            )
        else:
            self.elapsed = time.time() - self.start_time
        return self.elapsed


def perf_component_label(component_name):
    labels = {
        "simulator": "Simulator",
        "full_motion_interpolation": "Linear Blend Skinning",
        "rendering": "Rendering",
        "frame_compositing": "Frame compositing",
    }
    return labels.get(component_name, component_name.replace("_", " ").capitalize())


class InvPhyTrainerWarp:
    def __init__(
        self,
        data_path,
        base_dir,
    ):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = "cuda:0"

        self.init_masks = None
        self.init_velocities = None
        self.object_restore_permutation = None
        if cfg.data_type == "real":
            self.dataset = RealData(visualize=False, save_gt=False)
            self.controller_points = self.dataset.controller_points
            self.structure_points = self.dataset.structure_points
            self.num_all_points = self.dataset.num_all_points
        elif cfg.data_type == "synthetic":
            print("synthetic data detected")
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
            num_object_points = len(points)
            print(f" num object springs {num_object_springs}")
            if controller_points is not None:
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

            points_torch = torch.tensor(points, dtype=torch.float32, device=cfg.device)
            springs_torch = torch.tensor(springs, dtype=torch.int32, device=cfg.device)
            rest_lengths_torch = torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device)
            masses_torch = torch.tensor(masses, dtype=torch.float32, device=cfg.device)

            logger.info("Applying Morton ordering to improve cache performance...")
            (
                points_torch,
                springs_torch,
                rest_lengths_torch,
                spring_perm,
            ) = self._apply_morton_reordering(
                points_torch,
                springs_torch,
                rest_lengths_torch,
                num_object_points,
            )
            self.spring_permutation = spring_perm

            return (
                points_torch,
                springs_torch,
                rest_lengths_torch,
                masses_torch,
                num_object_springs,
            )
        else:
            mask = mask.cpu().numpy()
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
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
            num_object_points = len(vertices)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(vertices))

            vertices_torch = torch.tensor(vertices, dtype=torch.float32, device=cfg.device)
            springs_torch = torch.tensor(springs, dtype=torch.int32, device=cfg.device)
            rest_lengths_torch = torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device)
            masses_torch = torch.tensor(masses, dtype=torch.float32, device=cfg.device)

            logger.info("Applying Morton ordering to improve cache performance (multi-object)...")
            (
                vertices_torch,
                springs_torch,
                rest_lengths_torch,
                spring_perm,
            ) = self._apply_morton_reordering(
                vertices_torch,
                springs_torch,
                rest_lengths_torch,
                num_object_points,
            )
            self.spring_permutation = spring_perm

            return (
                vertices_torch,
                springs_torch,
                rest_lengths_torch,
                masses_torch,
                num_object_springs,
            )

    def stable_lexsort(self, keys):
        """
        keys: list of 1D tensors, all same length.
            Order is MOST-significant -> LEAST-significant.
        Returns: permutation idx such that keys are sorted lexicographically.
        """
        assert len(keys) > 0
        n = keys[0].numel()
        idx = torch.arange(n, device=keys[0].device)

        for k in reversed(keys):
            idx = idx[torch.argsort(k[idx], stable=True)]
        return idx

    def _reorder_springs_spatial_blocking(
        self,
        springs,
        rest_lengths,
        num_object_points,
        block_size=32,
    ):
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

        obj_obj_mask = (i < N) & (j < N)
        ctrl_mask = ~obj_obj_mask

        if ctrl_mask.any():
            has_obj_ep = (i[ctrl_mask] < N) | (j[ctrl_mask] < N)
            assert has_obj_ep.all(), "Found controller-controller springs!"

        obj_ids = torch.nonzero(obj_obj_mask, as_tuple=False).squeeze(1)
        if obj_ids.numel() > 0:
            io = i[obj_ids]
            jo = j[obj_ids]
            a = torch.minimum(io, jo)
            b = torch.maximum(io, jo)

            ablk = a // block_size
            bblk = b // block_size

            perm_obj_local = self.stable_lexsort([ablk, bblk, a, b])
            obj_order = obj_ids[perm_obj_local]
        else:
            obj_order = obj_ids

        ctrl_ids = torch.nonzero(ctrl_mask, as_tuple=False).squeeze(1)
        if ctrl_ids.numel() > 0:
            ic = i[ctrl_ids]
            jc = j[ctrl_ids]

            obj_ep = torch.where(ic < N, ic, jc)
            ctrl_ep = torch.where(ic >= N, ic, jc)
            ctrl_id = (ctrl_ep - N).clamp_min(0)

            obj_blk = obj_ep // block_size

            perm_ctrl_local = self.stable_lexsort([obj_blk, obj_ep, ctrl_id])
            ctrl_order = ctrl_ids[perm_ctrl_local]

        else:
            ctrl_order = ctrl_ids

        perm = torch.cat([obj_order, ctrl_order], dim=0)

        assert perm.shape[0] == springs.shape[0], "Permutation size mismatch"
        assert torch.unique(perm).shape[0] == springs.shape[0], "Permutation has duplicates"

        print(f"Reordered springs with spatial blocking (block_size={block_size}):")
        print(f"  Object-object springs: {obj_ids.numel()}")
        print(f"  Controller-object springs: {ctrl_ids.numel()}")

        return springs[perm], rest_lengths[perm], perm

    def _apply_morton_reordering(
        self,
        vertices,
        springs,
        rest_lengths,
        num_object_points,
    ):
        """
        Reorder object vertices using Morton (Z-order) curve AND reorder springs for coalescing.

        Returns:
            new_vertices: reordered vertices
            new_springs: springs with remapped indices AND reordered for coalescing
            new_rest_lengths: rest_lengths reordered to match springs
            spring_permutation: permutation applied to springs (for reordering other per-spring arrays)
        """
        device = vertices.device

        assert springs.device == device
        assert rest_lengths.device == device

        obj_end = num_object_points
        has_controllers = (num_object_points < len(vertices))

        obj_verts = vertices[:obj_end].detach().cpu().numpy()

        mins = obj_verts.min(axis=0)
        maxs = obj_verts.max(axis=0)
        range_vals = maxs - mins
        range_vals[range_vals < 1e-8] = 1.0
        normalized = (obj_verts - mins) / range_vals

        BITS = 21
        MAX_VAL = (1 << BITS) - 1
        int_coords = (normalized * MAX_VAL).astype(np.uint64)

        def part1by2(n):
            n = np.uint64(n)
            n = (n | (n << 32)) & np.uint64(0x1f00000000ffff)
            n = (n | (n << 16)) & np.uint64(0x1f0000ff0000ff)
            n = (n | (n << 8)) & np.uint64(0x100f00f00f00f00f)
            n = (n | (n << 4)) & np.uint64(0x10c30c30c30c30c3)
            n = (n | (n << 2)) & np.uint64(0x1249249249249249)
            return n

        x = part1by2(int_coords[:, 0])
        y = part1by2(int_coords[:, 1])
        z = part1by2(int_coords[:, 2])
        morton_codes = x | (y << 1) | (z << 2)

        perm = np.argsort(morton_codes, kind="stable")
        inv_perm = np.empty(obj_end, dtype=np.int64)
        inv_perm[perm] = np.arange(obj_end)

        perm_torch = torch.from_numpy(perm).to(device).long()
        inv_perm_torch = torch.from_numpy(inv_perm).to(device).long()
        self.object_restore_permutation = inv_perm_torch.detach().cpu()

        reordered_obj_verts = torch.index_select(vertices[:obj_end], 0, perm_torch)
        if has_controllers:
            new_vertices = torch.cat([reordered_obj_verts, vertices[obj_end:]], dim=0)
        else:
            new_vertices = reordered_obj_verts

        springs_dtype = springs.dtype
        new_springs = springs.clone()

        for col in [0, 1]:
            obj_mask = springs[:, col] < obj_end
            obj_indices = springs[obj_mask, col].long()
            remapped_indices = inv_perm_torch[obj_indices].to(springs_dtype)
            new_springs[obj_mask, col] = remapped_indices

        logger.info(f"Applied Morton reordering to {obj_end} object vertices")

        (
            new_springs,
            new_rest_lengths,
            spring_permutation,
        ) = self._reorder_springs_spatial_blocking(
            new_springs,
            rest_lengths,
            num_object_points,
            block_size=32,
        )
        return new_vertices, new_springs, new_rest_lengths, spring_permutation

    def _restore_original_object_order(self, vertices):
        if self.object_restore_permutation is None:
            return vertices

        object_count = self.object_restore_permutation.numel()
        if vertices.shape[1] < object_count:
            raise ValueError(
                "Saved trajectory has fewer nodes than the object restore permutation: "
                f"{vertices.shape[1]} < {object_count}"
            )

        restore_perm = self.object_restore_permutation.to(vertices.device)
        restored_objects = torch.index_select(vertices[:, :object_count], 1, restore_perm)
        if vertices.shape[1] == object_count:
            return restored_objects

        return torch.cat([restored_objects, vertices[:, object_count:]], dim=1)

    def _capture_collision_pruning_export_frame(
        self,
        exports,
        frame_count,
        x,
        current_pos,
        current_quat,
        n_vert_single_obj,
        n_gaussians_single_obj,
    ):
        object_nodes = x[:n_vert_single_obj].detach().cpu().unsqueeze(0)
        object_nodes = self._restore_original_object_order(object_nodes)[0]
        exports[int(frame_count)] = {
            "mass_nodes": object_nodes.numpy(),
            "gaussian_xyz": current_pos[:n_gaussians_single_obj]
            .detach()
            .cpu()
            .numpy(),
            "gaussian_quat": current_quat[:n_gaussians_single_obj]
            .detach()
            .cpu()
            .numpy(),
        }
        print(
            "[CollisionPruningExport] captured frame "
            f"{frame_count}: {n_vert_single_obj} mass nodes, "
            f"{n_gaussians_single_obj} gaussians"
        )

    def _write_collision_pruning_export(
        self,
        export_path,
        export_frames,
        exports,
        gaussian_rgb_single,
        gaussian_scales_rest,
        gaussian_opacity,
        n_vert_single_obj,
        n_gaussians_single_obj,
    ):
        missing_frames = [
            int(frame) for frame in export_frames if int(frame) not in exports
        ]
        if missing_frames:
            raise RuntimeError(
                "Missing requested collision-pruning export frames: "
                f"{missing_frames}"
            )

        export_path = os.path.abspath(export_path)
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        ordered_frames = np.asarray(export_frames, dtype=np.int32)
        mass_nodes = np.stack(
            [exports[int(frame)]["mass_nodes"] for frame in export_frames],
            axis=0,
        )
        gaussian_xyz = np.stack(
            [exports[int(frame)]["gaussian_xyz"] for frame in export_frames],
            axis=0,
        )
        gaussian_quat = np.stack(
            [exports[int(frame)]["gaussian_quat"] for frame in export_frames],
            axis=0,
        )
        np.savez_compressed(
            export_path,
            frames=ordered_frames,
            mass_nodes=mass_nodes,
            gaussian_xyz=gaussian_xyz,
            gaussian_quat=gaussian_quat,
            gaussian_rgb=gaussian_rgb_single.detach().cpu().numpy(),
            gaussian_scales_rest=gaussian_scales_rest.detach().cpu().numpy(),
            gaussian_opacity=gaussian_opacity.detach().cpu().numpy(),
            c2ws=np.asarray(cfg.c2ws),
            w2cs=np.asarray(cfg.w2cs),
            intrinsics=np.asarray(cfg.intrinsics),
            WH=np.asarray(cfg.WH, dtype=np.int32),
            object_mass_node_count=np.asarray(n_vert_single_obj, dtype=np.int32),
            gaussian_count=np.asarray(n_gaussians_single_obj, dtype=np.int32),
            exported_in_original_object_order=np.asarray(True),
            export_schema_version=np.asarray(2, dtype=np.int32),
            gaussian_kernel_parameterization=np.asarray("scale_quat_covariance"),
        )
        print(f"[CollisionPruningExport] wrote {export_path}")

    def _run_collision_pruning_export_only(
        self,
        export_path,
        export_frames,
        export_frame_set,
        controller_points,
        rotation_cache,
        current_pos,
        current_quat,
        gaussian_rgb_single,
        gaussian_scales_rest,
        gaussian_opacity,
        n_vert_single_obj,
        n_gaussians_single_obj,
    ):
        exports = {}
        max_export_frame = max(export_frames)
        prev_target = controller_points[0]

        x = wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
        if 0 in export_frame_set:
            self._capture_collision_pruning_export_frame(
                exports,
                0,
                x,
                current_pos,
                current_quat,
                n_vert_single_obj,
                n_gaussians_single_obj,
            )

        for frame_count in range(1, max_export_frame + 1):
            current_target = controller_points[frame_count]
            self.simulator.set_controller_interactive(prev_target, current_target)

            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()
            wp.capture_launch(self.simulator.forward_graph)
            wp.synchronize()

            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            current_pos, current_quat = lbs_with_rotation_reuse(
                current_mass_nodes=x,
                cache=rotation_cache,
            )

            if frame_count in export_frame_set:
                self._capture_collision_pruning_export_frame(
                    exports,
                    frame_count,
                    x,
                    current_pos,
                    current_quat,
                    n_vert_single_obj,
                    n_gaussians_single_obj,
                )

            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )
            prev_target = current_target

            if export_frame_set.issubset(exports.keys()):
                break

        self._write_collision_pruning_export(
            export_path,
            export_frames,
            exports,
            gaussian_rgb_single,
            gaussian_scales_rest,
            gaussian_opacity,
            n_vert_single_obj,
            n_gaussians_single_obj,
        )

    def interactive_playground(
        self,
        model_path,
        gs_path,
        output_dir=None,
        window=None,
        cuda_ctx=None,
        num_views=1,
        collision_pruning_export_path=None,
    ):
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
        init_velocities = self.init_velocities
        controller_points = self.controller_points
        object_massnodes_total = self.num_all_points
        export_requested = collision_pruning_export_path is not None
        if export_requested and frame_len <= 0:
            raise ValueError("collision-pruning export requires at least one frame")
        export_frames = list(range(frame_len)) if export_requested else []
        export_frame_set = set(export_frames)

        print(
            f"[Quality] single-instance object mass node {n_vert_single_obj}, "
            f"controller mass node {n_vert_single_ctrl}"
        )
        print(
            f"[Quality] total mass nodes {self.init_vertices.shape[0]}, "
            f"expected {object_massnodes_total + n_vert_single_ctrl}"
        )

        self.simulator = SpringMassSystemWarp(
            base_springs=self.init_springs,
            base_rest_lengths=self.init_rest_lengths,
            init_masses=self.init_masses,
            init_masks=self.init_masks,
            signed_incidence_map=None,
            max_incident_springs=0,
            init_vertices=self.init_vertices,
            init_velocities=init_velocities,
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
            object_massnodes_total=object_massnodes_total,
            object_massnodes_single=n_vert_single_obj,
            controller_massnodes_single=n_vert_single_ctrl,
            controller_rest_location=controller_points[0],
            number_of_instance=1,
        )

        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )

        if self.simulator.object_collision_flag:
            self.simulator.create_resting_case()

        self.simulator.create_cuda_graph()

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        raw_gaussian_count = int(gaussians._xyz.shape[0])
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        render_gaussians = gaussians
        n_gaussians_single_obj = gaussians._xyz.shape[0]
        rendered_gaussian_count = int(n_gaussians_single_obj)
        xyz_rest_single = gaussians.get_xyz[:n_gaussians_single_obj]
        rot_rest_single = gaussians.get_rotation[:n_gaussians_single_obj]
        gaussian_scales_rest = gaussians.get_scaling[:n_gaussians_single_obj]
        gaussian_opacity_single = gaussians.get_opacity[:n_gaussians_single_obj]
        gaussian_rgb_single = SH2RGB(
            gaussians.get_features_dc[:n_gaussians_single_obj, 0, :]
        ).clamp(0.0, 1.0)

        torch.cuda.empty_cache()

        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        current_pos = gaussians.get_xyz
        current_rot = rot_rest_single

        rest_mass_node_single = prev_x[:n_vert_single_obj]
        relations_single = get_topk_indices(rest_mass_node_single, K=3)
        weights_single, weights_indices_single = knn_weights_sparse(
            rest_mass_node_single,
            current_pos[:n_gaussians_single_obj],
            K=3,
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
            number_of_instance=1,
        )

        prev_target = controller_points[0]
        current_target = prev_target

        if export_requested and window is None:
            self._run_collision_pruning_export_only(
                collision_pruning_export_path,
                export_frames,
                export_frame_set,
                controller_points,
                rotation_cache,
                current_pos,
                current_rot,
                gaussian_rgb_single,
                gaussian_scales_rest,
                gaussian_opacity_single,
                n_vert_single_obj,
                n_gaussians_single_obj,
            )
            return

        glfw.make_context_current(window)
        width, height = cfg.WH
        available_views = min(len(cfg.intrinsics), len(cfg.w2cs))
        if num_views < 1 or num_views > available_views:
            raise ValueError(
                f"num_views must be between 1 and {available_views}. Received: {num_views}"
            )

        intrinsic = cfg.intrinsics[0]
        w2c = cfg.w2cs[0]

        background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        view, K_cuda = self._create_gs_view(w2c, intrinsic, height, width)
        render_views = [(0, view)]
        for view_idx in range(1, num_views):
            extra_view, _ = self._create_gs_view(
                cfg.w2cs[view_idx], cfg.intrinsics[view_idx], height, width
            )
            render_views.append((view_idx, extra_view))
        image_path = cfg.bg_img_path
        overlay = cv2.imread(image_path)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)
        assert (
            overlay.shape[0] == height and overlay.shape[1] == width
        ), f"overlay {tuple(overlay.shape)} != (H,W,3)=({height},{width},3)"

        lights = torch.tensor(
            [
                [0, 0, -3],
                [1, 0.5, -2],
                [-3, -0.5, -5],
            ],
            device=cfg.device,
            dtype=torch.float32,
        )
        coeffs = torch.tensor([0.95, 0.97, 0.98], device=cfg.device, dtype=torch.float32)
        w2c_cuda = torch.tensor(w2c, dtype=torch.float32, device=cfg.device)
        coeffs_b = coeffs.view(-1, 1, 1)
        w2c_T = w2c_cuda.T.contiguous()
        intrinsic_T = K_cuda.T.contiguous()
        inv_Lz = 1.0 / lights[:, 2]
        BYTES_PER_PIXEL = 4
        pbo_size = width * height * BYTES_PER_PIXEL
        row_pitch = width * BYTES_PER_PIXEL

        tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        pbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, pbo_size, None, gl.GL_STREAM_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        reg = RegisteredBuffer(int(pbo), graphics_map_flags.WRITE_DISCARD)

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
            sid = gl.glCreateShader(kind)
            gl.glShaderSource(sid, src)
            gl.glCompileShader(sid)
            if not gl.glGetShaderiv(sid, gl.GL_COMPILE_STATUS):
                raise RuntimeError(gl.glGetShaderInfoLog(sid).decode())
            return sid

        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, _compile(gl.GL_VERTEX_SHADER, VS))
        gl.glAttachShader(prog, _compile(gl.GL_FRAGMENT_SHADER, FS))
        gl.glLinkProgram(prog)
        if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
            raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())
        gl.glUseProgram(prog)
        gl.glUniform1i(gl.glGetUniformLocation(prog, "uTex"), 0)
        gl.glUseProgram(0)
        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)

        pbo_stream = cuda_driver.Stream()

        cpy2d = cuda_driver.Memcpy2D()
        cpy2d.src_pitch = row_pitch
        cpy2d.dst_pitch = row_pitch
        cpy2d.width_in_bytes = row_pitch
        cpy2d.height = height

        frame_rgba = torch.empty((height, width, 4), dtype=torch.uint8, device=cfg.device)
        frame = torch.empty_like(overlay)
        rgb_temp = torch.empty((height, width, 3), dtype=overlay.dtype, device=cfg.device)

        if output_dir:
            eval_render_paths = {
                view_idx: os.path.join(output_dir, str(view_idx))
                for view_idx in range(num_views)
            }
            view_render_path = os.path.join(output_dir, "output")
            os.makedirs(view_render_path, exist_ok=True)
            for render_path in eval_render_paths.values():
                os.makedirs(render_path, exist_ok=True)
            traj_save_path = os.path.join(output_dir, "inference.pkl")

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
        frame_count = 0
        collision_pruning_exports = {}

        traj_frames_cpu = []
        x0 = wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
        traj_frames_cpu.append(x0.detach().cpu())
        try:
            while True:
                total_timer.start()

                if frame_count == 0:
                    x = wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False)
                else:
                    sim_timer.start()

                    self.simulator.set_controller_interactive(prev_target, current_target)

                    if self.simulator.object_collision_flag:
                        self.simulator.update_collision_graph()
                    wp.capture_launch(self.simulator.forward_graph)
                    wp.synchronize()

                    x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                    traj_frames_cpu.append(x.detach().cpu())

                    self.simulator.set_init_state(
                        self.simulator.wp_states[-1].wp_x,
                        self.simulator.wp_states[-1].wp_v,
                    )

                    sim_time = sim_timer.stop()

                    if frame_count > 1:
                        component_times["simulator"].append(sim_time)

                render_timer.start()

                results = render_gaussian(view, render_gaussians, None, background)
                rendering = results["render"]
                additional_renderings = {}
                if output_dir and num_views > 1:
                    for view_idx, extra_view in render_views[1:]:
                        additional_renderings[view_idx] = render_gaussian(
                            extra_view, render_gaussians, None, background
                        )["render"]
                image = rendering.permute(1, 2, 0).detach()

                render_time = render_timer.stop()
                if frame_count > 1:
                    component_times["rendering"].append(render_time)

                frame_timer.start()

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

                masks = get_shadow_masks_batched_downsampled(
                    points=x,
                    intrinsic_T=intrinsic_T,
                    w2c_T=w2c_T,
                    W=width, H=height,
                    image_mask=image_mask,
                    lights=lights,
                    inv_Lz=inv_Lz,
                    kernel_size=7,
                    scale=2,
                    use_half=False,
                    upsample_mode="bilinear",
                    post_blur=False,
                )

                M = masks.to(frame.dtype)
                A = torch.prod(1.0 - M + M * coeffs_b, dim=0)
                frame.mul_(A.unsqueeze(-1))

                frame_u8 = frame.clamp_(0, 255).to(torch.uint8)
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

                gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
                gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, pbo)
                gl.glTexSubImage2D(
                    gl.GL_TEXTURE_2D,
                    0,
                    0,
                    0,
                    width,
                    height,
                    gl.GL_RGBA,
                    gl.GL_UNSIGNED_BYTE,
                    None,
                )
                gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

                gl.glViewport(0, 0, width, height)
                gl.glDisable(gl.GL_DEPTH_TEST)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                gl.glUseProgram(prog)
                gl.glActiveTexture(gl.GL_TEXTURE0)
                gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
                gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                gl.glUseProgram(0)

                glfw.swap_buffers(window)
                glfw.poll_events()

                frame_comp_time = frame_timer.stop()
                if frame_count > 1:
                    component_times["frame_compositing"].append(frame_comp_time)

                if frame_count > 0:
                    with torch.no_grad():
                        interp_timer.start()

                        current_pos, current_rot = lbs_with_rotation_reuse(
                            current_mass_nodes=x,
                            cache=rotation_cache,
                        )

                        interp_time = interp_timer.stop()
                        gaussians._xyz = current_pos
                        gaussians._rotation = current_rot

                    if frame_count > 1:
                        component_times["full_motion_interpolation"].append(interp_time)

                if export_requested and frame_count in export_frame_set:
                    with torch.no_grad():
                        self._capture_collision_pruning_export_frame(
                            collision_pruning_exports,
                            frame_count,
                            x,
                            current_pos,
                            current_rot,
                            n_vert_single_obj,
                            n_gaussians_single_obj,
                        )

                prev_x = x.clone()

                total_time = total_timer.stop()
                if frame_count > 1:
                    component_times["total"].append(total_time)

                if output_dir:
                    torchvision.utils.save_image(
                        rendering,
                        os.path.join(
                            eval_render_paths[0],
                            f"{frame_count:05d}.png",
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
                    save_path = os.path.join(view_render_path, f"{frame_count:05d}.png")
                    img_rgb = frame_rgba[:, :, :3]
                    img_rgb = img_rgb.permute(2, 0, 1).float() / 255.0
                    torchvision.utils.save_image(img_rgb, save_path)

                if export_requested and export_frame_set.issubset(
                    collision_pruning_exports.keys()
                ):
                    print(
                        "[CollisionPruningExport] collected full frame sequence; "
                        "stopping quality playback early"
                    )
                    break

                frame_count += 1

                prev_target = current_target
                if frame_count < frame_len:
                    current_target = controller_points[frame_count]
                else:
                    print("Reached end of recorded control sequence")
                    break


        finally:
            if frame_count > 1:
                frames_used_for_stats = len(component_times["total"])

                print(f"\n=== Final Summary (averaged over {frames_used_for_stats} frames) ===")
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
                print(
                    "Gaussians: "
                    f"{raw_gaussian_count} raw, "
                    f"{rendered_gaussian_count} after opacity filter"
                )
                log_lines.append(
                    "Gaussians: "
                    f"{raw_gaussian_count} raw, "
                    f"{rendered_gaussian_count} after opacity filter"
                )
                components_to_report = [
                    "simulator",
                    "full_motion_interpolation",
                    "rendering",
                    "frame_compositing",
                ]

                metrics = {
                    "frames_used_for_stats": int(frames_used_for_stats),
                    "average_fps": float(average_fps),
                    "average_total_frame_time_ms": float(average_frame_time * 1000.0),
                    "gaussian_path": str(gs_path),
                    "raw_gaussian_count": raw_gaussian_count,
                    "rendered_gaussian_count": rendered_gaussian_count,
                    "opacity_filter_threshold": 0.1,
                }
                for component_name in components_to_report:
                    component_times_list = component_times.get(component_name, [])
                    if component_times_list:
                        average_component_time = np.mean(component_times_list)
                        time_share_percentage = (average_component_time / average_frame_time) * 100
                        readable_name = perf_component_label(component_name)
                        print(f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)")
                        log_lines.append(f"{readable_name}: {average_component_time * 1000:.2f} ms ({time_share_percentage:.1f}%)")
                        metrics[f"average_{component_name}_ms"] = float(
                            average_component_time * 1000.0
                        )
                        metrics[f"average_{component_name}_share_pct"] = float(
                            time_share_percentage
                        )

                if output_dir is not None:
                    os.makedirs(output_dir, exist_ok=True)
                    log_file_path = os.path.join(output_dir, "performance_summary.txt")
                    with open(log_file_path, "w") as log_file:
                        log_file.write("\n".join(log_lines))
                    metrics_path = os.path.join(output_dir, "performance_summary.json")
                    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
                        json.dump(metrics, metrics_file, indent=2)

            if output_dir is not None:
                vertices = torch.stack(traj_frames_cpu, dim=0)
                vertices = self._restore_original_object_order(vertices)
                with open(traj_save_path, "wb") as f:
                    pickle.dump(vertices.numpy(), f)
                print(f"[Saved] inference.pkl -> {traj_save_path}, shape={tuple(vertices.shape)}")

            if export_requested:
                self._write_collision_pruning_export(
                    collision_pruning_export_path,
                    export_frames,
                    collision_pruning_exports,
                    gaussian_rgb_single,
                    gaussian_scales_rest,
                    gaussian_opacity_single,
                    n_vert_single_obj,
                    n_gaussians_single_obj,
                )

            reg.unregister()
            gl.glDeleteProgram(prog)
            gl.glDeleteTextures([tex])
            gl.glDeleteBuffers(1, [pbo])
            gl.glDeleteVertexArrays(1, [vao])
            cuda_ctx.pop()


    def _create_gs_view(self, w2c, intrinsic, height, width):
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        K = torch.tensor(intrinsic, dtype=torch.float32, device="cuda")
        zoom_out = 1
        K[0, 0] /= zoom_out
        K[1, 1] /= zoom_out
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
    points,
    intrinsic_T,
    w2c_T,
    W: int,
    H: int,
    image_mask,
    lights,
    inv_Lz,
    kernel_size: int = 7,
    scale: int = 2,
    use_half: bool = False,
    upsample_mode: str = "bilinear",
    post_blur: bool = False,
):
    """
    Returns one shadow mask per light as an (L, H, W) CUDA bool tensor.
    """
    assert scale >= 1
    if scale == 1:
        raise NotImplementedError("Use non-downsampled path when scale=1.")

    device = points.device
    dtype = torch.float16 if use_half else points.dtype

    points = points.to(dtype)
    intrinsic_T = intrinsic_T.to(dtype)
    w2c_T = w2c_T.to(dtype)
    lights = lights.to(dtype)
    inv_Lz = inv_Lz.to(dtype)

    Hs, Ws = H // scale, W // scale
    N = points.shape[0]
    L = lights.shape[0]

    P = w2c_T[:, :3] @ intrinsic_T

    ones = torch.ones((N, 1), device=device, dtype=dtype)
    base4 = torch.cat([points, ones], dim=1)
    pix3_base = base4 @ P

    zeros1 = torch.zeros((L, 1), device=device, dtype=dtype)
    light4 = torch.cat([lights, zeros1], dim=1)
    pix3_dir = light4 @ P

    t_base = -points[:, 2].to(dtype)
    t = inv_Lz.view(L, 1) * t_base.view(1, N)
    pix3 = pix3_base.unsqueeze(0) + t.unsqueeze(-1) * pix3_dir.unsqueeze(1)
    z = torch.clamp(pix3[..., 2:3], min=1e-12)
    pix = pix3[..., :2] / z

    x = torch.floor(pix[..., 0] / scale).to(torch.int64)
    y = torch.floor(pix[..., 1] / scale).to(torch.int64)
    valid = (x >= 0) & (x < Ws) & (y >= 0) & (y < Hs)

    idx = (y * Ws + x)
    idx = idx.masked_fill_(~valid, 0)

    shadow_flat = torch.zeros((L, Hs * Ws), device=device, dtype=torch.float32)
    if hasattr(shadow_flat, "scatter_reduce_"):
        src = valid.to(torch.float32)
        shadow_flat.scatter_reduce_(
            dim=1,
            index=idx,
            src=src,
            reduce="amax",
            include_self=False,
        )
    else:
        vmask = valid.view(-1)
        rows = torch.arange(L, device=device).view(L, 1).expand_as(idx).view(-1)[vmask]
        cols = idx.view(-1)[vmask]
        shadow_flat.index_put_(
            (rows, cols),
            torch.ones_like(cols, dtype=torch.float32),
            accumulate=True,
        )
        shadow_flat.clamp_(0, 1.0)

    shadow_lo = shadow_flat.view(1, L, Hs, Ws)
    shadow_lo = shadow_lo.contiguous(memory_format=torch.channels_last)

    k_lo = max(1, int(round(kernel_size / scale)) | 1)
    shadow_lo = F.max_pool2d(
        shadow_lo,
        kernel_size=(k_lo, 1),
        stride=1,
        padding=(k_lo // 2, 0),
    )
    shadow_lo = F.max_pool2d(
        shadow_lo,
        kernel_size=(1, k_lo),
        stride=1,
        padding=(0, k_lo // 2),
    )
    shadow_lo = 1.0 - F.max_pool2d(
        1.0 - shadow_lo,
        kernel_size=(k_lo, 1),
        stride=1,
        padding=(k_lo // 2, 0),
    )
    shadow_lo = 1.0 - F.max_pool2d(
        1.0 - shadow_lo,
        kernel_size=(1, k_lo),
        stride=1,
        padding=(0, k_lo // 2),
    )

    occ = image_mask.view(1, 1, H, W).float()
    occ_lo = F.avg_pool2d(occ, kernel_size=scale, stride=scale)
    occ_lo = (occ_lo > 0.5).to(shadow_lo.dtype)
    shadow_lo.mul_(1.0 - occ_lo)

    shadow_hi = F.interpolate(
        shadow_lo,
        size=(H, W),
        mode=upsample_mode,
        align_corners=False if upsample_mode == "bilinear" else None,
    )
    if post_blur:
        shadow_hi = F.avg_pool2d(shadow_hi, kernel_size=3, stride=1, padding=1)
    masks = (shadow_hi > 0.5).squeeze(0)

    return masks
