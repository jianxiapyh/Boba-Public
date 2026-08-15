#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
from contextlib import nullcontext

import torch
from ..scene.gaussian_model import GaussianModel
from .._gsplat_vendor import rasterization, rasterization_shared_template


def _profile(profiler, name):
    if profiler is None:
        return nullcontext()
    return profiler.record(name)


def _is_shared_template_render_input(pc) -> bool:
    return bool(getattr(pc, "uses_shared_template_rendering", False))


def _densify_projection_metadata(
    info,
    num_gaussians: int,
    num_cameras: int = 1,
):
    camera_ids = info.get("camera_ids")
    gaussian_ids = info.get("gaussian_ids")
    radii = info.get("radii")
    means2d = info.get("means2d")

    if camera_ids is None or gaussian_ids is None or radii is None or means2d is None:
        return info

    dense_radii = torch.zeros(
        (num_cameras, num_gaussians, radii.shape[-1]),
        dtype=radii.dtype,
        device=radii.device,
    )
    dense_means2d = torch.zeros(
        (num_cameras, num_gaussians, means2d.shape[-1]),
        dtype=means2d.dtype,
        device=means2d.device,
    )

    dense_radii[camera_ids, gaussian_ids] = radii
    dense_means2d[camera_ids, gaussian_ids] = means2d

    dense_info = dict(info)
    dense_info["radii"] = dense_radii
    dense_info["means2d"] = dense_means2d
    return dense_info


def _build_intrinsics_matrix(viewpoint_camera, device_tensor: torch.Tensor):
    if viewpoint_camera.K is not None:
        focal_length_x = viewpoint_camera.K[0, 0]
        focal_length_y = viewpoint_camera.K[1, 1]
        cx = viewpoint_camera.K[0, 2]
        cy = viewpoint_camera.K[1, 2]
        K = torch.tensor(
            [
                [focal_length_x, 0, cx],
                [0, focal_length_y, cy],
                [0, 0, 1.0],
            ]
        ).to(device_tensor)
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
        focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
        K = torch.tensor(
            [
                [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
                [0, focal_length_y, viewpoint_camera.image_height / 2.0],
                [0, 0, 1],
            ]
        ).to(device_tensor)
    return K


def _format_gsplat_render_output(render_colors, render_alphas, info):
    rendered_image = render_colors[0].permute(2, 0, 1)[:3]
    rendered_depth = render_colors[0].permute(2, 0, 1)[3:]
    rendered_alphas = render_alphas[0].permute(2, 0, 1)

    radii = info["radii"].squeeze(0)
    if radii.ndim > 1:
        radii = radii.max(dim=-1).values

    try:
        info["means2d"].retain_grad()
    except Exception:
        pass

    screenspace_points = info["means2d"]
    rendered_image = torch.cat((rendered_image, rendered_alphas), dim=0)
    depth_image = rendered_depth.squeeze(0)

    return rendered_image, depth_image, screenspace_points, radii


def _format_gsplat_batch_images_output(
    render_colors,
    render_alphas,
    info,
    num_gaussians: int,
):
    if render_colors.dim() != 5 or render_alphas.dim() != 5:
        raise ValueError(
            "batch image rendering expects render_colors/render_alphas with "
            f"shape [B, C, H, W, D], got {tuple(render_colors.shape)} and "
            f"{tuple(render_alphas.shape)}"
        )
    if render_colors.shape[1] != 1:
        raise NotImplementedError(
            "render_mode='batch_images' currently supports num_views=1 only."
        )

    rgb = render_colors[:, 0, :, :, :3]
    alpha = render_alphas[:, 0]
    rendered_image = render_colors.new_empty(
        (
            render_colors.shape[0],
            4,
            render_colors.shape[2],
            render_colors.shape[3],
        )
    )
    rendered_image[:, :3].copy_(rgb.permute(0, 3, 1, 2))
    rendered_image[:, 3:4].copy_(alpha.permute(0, 3, 1, 2))

    return rendered_image, None, None, None


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    antialiased=False,
    profiler=None,
):
    return render_gsplat(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        scaling_modifier=scaling_modifier,
        override_color=override_color,
        antialiased=antialiased,
        profiler=profiler,
    )


# This is code is adapted from ChatSim background gaussians model: 
# https://github.com/yifanlu0227/ChatSim/blob/main/chatsim/background/gaussian-splatting/gaussian_renderer/gsplat_renderer.py
def render_gsplat(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    antialiased=True,
    render_normals=False,
    profiler=None,
):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    with _profile(profiler, "render_gsplat_total_ms"):
        if _is_shared_template_render_input(pc):
            if render_normals:
                raise NotImplementedError(
                    "Shared-template gsplat rendering does not support normal rendering."
                )
            return render_gsplat_shared_template(
                viewpoint_camera,
                pc,
                pipe,
                bg_color,
                scaling_modifier=scaling_modifier,
                override_color=override_color,
                antialiased=antialiased,
                profiler=profiler,
                record_total=False,
            )

        with _profile(profiler, "prepare_inputs_ms"):
            K = _build_intrinsics_matrix(viewpoint_camera, pc.get_xyz)
            means3D = pc.get_xyz
            opacity = pc.get_opacity
            scales = pc.get_scaling * scaling_modifier
            rotations = pc.get_rotation
            batch_image_mode = bool(
                getattr(pc, "uses_batch_image_rendering", False)
            )
            if batch_image_mode and render_normals:
                raise NotImplementedError(
                    "Batch image gsplat rendering does not support normal rendering."
                )

            if override_color is not None:
                colors = override_color # [N, 3]
                sh_degree = None
            else:
                colors = pc.get_features # [N, K, 3]
                sh_degree = pc.active_sh_degree

            viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]
            rasterize_mode = 'classic' if not antialiased else 'antialiased'
            if batch_image_mode:
                batch_size = int(means3D.shape[0])
                viewmats = viewmat[None, None].expand(batch_size, 1, 4, 4)
                Ks = K[None, None].expand(batch_size, 1, 3, 3)
                backgrounds = bg_color[None, None].expand(batch_size, 1, -1)
                standard_render_mode = "RGB"
            else:
                viewmats = viewmat[None]
                Ks = K[None]
                backgrounds = bg_color[None]
                standard_render_mode = "RGB+ED"

        render_colors, render_alphas, info = rasterization(
            means=means3D,    # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,    # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            backgrounds=backgrounds,
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=sh_degree,
            render_mode=standard_render_mode,
            rasterize_mode=rasterize_mode,
            absgrad=True,
            profiler=profiler,
        )
        with _profile(profiler, "format_output_ms"):
            if batch_image_mode:
                rendered_image, depth_image, screenspace_points, radii = (
                    _format_gsplat_batch_images_output(
                        render_colors,
                        render_alphas,
                        info,
                        num_gaussians=means3D.shape[-2],
                    )
                )
            else:
                rendered_image, depth_image, screenspace_points, radii = _format_gsplat_render_output(
                    render_colors,
                    render_alphas,
                    info,
                )

    ##### Our normal rendering #####
    if render_normals:

        render_extras = {}

        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # (N, 3)

        # compute normal image (reference: GaussianShader)
        normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized)
        normal_normed = normal * 0.5 + 0.5          # from [-1, 1] to [0, 1]
        render_extras["normal"] = normal_normed

        out_extras = {}
        for k in render_extras.keys():
            if render_extras[k] is None: continue
            render_colors = rasterization(
                means=means3D,    # [N, 3]
                quats=rotations,  # [N, 4]
                scales=scales,    # [N, 3]
                opacities=opacity.squeeze(-1),  # [N,]
                colors=render_extras[k],   # [N, 3] for normal
                viewmats=viewmat[None],  # [1, 4, 4]
                Ks=K[None],  # [1, 3, 3]
                backgrounds=None, # [1, 3]
                width=int(viewpoint_camera.image_width),
                height=int(viewpoint_camera.image_height),
                packed=False,
                sh_degree=None,
                render_mode='RGB+ED',
            )[0]
            image = render_colors[0].permute(2, 0, 1)[:3]   # [1, H, W, 4] -> [3, H, W]
            out_extras[k] = image

        for k in ["normal"]:
            if k in out_extras.keys():
                out_extras[k] = (out_extras[k] - 0.5) * 2. # from [0, 1] to [-1, 1]
    
        # normalize the normal map
        normal_image = out_extras["normal"]
        normal_image = normal_image.permute(1, 2, 0) # (H, W, 3)
        normal_image = torch.nn.functional.normalize(normal_image, p=2, dim=-1)
    else:
        normal_image = None

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return_pkg = {
        "render": rendered_image,
        "depth": depth_image,
        "normal": normal_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": None if radii is None else radii > 0,
        "radii": radii,
    }

    return return_pkg


def render_gsplat_shared_template(
    viewpoint_camera,
    pc,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    antialiased=True,
    profiler=None,
    record_total=True,
):
    total_scope = (
        _profile(profiler, "render_gsplat_total_ms")
        if record_total
        else nullcontext()
    )
    with total_scope:
        with _profile(profiler, "prepare_inputs_ms"):
            means3D = pc.get_xyz
            K = _build_intrinsics_matrix(viewpoint_camera, means3D)
            rotations = pc.get_rotation
            scales = pc.get_template_scaling * scaling_modifier
            opacity = pc.get_template_opacity

            if override_color is not None:
                colors = override_color
                sh_degree = None
            else:
                colors = pc.get_template_features
                sh_degree = pc.active_sh_degree

            viewmat = viewpoint_camera.world_view_transform.transpose(0, 1)
            rasterize_mode = "classic" if not antialiased else "antialiased"
            batch_image_mode = bool(
                getattr(pc, "uses_batch_image_rendering", False)
            )

        shared_template_render_mode = "RGB" if batch_image_mode else "RGB+ED"
        render_colors, render_alphas, info = rasterization_shared_template(
            means=means3D,
            quats=rotations,
            scales=scales,
            opacities=opacity.squeeze(-1),
            colors=colors,
            gaussians_per_instance=pc.gaussians_per_instance,
            viewmats=viewmat[None],
            Ks=K[None],
            backgrounds=bg_color[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            sh_degree=sh_degree,
            render_mode=shared_template_render_mode,
            rasterize_mode=rasterize_mode,
            absgrad=True,
            batch_image_mode=batch_image_mode,
            profiler=profiler,
        )
        if not batch_image_mode:
            with _profile(profiler, "densify_projection_metadata_ms"):
                info = _densify_projection_metadata(
                    info,
                    num_gaussians=means3D.shape[0],
                    num_cameras=1,
                )
        with _profile(profiler, "format_output_ms"):
            if batch_image_mode:
                rendered_image, depth_image, screenspace_points, radii = (
                    _format_gsplat_batch_images_output(
                        render_colors,
                        render_alphas,
                        info,
                        num_gaussians=means3D.shape[0],
                    )
                )
            else:
                rendered_image, depth_image, screenspace_points, radii = (
                    _format_gsplat_render_output(
                        render_colors,
                        render_alphas,
                        info,
                    )
                )

    return {
        "render": rendered_image,
        "depth": depth_image,
        "normal": None,
        "viewspace_points": screenspace_points,
        "visibility_filter": None if radii is None else radii > 0,
        "radii": radii,
    }
