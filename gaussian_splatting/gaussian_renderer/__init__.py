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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from ..scene.gaussian_model import GaussianModel
from ..utils.sh_utils import eval_sh
from torch.nn import functional as F
from gsplat import rasterization


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, use_gsplat=True, antialiased=False, separate_sh = False, use_trained_exp=False):
    if use_gsplat:
        return render_gsplat(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, antialiased)
    else:
        return render_3dgs(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, separate_sh, override_color, use_trained_exp)


# This is code is adapted from ChatSim background gaussians model: 
# https://github.com/yifanlu0227/ChatSim/blob/main/chatsim/background/gaussian-splatting/gaussian_renderer/gsplat_renderer.py
def render_gsplat(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, antialiased = True, render_normals = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Set up rasterization configuration
    if viewpoint_camera.K is not None:
        # print("====== Use camera K ======")
        # focal_length_x, focal_length_y, cx, cy = viewpoint_camera.K
        focal_length_x, focal_length_y, cx, cy = viewpoint_camera.K[0, 0], viewpoint_camera.K[1, 1], viewpoint_camera.K[0, 2], viewpoint_camera.K[1, 2]
        K = torch.tensor([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1.0]
        ]).to(pc.get_xyz)
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
        ).to(pc.get_xyz)

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation

    if override_color is not None:
        colors = override_color # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]

    rasterize_mode = 'classic' if not antialiased else 'antialiased'

    render_colors, render_alphas, info = rasterization(
        means=means3D,    # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,    # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
        render_mode='RGB+ED',
        rasterize_mode=rasterize_mode,
        absgrad=True
    )
    # [1, H, W, 4] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)[:3]
    # [1, H, W, 4] -> [1, H, W]
    rendered_depth = render_colors[0].permute(2, 0, 1)[3:]
    # [1, H, W, 1] -> [1, H, W]
    rendered_alphas = render_alphas[0].permute(2, 0, 1)

    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    screenspace_points = info["means2d"]

    ##### Convert into our own return format #####
    # concatenate RGB image with alpha image
    rendered_image = torch.cat((rendered_image, rendered_alphas), dim=0)
    depth_image = rendered_depth.squeeze(0)  # (1, H, W) -> (H, W)

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
        "visibility_filter" : radii > 0,
        "radii": radii,
    }

    return return_pkg


def render_3dgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
