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

import os
import sys
from PIL import Image
from typing import NamedTuple
from ..scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from ..utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from ..utils.sh_utils import SH2RGB
from ..scene.gaussian_model import BasicPointCloud

import pickle
import trimesh
import open3d as o3d
import cv2


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):

        # Extract all meshes from the scene
        meshes = []
        for name, geometry in scene_or_mesh.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)

        # Combine all meshes if there are multiple
        if len(meshes) > 1:
            combined_mesh = trimesh.util.concatenate(meshes)
        elif len(meshes) == 1:
            combined_mesh = meshes[0]
        else:
            raise ValueError("No valid meshes found in the GLB file")

        # Get model metadata
        metadata = {
            "vertices": combined_mesh.vertices.shape[0],
            "faces": combined_mesh.faces.shape[0],
            "bounds": combined_mesh.bounds.tolist(),
            "center_mass": combined_mesh.center_mass.tolist(),
            "is_watertight": combined_mesh.is_watertight,
            "original_scene": combined_mesh,  # Keep reference to original scene
        }

        mesh = combined_mesh
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    image: np.array = None
    normal: np.array = None
    depth: np.array = None
    K: np.array = None
    occ_mask: np.array = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(
    cam_extrinsics,
    cam_intrinsics,
    depths_params,
    images_folder,
    depths_folder,
    test_cam_names_list,
):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split(".")[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = (
            os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png")
            if depths_folder != ""
            else ""
        )

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            depth_params=depth_params,
            image_path=image_path,
            image_name=image_name,
            depth_path=depth_path,
            width=width,
            height=height,
            is_test=image_name in test_cam_names_list,
        )
        cam_infos.append(cam_info)

    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array(
                [depths_params[key]["scale"] for key in depths_params]
            )
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(
                f"Error: depth_params.json file not found at path '{depth_params_file}'."
            )
            sys.exit(1)
        except Exception as e:
            print(
                f"An unexpected error occurred when trying to open depth_params.json file: {e}"
            )
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [
                name for idx, name in enumerate(cam_names) if idx % llffhold == 0
            ]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), "r") as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir),
        depths_folder=os.path.join(path, depths) if depths != "" else "",
        test_cam_names_list=test_cam_names_list,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False,
    )
    return scene_info


def readCamerasFromTransforms(
    path, transformsfile, depths_folder, white_background, is_test, extension=".png"
):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            depth_path = (
                os.path.join(depths_folder, f"{image_name}.png")
                if depths_folder != ""
                else ""
            )

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    depth_path=depth_path,
                    depth_params=None,
                    is_test=is_test,
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder = os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", depths_folder, white_background, False, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", depths_folder, white_background, True, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=True,
    )
    return scene_info


# def readQQTTSceneInfo(path, images, depths, eval, train_test_exp, use_masks=False, mesh_path=None):
#     # currently ignore parameter such as: images, depths, eval, train_test_exp

#     # read metadata
#     with open(os.path.join(path, 'metadata.json'), 'r') as f:
#         data = json.load(f)

#     # read cameras
#     intrinsics = np.array(data["intrinsics"])
#     WH = data["WH"]
#     width, height = WH
#     c2ws = pickle.load(open(os.path.join(path, 'calibrate.pkl'), 'rb'))
#     num_cam = len(intrinsics)
#     assert num_cam == len(c2ws), "Number of cameras and camera poses mismatched"

#     cam_infos_unsorted = []
#     for cam_i in range(num_cam):
#         c2w = c2ws[cam_i]
#         K = intrinsics[cam_i]

#         # get the world-to-camera transform and set R, T
#         w2c = np.linalg.inv(c2w)
#         R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#         T = w2c[:3, 3]

#         image_path = os.path.join(path, 'color', str(cam_i), '0.png')
#         image_name = f'cam{cam_i}_0'
#         image = Image.open(image_path) if os.path.exists(image_path) else None

#         # (Optional) use additional masks
#         if use_masks and image is not None:
#             mask_info_path = os.path.join(path, 'mask', f'mask_info_{cam_i}.json')
#             with open(mask_info_path, 'r') as f:
#                 mask_info = json.load(f)                                      # example: {"0": "hand", "1": "twine", "2": "hand"}
#             twine_id = [k for k, v in mask_info.items() if v == "twine"][0]   # assume only one twine
#             mask_path = os.path.join(path, 'mask', str(cam_i), twine_id, '0.png')
#             mask = np.array(Image.open(mask_path))
#             image_rgba = np.concatenate([np.array(image), mask[:, :, None]], axis=-1)
#             image = Image.fromarray(image_rgba)

#         # assume centered principal point at this moment, use K instead
#         focal_length_x = K[0, 0]
#         focal_length_y = K[1, 1]
#         FovY = focal2fov(focal_length_y, height)
#         FovX = focal2fov(focal_length_x, width)

#         # load depth
#         depth_path = os.path.join(path, 'depth', str(cam_i), '0.npy')
#         depth = np.load(depth_path) / 1000.0 if os.path.exists(depth_path) else None  # in mm, convert to m

#         # load normal
#         # normal_path = os.path.join(path, 'normal_omnidata', str(cam_i), '0_normal.png')
#         normal_path = os.path.join(path, 'normal_metric3d', str(cam_i), '0.png')
#         normal = np.array(Image.open(normal_path)) if os.path.exists(normal_path) else None

#         if normal is not None:
#             normal = normal.astype(np.float32) / 255.0  # normalize to [0, 1]
#             normal = (normal - 0.5) * 2                 # normalize to [-1, 1]
#             W2C = getWorld2View2(R, T)
#             C2W = np.linalg.inv(W2C)
#             normal = normal @ C2W[:3, :3].T             # transform normal to world space

#         cam_infos_unsorted.append(CameraInfo(uid=cam_i, R=R, T=T, FovY=FovY, FovX=FovX,
#                             image_path=image_path, image_name=image_name,
#                             width=width, height=height, depth_path="", depth_params=None, is_test=False,
#                             K=K, image=image, normal=normal, depth=depth))

#     test_cam_infos = []
#     test_c2ws = pickle.load(open(os.path.join(path, 'interp_poses.pkl'), 'rb'))
#     for cam_i, c2w in enumerate(test_c2ws):
#         dummy_cam_id = 1
#         K = intrinsics[dummy_cam_id]
#         w2c = np.linalg.inv(c2w)
#         R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#         T = w2c[:3, 3]
#         image_path = os.path.join(path, 'color', str(dummy_cam_id), '0.png')
#         image_name = f'test_cam{cam_i}_0'
#         focal_length_x = K[0, 0]
#         focal_length_y = K[1, 1]
#         FovY = focal2fov(focal_length_y, height)
#         FovX = focal2fov(focal_length_x, width)
#         test_cam_infos.append(CameraInfo(uid=cam_i, R=R, T=T, FovY=FovY, FovX=FovX,
#                             image_path=image_path, image_name=image_name,
#                             width=width, height=height, depth_path="", depth_params=None, is_test=True,
#                             K=K, image=None, normal=None, depth=None))

#     cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
#     train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
#     test_cam_infos = [c for c in test_cam_infos if c.is_test]

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     # read point cloud
#     frame_idx = 0

#     # pcd_xyz_path = os.path.join(path, 'pcd', str(frame_idx), 'points.npy')
#     # pcd_color_path = os.path.join(path, 'pcd', str(frame_idx), 'colors.npy')  # [0-1]
#     # xyz = np.load(pcd_xyz_path)   # [N, 3]
#     # rgb = np.load(pcd_color_path) # [N, 3]

#     if use_masks:
#         data = np.load(os.path.join(path, 'pcd', str(frame_idx), 'first_frame_object.npz'))
#     else:
#         data = np.load(os.path.join(path, 'pcd', str(frame_idx), 'first_frame_total.npz'))
#     xyz = data['points']
#     rgb = data['colors']     # [0-1]
#     normals = np.zeros_like(xyz)

#     # sample init points from mesh if mesh_path is provided
#     if mesh_path:
#         print("Init points from mesh...", mesh_path)
#         xyz, rgb, normals = sample_pcd_from_mesh(mesh_path, POINT_PER_TRIANGLE=30)

#     pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals)

#     ply_path = os.path.join(path, 'pcd', str(frame_idx), 'points3D.ply')  # mimic other two dataloaders
#     storePly(ply_path, xyz, rgb)

#     # return scene info
#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path,
#                            is_nerf_synthetic=False)
#     return scene_info


def readQQTTSceneInfo(
    path,
    images,
    depths,
    eval,
    train_test_exp,
    use_masks=False,
    gs_init_opt="pcd",
    pts_per_triangles=30,
    use_high_res=False,
):
    # currently ignore parameter such as: images, depths, eval, train_test_exp

    # read metadata
    camera_info_path = os.path.join(path, "camera_meta.pkl")
    with open(camera_info_path, "rb") as f:
        camera_info = pickle.load(f)

    # read cameras
    intrinsics = [np.array(intr) for intr in camera_info["intrinsics"]]
    c2ws = camera_info["c2ws"]
    num_cam = len(intrinsics)
    assert num_cam == len(c2ws), "Number of cameras and camera poses mismatched"

    H, W = 480, 848  # fixed resolution

    if use_high_res:
        upsample = 4
        H = int(H * upsample)
        W = int(W * upsample)
        for intr in intrinsics:
            intr[0, 0] *= upsample
            intr[1, 1] *= upsample
            intr[0, 2] *= upsample
            intr[1, 2] *= upsample

    # get camera infos
    cam_infos_unsorted = []
    for cam_i in range(num_cam):
        c2w = c2ws[cam_i]
        K = intrinsics[cam_i]

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        image_path = (
            os.path.join(path, str(cam_i) + ".png")
            if not use_high_res
            else os.path.join(path, str(cam_i) + "_high.png")
        )
        image_name = f"cam{cam_i}"
        image = Image.open(image_path) if os.path.exists(image_path) else None

        # use additional masks
        if use_masks and image is not None:
            mask_path = (
                os.path.join(path, "mask_" + str(cam_i) + ".png")
                if not use_high_res
                else os.path.join(path, "mask_" + str(cam_i) + "_high.png")
            )
            mask = np.array(Image.open(mask_path))
            if len(mask.shape) == 3:
                mask = mask[:, :, -1]  # take the alpha channel
            image_rgba = np.concatenate([np.array(image), mask[:, :, None]], axis=-1)
            image = Image.fromarray(image_rgba)

        # this is dummy term for center principal point assumption (not used)
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, H)
        FovX = focal2fov(focal_length_x, W)

        # load depth
        depth_path = os.path.join(path, str(cam_i) + "_depth.npy")
        depth = (
            np.load(depth_path) / 1000.0 if os.path.exists(depth_path) else None
        )  # in mm, convert to m

        # load normal
        normal_path = os.path.join(path, str(cam_i) + "_normal_metric3d.png")
        normal = (
            np.array(Image.open(normal_path)) if os.path.exists(normal_path) else None
        )

        # load occ mask
        occ_mask_path = (
            os.path.join(path, "mask_human_" + str(cam_i) + ".png")
            if not use_high_res
            else os.path.join(path, "mask_human_" + str(cam_i) + "_high.png")
        )
        occ_mask = (
            np.array(Image.open(occ_mask_path))
            if os.path.exists(occ_mask_path)
            else None
        )
        if occ_mask is not None:
            if len(occ_mask.shape) == 3:
                occ_mask = occ_mask[:, :, -1]  # take the alpha channel
            occ_mask = occ_mask.astype(np.float32) / 255.0
            kernel_size = 8
            occ_mask = cv2.dilate(
                occ_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1
            )  # dilate to avoid boundary artifacts

        if normal is not None:
            normal = normal.astype(np.float32) / 255.0  # normalize to [0, 1]
            normal = (normal - 0.5) * 2  # normalize to [-1, 1]
            W2C = getWorld2View2(R, T)
            C2W = np.linalg.inv(W2C)
            normal = normal @ C2W[:3, :3].T  # transform normal to world space

        cam_infos_unsorted.append(
            CameraInfo(
                uid=cam_i,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image_path=image_path,
                image_name=image_name,
                width=W,
                height=H,
                depth_path="",
                depth_params=None,
                is_test=False,
                K=K,
                image=image,
                normal=normal,
                depth=depth,
                occ_mask=occ_mask,
            )
        )

    test_cam_infos = []
    test_c2ws = pickle.load(open(os.path.join(path, "interp_poses.pkl"), "rb"))
    for cam_i, c2w in enumerate(test_c2ws):
        dummy_cam_id = 1
        K = intrinsics[dummy_cam_id]
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        image_path = os.path.join(path, "color", str(dummy_cam_id), "0.png")
        image_name = f"test_cam{cam_i}_0"
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, H)
        FovX = focal2fov(focal_length_x, W)
        test_cam_infos.append(
            CameraInfo(
                uid=cam_i,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image_path=image_path,
                image_name=image_name,
                width=W,
                height=H,
                depth_path="",
                depth_params=None,
                is_test=True,
                K=K,
                image=None,
                normal=None,
                depth=None,
            )
        )

    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in test_cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # read point cloud ('pcd', 'mesh', 'hybrid')
    all_xyz, all_rgb, all_normals = [], [], []
    if gs_init_opt in ["pcd", "hybrid"]:
        print("Init points from pcd...")
        pcd_path = os.path.join(path, "observation.ply")
        if os.path.exists(pcd_path):
            pcd = o3d.io.read_point_cloud(pcd_path)
            xyz = np.asarray(pcd.points)
            rgb = np.asarray(pcd.colors)
            all_xyz.append(xyz)
            all_rgb.append(rgb)
            all_normals.append(np.zeros((xyz.shape[0], 3)))

    if gs_init_opt in ["mesh", "hybrid"]:
        print("Init points from mesh...")
        mesh_path = os.path.join(path, "shape_prior.glb")
        if os.path.exists(mesh_path):
            xyz, rgb, normals = sample_pcd_from_mesh(
                mesh_path, POINT_PER_TRIANGLE=pts_per_triangles
            )
            all_xyz.append(xyz)
            all_rgb.append(rgb)
            all_normals.append(normals)

    assert len(all_xyz) > 0, "No point cloud or mesh found for initialization"

    all_xyz = np.concatenate(all_xyz, axis=0)
    all_rgb = np.concatenate(all_rgb, axis=0)
    all_normals = np.concatenate(all_normals, axis=0)

    pcd = BasicPointCloud(points=all_xyz, colors=all_rgb, normals=all_normals)

    ply_path = os.path.join(path, "points3D.ply")  # mimic other two dataloaders
    storePly(ply_path, all_xyz, all_rgb, all_normals)

    # return scene info
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False,
    )
    return scene_info


def sample_pcd_from_mesh(mesh_path, POINT_PER_TRIANGLE=5):
    """
    Sample points from uv-textured mesh
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    has_uv_texture = np.asarray(mesh.triangle_uvs).shape[0] != 0
    if has_uv_texture:
        uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)
        texture = np.asarray(mesh.textures[0])
    else:
        vertex_colors = np.asarray(mesh.vertex_colors)
        if vertex_colors.shape[0] != vertices.shape[0]:
            raise ValueError("Mesh has no texture or valid vertex colors.")

    mesh.compute_triangle_normals()
    triangles_normals = np.asarray(mesh.triangle_normals)

    n_triangles = triangles.shape[0]
    total_sample_points = n_triangles * POINT_PER_TRIANGLE

    sampled_points = np.zeros((total_sample_points, 3), dtype=np.float32)
    sampled_colors = np.zeros((total_sample_points, 3), dtype=np.float32)
    sampled_normals = np.zeros((total_sample_points, 3), dtype=np.float32)

    for i in range(n_triangles):
        tri_vertices = vertices[triangles[i]]

        # generate barycentric coordinates
        r1 = np.random.rand(POINT_PER_TRIANGLE)
        r2 = np.random.rand(POINT_PER_TRIANGLE)
        u = 1 - np.sqrt(r1)
        v = r2 * np.sqrt(r1)
        w = 1 - u - v
        barycentric = np.vstack((u, v, w)).T
        points = np.dot(barycentric, tri_vertices)

        if has_uv_texture:
            tri_uvs = uvs[i]
            uv_points = np.dot(barycentric, tri_uvs)
            # convert uv to texture coordinates
            px = np.clip(
                (uv_points[:, 0] * texture.shape[1]).astype(int),
                0,
                texture.shape[1] - 1,
            )
            py = np.clip(
                (uv_points[:, 1] * texture.shape[0]).astype(int),
                0,
                texture.shape[0] - 1,
            )
            colors = texture[py, px] / 255.0
        else:
            # interpolate vertex colors using barycentric coords
            tri_colors = vertex_colors[triangles[i]]
            colors = np.dot(barycentric, tri_colors)

        normals = triangles_normals[i]

        start_idx = i * POINT_PER_TRIANGLE
        end_idx = (i + 1) * POINT_PER_TRIANGLE
        sampled_points[start_idx:end_idx] = points
        sampled_colors[start_idx:end_idx] = colors
        sampled_normals[start_idx:end_idx] = normals

    return sampled_points, sampled_colors, sampled_normals


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "QQTT": readQQTTSceneInfo,
}
