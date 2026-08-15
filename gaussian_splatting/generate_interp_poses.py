import numpy as np
import scipy.interpolate
import pickle
import os


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def generate_interpolated_path(poses: np.ndarray,
                               n_interp: int,
                               spline_degree: int = 5,
                               smoothness: float = .03,
                               rot_weight: float = .1):
    """Creates a smooth spline path between input keyframe camera poses.
    Adapted from https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py
    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
        poses: (n, 3, 4) array of input pose keyframes.
        n_interp: returned path will have n_interp * (n - 1) total poses.
        spline_degree: polynomial degree of B-spline.
        smoothness: parameter for spline smoothing, 0 forces exact interpolation.
        rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
        Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points)


if __name__ == '__main__':
    root_dir = "./data/gaussian_data"
    for scene_name in sorted(os.listdir(root_dir)):
        scene_dir = os.path.join(root_dir, scene_name)
        print(f'Processing {scene_name}')
        camera_path = os.path.join(scene_dir, 'camera_meta.pkl')
        with open(camera_path, 'rb') as f:
            camera_meta = pickle.load(f)
        c2ws = camera_meta['c2ws']
        pose_0 = c2ws[0]
        pose_1 = c2ws[1]
        pose_2 = c2ws[2]
        n_interp = 50
        poses_01 = np.stack([pose_0, pose_1], 0)[:, :3, :]
        interp_poses_01 = generate_interpolated_path(poses_01, n_interp)
        poses_12 = np.stack([pose_1, pose_2], 0)[:, :3, :]
        interp_poses_12 = generate_interpolated_path(poses_12, n_interp)
        poses_20 = np.stack([pose_2, pose_0], 0)[:, :3, :]
        interp_poses_20 = generate_interpolated_path(poses_20, n_interp)
        interp_poses = np.concatenate([interp_poses_01, interp_poses_12, interp_poses_20], 0)
        output_poses = [np.vstack([pose, np.array([0, 0, 0, 1])]) for pose in interp_poses]
        pickle.dump(output_poses, open(os.path.join(scene_dir, 'interp_poses.pkl'), 'wb'))
        