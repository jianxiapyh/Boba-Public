import torch
import kornia
from torch.profiler import profile, ProfilerActivity, record_function
import torch.nn.functional as Func
from typing import Optional

def quat2mat(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.shape[0], 3, 3)).to(q.device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def mat2quat(rot):
    t = torch.clamp(rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2], min=-1)
    q = torch.zeros((rot.shape[0], 4)).to(rot.device)

    mask_0 = t > -1
    t_0 = torch.sqrt(t[mask_0] + 1)
    q[mask_0, 0] = 0.5 * t_0
    t_0 = 0.5 / t_0
    q[mask_0, 1] = (rot[mask_0, 2, 1] - rot[mask_0, 1, 2]) * t_0
    q[mask_0, 2] = (rot[mask_0, 0, 2] - rot[mask_0, 2, 0]) * t_0
    q[mask_0, 3] = (rot[mask_0, 1, 0] - rot[mask_0, 0, 1]) * t_0

    # i = 0, j = 1, k = 2
    mask_1 = ~mask_0 & (rot[:, 0, 0] >= rot[:, 1, 1]) & (rot[:, 0, 0] >= rot[:, 2, 2])
    t_1 = torch.sqrt(1 + rot[mask_1, 0, 0] - rot[mask_1, 1, 1] - rot[mask_1, 2, 2])
    t_1 = 0.5 / t_1
    q[mask_1, 0] = (rot[mask_1, 2, 1] - rot[mask_1, 1, 2]) * t_1
    q[mask_1, 1] = 0.5 * t_1
    q[mask_1, 2] = (rot[mask_1, 1, 0] + rot[mask_1, 0, 1]) * t_1
    q[mask_1, 3] = (rot[mask_1, 2, 0] + rot[mask_1, 0, 2]) * t_1

    # i = 1, j = 2, k = 0
    mask_2 = ~mask_0 & (rot[:, 1, 1] >= rot[:, 2, 2]) & (rot[:, 1, 1] > rot[:, 0, 0])
    t_2 = torch.sqrt(1 + rot[mask_2, 1, 1] - rot[mask_2, 0, 0] - rot[mask_2, 2, 2])
    t_2 = 0.5 / t_2
    q[mask_2, 0] = (rot[mask_2, 0, 2] - rot[mask_2, 2, 0]) * t_2
    q[mask_2, 1] = (rot[mask_2, 2, 1] + rot[mask_2, 1, 2]) * t_2
    q[mask_2, 2] = 0.5 * t_2
    q[mask_2, 3] = (rot[mask_2, 0, 1] + rot[mask_2, 1, 0]) * t_2

    # i = 2, j = 0, k = 1
    mask_3 = ~mask_0 & (rot[:, 2, 2] > rot[:, 0, 0]) & (rot[:, 2, 2] > rot[:, 1, 1])
    t_3 = torch.sqrt(1 + rot[mask_3, 2, 2] - rot[mask_3, 0, 0] - rot[mask_3, 1, 1])
    t_3 = 0.5 / t_3
    q[mask_3, 0] = (rot[mask_3, 1, 0] - rot[mask_3, 0, 1]) * t_3
    q[mask_3, 1] = (rot[mask_3, 0, 2] + rot[mask_3, 2, 0]) * t_3
    q[mask_3, 2] = (rot[mask_3, 1, 2] + rot[mask_3, 2, 1]) * t_3
    q[mask_3, 3] = 0.5 * t_3

    assert torch.allclose(mask_1 + mask_2 + mask_3 + mask_0, torch.ones_like(mask_0))
    return q

def interpolate_motions(bones, motions, relations, xyz, rot=None, quat=None, weights=None, device='cuda', step='n/a'):
    # bones: (n_bones, 3)
    # motions: (n_bones, 3)
    # relations: (n_bones, k)
    # indices: (n_bones,)
    # xyz: (n_particles, 3)
    # rot: (n_particles, 3, 3)
    # quat: (n_particles, 4)
    # weights: (n_particles, n_bones)

    n_bones, _ = bones.shape
    n_particles, _ = xyz.shape

    # Compute the bone transformations
    bone_transforms = torch.zeros((n_bones, 4, 4),  device=device)

    n_adj = relations.shape[1]
    
    adj_bones = bones[relations] - bones[:, None]  # (n_bones, n_adj, 3)
    adj_bones_new = (bones[relations] + motions[relations]) - (bones[:, None] + motions[:, None])  # (n_bones, n_adj, 3)

    W = torch.eye(n_adj, device=device)[None].repeat(n_bones, 1, 1)  # (n_bones, n_adj, n_adj)

    # fit a transformation
    F = adj_bones_new.permute(0, 2, 1) @ W @ adj_bones  # (n_bones, 3, 3)
    
    cov_rank = torch.linalg.matrix_rank(F)  # (n_bones,)
    
    cov_rank_3_mask = cov_rank == 3  # (n_bones,)
    cov_rank_2_mask = cov_rank == 2  # (n_bones,)
    cov_rank_1_mask = cov_rank == 1  # (n_bones,)

    F_2_3 = F[cov_rank_2_mask | cov_rank_3_mask]  # (n_bones, 3, 3)
    F_1 = F[cov_rank_1_mask]  # (n_bones, 3, 3)

    # 2 or 3
    try:
        U, S, V = torch.svd(F_2_3)  # S: (n_bones, 3)
        S = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(F_2_3.shape[0], 1, 1)
        neg_det_mask = torch.linalg.det(F_2_3) < 0
        if neg_det_mask.sum() > 0:
            print(f'[step {step}] F det < 0 for {neg_det_mask.sum()} bones')
            S[neg_det_mask, -1, -1] = -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)
    except:
        print(f'[step {step}] SVD failed')
        import ipdb; ipdb.set_trace()

    neg_1_det_mask = torch.abs(torch.linalg.det(R) + 1) < 1e-3
    pos_1_det_mask = torch.abs(torch.linalg.det(R) - 1) < 1e-3
    bad_det_mask = ~(neg_1_det_mask | pos_1_det_mask)

    if neg_1_det_mask.sum() > 0:
        print(f'[step {step}] det -1')
        S[neg_1_det_mask, -1, -1] *= -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)

    try:
        assert bad_det_mask.sum() == 0
    except:
        print(f'[step {step}] Bad det')
        import ipdb; ipdb.set_trace()

    try:
        if cov_rank_1_mask.sum() > 0:
            print(f'[step {step}] F rank 1 for {cov_rank_1_mask.sum()} bones')
            U, S, V = torch.svd(F_1)  # S: (n_bones', 3)
            assert torch.allclose(S[:, 1:], torch.zeros_like(S[:, 1:]))
            x = torch.tensor([1., 0., 0.], device=device, dtype=torch.float32)[None].repeat(F_1.shape[0], 1)  # (n_bones', 3)
            axis = U[:, :, 0]  # (n_bones', 3)
            perp_axis = torch.linalg.cross(axis, x)  # (n_bones', 3)

            perp_axis_norm_mask = torch.norm(perp_axis, dim=1) < 1e-6

            R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
            if perp_axis_norm_mask.sum() > 0:
                print(f'[step {step}] Perp axis norm 0 for {perp_axis_norm_mask.sum()} bones')
                R[perp_axis_norm_mask] = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(perp_axis_norm_mask.sum(), 1, 1)

            perp_axis = perp_axis[~perp_axis_norm_mask]  # (n_bones', 3)
            x = x[~perp_axis_norm_mask]  # (n_bones', 3)

            perp_axis = perp_axis / torch.norm(perp_axis, dim=1, keepdim=True)  # (n_bones', 3)
            third_axis = torch.linalg.cross(x, perp_axis)  # (n_bones', 3)
            assert ((torch.norm(third_axis, dim=1) - 1).abs() < 1e-6).all()
            third_axis_after = torch.linalg.cross(axis, perp_axis)  # (n_bones', 3)

            X = torch.stack([x, perp_axis, third_axis], dim=-1)
            Y = torch.stack([axis, perp_axis, third_axis_after], dim=-1)
            R[~perp_axis_norm_mask] = Y @ X.permute(0, 2, 1)
    except:
        R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
        R[:, 0, 0] = 1
        R[:, 1, 1] = 1
        R[:, 2, 2] = 1

    try:
        bone_transforms[:, :3, :3] = R
    except:
        print(f'[step {step}] Bad R')
        bone_transforms[:, 0, 0] = 1
        bone_transforms[:, 1, 1] = 1
        bone_transforms[:, 2, 2] = 1
    bone_transforms[:, :3, 3] = motions

    # Compute the weights
    if weights is None:
        weights = torch.ones((n_particles, n_bones), device=device)

        dist = torch.cdist(xyz[None], bones[None])[0]  # (n_particles, n_bones)
        dist = torch.clamp(dist, min=1e-4)
        weights = 1 / dist
        # weights_topk = torch.topk(weights, 5, dim=1, largest=True, sorted=True)
        # weights[weights < weights_topk.values[:, -1:]] = 0.
        weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)
        # weights[weights < 0.01] = 0.
        # weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)
    
    # Compute the transformed particles
    xyz_transformed = torch.zeros((n_particles, n_bones, 3), device=device)

    xyz_transformed = xyz[:, None] - bones[None]  # (n_particles, n_bones, 3)
    # xyz_transformed = (bone_transforms[:, :3, :3][None].repeat(n_particles, 1, 1, 1)\
    #         .reshape(n_particles * n_bones, 3, 3) @ xyz_transformed.reshape(n_particles * n_bones, 3, 1)).reshape(n_particles, n_bones, 3)
    xyz_transformed = torch.einsum('ijk,jkl->ijl', xyz_transformed, bone_transforms[:, :3, :3].permute(0, 2, 1))  # (n_particles, n_bones, 3)
    xyz_transformed = xyz_transformed + bone_transforms[:, :3, 3][None] + bones[None]  # (n_particles, n_bones, 3)
    xyz_transformed = (xyz_transformed * weights[:, :, None]).sum(dim=1)  # (n_particles, 3)

    def quaternion_multiply(q1, q2):
        # q1: bsz x 4
        # q2: bsz x 4
        q = torch.zeros_like(q1)
        q[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
        q[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
        q[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
        q[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
        return q

    if quat is not None:
        # base_quats = kornia.geometry.conversions.rotation_matrix_to_quaternion(bone_transforms[:, :3, :3])  # (n_bones, 4)
        base_quats = mat2quat(bone_transforms[:, :3, :3])  # (n_bones, 4)
        base_quats = torch.nn.functional.normalize(base_quats, dim=-1)  # (n_particles, 4)
        quats = (base_quats[None] * weights[:, :, None]).sum(dim=1)  # (n_particles, 4)
        quats = torch.nn.functional.normalize(quats, dim=-1)
        rot = quaternion_multiply(quats, quat)

    # xyz_transformed: (n_particles, 3)
    # rot: (n_particles, 3, 3) / (n_particles, 4)
    # weights: (n_particles, n_bones)
    return xyz_transformed, rot, weights


def create_relation_matrix(points, K=5):
    """
    Create an NxN relation matrix where each row has 1s for the top K closest neighbors and 0s elsewhere.
    
    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing 3D points.
        K (int): Number of closest neighbors to mark as 1.
        
    Returns:
        torch.Tensor: NxN relation matrix with dtype int.
    """
    N = points.shape[0]

    # Compute pairwise squared Euclidean distances
    dist_matrix = torch.cdist(points, points, p=2)  # (N, N)

    # Get the indices of the top K closest neighbors (excluding self)
    topk_indices = torch.topk(dist_matrix, K + 1, largest=False).indices[:, 1:]  # Skip self (0 distance)

    # Create the NxN relation matrix
    relation_matrix = torch.zeros((N, N), dtype=torch.int)

    # Scatter 1s for the top K neighbors
    batch_indices = torch.arange(N).unsqueeze(1).expand(-1, K)
    relation_matrix[batch_indices, topk_indices] = 1

    return relation_matrix


def get_topk_indices(points, K=5):
    """
    Compute the indices of the top K closest neighbors for each point.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing 3D points.
        K (int): Number of closest neighbors to retrieve.

    Returns:
        torch.Tensor: Tensor of shape (N, K) containing the indices of the top K closest neighbors.
    """
    # Compute pairwise squared Euclidean distances
    dist_matrix = torch.cdist(points, points, p=2)  # (N, N)

    # Get the indices of the top K closest neighbors (excluding self)
    topk_indices = torch.topk(dist_matrix, K + 1, largest=False).indices[:, 1:]  # Skip self (0 distance)

    return topk_indices


def knn_weights(bones, pts, K=5):
    dist = torch.norm(pts[:, None] - bones, dim=-1)  # (n_pts, n_bones)
    _, indices = torch.topk(dist, K, dim=-1, largest=False)
    bones_selected = bones[indices]  # (N, k, 3)
    dist = torch.norm(bones_selected - pts[:, None], dim=-1)  # (N, k)
    weights = 1 / (dist + 1e-6)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # (N, k)
    weights_all = torch.zeros((pts.shape[0], bones.shape[0]), device=pts.device)  # TODO: prevent init new one
    # weights_all[torch.arange(pts.shape[0])[:, None], indices] = weights
    weights_all[torch.arange(pts.shape[0], device=pts.device)[:, None], indices] = weights
    return weights_all



def calc_weights_vals_from_indices(bones, pts, indices):
    # bones: (n_bones, 3)
    # pts: (n_particles, 3)
    # indices: (n_particles, k) indices of k nearest bones per particle

    nearest_bones = bones[indices]  # (n_particles, k, 3)
    pts_expanded = pts.unsqueeze(1)  # (n_particles, 1, 3)
    distances = torch.norm(pts_expanded - nearest_bones, dim=2)
    weights_vals = 1.0 / (distances + 1e-6)
    weights_vals = weights_vals / weights_vals.sum(dim=1, keepdim=True)  # (n_particles, k)    
    return weights_vals


def knn_weights_sparse(bones, pts, K=5):
    dist = torch.norm(pts[:, None].cpu() - bones.cpu(), dim=-1)  # (n_pts, n_bones)
    weights_vals, indices = torch.topk(dist, K, dim=-1, largest=False)
    weights_vals = weights_vals.to(pts.device)
    indices = indices.to(pts.device)
    weights_vals = 1 / (weights_vals + 1e-6)
    weights_vals = weights_vals / weights_vals.sum(dim=-1, keepdim=True)  # (N, k)
    torch.cuda.empty_cache()
    return weights_vals, indices

def rotmat_to_quat_fast(R: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Vectorized rotation-matrix -> quaternion (scalar-first [w,x,y,z]).
    R: (..., 3, 3) orthonormal rotation matrices
    Returns: (..., 4) unit quaternions with w >= 0 (canonical hemisphere)
    """
    R = R.contiguous()
    m00 = R[..., 0, 0]; m01 = R[..., 0, 1]; m02 = R[..., 0, 2]
    m10 = R[..., 1, 0]; m11 = R[..., 1, 1]; m12 = R[..., 1, 2]
    m20 = R[..., 2, 0]; m21 = R[..., 2, 1]; m22 = R[..., 2, 2]

    trace = m00 + m11 + m22
    t_mask = trace > 0

    # Case 0: trace > 0
    t0 = (trace + 1.0).clamp_min(eps)
    s0 = torch.sqrt(t0)
    w0 = 0.5 * s0
    inv4w = 0.25 / (w0 + eps)
    x0 = (m21 - m12) * inv4w
    y0 = (m02 - m20) * inv4w
    z0 = (m10 - m01) * inv4w

    # Find index of largest diagonal for other cases
    diag = torch.stack([m00, m11, m22], dim=-1)  # (..., 3)
    i = diag.argmax(dim=-1)

    # Case 1: m00 is largest
    t1 = (1.0 + m00 - m11 - m22).clamp_min(eps)
    s1 = torch.sqrt(t1)
    x1 = 0.5 * s1
    inv4x = 0.25 / (x1 + eps)
    w1 = (m21 - m12) * inv4x
    y1 = (m01 + m10) * inv4x
    z1 = (m02 + m20) * inv4x

    # Case 2: m11 is largest
    t2 = (1.0 + m11 - m00 - m22).clamp_min(eps)
    s2 = torch.sqrt(t2)
    y2 = 0.5 * s2
    inv4y = 0.25 / (y2 + eps)
    w2 = (m02 - m20) * inv4y
    x2 = (m01 + m10) * inv4y
    z2 = (m12 + m21) * inv4y

    # Case 3: m22 is largest
    t3 = (1.0 + m22 - m00 - m11).clamp_min(eps)
    s3 = torch.sqrt(t3)
    z3 = 0.5 * s3
    inv4z = 0.25 / (z3 + eps)
    w3 = (m10 - m01) * inv4z
    x3 = (m02 + m20) * inv4z
    y3 = (m12 + m21) * inv4z

    # Assemble candidates
    q_t = torch.stack([w0, x0, y0, z0], dim=-1)
    q_0 = torch.stack([w1, x1, y1, z1], dim=-1)
    q_1 = torch.stack([w2, x2, y2, z2], dim=-1)
    q_2 = torch.stack([w3, x3, y3, z3], dim=-1)

    # Select by masks: first prefer trace>0, else by argmax(diag)
    # Build selection for non-trace branch
    sel0 = (i == 0)
    sel1 = (i == 1)
    sel2 = (i == 2)
    q_sel = torch.where(sel0.unsqueeze(-1), q_0, torch.where(sel1.unsqueeze(-1), q_1, q_2))
    q = torch.where(t_mask.unsqueeze(-1), q_t, q_sel)

    # Canonicalize hemisphere (w >= 0), then normalize once
    sign = torch.where(q[..., :1] >= 0, 1.0, -1.0)
    q = q * sign
    q = Func.normalize(q, dim=-1, eps=eps)
    return q

def _quat_mul(q1, q2):
    # q1,q2: (..., 4) scalar-first
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)
    # Stack once; avoids zeros_like + indexing overhead
    return torch.stack((
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ), dim=-1)


def interpolate_motions_speedup_rotation_reuse(
    bones, motions, relations, xyz, rot=None, quat=None,
    weights=None, weights_indices=None, device='cuda', step='n/a',
    R_cache=None, F_prev=None, valid=None,
    tau_F=2e-3,
    static_cache=None,
):
    assert static_cache is not None, "Pass the precomputed static cache"

    device = bones.device
    active   = static_cache["active"]              # (M,)
    wi_local = static_cache["wi_local_T"]            # (N,k)
    rel_act  = static_cache["rel_act"]             # (M,n_adj)
    used_local = static_cache["used_local"]      # (M,) bool

    # ----- Recompute geometry this frame (bones move!) -----
    bones_act = static_cache["bones_act"]                 # rest
    B = static_cache["adj_bones_rest_act"].contiguous()
    motions_act = motions.index_select(0, active)
    nb_motions  = motions.index_select(0, rel_act.reshape(-1)).reshape(rel_act.shape + (3,))
    adj_bones_new = B + nb_motions - motions_act[:, None, :]  # (M,n_adj,3)


    # --- Compute F for active bones ---
    with record_function("LBS/compute-F(active)"):
        A = adj_bones_new.transpose(1, 2).contiguous()  # (M,3,n_adj)
        F_act = torch.bmm(A, B)                         # (M,3,3)

    if F_act.dtype != torch.float32:
        F_act = F_act.to(torch.float32)
        
    # --- Reuse vs recompute ---
    with record_function("LBS/Polar-reuse(active)"):
        dF = torch.linalg.matrix_norm(
            F_act - F_prev.index_select(0, active), ord='fro', dim=(-2, -1)
        )

        #for debugging
        # v = valid.index_select(0, active)
        # print(f"[telemetry] valid={int(v.sum())}/{active.numel()}  used_local={int(used_local.sum())}/{active.numel()}")

        # dF = torch.linalg.matrix_norm(F_act - F_prev.index_select(0, active), ord='fro', dim=(-2,-1))
        # print(f"[telemetry] dF min/med/max = {dF.min().item():.3e} / {dF.median().item():.3e} / {dF.max().item():.3e}")
        # print(f"[telemetry] (dF < tau_F) = {int((dF < tau_F).sum())}/{dF.numel()}")

        reuse_mask = valid.index_select(0, active) & used_local & (dF < tau_F)

        # Optional det guard only on candidates
        # cand = (reuse_mask & used_local).nonzero(as_tuple=False).squeeze(1)
        # if cand.numel():
        #     ok = (torch.linalg.det(F_act.index_select(0, cand)) > 1e-7)
        #     reuse_mask[cand] = ok

        # Start from cache; overwrite only rows that changed
        R_act = R_cache.index_select(0, active).clone()          # (M,3,3)

        rec_idx_local = (~reuse_mask).nonzero(as_tuple=False).squeeze(1)
        if rec_idx_local.numel():
            F_rec = F_act.index_select(0, rec_idx_local)         # (m,3,3)

            # Polar via eig (keep your path)  --- you can swap to SVD if you prefer
            X = F_rec
            G = X.transpose(-2, -1) @ X
            G = 0.5 * (G + G.transpose(-2, -1))
            w, V = torch.linalg.eigh(G)
            idx = torch.argsort(w, dim=-1, descending=True)
            w   = w.gather(-1, idx)
            V   = V.gather(-1, idx.unsqueeze(-2).expand_as(V))
            vmax_idx = V.abs().argmax(dim=-2, keepdim=True)
            sgn = torch.sign(V.gather(-2, vmax_idx))
            V = V * sgn
            Svals = w.clamp_min(1e-12).sqrt()
            Sinv  = torch.diag_embed(1.0 / Svals)
            U     = (X @ V) @ Sinv
            Vh    = V.transpose(-2, -1)
            need_flip = (torch.linalg.det(U) * torch.linalg.det(Vh)) < 0
            if need_flip.any():
                U[need_flip, :, 2] *= -1
            R_rec = (U @ Vh).to(F_act.dtype)                      # (m,3,3)

            # Write back: local frame + global caches
            rec_idx_global = active.index_select(0, rec_idx_local)
            R_act.index_copy_(0, rec_idx_local, R_rec)
            R_cache.index_copy_(0, rec_idx_global, R_rec)
            F_prev.index_copy_(0, rec_idx_global, F_rec)
            valid.index_fill_(0, rec_idx_global, True)
            used_local.index_fill_(0, rec_idx_local, True)  

        #for debugging
        #print(f"[dbg] recomputed {rec_idx_local.numel()} / {active.numel()} bones this frame")
        # # Check how many cached rotations are still identity *after* the update:
        # I = torch.eye(3, device=R_cache.device, dtype=R_cache.dtype)
        # # Among ACTIVE bones, how many rotations are (almost) identity?
        # R_active = R_cache.index_select(0, active)                                  # (M,3,3)
        # is_eye_rows = torch.all(
        #     torch.isclose(R_active, I.expand_as(R_active), atol=1e-6, rtol=0.0),
        #     dim=(1, 2)
        # )                                                                            # (M,)
        # num_eye = int(is_eye_rows.sum())
        # print(f"[dbg] identity-like rotations among active: {num_eye} / {active.numel()}")

    # --- Gather per-particle neighbors (from active subset) ---
    with record_function("LBS/gather-kNN"):
        R_sel = R_act[wi_local]            # (N,k,3,3)
        t_sel = motions_act[wi_local]      # (N,k,3)
        b_sel = bones_act[wi_local]        # (N,k,3)

    # --- Transform & blend ---
    with record_function("LBS/transform+blend"):
        xyz_local = xyz[:, None, :] - b_sel                        # (N,k,3)
        rotated   = torch.einsum('nkij,nkj->nki', R_sel, xyz_local)# (N,k,3)
        xyz_out   = ((rotated + t_sel + b_sel) * weights[..., None]).sum(dim=1)

    # Optional quaternion output (unchanged from your version)
    ret_rot = rot
    if quat is not None:
        with record_function("LBS/quats"):
            bone_quats_act = rotmat_to_quat_fast(R_act)           # (M,4)   
            base_quats = bone_quats_act[wi_local]                 # (N,k,4)
            quats = (base_quats * weights[..., None]).sum(dim=1)
            quats = Func.normalize(quats, dim=-1, eps=1e-12)
            ret_rot = Func.normalize(_quat_mul(quats, quat), dim=-1, eps=1e-12)

    return xyz_out, ret_rot, (weights, weights_indices)

def interpolate_motions_speedup(bones, motions, relations, xyz, rot=None, quat=None, weights=None, weights_indices=None, device='cuda', step='n/a'):
    # bones: (n_bones, 3) bone positions
    # motions: (n_bones, 3) bone motions/displacements
    # relations: (n_bones, k_adj) bone adjacency relationships - which bones are connected to each other
    # xyz: (n_particles, 3) particle positions
    # weights: (n_particles, k) weights for k nearest bones per particle
    # weights_indices: (n_particles, k) indices of k nearest bones per particle
    # rot: (n_particles, 3, 3) optional rotation matrices
    # quat: (n_particles, 4) optional quaternions

    n_bones, _ = bones.shape
    n_particles, k_nearest = xyz.shape

    # Compute the bone transformations
    bone_transforms = torch.zeros((n_bones, 4, 4),  device=device)

    n_adj = relations.shape[1]
    
    adj_bones = bones[relations] - bones[:, None]  # (n_bones, n_adj, 3)
    adj_bones_new = (bones[relations] + motions[relations]) - (bones[:, None] + motions[:, None])  # (n_bones, n_adj, 3)

    W = torch.eye(n_adj, device=device)[None].repeat(n_bones, 1, 1)  # (n_bones, n_adj, n_adj)

    # fit a transformation
    F = adj_bones_new.permute(0, 2, 1) @ W @ adj_bones  # (n_bones, 3, 3)
    
    cov_rank = torch.linalg.matrix_rank(F)  # (n_bones,)
    
    cov_rank_3_mask = cov_rank == 3  # (n_bones,)
    cov_rank_2_mask = cov_rank == 2  # (n_bones,)
    cov_rank_1_mask = cov_rank == 1  # (n_bones,)
    num_rank1 = int(cov_rank_1_mask.sum().item())
    print(f"[step {step}] F rank 1 for {num_rank1} bones")

    F_2_3 = F[cov_rank_2_mask | cov_rank_3_mask]  # (n_bones, 3, 3)
    F_1 = F[cov_rank_1_mask]  # (n_bones, 3, 3)

    # 2 or 3
    try:
        U, S, V = torch.svd(F_2_3)  # S: (n_bones, 3)
        S = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(F_2_3.shape[0], 1, 1)
        neg_det_mask = torch.linalg.det(F_2_3) < 0
        if neg_det_mask.sum() > 0:
            print(f'[step {step}] F det < 0 for {neg_det_mask.sum()} bones')
            S[neg_det_mask, -1, -1] = -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)
    except:
        print(f'[step {step}] SVD failed')
        import ipdb; ipdb.set_trace()

    neg_1_det_mask = torch.abs(torch.linalg.det(R) + 1) < 1e-3
    pos_1_det_mask = torch.abs(torch.linalg.det(R) - 1) < 1e-3
    bad_det_mask = ~(neg_1_det_mask | pos_1_det_mask)

    if neg_1_det_mask.sum() > 0:
        print(f'[step {step}] det -1')
        S[neg_1_det_mask, -1, -1] *= -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)

    try:
        assert bad_det_mask.sum() == 0
    except:
        print(f'[step {step}] Bad det')
        import ipdb; ipdb.set_trace()

    try:
        if cov_rank_1_mask.sum() > 0:
            print(f'[step {step}] F rank 1 for {cov_rank_1_mask.sum()} bones')
            U, S, V = torch.svd(F_1)  # S: (n_bones', 3)
            assert torch.allclose(S[:, 1:], torch.zeros_like(S[:, 1:]))
            x = torch.tensor([1., 0., 0.], device=device, dtype=torch.float32)[None].repeat(F_1.shape[0], 1)  # (n_bones', 3)
            axis = U[:, :, 0]  # (n_bones', 3)
            perp_axis = torch.linalg.cross(axis, x)  # (n_bones', 3)

            perp_axis_norm_mask = torch.norm(perp_axis, dim=1) < 1e-6

            R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
            if perp_axis_norm_mask.sum() > 0:
                print(f'[step {step}] Perp axis norm 0 for {perp_axis_norm_mask.sum()} bones')
                R[perp_axis_norm_mask] = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(perp_axis_norm_mask.sum(), 1, 1)

            perp_axis = perp_axis[~perp_axis_norm_mask]  # (n_bones', 3)
            x = x[~perp_axis_norm_mask]  # (n_bones', 3)

            perp_axis = perp_axis / torch.norm(perp_axis, dim=1, keepdim=True)  # (n_bones', 3)
            third_axis = torch.linalg.cross(x, perp_axis)  # (n_bones', 3)
            assert ((torch.norm(third_axis, dim=1) - 1).abs() < 1e-6).all()
            third_axis_after = torch.linalg.cross(axis, perp_axis)  # (n_bones', 3)

            X = torch.stack([x, perp_axis, third_axis], dim=-1)
            Y = torch.stack([axis, perp_axis, third_axis_after], dim=-1)
            R[~perp_axis_norm_mask] = Y @ X.permute(0, 2, 1)
    except:
        R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
        R[:, 0, 0] = 1
        R[:, 1, 1] = 1
        R[:, 2, 2] = 1

    try:
        bone_transforms[:, :3, :3] = R
    except:
        print(f'[step {step}] Bad R')
        bone_transforms[:, 0, 0] = 1
        bone_transforms[:, 1, 1] = 1
        bone_transforms[:, 2, 2] = 1
    bone_transforms[:, :3, 3] = motions

    # Compute the weights
    # if weights is None:
    #     weights = torch.ones((n_particles, n_bones), device=device)

    #     dist = torch.cdist(xyz[None], bones[None])[0]  # (n_particles, n_bones)
    #     dist = torch.clamp(dist, min=1e-4)
    #     weights = 1 / dist
    #     # weights_topk = torch.topk(weights, 5, dim=1, largest=True, sorted=True)
    #     # weights[weights < weights_topk.values[:, -1:]] = 0.
    #     weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)
    #     # weights[weights < 0.01] = 0.
    #     # weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)
    
    # Compute the transformed particles
    # xyz_transformed = torch.zeros((n_particles, n_bones, 3), device=device)

    # xyz_transformed = xyz[:, None] - bones[None]  # (n_particles, n_bones, 3)
    # # xyz_transformed = (bone_transforms[:, :3, :3][None].repeat(n_particles, 1, 1, 1)\
    # #         .reshape(n_particles * n_bones, 3, 3) @ xyz_transformed.reshape(n_particles * n_bones, 3, 1)).reshape(n_particles, n_bones, 3)
    # xyz_transformed = torch.einsum('ijk,jkl->ijl', xyz_transformed, bone_transforms[:, :3, :3].permute(0, 2, 1))  # (n_particles, n_bones, 3)
    # xyz_transformed = xyz_transformed + bone_transforms[:, :3, 3][None] + bones[None]  # (n_particles, n_bones, 3)
    # xyz_transformed = (xyz_transformed * weights[:, :, None]).sum(dim=1)  # (n_particles, 3)


    selected_bones = bones[weights_indices]  # (n_particles, k, 3)
    selected_transforms = bone_transforms[weights_indices]  # (n_particles, k, 4, 4)

    # Transform each point with only its k nearest bones
    # xyz_expanded = xyz[:, None].unsqueeze(1).expand(-1, k_nearest, -1)  # (n_particles, k, 3)
    # xyz_local = xyz_expanded - selected_bones  # (n_particles, k, 3)
    xyz_local = xyz.unsqueeze(1) - selected_bones  # (n_particles, k, 3)
    
    # Apply rotation to local coordinates 
    rotated_local = torch.einsum('nkij,nkj->nki', selected_transforms[:, :, :3, :3], xyz_local)  # (n_particles, k, 3)
    
    # Apply translation and add back bone positions
    transformed_pts = rotated_local + selected_transforms[:, :, :3, 3] + selected_bones  # (n_particles, k, 3)
    
    # Apply weights to get final positions
    xyz_transformed = torch.sum(transformed_pts * weights[:, :, None], dim=1)  # (n_particles, 3)


    def quaternion_multiply(q1, q2):
        # q1: bsz x 4
        # q2: bsz x 4
        q = torch.zeros_like(q1)
        q[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
        q[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
        q[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
        q[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
        return q

    if quat is not None:
        # base_quats = kornia.geometry.conversions.rotation_matrix_to_quaternion(bone_transforms[:, :3, :3])  # (n_bones, 4)
        # base_quats = mat2quat(bone_transforms[:, :3, :3])  # (n_bones, 4)
        # base_quats = torch.nn.functional.normalize(base_quats, dim=-1)  # (n_particles, 4)
        # quats = (base_quats[None] * weights[:, :, None]).sum(dim=1)  # (n_particles, 4)
        # quats = torch.nn.functional.normalize(quats, dim=-1)

        from kornia.geometry.conversions import rotation_matrix_to_quaternion

        selected_rot_matrices = selected_transforms[:, :, :3, :3]  # (n_particles, k, 3, 3)
        n_particles, k_weights = weights_indices.shape
        batch_rot_matrices = selected_rot_matrices.reshape(-1, 3, 3)  # (n_particles*k, 3, 3)
        
        try:
            base_quats = rotation_matrix_to_quaternion(batch_rot_matrices)  # (n_particles*k, 4)
        except:
            print('use mat2quat')
            base_quats = mat2quat(batch_rot_matrices)  # (n_particles*k, 4)
            
        base_quats = base_quats.reshape(n_particles, k_weights, 4)  # (n_particles, k, 4)
        base_quats = torch.nn.functional.normalize(base_quats, dim=-1)
        quats = torch.sum(base_quats * weights[:, :, None], dim=1)  # (n_particles, 4)
        quats = torch.nn.functional.normalize(quats, dim=-1)

        rot = quaternion_multiply(quats, quat)

    # Return sparse weights representation for reuse
    weights_sparse = (weights, weights_indices)

    # xyz_transformed: (n_particles, 3)
    # rot: (n_particles, 3, 3) / (n_particles, 4)
    # weights: (n_particles, n_bones)
    return xyz_transformed, rot, weights_sparse

def polar_ns(F, iters=3, eps=1e-6):
    # F: (B,3,3) float32
    B = F.shape[0]
    I = torch.eye(3, device=F.device, dtype=F.dtype).expand(B,3,3)
    # scale for convergence
    X = F / F.norm(dim=(-2,-1), keepdim=True).clamp_min(eps)
    for _ in range(iters):
        XtX = X.transpose(-2, -1) @ X
        X = 0.5 * (X @ (3*I - XtX))
    R = X
    # Proper rotation fix (det>0)
    flip = torch.linalg.det(R) < 0
    if flip.any():
        R[flip, :, 2] *= -1
    return R


def update_bone_rotations_only(bones, motions, static_bones, R_cache, F_prev, valid, tau_F=5e-3):
    active  = static_bones["active"]      # (M',)
    rel_act = static_bones["rel_act"]     # (M', n_adj)

    bones_act  = bones.index_select(0, active)
    nb_bones   = bones.index_select(0, rel_act.reshape(-1)).reshape(rel_act.shape + (3,))
    adj_bones  = nb_bones - bones_act[:, None, :]

    motions_act = motions.index_select(0, active)
    nb_motions  = motions.index_select(0, rel_act.reshape(-1)).reshape(rel_act.shape + (3,))
    adj_bones_new = adj_bones + nb_motions - motions_act[:, None, :]

    A = adj_bones_new.transpose(1, 2).contiguous()
    B = adj_bones.contiguous()
    F_act = torch.bmm(A, B)

    dF = torch.linalg.matrix_norm(F_act - F_prev.index_select(0, active), ord='fro', dim=(-2,-1))
    reuse = (valid.index_select(0, active) & (dF < tau_F))

    R_act = R_cache.index_select(0, active).clone()
    rec_idx_local = (~reuse).nonzero(as_tuple=False).squeeze(1)
    if rec_idx_local.numel():
        F_rec = F_act.index_select(0, rec_idx_local)
        U,S,Vh = torch.linalg.svd(F_rec, full_matrices=False)
        need_flip = (torch.linalg.det(U) * torch.linalg.det(Vh)) < 0
        if need_flip.any(): Vh[need_flip, 2, :] *= -1
        R_rec = (U @ Vh).to(F_act.dtype)

        rec_idx_global = active.index_select(0, rec_idx_local)
        R_act.index_copy_(0, rec_idx_local, R_rec)
        R_cache.index_copy_(0, rec_idx_global, R_rec)
        F_prev.index_copy_(0, rec_idx_global, F_rec)
        valid.index_fill_(0, rec_idx_global, True)

    return active, bones_act, motions_act, R_act


@torch.no_grad()
def build_rotation_reuse_cache(
    #these are shared
    weights_indices: torch.Tensor,
    weights: torch.Tensor,
    relations: torch.Tensor,
    mass_nodes_rest: torch.Tensor,
    gaussians_xyz_rest: torch.Tensor,
    gaussians_quat_rest: torch.Tensor,
    device: torch.device,
    mass_node_per_instance: int,
    gaussians_per_instance: int,
    number_of_instance: int,
):
    assert weights_indices.device == mass_nodes_rest.device
    assert relations.device == mass_nodes_rest.device
    assert weights_indices.dtype in (torch.int32, torch.int64)
    assert relations.dtype in (torch.int32, torch.int64)
    """
    Build cache for LBS optimization.
    Output: cache: Dictionary containing all static and dynamic data for rotation reuse
    """  
    object_mass_node_total = number_of_instance * mass_node_per_instance  
    # Precompute rest-pose bone-to-neighbor vectors
    rest_bone_to_neighbors = (
        mass_nodes_rest[relations] - mass_nodes_rest[:, None, :]
    ).contiguous()  # (B, n_adj, 3)
    
    # FP16 version
    R_cache = torch.eye(3, device=device, dtype=torch.float16).repeat(object_mass_node_total, 1, 1).contiguous()
    
    F_prev = torch.zeros(object_mass_node_total, 3, 3, device=device, dtype=torch.float32).contiguous()
    rotation_computed = torch.zeros(object_mass_node_total, device=device, dtype=torch.bool).contiguous()
    
    weights_f32 = weights.to(device=device, dtype=torch.float32).contiguous()
    weights_f16 = weights.to(device=device, dtype=torch.float16).contiguous()

    bones_rest_per_gaussian = mass_nodes_rest[weights_indices]        # (Ng,K,3)
    xyz_local = gaussians_xyz_rest[:,None,:] - bones_rest_per_gaussian # (Ng,K,3)

    #newly added static data
    xyz_local_w = (xyz_local.to(torch.float16) * weights_f16.unsqueeze(-1)).contiguous()    # fp16
    
    bones_rest_blend = (bones_rest_per_gaussian.to(torch.float16) * weights_f16.unsqueeze(-1)).sum(dim=1).contiguous()
    # basically store quaternion value if rotation is the same (in bone major format)    
    Q_cache_bm = torch.zeros(mass_node_per_instance, number_of_instance, 4, device=device, dtype=torch.float16)
    Q_cache_bm[..., 0] = 1.0
    Q_cache_bm = Q_cache_bm.contiguous()

    motions_bm = torch.empty(mass_node_per_instance, number_of_instance, 3,
                         device=device, dtype=torch.float32).contiguous()

    def build_W_csr(weights_indices_ik: torch.Tensor, weights_ik: torch.Tensor, Nb_: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        Ng_, K_ = weights_indices_ik.shape
        row = torch.arange(Ng_, device=device, dtype=torch.int64).repeat_interleave(K_)  # (Ng*K,)
        col = weights_indices_ik.reshape(-1).to(device=device, dtype=torch.int64)                                      # (Ng*K,)
        #updated to use float16
        val = weights_ik.reshape(-1).to(device=device, dtype=dtype) 

        W = torch.sparse_coo_tensor(
            torch.stack([row, col], dim=0),
            val,
            size=(Ng_, Nb_),
            device=device,
            #updated to use float16
            dtype=dtype,
        ).coalesce()
        return W.to_sparse_csr()
    weights_indices_i64 = weights_indices.to(torch.int64).contiguous()

    W_csr_f32 = build_W_csr(weights_indices_i64, weights_f32, mass_node_per_instance, dtype=torch.float32)  # dtype=float32 inside
    W_csr_f16 = build_W_csr(weights_indices_i64, weights_f16, mass_node_per_instance, dtype=torch.float16)  # dtype=float16 inside

    gs_rest = Func.normalize(gaussians_quat_rest, dim=-1, eps=1e-6).to(device=device, dtype=torch.float16).contiguous()


    cache = {
        # === Static geometry (never changes) ===
        "mass_nodes_rest": mass_nodes_rest,  # (B, 3) Rest bone positions
        "gaussians_quat_rest": gs_rest,  # (N, 4) Rest Gaussian quaternions
        "relations": relations,  # (B, n_adj) Bone adjacency graph
        "weights_indices": weights_indices,  # (N, K) Bone indices per Gaussian
        "rest_bone_to_neighbors": rest_bone_to_neighbors,  # (B, n_adj, 3) Precomputed rest vectors
        "mass_nodes_per_instance": mass_node_per_instance,
        "gaussians_per_instance": gaussians_per_instance,
        "number_of_instance": number_of_instance,
        "xyz_local_w": xyz_local_w,
        "bones_rest_blend": bones_rest_blend,
        "W_csr_f32": W_csr_f32,  # Sparse weight matrix in CSR format for fast multiplication
        "W_csr_f16": W_csr_f16,  # Sparse weight matrix in CSR format for fast multiplication
        # === Dynamic state (modified each frame) ===
        # batched vector cover all instances
        "R_cache": R_cache,  # (B, 3, 3) Cached rotation matrices
        "F_prev": F_prev,  # (B, 3, 3) Previous deformation gradients
        "rotation_computed": rotation_computed,  # (B,) Has rotation been computed at least once?
        "Q_cache_bm": Q_cache_bm,
        "motions_bm_fp32": motions_bm,
    }
    
    return cache

# @torch.compile(dynamic=True)
def quat_mul_norm_fused(q, rest):
    # q: (Ng,I,4)  rest: (Ng,1,4)
    qw, qx, qy, qz = q.unbind(-1)
    rw, rx, ry, rz = rest.unbind(-1)

    quat = torch.stack([
        qw*rw - qx*rx - qy*ry - qz*rz,
        qw*rx + qx*rw + qy*rz - qz*ry,
        qw*ry - qx*rz + qy*rw + qz*rx,
        qw*rz + qx*ry - qy*rx + qz*rw,
    ], dim=-1)

    inv = torch.rsqrt((quat * quat).sum(dim=-1, keepdim=True) + 1e-6)
    return quat * inv

#pyh updated for batched version
@torch.no_grad()
def lbs_with_rotation_reuse(
    current_mass_nodes: torch.Tensor,  # This now contains all mass nodes across all instances
    cache: dict,                       # rotation_cache with all state
    tau_F: float = 5e-5,                # threshod need to tested, 5e-5 seems a little bit bad for visualiazation 
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear Blend Skinning with rotation matrix caching and reuse."""
    
    # Unpack cache 
    mass_nodes_rest = cache["mass_nodes_rest"]
    gaussians_quat_rest = cache["gaussians_quat_rest"]
    relations = cache["relations"]
    weights_indices = cache["weights_indices"]
    rest_bone_to_neighbors = cache["rest_bone_to_neighbors"]
    mass_nodes_per_instance = cache["mass_nodes_per_instance"]
    gaussians_per_instance = cache["gaussians_per_instance"]
    number_of_instance = cache["number_of_instance"]
    R_cache = cache["R_cache"]
    F_prev = cache["F_prev"]
    rotation_computed = cache["rotation_computed"]
    
    current_mass_nodes_by_instance = current_mass_nodes.reshape(number_of_instance, mass_nodes_per_instance, 3)

    mass_node_rest_template = mass_nodes_rest.view(1, mass_nodes_per_instance, 3)
    motions = current_mass_nodes_by_instance - mass_node_rest_template

    neighbor_motions = motions[:, relations]  # (inst, bones, k,3)
    rest_bone_to_neighbors_template = rest_bone_to_neighbors.view(1, mass_nodes_per_instance,relations.shape[1],3)

    # Compute current vectors from each bone to its neighbors
    current_bone_to_neighbors = (
        rest_bone_to_neighbors_template + neighbor_motions - motions[:,:, None, :]
    )  


    # --- Compute deformation gradient F ---
    F = torch.einsum("ibja,bjc->ibac", current_bone_to_neighbors, rest_bone_to_neighbors)
    
    F = F.reshape(number_of_instance * mass_nodes_per_instance, 3, 3).contiguous()             # (I*Nb, 3, 3)

    if F.dtype != torch.float32:
        F = F.to(torch.float32)

    # Compute change in deformation gradient from previous frame
    dF = torch.linalg.matrix_norm(F - F_prev, ord='fro', dim=(-2, -1))  

    # For debugging
    # print(f"[telemetry] valid={int(valid.sum())}/{valid.numel()}  rotation_computed={int(rotation_computed.sum())}/{rotation_computed.numel()}")
    # print(f"[telemetry] dF min/med/max = {dF.min().item():.3e} / {dF.median().item():.3e} / {dF.max().item():.3e}")
    # print(f"[telemetry] (dF < tau_F) = {int((dF < tau_F).sum())}/{dF.numel()}")

    # Reuse rotation if: valid cache AND rotation computed before AND change is small
    can_reuse_rotation = rotation_computed & (dF < tau_F)

    # Find bones that need rotation recomputation
    bones_to_recompute = (~can_reuse_rotation).nonzero(as_tuple=False).squeeze(1)
    
    if bones_to_recompute.numel() > 0:
        F_to_compute = F.index_select(0, bones_to_recompute)  # (m, 3, 3)

        # Polar decomposition via eigenvalue decomposition
        X = F_to_compute
        G = X.transpose(-2, -1) @ X
        G = 0.5 * (G + G.transpose(-2, -1))
        eigenvalues, eigenvectors = torch.linalg.eigh(G)
        
        # Sort eigenvalues/vectors in descending order
        sort_idx = torch.argsort(eigenvalues, dim=-1, descending=True)
        eigenvalues = eigenvalues.gather(-1, sort_idx)
        eigenvectors = eigenvectors.gather(-1, sort_idx.unsqueeze(-2).expand_as(eigenvectors))
        
        # Fix sign ambiguity
        max_component_idx = eigenvectors.abs().argmax(dim=-2, keepdim=True)
        sign = torch.sign(eigenvectors.gather(-2, max_component_idx))
        eigenvectors = eigenvectors * sign
        
        # Compute rotation
        singular_values = eigenvalues.clamp_min(1e-12).sqrt()
        singular_values_inv = torch.diag_embed(1.0 / singular_values)
        U = (X @ eigenvectors) @ singular_values_inv
        Vh = eigenvectors.transpose(-2, -1)
        
        # Ensure proper rotation (determinant = +1)
        needs_reflection_fix = (torch.linalg.det(U) * torch.linalg.det(Vh)) < 0
        if needs_reflection_fix.any():
            U[needs_reflection_fix, :, 2] *= -1
            
        R_computed = (U @ Vh).to(F.dtype)  

        # sign stabilize vs previous cached quat
        Q_new_f32 = rotmat_to_quat_fast(R_computed).float()  # (m,4)
        inst = bones_to_recompute // mass_nodes_per_instance
        bone = bones_to_recompute % mass_nodes_per_instance
        idx = (bone * number_of_instance + inst).to(torch.int64)

        Q_flat16 = cache["Q_cache_bm"].reshape(mass_nodes_per_instance * number_of_instance, 4)
        Q_old_f32 = Q_flat16.index_select(0, idx).float()
        flip = (Q_new_f32 * Q_old_f32).sum(dim=-1, keepdim=True) < 0        
        Q_new_f32 = torch.where(flip, -Q_new_f32, Q_new_f32)
        # write back
        Q_flat16.index_copy_(0, idx, Q_new_f32.to(torch.float16))
        
        # Update caches 
        R_cache.index_copy_(0, bones_to_recompute, R_computed.to(torch.float16))
        F_prev.index_copy_(0, bones_to_recompute, F_to_compute)
        rotation_computed.index_fill_(0, bones_to_recompute, True)
        
    # --- Gather bone data for each Gaussian ---
    R_per_instance = R_cache.view(number_of_instance, mass_nodes_per_instance, 3, 3)
    R_per_gaussian = R_per_instance[:, weights_indices]              

    xyz_rot_sum = torch.einsum("inkab,nkb->nia", R_per_gaussian, cache["xyz_local_w"])  # (Ng,I,3)

    # We are keeping this in fp32 for accuracy
    cache["motions_bm_fp32"].copy_(motions.transpose(0, 1))  # (Nb,I,3)
    
    M_mat = cache["motions_bm_fp32"].reshape(mass_nodes_per_instance, number_of_instance * 3)         # view
    M_out = torch.sparse.mm(cache["W_csr_f32"], M_mat)                      # (Ng, I*3)
    motion_sum = M_out.view(gaussians_per_instance, number_of_instance, 3)                      # view
    
    xyz_def_nia = xyz_rot_sum + motion_sum + cache["bones_rest_blend"][:, None, :]     # (Ng,I,3)
    xyz_deformed = xyz_def_nia.permute(1, 0, 2).reshape(number_of_instance * gaussians_per_instance, 3)

    Q_mat = cache["Q_cache_bm"].reshape(mass_nodes_per_instance, number_of_instance * 4)   # view, no copy
    Q_out = torch.sparse.mm(cache["W_csr_f16"], Q_mat)                                     # (Ng, I*4)
    q = Q_out.view(gaussians_per_instance, number_of_instance, 4) 
    rest = gaussians_quat_rest.view(gaussians_per_instance, 1, 4)
    quat_def_nic = quat_mul_norm_fused(q, rest)  # (Ng
    quat_deformed = quat_def_nic.permute(1, 0, 2).reshape(number_of_instance * gaussians_per_instance, 4)

    #TODO maybe in single instance batch, we just pass fp16 format then upcast to float32 for rendering?
    quat_deformed = quat_deformed.to(torch.float32)
    #print(xyz_deformed.dtype, quat_deformed.dtype)
    return xyz_deformed, quat_deformed
