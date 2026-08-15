import torch
import kornia
from torch.profiler import profile, ProfilerActivity, record_function
import torch.nn.functional as Func

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

def interpolate_motions_speedup_pyh(bones, motions, relations, xyz, rot=None, quat=None,
                                weights=None, weights_indices=None, device='cuda', step='n/a'):
    n_bones, _ = bones.shape
    n_particles = xyz.shape[0]

    with record_function("LBS/setup"):
        bone_transforms = torch.zeros((n_bones, 4, 4), device=device)
        n_adj = relations.shape[1]

    with record_function("LBS/build-adjacencies"):
        adj_bones = bones[relations] - bones[:, None]  # (n_bones, n_adj, 3)
        adj_bones_new = adj_bones + motions[relations] - motions[:, None]          # (B, n_adj, 3)

    with record_function("LBS/compute-F"):
        A = adj_bones_new.transpose(1, 2).contiguous()  # (B, 3, n_adj)
        B = adj_bones.contiguous()                      # (B, n_adj, 3)
        F = torch.bmm(A, B)                             # (B, 3, 3), contiguous

    with record_function("LBS/Polar-eig FP64"):
        # F: (B,3,3), float32 cuda, contiguous
        F32 = F.contiguous()
        F64 = F32
        
        #F64 = F32.to(torch.float64)  # do the sensitive eig math in fp64

        # 1) SPD gram matrix, explicitly symmetrized to kill round-off
        G = F64.transpose(-2, -1) @ F64           # (B,3,3)
        G = 0.5 * (G + G.transpose(-2, -1))

        # 2) Eigendecomposition: right singular vectors (columns of V) & singular values^2
        w, V = torch.linalg.eigh(G)               # eigenvalues ascending

        # 3) Sort to DESCENDING to mirror SVD ordering
        idx = torch.argsort(w, dim=-1, descending=True)
        w  = w.gather(-1, idx)
        V  = V.gather(-1, idx.unsqueeze(-2).expand_as(V))

        # 4) Deterministic sign convention for eigenvectors:
        #    make the largest-magnitude entry in each column positive
        vmax_idx = V.abs().argmax(dim=-2, keepdim=True)    # (B,1,3) col-wise argmax over rows
        sgn = torch.sign(V.gather(-2, vmax_idx))           # (B,1,3) in {+1,-1}
        V = V * sgn

        # 5) Singular values (like S from SVD) and their inverse (guard tiny)
        eps64 = 1e-12
        Svals = w.clamp_min(eps64).sqrt()                  # (B,3)
        Sinv  = torch.diag_embed(1.0 / Svals)              # (B,3,3)

        # 6) Reconstruct U exactly as SVD would: U = F V S^{-1}
        U = (F64 @ V) @ Sinv                               # (B,3,3)
        Vh = V.transpose(-2, -1)                           # (B,3,3)

        # 7) Reflection fix — keep the SAME convention as your SVD path (flip last col of U)
        need_flip = (torch.linalg.det(U) * torch.linalg.det(Vh)) < 0
        if need_flip.any():
            sign = (1.0 - 2.0 * need_flip.to(U.dtype)).unsqueeze(-1)  # (B,1) in {+1,-1}
            U[..., :, 2] *= sign

        # 8) Final rotation (back to fp32)
        R = (U @ Vh).to(F32.dtype)

        # 9) Provide S for the rank test/fallback exactly like SVD
        S = Svals.to(F32.dtype)  # (B,3)  

    with record_function("LBS/build-transforms"):
        try :
            bone_transforms[:, :3, :3] = R
        except:
            print(f'[step {step}] Bad R')
            bone_transforms[:, 0, 0] = 1
            bone_transforms[:, 1, 1] = 1
            bone_transforms[:, 2, 2] = 1
        bone_transforms[:, :3,  3] = motions

    with record_function("LBS/gather-kNN"):
        selected_bones      = bones[weights_indices]           # (n_particles, k, 3)
        R_sel = R[weights_indices]                 # (N, k, 3, 3)
        t_sel = motions[weights_indices]           # (N, k, 3)
        b_sel = bones[weights_indices]             # (N, k, 3)
        #selected_transforms = bone_transforms[weights_indices]   # (N, k, 4, 4) 

    with record_function("LBS/local-coords"):
        #xyz_local = xyz.unsqueeze(1) - selected_bones 
        xyz_local = xyz.unsqueeze(1) - b_sel          # (n_particles, k, 3)

    with record_function("LBS/rotate-einsum"):
        # rotated_local = torch.einsum('nkij,nkj->nki',
        #                              selected_transforms[:, :, :3, :3], xyz_local)
        rot = R_sel.reshape(-1,3,3) @ xyz_local.reshape(-1,3,1)
        rotated_local = rot.reshape_as(xyz_local).squeeze(-1)

    with record_function("LBS/translate+weight"):
        # transformed_pts = rotated_local + selected_transforms[:, :, :3, 3] + selected_bones
        # xyz_transformed = torch.sum(transformed_pts * weights[:, :, None], dim=1)
        transformed   = rotated_local + t_sel + b_sel
        xyz_transformed = (transformed * weights[..., None]).sum(dim=1)

    ret_rot = rot  # keep contract
    if quat is not None:
        with record_function("LBS/quats-from-bones"):
            bone_quats = rotmat_to_quat_fast(R)            # (n_bones, 4)
            base_quats = bone_quats[weights_indices]       # (N, k, 4)
            quats = (base_quats * weights[..., None]).sum(dim=1)
            quats = Func.normalize(quats, dim=-1, eps=1e-12)
            ret_rot = Func.normalize(_quat_mul(quats, quat), dim=-1, eps=1e-12)

    weights_sparse = (weights, weights_indices)
    return xyz_transformed, ret_rot, weights_sparse



@torch.no_grad()
def build_lbs_static_cache(weights_indices, relations, bones):
    device = bones.device
    active = torch.unique(weights_indices).to(device)             # (M,)

    inv_lut = torch.full((bones.shape[0],), -1, device=device, dtype=torch.long)
    inv_lut[active] = torch.arange(active.numel(), device=device, dtype=torch.long)
    wi_local = inv_lut[weights_indices]                           # (N,k)

    rel_act = relations[active]                                   # (M,n_adj)

    # rest adjacencies (bones assumed static/rest)
    adj_bones_rest = bones[relations] - bones[:, None, :]         # (B,n_adj,3)
    adj_bones_rest_act = adj_bones_rest[active]                   # (M,n_adj,3)


    return {
        "active": active,
        "wi_local": wi_local,
        "rel_act": rel_act,
        "adj_bones_rest_act": adj_bones_rest_act,
        "bones_act": bones[active],                               # (M,3)
        "used_local": torch.zeros(active.numel(), dtype=torch.bool, device=device),
    }


#try to drop weights with fixed threshold, then renormalize
@torch.no_grad()
def build_lbs_static_cache_drop_weight(weights_indices, weights, relations, bones, thr=0.05, T=3):
    device = bones.device
    N, K = weights.shape

    # normalize
    w = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # threshold (optional): push tiny weights further down before topk
    w = torch.where(w >= thr, w, torch.zeros_like(w))

    # guarantee at least one kept
    none = (w.sum(dim=1) == 0)
    if none.any():
        jmax = weights.argmax(dim=1)
        w[torch.arange(N, device=device), jmax] = 1.0

    # Top-T selection and renorm on just those T
    w_top, col_top = w.topk(T, dim=1, largest=True, sorted=False)     # (N,T)
    w_top = w_top / w_top.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # active bones only among those kept
    kept_global = weights_indices.gather(1, col_top)                  # (N,T)
    active = torch.unique(kept_global)                                # (M_used,)

    inv_lut = torch.full((bones.shape[0],), -1, device=device, dtype=torch.long)
    inv_lut[active] = torch.arange(active.numel(), device=device, dtype=torch.long)
    wi_local_T = inv_lut[kept_global]                                 # (N,T)

    rel_act = relations.index_select(0, active)                       # (M_used,n_adj)
    B_rest  = (bones[rel_act] - bones[active][:,None,:]).contiguous() # (M_used,n_adj,3)

    cache = {
        "active": active,
        "wi_local_T": wi_local_T,             # <= compact neighbor map
        "rel_act": rel_act,
        "adj_bones_rest_act": B_rest,
        "bones_act": bones.index_select(0, active).contiguous(),
        "used_local": torch.zeros(active.numel(), dtype=torch.bool, device=device),
    }
    return cache, w_top


#this is the R-Reuse with dropping weight
#this is wrong masked version, it always use cached value (identity matrix)
#because at first frame ~(used local) is all True s
def interpolate_motions_speedup_reuse_static_cache_dropping_weight(
    bones, motions, relations, xyz, rot=None, quat=None,
    weights=None, weights_indices=None, device='cuda', step='n/a',
    R_cache=None, F_prev=None, valid=None,
    tau_F=5e-3,
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
        reuse_mask = (valid.index_select(0, active) & (dF < tau_F)) | (~used_local)

        # Optional det guard only on candidates
        cand = (reuse_mask & used_local).nonzero(as_tuple=False).squeeze(1)
        if cand.numel():
            ok = (torch.linalg.det(F_act.index_select(0, cand)) > 1e-7)
            reuse_mask[cand] = ok

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
            used_local.index_fill_(0, rec_idx_global, True)  

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

def interpolate_motions_speedup_reuse_static_cache_dropping_weight(
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

# #this is the R-Reuse
def interpolate_motions_speedup_reuse_static_cache(
    bones, motions, relations, xyz, rot=None, quat=None,
    weights=None, weights_indices=None, device='cuda', step='n/a',
    R_cache=None, F_prev=None, valid=None,
    tau_F=2e-3,
    static_cache=None,
):
    assert static_cache is not None, "Pass the precomputed static cache"

    device = bones.device
    active   = static_cache["active"]              # (M,)
    wi_local = static_cache["wi_local"]            # (N,k)
    rel_act  = static_cache["rel_act"]             # (M,n_adj)
    used_local = static_cache["used_local"]      # (M,) bool

    # # ----- Recompute geometry this frame (bones move!) -----
    # bones_act  = bones.index_select(0, active)                                     # (M,3)
    # nb_bones   = bones.index_select(0, rel_act.reshape(-1)).reshape(rel_act.shape + (3,))  # (M,n_adj,3)
    # adj_bones  = nb_bones - bones_act[:, None, :]                                  # (M,n_adj,3)

    # motions_act = motions.index_select(0, active)                                   # (M,3)
    # nb_motions  = motions.index_select(0, rel_act.reshape(-1)).reshape(rel_act.shape + (3,))
    # adj_bones_new = adj_bones + nb_motions - motions_act[:, None, :]               # (M,n_adj,3)

    #     # --- Compute F for active bones ---
    # with record_function("LBS/compute-F(active)"):
    #     A = adj_bones_new.transpose(1, 2).contiguous()  # (M,3,n_adj)
    #     B = adj_bones.contiguous()                      # (M,n_adj,3)
    #     F_act = torch.bmm(A, B)                         # (M,3,3)


    bones_act = static_cache["bones_act"]                 # rest
    B = static_cache["adj_bones_rest_act"].contiguous()
    motions_act = motions.index_select(0, active)
    nb_motions  = motions.index_select(0, rel_act.reshape(-1)).reshape(rel_act.shape + (3,))
    adj_bones_new = B + nb_motions - motions_act[:, None, :]  # (M,n_adj,3)

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
        reuse_mask = (valid.index_select(0, active) & (dF < tau_F)) | (~used_local)

        # Optional det guard only on candidates
        cand = (reuse_mask & used_local).nonzero(as_tuple=False).squeeze(1)
        if cand.numel():
            ok = (torch.linalg.det(F_act.index_select(0, cand)) > 1e-7)
            reuse_mask[cand] = ok

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
            used_local.index_fill_(0, rec_idx_global, True)  

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

#this is on the fly 
def interpolate_motions_speedup_reuse(
    bones, motions, relations, xyz, rot=None, quat=None,
    weights=None, weights_indices=None, device='cuda', step='n/a',
    R_cache=None, F_prev=None, valid=None, 
    # tuning knobs
    tau_F=5e-3,            # F-delta threshold for reuse
    static=None,
    # tau_rot_small=1e-2,  # optional: if you add a "very-small-change" fast projection tier
):
    n_bones, _ = bones.shape

    with record_function("LBS/active-LUT"): 
        # active bones in this batch 
        active = torch.unique(weights_indices) # (M,) 
        M = active.numel() 
        if M == 0: 
            # Degenerate case: nothing to skin; return inputs 
            # (Keep contract: rot path returns 'rot' below) 
            return xyz, rot if rot is not None else None, (weights, weights_indices) 
            
        # inverse map: global bone id -> local active index 
        inv_lut = torch.full((n_bones,), -1, device=device, dtype=torch.long) 
        inv_lut[active] = torch.arange(M, device=device, dtype=torch.long) 
        # map weights_indices to active-local indices once 
        wi_local = inv_lut[weights_indices] # (N, k)

# ------- Build adjacencies ONLY for active bones ------- 
    with record_function("LBS/build-adjacencies(active)"):
        rel_act = relations[active] # (M, n_adj) 
        bones_act = bones[active] # (M, 3) 
        motions_act = motions[active] # (M, 3)

        # neighbors for active bones 
        nb_bones = bones[rel_act] # (M, n_adj, 3) 
        nb_motions = motions[rel_act] # (M, n_adj, 3)
        adj_bones = nb_bones - bones_act[:, None, :]
        adj_bones_new  = adj_bones + nb_motions - motions_act[:, None, :]  # (M,n_adj,3)

    # ------- Compute F ONLY for active bones -------
    with record_function("LBS/compute-F(active)"):
        A = adj_bones_new.transpose(1, 2).contiguous()        # (M, 3, n_adj)
        B = adj_bones.contiguous()                            # (M, n_adj, 3)
        F_act = torch.bmm(A, B)                               # (M, 3, 3) float32

    # ---------- F-delta reuse + eig recompute for changed ----------
    with record_function("LBS/Polar-reuse-Fdelta+Eig(active)"):
        # cheap early gate: ||F - F_prev||_F
        dF = torch.linalg.matrix_norm(F_act - F_prev.index_select(0, active), ord='fro', dim=(-2, -1))

        reuse_mask = valid.index_select(0, active) & (dF < tau_F)

        cand = reuse_mask.nonzero(as_tuple=False).squeeze(1)
        if cand.numel():
            ok = (torch.linalg.det(F_act.index_select(0, cand)) > 1e-7)
            good = torch.zeros_like(reuse_mask, dtype=torch.bool)
            good[cand] = ok
            reuse_mask = good

        #pyh void reallocation and double gather
        R_act = R_cache.index_select(0, active).clone()   # (M,3,3)
        rec_idx_local = (~reuse_mask).nonzero(as_tuple=False).squeeze(1)  
        if rec_idx_local.numel():
            # Grab their F
            F_rec = F_act.index_select(0, rec_idx_local)     # (m,3,3)

            ###
            X = F_rec                                       

            G = X.transpose(-2, -1) @ X
            G = 0.5 * (G + G.transpose(-2, -1))
            w, V = torch.linalg.eigh(G)                       # ascending
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

            R_rec = (U @ Vh).to(F_act.dtype)

            rec_idx_global = active.index_select(0, rec_idx_local)
            R_act.index_copy_(0, rec_idx_local, R_rec)
            R_cache.index_copy_(0, rec_idx_global, R_rec)
            F_prev.index_copy_(0, rec_idx_global, F_rec)
            valid.index_fill_(0, rec_idx_global, True)

    # ------- Gather only what you need (avoid full R) -------
    with record_function("LBS/gather-kNN"):
        # Map to local active indices
        R_sel = R_act[wi_local]                               # (N, k, 3, 3)
        t_sel = motions_act[wi_local]                         # (N, k, 3)
        b_sel = bones_act[wi_local]                           # (N, k, 3)

    # ------- Transform points -------
    with record_function("LBS/local-coords"):
        xyz_local = xyz.unsqueeze(1) - b_sel                  # (N, k, 3)

    with record_function("LBS/rotate-matmul"):
        rot = R_sel.reshape(-1, 3, 3) @ xyz_local.reshape(-1, 3, 1)
        rotated_local = rot.reshape_as(xyz_local).squeeze(-1)

    with record_function("LBS/translate+weight"):
        transformed = rotated_local + t_sel + b_sel
        xyz_transformed = (transformed * weights[..., None]).sum(dim=1)

    # ------- Optional quaternion output, ONLY for active -------
    ret_rot = rot  # keep contract
    if quat is not None:
        with record_function("LBS/quats-from-active-bones"):
            bone_quats_act = rotmat_to_quat_fast(R_act)          # (M, 4)
            base_quats = bone_quats_act[wi_local]                # (N, k, 4)
            quats = (base_quats * weights[..., None]).sum(dim=1)
            quats = Func.normalize(quats, dim=-1, eps=1e-12)
            ret_rot = Func.normalize(_quat_mul(quats, quat), dim=-1, eps=1e-12)

    return xyz_transformed, ret_rot, (weights, weights_indices)


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


# def cluster_transform_and_apply(current_pos, clusters, clusters_meta, static_bones, bones_act, motions_act, R_act):
#     labels  = clusters["labels"]     # (N,)
#     wi_c    = clusters_meta["wi_c"]  # (C,Kc) global bone ids
#     w_c     = clusters_meta["w_c"]   # (C,Kc)

#     active  = static_bones["active"]
#     inv_lut = static_bones["inv_lut"]
#     wi_local_c = inv_lut[wi_c]       # (C,Kc)

#     # gather per-cluster bones
#     R_c = R_act[wi_local_c]          # (C,Kc,3,3)
#     b_c = bones_act[wi_local_c]      # (C,Kc,3)
#     t_c = motions_act[wi_local_c]    # (C,Kc,3)

#     # one affine per cluster: A_c, t_c
#     A_c = (w_c[..., None, None] * R_c).sum(dim=1)                                          # (C,3,3)
#     t_c = (w_c[..., None] * (t_c + b_c - torch.einsum('ckij,ckj->cki', R_c, b_c))).sum(dim=1)  # (C,3)

#     # apply per point via its cluster label (no Python loops)
#     A_by_pt = A_c.index_select(0, labels)                    # (N,3,3)
#     t_by_pt = t_c.index_select(0, labels)                    # (N,3)
#     Xout    = torch.bmm(A_by_pt, current_pos.unsqueeze(-1)).squeeze(-1) + t_by_pt
#     return Xout

# def cluster_rigid_and_apply(current_pos, current_rot, clusters_meta, static_bones,
#                             bones_act, motions_act, R_act):
#     labels  = clusters_meta["labels"]        # (N,)
#     wi_c    = clusters_meta["wi_c"]          # (C,Kc) global ids
#     w_c     = clusters_meta["w_c"]           # (C,Kc)
#     x_c     = clusters_meta["centroid"]      # (C,3)
#     C       = clusters_meta["C"]

#     active  = static_bones["active"]
#     inv_lut = static_bones["inv_lut"]
#     wi_local_c = inv_lut[wi_c]               # (C,Kc)

#     # gather per-cluster bone transforms
#     Rg = R_act[wi_local_c]                   # (C,Kc,3,3)
#     bg = bones_act[wi_local_c]               # (C,Kc,3)
#     tg = motions_act[wi_local_c]             # (C,Kc,3)

#     # affine from weighted LBS
#     A_c = (w_c[...,None,None] * Rg).sum(dim=1)                                        # (C,3,3)
#     t_aff = (w_c[...,None] * (tg + bg - torch.einsum('ckij,ckj->cki', Rg, bg))).sum(dim=1)  # (C,3)

#     # rigid: project A_c to rotation, then align translation at cluster centroid
#     U,S,Vh = torch.linalg.svd(A_c, full_matrices=False)
#     det = torch.linalg.det(U) * torch.linalg.det(Vh)
#     Vh[det < 0, 2, :] *= -1
#     R_c = U @ Vh                                                                     # (C,3,3)

#     x_c_prime = torch.einsum('cij,cj->ci', A_c, x_c) + t_aff                          # via affine
#     t_c = x_c_prime - torch.einsum('cij,cj->ci', R_c, x_c)                            # align center

#     # per-cluster quaternion
#     q_c = rotmat_to_quat_fast(R_c)                                                         # (C,4)

#     # apply to every Gaussian (vectorized by label)
#     R_by_pt = R_c.index_select(0, labels)                                             # (N,3,3)
#     t_by_pt = t_c.index_select(0, labels)                                             # (N,3)
#     pos_out = torch.einsum('nij,nj->ni', R_by_pt, current_pos) + t_by_pt              # (N,3)

#     if current_rot is not None:
#         q_by_pt  = q_c.index_select(0, labels)                                        # (N,4)
#         rot_out  = torch.nn.functional.normalize(_quat_mul(q_by_pt, current_rot), dim=-1)
#     else:
#         rot_out = None

#     return pos_out, rot_out


def interpolate_motions_speedup_reuse_static_cache_cluster(
    bones, motions, relations, xyz, rot=None, quat=None,
    weights=None, weights_indices=None, device='cuda', step='n/a',
    R_cache=None, F_prev=None, valid=None,
    tau_F=1e-2,
    static_cache=None,
    clusters_meta=None,          # <<< NEW: pass this to enable cluster mode
):
    assert static_cache is not None

    device = bones.device
    active   = static_cache["active"]
    rel_act  = static_cache["rel_act"]
    inv_lut  = static_cache["inv_lut"]          # <<< ensure you store this in your cache
    used_local = static_cache.get("used_local", None)


    bones_act = static_cache["bones_act"]                 # rest
    B = static_cache["B_rest"].contiguous()
    motions_act = motions.index_select(0, active)
    nb_motions  = motions.index_select(0, rel_act.reshape(-1)).reshape(rel_act.shape + (3,))
    adj_bones_new = B + nb_motions - motions_act[:, None, :]  # (M,n_adj,3)

    # --- F, reuse, recompute R_act (unchanged) ---
    A = adj_bones_new.transpose(1, 2).contiguous()
    #B = adj_bones.contiguous()
    F_act = torch.bmm(A, B)

    dF = torch.linalg.matrix_norm(F_act - F_prev.index_select(0, active), ord='fro', dim=(-2, -1))
    reuse_mask = valid.index_select(0, active) & (dF < tau_F)
    if used_local is not None:
        reuse_mask = reuse_mask | (~used_local)

    # Optional det guard only on candidates
    cand = (reuse_mask & used_local).nonzero(as_tuple=False).squeeze(1)
    if cand.numel():
        ok = (torch.linalg.det(F_act.index_select(0, cand)) > 1e-7)
        reuse_mask[cand] = ok

    R_act = R_cache.index_select(0, active).clone()
    rec_idx_local = (~reuse_mask).nonzero(as_tuple=False).squeeze(1)
    if rec_idx_local.numel():
        F_rec = F_act.index_select(0, rec_idx_local)

        # === eig-based polar, same as your original ===
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
        R_rec = (U @ Vh).to(F_act.dtype)
        # ---------------------------------------------

        rec_idx_global = active.index_select(0, rec_idx_local)
        R_act.index_copy_(0, rec_idx_local, R_rec)
        R_cache.index_copy_(0, rec_idx_global, R_rec)
        F_prev.index_copy_(0, rec_idx_global, F_rec)
        valid.index_fill_(0, rec_idx_global, True)
        used_local.index_fill_(0, rec_idx_global, True)  

    # =============== CLUSTER MODE (eig-based, no SVD) ===============
    wi_c   = clusters_meta["wi_c"]
    w_c    = clusters_meta["w_c"]
    labels = clusters_meta["labels"]
    x_c    = clusters_meta["centroid"]
    C, Kc  = wi_c.shape

    # local map: global bone id -> local row in ACTIVE
    inv_lut = static_cache.get("inv_lut", None)
    if inv_lut is None:
        inv_lut = torch.full((bones.shape[0],), -1, device=device, dtype=torch.long)
        inv_lut[active] = torch.arange(active.numel(), device=device, dtype=torch.long)

    wi_local_c = inv_lut[wi_c]                      # (C,Kc)

    # Gather per-cluster bone transforms
    Rg = R_act[wi_local_c]                          # (C,Kc,3,3)
    bg = bones_act[wi_local_c]                      # (C,Kc,3)
    tg = motions_act[wi_local_c]                    # (C,Kc,3)

    # 1) Affine from weighted LBS
    A_c   = (w_c[..., None, None] * Rg).sum(dim=1)  # (C,3,3)
    t_aff = (w_c[..., None] *
            (tg + bg - torch.einsum('ckij,ckj->cki', Rg, bg))).sum(dim=1)  # (C,3)

    # 2) Project A_c to closest rotation via eig-based polar (same scheme as above)
    Xc = A_c
    Gc = Xc.transpose(-2, -1) @ Xc
    Gc = 0.5 * (Gc + Gc.transpose(-2, -1))
    wc, Vc = torch.linalg.eigh(Gc)
    idxc   = torch.argsort(wc, dim=-1, descending=True)
    wc     = wc.gather(-1, idxc)
    Vc     = Vc.gather(-1, idxc.unsqueeze(-2).expand_as(Vc))
    vmaxc  = Vc.abs().argmax(dim=-2, keepdim=True)
    sgnc   = torch.sign(Vc.gather(-2, vmaxc))
    Vc     = Vc * sgnc
    Svalsc = wc.clamp_min(1e-12).sqrt()
    Sinvc  = torch.diag_embed(1.0 / Svalsc)
    Uc     = (Xc @ Vc) @ Sinvc
    Vhc    = Vc.transpose(-2, -1)
    need_flip_c = (torch.linalg.det(Uc) * torch.linalg.det(Vhc)) < 0
    if need_flip_c.any():
        Uc[need_flip_c, :, 2] *= -1
    R_c = (Uc @ Vhc).to(A_c.dtype)                  # (C,3,3)

    # 3) Fix translation so the cluster centroid maps exactly
    x_c_prime = torch.einsum('cij,cj->ci', A_c, x_c) + t_aff
    t_c = x_c_prime - torch.einsum('cij,cj->ci', R_c, x_c)      # (C,3)

    # 4) Broadcast to all gaussians by cluster id
    R_by_pt = R_c.index_select(0, labels)                       # (N,3,3)
    t_by_pt = t_c.index_select(0, labels)                       # (N,3)
    xyz_out = torch.einsum('nij,nj->ni', R_by_pt, xyz) + t_by_pt

    # 5) Quaternion output: reuse your converter (no SVD)
    ret_rot = rot
    if quat is not None:
        q_c = rotmat_to_quat_fast(R_c)                          # (C,4)
        q_by_pt = q_c.index_select(0, labels)                   # (N,4)
        # compose: q_cluster ⊗ quat
        w1,x1,y1,z1 = q_by_pt.unbind(-1)
        w2,x2,y2,z2 = quat.unbind(-1)
        ret_rot = torch.stack((
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2), dim=-1)
        ret_rot = Func.normalize(ret_rot, dim=-1, eps=1e-12)

    return xyz_out, ret_rot, (weights, weights_indices)

@torch.no_grad()
def build_lbs_static_cache_drop_weight_upto_T(weights_indices, weights, relations, bones, thr=0.05, T=3):
    device = bones.device
    N, K = weights.shape

    assert T >= 1 and T <= K, "T must be in [1, K]"

    # normalize
    w = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    # threshold (optional): push tiny weights further down before topk
    w = torch.where(w >= thr, w, torch.zeros_like(w))

    # guarantee at least one kept
    none = (w.sum(dim=1) == 0)
    if none.any():
        jmax = weights.argmax(dim=1)
        w[torch.arange(N, device=device), jmax] = 1.0

    # Top-T selection and renorm on just those T
    w_top, col_top = w.topk(T, dim=1, largest=True, sorted=False)     # (N,T)

    keep_mask = (w_top > 0)  # (N,T) bool
    # set non-kept weights to zero explicitly (already zero, but being explicit is clearer)
    w_top = torch.where(keep_mask, w_top, torch.zeros_like(w_top))

    denom = w_top.sum(dim=1, keepdim=True).clamp_min(1e-8)
    w_top = w_top / denom  # zeros stay zeros

    col_top_safe = col_top.clone()
    kept_global = weights_indices.gather(1, col_top_safe)  # (N,T)
    kept_global = torch.where(keep_mask, kept_global, torch.full_like(kept_global, -1))

    active = torch.unique(kept_global[kept_global >= 0])
    if active.numel() == 0:
        # Extremely unlikely due to step 3; pick a fallback to avoid empty cache.
        active = weights_indices[0, 0:1]

    inv_lut = torch.full((bones.shape[0],), -1, device=device, dtype=torch.long)
    inv_lut[active] = torch.arange(active.numel(), device=device, dtype=torch.long)

    # map kept global bone ids to *local* ids (pad with -1 where masked)
    wi_local_T = torch.full_like(kept_global, -1)
    pos_mask = (kept_global >= 0)
    wi_local_T[pos_mask] = inv_lut[kept_global[pos_mask]]  # (N,T) in [0..M_used-1] or -1

    # 10) compact adjacency and rest offsets *only for active bones*
    rel_act = relations.index_select(0, active)                     # (M_used, n_adj)
    B_rest  = (bones[rel_act] - bones[active][:, None, :]).contiguous()  # (M_used, n_adj, 3)

    cache = {
        "active": active,
        "wi_local_T": wi_local_T,             # <= compact neighbor map
        "rel_act": rel_act,
        "adj_bones_rest_act": B_rest,
        "bones_act": bones.index_select(0, active).contiguous(),
        "used_local": torch.zeros(active.numel(), dtype=torch.bool, device=device),
        "keep_mask": keep_mask,               # (N,T) bool
    }
    return cache, w_top



# @torch.no_grad()
# def build_lbs_static_cache_drop_weight(weights_indices, weights, relations, bones, thr=0.05):
#     device = bones.device
#     N, K = weights.shape

#     # 1) prune small weights, keep at least one per row, renormalize
#     w = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
#     keep = w >= thr
#     none = ~keep.any(dim=1)
#     if none.any():
#         jmax = w.argmax(dim=1)
#         keep[torch.arange(N, device=device), jmax] = True
#     w_eff = (w * keep)
#     w_eff = w_eff / w_eff.sum(dim=1, keepdim=True).clamp_min(1e-8)

#     # 2) build active set **from kept entries only**
#     kept_global = weights_indices[keep]                  # 1D view of kept bone ids
#     active = torch.unique(kept_global)                   # (M_used,)

#     # 3) inv LUT for used bones only
#     inv_lut = torch.full((bones.shape[0],), -1, device=device, dtype=torch.long)
#     inv_lut[active] = torch.arange(active.numel(), device=device, dtype=torch.long)

#     # 4) wi_local with a safe fallback for dropped columns
#     wi_local_tmp = inv_lut[weights_indices]              # (N,K), -1 for dropped bone ids
#     # per-row fallback: any kept column (use the first True in `keep`)
#     fallback_col = keep.float().argmax(dim=1)            # (N,)
#     fallback_idx = wi_local_tmp[torch.arange(N, device=device), fallback_col]  # (N,)
#     wi_local = torch.where(
#         keep,                                            # kept → mapped id
#         wi_local_tmp,
#         fallback_idx.view(N, 1).expand(N, K)            # dropped → duplicate a kept id
#     )

#     # 5) adjacency for used bones, and B at rest (constant)
#     rel_act = relations.index_select(0, active)          # (M_used, n_adj)
#     B_rest  = (bones[rel_act] - bones[active][:,None,:]) # (M_used, n_adj, 3)

#     # 6) used_local (all True now, since active came from kept)
#     used_local = torch.ones(active.numel(), dtype=torch.bool, device=device)

#     cache = {
#         "active": active,
#         "wi_local": wi_local,
#         "rel_act": rel_act,
#         "adj_bones_rest_act": B_rest,
#         "bones_act": bones[active],
#         "used_local": used_local,
#     }
#     return cache, w_eff
