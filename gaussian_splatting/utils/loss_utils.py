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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def normal_loss(network_output, gt, alpha_mask=None):
    '''
    Use normal to regularize the normal prediction
    Adapted from Instant-NGP-PP (https://github.com/zhihao-lin/instant-ngp-pp)
    '''
    assert network_output.shape[-1] == 3                                 # expected shape: (H, W, 3)
    normal_pred = F.normalize(network_output, p=2, dim=-1)               # (H, W, 3)
    normal_gt = F.normalize(gt, p=2, dim=-1)                             # (H, W, 3)
    
    if alpha_mask is not None:
        # mask = alpha_mask.squeeze().unsqueeze(-1)
        mask = (alpha_mask.squeeze() > 0.5).float().unsqueeze(-1)
        # normal_pred = normal_pred * mask
        normal_gt = normal_gt * mask
    
    l1_loss = torch.abs(normal_pred - normal_gt).mean()                  # L1 loss (H, W, 3)
    cos_loss = -torch.sum(normal_pred * normal_gt, axis=-1).mean()       # Cosine similarity loss (H, W, 3)
    return l1_loss + 0.1 * cos_loss


def depth_loss(network_output, gt, alpha_mask=None):
    '''
    Use disparity to regularize the depth prediction
    '''
    # valid_mask = (gt > 0).float()
    assert (gt < 0.0).sum() == 0, "Depth map should be non-negative"

    if alpha_mask is not None:
        # mask = alpha_mask.squeeze()
        mask = (alpha_mask.squeeze() > 0.5).float()
        # network_output = network_output * mask
        gt = gt * mask
        
    # network_output = network_output * valid_mask
    # gt = gt * valid_mask

    # disp_pred = 1.0 / (network_output + 1e-6)
    # disp_gt = 1.0 / (gt + 1e-6)
    # l1_loss = torch.abs(disp_pred - disp_gt).mean()

    l1_loss = torch.abs(network_output - gt).mean()
    
    return l1_loss


def anisotropic_loss(gaussians_scale, r=3):
    '''
    Use to regularize gaussians size to be isotropic (avoid over-stretching gaussians)
    Reference from PhysGaussian (https://arxiv.org/pdf/2311.12198)
    '''
    # L_aniso = mean( max( max(scale)/min(scale), r ) - r)
    eps = 1e-6
    max_scale = torch.max(gaussians_scale, dim=-1).values
    min_scale = torch.min(gaussians_scale, dim=-1).values
    return torch.mean(torch.clamp(max_scale / (min_scale + eps), min=r) - r)