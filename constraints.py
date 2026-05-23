"""
Energy/Loss functions for Test-Time Training (TTT) in TTT3R.

This module contains various loss functions used during TTT optimization,
including pose supervision, intrinsics supervision, and multiview consistency losses.
"""

import numpy as np

import torch
import torch.nn.functional as F
from matplotlib import cm
from typing import Dict, Optional, Tuple
from utils.geometry import (to_homogeneous, sample_view_pairs, invert_se3, sample_view_triplets, triangle_similarity_loss)

# ----------------------
# Pose supervision
# 1. Relative pose energy
# 2. Direct pose energy
# ----------------------
def relative_pose_energy(predicted_poses: torch.Tensor,
                        gt_extrinsics: torch.Tensor,
                        num_pairs: int = None,
                        rot_loss_type: str = 'cosine',
                        trans_loss_type: str = 'triangle_similarity',
                        enforce_canonical_reference: bool = False) -> tuple:
    """
    Compute both relative rotation geodesic loss and relative translation loss.
    Samples view pairs/triplets internally for rotation/translation losses.
    
    Args:
        predicted_poses: Predicted camera poses (B, S, 4, 4) - camera-to-world transforms
        gt_extrinsics: Ground truth extrinsics (B, S, 4, 4) - world-to-camera transforms
        num_pairs: Number of view pairs to sample
        rot_loss_type: Type of rotation loss ('cosine', 'angle')
        trans_loss_type: Type of translation loss ('triangle_similarity', 'normed_smooth_l1', 'smooth_l1')
        enforce_canonical_reference: If True, enforces first camera at origin with identity rotation (for VGGT)
        
    Returns:
        (rot_loss, trans_loss): Tuple of rotation and translation losses
    """
    assert predicted_poses.ndim == 4 and gt_extrinsics.ndim == 4
    predicted_poses = to_homogeneous(predicted_poses)
    gt_extrinsics = to_homogeneous(gt_extrinsics)
    
    # Convert extrinsics (world-to-camera) to poses (camera-to-world) for comparison
    gt_poses = invert_se3(gt_extrinsics)

    # Sample pairs for rotation loss
    num_views = predicted_poses.shape[1]
    pairs = sample_view_pairs(num_views, num_pairs, device=predicted_poses.device)
    if num_pairs is None:
        num_pairs = pairs.shape[0]
    K = pairs.shape[0]
    if K == 0:
        zero = torch.zeros((), device=predicted_poses.device, dtype=predicted_poses.dtype)
        return zero, zero

    # Gather once
    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    Tpi = predicted_poses[:, i_idx, :, :]
    Tpj = predicted_poses[:, j_idx, :, :]
    Tgi = gt_poses[:, i_idx, :, :]
    Tgj = gt_poses[:, j_idx, :, :]

    # Relative transforms
    Tpi_inv = invert_se3(Tpi)
    Tgi_inv = invert_se3(Tgi)
    Tpred_rel = Tpi_inv @ Tpj
    Tgt_rel = Tgi_inv @ Tgj

    # Rotation geodesic loss
    R_pred_rel = Tpred_rel[..., :3, :3]
    R_gt_rel = Tgt_rel[..., :3, :3]
    RtR = R_pred_rel.transpose(-1, -2) @ R_gt_rel
    trace = RtR.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    if rot_loss_type == 'angle':
        cos = (trace - 1.0) * 0.5
        cos = torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)
        rot_loss = torch.acos(cos).mean()
    elif rot_loss_type == 'cosine':
        rot_loss = ((3.0 - trace) * 0.5).mean()
    else:
        raise ValueError(f"Invalid rot_loss_type: {rot_loss_type}")

    # Translation loss
    if trans_loss_type == "triangle_similarity":
        # Use triangle similarity loss - coordinate frame invariant and scale invariant
        # Sample triplets for triangle-based comparison
        triplets = sample_view_triplets(num_views, num_triplets=None, device=predicted_poses.device)
        trans_loss = triangle_similarity_loss(predicted_poses, gt_poses, triplets)
    elif trans_loss_type == "normed_smooth_l1":
        # Normalize relative translation vectors by average L2 norm across pairs before smooth L1 loss
        t_pred_rel = Tpred_rel[..., :3, 3]  # (B, K, 3) where K is number of pairs
        t_gt_rel = Tgt_rel[..., :3, 3]      # (B, K, 3)
        
        # Compute L2 norms for each relative translation pair
        t_pred_rel_norms = torch.norm(t_pred_rel, dim=-1, keepdim=False)  # (B, K)
        t_gt_rel_norms = torch.norm(t_gt_rel, dim=-1, keepdim=False)      # (B, K)
        
        # Compute average norms across pairs for each batch
        t_pred_avg_norm = torch.mean(t_pred_rel_norms, dim=1, keepdim=True) + 1e-8  # (B, 1)
        t_gt_avg_norm = torch.mean(t_gt_rel_norms, dim=1, keepdim=True) + 1e-8      # (B, 1)
        
        # Normalize relative translation vectors by average norms
        t_pred_rel_normalized = t_pred_rel / t_pred_avg_norm.unsqueeze(-1)  # (B, K, 3)
        t_gt_rel_normalized = t_gt_rel / t_gt_avg_norm.unsqueeze(-1)        # (B, K, 3)
        
        # Apply smooth L1 loss to normalized relative vectors
        trans_loss = F.smooth_l1_loss(t_pred_rel_normalized, t_gt_rel_normalized, reduction='mean')
    elif trans_loss_type == 'smooth_l1':
        # Default: Smooth L1 over vector components, then reduce to (B, K)
        t_pred_rel = Tpred_rel[..., :3, 3]
        t_gt_rel = Tgt_rel[..., :3, 3]
        trans_loss = F.smooth_l1_loss(t_pred_rel, t_gt_rel, reduction='mean')
    else:
        raise ValueError(f"Invalid trans_loss_type: {trans_loss_type}")

    # Optionally enforce canonical reference constraint (for VGGT)
    if enforce_canonical_reference:
        # In VGGT, the first camera defines the world coordinate system
        # and should be at origin (zero translation) with identity rotation
        reference_pose = predicted_poses[:, 0]  # (B, 4, 4)
        
        # Compute rotation loss using geodesic distance (consistent with relative rotation loss)
        R_ref = reference_pose[:, :3, :3]  # (B, 3, 3)
        I_3x3 = torch.eye(3, device=R_ref.device, dtype=R_ref.dtype).unsqueeze(0).expand_as(R_ref)
        
        # Compute R^T @ I = R^T (for geodesic distance from identity)
        RtI = R_ref.transpose(-1, -2) @ I_3x3  # This simplifies to R^T
        trace = RtI.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)  # (B,)
        
        # Use same loss formulation as relative rotation
        if rot_loss_type == 'angle':
            cos = (trace - 1.0) * 0.5
            cos = torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)
            ref_rot_loss = torch.acos(cos).mean()
        elif rot_loss_type == 'cosine':
            ref_rot_loss = ((3.0 - trace) * 0.5).mean()
        else:
            raise ValueError(f"Invalid rot_loss_type: {rot_loss_type}")
        
        # Compute translation loss - enforce to be exactly zero
        ref_trans_loss = F.smooth_l1_loss(reference_pose[:, :3, 3], torch.zeros_like(reference_pose[:, :3, 3]))
        
        # Add weighted constraints to both rotation and translation losses
        # This ensures the reference camera stays at canonical pose (origin with no rotation)
        rot_loss = rot_loss + num_views * ref_rot_loss
        trans_loss = trans_loss + num_views * ref_trans_loss

    return rot_loss, trans_loss

def direct_pose_energy(predicted_poses: torch.Tensor,
                       gt_extrinsics: torch.Tensor,
                       rot_loss_type: str = 'cosine',
                       trans_loss_type: str = 'smooth_l1',
                       canonicalize_to_first_view: bool = False,
                       pose_type: str = 'abs') -> tuple:
    """
    Compute direct pose supervision loss by comparing predicted poses to ground truth extrinsics.
    
    Args:
        predicted_poses: Predicted camera poses (B, S, 4, 4) - camera-to-world transforms
        gt_extrinsics: Ground truth extrinsics (B, S, 4, 4) - world-to-camera transforms
        rot_loss_type: Type of rotation loss ('cosine', 'angle', 'smooth_l1')
        trans_loss_type: Type of translation loss ('smooth_l1', 'l2', 'l1')
            - 'smooth_l1': Standard smooth L1 loss on translation vectors
            - 'l2': Mean squared error on translation vectors
            - 'l1': L1 loss on translation vectors

        canonicalize_to_first_view: If True, canonicalize both predicted and GT to their respective first views
        pose_type: Type of pose ('abs', 'rel')
    Returns:
        (rot_loss, trans_loss): Tuple of rotation and translation losses
    """
    assert predicted_poses.ndim == 4 and gt_extrinsics.ndim == 4
    predicted_poses = to_homogeneous(predicted_poses)
    gt_extrinsics = to_homogeneous(gt_extrinsics)
    
    # Convert extrinsics (world-to-camera) to poses (camera-to-world) for comparison
    gt_poses = invert_se3(gt_extrinsics)
    
    B, S = predicted_poses.shape[:2]
    
    # Optionally canonicalize both predicted and GT to their respective first views
    if canonicalize_to_first_view:
        if S == 1:
            # For single view, set both to identity to avoid coordinate system mismatch
            # This makes the loss zero, which is correct since we can't learn relative geometry
            return torch.zeros((), device=predicted_poses.device, dtype=predicted_poses.dtype), \
                torch.zeros((), device=predicted_poses.device, dtype=predicted_poses.dtype)
        else:
            # Canonicalize GT poses to GT first view
            gt_pose_ref = gt_poses[:, 0:1]            # (B, 1, 4, 4)
            gt_pose_ref_inv = invert_se3(gt_pose_ref) # (B, 1, 4, 4)
            gt_poses = gt_pose_ref_inv @ gt_poses     # (B, S, 4, 4)
            
            if pose_type == 'rel':
                # Canonicalize predicted poses to predicted first view
                pred_pose_ref = predicted_poses[:, 0:1]       # (B, 1, 4, 4)
                pred_pose_ref_inv = invert_se3(pred_pose_ref) # (B, 1, 4, 4)
                predicted_poses = pred_pose_ref_inv @ predicted_poses  # (B, S, 4, 4)
    
    # Extract rotation matrices and translation vectors (now both are camera-to-world)
    R_pred = predicted_poses[..., :3, :3]  # (B, S, 3, 3)
    t_pred = predicted_poses[..., :3, 3]   # (B, S, 3)
    R_gt = gt_poses[..., :3, :3]           # (B, S, 3, 3)
    t_gt = gt_poses[..., :3, 3]            # (B, S, 3)
    
    # Compute rotation loss using geodesic distance
    if rot_loss_type == 'cosine':
        # Geodesic distance using cosine formulation: d = (3 - trace(R_pred^T @ R_gt)) / 2
        RtR = R_pred.transpose(-1, -2) @ R_gt  # (B, S, 3, 3)
        trace = RtR.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)  # (B, S)
        rot_loss = ((3.0 - trace) * 0.5).mean()
    elif rot_loss_type == 'angle':
        # Direct angle computation from geodesic distance
        RtR = R_pred.transpose(-1, -2) @ R_gt  # (B, S, 3, 3)
        trace = RtR.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)  # (B, S)
        cos = (trace - 1.0) * 0.5
        cos = torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7)
        rot_loss = torch.acos(cos).mean()
    elif rot_loss_type == 'smooth_l1':
        # Frobenius norm between rotation matrices
        # rot_loss = F.mse_loss(R_pred, R_gt)
        rot_loss = F.smooth_l1_loss(R_pred, R_gt, reduction='mean')
    else:
        raise ValueError(f"Invalid rot_loss_type: {rot_loss_type}")
    
    # Compute translation loss
    if trans_loss_type == 'smooth_l1':
        trans_loss = F.smooth_l1_loss(t_pred, t_gt, reduction='mean')
    elif trans_loss_type == 'normed_l1':
        # Normalize translation vectors by average L2 norm across views before smooth L1 loss
        # Compute L2 norms for each view
        t_pred_norms = torch.norm(t_pred, dim=-1, keepdim=False)  # (B, S)
        t_gt_norms = torch.norm(t_gt, dim=-1, keepdim=False)      # (B, S)
        
        # Compute average norms across views for each batch
        t_pred_avg_norm = torch.mean(t_pred_norms, dim=1, keepdim=True) + 1e-8  # (B, 1)
        t_gt_avg_norm = torch.mean(t_gt_norms, dim=1, keepdim=True) + 1e-8      # (B, 1)
        
        # Normalize translation vectors by average norms
        t_pred_normalized = t_pred / t_pred_avg_norm.unsqueeze(-1)  # (B, S, 3)
        t_gt_normalized = t_gt / t_gt_avg_norm.unsqueeze(-1)        # (B, S, 3)
        
        # Apply smooth L1 loss to normalized vectors
        # trans_loss = F.smooth_l1_loss(t_pred_normalized, t_gt_normalized, reduction='mean')
        trans_loss = F.l1_loss(t_pred_normalized, t_gt_normalized, reduction='mean')
    elif trans_loss_type == 'l2':
        trans_loss = F.mse_loss(t_pred, t_gt)
    elif trans_loss_type == 'l1':
        trans_loss = F.l1_loss(t_pred, t_gt)
    else:
        raise ValueError(f"Invalid trans_loss_type: {trans_loss_type}")
    
    return rot_loss, trans_loss

# ----------------------
# Intrinsics energy
# 1. Direct intrinsics energy
# 2. Ray consistency intrinsics energy
# ----------------------
def direct_intrinsics_energy(pred_intrinsics: torch.Tensor, gt_intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Direct supervision loss between predicted and ground truth intrinsics.
    
    Args:
        pred_intrinsics: Predicted intrinsics (B, S, 3, 3) where B=batch_size, S=num_views
        gt_intrinsics: Ground truth intrinsics (B, S, 3, 3) - same shape as predicted
        
    Note: Both inputs should have the same shape. The interface converts dataset format
    (S, 3, 3) to (B, S, 3, 3) by adding batch dimension.
    
    Returns:
        loss: Scalar loss value
    """
    # Verify shapes match
    if pred_intrinsics.shape != gt_intrinsics.shape:
        raise ValueError(
            f"Shape mismatch: pred_intrinsics {pred_intrinsics.shape} vs "
            f"gt_intrinsics {gt_intrinsics.shape}. Both should have shape (B, S, 3, 3)."
        )
    
    # Focus on the important intrinsic parameters: fx, fy, cx, cy
    # Intrinsics matrix format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    
    # Extract focal lengths (fx, fy)
    pred_fx, pred_fy = pred_intrinsics[..., 0, 0], pred_intrinsics[..., 1, 1]  # (B, S) or (B,)
    gt_fx, gt_fy = gt_intrinsics[..., 0, 0], gt_intrinsics[..., 1, 1]
    
    # Extract principal point (cx, cy)
    pred_cx, pred_cy = pred_intrinsics[..., 0, 2], pred_intrinsics[..., 1, 2]  # (B, S) or (B,)
    gt_cx, gt_cy = gt_intrinsics[..., 0, 2], gt_intrinsics[..., 1, 2]
    
    # Relative losses for focal lengths (more important)
    fx_loss = F.smooth_l1_loss(pred_fx, gt_fx, reduction='mean')
    fy_loss = F.smooth_l1_loss(pred_fy, gt_fy, reduction='mean')
    
    # Absolute losses for principal point (less critical, usually near image center)
    cx_loss = F.smooth_l1_loss(pred_cx, gt_cx, reduction='mean')
    cy_loss = F.smooth_l1_loss(pred_cy, gt_cy, reduction='mean')
    
    # Combine losses with higher weight on focal lengths
    total_loss = (2.0 * (fx_loss + fy_loss) + 1.0 * (cx_loss + cy_loss)) / 6
    
    return total_loss

def intrinsics_ray_consistency_energy(
        local_points: torch.Tensor,
        intrinsics: torch.Tensor,
        stride: int = 1,
    ) -> torch.Tensor:
    """
    Encourage predicted local point map to follow the pinhole camera model given intrinsics.

    Args:
        local_points: (B, N, H, W, 3) with [X, Y, Z] in camera coords, Z > 0
        intrinsics: (B, N, 3, 3) camera intrinsics

    Returns:
        Mean L1 loss between predicted normalized rays [X/Z, Y/Z]
        and rays implied by intrinsics: [(u-cx)/fx, (v-cy)/fy].
    """
    assert local_points.ndim == 5 and local_points.shape[-1] == 3
    assert intrinsics.ndim == 4 and intrinsics.shape[-2:] == (3, 3)

    if stride > 1:
        local_points = local_points[:, :, ::stride, ::stride]
    B, N, H, W, _ = local_points.shape

    # Extract intrinsics
    intrinsics = intrinsics.to(device=local_points.device, dtype=local_points.dtype)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]

    # Build pixel grid (u, v)
    u = torch.arange(W, device=local_points.device, dtype=local_points.dtype)[None, None, None, :].expand(B, N, H, W)
    v = torch.arange(H, device=local_points.device, dtype=local_points.dtype)[None, None, :, None].expand(B, N, H, W)

    # Predicted normalized rays
    z = local_points[..., 2] + 1e-8
    ray_pred = local_points[..., :2] / z[..., None]

    # Rays from intrinsics
    ray_gt_x = (u - cx[..., None, None]) / fx[..., None, None]
    ray_gt_y = (v - cy[..., None, None]) / fy[..., None, None]
    ray_gt = torch.stack([ray_gt_x, ray_gt_y], dim=-1)

    # return F.smooth_l1_loss(ray_pred, ray_gt, reduction='mean')
    return F.l1_loss(ray_pred, ray_gt, reduction='mean')

def estimate_intrinsics_from_local_points(local_points: torch.Tensor) -> torch.Tensor:
    """
    Estimate camera intrinsics from 3D local points using least squares fitting to pinhole model.
    
    Given local_points in camera coordinates, this function estimates the intrinsics matrix
    by solving the pinhole projection equations:
        u = fx * (X/Z) + cx
        v = fy * (Y/Z) + cy
    
    Args:
        local_points: (B, N, H, W, 3) with [X, Y, Z] in camera coords, Z > 0
    
    Returns:
        estimated_intrinsics: (B, N, 3, 3) estimated camera intrinsics matrix
    """
    assert local_points.ndim == 5 and local_points.shape[-1] == 3
    
    B, N, H, W, _ = local_points.shape
    
    # Build pixel grid (u, v)
    u = torch.arange(W, device=local_points.device, dtype=local_points.dtype)[None, None, None, :].expand(B, N, H, W)
    v = torch.arange(H, device=local_points.device, dtype=local_points.dtype)[None, None, :, None].expand(B, N, H, W)
    
    # Extract X, Y, Z from local_points
    X = local_points[..., 0]  # (B, N, H, W)
    Y = local_points[..., 1]  # (B, N, H, W)
    Z = local_points[..., 2] + 1e-8  # (B, N, H, W), add epsilon to avoid division by zero
    
    # Flatten spatial dimensions for each batch and view
    u_flat = u.reshape(B, N, -1)  # (B, N, H*W)
    v_flat = v.reshape(B, N, -1)  # (B, N, H*W)
    X_over_Z = (X / Z).reshape(B, N, -1)  # (B, N, H*W)
    Y_over_Z = (Y / Z).reshape(B, N, -1)  # (B, N, H*W)
    
    # Estimate fx and cx: u = fx * (X/Z) + cx
    # This is a linear regression: u = fx * x_norm + cx
    # Solution: fx = cov(x_norm, u) / var(x_norm), cx = mean(u) - fx * mean(x_norm)
    mean_X_over_Z = X_over_Z.mean(dim=-1, keepdim=True)  # (B, N, 1)
    mean_u = u_flat.mean(dim=-1, keepdim=True)  # (B, N, 1)
    
    cov_X_u = ((X_over_Z - mean_X_over_Z) * (u_flat - mean_u)).mean(dim=-1)  # (B, N)
    var_X = ((X_over_Z - mean_X_over_Z) ** 2).mean(dim=-1) + 1e-8  # (B, N)
    
    pred_fx = cov_X_u / var_X  # (B, N)
    pred_cx = mean_u.squeeze(-1) - pred_fx * mean_X_over_Z.squeeze(-1)  # (B, N)
    
    # Estimate fy and cy: v = fy * (Y/Z) + cy
    mean_Y_over_Z = Y_over_Z.mean(dim=-1, keepdim=True)  # (B, N, 1)
    mean_v = v_flat.mean(dim=-1, keepdim=True)  # (B, N, 1)
    
    cov_Y_v = ((Y_over_Z - mean_Y_over_Z) * (v_flat - mean_v)).mean(dim=-1)  # (B, N)
    var_Y = ((Y_over_Z - mean_Y_over_Z) ** 2).mean(dim=-1) + 1e-8  # (B, N)
    
    pred_fy = cov_Y_v / var_Y  # (B, N)
    pred_cy = mean_v.squeeze(-1) - pred_fy * mean_Y_over_Z.squeeze(-1)  # (B, N)
    
    # Build intrinsics matrix (B, N, 3, 3)
    estimated_intrinsics = torch.zeros(B, N, 3, 3, device=local_points.device, dtype=local_points.dtype)
    estimated_intrinsics[:, :, 0, 0] = pred_fx  # fx
    estimated_intrinsics[:, :, 1, 1] = pred_fy  # fy
    estimated_intrinsics[:, :, 0, 2] = pred_cx  # cx
    estimated_intrinsics[:, :, 1, 2] = pred_cy  # cy
    estimated_intrinsics[:, :, 2, 2] = 1.0      # bottom-right
    
    return estimated_intrinsics


def intrinsics_ray_consistency_energy_v2(
        local_points: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
    """
    Estimate intrinsics from predicted local_points and compare with GT intrinsics.
    
    This function:
    1. Estimates fx, fy, cx, cy from local_points using least squares fitting to pinhole model
    2. Compares estimated intrinsics with GT intrinsics using smooth L1 loss (similar to direct_intrinsics_energy)

    Args:
        local_points: (B, N, H, W, 3) with [X, Y, Z] in camera coords, Z > 0
        intrinsics: (B, N, 3, 3) camera intrinsics (ground truth)

    Returns:
        Loss comparing estimated intrinsics with ground truth intrinsics
    """
    assert local_points.ndim == 5 and local_points.shape[-1] == 3
    assert intrinsics.ndim == 4 and intrinsics.shape[-2:] == (3, 3)

    # Estimate intrinsics from local points
    estimated_intrinsics = estimate_intrinsics_from_local_points(local_points)
    
    # Extract GT intrinsics
    intrinsics = intrinsics.to(device=local_points.device, dtype=local_points.dtype)
    gt_fx = intrinsics[..., 0, 0]  # (B, N)
    gt_fy = intrinsics[..., 1, 1]  # (B, N)
    gt_cx = intrinsics[..., 0, 2]  # (B, N)
    gt_cy = intrinsics[..., 1, 2]  # (B, N)
    
    # Extract estimated intrinsics
    pred_fx = estimated_intrinsics[..., 0, 0]  # (B, N)
    pred_fy = estimated_intrinsics[..., 1, 1]  # (B, N)
    pred_cx = estimated_intrinsics[..., 0, 2]  # (B, N)
    pred_cy = estimated_intrinsics[..., 1, 2]  # (B, N)
    
    # Compute losses similar to direct_intrinsics_energy
    # Focal lengths (more important)
    fx_loss = F.smooth_l1_loss(pred_fx, gt_fx, reduction='mean')
    fy_loss = F.smooth_l1_loss(pred_fy, gt_fy, reduction='mean')
    
    # Principal point (less critical)
    cx_loss = F.smooth_l1_loss(pred_cx, gt_cx, reduction='mean')
    cy_loss = F.smooth_l1_loss(pred_cy, gt_cy, reduction='mean')
    
    # Combine losses with higher weight on focal lengths
    total_loss = (2.0 * (fx_loss + fy_loss) + 1.0 * (cx_loss + cy_loss)) / 6
    
    return total_loss

# ----------------------
# Multiview Consistency Energy
# 1. Depth Multiview Consistency Energy
# 2. Depth Supervision Energy
# ----------------------
def depth_energy(predicted_depth: torch.Tensor,
                 predicted_depth_conf: torch.Tensor,
                 gt_depth: torch.Tensor,
                 gt_depth_mask: torch.Tensor,
                 confidence_percentile: float = 0.5,
                 alignment_mode: str = "per_scene",
                 scale_reg_weight: float = 0.0,
                 shift_reg_weight: float = 0.0) -> torch.Tensor:
    """
    Compute scale-shift invariant depth supervision loss with confidence weighting.
    
    This loss is invariant to affine transformations (scale and shift) of the predicted depth,
    making it suitable for monocular depth estimation where absolute scale is ambiguous.
    The loss is weighted by predicted confidence to allow the model to indicate uncertainty.
    
    **Alignment Strategy:**
    Depending on ``alignment_mode``, scale and shift parameters are computed either per
    batch sample across all views (``alignment_mode='per_scene'``) or per view
    (``alignment_mode='per_view'``), using only pixels that are:
    1. Valid in the ground truth mask (gt_depth_mask)
    2. In the top X% of predicted confidence values (default: top 50%)
    
    The confidence percentile threshold is computed independently per view when
    ``alignment_mode='per_view'`` and per scene when ``alignment_mode='per_scene'``.
    The final loss is then computed on all GT-valid pixels.
    
    **Confidence Weighting:**
    The confidence map is assumed to be output with activation: depth_conf = 1 + exp(conf),
    representing precision (inverse variance). The loss follows the formulation:
    
        L = depth_conf * error - log(depth_conf)
    
    This is equivalent to: L = (1/σ²) * error + log(σ²) where σ² = 1/depth_conf
    
    This encourages the model to:
    - Predict high confidence when errors are low (increases loss if confident but wrong)
    - Predict low confidence when errors are high (allows model to downweight uncertain predictions)
    - Avoid always predicting low confidence via the regularization term -log(depth_conf)
    
    Args:
        predicted_depth: Predicted depth values (B, S, H, W), (B, S, 1, H, W), or (B, S, H, W, 1)
        predicted_depth_conf: Predicted confidence values (B, S, H, W), (B, S, 1, H, W), or (B, S, H, W, 1)
                             Output with activation depth_conf = 1 + exp(conf), so values are >= 1
                             Higher values indicate higher confidence (lower uncertainty)
        gt_depth: Ground truth depth values (B, S, H, W), (B, S, 1, H, W), or (B, S, H, W, 1)
        gt_depth_mask: Binary mask for valid depth pixels (B, S, H, W), (B, S, 1, H, W), or (B, S, H, W, 1)
        confidence_percentile: Percentile of confidence values to use for alignment (default: 0.5 = top 50%, 0.4 = top 60%, 0.3 = top 70%)
        alignment_mode: Alignment mode for scale and shift computation (default: "per_scene")
        Valid values:
            - "per_scene": Compute scale and shift per scene across all views
            - "per_view": Compute scale and shift per view
        Note:
            - "per_scene" is more robust to outliers and provides consistent scale and shift across all views of the same scene.
            - "per_view" is more sensitive to local variations and provides per-view scale and shift.
    
    Additional args:
        scale_reg_weight: Weight for penalizing deviation of scale from 1.0 (default: 0.0, disabled)
        shift_reg_weight: Weight for penalizing deviation of shift from 0.0 (default: 0.0, disabled)

    Returns:
        Scalar depth loss value
    """
    # Normalize shapes to (B, S, H, W)
    # Handle both (B, S, H, W, 1) and (B, S, 1, H, W) formats
    if predicted_depth.ndim == 5:
        if predicted_depth.shape[-1] == 1:
            predicted_depth = predicted_depth.squeeze(-1)
        elif predicted_depth.shape[-3] == 1:
            predicted_depth = predicted_depth.squeeze(-3)
    
    if predicted_depth_conf.ndim == 5:
        if predicted_depth_conf.shape[-1] == 1:
            predicted_depth_conf = predicted_depth_conf.squeeze(-1)
        elif predicted_depth_conf.shape[-3] == 1:
            predicted_depth_conf = predicted_depth_conf.squeeze(-3)
    
    if gt_depth.ndim == 5:
        if gt_depth.shape[-1] == 1:
            gt_depth = gt_depth.squeeze(-1)
        elif gt_depth.shape[-3] == 1:
            gt_depth = gt_depth.squeeze(-3)
    
    if gt_depth_mask.ndim == 5:
        if gt_depth_mask.shape[-1] == 1:
            gt_depth_mask = gt_depth_mask.squeeze(-1)
        elif gt_depth_mask.shape[-3] == 1:
            gt_depth_mask = gt_depth_mask.squeeze(-3)
    
    assert predicted_depth.shape == gt_depth.shape, \
        f"Shape mismatch: predicted_depth {predicted_depth.shape} vs gt_depth {gt_depth.shape}"
    assert predicted_depth.shape == gt_depth_mask.shape, \
        f"Shape mismatch: predicted_depth {predicted_depth.shape} vs gt_depth_mask {gt_depth_mask.shape}"
    assert predicted_depth.shape == predicted_depth_conf.shape, \
        f"Shape mismatch: predicted_depth {predicted_depth.shape} vs predicted_depth_conf {predicted_depth_conf.shape}"
    
    device = predicted_depth.device
    dtype = predicted_depth.dtype

    if alignment_mode not in {"per_view", "per_scene"}:
        raise ValueError(
            f"Invalid alignment_mode '{alignment_mode}'. Expected 'per_view' or 'per_scene'."
        )
    
    # Convert mask to boolean and flatten spatial dimensions
    gt_mask = gt_depth_mask.bool()  # (B, S, H, W)
    
    # Flatten to (B*S, H*W) for easier processing
    B, S, H, W = predicted_depth.shape
    pred_flat = predicted_depth.reshape(B * S, H * W)  # (B*S, H*W)
    gt_flat = gt_depth.reshape(B * S, H * W)          # (B*S, H*W)
    conf_flat = predicted_depth_conf.reshape(B * S, H * W)  # (B*S, H*W)
    gt_mask_flat = gt_mask.reshape(B * S, H * W)     # (B*S, H*W)
    
    conf_thresholds_expanded = None
    if alignment_mode == "per_scene":
        conf_per_scene = conf_flat.reshape(B, S, H * W)  # (B, S, H*W)
        gt_mask_per_scene = gt_mask_flat.reshape(B, S, H * W)  # (B, S, H*W)

        conf_thresholds = []
        for b in range(B):
            valid_confs_scene = conf_per_scene[b][gt_mask_per_scene[b]]  # (N_valid,)
            if valid_confs_scene.numel() > 0:
                threshold = torch.quantile(valid_confs_scene, confidence_percentile)
            else:
                threshold = torch.tensor(0.0, device=device, dtype=dtype)
            conf_thresholds.append(threshold)
        conf_thresholds = torch.stack(conf_thresholds)  # (B,)
        conf_thresholds_expanded = conf_thresholds.view(B, 1, 1).expand(B, S, H * W).reshape(B * S, H * W)
    else:
        conf_per_view = conf_flat.reshape(B, S, H * W)  # (B, S, H*W)
        gt_mask_per_view = gt_mask_flat.reshape(B, S, H * W)  # (B, S, H*W)

        conf_thresholds = torch.zeros((B, S), device=device, dtype=dtype)
        for b in range(B):
            for s in range(S):
                valid_confs_view = conf_per_view[b, s][gt_mask_per_view[b, s]]  # (N_valid,)
                if valid_confs_view.numel() > 0:
                    conf_thresholds[b, s] = torch.quantile(valid_confs_view, confidence_percentile)

        conf_thresholds_expanded = conf_thresholds.view(B, S, 1).expand(B, S, H * W).reshape(B * S, H * W)
    
    # Create confidence mask: top percentage means conf >= threshold
    conf_mask_flat = conf_flat >= conf_thresholds_expanded  # (B*S, H*W)
     
    # Combined mask for alignment: both GT valid AND top percentage confidence
    alignment_mask_flat = gt_mask_flat & conf_mask_flat  # (B*S, H*W)
    
    # Count valid pixels per sequence for alignment
    num_valid_alignment = alignment_mask_flat.sum(dim=1)  # (B*S,)
    eps = 1e-8

    # Early return if no valid pixels for alignment in the entire batch
    if num_valid_alignment.sum() == 0:
        return torch.zeros((), device=device, dtype=dtype)
    
    # Create masked versions for alignment (invalid values set to 0)
    pred_masked = torch.where(alignment_mask_flat, pred_flat, torch.zeros_like(pred_flat))
    gt_masked = torch.where(alignment_mask_flat, gt_flat, torch.zeros_like(gt_flat))
    
    scale_values = None
    shift_values = None

    if alignment_mode == "per_scene":
        pred_masked_per_batch = pred_masked.reshape(B, S, H * W)  # (B, S, H*W)
        gt_masked_per_batch = gt_masked.reshape(B, S, H * W)      # (B, S, H*W)
        num_valid_per_batch = num_valid_alignment.reshape(B, S)   # (B, S)

        total_valid_per_batch = num_valid_per_batch.sum(dim=1).clamp(min=1)  # (B,)
        pred_sum_per_batch = pred_masked_per_batch.sum(dim=(1, 2))  # (B,)
        gt_sum_per_batch = gt_masked_per_batch.sum(dim=(1, 2))      # (B,)
        pred_mean = pred_sum_per_batch / total_valid_per_batch  # (B,)
        gt_mean = gt_sum_per_batch / total_valid_per_batch      # (B,)

        pred_flat_per_batch = pred_flat.reshape(B, S, H * W)  # (B, S, H*W)
        alignment_mask_per_batch = alignment_mask_flat.reshape(B, S, H * W)  # (B, S, H*W)

        pred_centered = (pred_flat_per_batch - pred_mean.view(B, 1, 1)) * alignment_mask_per_batch.float()
        gt_centered = (gt_flat.reshape(B, S, H * W) - gt_mean.view(B, 1, 1)) * alignment_mask_per_batch.float()

        pred_var = (pred_centered ** 2).sum(dim=(1, 2)) / total_valid_per_batch  # (B,)
        # gt_var = (gt_centered ** 2).sum(dim=(1, 2)) / total_valid_per_batch  # (B,)
        covariance = (pred_centered * gt_centered).sum(dim=(1, 2)) / total_valid_per_batch  # (B,)

        scale = torch.where(
            pred_var > eps,
            covariance / (pred_var + eps),
            torch.ones_like(pred_var)
        )  # (B,)
        shift = gt_mean - scale * pred_mean  # (B,)

        pred_aligned = (
            scale.view(B, 1, 1) * pred_flat_per_batch + shift.view(B, 1, 1)
        ).reshape(B * S, H * W)  # (B*S, H*W)

        scale_values = scale.view(-1)
        shift_values = shift.view(-1)
    else:
        num_views = B * S
        alignment_mask_float = alignment_mask_flat.float()
        total_valid_per_view = num_valid_alignment.clamp(min=1)
        pred_sum_per_view = pred_masked.sum(dim=1)  # (B*S,)
        gt_sum_per_view = gt_masked.sum(dim=1)      # (B*S,)
        pred_mean = pred_sum_per_view / total_valid_per_view  # (B*S,)
        gt_mean = gt_sum_per_view / total_valid_per_view      # (B*S,)

        pred_centered = (pred_flat - pred_mean.view(num_views, 1)) * alignment_mask_float  # (B*S, H*W)
        gt_centered = (gt_flat - gt_mean.view(num_views, 1)) * alignment_mask_float        # (B*S, H*W)

        pred_var = (pred_centered ** 2).sum(dim=1) / total_valid_per_view  # (B*S,)
        # gt_var = (gt_centered ** 2).sum(dim=1) / total_valid_per_view  # (B*S,)
        covariance = (pred_centered * gt_centered).sum(dim=1) / total_valid_per_view      # (B*S,)

        scale = torch.where(
            pred_var > eps,
            covariance / (pred_var + eps),
            torch.ones_like(pred_var)
        )  # (B*S,)

        shift = gt_mean - scale * pred_mean  # (B*S,)

        pred_aligned = scale.view(num_views, 1) * pred_flat + shift.view(num_views, 1)  # (B*S, H*W)

        scale_values = scale.view(-1)
        shift_values = shift.view(-1)
    # import ipdb; ipdb.set_trace()
    # Compute smooth L1 loss for each pixel using PyTorch built-in
    # reduction='none' returns per-element loss without averaging
    l1_loss = F.l1_loss(pred_aligned, gt_flat, reduction='none')  # (B*S, H*W)
    
    # Apply confidence weighting: loss = depth_conf * error - log(depth_conf)
    # Clamp confidence to prevent numerical issues (should be >= 1 due to activation)
    conf_clamped = torch.clamp(conf_flat, min=1.0, max=1e10)  # (B*S, H*W)
    
    # Weighted loss per pixel: confidence * error - log(confidence)
    weighted_loss = conf_clamped * l1_loss - torch.log(conf_clamped)  # (B*S, H*W)
    # weighted_loss = conf_clamped * l1_loss / conf_clamped.sum(dim=1, keepdim=True)
    # weighted_loss = l1_loss
    
    # Apply GT mask for final loss computation (use all valid GT pixels)
    weighted_loss_masked = weighted_loss * gt_mask_flat.float()  # (B*S, H*W)
    
    # Average over all valid GT pixels
    # Masked pixels contribute 0, so we sum all and divide by total valid count
    num_valid_gt_total = gt_mask_flat.sum().clamp(min=1)
    total_loss = weighted_loss_masked.sum() / num_valid_gt_total

    if scale_reg_weight != 0.0 or shift_reg_weight != 0.0:
        if scale_values is None or shift_values is None:
            raise RuntimeError("Scale and shift values must be computed before applying regularization.")
        if scale_reg_weight != 0.0:
            scale_reg = ((scale_values - 1.0) ** 2).mean()
            total_loss = total_loss + scale_reg_weight * scale_reg
        if shift_reg_weight != 0.0:
            shift_reg = (shift_values ** 2).mean()
            total_loss = total_loss + shift_reg_weight * shift_reg
        print(f"Scale values: {scale_values.mean()}, Shift values: {shift_values.mean()}")
    
    return total_loss, scale_values, shift_values


def redundancy_consistency_energy(depth_points: torch.Tensor,
                                  depth_points_conf: torch.Tensor,
                                  world_points: torch.Tensor,
                                  world_points_conf: torch.Tensor,
                                  use_confidence: bool = False,
                                  epsilon: float = 1e-6) -> torch.Tensor:
    """
    Enforce consistency between depth-derived world points and directly predicted world points.

    Args:
        depth_points: (B, S, H, W, 3) or (B, S, N, 3) tensor of 3D points reconstructed from depth maps.
        depth_points_conf: matching confidence tensor (optional, can be None).
        world_points: (B, S, H, W, 3) tensor from global point head.
        world_points_conf: matching confidence tensor (optional, can be None).
        use_confidence: if True, weight the loss by combined confidence.
        epsilon: numerical stability term for weighting.

    Returns:
        Scalar tensor of redundancy consistency loss.
    """
    assert depth_points.shape == world_points.shape, "depth_points and world_points must have identical shapes"

    point_diff = depth_points - world_points.detach()
    l1_dist = torch.abs(point_diff).sum(dim=-1)

    if use_confidence and (depth_points_conf is not None or world_points_conf is not None):
        depth_w = depth_points_conf if depth_points_conf is not None else torch.ones_like(l1_dist)
        # world_w = world_points_conf if world_points_conf is not None else torch.ones_like(l1_dist)
        combined_weight = depth_w
        loss = (combined_weight * l1_dist).sum() / (combined_weight.sum() + epsilon)
    else:
        loss = l1_dist.mean()
    
    # conf_diff = depth_points_conf - world_points_conf.detach()
    # l1_dist_conf = torch.abs(conf_diff).sum(dim=-1)
    # loss = loss + l1_dist_conf.mean()
    
    return loss