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

from constraints import (direct_pose_energy, direct_intrinsics_energy, \
    intrinsics_ray_consistency_energy_v2, estimate_intrinsics_from_local_points, \
    depth_energy, redundancy_consistency_energy)
from objective import depth_multiview_consistency_energy

# ----------------------
# Combined TCO Loss
# ----------------------
def compute_tco_losses(
        predictions: Dict[str, torch.Tensor],
        images: torch.Tensor,
        depths: Optional[torch.Tensor] = None,
        depth_masks: Optional[torch.Tensor] = None,
        depth_alignment_mode: str = "per_scene",
        depth_confidence_percentile: float = 0.5,
        depth_scale_reg_weight: float = 0.0,
        depth_shift_reg_weight: float = 0.0,
        gt_extrinsics: Optional[torch.Tensor] = None,
        gt_intrinsics: Optional[torch.Tensor] = None,
        lambda_pose: float = 1.0,
        pose_translation_weight: float = 1.0,
        lambda_intrinsics: float = 0.0,
        lambda_depth: float = 0.0,
        lambda_mv_consistency: float = 0.0,
        pose_rot_loss_type: str = 'cosine',
        pose_trans_loss_type: str = 'normed_l1',
        mv_depth_weight: float = 0.0,
        mv_normal_weight: float = 0.0,
        mv_rgb_weight: float = 1.0,
        num_view_groups: int = 2,
        use_cos_weighting: bool = True,
        lambda_redundancy: float = 0.0,
        canonicalize_to_first_view: bool = True,
        meta_info: Dict[str, object] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute combined TTT losses using available energy terms.

    Uses the provided weights to combine:
    - direct pose energy (if gt_extrinsics provided and poses predicted)
    - direct intrinsics energy (if intrinsics supervision weight > 0)
    - depth supervision energy (if lambda_depth > 0 and depths/depth_masks provided)
    - depth-based multiview consistency (if lambda_mv_consistency > 0 and depth available)
    """

    device = images.device
    dtype = images.dtype

    total_loss = torch.zeros((), device=device, dtype=dtype)
    loss_dict: Dict[str, torch.Tensor] = {
        'pose_rot_loss': torch.zeros((), device=device, dtype=dtype),
        'pose_trans_loss': torch.zeros((), device=device, dtype=dtype),
        'intrinsics_loss': torch.zeros((), device=device, dtype=dtype),
        'depth_loss': torch.zeros((), device=device, dtype=dtype),
        'mv_consistency_loss': torch.zeros((), device=device, dtype=dtype),
        'redundancy_loss': torch.zeros((), device=device, dtype=dtype),
    }
    
    # ----------------------
    # Pose supervision (direct or relative depending on config)
    # ----------------------
    if lambda_pose > 0 and gt_extrinsics is not None and 'camera_poses' in predictions:
        predicted_poses = predictions['camera_poses']  # (B, S, 4, 4)
        # Align GT dtype/device to predictions to avoid mixed-precision errors
        gt_extrinsics = gt_extrinsics.to(device=predicted_poses.device, dtype=predicted_poses.dtype)
        # Direct pose supervision; canonicalization handled inside direct_pose_energy
        rot_loss, trans_loss = direct_pose_energy(
            predicted_poses=predicted_poses,
            gt_extrinsics=gt_extrinsics,
            rot_loss_type=pose_rot_loss_type,
            trans_loss_type=pose_trans_loss_type,
            canonicalize_to_first_view=canonicalize_to_first_view,
            pose_type=predictions.get('pose_type', 'abs'),
        )
        # Allow weighting translation term
        pose_loss = rot_loss + pose_translation_weight * trans_loss
        total_loss = total_loss + lambda_pose * pose_loss
        loss_dict['pose_rot_loss'] = rot_loss.detach()
        loss_dict['pose_trans_loss'] = trans_loss.detach()

    # ----------------------
    # Intrinsics supervision
    # ----------------------
    if lambda_intrinsics > 0:
        if 'intrinsics' in predictions and gt_intrinsics is not None:
            pred_intr = predictions['intrinsics']
            gt_intrinsics = gt_intrinsics.to(device=pred_intr.device, dtype=pred_intr.dtype)
            intr_loss = direct_intrinsics_energy(pred_intr, gt_intrinsics)
            total_loss = total_loss + lambda_intrinsics * intr_loss
            loss_dict['intrinsics_loss'] = intr_loss.detach()
        elif 'local_points' in predictions and gt_intrinsics is not None:
            pred_local_points = predictions['local_points']
            gt_intrinsics = gt_intrinsics.to(device=pred_local_points.device, dtype=pred_local_points.dtype)
            intr_loss = intrinsics_ray_consistency_energy_v2(pred_local_points, gt_intrinsics)
            # intr_loss = intrinsics_ray_consistency_energy(pred_local_points, gt_intrinsics)
            total_loss = total_loss + lambda_intrinsics * intr_loss
            loss_dict['intrinsics_loss'] = intr_loss.detach()
        else:
            loss_dict['intrinsics_loss'] = torch.zeros((), device=device, dtype=dtype)

    # ----------------------
    # Depth supervision (scale-shift invariant)
    # ----------------------
    if lambda_depth > 0:
        if 'depth' in predictions and depths is not None and depth_masks is not None:
            pred_depth = predictions['depth']
            pred_depth_conf = predictions.get('depth_conf', None)
            # Align GT dtype/device to predictions
            depths = depths.to(device=pred_depth.device, dtype=pred_depth.dtype)
            depth_masks = depth_masks.to(device=pred_depth.device, dtype=pred_depth.dtype)
            
            depth_loss, scales, shifts = depth_energy(
                predicted_depth=pred_depth,
                predicted_depth_conf=pred_depth_conf,
                gt_depth=depths,
                gt_depth_mask=depth_masks,
                alignment_mode=depth_alignment_mode,
                confidence_percentile=depth_confidence_percentile,
                scale_reg_weight=depth_scale_reg_weight,
                shift_reg_weight=depth_shift_reg_weight,
            )
            total_loss = total_loss + lambda_depth * depth_loss
            loss_dict['depth_loss'] = depth_loss.detach()
        else:
            loss_dict['depth_loss'] = torch.zeros((), device=device, dtype=dtype)

    # ----------------------
    # Multiview consistency (depth-based 2DGS)
    # ----------------------
    if lambda_mv_consistency > 0:
        if 'depth' in predictions and 'camera_poses' in predictions and 'intrinsics' in predictions:
            depth_for_mv = predictions['depth']
            # Normalize depth shape to (B, S, H, W)
            if depth_for_mv.ndim == 5 and depth_for_mv.shape[-1] == 1:
                depth_for_mv = depth_for_mv.squeeze(-1)
            if depth_for_mv.ndim == 3:
                depth_for_mv = depth_for_mv.unsqueeze(0)

            # gt depths for multiview consistency
            if depths is not None and depth_masks is not None:
                gt_depths = depths.squeeze(2)
                gt_depths_mask = depth_masks.squeeze(2)
                assert scales is not None and shifts is not None, "Scale and shifts must be assigned"
                rescaled_gt_depths = gt_depths_mask * (gt_depths - shifts) / scales # TODO: scale and shifts may be unassigned, need to resolve the issue later
            else:
                rescaled_gt_depths = None
            mv_loss = depth_multiview_consistency_energy(
                depth=depth_for_mv,
                camera_poses=predictions['camera_poses'],
                intrinsics=predictions['intrinsics'],
                real_depth_gt=rescaled_gt_depths,
                rgb_images=images,
                depth_conf=predictions.get('depth_conf', None),
                rgb_weight=mv_rgb_weight,
                depth_weight=mv_depth_weight,
                normal_weight=mv_normal_weight,
                meta_info=meta_info,
                num_view_groups=num_view_groups,
                use_cos_weighting=use_cos_weighting,
            )
            total_loss = total_loss + lambda_mv_consistency * mv_loss
            loss_dict['mv_consistency_loss'] = mv_loss.detach()
        elif 'local_points' in predictions and 'camera_poses' in predictions:
            # Pi3 case: compute depth and intrinsics from local_points
            local_points = predictions['local_points']  # (B, N, H, W, 3)
            
            # Extract depth from local_points (Z component)
            depth_for_mv = local_points[..., 2]  # (B, N, H, W)
            
            # Estimate intrinsics from local_points using the shared function
            estimated_intrinsics = estimate_intrinsics_from_local_points(local_points)
            
            # Call the same multiview consistency function
            mv_loss = depth_multiview_consistency_energy(
                depth=depth_for_mv,
                camera_poses=predictions['camera_poses'],
                intrinsics=estimated_intrinsics,
                real_depth_gt=None,  # Pi3 doesn't use GT depth for MV consistency
                rgb_images=images,
                depth_conf=predictions.get('conf', None),  # Pi3 uses 'conf' key
                rgb_weight=mv_rgb_weight,
                depth_weight=mv_depth_weight,
                normal_weight=mv_normal_weight,
                meta_info=meta_info,
                num_view_groups=num_view_groups,
                use_cos_weighting=use_cos_weighting,
            )
            total_loss = total_loss + lambda_mv_consistency * mv_loss
            loss_dict['mv_consistency_loss'] = mv_loss.detach()
        else:
            loss_dict['mv_consistency_loss'] = torch.zeros((), device=device, dtype=dtype)

    # ----------------------
    # Redundancy consistency (depth vs global point head)
    # ----------------------
    if lambda_redundancy > 0:
        if 'depth_points' in predictions and 'world_points' in predictions:
            depth_pts = predictions['depth_points']
            world_pts = predictions['world_points']
            depth_conf = predictions.get('depth_conf', None)
            world_conf = predictions.get('world_points_conf', None)
            redund_loss = redundancy_consistency_energy(
                depth_points=depth_pts,
                depth_points_conf=depth_conf,
                world_points=world_pts,
                world_points_conf=world_conf,
            )
            total_loss = total_loss + lambda_redundancy * redund_loss
            loss_dict['redundancy_loss'] = redund_loss.detach()
        else:
            loss_dict['redundancy_loss'] = torch.zeros((), device=device, dtype=dtype)

    return total_loss, loss_dict