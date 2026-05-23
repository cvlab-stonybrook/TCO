import torch
from typing import Optional, Dict
import numpy as np
import matplotlib.cm as cm
from utils.geometry import unproject_depth_to_points, to_homogeneous, invert_se3, mat_to_quat


def _rotation_matrix_to_wxyz_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion in WXYZ order (as expected by gsplat)."""
    q_xyzw = mat_to_quat(R)  # (..., 4) [x,y,z,w]
    q_wxyz = torch.cat([q_xyzw[..., 3:4], q_xyzw[..., 0:3]], dim=-1)
    return q_wxyz


def depth_multiview_consistency_energy(depth: torch.Tensor,
                                       camera_poses: torch.Tensor,
                                       intrinsics: torch.Tensor,
                                       depth_conf: Optional[torch.Tensor] = None,
                                       real_depth_gt: Optional[torch.Tensor] = None,
                                       rgb_images: Optional[torch.Tensor] = None,
                                       rgb_weight: float = 1.0,
                                       depth_weight: float = 0.0,
                                       normal_weight: float = 0.0,
                                       scale_factor: float = 0.5, # TODO: for tuning, remove later, 0.5 originally
                                       thickness_factor: float = 1e-3,
                                       eps2d: float = 0.3,
                                       num_view_groups: int = 2,
                                       use_cos_weighting: bool = True,
                                       meta_info: Optional[Dict[str, object]] = None) -> torch.Tensor:
    """
    Ultra-efficient multiview consistency using 2D Gaussian Splatting.
    
    Leverages gsplat's 2DGS with normals derived from local point maps for 
    geometrically-aware feature and RGB splatting between views.
    
    Args:
        depth: (B, N, H_img, W_img) - predicted depth maps
        camera_poses: (B, N, 4, 4) - camera poses (camera-to-world)
        intrinsics: (B, N, 3, 3) - camera intrinsics for each view (at H_img, W_img resolution)
        rgb_images: (B, N, 3, H_img, W_img) - RGB images
        depth_conf: Optional (B, N, H_img, W_img) - depth confidence values
        rgb_weight: weight for RGB consistency loss
        depth_weight: weight for depth consistency loss (includes depth gradient matching)
        normal_weight: weight for normal consistency loss
        scale_factor: footprint scale along image tangents (smaller → crisper)
        thickness_factor: footprint scale along normal (smaller → thinner)
        eps2d: minimal 2D covariance epsilon in gsplat (smaller → crisper)
        num_view_groups: number of groups to split views into; each group takes turns as source
        use_cos_weighting: if True, scale splat footprints by cos(normal, optical axis) to shrink
            edge-on surfaces; if False, use uniform scale regardless of orientation
        
    Returns:
        combined_loss: weighted combination of consistency losses using 2DGS
    """
    # Import gsplat lazily to keep module import light
    try:
        import gsplat
    except Exception as e:
        raise ImportError("gsplat is required for 2DGS-based multiview consistency. Please install gsplat.") from e

    assert depth.ndim == 4, "depth must be (B, N, H, W)"
    assert camera_poses.ndim == 4 and camera_poses.shape[-2:] == (4, 4), "camera_poses must be (B, N, 4, 4)"
    assert intrinsics.ndim == 4 and intrinsics.shape[-2:] == (3, 3), "intrinsics must be (B, N, 3, 3)"

    device = depth.device
    dtype = depth.dtype
    B, S, H, W = depth.shape

    # Optional visualization via wandb (only logs when wandb is initialized)
    wandb_module = None
    try:
        import wandb  # type: ignore
        if getattr(wandb, "run", None) is not None:
            wandb_module = wandb
    except ImportError:
        pass

    # If RGB provided, ensure correct shape and dtype
    if rgb_images is not None:
        assert rgb_images.ndim == 5 and rgb_images.shape[2] == 3, "rgb_images must be (B, N, 3, H, W)"
        rgb_images = rgb_images.to(device=device, dtype=dtype)
    
    # Depth confidence optional; if provided, ensure proper shape
    if depth_conf is not None:
        assert depth_conf.shape == depth.shape, "depth_conf must match depth shape (B, N, H, W)"
        depth_conf = depth_conf.to(device=device, dtype=dtype)

    # Accumulators
    total_rgb_loss = torch.zeros((), device=device, dtype=dtype)
    total_depth_loss = torch.zeros((), device=device, dtype=dtype)
    total_normal_loss = torch.zeros((), device=device, dtype=dtype)
    num_terms_rgb = 0
    num_terms_depth = 0
    num_terms_normal = 0

    # Precompute world-to-camera for targets per batch/view
    cam_poses_h = to_homogeneous(camera_poses)
    world_to_cam = invert_se3(cam_poses_h)  # (B, N, 4, 4)

    # eps for numerical stability
    eps = 1e-6 if dtype == torch.float32 else 1e-4

    # Process each scene independently (gsplat doesn't mix different scenes in one call)
    for b in range(B):
        # Split views into even groups
        if S <= 1:
            continue
        
        # Determine group size based on num_view_groups
        num_groups = min(num_view_groups, S)  # Can't have more groups than views
        views_per_group = S // num_groups
        
        # Multi-view consistency requires at least 2 groups
        assert num_groups >= 2, f"num_groups must be >= 2 for multi-view consistency, got {num_groups} (S={S}, num_view_groups={num_view_groups})"
        
        # Randomly permute views first for variety
        perm = torch.randperm(S, device=device)
        
        # Split into groups
        view_groups = []
        for g in range(num_groups):
            start_idx = g * views_per_group
            if g == num_groups - 1:
                # Last group gets any remaining views
                end_idx = S
            else:
                end_idx = (g + 1) * views_per_group
            view_groups.append(perm[start_idx:end_idx])
        
        # Loop over each group as source, rendering into all other groups
        for group_idx in range(min(1, num_groups // 2)): # for faster inference, only project from the first group to all the groups
            src_idx = view_groups[group_idx]
            
            # Target view is a single view sampled from non-source views
            # non_src_views = torch.cat([view_groups[i] for i in range(num_groups) if i != group_idx], dim=0)
            # tgt_idx = non_src_views[torch.randint(len(non_src_views), (1,))]  # (1,) # TODO: for debugging, use the first non-source view
            tgt_idx = perm
            # tgt_idx = non_src_views

            K_src = src_idx.shape[0]

            # Unproject depths for all source views in one call
            depth_b_src = depth[b, src_idx].unsqueeze(0)           # (1, K_src, H, W)
            intr_b_src = intrinsics[b, src_idx].unsqueeze(0)       # (1, K_src, 3, 3)
            points_cam, valid_mask = unproject_depth_to_points(depth_b_src, intr_b_src) # (1, K_src, H, W, 3) and (1, K_src, H, W)
            points_cam = points_cam[0]       # (K_src, H, W, 3)
            valid_mask = valid_mask[0]       # (K_src, H, W)
            
            # Tangent bases from image gradients (vectorized across K_src)
            v_v, v_u = torch.gradient(points_cam, dim=[1, 2])  # dP/dv (H): (K_src, H, W, 3), dP/du (W): (K_src, H, W, 3)
            v_v = -v_v # To make it compatible with the right-handed coordinate system

            v_u_norm = torch.norm(v_u, dim=-1, keepdim=True).clamp_min(eps) # (K_src, H, W, 1)
            v_v_norm = torch.norm(v_v, dim=-1, keepdim=True).clamp_min(eps) # (K_src, H, W, 1)
            t_u = v_u / v_u_norm # (K_src, H, W, 3)
            t_v = v_v / v_v_norm # (K_src, H, W, 3)
            n = torch.cross(t_u, t_v, dim=-1)
            n_norm = torch.norm(n, dim=-1, keepdim=True).clamp_min(eps)
            n = n / n_norm # (K_src, H, W, 3)
            # Recompute t_v to ensure an orthonormal right-handed frame
            t_v = torch.cross(n, t_u, dim=-1)
            # update v_v_norm by projecting v_v onto the new t_v
            v_v_norm = torch.norm(torch.sum(v_v * t_v, dim=-1, keepdim=True), dim=-1, keepdim=True).clamp_min(eps)

            # Compute cosine of angle between normal and optical axis (z-axis in camera coords)
            # For surfaces PARALLEL to image plane (normal perpendicular to plane, facing camera), cos ≈ 1 (larger scale)
            # For surfaces PERPENDICULAR to image plane (normal parallel to plane, side-facing), cos ≈ 0 (smaller scale, noisy)
            cos_normal_optical = torch.abs(n[..., 2])  # (K_src, H, W)
            
            # Scales adapted by normal orientation
            cos_weight = cos_normal_optical if use_cos_weighting else 1.0
            s_u = scale_factor * v_u_norm.squeeze(-1) * cos_weight  # (K_src, H, W)
            s_v = scale_factor * v_v_norm.squeeze(-1) * cos_weight  # (K_src, H, W)
            s_n = thickness_factor * 0.5 * (s_u + s_v) # (K_src, H, W)
            scales = torch.stack([s_u, s_v, s_n], dim=-1)  # (K_src, H, W, 3)

            # Per-pixel local rotation in camera (columns: t_u, t_v, n), then map to world
            R_pix_cam = torch.stack([t_u, t_v, n], dim=-1)  # (K_src, H, W, 3, 3)

            R_src = cam_poses_h[b, src_idx, :3, :3]  # (K_src, 3, 3)
            R_src_exp = R_src.view(K_src, 1, 1, 3, 3).expand(K_src, H, W, 3, 3)
            R_pix_world = torch.matmul(R_src_exp, R_pix_cam)  # (K_src, H, W, 3, 3)
            quats_wxyz = _rotation_matrix_to_wxyz_quat(R_pix_world)  # (K_src, H, W, 4)

            # Means in world: reshape to avoid ambiguous broadcasting in matmul
            t_src = cam_poses_h[b, src_idx, :3, 3]  # (K_src, 3)
            R_src_T = R_src.transpose(1, 2)  # (K_src, 3, 3)
            points_cam_flat = points_cam.view(K_src, H * W, 3)
            means_world = torch.bmm(points_cam_flat, R_src_T).view(K_src, H, W, 3) + t_src.view(K_src, 1, 1, 3) # (K_src, H, W, 3)

            # Opacities
            if depth_conf is not None:
                # depth_conf = 1 + exp(conf) ∈ (1, ∞). Map to (0, 1): w = (c-1)/c = 1 - 1/c
                assert torch.all(depth_conf[b, src_idx] >= 1), "depth_conf must be greater than or equal to 1"
                conf_src = depth_conf[b, src_idx].to(dtype=dtype) # (K_src, H, W)
                conf_src = torch.clamp(conf_src, min=1.0 + 1e-6)
                conf_w = 1.0 - (1.0 / conf_src)
                opa = conf_w * valid_mask.float()
            else:
                opa = valid_mask.float()
            opa = torch.clamp(opa, 0.0, 1.0)

            # Colors from source features/rgb at image resolution
            colors_rgb = None
            if rgb_images is not None and rgb_weight > 0:
                rgb_src = rgb_images[b, src_idx]  # (K_src, 3, H, W)
                colors_rgb = rgb_src.permute(0, 2, 3, 1).contiguous()  # (K_src, H, W, 3)

            # Flatten across all source views and spatial locations, keep only valid
            flat_mask = valid_mask.reshape(K_src * H * W)
            means_b = means_world.reshape(K_src * H * W, 3)[flat_mask]
            quats_b = quats_wxyz.reshape(K_src * H * W, 4)[flat_mask]
            scales_b = scales.reshape(K_src * H * W, 3)[flat_mask]
            scales_b = torch.clamp(scales_b, min=eps)
            opa_b = opa.reshape(K_src * H * W)[flat_mask]

            cols_rgb_b = None
            if colors_rgb is not None:
                cols_rgb_b = colors_rgb.reshape(K_src * H * W, -1)[flat_mask]

            if means_b.numel() == 0:
                continue

            # Target cameras (vectorized across all target views)
            viewmats_b = world_to_cam[b, tgt_idx]  # (K_tgt, 4, 4)
            Ks_b = intrinsics[b, tgt_idx]         # (K_tgt, 3, 3)

            # Visibility from expected depth across all targets
            N_b = means_b.shape[0]
            dummy_cols = torch.ones((N_b, 1), device=device, dtype=dtype)
            dummy_cols_cam = dummy_cols.unsqueeze(0).expand(viewmats_b.shape[0], N_b, 1)
            render_depth, render_alp, render_normals, _, _, _, _ = gsplat.rasterization_2dgs(
                means=means_b,
                quats=quats_b,
                scales=scales_b,
                opacities=opa_b,
                colors=dummy_cols_cam,
                viewmats=viewmats_b,
                Ks=Ks_b,
                width=W,
                height=H,
                packed=False,
                eps2d=eps2d,
                tile_size=16,
                render_mode='ED',
            )
            # Shapes: depth [K_tgt, H, W, 1], alpha [K_tgt, H, W, 1]
            pred_depth = render_depth[..., 0]  # (K_tgt, H, W)
            pred_normals = render_normals[..., :3]  # (K_tgt, H, W, 3)
            alpha_vis = render_alp[..., 0]     # (K_tgt, H, W)
            tgt_depth = depth[b, tgt_idx]      # (K_tgt, H, W)
            if real_depth_gt is not None:
                real_tgt_depth = real_depth_gt[b, tgt_idx]
            # vis_mask = (pred_depth <= tgt_depth + 1e-4).float()
            vis_mask = (pred_depth <= tgt_depth + tgt_depth.mean() * 0.0001).float() # TODO: tune 0.01
            weight_map = alpha_vis * vis_mask
            weight_denom = weight_map.sum() + eps

            # Normal rendering loss across all targets (compare to normals from target depth)
            if pred_normals is not None:
                # Compute GT normals in camera coordinates from target depth
                tgt_depth_in_n = tgt_depth.unsqueeze(0)    # (1, K_tgt, H, W)
                Ks_in_n = Ks_b.unsqueeze(0)               # (1, K_tgt, 3, 3)
                tgt_pts_cam_n, tgt_valid_n = unproject_depth_to_points(tgt_depth_in_n, Ks_in_n)  # (1,K,H,W,3),(1,K,H,W)
                tgt_pts_cam_n = tgt_pts_cam_n[0]  # (K_tgt, H, W, 3)
                # Gradients over height, width (v, u)
                v_v_n, v_u_n = torch.gradient(tgt_pts_cam_n, dim=[1, 2])
                v_v_n = -v_v_n # To make it compatible with the right-handed coordinate system
                # Compute normals via cross product and normalize
                tgt_normals = torch.cross(v_u_n, v_v_n, dim=-1)
                norm = torch.norm(tgt_normals, dim=-1, keepdim=True).clamp_min(eps)
                tgt_normals = tgt_normals / norm
                # Keep for visualization
                tgt_normals_for_vis = tgt_normals
                # L1 difference per pixel
                if normal_weight > 0:
                    normal_diff = torch.abs(pred_normals - tgt_normals).mean(dim=-1)  # (K_tgt, H, W)
                    total_normal_loss = total_normal_loss + (weight_map * normal_diff).sum() / weight_denom
                    num_terms_normal += 1

            # RGB rendering across all targets
            tgt_rgb = rgb_images[b, tgt_idx].permute(0, 2, 3, 1).contiguous()  # (K_tgt,H,W,3)
            pred_rgb = None
            if cols_rgb_b is not None and rgb_weight > 0:
                cols_rgb_cam = cols_rgb_b.unsqueeze(0).expand(viewmats_b.shape[0], N_b, cols_rgb_b.shape[-1])
                render_cols, _, _, _, _, _, _ = gsplat.rasterization_2dgs(
                    means=means_b,
                    quats=quats_b,
                    scales=scales_b,
                    opacities=opa_b,
                    colors=cols_rgb_cam,
                    viewmats=viewmats_b,
                    Ks=Ks_b,
                    width=W,
                    height=H,
                    packed=False,
                    eps2d=eps2d,
                    tile_size=16,
                    render_mode='RGB',
                )
                pred_rgb = render_cols  # (K_tgt, H, W, 3)
                rgb_diff = torch.abs(pred_rgb - tgt_rgb).mean(dim=-1)  # (K_tgt, H, W)
                total_rgb_loss = total_rgb_loss + (weight_map * rgb_diff).sum() / weight_denom # TODO: use confidence-aware loss
                num_terms_rgb += 1

            # Depth rendering across all targets (includes depth gradient matching)
            if depth_weight > 0:
                # Reuse the rendered depth (pred_depth) from the ED pass above
                # Compare with target depth maps using L1 difference
                depth_diff = torch.abs(pred_depth - tgt_depth)  # (K_tgt, H, W)

                # Add depth gradient matching
                pred_grad_y, pred_grad_x = torch.gradient(pred_depth, dim=[1, 2])
                tgt_grad_y, tgt_grad_x = torch.gradient(tgt_depth, dim=[1, 2])
                grad_diff = torch.abs(pred_grad_x - tgt_grad_x) + torch.abs(pred_grad_y - tgt_grad_y)

                # Combine depth value and gradient differences
                combined_depth_diff = depth_diff + grad_diff
                total_depth_loss = total_depth_loss + (weight_map * combined_depth_diff).sum() / weight_denom
                num_terms_depth += 1

            # Log a limited number of visualizations to wandb
            if (
                wandb_module is not None
                and meta_info is not None
            ):
                views_to_log = tgt_idx.shape[0]

                def stack_views(array: np.ndarray, repeat_channels: bool = False, apply_cmap: bool = False, global_min: float = None, global_max: float = None) -> np.ndarray:
                    slices = []
                    for i in range(views_to_log):
                        view = array[i]
                        if not np.issubdtype(view.dtype, np.floating):
                            view = view.astype(np.float32)
                        v_min = view.min() if global_min is None else global_min
                        v_max = view.max() if global_max is None else global_max
                        if v_min < 0 or v_max > 1:
                            if v_max > v_min:
                                view = (view - v_min) / (v_max - v_min)
                            else:
                                view = np.zeros_like(view, dtype=view.dtype)
                        if apply_cmap and view.ndim == 2:
                            colored = cm.get_cmap('viridis')(view)
                            view = colored[..., :3]
                        elif view.ndim == 2 or repeat_channels:
                            view = np.repeat(view[..., None], 3, axis=-1)
                        slices.append(view)
                    return np.concatenate(slices, axis=1)

                tgt_vis = stack_views(tgt_rgb.detach().clamp(0.0, 1.0).float().cpu().numpy())
                pred_vis = stack_views(pred_rgb.detach().clamp(0.0, 1.0).float().cpu().numpy()) if pred_rgb is not None else None
                depth_global_min = None
                depth_global_max = None
                if pred_depth is not None:
                    depth_global_min = pred_depth.detach().min().item()
                    depth_global_max = pred_depth.detach().max().item()
                if tgt_depth is not None:
                    tgt_min = tgt_depth.detach().min().item()
                    tgt_max = tgt_depth.detach().max().item()
                    if depth_global_min is None or tgt_min < depth_global_min:
                        depth_global_min = tgt_min
                    if depth_global_max is None or tgt_max > depth_global_max:
                        depth_global_max = tgt_max

                pred_depth_vis = stack_views(pred_depth.detach().float().cpu().numpy(), apply_cmap=True, global_min=depth_global_min, global_max=depth_global_max) if pred_depth is not None else None
                gt_depth_vis = stack_views(tgt_depth.detach().float().cpu().numpy(), apply_cmap=True, global_min=depth_global_min, global_max=depth_global_max)
                if real_depth_gt is not None:
                    real_depth_gt_vis = stack_views(real_tgt_depth.detach().float().cpu().numpy(), apply_cmap=True, global_min=depth_global_min, global_max=depth_global_max)
                weight_vis = stack_views(weight_map.detach().clamp(0.0, 1.0).float().cpu().numpy(), repeat_channels=True)
                if pred_normals is not None:
                    pred_normals_img = (pred_normals.clamp(-1.0, 1.0) * 0.5 + 0.5).detach().float().cpu().numpy()
                    pred_normals_vis = stack_views(pred_normals_img, apply_cmap=False, global_min=0.0, global_max=1.0)
                if tgt_normals_for_vis is not None:
                    gt_normals_img = (tgt_normals_for_vis.clamp(-1.0, 1.0) * 0.5 + 0.5).detach().float().cpu().numpy()
                    gt_normals_vis = stack_views(gt_normals_img, apply_cmap=False, global_min=0.0, global_max=1.0)

                vis_list = [weight_vis, tgt_vis]
                if pred_rgb is not None:
                    vis_list.append(pred_vis)
                if gt_depth_vis is not None:
                    vis_list.append(gt_depth_vis)
                if real_depth_gt is not None:
                    vis_list.append(real_depth_gt_vis)
                if pred_depth is not None:
                    vis_list.append(pred_depth_vis)
                if tgt_normals_for_vis is not None:
                    vis_list.append(gt_normals_vis)
                if pred_normals is not None:
                    vis_list.append(pred_normals_vis)

                grid_vis = np.concatenate(vis_list, axis=0)

                sample_idx = meta_info.get('sample_idx', None)
                scene_name = meta_info.get('scene_name', None)
                step_value = meta_info.get('tco_step', None)
                assert sample_idx is not None and scene_name is not None and step_value is not None, \
                    "sample_idx, scene_name and step_value must be provided"

                log_payload = {
                    f"depth_mv_consistency/vis_target_render_idx{sample_idx:03d}_scene{scene_name}_step{step_value:03d}_group{group_idx:02d}": wandb_module.Image(grid_vis)
                }

                wandb_module.log(log_payload, commit=False)

    # Normalize aggregated losses
    loss = torch.zeros((), device=device, dtype=dtype)
    if num_terms_rgb > 0 and rgb_weight > 0:
        loss = loss + rgb_weight * (total_rgb_loss / max(1, num_terms_rgb))
    if num_terms_depth > 0 and depth_weight > 0:
        loss = loss + depth_weight * (total_depth_loss / max(1, num_terms_depth))
    if num_terms_normal > 0 and normal_weight > 0:
        loss = loss + normal_weight * (total_normal_loss / max(1, num_terms_normal))
    return loss