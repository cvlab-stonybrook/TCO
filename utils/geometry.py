import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def se3_inverse(T):
    """
    Computes the inverse of a batch of SE(3) matrices.
    T: Tensor of shape (B, 4, 4)
    """
    if len(T.shape) == 2:
        T = T[None]
        unseq_flag = True
    else:
        unseq_flag = False

    if torch.is_tensor(T):
        R = T[:, :3, :3]
        t = T[:, :3, 3].unsqueeze(-1)
        R_inv = R.transpose(-2, -1)
        t_inv = -torch.matmul(R_inv, t)
        T_inv = torch.cat([
            torch.cat([R_inv, t_inv], dim=-1),
            torch.tensor([0, 0, 0, 1], device=T.device, dtype=T.dtype).repeat(T.shape[0], 1, 1)
        ], dim=1)
    else:
        R = T[:, :3, :3]
        t = T[:, :3, 3, np.newaxis]

        R_inv = np.swapaxes(R, -2, -1)
        t_inv = -R_inv @ t

        bottom_row = np.zeros((T.shape[0], 1, 4), dtype=T.dtype)
        bottom_row[:, :, 3] = 1

        top_part = np.concatenate([R_inv, t_inv], axis=-1)
        T_inv = np.concatenate([top_part, bottom_row], axis=1)

    if unseq_flag:
        T_inv = T_inv[0]
    return T_inv

def get_pixel(H, W):
    # get 2D pixels (u, v) for image_a in cam_a pixel space
    u_a, v_a = np.meshgrid(np.arange(W), np.arange(H))
    # u_a = np.flip(u_a, axis=1)
    # v_a = np.flip(v_a, axis=0)
    pixels_a = np.stack([
        u_a.flatten() + 0.5, 
        v_a.flatten() + 0.5, 
        np.ones_like(u_a.flatten())
    ], axis=0)
    
    return pixels_a

def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, z_far=0, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)
    if z_far > 0:
        valid_mask = valid_mask & (depthmap < z_far)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    # assert camera_intrinsics[0, 1] == 0.0
    # assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    # Invalid any depth > 80m
    valid_mask = valid_mask
    return X_cam, valid_mask

def unproject_depth_to_points(depth: torch.Tensor, 
                             intrinsics: torch.Tensor, 
                             min_depth: float = 1e-3) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Unproject depth maps to 3D points in camera coordinate system.
    
    Args:
        depth: (B, N, H, W) - depth values
        intrinsics: (B, N, 3, 3) - camera intrinsics
        min_depth: minimum valid depth to avoid numerical issues
        
    Returns:
        points: (B, N, H, W, 3) - 3D points in camera coordinates
        valid_mask: (B, N, H, W) - mask for valid points (optional)
    """
    # Handle depth with extra channel dimension (e.g., from depth head)
    if depth.dim() == 5:
        depth = depth.squeeze(-1)  # Remove channel dimension: (B, N, H, W, 1) -> (B, N, H, W)
    
    B, N, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype
    
    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    # Expand to match batch and view dimensions
    x = x[None, None, :, :].expand(B, N, H, W)  # (B, N, H, W)
    y = y[None, None, :, :].expand(B, N, H, W)  # (B, N, H, W)
    
    # Extract intrinsics
    fx = intrinsics[..., 0, 0]  # (B, N)
    fy = intrinsics[..., 1, 1]  # (B, N)
    cx = intrinsics[..., 0, 2]  # (B, N)
    cy = intrinsics[..., 1, 2]  # (B, N)
    
    # Expand intrinsics to match spatial dimensions
    fx = fx[..., None, None]  # (B, N, 1, 1)
    fy = fy[..., None, None]  # (B, N, 1, 1)
    cx = cx[..., None, None]  # (B, N, 1, 1)
    cy = cy[..., None, None]  # (B, N, 1, 1)
    
    # Compute 3D points
    z = depth
    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy
    
    # Stack to get points
    points = torch.stack([x_3d, y_3d, z], dim=-1)  # (B, N, H, W, 3)
    
    # Create valid mask based on depth and confidence
    valid_mask = z > min_depth  # (B, N, H, W)
    
    return points, valid_mask


def homogenize_points(
    points,
):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def to_homogeneous(T: torch.Tensor) -> torch.Tensor:
    """
    Ensure pose matrices are homogeneous (..., 4, 4). Accepts (..., 3, 4) or (..., 4, 4).
    """
    if T.shape[-2:] == (4, 4):
        return T
    if T.shape[-2:] == (3, 4):
        pad = T.new_zeros((*T.shape[:-2], 1, 4))
        pad[..., 0, 3] = 1.0
        return torch.cat([T, pad], dim=-2)
    raise AssertionError(f"Pose tensor must be (...,3,4) or (...,4,4), got {T.shape[-2:]}.")


def get_gt_warp(depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode = 'bilinear', relative_depth_error_threshold = 0.05, H = None, W = None):
    
    if H is None:
        B,H,W = depth1.shape
    else:
        B = depth1.shape[0]
    with torch.no_grad():
        x1_n = torch.meshgrid(
            *[
                torch.linspace(
                    -1 + 1 / n, 1 - 1 / n, n, device=depth1.device
                )
                for n in (B, H, W)
            ],
            indexing = 'ij'
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        mask, x2 = warp_kpts(
            x1_n.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
            depth_interpolation_mode = depth_interpolation_mode,
            relative_depth_error_threshold = relative_depth_error_threshold,
        )
        prob = mask.float().reshape(B, H, W)
        x2 = x2.reshape(B, H, W, 2)
        return x2, prob

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, smooth_mask = False, return_relative_depth_error = False, depth_interpolation_mode = "bilinear", relative_depth_error_threshold = 0.05):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    # https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/src/loftr/utils/geometry.py adapted from here
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>, should be normalized in (-1,1)
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    (
        n,
        h,
        w,
    ) = depth0.shape
    if depth_interpolation_mode == "combined":
        # Inspired by approach in inloc, try to fill holes from bilinear interpolation by nearest neighbour interpolation
        if smooth_mask:
            raise NotImplementedError("Combined bilinear and NN warp not implemented")
        valid_bilinear, warp_bilinear = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "bilinear",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        valid_nearest, warp_nearest = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "nearest-exact",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        nearest_valid_bilinear_invalid = (~valid_bilinear).logical_and(valid_nearest) 
        warp = warp_bilinear.clone()
        warp[nearest_valid_bilinear_invalid] = warp_nearest[nearest_valid_bilinear_invalid]
        valid = valid_bilinear | valid_nearest
        return valid, warp
        
        
    kpts0_depth = F.grid_sample(depth0[:, None], kpts0[:, :, None], mode = depth_interpolation_mode, align_corners=False)[
        :, 0, :, 0
    ]
    kpts0 = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    # Sample depth, get calculable_mask on depth != 0
    # nonzero_mask = kpts0_depth != 0
    # Sample depth, get calculable_mask on depth > 0
    nonzero_mask = kpts0_depth > 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
    kpts0_cam = kpts0_n

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0 = torch.stack(
        (2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1
    )  # from [0.5,h-0.5] -> [-1+1/h, 1-1/h]
    # w_kpts0[~covisible_mask, :] = -5 # xd

    w_kpts0_depth = F.grid_sample(
        depth1[:, None], w_kpts0[:, :, None], mode=depth_interpolation_mode, align_corners=False
    )[:, 0, :, 0]
    
    relative_depth_error = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs()
    if not smooth_mask:
        consistent_mask = relative_depth_error < relative_depth_error_threshold
    else:
        consistent_mask = (-relative_depth_error/smooth_mask).exp()
    valid_mask = nonzero_mask * covisible_mask * consistent_mask
    if return_relative_depth_error:
        return relative_depth_error, w_kpts0
    else:
        return valid_mask, w_kpts0


def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def opencv_camera_to_plucker(poses, K, H, W):
    device = poses.device
    B = poses.shape[0]

    pixel = torch.from_numpy(get_pixel(H, W).astype(np.float32)).to(device).T.reshape(H, W, 3)[None].repeat(B, 1, 1, 1)         # (3, H, W)
    pixel = torch.einsum('bij, bhwj -> bhwi', torch.inverse(K), pixel)
    ray_directions = torch.einsum('bij, bhwj -> bhwi', poses[..., :3, :3], pixel)

    ray_origins = poses[..., :3, 3][:, None, None].repeat(1, H, W, 1)

    ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    plucker_ray = torch.cat([ray_directions, plucker_normal], dim=-1)

    return plucker_ray


def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None) -> torch.BoolTensor:
    """
    Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.
    
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge

def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def unproject_depth_to_world_points_torch(
    depth_map: torch.Tensor, 
    extrinsics: torch.Tensor, 
    intrinsics: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Unproject depth maps to 3D world coordinates using PyTorch operations.
    
    Args:
        depth_map (torch.Tensor): Depth maps of shape (B, S, H, W) or (S, H, W) or (B, S, H, W, 1)
        extrinsics (torch.Tensor): Camera extrinsic matrices of shape (B, S, 3, 4) or (S, 3, 4) or (B, S, 4, 4)
        intrinsics (torch.Tensor): Camera intrinsic matrices of shape (B, S, 3, 3) or (S, 3, 3)
        eps (float): Minimum depth threshold for valid points
        
    Returns:
        torch.Tensor: World coordinates of shape (B, S, H, W, 3) or (S, H, W, 3)
    """
    # Handle input shapes
    original_shape = depth_map.shape
    if len(depth_map.shape) == 3:  # (S, H, W)
        depth_map = depth_map.unsqueeze(0)  # (1, S, H, W)
        extrinsics = extrinsics.unsqueeze(0)  # (1, S, 3, 4) or (1, S, 4, 4)
        intrinsics = intrinsics.unsqueeze(0)  # (1, S, 3, 3)
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    # Handle depth map with trailing dimension
    if len(depth_map.shape) == 5:  # (B, S, H, W, 1)
        depth_map = depth_map.squeeze(-1)  # (B, S, H, W)
    
    B, S, H, W = depth_map.shape
    device = depth_map.device
    
    # Handle extrinsics: convert (B, S, 4, 4) to (B, S, 3, 4) if needed
    if extrinsics.shape[-1] == 4 and extrinsics.shape[-2] == 4:
        extrinsics = extrinsics[..., :3, :]  # (B, S, 3, 4)
    
    # Create pixel coordinate grids
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # Reshape depth and pixel coordinates for batch processing
    depth_flat = depth_map.reshape(B * S, H, W)  # (B*S, H, W)
    u_coords = u_coords.expand(B * S, -1, -1)    # (B*S, H, W)
    v_coords = v_coords.expand(B * S, -1, -1)    # (B*S, H, W)
    
    # Extract intrinsic parameters
    intrinsics_flat = intrinsics.reshape(B * S, 3, 3)  # (B*S, 3, 3)
    fx = intrinsics_flat[:, 0, 0]  # (B*S,)
    fy = intrinsics_flat[:, 1, 1]  # (B*S,)
    cx = intrinsics_flat[:, 0, 2]  # (B*S,)
    cy = intrinsics_flat[:, 1, 2]  # (B*S,)
    
    # Unproject to camera coordinates
    x_cam = (u_coords - cx.view(-1, 1, 1)) * depth_flat / fx.view(-1, 1, 1)
    y_cam = (v_coords - cy.view(-1, 1, 1)) * depth_flat / fy.view(-1, 1, 1)
    z_cam = depth_flat
    
    # Stack camera coordinates (B*S, H, W, 3)
    cam_coords = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    
    # Reshape for transformation: (B*S, H*W, 3)
    cam_coords_flat = cam_coords.reshape(B * S, H * W, 3)
    
    # Get extrinsics and compute camera-to-world transformation
    extrinsics_flat = extrinsics.reshape(B * S, 3, 4)  # (B*S, 3, 4)
    
    # Extract rotation and translation from extrinsics (world-to-camera)
    R_w2c = extrinsics_flat[:, :3, :3]  # (B*S, 3, 3)
    t_w2c = extrinsics_flat[:, :3, 3]   # (B*S, 3)
    
    # Compute camera-to-world transformation (inverse of world-to-camera)
    # R_c2w = R_w2c^T, t_c2w = -R_c2w @ t_w2c
    R_c2w = R_w2c.transpose(-1, -2)  # (B*S, 3, 3)
    t_c2w = -torch.bmm(R_c2w, t_w2c.unsqueeze(-1)).squeeze(-1)  # (B*S, 3)
    
    # Transform camera coordinates to world coordinates
    # world_coords = R_c2w @ cam_coords + t_c2w
    world_coords_flat = torch.bmm(cam_coords_flat, R_c2w.transpose(-1, -2)) + t_c2w.unsqueeze(1)
    
    # Reshape back to spatial dimensions
    world_coords = world_coords_flat.reshape(B, S, H, W, 3)
    
    # Remove batch dimension if it was added
    if squeeze_batch:
        world_coords = world_coords.squeeze(0)  # (S, H, W, 3)
    
    return world_coords


def extrinsics_to_se3(extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Convert extrinsics matrix (3x4) to SE3 matrix (4x4).
    
    Args:
        extrinsics: (..., 3, 4) extrinsics matrices [R|t]
    
    Returns:
        se3: (..., 4, 4) SE3 transformation matrices
    """
    # Get the batch dimensions
    batch_shape = extrinsics.shape[:-2]
    device = extrinsics.device
    dtype = extrinsics.dtype
    
    # Create SE3 matrices with proper batch dimensions
    se3_matrices = torch.zeros(*batch_shape, 4, 4, device=device, dtype=dtype)
    se3_matrices[..., :3, :4] = extrinsics
    se3_matrices[..., 3, 3] = 1.0
    
    return se3_matrices

def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a depth map to world coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).
        extrinsic (np.ndarray): Camera extrinsic matrix of shape (3, 4). OpenCV camera coordinate convention, cam from world.

    Returns:
        tuple[np.ndarray, np.ndarray]: World coordinates (H, W, 3) and valid depth mask (H, W).
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # Multiply with the inverse of extrinsic matrix to transform to world coordinates
    # extrinsic_inv is 4x4 (note closed_form_inverse_OpenCV is batched, the output is (N, 4, 4))
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask

def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.
    Quaternion Order: XYZW (scalar-last)
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

# ----------------------
# VGGT Pose Encoding Utilities (from vggt/utils)
# ----------------------
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) but with a zero subgradient where x is 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert a unit quaternion to a standard form: one in which the real part is non negative."""
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as rotation matrices to quaternions.
    Returns quaternions with real part last (XYZW order).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    out = out[..., [1, 2, 3, 0]]  # Convert from rijk to ijkr
    out = standardize_quaternion(out)
    return out

def pose_encoding_to_extri_intri(
    pose_encoding, image_size_hw=None, pose_encoding_type="absT_quaR_FoV", build_intrinsics=True
):
    """Convert a pose encoding back to camera extrinsics and intrinsics.
    
    Args:
        pose_encoding: Encoded camera pose (BxSx9) with T(3D) + quat(4D) + FoV(2D)
        image_size_hw: (H, W) tuple for image size
        pose_encoding_type: Type of encoding (only "absT_quaR_FoV" supported)
        build_intrinsics: Whether to build intrinsics matrix
    
    Returns:
        extrinsics: BxSx3x4 camera extrinsics [R|t]
        intrinsics: BxSx3x3 camera intrinsics or None
    """
    intrinsics = None

    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)

        if build_intrinsics and image_size_hw is not None:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = W / 2
            intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0
    else:
        raise NotImplementedError(f"Pose encoding type {pose_encoding_type} not supported")

    return extrinsics, intrinsics

def invert_se3(T: torch.Tensor) -> torch.Tensor:
    """
    Invert batched SE(3) transforms.
    Args: T (..., 4, 4) with bottom row [0,0,0,1]
    Returns: T_inv with same shape
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_t = R.transpose(-1, -2)
    t_inv = -(R_t @ t.unsqueeze(-1)).squeeze(-1)
    T_inv = T.clone()
    T_inv[..., :3, :3] = R_t
    T_inv[..., :3, 3] = t_inv
    return T_inv

def sample_view_pairs(num_views: int, num_pairs: int = None, device: torch.device = "cuda") -> torch.Tensor:
    """Sample unique unordered pairs of view indices (i < j). Returns (K, 2)."""
    # Build all unique unordered pairs once
    pairs = [(i, j) for i in range(num_views) for j in range(i + 1, num_views)]
    if len(pairs) == 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    all_pairs = torch.tensor(pairs, device=device, dtype=torch.long)
    total = all_pairs.shape[0]
    if (num_pairs is None) or (num_pairs >= total):
        return all_pairs
    if num_pairs <= 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    # random sampling without replacement over all pairs
    perm = torch.randperm(total, device=device)
    return all_pairs[perm[:num_pairs]]

def sample_view_triplets(num_views: int, num_triplets: int = None, device: torch.device = "cuda") -> torch.Tensor:
    """Sample unique triplets of view indices (i < j < k). Returns (K, 3)."""
    # Build all unique triplets once
    triplets = [(i, j, k) for i in range(num_views) 
                for j in range(i + 1, num_views) 
                for k in range(j + 1, num_views)]
    if len(triplets) == 0:
        return torch.empty((0, 3), device=device, dtype=torch.long)
    all_triplets = torch.tensor(triplets, device=device, dtype=torch.long)
    total = all_triplets.shape[0]
    if (num_triplets is None) or (num_triplets >= total):
        return all_triplets
    if num_triplets <= 0:
        return torch.empty((0, 3), device=device, dtype=torch.long)
    # random sampling without replacement over all triplets
    perm = torch.randperm(total, device=device)
    return all_triplets[perm[:num_triplets]]

def triangle_similarity_loss(camera_poses_pred: torch.Tensor,
                           camera_poses_gt: torch.Tensor,
                           triplets: torch.Tensor) -> torch.Tensor:
    """
    Compute translation loss using triangle similarity between camera center triplets.
    This approach is coordinate frame invariant and scale-invariant.
    
    Args:
        camera_poses_pred: (B, N, 4, 4) predicted camera poses
        camera_poses_gt: (B, N, 4, 4) ground truth camera poses  
        triplets: (K, 3) triplet indices
        
    Returns:
        Triangle similarity loss comparing normalized side length ratios
    """
    K = triplets.shape[0]
    if K == 0:
        return torch.zeros((), device=camera_poses_pred.device, dtype=camera_poses_pred.dtype)
    
    # Extract camera centers (translation components)
    centers_pred = camera_poses_pred[..., :3, 3]  # (B, N, 3)
    centers_gt = camera_poses_gt[..., :3, 3]      # (B, N, 3)
    
    # Gather triplet centers
    i_idx, j_idx, k_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    
    # Predicted triangle vertices
    ci_pred = centers_pred[:, i_idx]  # (B, K, 3)
    cj_pred = centers_pred[:, j_idx]  # (B, K, 3)
    ck_pred = centers_pred[:, k_idx]  # (B, K, 3)
    
    # Ground truth triangle vertices
    ci_gt = centers_gt[:, i_idx]      # (B, K, 3)
    cj_gt = centers_gt[:, j_idx]      # (B, K, 3)
    ck_gt = centers_gt[:, k_idx]      # (B, K, 3)
    
    # Compute triangle side lengths
    side1_pred = torch.norm(ci_pred - cj_pred, dim=-1)  # (B, K)
    side2_pred = torch.norm(cj_pred - ck_pred, dim=-1)  # (B, K)
    side3_pred = torch.norm(ci_pred - ck_pred, dim=-1)  # (B, K)
    
    side1_gt = torch.norm(ci_gt - cj_gt, dim=-1)      # (B, K)
    side2_gt = torch.norm(cj_gt - ck_gt, dim=-1)      # (B, K)
    side3_gt = torch.norm(ci_gt - ck_gt, dim=-1)      # (B, K)
    
    # Normalize by perimeter for scale invariance
    perimeter_pred = side1_pred + side2_pred + side3_pred + 1e-8
    perimeter_gt = side1_gt + side2_gt + side3_gt + 1e-8
    
    # Normalized side ratios (scale-invariant triangle shape)
    ratio1_pred = side1_pred / perimeter_pred
    ratio2_pred = side2_pred / perimeter_pred  
    ratio3_pred = side3_pred / perimeter_pred
    
    ratio1_gt = side1_gt / perimeter_gt
    ratio2_gt = side2_gt / perimeter_gt
    ratio3_gt = side3_gt / perimeter_gt
    
    # Compare triangle shapes using normalized ratios
    loss1 = F.smooth_l1_loss(ratio1_pred, ratio1_gt, reduction='mean')
    loss2 = F.smooth_l1_loss(ratio2_pred, ratio2_gt, reduction='mean')
    loss3 = F.smooth_l1_loss(ratio3_pred, ratio3_gt, reduction='mean')
    
    return loss1 + loss2 + loss3
