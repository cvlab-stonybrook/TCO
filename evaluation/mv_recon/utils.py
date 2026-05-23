# Reference: https://github.com/CUT3R/CUT3R/blob/main/eval/mv_recon/utils.py

import numpy as np
from scipy.spatial import cKDTree as KDTree
from typing import Tuple, Optional


def umeyama(X, Y):
    """
    Estimates the Sim(3) transformation between `X` and `Y` point sets.

    Estimates c, R and t such as c * R @ X + t ~ Y.

    Parameters
    ----------
    X : numpy.array
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the point set.
    Y : numpy.array
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].

    Returns
    -------
    c : float
        Scale factor.
    R : numpy.array
        (3, 3) shaped rotation matrix.
    t : numpy.array
        (3, 1) shaped translation vector.
    """
    mu_x = X.mean(axis=1).reshape(-1, 1)
    mu_y = Y.mean(axis=1).reshape(-1, 1)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t


def robust_umeyama(X: np.ndarray, Y: np.ndarray, 
                   max_iterations: int = 1000,
                   inlier_threshold: Optional[float] = None,
                   min_inlier_ratio: float = 0.5,
                   adaptive_threshold: bool = True,
                   confidence: float = 0.99,
                   min_sample_size: Optional[int] = None,
                   verbose: bool = False) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Robust estimation of Sim(3) transformation using RANSAC with Umeyama algorithm.
    
    This function is more robust to outliers and noisy correspondences compared to
    the standard Umeyama algorithm.
    
    Parameters
    ----------
    X : numpy.ndarray
        (m, n) shaped numpy array. m is the dimension of the points (typically 3),
        n is the number of points in the source point set.
    Y : numpy.ndarray
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
        That is, Y[:, i] must be the point corresponding to X[:, i].
    max_iterations : int, optional
        Maximum number of RANSAC iterations (default: 1000).
    inlier_threshold : float, optional
        Distance threshold for considering a point as an inlier.
        If None, will be automatically estimated (default: None).
    min_inlier_ratio : float, optional
        Minimum ratio of inliers required for a valid model (default: 0.5).
    adaptive_threshold : bool, optional
        If True, use adaptive threshold based on MAD (default: True).
    confidence : float, optional
        Desired confidence level for RANSAC (default: 0.99).
    min_sample_size : int, optional
        Minimum number of points for computing transformation.
        If None, uses 4 for 3D points, 3 for 2D points (default: None).
    verbose : bool, optional
        If True, print debug information (default: False).
        
    Returns
    -------
    c : float
        Scale factor of the best transformation.
    R : numpy.ndarray
        (m, m) shaped rotation matrix of the best transformation.
    t : numpy.ndarray
        (m, 1) shaped translation vector of the best transformation.
    inlier_mask : numpy.ndarray
        Boolean array indicating which correspondences are inliers.
    """
    m, n = X.shape  # m: dimension, n: number of points
    
    # Set minimum sample size based on dimension if not provided
    if min_sample_size is None:
        min_sample_size = max(4, m + 1)  # At least m+1 points for m-dimensional space
    
    # Ensure we have enough points
    if n < min_sample_size:
        if verbose:
            print(f"Warning: Not enough points ({n} < {min_sample_size}), using standard Umeyama")
        c, R, t = umeyama(X, Y)
        return c, R, t, np.ones(n, dtype=bool)
    
    # Estimate initial threshold if not provided
    if inlier_threshold is None:
        # Quick estimation using a subset of points
        sample_size = min(100, n)
        indices = np.random.choice(n, sample_size, replace=False)
        c_init, R_init, t_init = umeyama(X[:, indices], Y[:, indices])
        Y_pred = c_init * R_init @ X + t_init
        errors = np.linalg.norm(Y - Y_pred, axis=0)
        
        if adaptive_threshold:
            # Use MAD (Median Absolute Deviation) for robust threshold estimation
            median_error = np.median(errors)
            mad = np.median(np.abs(errors - median_error))
            # MAD-based threshold (2.5 * MAD covers ~98% of inliers for normal distribution)
            inlier_threshold = median_error + 2.5 * 1.4826 * mad
        else:
            # Use percentile-based threshold
            inlier_threshold = np.percentile(errors, 75)
        
        if verbose:
            print(f"Estimated inlier threshold: {inlier_threshold:.6f}")
    
    best_c, best_R, best_t = None, None, None
    best_inliers = []
    best_num_inliers = 0
    
    # Calculate adaptive number of iterations based on confidence
    def update_num_iterations(num_inliers, num_total, confidence, min_sample_size):
        if num_inliers == 0:
            return max_iterations
        inlier_ratio = num_inliers / num_total
        if inlier_ratio >= 0.999:  # Almost all points are inliers
            return 1
        try:
            num_iter = np.log(1 - confidence) / np.log(1 - inlier_ratio ** min_sample_size)
            return int(min(num_iter, max_iterations))
        except:
            return max_iterations
    
    iterations_needed = max_iterations
    
    for iteration in range(max_iterations):
        # Randomly sample minimum number of points
        sample_indices = np.random.choice(n, min_sample_size, replace=False)
        X_sample = X[:, sample_indices]
        Y_sample = Y[:, sample_indices]
        
        try:
            # Compute transformation using sample
            c_candidate, R_candidate, t_candidate = umeyama(X_sample, Y_sample)
            
            # Apply transformation to all points
            Y_pred = c_candidate * R_candidate @ X + t_candidate
            
            # Compute errors
            errors = np.linalg.norm(Y - Y_pred, axis=0)
            
            # Determine inliers
            inlier_mask = errors < inlier_threshold
            num_inliers = np.sum(inlier_mask)
            
            # Update best model if this one is better
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                best_inliers = np.where(inlier_mask)[0]
                
                # Update number of iterations needed
                iterations_needed = update_num_iterations(
                    num_inliers, n, confidence, min_sample_size
                )
                
                if verbose and iteration % 100 == 0:
                    print(f"Iteration {iteration}: {num_inliers}/{n} inliers, "
                          f"iterations needed: {iterations_needed}")
            
            # Early termination if we've done enough iterations
            if iteration >= iterations_needed:
                break
                
        except np.linalg.LinAlgError:
            # Singular matrix, skip this sample
            continue
    
    # Check if we found enough inliers
    if best_num_inliers < min_inlier_ratio * n:
        if verbose:
            print(f"Warning: Only found {best_num_inliers}/{n} inliers "
                  f"(< {min_inlier_ratio * n:.0f} required)")
    
    # Recompute transformation using all inliers
    if len(best_inliers) >= min_sample_size:
        X_inliers = X[:, best_inliers]
        Y_inliers = Y[:, best_inliers]
        best_c, best_R, best_t = umeyama(X_inliers, Y_inliers)
        
        # Final inlier check with refined model
        Y_pred_final = best_c * best_R @ X + best_t
        errors_final = np.linalg.norm(Y - Y_pred_final, axis=0)
        final_inlier_mask = errors_final < inlier_threshold
        
        if verbose:
            print(f"Final model: {np.sum(final_inlier_mask)}/{n} inliers")
            print(f"Scale: {best_c:.4f}, Translation norm: {np.linalg.norm(best_t):.4f}")
    else:
        # Fallback to standard Umeyama if RANSAC fails
        if verbose:
            print("RANSAC failed, falling back to standard Umeyama")
        best_c, best_R, best_t = umeyama(X, Y)
        final_inlier_mask = np.ones(n, dtype=bool)
    
    return best_c, best_R, best_t, final_inlier_mask


def umeyama_auto(X: np.ndarray, Y: np.ndarray, 
                 robust: bool = True,
                 noise_ratio_threshold: float = 0.1,
                 **kwargs) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Automatically choose between standard and robust Umeyama based on data characteristics.
    
    This is a convenience wrapper that analyzes the input data and decides whether
    to use the standard Umeyama algorithm or the robust RANSAC-based version.
    
    Parameters
    ----------
    X : numpy.ndarray
        (m, n) shaped numpy array. m is the dimension of the points,
        n is the number of points in the source point set.
    Y : numpy.ndarray
        (m, n) shaped numpy array. Indexes should be consistent with `X`.
    robust : bool, optional
        If True, always use robust version. If False, use heuristics (default: True).
    noise_ratio_threshold : float, optional
        If estimated noise ratio exceeds this, use robust version (default: 0.1).
    **kwargs : dict
        Additional arguments passed to robust_umeyama if used.
        
    Returns
    -------
    c : float
        Scale factor.
    R : numpy.ndarray
        (m, m) shaped rotation matrix.
    t : numpy.ndarray
        (m, 1) shaped translation vector.
        
    Notes
    -----
    The function returns only c, R, t for compatibility with the original umeyama.
    If you need the inlier mask, call robust_umeyama directly.
    """
    if not robust:
        # Quick noise estimation
        c_quick, R_quick, t_quick = umeyama(X, Y)
        Y_pred = c_quick * R_quick @ X + t_quick
        errors = np.linalg.norm(Y - Y_pred, axis=0)
        
        # Use robust statistics to estimate noise
        median_error = np.median(errors)
        mad = np.median(np.abs(errors - median_error))
        threshold = median_error + 3 * 1.4826 * mad
        
        noise_ratio = np.mean(errors > threshold)
        
        if noise_ratio < noise_ratio_threshold:
            # Low noise, use standard algorithm
            return c_quick, R_quick, t_quick
    
    # Use robust version
    c, R, t, _ = robust_umeyama(X, Y, **kwargs)
    return c, R, t


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)

    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)

        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)

        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median


def compute_iou(pred_vox, target_vox):
    # Get voxel indices
    v_pred_indices = [voxel.grid_index for voxel in pred_vox.get_voxels()]
    v_target_indices = [voxel.grid_index for voxel in target_vox.get_voxels()]

    # Convert to sets for set operations
    v_pred_filled = set(tuple(np.round(x, 4)) for x in v_pred_indices)
    v_target_filled = set(tuple(np.round(x, 4)) for x in v_target_indices)

    # Compute intersection and union
    intersection = v_pred_filled & v_target_filled
    union = v_pred_filled | v_target_filled

    # Compute IoU
    iou = len(intersection) / len(union)
    return iou