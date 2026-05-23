import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf
import time

from typing import List, Optional, Tuple
from omegaconf import DictConfig
from PIL import Image

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# from base_models.pi3.pi3 import Pi3
from .geometry import se3_inverse

def load_images(filelist: List[str], PIXEL_LIMIT: int = 255000, new_width: Optional[int] = None, verbose: bool = False):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [] 
    
    # --- 1. Load image paths or video frames ---
    for img_path in filelist:
        try:
            # sources.append(Image.open(img_path).convert('RGB'))
            image = Image.open(img_path)
            sources.append(image)
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    if verbose:
        print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    if new_width is None:
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    else:
        TARGET_W, TARGET_H = new_width, round(H_orig * (new_width / W_orig) / 14) * 14
    if verbose:
        print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = tvf.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)


def load_and_resize14(filelist: List[str], new_width: int, device: str, verbose: bool):
    imgs = load_images(filelist, new_width=new_width, verbose=verbose).to(device)
    
    # Convert to float32 to ensure compatibility with interpolation
    if imgs.dtype != torch.float32:
        imgs = imgs.float()

    ori_h, ori_w = imgs.shape[-2:]
    patch_h, patch_w = ori_h // 14, ori_w // 14
    # (N, 3, h, w) -> (1, N, 3, h_14, w_14)
    imgs = F.interpolate(imgs, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=False, antialias=True).unsqueeze(0)
    return imgs

def infer_monodepth(file: str, model: nn.Module, hydra_cfg: DictConfig):

    imgs = load_and_resize14([file], new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    points = pred['local_points'][0]         # (1, h_14, w_14, 3)
    depth_map = points[0, ..., -1].detach()  # (h_14, w_14)
    return depth_map  # torch.Tensor


def infer_videodepth(filelist: str, model: nn.Module, hydra_cfg: DictConfig):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)
    end = time.time()

    depth_map = pred['local_points'][0, ..., -1]  # (N, h_14, w_14)
    depth_conf = pred['conf'][0, ..., 0]          # (N, h_14, w_14)
    return end - start, depth_map, depth_conf


def infer_cameras_w2c(filelist: str, model: nn.Module, hydra_cfg: DictConfig):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    poses_c2w_all = pred['camera_poses'].cpu()
    extrinsics = se3_inverse(poses_c2w_all[0])

    return extrinsics, None


def infer_cameras_c2w(filelist: str, model: nn.Module, hydra_cfg: DictConfig):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    poses_c2w_all = pred['camera_poses'].cpu()

    return poses_c2w_all[0], None

def infer_cameras_c2w_w_priors(filelist: str, depth_filelist: str, model: nn.Module, hydra_cfg: DictConfig, **kwargs):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)
    depths = load_and_resize14(depth_filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)
    depths = depths / hydra_cfg.depth_scale
    depth_masks = ((depths > hydra_cfg.min_valid_depth) & (depths < hydra_cfg.max_valid_depth)).float()

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
        pred = model(imgs, depths=depths, depth_masks=depth_masks,
                     gt_extrinsics=None,
                     gt_intrinsics=None,
                     # ----- TCO parameters -----
                     tco_steps=hydra_cfg.tco_steps,
                     tco_lr=hydra_cfg.tco_lr,
                     # ----- depth energy parameters -----
                     lambda_depth=hydra_cfg.lambda_depth,
                     depth_alignment_mode=hydra_cfg.depth_alignment_mode,
                     depth_confidence_percentile=hydra_cfg.depth_confidence_percentile,
                     depth_scale_reg_weight=hydra_cfg.depth_scale_reg_weight,
                     depth_shift_reg_weight=hydra_cfg.depth_shift_reg_weight,
                     # ----- multiview consistency energy parameters -----
                     lambda_mv_consistency=hydra_cfg.lambda_mv_consistency,
                     mv_depth_weight=hydra_cfg.mv_depth_weight,
                     mv_normal_weight=hydra_cfg.mv_normal_weight,
                     mv_rgb_weight=hydra_cfg.mv_rgb_weight,
                     num_view_groups=hydra_cfg.num_view_groups,
                     use_cos_weighting=hydra_cfg.use_cos_weighting,
                     # ----- redundancy consistency energy parameters -----
                     lambda_redundancy=hydra_cfg.lambda_redundancy,
                     # ----- LoRA parameters -----
                     lora_rank=hydra_cfg.lora_rank,
                     lora_alpha=hydra_cfg.lora_alpha,
                     lora_target_modules=hydra_cfg.lora_target_modules,
                     lora_dropout=hydra_cfg.lora_dropout,
                     **kwargs)

    poses_c2w_all = pred['camera_poses'].cpu()

    return poses_c2w_all[0], None

def infer_mv_pointclouds(filelist: str, model: nn.Module, hydra_cfg: DictConfig, data_size: Tuple[int, int]):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)
    
    global_points = pred['points'][0]  # (N, h, w, 3)
    global_points = F.interpolate(
        global_points.permute(0, 3, 1, 2), data_size,
        mode="bilinear", align_corners=False, antialias=True
    ).permute(0, 2, 3, 1)  # align to gt

    return global_points.cpu().numpy()

def infer_mv_pointclouds_no_priors(data: dict, model: nn.Module, hydra_cfg: DictConfig, data_size: Tuple[int, int], output_key: str):
    imgs = data['images'].to(hydra_cfg.device).unsqueeze(0)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    global_points = pred[output_key][0]  # (N, h, w, 3)
    return global_points.cpu().numpy()

def infer_mv_pointclouds_w_priors(data: dict, model: nn.Module, hydra_cfg: DictConfig, **kwargs):
    imgs = data['images'].to(hydra_cfg.device).unsqueeze(0)
    intrinsics = torch.from_numpy(data['intrs']).to(hydra_cfg.device).unsqueeze(0)
    extrinsics = torch.from_numpy(data['extrs']).to(hydra_cfg.device).unsqueeze(0)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Enable gradients for TTT steps
    with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
        pred = model(# ----- Inputs -----
                     imgs,
                     gt_extrinsics=extrinsics,
                     gt_intrinsics=intrinsics,
                     # ----- TTT parameters -----
                     ttt_steps=20,
                     ttt_lr=1e-3,
                     # ----- camera energy parameters -----
                     lambda_pose=1,
                     pose_translation_weight=1,
                     lambda_intrinsics=0.01,
                     ttt_num_pose_pairs=None,
                     pose_rot_loss_type='angle',
                     pose_trans_loss_type='normed_l1',
                     # ----- multiview consistency energy parameters -----
                     lambda_mv_consistency=1,
                     mv_rgb_weight=1,
                     mv_src_view_ratio=0.5,
                     # ----- LoRA parameters -----
                     lora_rank=4,
                     lora_alpha=16,
                     lora_target_modules=['qkv', 'proj', 'fc1', 'fc2'],
                     lora_dropout=0.,
                     **kwargs)

    global_points = pred['points'][0]  # (N, h, w, 3)

    return global_points.cpu().numpy()

def infer_mv_pointclouds_w_priors_v2(data: dict, model: nn.Module, hydra_cfg: DictConfig, **kwargs):
    imgs = data['images'].to(hydra_cfg.device).unsqueeze(0)
    intrinsics = torch.from_numpy(data['intrs']).to(hydra_cfg.device).unsqueeze(0)
    extrinsics = torch.from_numpy(data['extrs']).to(hydra_cfg.device).unsqueeze(0)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Enable gradients for TTT steps
    with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
        pred = model(# ----- Inputs -----
                    imgs,
                    gt_extrinsics=extrinsics,
                    gt_intrinsics=intrinsics,
                    # ----- TCO parameters -----
                    tco_steps=hydra_cfg.tco_steps,
                    tco_lr=hydra_cfg.tco_lr,
                    # ----- camera energy parameters -----
                    lambda_pose=hydra_cfg.lambda_pose,
                    pose_translation_weight=hydra_cfg.pose_translation_weight,
                    enforce_canonical_reference=hydra_cfg.enforce_canonical_reference,
                    lambda_intrinsics=hydra_cfg.lambda_intrinsics,
                    tco_num_pose_pairs=hydra_cfg.tco_num_pose_pairs,
                    pose_rot_loss_type=hydra_cfg.pose_rot_loss_type,
                    pose_trans_loss_type=hydra_cfg.pose_trans_loss_type,
                    # ----- depth energy parameters -----
                    lambda_depth=0,
                    # ----- multiview consistency energy parameters -----
                    lambda_mv_consistency=hydra_cfg.lambda_mv_consistency,
                    mv_depth_weight=hydra_cfg.mv_depth_weight,
                    mv_normal_weight=hydra_cfg.mv_normal_weight,
                    mv_rgb_weight=hydra_cfg.mv_rgb_weight,
                    num_view_groups=hydra_cfg.num_view_groups,
                    use_cos_weighting=hydra_cfg.use_cos_weighting,
                    # ----- redundancy consistency energy parameters -----
                    lambda_redundancy=hydra_cfg.lambda_redundancy,
                    # ----- LoRA parameters -----
                    lora_rank=hydra_cfg.lora_rank,
                    lora_alpha=hydra_cfg.lora_alpha,
                    lora_target_modules=hydra_cfg.lora_target_modules,
                    lora_dropout=hydra_cfg.lora_dropout,
                    **kwargs)

    global_points = pred['points'][0]  # (N, h, w, 3)

    return global_points.cpu().numpy()