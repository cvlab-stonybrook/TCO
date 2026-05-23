import sys
from pathlib import Path
# Add vggt's parent directory to path so its internal imports work
sys.path.insert(0, str(Path(__file__).parent / "base_models" / "vggt"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List, Optional, Tuple, Any

from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten
from utils.geometry import unproject_depth_to_world_points_torch, extrinsics_to_se3, pose_encoding_to_extri_intri, invert_se3
from tco_loss import compute_tco_losses

from peft.lora import LoRAWrapper, LoRAModel

class TCO_VGGT_LoRA(VGGT, LoRAModel):
    """
    TBD
    """
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=True,
        enable_depth=True,
        enable_track=False  # Disable tracking by default for TCO
    ):
        # Initialize VGGT parent class
        VGGT.__init__(
            self,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            enable_camera=enable_camera,
            enable_point=enable_point,
            enable_depth=enable_depth,
            enable_track=enable_track
        )
        
        # Initialize LoRAModel mixin
        self._init_lora_model()
        
        # Store embedding dimension for LoRA adapters
        self.embed_dim = embed_dim
    
    def extract_patch_tokens(self, images: torch.Tensor):
        """
        Extract patch tokens from images. This is done once before TTT.
        
        Args:
            images: Input images [B, S, 3, H, W]
        
        Returns:
            patch_tokens: Extracted patch tokens [B*S, P, C]
            images_normalized: Normalized images [B*S, 3, H, W]
        """
        B, S, C_in, H, W = images.shape
        
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        
        # Normalize images and reshape for patch embed
        images_normalized = (images - self.aggregator._resnet_mean) / self.aggregator._resnet_std
        
        # Reshape to [B*S, C, H, W] for patch embedding
        images_flat = images_normalized.view(B * S, C_in, H, W)
        patch_tokens = self.aggregator.patch_embed(images_flat)
        
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        
        return patch_tokens
    
    def aggregate_with_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        B: int,
        S: int,
        H: int,
        W: int,
        register_token_override: torch.Tensor = None,
        use_checkpoint: bool = False
    ):
        """
        Apply aggregator blocks with pre-computed patch tokens.
        
        Args:
            patch_tokens: Pre-computed patch tokens [B*S, P, C]
            B: Batch size
            S: Sequence length
            H: Image height
            W: Image width
            register_token_override: Optional override for register tokens
            use_checkpoint: Whether to use gradient checkpointing
        """
        _, P, C = patch_tokens.shape
        
        # Use override tokens if provided, otherwise use default
        if register_token_override is None:
            register_token = self.aggregator.register_token
        else:
            register_token = register_token_override
        
        # Process camera and register tokens
        camera_token = slice_expand_and_flatten(self.aggregator.camera_token, B, S)
        register_token_expanded = slice_expand_and_flatten(register_token, B, S)
        
        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token_expanded, patch_tokens], dim=1)
        
        pos = None
        if self.aggregator.rope is not None:
            pos = self.aggregator.position_getter(
                B * S, H // self.aggregator.patch_size, W // self.aggregator.patch_size,
                device=patch_tokens.device
            )
        
        if self.aggregator.patch_start_idx > 0:
            # Do not use position embedding for special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.aggregator.patch_start_idx, 2).to(patch_tokens.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        
        # Update P because we added special tokens
        _, P, C = tokens.shape
        
        frame_idx = 0
        global_idx = 0
        output_list = []

        for block_idx in range(self.aggregator.aa_block_num):
            for attn_type in self.aggregator.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, use_checkpoint=use_checkpoint
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos, use_checkpoint=use_checkpoint
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            for i in range(len(frame_intermediates)):
                # if block_idx in [4, 11, 17, 23]:
                    # Concat frame and global intermediates
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
        
        del concat_inter
        del frame_intermediates
        del global_intermediates
        
        return output_list, self.aggregator.patch_start_idx
    
    def _process_frame_attention(
        self, tokens, B, S, P, C, frame_idx, pos=None, use_checkpoint=False
    ):
        """Process frame attention blocks."""
        # Reshape for frame attention
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B * S, P, C)
        
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B * S, P, 2)
        
        intermediates = []
        
        for _ in range(self.aggregator.aa_block_size):
            # Apply block
            if use_checkpoint:
                tokens = checkpoint(
                    self.aggregator.frame_blocks[frame_idx], tokens, pos,
                    use_reentrant=self.aggregator.use_reentrant
                )
            else:
                tokens = self.aggregator.frame_blocks[frame_idx](tokens, pos=pos)
            
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        
        return tokens, frame_idx, intermediates
    
    def _process_global_attention(
        self, tokens, B, S, P, C, global_idx, pos=None, use_checkpoint=False
    ):
        """Process global attention blocks."""
        # Reshape for global attention
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S * P, C)
        
        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S * P, 2)
        
        intermediates = []
        
        for _ in range(self.aggregator.aa_block_size):
            # Apply block
            if use_checkpoint:
                tokens = checkpoint(
                    self.aggregator.global_blocks[global_idx], tokens, pos,
                    use_reentrant=self.aggregator.use_reentrant
                )
            else:
                tokens = self.aggregator.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
        
        return tokens, global_idx, intermediates
    
    # ----------------------
    # LoRA target modules (implements LoRAModel interface)
    # ----------------------
    def get_lora_target_modules(self) -> List[nn.Module]:
        """
        Return the list of modules to apply LoRA adapters to.
        
        For VGGT, we apply LoRA to frame_blocks and global_blocks in the aggregator.
        
        Returns:
            List of nn.Module objects (frame_blocks + global_blocks)
        """
        modules = []
        # Add frame blocks
        for block in self.aggregator.frame_blocks:
            modules.append(block)
        # Add global blocks
        for block in self.aggregator.global_blocks:
            modules.append(block)
        return modules

    @torch.no_grad()
    def forward_original_vggt(self, images: torch.Tensor):
        patch_tokens = self.extract_patch_tokens(images)
        B, S, _, H, W = images.shape

        aggregated_tokens_list, patch_start_idx = self.aggregate_with_patch_tokens(
            patch_tokens, B, S, H, W, register_token_override=None, use_checkpoint=False)

        predictions_orig = {}
        with torch.amp.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions_orig["pose_enc"] = pose_enc_list[-1]
                extrinsics, intrinsics_orig = pose_encoding_to_extri_intri(
                    pose_enc_list[-1], image_size_hw=(H, W), build_intrinsics=True
                )
                predictions_orig["extrinsics"] = extrinsics_to_se3(extrinsics)
                camera_poses_orig = invert_se3(predictions_orig["extrinsics"])
                predictions_orig["intrinsics"] = intrinsics_orig
                predictions_orig["camera_poses"] = camera_poses_orig

                # naivly set camera poses and intrinsics to gt camera poses and intrinsics
                # print("naivly set camera poses to gt camera poses")
                # from ttt3r.utils.ttt3r_utils import to_homogeneous
                # gt_poses = to_homogeneous(invert_se3(gt_extrinsics.float()))
                # gt_pose_ref = gt_poses[:, 0:1]            # (B, 1, 4, 4)
                # gt_pose_ref_inv = invert_se3(gt_pose_ref) # (B, 1, 4, 4)
                # gt_poses = gt_pose_ref_inv @ gt_poses     # (B, S, 4, 4)
                # # compute the translation scale per scene
                # t_pred_avg_norm = torch.mean(torch.norm(predictions_orig["camera_poses"][..., :3, 3], dim=-1, keepdim=True), dim=1, keepdim=True)
                # t_gt_avg_norm = torch.mean(torch.norm(gt_poses[..., :3, 3], dim=-1, keepdim=True), dim=1, keepdim=True)
                # t_scale = t_pred_avg_norm / t_gt_avg_norm
                # gt_poses[..., :3, 3] = t_scale * gt_poses[..., :3, 3]
                # predictions_orig["camera_poses"] = gt_poses
                # predictions_orig["intrinsics"] = gt_intrinsics.float()

            
            if self.depth_head is not None:
                depth_orig, depth_conf_orig = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions_orig["depth"] = depth_orig
                predictions_orig["depth_conf"] = depth_conf_orig

        return patch_tokens, predictions_orig

    def backbone_peft_setup(self, 
        lr: float, 
        lora_rank: int, 
        lora_alpha: float, 
        lora_target_modules: List[str], 
        lora_dropout: float = 0.0, 
        merge_weights: bool = False
    ):
        """Setup PEFT parameters and optimizer."""
        self.reset_lora_adapters(merge_weights=merge_weights)
        self.setup_lora_adapters(lora_rank, lora_alpha, lora_target_modules, lora_dropout)
        params_to_optimize = self.get_lora_parameters()
        optimizer = torch.optim.Adam(params_to_optimize, lr=lr) if params_to_optimize else None
        return optimizer, params_to_optimize

    def heads_ft_setup(self, 
        lr: float,
        finetune_depth_head: bool,
        finetune_camera_head: bool,
    ):
        """Setup parameters and optimizer for heads."""
        head_params_to_optimize = []
        if finetune_depth_head and self.depth_head is not None:
            head_params_to_optimize.extend(self.depth_head.parameters())
        if finetune_camera_head and self.camera_head is not None:
            head_params_to_optimize.extend(self.camera_head.parameters())
        print(f"Finetuning heads: depth_head={finetune_depth_head}, camera_head={finetune_camera_head}")
        print(f"Heads optimizer: {len(head_params_to_optimize)} parameters (lr={lr})")
        optimizer = torch.optim.Adam(head_params_to_optimize, lr=lr) if head_params_to_optimize else None
        return optimizer, head_params_to_optimize
        
    def forward(
        self,
        # ----- Inputs -----
        images: torch.Tensor,
        depths: torch.Tensor = None,
        depth_masks: torch.Tensor = None,
        gt_extrinsics: torch.Tensor = None,
        gt_intrinsics: torch.Tensor = None,
        # ----- TCO parameters -----
        tco_steps: int = 0,
        tco_lr: float = 1e-4,
        finetune_depth_head: bool = False,
        finetune_camera_head: bool = False,
        heads_lr: float = 1e-5,
        # ----- depth energy parameters -----
        lambda_depth: float = 0.0,
        depth_alignment_mode: str = "per_scene",
        depth_confidence_percentile: float = 0.5,
        depth_scale_reg_weight: float = 0.0,
        depth_shift_reg_weight: float = 0.0,
        # ----- camera energy parameters -----
        lambda_pose: float = 0.0,
        pose_translation_weight: float = 1.0,
        lambda_intrinsics: float = 0.0,
        pose_rot_loss_type: str = 'cosine',
        pose_trans_loss_type: str = 'normed_l1',
        # ----- multiview consistency energy parameters -----
        lambda_mv_consistency: float = 0.0,
        mv_depth_weight: float = 0.0,
        mv_normal_weight: float = 0.0,
        mv_rgb_weight: float = 1.0,
        num_view_groups: int = 2,
        use_cos_weighting: bool = True,
        # ----- redundancy consistency energy parameters -----
        lambda_redundancy: float = 0.0,
        # ----- LoRA parameters -----
        lora_rank: int = 4,
        lora_alpha: float = 16.0,
        lora_target_modules: List[str] = None,
        lora_dropout: float = 0.0,
        **kwargs
    ):
        """
        Forward pass with TCO.
        
        Args:
            images: Input images [B, S, 3, H, W] or [S, 3, H, W]
            depths: Input depths [B, S, 1, H, W] or [S, 1, H, W]
            depth_masks: Input depth masks [B, S, 1, H, W] or [S, 1, H, W]
            gt_extrinsics: Ground truth camera extrinsics for TCO supervision
            gt_intrinsics: Ground truth camera intrinsics for TCO supervision
            tco_steps: Number of outer TCO optimization steps (LoRA updates)
            tco_lr: Learning rate for LoRA parameter updates
            finetune_depth_head: If True, finetune depth_head parameters during TCO
            finetune_camera_head: If True, finetune camera_head parameters during TCO
            heads_lr: Learning rate for depth_head and camera_head finetuning
            lambda_*: Various loss weights
            lora_*: LoRA configuration parameters
        """        
        # Make input shapes compatible
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if depths is not None and len(depths.shape) == 4:
            depths = depths.unsqueeze(0)
        if depth_masks is not None and len(depth_masks.shape) == 4:
            depth_masks = depth_masks.unsqueeze(0)
                        
        B, S, _, H, W = images.shape
        
        # ------------------------------------------------------------
        # Get original predictions without TCO
        # ------------------------------------------------------------
        patch_tokens, predictions_orig = self.forward_original_vggt(images)


        # No TCO: return original predictions
        if tco_steps is None or tco_steps <= 0:
            predictions_orig["images"] = images
            predictions_orig["depth_points"] = unproject_depth_to_world_points_torch(
                predictions_orig["depth"], predictions_orig["extrinsics"], predictions_orig["intrinsics"]
            )
            predictions_orig["depth_conf"] = predictions_orig["depth_conf"]
            if "depth_points" in predictions_orig:
                predictions_orig["points"] = predictions_orig["depth_points"]
            if "depth_conf" in predictions_orig:
                predictions_orig["conf"] = predictions_orig["depth_conf"]
            return predictions_orig
        
        # ----------------------
        # TCO
        # ----------------------
        print("\n" + "="*80)
        print("Starting TCO")
        print("="*80)
        print(f"Use cos weighting: {use_cos_weighting}")
        
        # Params to optimize for LoRA and (optional) heads
        optimizer, params_to_optimize = self.backbone_peft_setup(
            lr=tco_lr,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            merge_weights=False
        )
        heads_optimizer, head_params_to_optimize = self.heads_ft_setup(
            lr=heads_lr,
            finetune_depth_head=finetune_depth_head,
            finetune_camera_head=finetune_camera_head
        )

        # Temporarily freeze other params (except LoRA and heads being finetuned)
        prev_requires_grad = []
        for name, p in self.named_parameters():
            # Skip LoRA parameters
            if 'lora_A' in name or 'lora_B' in name:
                continue
            # Skip depth_head parameters if finetuning
            if finetune_depth_head and 'depth_head' in name:
                continue
            # Skip camera_head parameters if finetuning
            if finetune_camera_head and 'camera_head' in name:
                continue
            prev_requires_grad.append((name, p.requires_grad))
            p.requires_grad_(False)
        
        # Ensure LoRA parameters have gradients enabled
        self.enable_lora_gradients()
        
        # Run TCO optimization steps (update LoRA parameters)
        for step in range(tco_steps):
            with torch.amp.autocast(device_type='cuda', enabled=True):
                aggregated_tokens_list, patch_start_idx = self.aggregate_with_patch_tokens(
                    patch_tokens, B, S, H, W,
                    register_token_override=None,
                    use_checkpoint=True
                )

            with torch.amp.autocast(device_type='cuda', enabled=True):
                # Get predictions
                predictions = {}
                
                # For mv rendering feature map consistency
                if self.camera_head is not None:
                    pose_enc_list = self.camera_head(aggregated_tokens_list)
                    predictions["pose_enc"] = pose_enc_list[-1]
                    extrinsics, predicted_intrinsics = pose_encoding_to_extri_intri(
                        pose_enc_list[-1], image_size_hw=(H, W), build_intrinsics=True
                    )
                    predictions["extrinsics"] = extrinsics_to_se3(extrinsics)
                    camera_poses = invert_se3(predictions["extrinsics"])
                    predictions["camera_poses"] = camera_poses
                    predictions["intrinsics"] = predicted_intrinsics
                
                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    predictions["depth"] = depth
                    predictions["depth_conf"] = depth_conf
                    
                    if self.camera_head is not None:
                        predictions["depth_points"] = unproject_depth_to_world_points_torch(
                            predictions["depth"], predictions["extrinsics"], predictions["intrinsics"]
                        )
                        predictions["depth_conf"] = predictions["depth_conf"]

            # Compute supervision losses (from known priors like depth, pose)
            loss, loss_dict = compute_tco_losses(
                predictions=predictions,
                images=images,
                depths=depths,
                depth_masks=depth_masks,
                depth_alignment_mode=depth_alignment_mode,
                depth_confidence_percentile=depth_confidence_percentile,
                depth_scale_reg_weight=depth_scale_reg_weight,
                depth_shift_reg_weight=depth_shift_reg_weight,
                gt_extrinsics=gt_extrinsics,
                gt_intrinsics=gt_intrinsics,
                lambda_pose=lambda_pose,
                lambda_depth=lambda_depth,
                pose_translation_weight=pose_translation_weight,
                lambda_intrinsics=lambda_intrinsics,
                lambda_mv_consistency=lambda_mv_consistency,
                pose_rot_loss_type=pose_rot_loss_type,
                pose_trans_loss_type=pose_trans_loss_type,
                mv_depth_weight=mv_depth_weight,
                mv_normal_weight=mv_normal_weight,
                mv_rgb_weight=mv_rgb_weight,
                num_view_groups=num_view_groups,
                use_cos_weighting=use_cos_weighting,
                lambda_redundancy=lambda_redundancy,
                meta_info={"sample_idx": kwargs.get("sample_idx", None),
                          "tco_step": step,
                          "scene_name": kwargs.get("scene_name", None)}
            )

            # Extract individual loss components
            pose_rot_loss = loss_dict['pose_rot_loss']
            pose_trans_loss = loss_dict['pose_trans_loss']
            intrinsics_loss = loss_dict['intrinsics_loss']
            depth_loss = loss_dict['depth_loss']
            mv_consistency_loss = loss_dict['mv_consistency_loss']

            # Pretty console logging
            if step == 0:
                header = f"{'STEP':>4} | {'POSE_R':>8} | {'POSE_T':>8} | {'INTR':>8} | {'DEPTH':>8} | {'MV':>8} | {'RED':>8} | {'TOTAL':>8}"
                print(header)
                print('-' * len(header))
            print(
                f"{step:4d} | "
                f"{pose_rot_loss.item():8.4f} | "
                f"{pose_trans_loss.item():8.8f} | "
                f"{intrinsics_loss.item():8.4f} | "
                f"{depth_loss.item():8.4f} | "
                f"{mv_consistency_loss.item():8.4f} | "
                f"{loss_dict['redundancy_loss'].item():8.4f} | "
                f"{loss.item():8.4f}"
            )

            # Optimization step for LoRA parameters and heads
            if optimizer is not None or heads_optimizer is not None:
                # Zero gradients for both optimizers
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                if heads_optimizer is not None:
                    heads_optimizer.zero_grad(set_to_none=True)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for LoRA parameters
                if params_to_optimize:
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
                
                # Gradient clipping for head parameters
                if head_params_to_optimize:
                    torch.nn.utils.clip_grad_norm_(head_params_to_optimize, max_norm=1.0)
                
                # Step both optimizers
                if optimizer is not None:
                    optimizer.step()
                if heads_optimizer is not None:
                    heads_optimizer.step()
        
        # Restore requires_grad flags
        for name, req in prev_requires_grad:
            param = dict(self.named_parameters())[name]
            param.requires_grad_(req)
        
        # Disable LoRA gradients after training
        self.disable_lora_gradients()
        
        # Final forward pass with adapted model
        with torch.no_grad():
            aggregated_tokens_list, patch_start_idx = self.aggregate_with_patch_tokens(
                patch_tokens, B, S, H, W,
                register_token_override=None,
                use_checkpoint=False
            )
            
            predictions = {}
            
            with torch.amp.autocast(device_type='cuda', enabled=False):
                if self.camera_head is not None:
                    pose_enc_list = self.camera_head(aggregated_tokens_list)
                    predictions["pose_enc"] = pose_enc_list[-1]
                    extrinsics, predicted_intrinsics = pose_encoding_to_extri_intri(
                        pose_enc_list[-1], image_size_hw=(H, W), build_intrinsics=True
                    )
                    predictions["extrinsics"] = extrinsics_to_se3(extrinsics)
                    camera_poses = invert_se3(predictions["extrinsics"])
                    predictions["camera_poses"] = camera_poses
                    predictions["intrinsics"] = predicted_intrinsics
                
                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    predictions["depth"] = depth
                    predictions["depth_conf"] = depth_conf
                    
                    if self.camera_head is not None:
                        predictions["depth_points"] = unproject_depth_to_world_points_torch(
                            predictions["depth"], predictions["extrinsics"], predictions["intrinsics"]
                        )
            
            predictions["images"] = images
            
            # Add compatibility keys
            if "depth_points" in predictions:
                predictions["points"] = predictions["depth_points"]
            if "depth_conf" in predictions:
                predictions["conf"] = predictions["depth_conf"]

            print("\n" + "="*80)
            print("TCO Complete")
            print("="*80 + "\n")
            
            return predictions
