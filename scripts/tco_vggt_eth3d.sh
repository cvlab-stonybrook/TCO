#!/bin/bash
CUDA_VISIBLE_DEVICES=${GPUS:-0} python evaluation/mv_recon/eval_vggt.py \
    tco_steps=40 \
    tco_lr=5e-4 \
    pose_translation_weight=2 \
    lambda_intrinsics=0.01 \
    lambda_mv_consistency=0.2 \
    num_view_groups=100 \
    save_suffix=vggt \
    eval_datasets=[ETH3D] \
    use_cos_weighting=True
