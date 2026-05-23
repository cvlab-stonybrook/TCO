#!/bin/bash
NUM_GPUS=${NUM_GPUS:-1}
CUDA_VISIBLE_DEVICES=${GPUS:-0} python evaluation/mv_recon/eval_vggt_parallel.py \
    num_gpus=${NUM_GPUS} \
    tco_steps=40 \
    tco_lr=1e-3 \
    pose_translation_weight=2 \
    lambda_intrinsics=0.01 \
    lambda_mv_consistency=0.2 \
    num_view_groups=100 \
    save_suffix=vggt \
    eval_datasets=[7scenes-sparse] \
    use_cos_weighting=True
