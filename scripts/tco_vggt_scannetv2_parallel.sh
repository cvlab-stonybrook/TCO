#!/bin/bash
NUM_GPUS=${NUM_GPUS:-1}
CUDA_VISIBLE_DEVICES=${GPUS:-0} python evaluation/relpose/eval_dist_tco_vggt_parallel.py \
    num_gpus=${NUM_GPUS} \
    tco_steps=40 \
    tco_lr=2e-4 \
    lambda_mv_consistency=1 \
    mv_depth_weight=1 \
    mv_rgb_weight=0 \
    num_view_groups=100 \
    pose_eval_stride=1 \
    save_suffix=vggt \
    eval_datasets=[scannetv2] \
    use_cos_weighting=True
