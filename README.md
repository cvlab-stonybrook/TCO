<div align="center">
<h1>TCO: Learning 3D Reconstruction with Priors in Test Time</h1>

<a href="https://arxiv.org/abs/2604.03878"><img src="https://img.shields.io/badge/arXiv-2604.03878-b31b1b" alt="arXiv"></a>
<a href="https://github.com/cvlab-stonybrook/TCO"><img src="https://img.shields.io/badge/GitHub-TCO-blue" alt="GitHub"></a>

**[CVLab, Stony Brook University](https://www3.cs.stonybrook.edu/~cvl/)**

Lei Zhou\*, Haoyu Wu, Akshat Dave, Dimitris Samaras
</div>

```bibtex
@article{zhou2026learning,
  title={Learning 3D Reconstruction with Priors in Test Time},
  author={Zhou, Lei and Wu, Haoyu and Dave, Akshat and Samaras, Dimitris},
  journal={arXiv preprint arXiv:2604.03878},
  year={2026}
}
```

## Overview

Test-time Constrained Optimization (TCO) improves the 3D predictions of feed-forward multiview Transformers (e.g., [VGGT](https://github.com/facebookresearch/vggt)) at test time, **without retraining or modifying** the pretrained network. Rather than feeding priors into the architecture, TCO casts them as constraints on the predictions and optimizes lightweight [LoRA](https://arxiv.org/abs/2106.09685) adapters on the frozen shared decoder for each test scene. The optimization loss consists of:

- **A self-supervised objective**: multiview prediction compatibility, implemented via [2DGS](https://arxiv.org/abs/2403.17888)-based photometric/geometric rendering across views.
- **Prior penalty terms**: camera pose, intrinsics, and/or depth constraints when available.


## Quick Start

Clone this repository and install the dependencies:

```bash
git clone https://github.com/cvlab-stonybrook/TCO.git && cd TCO
conda create -n tco python=3.11
conda activate tco
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # adjust for your CUDA version
pip install -r requirements.txt
```

## Data Preparation

Please follow the instructions in [Pi3](https://github.com/yyfz233/Pi3) to download and process the datasets.
Then, organize datasets under `data/` (or symlink them):

```
data/
├── scannetv2/          # ScanNet v2 (camera pose estimation)
├── 7scenes/            # 7-Scenes (point map estimation)
├── nrgbd/              # Neural RGBD (point map estimation)
├── dtu/                # DTU (point map estimation)
└── eth3d/              # ETH3D (point map estimation)
```


## Usage

### Point Map Estimation with Camera Priors

Evaluate on ETH3D, 7-Scenes, DTU, or NRGBD:

```bash
bash scripts/tco_vggt_eth3d.sh
bash scripts/tco_vggt_7scenes.sh
bash scripts/tco_vggt_dtu.sh
bash scripts/tco_vggt_nrgbd.sh
```

### Camera Pose Estimation with Depth Priors

Evaluate on ScanNet v2:

```bash
bash scripts/tco_vggt_scannetv2.sh
```

### Custom Runs

All scripts are thin wrappers around Hydra-configured Python entry points. You can override any config parameter from the command line:

```bash
# Point map estimation
CUDA_VISIBLE_DEVICES=0 python evaluation/mv_recon/eval_vggt.py \
    tco_steps=40 \
    tco_lr=5e-4 \
    lambda_mv_consistency=0.2 \
    num_view_groups=100 \
    eval_datasets=[ETH3D]

# Camera pose estimation
CUDA_VISIBLE_DEVICES=0 python evaluation/relpose/eval_dist_tco_vggt.py \
    tco_steps=40 \
    tco_lr=2e-4 \
    lambda_mv_consistency=1 \
    mv_depth_weight=1 \
    eval_datasets=[scannetv2]
```


## Project Structure

```
TCO/
├── tco_vggt_lora.py        # TCO_VGGT_LoRA: VGGT + LoRA + TCO optimization loop
├── tco_loss.py              # Combined TCO loss (Eq. 4 in the paper)
├── constraints.py           # Prior penalty terms: pose, intrinsics, depth (Sec. 3.2.2)
├── objective.py             # Prediction compatibility via 2DGS rendering (Sec. 3.2.1)
├── peft/
│   └── lora.py              # LoRA adapter implementation
├── utils/                   # Inference wrappers, geometry, I/O utilities
├── evaluation/
│   ├── mv_recon/            # Point map estimation eval (ETH3D, DTU, 7-Scenes, NRGBD)
│   └── relpose/             # Camera pose estimation eval (ScanNet v2)
├── datasets/                # Dataset loaders (7-Scenes, NRGBD, DTU, ETH3D)
├── configs/                 # Hydra configs (eval tasks, dataset paths, hyperparams)
├── scripts/                 # Ready-to-run shell scripts
├── base_models/
│   └── vggt/                # VGGT model (facebook/VGGT-1B)
└── data/                    # Dataset root (symlink or download here)
```


## Key Hyperparameters

| Parameter | Description | Typical Range |
|---|---|---|
| `tco_steps` | Number of LoRA optimization steps per scene | 10–80 |
| `tco_lr` | Learning rate for LoRA parameters | 1e-4 – 1e-3 |
| `lambda_mv_consistency` | Weight for multiview photometric/geometric consistency | 0.2–1.0 |
| `lambda_depth` | Weight for depth prior penalty | 0–1 |
| `lambda_pose` | Weight for camera pose prior penalty | 0–1 |
| `lambda_intrinsics` | Weight for intrinsics prior penalty | 0–0.01 |
| `num_view_groups` | View groups for 2DGS rendering (higher = fewer views per group) | 2–100 (100 leads to 1 view per group) |
| `lora_rank` | Rank of LoRA adapters | 1–16 (default: 4) |


## Acknowledgements
Thanks to these great repositories:
- [Pi3](https://github.com/yyfz233/Pi3) — Pi3: Permutation-Equivariant Visual Geometry Learning
- [VGGT](https://github.com/facebookresearch/vggt) — Visual Geometry Grounded Transformer
- [gsplat](https://github.com/nerfstudio-project/gsplat) — Gaussian Splatting library
