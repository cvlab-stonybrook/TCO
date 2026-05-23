"""
Multi-GPU parallel evaluation for MV point-map reconstruction.
Uses subprocess.Popen with per-process CUDA_VISIBLE_DEVICES for true GPU
isolation (avoids gsplat CUDA kernel conflicts between co-resident processes).

When num_gpus=1 (default), behaves identically to eval_vggt.py.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation/mv_recon/eval_vggt_parallel.py \
        num_gpus=4 eval_datasets=[ETH3D] tco_steps=40 tco_lr=5e-4 ...
"""

import os
import json
import subprocess
import sys
import torch
import numpy as np
import open3d as o3d
import os.path as osp
import hydra
import logging
import traceback

from omegaconf import DictConfig, OmegaConf

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "base_models" / "vggt"))

from tco_vggt_lora import TCO_VGGT_LoRA
from utils.interfaces import infer_mv_pointclouds_w_priors_v2 as infer_mv_pointclouds_w_priors
from evaluation.mv_recon.utils import umeyama, accuracy, completion, robust_umeyama
from utils.messages import set_default_arg, write_csv
from utils.vis_utils import save_image_grid_auto


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-sequence evaluation (shared by single-GPU and worker paths)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_one_sequence(seq_idx, seq_name, ids, dataset, model, hydra_cfg,
                          dataset_name, output_root):
    """Evaluate a single sequence and return per-sequence metrics dict."""
    data = dataset.get_data(sequence_name=seq_name, ids=ids)
    images     = data['images']
    gt_pts     = data['pointclouds']
    valid_mask = data['valid_mask']

    seq_name_safe = seq_name.replace("/", "-")
    pred_pts = infer_mv_pointclouds_w_priors(
        data, model, hydra_cfg, sample_idx=seq_idx, scene_name=seq_name_safe)
    assert pred_pts.shape == gt_pts.shape, \
        f"Predicted {pred_pts.shape} != GT {gt_pts.shape}"

    save_image_grid_auto(images, osp.join(output_root, f"{seq_name_safe}.png"))
    colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)

    if hydra_cfg.use_robust_umeyama:
        c, R, t, _ = robust_umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
    else:
        c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
    pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T

    pred_pts = pred_pts[valid_mask].reshape(-1, 3)
    gt_pts   = gt_pts[valid_mask].reshape(-1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pred_pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name_safe}-pred.ply"), pcd)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
    pcd_gt.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name_safe}-gt.ply"), pcd_gt)

    threshold = 100 if "DTU" in dataset_name else 0.1
    reg = o3d.pipelines.registration.registration_icp(
        pcd, pcd_gt, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    pcd = pcd.transform(reg.transformation)

    pcd.estimate_normals()
    pcd_gt.estimate_normals()
    pred_normal = np.asarray(pcd.normals)
    gt_normal   = np.asarray(pcd_gt.normals)

    acc, acc_med, nc1, nc1_med   = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
    comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)

    return {
        "seq": seq_name_safe,
        "Acc-mean": acc,  "Acc-med": acc_med,
        "Comp-mean": comp, "Comp-med": comp_med,
        "NC1-mean": nc1,  "NC1-med": nc1_med,
        "NC2-mean": nc2,  "NC2-med": nc2_med,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Worker entry point (subprocess with single-GPU CUDA_VISIBLE_DEVICES)
# ═══════════════════════════════════════════════════════════════════════════════

def _is_cuda_fatal(exc):
    msg = str(exc).lower()
    return any(s in msg for s in [
        "illegal memory access", "device-side assert",
        "cuda error", "out of memory",
    ])


def run_worker(config_path, worker_id, num_workers):
    """Standalone worker — no Hydra.  Reads config from a JSON file."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s][worker-{worker_id}][%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(f"mv_recon-worker{worker_id}")

    with open(config_path) as f:
        cfg = json.load(f)

    hydra_cfg = OmegaConf.create(cfg["hydra_cfg"])
    OmegaConf.update(hydra_cfg, "device", "cuda:0")

    dataset_name     = cfg["dataset_name"]
    dataset_cfg_dict = cfg["dataset_cfg"]
    seq_items        = cfg["seq_items"]       # [[name, ids], ...]
    output_root      = cfg["output_root"]
    results_dir      = cfg["results_dir"]

    model = TCO_VGGT_LoRA.from_pretrained(
        cfg["hydra_cfg"]["vggt"]["pretrained_model_name_or_path"],
        enable_track=False,
    ).to("cuda:0").eval()
    dataset = hydra.utils.instantiate(dataset_cfg_dict)

    my_seqs = [(i, name, ids) for i, (name, ids) in enumerate(seq_items)
               if i % num_workers == worker_id]
    logger.info(f"{len(my_seqs)}/{len(seq_items)} sequences on GPU "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}")

    for global_idx, seq_name, ids in my_seqs:
        seq_idx = global_idx + 1
        try:
            result = evaluate_one_sequence(
                seq_idx, seq_name, ids, dataset, model,
                hydra_cfg, dataset_name, output_root)
            with open(osp.join(results_dir, f"seq_{global_idx:04d}.json"), "w") as f:
                json.dump(result, f)
            logger.info(
                f"{seq_idx}/{len(seq_items)} {result['seq']} | "
                f"Acc: {result['Acc-mean']:.4f} | Comp: {result['Comp-mean']:.4f} | "
                f"NC: {(result['NC1-mean']+result['NC2-mean'])/2:.4f}")
        except Exception as e:
            logger.error(f"Error on {seq_name}: {e}")
            traceback.print_exc()
            if _is_cuda_fatal(e):
                logger.error("Fatal CUDA error — aborting remaining sequences")
                break
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.error("CUDA context corrupted — aborting remaining sequences")
            break

    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  Debug filter (same as eval_vggt.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _filter_debug_sequences(dataset_name, seq_items):
    debug_indices = {
        "7scenes-sparse": {7, 8},
        "DTU":            {18, 21},
        "ETH3D":          {6, 9, 12},
        "NRGBD-sparse":   {8},
    }
    allowed = debug_indices.get(dataset_name)
    if allowed is None:
        return seq_items
    return [(n, ids) for i, (n, ids) in enumerate(seq_items, start=1) if i in allowed]


# ═══════════════════════════════════════════════════════════════════════════════
#  Orchestrator (Hydra)
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base="1.2", config_path="../../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    num_gpus = getattr(hydra_cfg, "num_gpus", 1)
    all_eval_datasets = hydra_cfg.eval_datasets
    all_data_info     = hydra_cfg.data
    logger = logging.getLogger("mv_recon-parallel")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_info = all_data_info[dataset_name]

        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        os.makedirs(output_root, exist_ok=True)

        with open(dataset_info.seq_id_map, "r") as f:
            seq_id_map: dict = json.load(f)

        seq_items = list(seq_id_map.items())
        if hydra_cfg.debug:
            seq_items = _filter_debug_sequences(dataset_name, seq_items)

        logger.info(
            f"[{idx_dataset}/{len(all_eval_datasets)}] {dataset_name}: "
            f"{len(seq_items)} sequences on {num_gpus} GPU(s)")

        results_dir = osp.join(output_root, "_parallel_results")
        os.makedirs(results_dir, exist_ok=True)

        cfg_dict         = OmegaConf.to_container(hydra_cfg, resolve=True)
        dataset_cfg_dict = OmegaConf.to_container(dataset_info.cfg, resolve=True)

        # ── Multi-GPU: launch isolated subprocesses ──────────────────────
        if num_gpus > 1:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            gpu_ids = visible.split(",") if visible else [str(i) for i in range(num_gpus)]
            if len(gpu_ids) < num_gpus:
                raise ValueError(
                    f"num_gpus={num_gpus} but only {len(gpu_ids)} GPUs visible "
                    f"(CUDA_VISIBLE_DEVICES={visible})")

            worker_config = {
                "hydra_cfg":    cfg_dict,
                "dataset_name": dataset_name,
                "dataset_cfg":  dataset_cfg_dict,
                "seq_items":    seq_items,
                "output_root":  output_root,
                "results_dir":  results_dir,
            }
            config_path = osp.join(results_dir, "worker_config.json")
            with open(config_path, "w") as f:
                json.dump(worker_config, f)

            script = osp.abspath(__file__)
            processes = []
            for rank in range(num_gpus):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[rank]
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                cmd = [
                    sys.executable, script,
                    "--worker-config", config_path,
                    "--worker-id", str(rank),
                    "--num-workers", str(num_gpus),
                ]
                p = subprocess.Popen(cmd, env=env)
                processes.append(p)
                logger.info(f"  Launched worker {rank} on GPU {gpu_ids[rank]} (pid {p.pid})")

            for p in processes:
                p.wait()

            failed = [i for i, p in enumerate(processes) if p.returncode != 0]
            if failed:
                logger.warning(f"Workers {failed} exited with errors")

        # ── Single GPU: run in-process ───────────────────────────────────
        else:
            model = TCO_VGGT_LoRA.from_pretrained(
                hydra_cfg.vggt.pretrained_model_name_or_path, enable_track=False
            ).to(hydra_cfg.device).eval()
            dataset = hydra.utils.instantiate(dataset_cfg_dict)

            for idx, (seq_name, ids) in enumerate(seq_items):
                seq_idx = idx + 1
                try:
                    result = evaluate_one_sequence(
                        seq_idx, seq_name, ids, dataset, model,
                        hydra_cfg, dataset_name, output_root)
                    with open(osp.join(results_dir, f"seq_{idx:04d}.json"), "w") as f:
                        json.dump(result, f)
                    logger.info(
                        f"{seq_idx}/{len(seq_items)} {result['seq']} | "
                        f"Acc: {result['Acc-mean']:.4f} | Comp: {result['Comp-mean']:.4f}")
                except Exception as e:
                    logger.error(f"Error on {seq_name}: {e}", exc_info=True)
                torch.cuda.empty_cache()

            del model
            torch.cuda.empty_cache()

        # ── Aggregate per-sequence results ───────────────────────────────
        all_results = []
        for fname in sorted(os.listdir(results_dir)):
            if fname.endswith(".json") and fname.startswith("seq_"):
                with open(osp.join(results_dir, fname)) as f:
                    all_results.append(json.load(f))

        if not all_results:
            logger.error(f"[{dataset_name}] No sequences completed successfully")
            continue

        samples_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-samples")
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            samples_file += f"-{hydra_cfg.save_suffix}"
        samples_file += ".csv"
        if osp.exists(samples_file):
            os.remove(samples_file)
        for r in all_results:
            write_csv(samples_file, r)

        num_samples = len(all_results)
        metric_keys = ["Acc-mean", "Acc-med", "Comp-mean", "Comp-med",
                       "NC1-mean", "NC1-med", "NC2-mean", "NC2-med"]
        metric_dict = {k: sum(r[k] for r in all_results) / num_samples
                       for k in metric_keys}
        metric_dict["NC-mean"] = (metric_dict["NC1-mean"] + metric_dict["NC2-mean"]) / 2
        metric_dict["NC-med"]  = (metric_dict["NC1-med"]  + metric_dict["NC2-med"])  / 2

        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, metric_dict)

        logger.info(f"[{dataset_name}] {num_samples}/{len(seq_items)} sequences evaluated:")
        logger.info(f"  Acc-mean: {metric_dict['Acc-mean']:.4f} | Acc-med: {metric_dict['Acc-med']:.4f}")
        logger.info(f"  Comp-mean: {metric_dict['Comp-mean']:.4f} | Comp-med: {metric_dict['Comp-med']:.4f}")
        logger.info(f"  NC-mean: {metric_dict['NC-mean']:.4f} | NC-med: {metric_dict['NC-med']:.4f}")

    logger.info("Finished all datasets.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Dispatch: orchestrator (Hydra) vs worker (standalone)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker-config", type=str, default=None)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args, _remaining = parser.parse_known_args()

    if args.worker_config:
        run_worker(args.worker_config, args.worker_id, args.num_workers)
    else:
        set_default_arg("evaluation", "mv_recon")
        os.environ["HYDRA_FULL_ERROR"] = "1"
        main()
