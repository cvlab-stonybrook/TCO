"""
Multi-GPU parallel evaluation for camera pose estimation (relpose-distance).
Uses subprocess.Popen with per-process CUDA_VISIBLE_DEVICES for true GPU
isolation (avoids gsplat CUDA kernel conflicts between co-resident processes).

When num_gpus=1 (default), behaves identically to eval_dist_tco_vggt.py.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation/relpose/eval_dist_tco_vggt_parallel.py \
        num_gpus=4 eval_datasets=[scannetv2] tco_steps=40 tco_lr=2e-4 ...
"""

import os
import json
import subprocess
import sys
import os.path as osp
import logging
import traceback
import numpy as np
import torch
import hydra

from omegaconf import DictConfig, OmegaConf

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "base_models" / "vggt"))

from tco_vggt_lora import TCO_VGGT_LoRA
from utils.interfaces import infer_cameras_c2w_w_priors
from utils.files import list_imgs_a_sequence, list_depths_a_sequence, get_all_sequences
from utils.messages import set_default_arg, write_csv, save_list_of_matrices
from evo_utils import (calculate_averages, load_traj, eval_metrics,
                       plot_trajectory, get_tum_poses, save_tum_poses)


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-sequence evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_one_sequence(seq_idx, seq, dataset_info_dict, model, hydra_cfg,
                          output_root):
    """Evaluate a single sequence. Returns result dict or None."""
    dataset_info = OmegaConf.create(dataset_info_dict)

    filelist = list_imgs_a_sequence(dataset_info, seq)
    depth_filelist = list_depths_a_sequence(dataset_info, seq)
    filelist = filelist[::hydra_cfg.pose_eval_stride]
    depth_filelist = depth_filelist[::hydra_cfg.pose_eval_stride]
    if hydra_cfg.pose_eval_num_frames is not None:
        filelist = filelist[:hydra_cfg.pose_eval_num_frames]
        depth_filelist = depth_filelist[:hydra_cfg.pose_eval_num_frames]

    pr_poses, pr_intrs = infer_cameras_c2w_w_priors(
        filelist, depth_filelist, model, hydra_cfg,
        sample_idx=seq_idx, scene_name=seq)
    pred_traj = get_tum_poses(pr_poses)

    seq_save_dir = osp.join(output_root, seq)
    os.makedirs(seq_save_dir, exist_ok=True)
    save_tum_poses(pred_traj, osp.join(seq_save_dir, "pred_traj.txt"),
                   verbose=hydra_cfg.verbose)
    np.save(osp.join(seq_save_dir, "pred_poses.npy"), pr_poses)
    save_list_of_matrices(pr_poses.numpy().tolist(),
                          osp.join(seq_save_dir, "pred_poses.json"))
    if pr_intrs is not None:
        np.save(osp.join(seq_save_dir, "pred_intrinsics.npy"), pr_intrs)
        save_list_of_matrices(pr_intrs.tolist(),
                              osp.join(seq_save_dir, "pred_intrinsics.json"))

    gt_traj = load_traj(
        gt_traj_file=dataset_info.anno.path.format(seq=seq),
        traj_format=dataset_info.anno.format,
        stride=hydra_cfg.pose_eval_stride,
        num_frames=hydra_cfg.pose_eval_num_frames,
    )
    if gt_traj is None:
        return None

    ate, rpe_trans, rpe_rot = eval_metrics(
        pred_traj, gt_traj, seq=seq,
        filename=osp.join(seq_save_dir, "eval_metric.txt"),
        verbose=hydra_cfg.verbose)
    plot_trajectory(
        pred_traj, gt_traj, title=seq,
        filename=osp.join(seq_save_dir, "vis.png"),
        verbose=hydra_cfg.verbose)

    return {
        "seq": seq,
        "seq_len": len(filelist),
        "ATE": ate,
        "RPE_trans": rpe_trans,
        "RPE_rot": rpe_rot,
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
    logger = logging.getLogger(f"relpose-worker{worker_id}")

    with open(config_path) as f:
        cfg = json.load(f)

    hydra_cfg = OmegaConf.create(cfg["hydra_cfg"])
    OmegaConf.update(hydra_cfg, "device", "cuda:0")

    dataset_name      = cfg["dataset_name"]
    dataset_info_dict = cfg["dataset_info"]
    seq_list          = cfg["seq_list"]
    output_root       = cfg["output_root"]
    results_dir       = cfg["results_dir"]

    model = TCO_VGGT_LoRA.from_pretrained(
        cfg["hydra_cfg"]["vggt"]["pretrained_model_name_or_path"],
        enable_track=False,
    ).to("cuda:0").eval()

    my_seqs = [(i, s) for i, s in enumerate(seq_list)
               if i % num_workers == worker_id]
    logger.info(f"{len(my_seqs)}/{len(seq_list)} sequences on GPU "
                f"{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}")

    for global_idx, seq in my_seqs:
        seq_idx = global_idx + 1
        try:
            result = evaluate_one_sequence(
                seq_idx, seq, dataset_info_dict, model, hydra_cfg, output_root)
            if result is not None:
                with open(osp.join(results_dir, f"seq_{global_idx:04d}.json"), "w") as f:
                    json.dump(result, f)
                logger.info(
                    f"{seq_idx}/{len(seq_list)} {seq} | "
                    f"ATE: {result['ATE']:.4f} | RPE-t: {result['RPE_trans']:.4f} | "
                    f"RPE-r: {result['RPE_rot']:.4f}")
            else:
                logger.warning(f"No GT trajectory for {seq}")
        except np.linalg.LinAlgError:
            logger.warning(f"LinAlgError on {seq}, skipping")
        except Exception as e:
            logger.error(f"Error on {seq}: {e}")
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
#  Orchestrator (Hydra)
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base="1.2", config_path="../../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    num_gpus = getattr(hydra_cfg, "num_gpus", 1)
    all_eval_datasets = hydra_cfg.eval_datasets
    all_data_info     = hydra_cfg.data
    logger = logging.getLogger("relpose-parallel")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_info = all_data_info[dataset_name]

        seq_list = get_all_sequences(dataset_info)
        if hydra_cfg.debug:
            seq_list = seq_list[::10]

        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        os.makedirs(output_root, exist_ok=True)

        logger.info(
            f"[{idx_dataset}/{len(all_eval_datasets)}] {dataset_name}: "
            f"{len(seq_list)} sequences on {num_gpus} GPU(s)")

        results_dir = osp.join(output_root, "_parallel_results")
        os.makedirs(results_dir, exist_ok=True)

        cfg_dict          = OmegaConf.to_container(hydra_cfg, resolve=True)
        dataset_info_dict = OmegaConf.to_container(dataset_info, resolve=True)

        # ── Multi-GPU: launch isolated subprocesses ──────────────────────
        if num_gpus > 1:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            gpu_ids = visible.split(",") if visible else [str(i) for i in range(num_gpus)]
            if len(gpu_ids) < num_gpus:
                raise ValueError(
                    f"num_gpus={num_gpus} but only {len(gpu_ids)} GPUs visible "
                    f"(CUDA_VISIBLE_DEVICES={visible})")

            worker_config = {
                "hydra_cfg":     cfg_dict,
                "dataset_name":  dataset_name,
                "dataset_info":  dataset_info_dict,
                "seq_list":      seq_list,
                "output_root":   output_root,
                "results_dir":   results_dir,
            }
            config_path = osp.join(results_dir, "worker_config.json")
            with open(config_path, "w") as f:
                json.dump(worker_config, f)

            script = osp.abspath(__file__)
            processes = []
            for rank in range(num_gpus):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_ids[rank]
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

            for idx, seq in enumerate(seq_list):
                seq_idx = idx + 1
                try:
                    result = evaluate_one_sequence(
                        seq_idx, seq, dataset_info_dict, model, hydra_cfg,
                        output_root)
                    if result is not None:
                        with open(osp.join(results_dir, f"seq_{idx:04d}.json"), "w") as f:
                            json.dump(result, f)
                        logger.info(
                            f"{seq_idx}/{len(seq_list)} {seq} | "
                            f"ATE: {result['ATE']:.4f} | RPE-t: {result['RPE_trans']:.4f}")
                except np.linalg.LinAlgError:
                    logger.warning(f"LinAlgError on {seq}, skipping")
                except Exception as e:
                    logger.error(f"Error on {seq}: {e}", exc_info=True)
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

        for r in all_results:
            write_csv(osp.join(output_root, "seq_metrics.csv"), {
                "dataset": dataset_name,
                "seq": r["seq"],
                "ATE": r["ATE"],
                "RPE trans": r["RPE_trans"],
                "RPE rot": r["RPE_rot"],
            })

        results_tuples = [(r["seq"], r["ATE"], r["RPE_trans"], r["RPE_rot"])
                          for r in all_results]
        avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results_tuples)

        dataset_metrics = {
            "ATE": avg_ate,
            "RPE trans": avg_rpe_trans,
            "RPE rot": avg_rpe_rot,
        }
        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, dataset_metrics)
        logger.info(
            f"{dataset_name} - {len(all_results)}/{len(seq_list)} seqs: {dataset_metrics}")

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
        set_default_arg("evaluation", "relpose-distance")
        os.environ["HYDRA_FULL_ERROR"] = "1"
        main()
