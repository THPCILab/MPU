"""Run one fixed-codebook ratio with locally prepared EMNIST inputs."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ratio", type=int, choices=[20, 40, 60, 80, 90], default=90)
    p.add_argument("--forward-smoke", action="store_true")
    p.add_argument("--feature-cache", type=Path, help="Optional trusted local PCA cache")
    p.add_argument("--warmup-checkpoint", type=Path, help="Optional trusted local warmup")
    args = p.parse_args()
    work = ROOT / "results/quantization_training" / f"ratio{args.ratio}"
    library = work / "results_static332_t90pct"
    library.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/fig5/device_runs/codebook_static_3bit_eps_2pow_minus332_losst90pct.npz"
    shutil.copy2(source, library / source.name)
    env = dict(os.environ, STATIC332_WORK_ROOT=str(work),
               STATIC332_SOURCE_ROOT=str(ROOT / "tools/quantization"))
    if args.feature_cache:
        if not args.feature_cache.is_file():
            p.error("The local PCA cache does not exist")
        env["STATIC332_FEATURE_CACHE"] = str(args.feature_cache.resolve())
    else:
        env["STATIC332_FEATURE_CACHE"] = str(work / "local_pca_features.pt")
    script = ROOT / "tools/quantization/train_device_codebook.py"
    if args.forward_smoke:
        subprocess.run([sys.executable, str(script), "--forward-smoke"], env=env, check=True)
        return
    warmup = work / "checkpoints/warmup.pt"
    warmup.parent.mkdir(parents=True, exist_ok=True)
    if args.warmup_checkpoint:
        if not args.warmup_checkpoint.is_file():
            p.error("The local warmup checkpoint does not exist")
        shutil.copy2(args.warmup_checkpoint, warmup)
    command = [
        "--preset", "main", "--task-set", "three", "--input-preprocess", "pca",
        "--train-samples", "40000", "--test-samples", "12000",
        "--hidden-dim", "64", "--num-layers", "1", "--input-rank", "64",
        "--batch-size", "512", "--base-lr", "0.0035",
        "--electrical-lr-scale", "1.0", "--private-delta-lr-scale", "1.0",
        "--private-delta-weight-decay", "0.0001", "--weight-decay", "0.0001",
        "--score-batches", "20", "--score-source", "warmup",
        "--score-method", "quantized_fisher_damage", "--guided-node-prior", "stage_refine",
        "--guided-node-refine-fraction", "0.35", "--guided-node-balance", "global",
        "--clements-layer-mode", "stage", "--warmup-shared-ratio", "0",
        "--use-final-warmup-state", "--require-cuda", "--head-mode", "task_specific",
        "--photonic-readout", "per_mesh_intensity", "--norm-type", "layer",
        "--freeze-electrical-after-warmup", "--skip-electronic-baselines",
        "--skip-optical-unfrozen", "--efficiency", "off",
        "--save-mask-matrices", "--save-confusion-matrices", "--save-condition-checkpoints",
        "--modes", "node_pruning", "--pretrain-epochs", "80", "--epochs", "80",
        "--phase-quantization", "shared", "--phase-bits", "3", "--no-resume", "--seed", "44",
        "--warmup-checkpoint", str(warmup),
        "--ratios", str(args.ratio),
        "--output", f"seed44_ratio{args.ratio}.json",
        "--checkpoint-dir", "checkpoints",
    ]
    if not args.feature_cache:
        command.append("--download")
    subprocess.run([sys.executable, str(script), *command], env=env, check=True)


if __name__ == "__main__":
    main()
