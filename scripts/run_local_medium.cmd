@echo off
setlocal

cd /d "%~dp0\.."

python tools\download_emnist.py

python train_mpu_pnn.py ^
  --preset main ^
  --task-set three ^
  --seed 44 ^
  --input-preprocess pca ^
  --train-samples 12000 ^
  --test-samples 3000 ^
  --hidden-dim 64 ^
  --num-layers 1 ^
  --input-rank 64 ^
  --epochs 30 ^
  --pretrain-epochs 30 ^
  --baseline-epochs 1 ^
  --batch-size 512 ^
  --base-lr 0.0035 ^
  --private-delta-weight-decay 0.0001 ^
  --score-batches 8 ^
  --score-source warmup ^
  --score-method fisher_damage ^
  --guided-node-prior stage_refine ^
  --guided-node-refine-fraction 0.35 ^
  --guided-node-balance global ^
  --clements-layer-mode stage ^
  --warmup-shared-ratio 0 ^
  --use-best-warmup-state ^
  --head-mode task_specific ^
  --photonic-readout per_mesh_intensity ^
  --norm-type layer ^
  --freeze-electrical-after-warmup ^
  --skip-electronic-baselines ^
  --skip-optical-unfrozen ^
  --efficiency off ^
  --ratios 0 50 90 100 ^
  --modes node_pruning clements_layer ^
  --save-mask-matrices ^
  --save-confusion-matrices ^
  --save-condition-checkpoints ^
  --track-train-accuracy ^
  --checkpoint-dir results\local_medium_ckpt ^
  --output results\local_medium.json ^
  --no-resume

python tools\plot_nature_figures.py --input results\local_medium.json --out-dir results\v27_nature_figures

endlocal
