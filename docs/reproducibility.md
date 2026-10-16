# Reproducibility Notes

## Continuous-sharing figure data

The continuous-sharing analysis plots are generated from:

- `data/main_10pct_sweep.json`: 10%-step accuracy, masks, confusion matrices, and fine-tuning histories.
- `data/warmup_main.json`: full warmup histories for the main warmup run.
- `data/replicate_20pct_sweep.json`: independent 20%-step confirmation run.
- `data/warmup_replicate.json`: full warmup histories for the replicate warmup run.

The key reported number is read from the main 10%-step sweep:

- 90% neuron-level sharing: 87.64% average test accuracy.
- 90% layer-level sharing: 80.38% average test accuracy.
- Difference: 7.26 percentage points.

## Regenerating Figures

```bash
python tools/plot_nature_figures.py
python tools/plot_architecture.py
python tools/export_figure_data.py
python tools/plot_fig5_quantization.py
```

## Regenerating Training Results

For the quick offline CPU demo:

```bash
python scripts/run_demo.py
```

The following Windows command is a medium training run (30 warmup and
30 fine-tuning epochs, public EMNIST download), not a quick demo:

```powershell
scripts\run_local_medium.cmd
```

For the main 10%-step result, run the four-GPU launcher below. It now generates
the warmup checkpoint automatically when none exists, then reuses it:

```bash
bash scripts/run_main_10pct_sweep_reuse_warmup.sh
```

For an independent 20%-step confirmation run (seven GPUs):

```bash
bash scripts/run_replicate_20pct_sweep.sh
```

The training script saves JSON histories, masks, confusion matrices, and checkpoints when the corresponding `--save-*` flags are enabled.

These launchers regenerate training outputs and can replace previous generated
shards/logs in their output trees. They do not modify archived inputs under data/.
Do not run the multi-GPU launchers on a single-GPU host unchanged.

For one-GPU warmup generation, the same parameters can be passed directly:

```bash
python train_mpu_pnn.py --preset main --task-set three --seed 44 --download --require-cuda --input-preprocess pca --train-samples 40000 --test-samples 12000 --hidden-dim 64 --num-layers 1 --input-rank 64 --pretrain-epochs 80 --epochs 80 --batch-size 512 --base-lr 0.0035 --head-mode task_specific --photonic-readout per_mesh_intensity --norm-type layer --warmup-shared-ratio 0 --use-best-warmup-state --skip-electronic-baselines --skip-optical-unfrozen --efficiency off --warmup-only --track-train-accuracy --checkpoint-dir results/warmup_main_ckpt --warmup-checkpoint results/warmup_main_ckpt/shared_warmup.pt --output results/local_warmup.json
```

For a single-GPU sharing condition, use the complete COMMON_ARGS protocol in the
main launcher with one ratio/mode instead of the parallel run_shard calls.
The archived curves can be regenerated without this expensive training step.

Plot/export commands accept sparse ratio grids: missing 90% panels are skipped,
not replaced by a different ratio. A one-row pattern grid is also supported.
For a new input with a separately saved warmup, pass
`--warmup-input path/to/matching_warmup.json` to both plotting and export.
Only the exact published main/replicate inputs auto-select their own companion;
a custom input never silently borrows a different experiment's warmup.

## Quantization and device-error study

The submission Fig. 5(d) source data and quantized training code follow the same
`tools/data/paper_data/results` layout. See
[fig5_reproducibility.md](fig5_reproducibility.md) for the per-condition seed
counts, common PCA inputs, fixed codebook and paired active-control error model.
CPU source-data validation and forward/backward smoke tests are available:

```bash
python tools/plot_fig5_quantization.py --validate-only
python tools/check_quantization_model.py
```

