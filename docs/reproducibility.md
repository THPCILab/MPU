# Reproducibility Notes

## Final Figure Data

The final paper figures are generated from:

- `data/v27_3090_fine10_merged.json`: 10%-step accuracy, masks, confusion matrices, and fine-tuning histories.
- `data/v27_3090_warmup.json`: full warmup histories for the seed-44 3090 run.
- `data/v27_4090_merged.json`: independent 20%-step confirmation run.
- `data/v27_4090_warmup.json`: full warmup histories for the seed-43 4090 run.

The key reported number is read from the 3090 fine-grid result:

- 90% neuron-level sharing: 87.64% average test accuracy.
- 90% layer-level sharing: 80.38% average test accuracy.
- Difference: 7.26 percentage points.

## Regenerating Figures

```bash
python tools/plot_nature_figures.py
python tools/plot_architecture.py
python tools/export_figure_data.py
```

## Regenerating Training Results

For a quick local sanity check:

```powershell
scripts\run_local_medium.cmd
```

For the 3090 fine 10%-step result, first produce or keep the warmup checkpoint expected by the script, then run:

```bash
bash scripts/run_3090_fine10_reuse_warmup.sh
```

For the 4090 7-GPU confirmation run:

```bash
bash scripts/run_4090_7gpu.sh
```

The training script saves JSON histories, masks, confusion matrices, and checkpoints when the corresponding `--save-*` flags are enabled.
