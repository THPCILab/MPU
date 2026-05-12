# Reproducibility Notes

## Final Figure Data

The final paper figures are generated from:

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
```

## Regenerating Training Results

For a quick local sanity check:

```powershell
scripts\run_local_medium.cmd
```

For the main 10%-step result, first produce or keep the warmup checkpoint expected by the script, then run:

```bash
bash scripts/run_main_10pct_sweep_reuse_warmup.sh
```

For an independent 20%-step confirmation run:

```bash
bash scripts/run_replicate_20pct_sweep.sh
```

The training script saves JSON histories, masks, confusion matrices, and checkpoints when the corresponding `--save-*` flags are enabled.

