# Reproducing Figure 5

Installation, the offline demo and model retraining concern Figure 5.
Figures 2–4 are supplied as final plotting tables in
[`paper_data/`](../paper_data/README.md).

## Panel mapping

| Panel | Numerical input or code |
| --- | --- |
| 5a | Architecture schematic in the assembled `results/Fig5.pdf` |
| 5b | Node-selection schematic; archived masks in `data/main_10pct_sweep.json` |
| 5c | Continuous-sharing curves in `data/main_10pct_sweep.json` |
| 5d | `data/fig5/quantization_runs.csv`, `nominal_runs/` and `device_runs/` |
| 5e | 90%-sharing confusion matrices in `data/main_10pct_sweep.json` |

The supplied `results/Fig5.pdf` and `results/main_figure.png` are assembled
publication artwork. Scripts regenerate numerical panels, not the manually
composed schematics. Panel 5c uses archived best-epoch test accuracy;
panel 5d uses final-epoch test accuracy.

## Replot without retraining

From the repository root, after [installation](installation.md):

```bash
python tools/plot_nature_figures.py
python tools/plot_fig5_quantization.py
python tools/export_figure_data.py
```

Outputs are under `results/generated/fig5/`:
`fig5c_accuracy.*`, `fig5d_quantization.*`, `fig5e_confusion.*`,
additional convergence/mask plots and optional MATLAB exports in `source_data/`.
Generated files are not tracked by Git.

Validate the quantization sources without writing outputs:

```bash
python tools/plot_fig5_quantization.py --validate-only
python tools/check_quantization_model.py
```

The source check recomputes 75 run/ratio values and 40 means from 26 JSONs.
Continuous, 3-bit QAT and 4-bit QAT use training seeds 42, 43 and 44;
5-bit QAT uses 42 and 44; 2-bit QAT and 3-bit PTQ use 44.

For a separate continuous-sharing input, both plotting and MATLAB export accept
`--input path/to/run.json --warmup-input path/to/matching_warmup.json`.
Missing ratio panels are skipped rather than replaced by another ratio.

## Model and training protocol

The Figure 5 model applies training-fitted 64-feature PCA to EMNIST ByClass,
then a complex 64-mode Clements mesh, intensity readout, LayerNorm/GELU and
task-specific heads. The tasks are digits, uppercase letters and lowercase
letters (10, 26 and 26 classes). Shared nodes use common operators; unshared
nodes retain task-private controls. See [training strategy](training_strategy.md)
for scoring and fixed-budget selection.

Continuous-sharing launchers use 40,000 training and 12,000 held-out examples
per task, 80 warmup epochs and 80 fine-tuning epochs. The main launcher uses
four GPUs; the independent replicate uses seven. Run these only on a suitable
CUDA machine:

```bash
bash scripts/run_main_10pct_sweep_reuse_warmup.sh
bash scripts/run_replicate_20pct_sweep.sh
```

The archived main and replicate curves have separate warmup files
(`data/warmup_main.json` and `data/warmup_replicate.json`).
New outputs are written under `results/`, not over the archived inputs.

## Finite-codebook and device-error study

`tools/quantization/` implements QAT/PTQ, fixed-codebook deployment, active-control
perturbations and phase-separated checkpoint recovery.
The 64-state table assigns a fixed complex residual to each codeword,
with relative Frobenius scale `2**(-3.32)` after removing common attenuation.
Its mean two-input power transmission is 90%. The measured 3.32-bit
amplitude statistic motivates the modeled scale; measured complex residual
directions and phase-error distributions are not inferred from it.

The second device-error curve adds 10-mV RMS-equivalent MZI drive uncertainty,
converted to `pi * 0.010 / sqrt(1.78)` radians per shifter.
Both curves evaluate the same seed-44, epoch-80 models. Five evaluation-noise
draws are not five independently trained models.

For one replacement ratio, after [CUDA installation](installation.md):

```bash
python scripts/run_fig5_device_errors.py --forward-smoke
python scripts/run_fig5_device_errors.py --ratio 90
```

Repeat the training command for 20, 40, 60 and 80. Without a local warmup,
each condition first trains an 80-epoch all-active model. Optional trusted
local inputs can be passed with `--feature-cache` and `--warmup-checkpoint`.
Do not load untrusted PyTorch files.

Full retraining can take many hours and can vary with CUDA, PCA and numerical
library versions. Replotting reproduces the archived curves directly;
fresh training reproduces the documented procedure, not identical trajectories.
Per-sample PCA caches and trained checkpoints remain local.
