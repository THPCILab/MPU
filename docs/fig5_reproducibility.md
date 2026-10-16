# Figure 5: source data and reproducibility

## Panel mapping

- **(a)** Architecture diagram; no numerical dataset.
- **(b)** Sharing-selection diagram and illustrative node masks. The complete
  archived exact-node masks are retained in the continuous-sharing JSON.
  This repository does not infer masks by tracing the illustration.
- **(c)** Continuous-sharing accuracy curves:
  `data/main_10pct_sweep.json`, comparing node and stage allocation.
- **(d)** Quantization and modeled device-error curves:
  `data/fig5/quantization_runs.csv`; underlying per-run JSONs are in
  `nominal_runs/` and `device_runs/`.
- **(e)** Confusion matrices at 90% continuous sharing:
  `data/main_10pct_sweep.json`.

`paper_data/Fig5_Sharing_and_Quantization/figure_data.mat` contains the numerical
curves for panels (c,d) and the confusion matrices for panel (e). The run-level
CSV retains the selected metric and seed; the summary CSV retains sample SD and
training-seed count. The existing Fig1--Fig4 export folders are individual
analysis plots, not the manuscript figure numbers.

## Rebuild and verify panel (d)

Run from the repository root:

```bash
pip install -r requirements.txt
python tools/plot_fig5_quantization.py --validate-only
python tools/plot_fig5_quantization.py
python tools/check_quantization_model.py
```

The validation recomputes all 75 run/ratio values from 26 JSON files and checks
all 40 plotted means. It also verifies passivity, power transmission and the
residual level of all 64 codewords. The PDF/PNG are written to
`results/analysis_panels/fig5d_quantization.*`; publication artwork remains
`results/Fig5.pdf`.

Continuous, 3-bit QAT and 4-bit QAT use seeds 42, 43 and 44. The 5-bit QAT
curve uses seeds 42 and 44; 2-bit QAT and 3-bit PTQ use seed 44.
These curves use final-epoch accuracy, whereas panel (c) uses the archived
best-epoch metric. They answer different questions and should not be combined
as one training protocol.

## Fixed-codebook and active-control error model

The deployed table has 64 nominal 3-bit theta--phi states. One fixed complex
matrix residual is assigned to each state and retained throughout training and
evaluation. After removal of common attenuation, its relative Frobenius error
is `2**(-3.32)`. Common field attenuation is `sqrt(0.90)`; all codewords are
passive and their average two-input power transmission is 90%.

The measured 3.32-bit statistic is amplitude RMSE after one fitted common
field multiplier per device. It motivates the modeled residual scale; it does
not supply measured complex residual directions or a measured phase-error
population. The 90% transmission is a prospective low-loss condition.

The two device-error curves use the same seed-44, epoch-80 checkpoints and
fixed codebook. The second adds 10 mV RMS-equivalent active-MZI uncertainty,
converted to `pi*0.010/sqrt(1.78) = 0.0235472` rad per theta/phi shifter.
Five evaluation-noise draws use seeds 441000--441004. They are evaluations
of one trained model, not five training seeds. No extra fine-tuning is applied.

## Training code and inputs

- `tools/quantization/engine.py`: quantized 64-mode Clements model, hard
  forward quantization with straight-through gradients, scoring and training.
- `train_device_codebook.py`: fixed passive table deployment.
- `paired_voltage_test.py`: same-checkpoint active-MZI perturbation.
- `feature_cache.py`: optional local PCA inputs and loader RNG state.
- `checkpoint_training.py`: explicit recovery hooks without changing update order.

Per-sample PCA caches and model checkpoints are not distributed. Retraining
downloads the public EMNIST ByClass dataset and prepares 40,000 training and
12,000 held-out examples per task, with 64 PCA features per example.
Warmup checkpoints and all new training outputs remain local and are ignored
by Git. An optional trusted local cache/checkpoint can be supplied.

On a CUDA-enabled machine:

```bash
python scripts/run_fig5_device_errors.py --forward-smoke
python scripts/run_fig5_device_errors.py --ratio 90
```

Repeat the second command for ratios 20, 40, 60 and 80. Training outputs go
to a separate `results/quantization_training/` tree; archived figure data
are not overwritten. Without a local warmup, each ratio first trains an
80-epoch all-active warmup. To reuse locally generated inputs:

```bash
python scripts/run_fig5_device_errors.py --ratio 90 --feature-cache /path/to/local_pca_features.pt --warmup-checkpoint /path/to/local_warmup.pt
```

Load trusted PyTorch files only. Full retraining is GPU-intensive and can vary
with CUDA, PyTorch, PCA solver/thread settings and numerical-library versions.
The fresh-input training route reproduces the procedure, not an exact replay
of the archived curves. Replotting and source-data validation reproduce the
published numerical curves directly and require no retraining.

Recovery files now have separate warmup and fine-tuning namespaces. Their keys
include masks, data tensors, initial model, codebook and optimizer settings.
An interrupted phase can resume its saved epoch without overwriting the next
phase's masks. Old phase-ambiguous recovery files are not loaded by this format;
retain them locally for reference, or start a new work directory.

## Archive origin

Machine-local paths and historical run identifiers remain inside the unchanged
JSON records to preserve source provenance, not in the public-facing directory
layout. SHA-256 values for plot inputs are exported
to `paper_data/Fig5_Sharing_and_Quantization/source_hashes.json`.
