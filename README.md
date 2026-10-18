# MPU: Task-Aware Neuron-Sharing Photonic Neural Network

This work studies how a fixed, passive **meta processing unit (MPU)** can replace selected 2-to-2 MZI neurons in a multi-task photonic neural network while retaining classification accuracy. This open-source repository serves as supporting documentation for the paper “Inverse-designed meta processing units for multi-task near-field photonic computing” (submitted to *Nature Communications*).

The model projects EMNIST ByClass images to 64 PCA features, applies a complex
64-mode Clements mesh, and uses square-law intensity readout, LayerNorm and GELU
followed by task-specific classification heads. The repository includes both
continuous-operator sharing and finite-codebook quantization studies.

![Figure 1: MPU architecture](results/MPU.png)

[Figure 1 (PDF)](results/Fig1.pdf)

## Main Result

At 90% MPU sharing, the learned neuron-level replacement pattern retains substantially more multi-task accuracy than sharing a coarse layer/stage prefix.

Figure 5(c) evaluates continuous-operator sharing; panel (d) separately evaluates
finite codebooks and modeled device errors. The latter uses simulated complex
residuals and a prospective 90% MPU transmission condition.

![Figure 5: Multi-task sharing and quantization](results/main_figure.png)

[Figure 5 (PDF)](results/Fig5.pdf)

## Repository Layout

```text
.
├── train_mpu_pnn.py              # Core training and pattern-search algorithm
├── tools/
│   ├── download_emnist.py         # EMNIST ByClass downloader
│   ├── merge_shards.py            # Merge multi-GPU JSON shards
│   ├── plot_nature_figures.py     # Rebuild paper figures
│   ├── plot_architecture.py       # Draw network schematic
│   ├── export_figure_data.py      # Export figure source data as .mat files
│   ├── plot_fig5_quantization.py  # Validate and redraw manuscript Fig. 5(d)
│   └── quantization/             # Quantized engine, fixed codebook and noise model
├── scripts/
│   ├── run_local_medium.cmd       # Medium Windows training example
│   ├── run_main_10pct_sweep_reuse_warmup.sh
│   ├── run_replicate_20pct_sweep.sh
│   └── run_fig5_device_errors.py  # Fixed-codebook CUDA reproduction
├── data/                          # Final JSON data used for figures
├── results/                       # Publication figures
├── paper_data/                    # Panel-labelled CSV/MAT files and Source_Data.xlsx
└── docs/
    ├── training_strategy.md       # Continuous-sharing method
    └── fig5_reproducibility.md    # Panel mapping and quantization protocol
```

## System requirements and installation

The offline demo needs only a normal CPU computer, Python and the packages
below. No photonic chip, GPU, private data, PCA cache or checkpoint is needed.
The clean reference environment is Python 3.9.13 on Windows 10.0.26300,
PyTorch 2.0.1+cpu, torchvision 0.15.2+cpu, NumPy 1.24.4, SciPy 1.10.1,
scikit-learn 1.3.2, Matplotlib 3.7.5 and pandas 2.0.3.
Use Python 3.9–3.11 with these pinned historical dependencies, not Python 3.12+.

Clone to a short directory, such as `E:\MPU` or `~/MPU`; long Windows
environment paths can exceed the wheel installer's path-length limit.

```bash
git clone https://github.com/THPCILab/MPU.git
cd MPU
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows cmd, or
`source .venv/bin/activate` on Linux/macOS, then:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

Allow about 5–15 minutes on a broadband-connected desktop (an estimate;
download speed dominates). The requirements select CPU PyTorch wheels.
For CUDA setup, analysis-only installation and platform details, see
[Installation](docs/installation.md).

## Offline demo and expected output

```bash
python scripts/run_demo.py
python -m unittest discover -s tests -v
```

The demo validates included numerical sources, regenerates the quantization
plot, checks a 64-state model's forward/backward operations, and trains a tiny
synthetic example of the input-to-output workflow. The generated examples are
software tests, not additional EMNIST or physical measurements.
No internet access is required after dependency installation.

Expected console messages include:

```text
Validated 75 run/ratio rows and 40 plotted means from 26 source JSONs.
PASS: 64-state deployment, forward/backward, passivity and active-phase perturbation.
PASS: offline CPU demo; archived inputs unchanged.
```

Outputs are `results/demo/quantization/fig5d_quantization.pdf` and `.png`,
CSV/MAT exports in `results/demo/quantization/source_data/`,
`results/demo/training.json` and `results/demo/demo_report.json`.
The demo took about 3.3 seconds on the reference desktop (Intel i7-13700F,
32 GB RAM, CPU only); allow up to a minute on a typical laptop.
Tiny-example accuracy is not an acceptance criterion.

Tests cover phase-safe recovery, identical interrupted/resumed CPU updates,
sparse ratio grids and warmup selection. An optional Linux/Windows Python 3.10
CI template is supplied in [docs/software_checks.yml](docs/software_checks.yml).
To enable GitHub Actions, copy it to `.github/workflows/tests.yml` through
GitHub or an account authorized to manage workflows. Remote CI is not claimed
as part of the local validation.

## Own data and reproduction of paper plots

```bash
python examples/train_custom_features.py --input my_features.npz --epochs 20 --sharing-ratio 50
```

The NPZ contains `x_train`, `y_train`, `x_test` and `y_test`;
features have shape `(samples, even_feature_dimension)` and labels are integers
starting at zero. See [Custom data](docs/custom_data.md) for preprocessing,
validation and outputs. This compact single-task example is not a replacement
for the multi-task EMNIST protocol.

Rebuild numerical analysis panels and MATLAB exports without retraining:

```bash
python tools/plot_nature_figures.py --out-dir results/validation/sharing
python tools/export_figure_data.py --fig-dir results/validation/sharing --out-dir results/validation/source_data
python tools/plot_fig5_quantization.py --out-dir results/validation/quantization
```

The assembled publication artwork also contains manually composed schematic
panels. The scripts regenerate numerical panels, not every assembled panel.
See [Reproducibility](docs/reproducibility.md) and
[Fig. 5 protocols](docs/fig5_reproducibility.md) for full training and seeds.
Full EMNIST retraining is GPU-intensive; `run_local_medium.cmd` is a medium
training example, not the quick demo.

## Manuscript Source Data

Numerical sources use manuscript panel names, for example
`paper_data/fig3/fig3.a/`, `paper_data/fig3/fig3.b/`,
`paper_data/fig3/fig3.c/`, `paper_data/fig4/fig4.c/` and
`paper_data/fig5/fig5.c/`.
`paper_data/Source_Data.xlsx` has one sheet per figure and matches the
panel-labelled UTF-8 CSV files. See [coverage and provenance](paper_data/README.md).
Fig3 contains final normalized operator comparisons and measured scan points,
with model fits stored separately; see [Figure 3 data](paper_data/fig3/README.md).
Fig4.c contains counts transcribed from the supplied figure; no sample-level
predictions are reconstructed. Fig4.b curve sources are still being identified.

## Method Summary

The model first performs a 0% shared warmup, where all optical phase offsets are task-specific. The trained warmup state is then used to score each 2-to-2 photonic neuron with a Fisher/gradient-damage proxy. Neurons with small replacement damage and high cross-task agreement are selected as shared MPU candidates.

Layer-level sharing uses a coarse stage-prefix prior. Neuron-level sharing starts from the same physically meaningful stage prior and refines it by swapping individual 2-to-2 neurons, keeping the exact same MPU ratio. This isolates the benefit of fine-grained replacement.

For full equations, hyperparameters, and figure provenance, see [docs/training_strategy.md](docs/training_strategy.md).

For the Fig. 5 panel-to-data mapping, seed counts, fixed-codebook training and
reproduction commands, see [docs/fig5_reproducibility.md](docs/fig5_reproducibility.md).

## Citation

If you use this code, please cite the associated manuscript. A BibTeX entry will be added after publication.

## License, version and support

Code is distributed under the [MIT license](LICENSE); EMNIST remains subject
to its upstream dataset terms. The submission software release is
[v1.0.0](https://github.com/THPCILab/MPU/releases/tag/v1.0.0);
`CITATION.cff` supplies machine-readable software metadata.
Use [GitHub Issues](https://github.com/THPCILab/MPU/issues) for questions and
bug reports. Per-sample PCA caches and trained checkpoints are not distributed.
The [software checklist](docs/software_checklist.md) maps submission
requirements to the supplied files and validation scope.
