# MPU: Task-Aware Neuron-Sharing Photonic Neural Network

This project implements training, task-aware node selection and evaluation for
multi-task MPU–MZI photonic neural networks. It accompanies the paper
**Inverse-designed meta processing units for multi-task near-field photonic computing**.

A fixed, passive **meta processing unit (MPU)** replaces selected two-mode MZI
nodes with shared optical operators, while the remaining programmable MZIs retain
task-specific flexibility. The computational model projects EMNIST ByClass to
64 PCA features, applies a complex 64-mode Clements mesh and square-law intensity
readout, then uses shared LayerNorm/GELU and task-specific classification heads.

![Figure 1: MPU–MZI architecture](results/MPU.png)

[Figure 1 PDF](results/Fig1.pdf)

## Main result

In the continuous-operator study, at the same MPU replacement budget,
task-aware neuron-level sharing preserves
more multi-task accuracy than coarse stage-level sharing, including at 90%
replacement. The project includes continuous-operator sharing, finite-codebook
QAT/PTQ comparisons and modeled device-error evaluation.

![Figure 5: Multi-task sharing and quantization](results/main_figure.png)

[Figure 5 PDF](results/Fig5.pdf)

## Training workflow

1. Prepare EMNIST inputs using PCA fitted on the training split.
2. Train an all-active, task-private warmup model.
3. Select shared nodes using cross-task agreement and a Fisher-style replacement
   score, keeping the number of shared nodes fixed.
4. Fine-tune, evaluate each task and compare sharing/quantization strategies.

See [training strategy](docs/training_strategy.md) for equations and selection details.

## Quick start

Use Python 3.9–3.11. The reference CPU environment is Windows, Python 3.9.13 and
PyTorch 2.0.1; dependencies are pinned in `requirements.txt`.
No GPU, photonic hardware, private feature cache or checkpoint is needed for the demo.

```bash
git clone https://github.com/THPCILab/MPU.git
cd MPU
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows cmd or
`source .venv/bin/activate` on Linux/macOS, then:

```bash
python -m pip install -r requirements.txt
python scripts/run_demo.py
python -m unittest discover -s tests -v
```

Installation typically takes 5–15 minutes; the CPU demo takes less than a minute.
A successful demo ends with `PASS: offline CPU demo; archived inputs unchanged.`
It writes plots, numerical exports and a small training example to `results/demo/`.
The small example exercises the training API rather than reproducing an EMNIST benchmark.
See [installation](docs/installation.md) for PowerShell and CUDA setup.

## Reproduce the computational results

Regenerate the Figure 5 numerical panels from the included results:

```bash
python tools/plot_nature_figures.py
python tools/plot_fig5_quantization.py
python tools/export_figure_data.py
```

Outputs go to `results/generated/fig5/`; supplied figures and source tables
remain unchanged. For full continuous-sharing retraining on a suitable multi-GPU machine:

```bash
bash scripts/run_main_10pct_sweep_reuse_warmup.sh
```

The medium Windows example is `scripts\run_local_medium.cmd`.
Full EMNIST training requires CUDA and can take many hours.
[Reproduction instructions](docs/reproduction.md) describe panel mapping,
training protocols, finite-codebook deployment and individual-ratio commands.

To train on your own feature data:

```bash
python examples/train_custom_features.py --input my_features.npz --epochs 20 --sharing-ratio 50
```

See [own-data instructions](docs/custom_data.md) for input shapes and preprocessing.

## Project layout

```text
train_mpu_pnn.py    Core multi-task model and node-selection training
tools/             Quantization models, plotting and data export
scripts/           Offline demo and training launchers
examples/          Custom-feature training example
data/              Archived numerical inputs and model parameters
paper_data/        Final plotting tables and Source_Data.xlsx
results/           Publication figures; generated outputs are Git-ignored
tests/             Regression and source-data checks
docs/              Installation, method and reproduction details
```

The executable installation, demo and reproduction pipeline covers **Figure 5**.
**Figures 2–4** are supplied as final plotting data, with a concise
[panel index](paper_data/README.md). The Figure 5 device-error study uses a
prospective 90% MPU-transmission model; it is distinct from physical measurements.

## License, citation and support

Code is distributed under the [MIT license](LICENSE); EMNIST retains its
upstream dataset terms. Cite the associated paper and identify the software
by a tagged release or commit. [CITATION.cff](CITATION.cff) supplies metadata;
[v1.0.0](https://github.com/THPCILab/MPU/releases/tag/v1.0.0) is the archived
software release, preceding later source-data additions.
Questions and bug reports: [GitHub Issues](https://github.com/THPCILab/MPU/issues).
Per-sample PCA caches and trained checkpoints are not distributed.
The [software checklist](docs/software_checklist.md) maps submission requirements
to the supplied project materials.
