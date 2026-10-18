# MPU: code and source data

Supporting materials for **Inverse-designed meta processing units for multi-task
near-field photonic computing**.

## Scope

- **Figure 5:** computational model, training code, archived numerical results,
  installation instructions and an offline demonstration.
- **Figures 2–4:** final plotting values in `paper_data/`, matched to the
  manuscript panels. These tables are not an experimental-platform reproduction package.

Figure 5 distinguishes continuous-operator sharing from finite-codebook
quantization and modeled device errors. The error study assumes prospective
90% per-MPU transmission; it is a software study, not a measured 64-mode network.

![Figure 5](results/main_figure.png)

## Quick start: Figure 5

Use Python 3.9–3.11. The reference environment is Windows, Python 3.9.13 and
CPU PyTorch 2.0.1; exact dependencies are pinned in `requirements.txt`.
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

Allow approximately 5–15 minutes for installation and less than a minute for
the CPU demo. A successful run ends with
`PASS: offline CPU demo; archived inputs unchanged.`
Plots, CSV/MAT exports and a small training example are written to
`results/demo/`. The small example tests the training API, not a paper benchmark.
See [installation](docs/installation.md) for PowerShell and CUDA instructions.

## Reproduce Figure 5 numerical panels

```bash
python tools/plot_nature_figures.py
python tools/plot_fig5_quantization.py
```

Outputs go to `results/generated/fig5/`; the supplied publication artwork and
source-data tables remain unchanged. Retraining on public EMNIST requires a
CUDA GPU and substantially more time.
See [reproduction](docs/reproduction.md) for panel mapping, protocols and commands.

## Repository guide

| Location | Contents |
| --- | --- |
| `paper_data/` | Final CSV tables, figure-panel index and `Source_Data.xlsx` |
| `data/` | Archived numerical inputs and model parameters |
| `train_mpu_pnn.py`, `tools/quantization/` | Figure 5 computational models |
| `scripts/`, `tools/` | Demo, training, plotting and export commands |
| `results/` | Supplied publication artwork; generated outputs are ignored by Git |
| `tests/` | Regression and source-data checks |

[Figure-panel index](paper_data/README.md) ·
[Training method](docs/training_strategy.md) ·
[Own-data example](docs/custom_data.md) ·
[Software checklist](docs/software_checklist.md)

## License, citation and support

Code is released under the [MIT license](LICENSE); EMNIST retains its upstream
dataset terms. Cite the associated paper and identify the software by a tagged
release or commit. [CITATION.cff](CITATION.cff) provides software metadata;
[v1.0.0](https://github.com/THPCILab/MPU/releases/tag/v1.0.0) is the archived
software release, preceding subsequent source-data additions.
Questions and bug reports: [GitHub Issues](https://github.com/THPCILab/MPU/issues).
Per-sample PCA caches and trained checkpoints are not distributed.
