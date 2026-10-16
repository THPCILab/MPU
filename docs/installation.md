# Installation and system requirements

## Tested environment

A new virtual environment without access to the old environment's site-packages
was installed and tested on 16 October 2026.

| Component | Version |
| --- | --- |
| OS reported by Python | Windows-10-10.0.26300-SP0 |
| Python | 3.9.13, 64 bit |
| PyTorch / torchvision | 2.0.1+cpu / 0.15.2+cpu |
| NumPy / SciPy | 1.24.4 / 1.10.1 |
| scikit-learn | 1.3.2 |
| Matplotlib / pandas | 3.7.5 / 2.0.3 |
| Desktop | Intel Core i7-13700F, 32 GB RAM |

No GPU or nonstandard hardware is required for the offline demo. A few GB of
free disk space suffice for the Python environment and repository, excluding
EMNIST. A Python 3.10 Ubuntu 22.04/Windows CI template is supplied in
software_checks.yml; these remote jobs have not been run as part of the local
checks. macOS is not locally
validated. The historical pinned package set targets Python 3.9–3.11.
It is a reproducibility baseline, not a claim of compatibility with every newer
dependency version.

## CPU installation

From a short directory:

```bash
git clone https://github.com/THPCILab/MPU.git
cd MPU
python -m venv .venv
```

Activate with `.venv\Scripts\activate` (Windows cmd),
`.\.venv\Scripts\Activate.ps1` (PowerShell), or
`source .venv/bin/activate` (Linux/macOS).
If PowerShell blocks activation, call `.venv\Scripts\python.exe` directly;
there is no need to change system execution policy.

```bash
python -m pip install -r requirements.txt
python -m pip check
python scripts/run_demo.py
```

Typical installation time is estimated at 5–15 minutes, including downloads on
a broadband-connected desktop. The largest tested Windows CPU wheel is about
175 MB. Very long Windows environment paths can cause wheel extraction to fail;
create the environment under a short path instead of editing the registry.

For plotting and numerical-source validation without PyTorch:

```bash
python -m pip install -r requirements-analysis.txt
python tools/plot_fig5_quantization.py --validate-only
```

## CUDA retraining

Full training needs an NVIDIA GPU and compatible driver; no photonic hardware
or commercial FDTD solver is needed. Use a separate environment. For PyTorch
2.0.1 / CUDA 11.8, follow the
[official version-specific instructions](https://pytorch.org/get-started/previous-versions/#v201):

```bash
python -m pip install -r requirements-analysis.txt
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python scripts/run_fig5_device_errors.py --forward-smoke
```

Do not then install the CPU requirements into the CUDA environment.
The local GPU forward smoke was also checked with the pre-existing
PyTorch 2.0.0+cu118 environment. Full 80+80-epoch EMNIST retraining was not
repeated in these software checks. Reserve many hours or longer for full sweeps;
runtime depends strongly on GPU, samples, conditions and PCA preparation.
The main shell launcher uses four GPUs (IDs 0–3); the replicate launcher uses
seven (IDs 0–6). For one GPU, use individual commands in the reproducibility
guide rather than the multi-GPU launchers.

## Local data and safety

Fresh EMNIST inputs are downloaded and processed locally. Per-sample PCA caches,
intermediate models and recovery checkpoints are excluded by .gitignore.
PyTorch files can contain pickle data: load only files you created or trust.
Keep separate output directories for different experiments.
