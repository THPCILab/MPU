# Code and software submission checklist

Documentation for the [Nature code/software checklist](https://www.nature.com/documents/GuidelinesCodePublication.pdf).

| Requirement | Supplied material |
| --- | --- |
| Source and version | Public repository; release v1.0.0 and its commit |
| OS, language and dependency versions | README; installation.md; requirements files |
| Hardware and resources | CPU offline demo; CUDA and public EMNIST documented separately |
| Installation and typical time | README and installation.md |
| Demo inputs and commands | Archived numerical sources; run_demo.py; generated tiny test input |
| Expected outputs and runtime | README; demo_report.json produced by each run |
| Own-data instructions | custom_data.md; examples/train_custom_features.py |
| Figure value reproduction | plot_fig5_quantization.py; run JSONs/CSVs; fig5_reproducibility.md |
| License | MIT, LICENSE |
| Feedback | GitHub Issues |

## Checks performed

- Installed the reference CPU packages in a new virtual environment.
- Ran the offline demo without private PCA caches or model checkpoints.
- Verified 75 run/ratio values, 40 plotted means and 26 underlying JSON files.
- Checked fixed-codebook deployment, finite gradients, passivity and active-phase perturbation.
- Tested recovery isolation, interrupted/resumed updates, sparse grids and matching warmup selection.
- Preserved archived numerical inputs and publication artwork.

The demo covers source reconstruction and the training API, not a new physical
experiment. Full EMNIST retraining was not repeated for these software checks.
Per-sample PCA caches and trained checkpoints are not distributed and are not
required for the offline demo or archived-curve regeneration. Fresh-data training
can vary with PCA, PyTorch, hardware and random-number implementations.

## Author actions at submission

The manuscript should cite the public version and MIT license instead of saying
that code is available only on reasonable request. Suggested wording:

> The code and numerical source data supporting the computational analyses are
> available at https://github.com/THPCILab/MPU (release v1.0.0) under the MIT
> license. Installation instructions, an offline demonstration and figure
> reproduction procedures are included in the repository.

Verify that Methods describe the model, protocol and example data consistently.
This software update does not edit manuscript files.
An independent installation by a colleague unfamiliar with the software is
recommended by Nature; it has not been claimed here.
A DOI can be added after author-controlled archival deposition. Until then
the versioned release and commit identify the submitted software.
