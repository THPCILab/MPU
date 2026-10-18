# Software submission checklist

This checklist follows the [Nature software guidelines](https://www.nature.com/documents/GuidelinesCodePublication.pdf).
The executable demonstration and computational reproduction cover **Figure 5**;
Figures 2–4 are supplied as final plotting data.

| Requirement | Repository material |
| --- | --- |
| Source and identifiable version | Git history, tagged software release and `CITATION.cff` |
| System, language and dependencies | [Installation](installation.md), pinned requirements |
| Installation and estimated time | README and installation guide |
| Demonstration with example inputs | `python scripts/run_demo.py` |
| Expected outputs and runtime | README; generated `results/demo/demo_report.json` |
| Figure 5 reproduction | [Protocols and commands](reproduction.md) |
| Own-data instructions | [Feature-data example](custom_data.md) |
| Numerical source data | [Figure-panel index](../source_data/README.md) |
| License and support | MIT `LICENSE`, GitHub Issues |

The CPU demo and regression tests run without private PCA caches, checkpoints
or photonic hardware. Complete EMNIST retraining requires CUDA and is separate
from the offline demo. The optional `software_checks.yml` is a CI template,
not evidence of a completed remote run.

For submission, identify the final commit or an archived release containing
the intended code and data. The v1.0.0 software release predates later figure-data
additions. Add an archival DOI when available, and keep the manuscript's Code
Availability statement consistent with that version and the MIT license.
Independent testing by a colleague is recommended by Nature.
