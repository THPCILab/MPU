# Manuscript Figure 5 source data

`fig5_source_data.mat` stores panel (c) continuous-sharing curves, panel (d)
quantization and device-error curves, and panel (e) confusion matrices.
`fig5.d/fig5.d_runs.csv` contains the 75 underlying run/ratio values;
`fig5.d/fig5.d_summary.csv` contains the 40 plotted means and training-seed counts.

Panel (a) is a schematic. Panel (b) is an explanatory sharing illustration;
the original run masks remain in `data/main_10pct_sweep.json`.
The publication PDF and PNG are included as `fig5.pdf` and `fig5.png`.
Folders `fig5.a` through `fig5.e` follow manuscript panel labels.
Other convergence and mask analyses are in `additional_analyses/`.

Rebuild with `python tools/plot_fig5_quantization.py`.
See `docs/fig5_reproducibility.md` for the metric, protocol and seed definitions.
