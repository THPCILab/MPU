# Manuscript Figure 5 source data

`figure_data.mat` stores panel (c) continuous-sharing curves, panel (d)
quantization and device-error curves, and panel (e) confusion matrices.
`panel_d_runs.csv` contains the 75 underlying run/ratio values;
`panel_d_summary.csv` contains the 40 plotted means and training-seed counts.

Panel (a) is a schematic. Panel (b) is an explanatory sharing illustration;
the original run masks remain in `data/main_10pct_sweep.json`.
The publication PDF and PNG are included as visual references.

Rebuild with `python tools/plot_fig5_quantization.py`.
See `docs/fig5_reproducibility.md` for the metric, protocol and seed definitions.
