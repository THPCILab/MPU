# Fig5.d

Quantization and modeled device-error training results.

- `fig5.d_runs.csv`: 75 run/ratio records, including training seed and source JSON.
- `fig5.d_summary.csv`: 40 plotted means, sample SD and training-seed counts.
- `fig5.d_source_data.mat`: the same panel-level numerical arrays.

Accuracies are percentages. This panel uses final-epoch results, not the
best-epoch statistic of Fig5.c. A blank sample SD means that a condition
contains only one training seed; it is not zero uncertainty.
The five active-control noise draws evaluate one trained model and are not
independent training runs. See docs/fig5_reproducibility.md.
