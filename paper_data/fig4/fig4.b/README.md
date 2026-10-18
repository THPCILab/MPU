# Figure 4b: training curves

The two CSV files contain the final numerical histories used by the source
plotting cells. Values are copied from the supplied data export, without
smoothing, interpolation or retraining.

- `fig4.b_single_task.csv`: 300 records for MZI, MZI (3-bit), MPU and MPU+MZI.
  Each configuration has training accuracy (%) and dimensionless loss columns.
- `fig4.b_dual_task.csv`: 300 records for Tasks 1 and 2, each with training
  accuracy (%) and dimensionless loss.
- `../../../data/fig4/fig4.b_training_history.json` in the repository: original
  history arrays, with accuracy stored as a fraction between 0 and 1.
- `provenance.json`: source identifiers, plot-cell indices and SHA-256 hashes.

The epoch column is 10, 20, ..., 3000. These are training-set histories, not
independent test accuracies or optical power measurements. The Fig4.c confusion
counts are provided separately and are not reconstructed from these curves.

## Field mapping

| Figure label | Original JSON key | Accuracy / loss fields |
| --- | --- | --- |
| MZI | `mzi_2layer_history` | `acc`, `loss` |
| MZI (3-bit) | `mzi_2layer_q3_history` | `acc`, `loss` |
| MPU | `dpu_history` | `acc`, `loss` |
| MPU+MZI | `dpu_mzi_single_history` | `acc`, `loss` |
| Task 1 | `dpu_mzi_multi_history` | `acc1`, `loss1` |
| Task 2 | `dpu_mzi_multi_history` | `acc2`, `loss2` |

The original `dpu` identifier is retained in the JSON; the figure label is MPU.
The histories follow the JSON read by the final plotting cells, rather than
older notebook console outputs. A run seed is not assigned where it is not
recorded. These arrays support the plotted trajectories, not a new independent
training-reproduction study.

The 6.8-percentage-point annotation compares the peaks of MPU+MZI and MPU over
their respective histories, not their values at the same epoch or a relative
percentage gain. All early loss values are retained, including those above the
0--5 display range in the dual-task plot.

## Export and redraw

From the repository root:

```bash
python tools/export_manuscript_source_data.py
python tools/plot_fig4_training.py --validate-only
python tools/plot_fig4_training.py
```

The plot command writes PDF, PNG and SVG previews under `results/fig4_training/`,
including a full-range loss view. It does not train a model. The `Fig4` worksheet
of `../../Source_Data.xlsx` contains the same CSV values in labelled blocks.
