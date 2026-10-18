# Fig4.c

Integer confusion counts for Tasks A and B, separately for training and test.
Values are transcribed from the supplied figure, not recovered sample-level
prediction records. Class label case is preserved.

- `fig4.c_confusion_counts.csv`: one row per matrix cell (64 rows).
- `fig4.c_confusion_summary.csv`: totals and accuracies calculated from the counts.
- `accuracy_percent` retains the computed precision;
  `displayed_accuracy_percent` retains the rounded figure annotation.

Each training matrix contains 192 examples; each test matrix contains 168.
Rebuild with `python tools/export_manuscript_source_data.py`.
