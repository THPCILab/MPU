# Manuscript source data

Folders and filenames follow manuscript figure panels: `fig4.c`, `fig5.a`,
`fig5.b`, `fig5.c`, `fig5.d` and `fig5.e`. This avoids the previous
analysis-only Fig1--Fig4 numbering, which did not refer to manuscript figures.

- `Source_Data.xlsx`: one worksheet per manuscript figure, with clearly
  labelled panel blocks, variable names and units.
- `fig4/`: Fig4.c confusion counts transcribed from the supplied artwork.
  Fig4.b curves are not included because the available histories have not
  been matched to the plotted traces.
- `fig5/`: archived continuous-sharing, quantization and confusion-matrix data.
- `fig5/additional_analyses/`: convergence and mask analyses, not additional
  manuscript figures.

CSV files are UTF-8, with one header row, numeric values in numeric columns and
explicit count/fraction/percentage units. Raw integer confusion counts are
retained; missing sample SD is not replaced by zero.
The workbook and CSVs have the same numerical values.

Rebuild the panel CSVs with `python tools/export_manuscript_source_data.py`.
Quantization exports are generated with `python tools/plot_fig5_quantization.py`.
Current quantitative coverage is Fig4.c and Fig5.c--e; this is not yet a complete
source-data package for every manuscript figure.
