# Changelog

## Unreleased

- Add final plotting tables for manuscript Figures 2–4 and the source-data workbook.
- Clarify that installation, demo and computational reproduction concern Figure 5.
- Remove duplicate preview/MAT exports; generate them under ignored results directories.
- Use manuscript panel identifiers for regenerated Figure 5 plots.

## 1.0.0 — 2026-10-16

- Separate warmup and fine-tuning recovery by training phase, sharing masks,
  input tensors, initial model, codebook and optimizer configuration.
- Replace dynamic source rewriting with explicit, optional training-loop hooks.
- Handle sparse replacement-ratio grids in plotting and MATLAB exports.
- Prevent accidental use of a different seed's warmup history.
- Add a CPU-only offline demo, custom-feature example and regression tests.
- Specify dependencies, installation, expected outputs, timing and data scope.
- Preserve the archived numerical inputs and publication artwork.
