# Changelog

## 1.0.0 — 2026-10-16

- Separate warmup and fine-tuning recovery by training phase, sharing masks,
  input tensors, initial model, codebook and optimizer configuration.
- Replace dynamic source rewriting with explicit, optional training-loop hooks.
- Handle sparse replacement-ratio grids in plotting and MATLAB exports.
- Prevent accidental use of a different seed's warmup history.
- Add a CPU-only offline demo, custom-feature example and regression tests.
- Specify dependencies, installation, expected outputs, timing and data scope.
- Preserve the archived numerical inputs and publication artwork.
