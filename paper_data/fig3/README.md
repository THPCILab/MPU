# Figure 3 source data

The CSVs contain the final numerical values used for the plotted comparisons.
They are grouped by manuscript panel; values are not inferred from figure pixels.

## Fig3.a: non-unitary operators

`fig3.a/fig3.a_values.csv` contains 20 normalized experimental/target pairs:
five devices, with elements ordered a11, a12, a21, a22. Experimental and target
distributions are each normalized to their own maximum for each device.
These dimensionless comparisons do not retain absolute transmission.

## Fig3.b: cascade comparison

`fig3.b/fig3.b_values.csv` contains the 16 final experimental/target pairs in
four plotted groups. Each group is normalized independently to its maximum.
`group_index`, `element_index` and `notebook_label` preserve the figure order;
they do not establish a separately verified physical input/output mapping.
The comparison describes normalized power distributions, not absolute throughput
or full-complex matrix fidelity. The FDTD field-map array and the schematics
are not included in this handoff.

## Fig3.c: reference scan and representative MPU response

- `fig3.c_scan.csv`: 84 measured currents, corresponding fitted currents and
  residuals; two datasets, two output ports and 21 scan settings per channel.
  Current is in microamperes; voltage and voltage squared are in V and V².
- `fig3.c_fit_parameters.csv`: saved fit parameters and fit-quality metrics.
- `fig3.c_dense_fits.csv`: 600 model evaluations per channel, 2,400 total;
  these are fitted curves, not additional measurements.
- `fig3.c_complex_matrices.csv`: experimental-fit and simulation matrices,
  before and after the adopted normalization. Polar radius is field amplitude;
  its square is power. Angles are provided in radians and degrees.
- `fig3.c_model.json`: numerical model parameters and matrix values.

The fitted current model is I(x) = k |A exp(i beta x) + B|² + offset,
where x = V² and beta = pi/1.78 rad/V². With coefficient matrices N formed
from A and B, the fitted operator is N_DUT inv(N_reference). Experimental and
simulation operators are each normalized by their own maximum amplitude and
rotated to set the a11 phase to zero. Independent intensity detection leaves
output-row phase gauges unconstrained; the complex comparison uses the adopted
model convention, rather than uniquely identifying every absolute matrix phase.
Saved simulation target values include the source plotting convention.

`dut3` identifies the representative device in the source records, not a third
independent repetition. Negative currents and offsets are retained.

## Access and validation

The `Fig3` worksheet in `../Source_Data.xlsx` contains the same numerical tables.
CSV column names encode quantities and units; normalized values are fractions,
not percentages. `provenance.json` records source identifiers and SHA-256 hashes.
Run `python tools/validate_fig3_source_data.py` from the repository root to check
the published tables, model evaluations, matrix relationships and file integrity.
