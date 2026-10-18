"""Validate Figure 3 numerical source data without rerunning experiments."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "source_data/fig3"
METADATA = Path(__file__).resolve().parents[1] / "reproduction_data/fig3/source_tables"


def read_csv(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def matches_source_hash(path, expected):
    content = path.read_bytes()
    candidates = {content}
    if path.suffix.lower() == ".csv":
        # Git may check text out with LF or CRLF; all other bytes must match.
        lf = content.replace(b"\r\n", b"\n")
        candidates.update((lf, lf.replace(b"\n", b"\r\n")))
    return any(hashlib.sha256(value).hexdigest() == expected for value in candidates)


def validate():
    for relative, key, groups, expected in [
        ("fig3.a/fig3.a_values.csv", "device_index", 5, 20),
        ("fig3.b/fig3.b_values.csv", "group_index", 4, 16),
    ]:
        rows = read_csv(relative)
        assert len(rows) == expected
        for group in range(1, groups + 1):
            selected = [r for r in rows if int(r[key]) == group]
            assert len(selected) == 4
            for column in ("experiment_normalized", "target_normalized"):
                values = np.array([float(r[column]) for r in selected])
                assert np.isfinite(values).all() and (values >= 0).all()
                np.testing.assert_allclose(values.max(), 1, atol=1e-12)
    folder = "fig3.c/"
    model = json.loads((METADATA / folder / "fig3.c_model.json").read_text(encoding="utf-8"))
    scan = read_csv(folder + "fig3.c_scan.csv")
    dense = read_csv(folder + "fig3.c_dense_fits.csv")
    assert len(scan) == 84 and len(dense) == 2400
    coefficients = []
    for name in ("reference", "dut3"):
        data = model["signals"][name]
        raw = np.asarray(data["coefficients"])
        c = raw[:, :, 0] + 1j * raw[:, :, 1]
        coefficients.append(c)
        for port in (1, 2):
            param = data["fit_summary"][port - 1]
            for rows, n in ((scan, 21), (dense, 600)):
                selected = [r for r in rows if r["dataset"] == name and int(r["port"]) == port]
                assert len(selected) == n
                x = np.array([float(r["voltage_squared_V2"]) for r in selected])
                predicted = param["k"] * abs(c[port-1, 0] * np.exp(
                    1j * model["phase_slope_rad_per_V2"] * x) + c[port-1, 1])**2 + param["offset_uA"]
                fitted = np.array([float(r["fit_uA"]) for r in selected])
                np.testing.assert_allclose(predicted, fitted, atol=1e-10, rtol=0)
                if rows is scan:
                    measured = np.array([float(r["measured_uA"]) for r in selected])
                    np.testing.assert_allclose(measured - fitted,
                        [float(r["residual_uA"]) for r in selected], atol=1e-12)
                    r2 = 1 - np.sum((measured-fitted)**2) / np.sum((measured-measured.mean())**2)
                    np.testing.assert_allclose(r2, param["R2"], atol=1e-12)

    def matrix(key):
        values = np.asarray(model[key])
        return values[:, :, 0] + 1j * values[:, :, 1]

    np.testing.assert_allclose(coefficients[1] @ np.linalg.inv(coefficients[0]),
                               matrix("dut3_matrix_raw"), atol=1e-9)
    for name in ("dut3", "sim"):
        raw = matrix(name + "_matrix_raw")
        normalized = raw / abs(raw).max() * np.exp(-1j * np.angle(raw[0, 0]))
        np.testing.assert_allclose(normalized, matrix(name + "_matrix_normalized"), atol=1e-12)
    matrices = read_csv(folder + "fig3.c_complex_matrices.csv")
    assert len(matrices) == 16
    for row in matrices:
        z = complex(float(row["real"]), float(row["imag"]))
        np.testing.assert_allclose(abs(z), float(row["amplitude"]), atol=1e-12)
        np.testing.assert_allclose(abs(z)**2, float(row["power"]), atol=1e-12)
    provenance = json.loads((METADATA / "provenance.json").read_text(encoding="utf-8"))
    for record in provenance["files"]:
        assert matches_source_hash(ROOT / record["file"], record["sha256"])
    print("PASS: Fig3.a/b final values; 84 measured currents; 2400 fitted values; 16 matrix elements; hashes.")


if __name__ == "__main__":
    validate()
