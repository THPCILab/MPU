import csv
import math
from pathlib import Path
import tempfile
import unittest

from tools.export_fig2_source_data import export, TABLES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_data/fig2"


def records(relative):
    with (DATA / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Figure2SourceTests(unittest.TestCase):
    def test_paper_data_is_data_only(self):
        for path in (ROOT / "paper_data").rglob("*"):
            if path.is_file():
                self.assertTrue(path.suffix in {".csv", ".xlsx"} or
                                path == ROOT / "paper_data/README.md", str(path))

    def test_exports_match_final_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            export(Path(tmp))
            for _, relative, _ in TABLES:
                with (DATA / relative).open(encoding="utf-8-sig", newline="") as stream:
                    final = list(csv.reader(stream))
                with (Path(tmp) / relative).open(encoding="utf-8-sig", newline="") as stream:
                    rebuilt = list(csv.reader(stream))
                self.assertEqual(final, rebuilt)

    def test_panel_shapes_and_units(self):
        grid = records("fig2.b/fig2.b_theta_phi.csv")
        self.assertEqual(len(grid), 64)
        self.assertEqual({(int(r["theta_target_code"]), int(r["phi_target_code"])) for r in grid},
                         {(t, p) for t in range(8) for p in range(8)})
        hist = records("fig2.b/fig2.b_precision_histogram.csv")
        self.assertEqual(len(hist), 14)
        for series in ("Simulation results", "Experimental results"):
            self.assertEqual(sum(int(r["device_count"]) for r in hist if r["series"] == series), 64)
        self.assertAlmostEqual(sum(float(r["simulation_precision_bits"]) for r in grid)/64,
                               7.099539804318763, places=12)
        bars = records("fig2.c/fig2.c_power_1550nm.csv")
        self.assertEqual(len(bars), 32)
        self.assertEqual({r["port"] for r in bars}, {"a11", "a12", "a21", "a22"})
        spectra = records("fig2.d/fig2.d_spectra.csv")
        self.assertEqual(len(spectra), 18003)
        for device in ("27", "57", "77"):
            rows = [r for r in spectra if r["device_id"] == device]
            self.assertEqual(len(rows), 6001)
            self.assertAlmostEqual(float(rows[0]["wavelength_nm"]), 1520)
            self.assertAlmostEqual(float(rows[-1]["wavelength_nm"]), 1580)
            at1550 = next(r for r in rows if math.isclose(float(r["wavelength_nm"]),1550,abs_tol=1e-9))
            for bar in (r for r in bars if r["device_id"] == device):
                value = float(bar["power_1550nm_db"])
                self.assertAlmostEqual(value, float(at1550[bar["port"] + "_smoothed_db"]), places=10)
        for row in bars:
            self.assertLessEqual(float(row["lower_db"]), float(row["power_1550nm_db"]))
            self.assertGreaterEqual(float(row["upper_db"]), float(row["power_1550nm_db"]))


if __name__ == "__main__":
    unittest.main()

