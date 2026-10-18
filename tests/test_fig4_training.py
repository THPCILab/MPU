import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.plot_fig4_training import validate

ROOT = Path(__file__).resolve().parents[1]


class Figure4TrainingTests(unittest.TestCase):
    def test_histories_and_regenerated_exports(self):
        single, dual = validate()
        self.assertEqual(len(single["epoch"]), 300)
        self.assertEqual(len(dual["epoch"]), 300)
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT / "tools/export_manuscript_source_data.py"),
                            "--out-dir", tmp], check=True, capture_output=True, text=True)
            for name in ("fig4.b_single_task.csv", "fig4.b_dual_task.csv"):
                with (Path(tmp) / "fig4/fig4.b" / name).open(encoding="utf-8-sig", newline="") as stream:
                    regenerated = list(csv.reader(stream))
                with (ROOT / "source_data/fig4/fig4.b" / name).open(encoding="utf-8-sig", newline="") as stream:
                    supplied = list(csv.reader(stream))
                self.assertEqual(regenerated, supplied)


if __name__ == "__main__":
    unittest.main()
