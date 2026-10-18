import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ManuscriptSourceTests(unittest.TestCase):
    def test_transcribed_counts_and_csv_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, str(ROOT / "tools/export_manuscript_source_data.py"),
                            "--out-dir", tmp], check=True, capture_output=True, text=True)
            out = Path(tmp)
            with (out / "fig4/fig4.c/fig4.c_confusion_counts.csv").open(encoding="utf-8") as stream:
                counts = list(csv.DictReader(stream))
            self.assertEqual(len(counts), 64)
            with (out / "fig4/fig4.c/fig4.c_confusion_summary.csv").open(encoding="utf-8") as stream:
                summary = list(csv.DictReader(stream))
            self.assertEqual([int(row["samples"]) for row in summary], [192, 168, 192, 168])
            self.assertEqual([int(row["correct"]) for row in summary], [164, 140, 161, 135])
            with (out / "fig5/fig5.c/fig5.c_accuracy.csv").open(encoding="utf-8") as stream:
                sharing = list(csv.DictReader(stream))
            self.assertEqual(len(sharing), 22)
            with (out / "fig5/fig5.e/fig5.e_confusion.csv").open(encoding="utf-8") as stream:
                confusion = list(csv.DictReader(stream))
            self.assertEqual(len(confusion), 2904)
            for mode in ("node_pruning", "clements_layer"):
                for task in ("Digits", "Uppercase", "Lowercase"):
                    selected = [r for r in confusion if r["mode"] == mode and r["task"] == task]
                    for label in {r["true_label"] for r in selected}:
                        row = [r for r in selected if r["true_label"] == label]
                        # Published JSON fractions are rounded to six decimal places.
                        # Retain that source precision rather than renormalizing the data.
                        self.assertLessEqual(abs(sum(float(r["row_normalized_fraction"]) for r in row) - 1),
                                             len(row) * 5.1e-7)
                        total = sum(int(r["count"]) for r in row)
                        for cell in row:
                            self.assertAlmostEqual(float(cell["row_normalized_fraction"]),
                                                   int(cell["count"]) / total, delta=5.1e-7)


if __name__ == "__main__":
    unittest.main()
