import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import figure_inputs
import export_figure_data as exporter
import plot_nature_figures as plots


class FigureInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = json.loads((ROOT / "data/main_10pct_sweep.json").read_text(encoding="utf-8-sig"))
        cls.replicate = json.loads((ROOT / "data/replicate_20pct_sweep.json").read_text(encoding="utf-8-sig"))

    def test_replicate_uses_its_own_warmup(self):
        payload = figure_inputs.prepare_payload(self.replicate, ROOT / "data/replicate_20pct_sweep.json")
        self.assertTrue(payload["efficiency_conditions"]["eta_off"]["warmup_source"].endswith("warmup_replicate.json"))

    def test_arbitrary_input_does_not_borrow_published_warmup(self):
        payload = figure_inputs.prepare_payload(self.main, ROOT / "results/custom.json")
        self.assertNotIn("warmup_source", payload["efficiency_conditions"]["eta_off"])

    def test_wrong_seed_warmup_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "seed"):
            figure_inputs.prepare_payload(self.replicate, ROOT / "custom.json", ROOT / "data/warmup_main.json")

    def test_replicate_without_90pct_plots_and_exports(self):
        payload = figure_inputs.prepare_payload(self.replicate, ROOT / "data/replicate_20pct_sweep.json")
        cond = exporter.condition(payload)
        with tempfile.TemporaryDirectory() as tmp, patch.object(plots, "save") as save:
            out = Path(tmp)
            plots.plot_main_curve(payload, out)
            plots.plot_confusion(payload, out)
            plots.plot_convergence(payload, out)
            plots.plot_patterns(payload, out)
            self.assertEqual(save.call_count, 2)
            exporter.export_fig1(cond, out, out)
            exporter.export_fig2(payload, cond, out, out)
            exporter.export_fig3(payload, cond, out, out)
            exporter.export_fig4(cond, out, out)
            mat = loadmat(out / "Fig1_Accuracy_Retention_Under_MPU_Replacement/figure_data.mat")
            self.assertNotIn("gain_at_90pct_percentage_points", mat)
            patterns = loadmat(out / "Fig4_MPU_Sharing_Patterns/figure_data.mat")
            self.assertEqual(patterns["shared_mpu_ratio_percent"].tolist(), [[20., 40., 60., 80.]])

    def test_one_ratio_no_endpoints_no_requested_even_patterns(self):
        payload = copy.deepcopy(self.main)
        cond = exporter.condition(payload)
        cond["curves"] = {"0.50": cond["curves"]["0.50"]}
        self.assertEqual(figure_inputs.pattern_ratios(cond), [50])
        with tempfile.TemporaryDirectory() as tmp, patch.object(plots, "save") as save:
            plots.plot_main_curve(payload, Path(tmp))
            plots.plot_patterns(payload, Path(tmp))
            plots.plot_patterns(payload, Path(tmp), [20, 40, 60, 80])
            self.assertEqual(save.call_count, 2)
            exporter.export_fig1(cond, Path(tmp), Path(tmp))
            exporter.export_fig4(cond, Path(tmp), Path(tmp))


if __name__ == "__main__":
    unittest.main()
