from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import train_custom_features as custom


class CustomDataTests(unittest.TestCase):
    def test_demo_schema_and_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "toy.npz", Path(tmp) / "training.json"
            custom.write_demo(source)
            values, dim, classes = custom.load_features(source)
            self.assertEqual((dim, classes), (8, 2))
            report = custom.run(source, output, epochs=1)
            self.assertTrue(output.exists())
            self.assertEqual(report["training_samples"], 32)
            self.assertEqual(len(report["finetune"]["mean_hist"]), 1)

    def test_complex_features_and_unseen_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "invalid.npz"
            base = dict(x_train=np.ones((4, 4)), x_test=np.ones((2, 4)),
                        y_train=np.array([0, 1, 0, 1]), y_test=np.array([0, 1]))
            bad = dict(base, x_train=base["x_train"].astype(complex))
            np.savez(source, **bad)
            with self.assertRaisesRegex(ValueError, "finite"):
                custom.load_features(source)
            np.savez(source, **dict(base, y_test=np.array([0, 2])))
            with self.assertRaisesRegex(ValueError, "unseen"):
                custom.load_features(source)


if __name__ == "__main__":
    unittest.main()
