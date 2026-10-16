"""Tiny deterministic regression tests; these data are not research results."""
import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/quantization"))
import checkpoint_training as recovery
import engine


class TinyModel(torch.nn.Module):
    def __init__(self, shared=0):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.register_buffer("shared_mask_even", torch.tensor([shared], dtype=torch.float32))

    def forward(self, x, tid):
        return self.linear(x)


def setup_run(shared=0):
    engine.set_seed(17)
    model = TinyModel(shared)
    dataset = TensorDataset(torch.randn(8, 4), torch.tensor([0, 1] * 4))
    train = {0: DataLoader(dataset, batch_size=4, shuffle=True,
                           generator=torch.Generator().manual_seed(55))}
    test = {0: DataLoader(dataset, batch_size=4)}
    return model, train, test


def train(exp, model, train_loaders, test_loaders, phase):
    return exp.train_model(model, train_loaders, test_loaders, [0], {"0": 1.0},
                           2, .0035, 1.0, .0001, "cpu", capture_best_state=True,
                           track_train_accuracy=True, training_phase=phase)


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_exp(self, root):
        exp = types.SimpleNamespace(train_model=engine.train_model)
        recovery.install(exp, root, 40, [], {}, ROOT / "README.md")
        return exp

    def test_warmup_and_finetune_are_separate(self):
        with tempfile.TemporaryDirectory() as root:
            exp = self.make_exp(root)
            model, loaders, test = setup_run(0)
            train(exp, model, loaders, test, "warmup")
            model, loaders, test = setup_run(1)
            result = train(exp, model, loaders, test, "finetune")
            self.assertEqual(len(result["mean_hist"]), 2)
            self.assertEqual(model.shared_mask_even.item(), 1)
            paths = list((Path(root) / "recovery_checkpoints").glob("*.pt"))
            self.assertEqual(len(paths), 2)
            self.assertEqual({p.name.split("_")[0] for p in paths}, {"warmup", "finetune"})

    def test_interrupted_resume_matches_uninterrupted(self):
        model, loaders, test = setup_run(1)
        expected = train(types.SimpleNamespace(train_model=engine.train_model),
                         model, loaders, test, "finetune")
        expected_state = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as root:
            exp = self.make_exp(root)
            original_save = recovery.EpochRecovery.save

            def interrupt(session, *args):
                original_save(session, *args)
                raise InterruptedError("Simulated process interruption after epoch 1")

            model, loaders, test = setup_run(1)
            with patch.object(recovery.EpochRecovery, "save", interrupt):
                with self.assertRaises(InterruptedError):
                    train(exp, model, loaders, test, "finetune")
            model, loaders, test = setup_run(1)
            actual = train(exp, model, loaders, test, "finetune")
            for key, value in expected_state.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
            for key in ("mean_hist", "train_loss_hist", "train_mean_hist", "task_hist"):
                self.assertEqual(expected[key], actual[key])

    def test_tampered_masks_are_rejected_before_restore(self):
        with tempfile.TemporaryDirectory() as root:
            exp = self.make_exp(root)
            model, loaders, test = setup_run(1)
            train(exp, model, loaders, test, "finetune")
            path = next((Path(root) / "recovery_checkpoints").glob("*.pt"))
            saved = torch.load(path, weights_only=False)
            saved["model"]["shared_mask_even"].zero_()
            torch.save(saved, path)
            model, loaders, test = setup_run(1)
            with self.assertRaisesRegex(RuntimeError, "Saved recovery masks"):
                train(exp, model, loaders, test, "finetune")
            self.assertEqual(model.shared_mask_even.item(), 1)


if __name__ == "__main__":
    unittest.main()
