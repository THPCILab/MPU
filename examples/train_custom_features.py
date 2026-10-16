"""Small CPU training example for user-provided real feature vectors.

The optional generated data test software plumbing only, not paper performance.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_mpu_pnn as engine


def write_demo(path):
    rng = np.random.default_rng(44)
    values = {}
    for split, count in (("train", 32), ("test", 16)):
        x = rng.standard_normal((count, 8)).astype(np.float32)
        values["x_" + split] = x
        values["y_" + split] = (x[:, 0] + x[:, 1] > 0).astype(np.int64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)


def load_features(path):
    with np.load(path, allow_pickle=False) as data:
        values = {key: data[key].copy() for key in ("x_train", "y_train", "x_test", "y_test")}
    dim = values["x_train"].shape[1] if values["x_train"].ndim == 2 else 0
    if dim < 2 or dim % 2:
        raise ValueError("Features need an even dimension of at least 2")
    for split in ("train", "test"):
        x, y = values["x_" + split], values["y_" + split]
        if (x.ndim != 2 or x.shape[1] != dim or not len(x)
                or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x)
                or not np.isfinite(x).all()):
            raise ValueError("Features must be nonempty finite arrays with matching dimensions")
        if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer) or (y < 0).any():
            raise ValueError("Labels must be one-dimensional nonnegative integers")
    classes = int(values["y_train"].max()) + 1
    if not np.array_equal(np.unique(values["y_train"]), np.arange(classes)):
        raise ValueError("Training labels must cover 0,...,number_of_classes-1")
    if classes < 2 or values["y_test"].max() >= classes:
        raise ValueError("Need at least two training classes, with no unseen test labels")
    return values, dim, classes


def run(input_path, output, epochs=2, sharing_ratio=50):
    if epochs < 1 or not 0 <= sharing_ratio <= 100:
        raise ValueError("epochs must be positive; sharing_ratio must be in [0,100]")
    torch.set_num_threads(2)
    engine.set_seed(44)
    values, dim, classes = load_features(input_path)
    loaders = {}
    for split in ("train", "test"):
        dataset = TensorDataset(torch.tensor(values["x_" + split], dtype=torch.float32),
                                torch.tensor(values["y_" + split], dtype=torch.long))
        loaders[split] = {0: DataLoader(dataset, batch_size=16, shuffle=split == "train",
                                      generator=torch.Generator().manual_seed(44))}
    model = engine.SharedMZINet(task_classes=[classes], input_rank=dim, hidden_dim=dim,
        num_layers=1, use_efficiency=False, head_mode="task_specific", global_classes=classes,
        photonic_readout="per_mesh_intensity", norm_type="layer",
        input_dim=dim, input_projection="direct")
    warmup = engine.train_model(model, loaders["train"], loaders["test"], [0], {"0": 1.0},
                                epochs, .0035, 1.0, .0001, "cpu")
    masks = engine.make_random_node_masks(sharing_ratio / 100, dim, 1, 44)
    model.apply_shared_masks(masks)
    result = engine.train_model(model, loaders["train"], loaders["test"], [0], {"0": 1.0},
                                epochs, .0035, 1.0, .0001, "cpu")
    report = {"input_features": dim, "training_samples": len(values["x_train"]),
              "test_samples": len(values["x_test"]), "sharing_ratio_percent": sharing_ratio,
              "seed": 44, "warmup": warmup, "finetune": result}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Wrote " + str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/custom_features/training.json")
    parser.add_argument("--make-demo", action="store_true", help="Create a small synthetic software-test dataset")
    parser.add_argument("--epochs", type=int, default=2, help="Epochs in each of warmup and finetuning")
    parser.add_argument("--sharing-ratio", type=float, default=50)
    args = parser.parse_args()
    if args.make_demo:
        if args.input.exists():
            parser.error("--make-demo will not overwrite an existing input")
        write_demo(args.input)
        print("Generated synthetic software-test data; not an EMNIST or hardware result.")
    run(args.input, args.output, args.epochs, args.sharing_ratio)


if __name__ == "__main__":
    main()
