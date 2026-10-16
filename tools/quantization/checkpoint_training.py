"""Phase-safe, atomic epoch recovery. Load only locally created checkpoints."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def mask_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if "shared_mask_" in name:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, value in items:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str((name, array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def dataset_digest(loaders):
    items = []
    for tid, loader in sorted(loaders.items()):
        if not hasattr(loader.dataset, "tensors"):
            raise TypeError("Recovery requires TensorDataset inputs with inspectable tensors")
        items.extend((str(tid) + "_" + str(i), value)
                     for i, value in enumerate(loader.dataset.tensors))
    return tensor_digest(items)


class EpochRecovery:
    def __init__(self, directory, context, diagnostics, evaluate_counts):
        self.context = context
        self.diagnostics = diagnostics
        self.evaluate_counts = evaluate_counts
        key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[:20]
        self.path = Path(directory) / (context["phase"] + "_" + key + "_latest.pt")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def restore(self, model, optimizer, scheduler, loaders, device):
        if not self.path.exists():
            return None
        state = torch.load(self.path, map_location="cpu", weights_only=False)
        if state.get("context") != self.context or mask_digest(model) != self.context["mask_sha256"]:
            raise RuntimeError("Recovery context or requested sharing masks do not match")
        for name, value in model.state_dict().items():
            if "shared_mask_" in name and not torch.equal(value.cpu(), state["model"][name].cpu()):
                raise RuntimeError("Saved recovery masks do not match the requested condition")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        for tid, loader in loaders.items():
            generator = getattr(loader, "generator", None)
            if generator is not None and str(tid) in state["loader_rng"]:
                generator.set_state(state["loader_rng"][str(tid)])
        self.diagnostics[:] = state["forward_diagnostics"]
        shared_count = sum(int(v.sum().item()) for k, v in model.state_dict().items()
                           if "shared_mask_" in k)
        self.evaluate_counts[shared_count] = state["epoch"]
        print("RESUME " + self.context["phase"] + " EPOCH " + str(state["epoch"]), flush=True)
        return state

    def save(self, model, optimizer, scheduler, loaders, epoch, histories):
        state = {
            "context": self.context, "epoch": epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "histories": histories,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader_rng": {str(tid): loader.generator.get_state() for tid, loader in loaders.items()
                           if getattr(loader, "generator", None) is not None},
            "forward_diagnostics": list(self.diagnostics),
        }
        tmp = self.path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, self.path)
        receipt = {"context": self.context, "epoch": epoch,
                   "accuracy_percent": histories["mean_hist"][-1] * 100,
                   "training_loss": histories["train_loss_hist"][-1]}
        self.path.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def install(exp, work_root, ratio, diagnostics, evaluate_counts, library_path):
    """Attach explicit training hooks; legacy phase-ambiguous files are not reused."""
    original = exp.train_model
    if getattr(original, "_epoch_recovery_installed", False):
        raise RuntimeError("Epoch recovery is already installed in this engine")
    signature = inspect.signature(original)
    library_sha = hashlib.sha256(Path(library_path).read_bytes()).hexdigest()

    @functools.wraps(original)
    def recoverable(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        model = values["model"]
        context = {
            "format": 2, "phase": values["training_phase"], "ratio_percent": ratio,
            "library_sha256": library_sha, "mask_sha256": mask_digest(model),
            "initial_model_sha256": tensor_digest(model.state_dict().items()),
            "train_data_sha256": dataset_digest(values["train_loaders"]),
            "test_data_sha256": dataset_digest(values["test_loaders"]),
            "epochs": values["epochs"], "base_lr": values["base_lr"],
            "electrical_lr_scale": values["electrical_lr_scale"],
            "weight_decay": values["weight_decay"], "task_ids": list(values["task_ids"]),
            "task_weights": values["task_weights"],
            "capture_best_state": values["capture_best_state"],
            "track_train_accuracy": values["track_train_accuracy"],
            "private_delta_lr_scale": getattr(model, "private_delta_lr_scale", 1.0),
            "private_delta_weight_decay": getattr(model, "private_delta_weight_decay", values["weight_decay"]),
            "trainable": [(n, list(v.shape)) for n, v in model.named_parameters() if v.requires_grad],
            "loaders": {str(k): {"size": len(v.dataset), "batch_size": v.batch_size,
                        "seed": v.generator.initial_seed() if v.generator is not None else None}
                        for k, v in values["train_loaders"].items()},
        }
        context = json.loads(json.dumps(context, sort_keys=True))
        values["checkpoint_session"] = EpochRecovery(
            Path(work_root) / "recovery_checkpoints", context, diagnostics, evaluate_counts)
        return original(*bound.args, **bound.kwargs)

    recoverable._epoch_recovery_installed = True
    exp.train_model = recoverable
