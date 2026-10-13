"""Add atomic epoch recovery to the existing training loop without changing updates."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import numpy as np
import torch

def install(exp, work_root, ratio, diagnostics, evaluate_counts, library_path):
    checkpoint = Path(work_root) / "recovery_checkpoints" / ("seed44_ratio%d_latest.pt" % ratio)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    library_sha = hashlib.sha256(Path(library_path).read_bytes()).hexdigest()

    def restore(model, optimizer, scheduler, loaders, device):
        if not checkpoint.exists():
            return None
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["library_sha256"] != library_sha or state["ratio_percent"] != ratio:
            raise RuntimeError("Recovery checkpoint does not match this condition")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        for tid, loader in loaders.items():
            generator = getattr(loader, "generator", None)
            if generator is not None and str(tid) in state["loader_rng"]:
                generator.set_state(state["loader_rng"][str(tid)])
        diagnostics[:] = state["forward_diagnostics"]
        shared_count = sum(int(getattr(layer, "shared_mask_" + kind).sum().item())
                           for layer in model.mzi_layers for kind in ("even", "odd"))
        evaluate_counts[shared_count] = state["epoch"]
        print("RESUME_EPOCH " + str(state["epoch"]), flush=True)
        return state

    def save(model, optimizer, scheduler, loaders, epoch, histories):
        state = {
            "ratio_percent": ratio, "epoch": epoch, "library_sha256": library_sha,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "histories": histories,
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "loader_rng": {str(tid): loader.generator.get_state() for tid, loader in loaders.items()
                           if getattr(loader, "generator", None) is not None},
            "forward_diagnostics": list(diagnostics),
        }
        tmp = checkpoint.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, checkpoint)
        receipt = {"ratio_percent": ratio, "epoch": epoch,
                   "accuracy_percent": histories["mean_hist"][-1] * 100,
                   "training_loss": histories["train_loss_hist"][-1],
                   "library_sha256": library_sha}
        checkpoint.with_suffix(".json").write_text(json.dumps(receipt) + "\n")

    source = inspect.getsource(exp.train_model)
    before = "    for epoch in range(epochs):"
    restored = """    start_epoch = 0
    recovery = _epoch_restore(model, optimizer, scheduler, train_loaders, device)
    if recovery is not None:
        start_epoch = recovery["epoch"]
        history = recovery["histories"]
        mean_hist = history["mean_hist"]
        train_loss_hist = history["train_loss_hist"]
        train_mean_hist = history["train_mean_hist"]
        train_task_hist = history["train_task_hist"]
        task_hist = history["task_hist"]
        best_state_dict = history["best_state_dict"]
        best_seen = history["best_seen"]
    for epoch in range(start_epoch, epochs):"""
    after = '        print(f"  epoch {epoch + 1:03d}/{epochs}: mean={mean_acc:.4f}")'
    saved = after + """
        _epoch_save(model, optimizer, scheduler, train_loaders, epoch + 1, {
            "mean_hist": mean_hist, "train_loss_hist": train_loss_hist,
            "train_mean_hist": train_mean_hist, "train_task_hist": train_task_hist,
            "task_hist": task_hist, "best_state_dict": best_state_dict, "best_seen": best_seen,
        })"""
    if source.count(before) != 1 or source.count(after) != 1:
        raise RuntimeError("Training source does not match audited recovery insertion points")
    patched = source.replace(before, restored).replace(after, saved)
    exp._epoch_restore = restore
    exp._epoch_save = save
    exec(compile(patched, "<epoch-recoverable-train-model>", "exec"), exp.__dict__)
    (Path(work_root) / "recovery_checkpoints" / ("patched_train_model_ratio%d.py" % ratio)).write_text(patched)
