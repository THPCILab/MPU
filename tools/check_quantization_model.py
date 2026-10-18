"""Bounded CPU checks requiring only the published code and codebook."""
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/quantization"))
import engine
import train_device_codebook as device


def main():
    torch.set_num_threads(2)
    with np.load(ROOT / "reproduction_data/fig5/device_runs/codebook_static_3bit_eps_2pow_minus332_losst90pct.npz") as z:
        library = z["matrices"]
    device.exp = engine
    device.install_library_forward(library)
    engine.set_seed(44)
    model = engine.SharedMZINet(task_classes=[10, 26, 26], input_rank=64,
        hidden_dim=64, num_layers=1, use_efficiency=False,
        head_mode="task_specific", global_classes=62,
        photonic_readout="per_mesh_intensity", norm_type="layer",
        input_dim=64, input_projection="direct", phase_quantization="shared", phase_bits=3)
    model.apply_shared_masks(engine.make_random_node_masks(.5, 64, 1, 42))
    x = torch.randn(4, 64)
    output = model(x, 0)
    assert output.shape == (4, 10) and torch.isfinite(output).all()
    output.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    transfer = model.mzi_layers[0].build_transfer(0)
    assert torch.linalg.svdvals(transfer).max() <= 1.00001
    for layer in model.mzi_layers:
        layer.active_phase_noise_rms = .0235472325
    with torch.no_grad():
        noisy = model(x, 0)
    assert torch.isfinite(noisy).all()
    print("PASS: 64-state deployment, forward/backward, passivity and active-phase perturbation.")


if __name__ == "__main__":
    main()
