"""Train 64x64 shared-MZI QAT with one frozen 64-state matrix codebook.

The adopted simulation convention is epsilon_F = 2**(-3.32) for each fixed
complex-matrix perturbation, followed by a common 90% power-transmission factor.
The same codebook is used during training and evaluation.  This is a
simulation convention motivated by amplitude measurements, not measured complex matrices.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("STATIC332_SOURCE_ROOT", str(Path(__file__).resolve().parent))).resolve()
WORK_ROOT = Path(os.environ.get("STATIC332_WORK_ROOT", str(REPO_ROOT / "results/quantization_training"))).resolve()
LIBRARY_PATH = WORK_ROOT / "results_static332_t90pct" / "codebook_static_3bit_eps_2pow_minus332_losst90pct.npz"
LIBRARY_SEED = 332064
ERROR_BITS = 3.32
POWER_TRANSMISSION = 0.90
LOSS_DB = -10.0 * math.log10(POWER_TRANSMISSION)
LEVELS = 8


def mzi_matrix(theta: float, phi: float) -> np.ndarray:
    half = 0.5 * theta
    gp = 1j * np.exp(0.5j * theta)
    pp = np.exp(1j * phi)
    return np.asarray(
        [
            [gp * pp * np.sin(half), gp * pp * np.cos(half)],
            [gp * np.cos(half), -gp * np.sin(half)],
        ],
        dtype=np.complex128,
    )


def passive_precision_matrix(u, source_error, epsilon):
    """Calibrate a fixed residual to exact precision, throughput and passivity.

    V has Frobenius norm sqrt(2). Its singular-value spread is capped so
    sqrt(T)*V is passive. A scalar search matches ||V-U||_F/||U||_F to epsilon.
    """
    d_cap = min(0.999, (1.0 / POWER_TRANSMISSION - 1.0) * (1.0 - 1e-6))

    def candidate(scale):
        if scale == 0.0:
            return u.copy()
        left, singular, right = np.linalg.svd(u + scale * source_error)
        s2 = singular ** 2
        spread = min(float((s2[0] - s2[1]) / s2.sum()), d_cap)
        calibrated_singular = np.sqrt([1.0 + spread, 1.0 - spread])
        return (left * calibrated_singular[None, :]) @ right

    lo, hi = 0.0, 1.0
    for _ in range(24):
        if np.linalg.norm(candidate(hi) - u) / np.linalg.norm(u) >= epsilon:
            break
        hi *= 2.0
    else:
        raise RuntimeError("Could not bracket target matrix precision")
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if np.linalg.norm(candidate(mid) - u) / np.linalg.norm(u) < epsilon:
            lo = mid
        else:
            hi = mid
    v = candidate(0.5 * (lo + hi))
    m = math.sqrt(POWER_TRANSMISSION) * v
    actual_error = float(np.linalg.norm(v - u) / np.linalg.norm(u))
    actual_transmission = float(np.linalg.norm(m) ** 2 / 2.0)
    max_singular = float(np.linalg.svd(m, compute_uv=False).max())
    if not np.isclose(actual_error, epsilon, atol=1e-12):
        raise RuntimeError("Precision calibration failed")
    if not np.isclose(actual_transmission, POWER_TRANSMISSION, atol=1e-12):
        raise RuntimeError("Transmission calibration failed")
    if max_singular > 1.0 + 1e-10:
        raise RuntimeError("Passive matrix constraint failed")
    return v, hi


def create_library() -> Path:
    LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    epsilon = 2.0 ** (-ERROR_BITS)
    amplitude_scale = math.sqrt(POWER_TRANSMISSION)
    rng = np.random.default_rng(LIBRARY_SEED)
    base = np.empty((64, 2, 2), dtype=np.complex128)
    errors = np.empty_like(base)
    matrices = np.empty_like(base)
    rows: list[dict] = []
    for code in range(64):
        ti, pi = divmod(code, LEVELS)
        theta = ti * math.pi / LEVELS
        phi = pi * 2.0 * math.pi / LEVELS
        u = mzi_matrix(theta, phi)
        e = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        e /= max(float(np.linalg.norm(e)), 1e-15)
        e *= float(np.linalg.norm(u)) * epsilon
        calibrated, _ = passive_precision_matrix(u, e, epsilon)
        e = calibrated - u
        deployed = calibrated * amplitude_scale
        base[code] = u
        errors[code] = e
        matrices[code] = deployed
        rows.append(
            {
                "codeword_index": code,
                "theta_index": ti,
                "phi_index": pi,
                "theta_rad": theta,
                "phi_rad": phi,
                "relative_matrix_error_frobenius": float(np.linalg.norm(e) / np.linalg.norm(u)),
                "insertion_loss_db": LOSS_DB,
                "matrix_amplitude_scale": amplitude_scale,
                "actual_average_power_transmission": float(np.linalg.norm(deployed) ** 2 / 2.0),
                "maximum_singular_value": float(np.linalg.svd(deployed, compute_uv=False).max()),
            }
        )

    np.savez_compressed(
        LIBRARY_PATH,
        matrices=matrices.astype(np.complex64),
        base_matrices=base.astype(np.complex64),
        errors=errors.astype(np.complex64),
        theta_indices=np.asarray([r["theta_index"] for r in rows], dtype=np.int64),
        phi_indices=np.asarray([r["phi_index"] for r in rows], dtype=np.int64),
        epsilon_rms=np.asarray(epsilon),
        equivalent_bits=np.asarray(ERROR_BITS),
        insertion_loss_db=np.asarray(LOSS_DB),
        amplitude_scale=np.asarray(amplitude_scale),
        seed=np.asarray(LIBRARY_SEED),
    )
    import csv

    with (LIBRARY_PATH.parent / "codebook_static_332_t90pct.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "status": "READY",
        "model": "64-state static complex codebook; one fixed matrix per 3-bit theta-phi codeword",
        "precision_convention": "relative Frobenius matrix error epsilon_F = 2^(-3.32)",
        "epsilon_F": epsilon,
        "error_distribution": "complex Gaussian directions projected to fixed Frobenius norm and passive singular-value bounds; scalar-calibrated to epsilon_F per codeword; generated once",
        "error_is_resampled_during_inference": False,
        "codebook_seed": LIBRARY_SEED,
        "insertion_loss_db": LOSS_DB,
        "power_transmission": POWER_TRANSMISSION,
        "matrix_amplitude_scale": amplitude_scale,
        "phase_source": "nominal 3-bit design theta/phi grid; not measured optical phase",
        "codebook_mapping": "code = theta_index*8 + phi_index; theta=k*pi/8, phi=m*pi/4, k,m=0..7",
    }
    (LIBRARY_PATH.parent / "codebook_static_332_t90pct_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return LIBRARY_PATH


def install_library_forward(library: np.ndarray) -> None:
    cls = exp.SharedClementsMZILayer
    if getattr(cls, "_static332_t90pct_patch", False):
        return

    def build_transfer(self, task_id):
        device = self.theta_even.device
        dtype = torch.complex64 if self.theta_even.dtype == torch.float32 else torch.complex128
        cached = getattr(self, "_static332_t90pct_library", None)
        if cached is None or cached.device != device:
            cached = torch.as_tensor(library, dtype=dtype, device=device)
            self._static332_t90pct_library = cached
        w = torch.eye(self.dim, device=device, dtype=dtype)
        even_idx_1 = self.even_idx_1.to(device)
        even_idx_2 = self.even_idx_2.to(device)
        odd_idx_1 = self.odd_idx_1.to(device)
        odd_idx_2 = self.odd_idx_2.to(device)
        levels = 1 << int(self.phase_bits)
        if levels != LEVELS:
            raise ValueError(f"The static library is 3-bit only; got {self.phase_bits} bits")

        theta_even, phi_even = self.task_phase(task_id, "even")
        theta_odd, phi_odd = self.task_phase(task_id, "odd")
        phase_rms = float(getattr(self, "active_phase_noise_rms", 0.0))
        if phase_rms:
            private_even = 1.0 - self.shared_mask_even
            private_odd = 1.0 - self.shared_mask_odd
            theta_even = theta_even + private_even * torch.randn_like(theta_even) * phase_rms
            phi_even = phi_even + private_even * torch.randn_like(phi_even) * phase_rms
            theta_odd = theta_odd + private_odd * torch.randn_like(theta_odd) * phase_rms
            phi_odd = phi_odd + private_odd * torch.randn_like(phi_odd) * phase_rms
        theta_even_code = self.quantize_theta(self.theta_even, ste=False)
        phi_even_code = self.quantize_phi(self.phi_even, ste=False)
        theta_odd_code = self.quantize_theta(self.theta_odd, ste=False)
        phi_odd_code = self.quantize_phi(self.phi_odd, ste=False)
        theta_even_index = torch.round(theta_even_code * levels / math.pi).long().clamp(0, levels - 1)
        phi_even_index = torch.remainder(torch.round(phi_even_code * levels / (2.0 * math.pi)).long(), levels)
        theta_odd_index = torch.round(theta_odd_code * levels / math.pi).long().clamp(0, levels - 1)
        phi_odd_index = torch.remainder(torch.round(phi_odd_code * levels / (2.0 * math.pi)).long(), levels)
        eta_even = self.eta("even")
        eta_odd = self.eta("odd")

        def insert(u, kind: str, k: int, theta_index, phi_index):
            shared = getattr(self, f"shared_mask_{kind}")[k].to(dtype)
            code = theta_index[k] * levels + phi_index[k]
            mat = cached[code]
            return tuple(
                ideal + shared * (mat[:, i, j] - ideal).detach()
                for ideal, (i, j) in zip(u, ((0, 0), (0, 1), (1, 0), (1, 1)))
            )

        for k in range(self.dim // 2):
            eta_e = None if eta_even is None else eta_even[k]
            u = self.complex_mzi(theta_even[k], phi_even[k], eta_e)
            u = insert(u, "even", k, theta_even_index, phi_even_index)
            w_e = torch.eye(self.dim, device=device, dtype=dtype)
            w_e[even_idx_1, even_idx_1] = u[0]
            w_e[even_idx_1, even_idx_2] = u[1]
            w_e[even_idx_2, even_idx_1] = u[2]
            w_e[even_idx_2, even_idx_2] = u[3]
            w = w_e @ w

            eta_o = None if eta_odd is None else eta_odd[k]
            u = self.complex_mzi(theta_odd[k], phi_odd[k], eta_o)
            u = insert(u, "odd", k, theta_odd_index, phi_odd_index)
            w_o = torch.eye(self.dim, device=device, dtype=dtype)
            w_o[odd_idx_1, odd_idx_1] = u[0]
            w_o[odd_idx_1, odd_idx_2] = u[1]
            w_o[odd_idx_2, odd_idx_1] = u[2]
            w_o[odd_idx_2, odd_idx_2] = u[3]
            w = w_o @ w
        return w

    cls.build_transfer = build_transfer
    cls._static332_t90pct_patch = True


def main() -> None:
    if sys.argv[1:] == ["--prepare-library"]:
        print(create_library())
        return
    forward_smoke = sys.argv[1:] == ["--forward-smoke"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    sys.path.insert(0, str(SOURCE_ROOT))
    global exp
    import engine as exp  # noqa: E402

    exp.ROOT = WORK_ROOT
    cache_path = Path(os.environ.get("STATIC332_FEATURE_CACHE", str(REPO_ROOT / "reproduction_data/fig5/replay/seed44_pca_features.pt")))
    if not forward_smoke and cache_path.is_file():
        from feature_cache import install as install_feature_cache
        install_feature_cache(exp, cache_path)
    elif not forward_smoke:
        print("Preparing EMNIST PCA inputs locally; no archived sample cache is distributed.")
    if not LIBRARY_PATH.is_file():
        raise FileNotFoundError(f"Run with --prepare-library first: {LIBRARY_PATH}")
    with np.load(LIBRARY_PATH, allow_pickle=False) as z:
        library = np.asarray(z["matrices"], dtype=np.complex64)
        epsilon = float(np.asarray(z["epsilon_rms"]).item())
        loss_db = float(np.asarray(z["insertion_loss_db"]).item())
        codebook_seed = int(np.asarray(z["seed"]).item())
    if library.shape != (64, 2, 2):
        raise ValueError(f"Unexpected codebook shape: {library.shape}")

    install_library_forward(library)
    if forward_smoke:
        specs = exp.task_specs("three")
        model = exp.SharedMZINet(
            task_classes=[int(s["classes"]) for s in specs],
            input_rank=64,
            hidden_dim=64,
            num_layers=1,
            use_efficiency=False,
            head_mode="task_specific",
            global_classes=62,
            photonic_readout="per_mesh_intensity",
            norm_type="layer",
            input_dim=64,
            input_projection="direct",
            phase_quantization="shared",
            phase_bits=3,
        ).cuda().eval()
        model.apply_shared_masks(exp.make_random_node_masks(0.5, 64, 1, 42))
        with torch.no_grad():
            out = model(torch.randn(4, 64, device="cuda"), 0)
        if not torch.isfinite(out).all():
            raise RuntimeError("Non-finite output in fixed-codebook forward smoke")
        print(f"forward_smoke=PASS shape={tuple(out.shape)} max_abs={out.abs().max().item():.6g}")
        return
    diagnostics = []
    evaluate_counts = {}
    original_evaluate = exp.evaluate

    def diagnostic_evaluate(model, loaders, task_ids, device):
        shared_count = sum(
            int(getattr(layer, "shared_mask_" + kind).sum().item())
            for layer in model.mzi_layers for kind in ("even", "odd")
        )
        evaluate_counts[shared_count] = evaluate_counts.get(shared_count, 0) + 1
        epoch_index = evaluate_counts[shared_count]
        if shared_count == 0 or epoch_index not in (1, 40, 80):
            return original_evaluate(model, loaders, task_ids, device)
        pending = {}
        captured = {}
        hooks = []

        def capture_norm(module, inputs):
            pending["intensity"] = inputs[0].detach()

        def capture_head(tid):
            def hook(module, inputs, output):
                if str(tid) in captured:
                    return
                raw = pending["intensity"]
                feature_variance = raw.var(dim=-1, unbiased=False).mean().item()
                captured[str(tid)] = {
                    "intensity_mean": raw.mean().item(),
                    "intensity_feature_variance": feature_variance,
                    "layernorm_epsilon": model.norms[0].eps,
                    "variance_over_layernorm_epsilon": feature_variance / model.norms[0].eps,
                    "logit_feature_variance": output.detach().var(dim=-1, unbiased=False).mean().item(),
                    "first_batch_unique_predictions": int(output.detach().argmax(-1).unique().numel()),
                    "first_batch_size": int(output.shape[0]),
                }
            return hook

        hooks.append(model.norms[0].register_forward_pre_hook(capture_norm))
        for tid in task_ids:
            hooks.append(model.heads[tid].register_forward_hook(capture_head(tid)))
        try:
            value = original_evaluate(model, loaders, task_ids, device)
        finally:
            for hook in hooks:
                hook.remove()
        optical_grad_squared = sum(
            p.grad.detach().abs().square().sum().item()
            for name, p in model.named_parameters()
            if "mzi_layers" in name and p.grad is not None
        )
        record = {
            "shared_mzi": shared_count,
            "evaluation_index": epoch_index,
            "mean_test_accuracy": value[0],
            "task_metrics_first_test_batch": captured,
            "last_training_batch_optical_gradient_l2": math.sqrt(optical_grad_squared),
        }
        diagnostics.append(record)
        print("FORWARD_DIAGNOSTIC " + json.dumps(record), flush=True)
        return value

    exp.evaluate = diagnostic_evaluate
    from paired_voltage_test import install as install_paired_voltage_test
    install_paired_voltage_test(exp)
    original_save = exp.save_results
    args = exp.parse_args()
    if args.phase_quantization != "shared" or args.phase_bits != 3:
        raise ValueError("Expected shared 3-bit QAT")
    if len(args.ratios) != 1:
        raise ValueError("Recovery-enabled shards require exactly one sharing ratio")
    from checkpoint_training import install
    install(exp, WORK_ROOT, int(args.ratios[0]), diagnostics, evaluate_counts, LIBRARY_PATH)
    metadata = {
        "condition": "measured_error_simulation_static_64_state_codebook_t90pct",
        "evidence_label": "measured-error simulation using the adopted matrix-error convention; not measured complex matrices",
        "codebook_entries": 64,
        "codebook_seed": codebook_seed,
        "codebook_sha256": hashlib.sha256(LIBRARY_PATH.read_bytes()).hexdigest(),
        "relative_matrix_error_frobenius": epsilon,
        "equivalent_precision_bits": ERROR_BITS,
        "insertion_loss_db_per_shared_mzi": loss_db,
        "power_transmission_per_shared_mpu": POWER_TRANSMISSION,
        "transmission_definition": "Frobenius norm squared / 2 for each deployed 2x2 codeword, equals 0.90; all singular values <= 1",
        "transmission_source": "assumed 90% single-MPU average power transmission",
        "residual_model": "static passive-constrained complex perturbation; precision calibrated after removing common attenuation",
        "matrix_amplitude_scale": 10.0 ** (-loss_db / 20.0),
        "forward_semantics": "one fixed matrix per hard shared 3-bit codeword; the same table is held fixed during training and evaluation; shared gradients use the ideal 3-bit MZI STE path",
        "private_mzis": "continuous task-specific phases, unchanged and lossless in this model",
        "phase_source": "nominal theta/phi grid; no per-codeword measured phase population is supplied",
        "training_seeds": [int(args.seed)],
        "execution_host": platform.node(),
        "execution_platform": platform.platform(),
        "paired_voltage_noise": "same final QAT checkpoint plus 10 mV RMS equivalent phase error on active MZIs; 5 test-noise draws; no additional training",
    }

    def save_results(path, payload):
        payload["static_332_t90pct_codebook"] = metadata
        payload["forward_diagnostics"] = diagnostics
        original_save(path, payload)

    exp.save_results = save_results
    exp.run_experiment(args)
    out_path = exp.ROOT / args.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    payload["static_332_t90pct_codebook"] = metadata
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
