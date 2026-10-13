import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


ROOT = Path(__file__).resolve().parent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_specs(name):
    if name == "letters":
        return [
            {"id": 0, "name": "Uppercase", "lo": 10, "hi": 35, "classes": 26},
            {"id": 1, "name": "Lowercase", "lo": 36, "hi": 61, "classes": 26},
        ]
    if name == "three":
        return [
            {"id": 0, "name": "Digits", "lo": 0, "hi": 9, "classes": 10},
            {"id": 1, "name": "Uppercase", "lo": 10, "hi": 35, "classes": 26},
            {"id": 2, "name": "Lowercase", "lo": 36, "hi": 61, "classes": 26},
        ]
    raise ValueError(f"Unknown task set: {name}")


def fit_pca_projection(train_tensors, rank):
    x = torch.cat([tensor.view(len(tensor), -1) for tensor in train_tensors], dim=0).float()
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    _, _, v = torch.pca_lowrank(centered, q=rank, center=False, niter=4)
    basis = v[:, :rank].contiguous()
    projected = centered @ basis
    scale = projected.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return mean, basis, scale


def apply_pca_projection(x, mean, basis, scale):
    flat = x.view(len(x), -1).float()
    return ((flat - mean) @ basis) / scale


def get_task_loaders(
    task_set,
    train_samples,
    test_samples,
    batch_size,
    seed,
    download=False,
    label_mode="local",
    input_preprocess="learned",
    input_rank=64,
):
    specs = task_specs(task_set)
    raw_train = datasets.EMNIST(root=str(ROOT / "data"), split="byclass", train=True, download=download)
    raw_test = datasets.EMNIST(root=str(ROOT / "data"), split="byclass", train=False, download=download)

    train_data = {}
    test_data = {}
    for spec in specs:
        tid = spec["id"]
        train_mask = (raw_train.targets >= spec["lo"]) & (raw_train.targets <= spec["hi"])
        x_train = raw_train.data[train_mask].clone().float() / 255.0
        y_train = raw_train.targets[train_mask].clone()
        if label_mode == "local":
            y_train = y_train - spec["lo"]
        elif label_mode != "global":
            raise ValueError(f"Unknown label_mode: {label_mode}")
        if train_samples is not None and train_samples > 0 and len(x_train) > train_samples:
            g = torch.Generator().manual_seed(seed + 1000 + tid)
            idx = torch.randperm(len(x_train), generator=g)[:train_samples]
            x_train, y_train = x_train[idx], y_train[idx]

        test_mask = (raw_test.targets >= spec["lo"]) & (raw_test.targets <= spec["hi"])
        x_test = raw_test.data[test_mask].clone().float() / 255.0
        y_test = raw_test.targets[test_mask].clone()
        if label_mode == "local":
            y_test = y_test - spec["lo"]
        elif label_mode != "global":
            raise ValueError(f"Unknown label_mode: {label_mode}")
        if test_samples is not None and test_samples > 0 and len(x_test) > test_samples:
            g = torch.Generator().manual_seed(seed + 2000 + tid)
            idx = torch.randperm(len(x_test), generator=g)[:test_samples]
            x_test, y_test = x_test[idx], y_test[idx]

        train_data[tid] = [x_train, y_train]
        test_data[tid] = [x_test, y_test]

    preprocess_profile = {"input_preprocess": input_preprocess}
    if input_preprocess == "learned":
        for tid in train_data:
            train_data[tid][0] = train_data[tid][0].view(len(train_data[tid][0]), -1)
            test_data[tid][0] = test_data[tid][0].view(len(test_data[tid][0]), -1)
        preprocess_profile["input_dim"] = 784
    elif input_preprocess == "pca":
        torch.manual_seed(seed + 9090)
        mean, basis, scale = fit_pca_projection([train_data[spec["id"]][0] for spec in specs], input_rank)
        explained = []
        # Compute a lightweight diagnostic: variance carried by each retained standardized component before scaling.
        all_train = torch.cat([train_data[spec["id"]][0].view(len(train_data[spec["id"]][0]), -1) for spec in specs], dim=0)
        projected = (all_train.float() - mean) @ basis
        var = projected.var(dim=0, unbiased=False)
        total_var = (all_train.float() - mean).var(dim=0, unbiased=False).sum().clamp_min(1e-8)
        explained = (var / total_var).cpu().tolist()
        for tid in train_data:
            train_data[tid][0] = apply_pca_projection(train_data[tid][0], mean, basis, scale)
            test_data[tid][0] = apply_pca_projection(test_data[tid][0], mean, basis, scale)
        preprocess_profile.update(
            {
                "input_dim": int(input_rank),
                "pca_rank": int(input_rank),
                "pca_explained_variance_sum": float(sum(explained)),
                "pca_component_variance": explained,
            }
        )
    else:
        raise ValueError(f"Unknown input_preprocess: {input_preprocess}")

    train_loaders = {}
    test_loaders = {}
    for spec in specs:
        tid = spec["id"]
        x_train, y_train = train_data[tid]
        x_test, y_test = test_data[tid]
        train_loaders[tid] = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loaders[tid] = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=batch_size,
            shuffle=False,
        )
    return specs, train_loaders, test_loaders, preprocess_profile


class MultiTaskMLP(nn.Module):
    def __init__(self, hidden_dims, task_classes, dropout=0.0, head_mode="task_specific", global_classes=62, input_dim=784):
        super().__init__()
        self.head_mode = head_mode
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
            if dropout:
                layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        if head_mode == "task_specific":
            self.heads = nn.ModuleList([nn.Linear(dims[-1], n) for n in task_classes])
            self.head = None
        elif head_mode == "shared_global":
            self.heads = None
            self.head = nn.Linear(dims[-1], global_classes)
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")

    def forward(self, x, task_id):
        z = self.trunk(x)
        if self.head_mode == "shared_global":
            return self.head(z)
        return self.heads[task_id](z)


class SharedClementsMZILayer(nn.Module):
    def __init__(
        self,
        dim,
        num_tasks,
        use_efficiency=False,
        efficiency_min=0.65,
        efficiency_init=0.96,
        phase_quantization="none",
        phase_bits=3,
    ):
        super().__init__()
        self.dim = dim
        self.num_tasks = num_tasks
        self.num_even_mzi = dim // 2
        self.num_odd_mzi = (dim - 2) // 2
        self.use_efficiency = use_efficiency
        self.efficiency_min = efficiency_min
        self.phase_quantization = phase_quantization
        self.phase_bits = phase_bits

        self.register_buffer("even_idx_1", torch.arange(0, dim, 2))
        self.register_buffer("even_idx_2", torch.arange(1, dim, 2))
        self.register_buffer("odd_idx_1", torch.arange(1, dim - 1, 2))
        self.register_buffer("odd_idx_2", torch.arange(2, dim, 2))

        self.theta_even = nn.Parameter(torch.empty(dim // 2, self.num_even_mzi))
        self.theta_odd = nn.Parameter(torch.empty(dim // 2, self.num_odd_mzi))
        self.phi_even = nn.Parameter(torch.empty(dim // 2, self.num_even_mzi))
        self.phi_odd = nn.Parameter(torch.empty(dim // 2, self.num_odd_mzi))
        nn.init.uniform_(self.theta_even, 0, math.pi)
        nn.init.uniform_(self.theta_odd, 0, math.pi)
        nn.init.uniform_(self.phi_even, 0, 2 * math.pi)
        nn.init.uniform_(self.phi_odd, 0, 2 * math.pi)

        self.delta_theta_even = nn.Parameter(torch.zeros(num_tasks, dim // 2, self.num_even_mzi))
        self.delta_theta_odd = nn.Parameter(torch.zeros(num_tasks, dim // 2, self.num_odd_mzi))
        self.delta_phi_even = nn.Parameter(torch.zeros(num_tasks, dim // 2, self.num_even_mzi))
        self.delta_phi_odd = nn.Parameter(torch.zeros(num_tasks, dim // 2, self.num_odd_mzi))

        self.register_buffer("shared_mask_even", torch.ones_like(self.theta_even))
        self.register_buffer("shared_mask_odd", torch.ones_like(self.theta_odd))

        if use_efficiency:
            p = (efficiency_init - efficiency_min) / max(1e-6, 1.0 - efficiency_min)
            p = min(0.999, max(0.001, p))
            init_logit = math.log(p / (1.0 - p))
            self.eta_logit_even = nn.Parameter(torch.full_like(self.theta_even, init_logit))
            self.eta_logit_odd = nn.Parameter(torch.full_like(self.theta_odd, init_logit))
        else:
            self.register_parameter("eta_logit_even", None)
            self.register_parameter("eta_logit_odd", None)

    def set_shared_masks(self, even, odd):
        self.shared_mask_even.copy_(even.to(self.shared_mask_even.device, self.shared_mask_even.dtype))
        self.shared_mask_odd.copy_(odd.to(self.shared_mask_odd.device, self.shared_mask_odd.dtype))

    def eta(self, kind):
        if not self.use_efficiency:
            return None
        logits = self.eta_logit_even if kind == "even" else self.eta_logit_odd
        return self.efficiency_min + (1.0 - self.efficiency_min) * torch.sigmoid(logits)

    def quantize_theta(self, theta, ste=True):
        """Quantize theta to the manuscript grid k*pi/2**bits, k=0,...,2**bits-1."""
        levels = 1 << self.phase_bits
        step = math.pi / levels
        indices = torch.round(theta / step).clamp(0, levels - 1)
        quantized = indices * step
        return theta + (quantized - theta).detach() if ste else quantized

    def quantize_phi(self, phi, ste=True):
        """Quantize the external phase to 2**bits periodic levels on [0, 2pi)."""
        levels = 1 << self.phase_bits
        wrapped = torch.remainder(phi, 2 * math.pi)
        indices = torch.remainder(torch.round(wrapped * levels / (2 * math.pi)), levels)
        quantized = indices * (2 * math.pi / levels)
        return phi + (quantized - phi).detach() if ste else quantized

    def quantize_phase_pair(self, theta, phi, ste=True):
        return self.quantize_theta(theta, ste=ste), self.quantize_phi(phi, ste=ste)

    def task_phase(self, task_id, kind):
        if kind == "even":
            shared = self.shared_mask_even
            theta_base, phi_base = self.theta_even, self.phi_even
            theta_delta = self.delta_theta_even[task_id]
            phi_delta = self.delta_phi_even[task_id]
        else:
            shared = self.shared_mask_odd
            theta_base, phi_base = self.theta_odd, self.phi_odd
            theta_delta = self.delta_theta_odd[task_id]
            phi_delta = self.delta_phi_odd[task_id]
        private = 1.0 - shared
        theta = theta_base + private * theta_delta
        phi = phi_base + private * phi_delta
        if self.phase_quantization == "shared":
            theta_q, phi_q = self.quantize_phase_pair(theta_base, phi_base, ste=True)
            theta = shared * theta_q + private * theta
            phi = shared * phi_q + private * phi
        elif self.phase_quantization == "all":
            theta, phi = self.quantize_phase_pair(theta, phi, ste=True)
        return theta, phi

    @staticmethod
    def complex_mzi(theta, phi, eta):
        """Complex 2x2 phase-modulated MZI block.

        Implements the standard 50:50-beamsplitter MZI unitary
        U(theta, phi) up to the physically meaningful global phase. If an
        efficiency parameter is used, it is treated as power transmissivity and
        converted to field-amplitude attenuation by sqrt(eta).
        """
        half = 0.5 * theta
        phase_phi = torch.exp(1j * phi)
        global_phase = 1j * torch.exp(0.5j * theta)
        amp = 1.0 if eta is None else torch.sqrt(eta.clamp_min(1e-8))
        u11 = amp * global_phase * phase_phi * torch.sin(half)
        u12 = amp * global_phase * phase_phi * torch.cos(half)
        u21 = amp * global_phase * torch.cos(half)
        u22 = -amp * global_phase * torch.sin(half)
        return u11, u12, u21, u22

    def build_transfer(self, task_id):
        device = self.theta_even.device
        dtype = torch.complex64 if self.theta_even.dtype == torch.float32 else torch.complex128
        w = torch.eye(self.dim, device=device, dtype=dtype)
        even_idx_1 = self.even_idx_1.to(device)
        even_idx_2 = self.even_idx_2.to(device)
        odd_idx_1 = self.odd_idx_1.to(device)
        odd_idx_2 = self.odd_idx_2.to(device)

        theta_even, phi_even = self.task_phase(task_id, "even")
        theta_odd, phi_odd = self.task_phase(task_id, "odd")
        eta_even = self.eta("even")
        eta_odd = self.eta("odd")

        for k in range(self.dim // 2):
            eta_e = None if eta_even is None else eta_even[k]
            u11, u12, u21, u22 = self.complex_mzi(theta_even[k], phi_even[k], eta_e)
            w_e = torch.eye(self.dim, device=device, dtype=dtype)
            w_e[even_idx_1, even_idx_1] = u11
            w_e[even_idx_1, even_idx_2] = u12
            w_e[even_idx_2, even_idx_1] = u21
            w_e[even_idx_2, even_idx_2] = u22
            w = w_e @ w

            eta_o = None if eta_odd is None else eta_odd[k]
            u11, u12, u21, u22 = self.complex_mzi(theta_odd[k], phi_odd[k], eta_o)
            w_o = torch.eye(self.dim, device=device, dtype=dtype)
            w_o[odd_idx_1, odd_idx_1] = u11
            w_o[odd_idx_1, odd_idx_2] = u12
            w_o[odd_idx_2, odd_idx_1] = u21
            w_o[odd_idx_2, odd_idx_2] = u22
            w = w_o @ w
        return w

    def forward(self, x, task_id):
        field = self.propagate_field(x.to(self.theta_even.device), task_id)
        return self.detect(field)

    def propagate_field(self, field, task_id):
        w = self.build_transfer(task_id)
        field = field.to(w.dtype) @ w.t()
        return field

    @staticmethod
    def detect(field):
        return field.abs().pow(2)


class SharedMZINet(nn.Module):
    def __init__(
        self,
        task_classes,
        input_rank=32,
        hidden_dim=72,
        num_layers=3,
        use_efficiency=False,
        efficiency_min=0.65,
        head_mode="task_specific",
        global_classes=62,
        photonic_readout="per_mesh_intensity",
        norm_type="layer",
        input_dim=784,
        input_projection="learned",
        phase_quantization="none",
        phase_bits=3,
    ):
        super().__init__()
        self.head_mode = head_mode
        self.photonic_readout = photonic_readout
        self.norm_type = norm_type
        self.input_projection = input_projection
        self.num_tasks = len(task_classes)
        self.input_rank = input_rank
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        if input_projection == "learned":
            self.proj_low = nn.Linear(input_dim, input_rank, bias=False)
            self.proj_in = nn.Linear(input_rank, hidden_dim)
        elif input_projection == "direct":
            if input_dim != hidden_dim:
                raise ValueError("Direct input projection requires input_dim == hidden_dim.")
            self.proj_low = nn.Identity()
            self.proj_in = nn.Identity()
        else:
            raise ValueError(f"Unknown input_projection: {input_projection}")
        self.act = nn.GELU()
        self.mzi_layers = nn.ModuleList(
            [
                SharedClementsMZILayer(
                    hidden_dim,
                    self.num_tasks,
                    use_efficiency=use_efficiency,
                    efficiency_min=efficiency_min,
                    phase_quantization=phase_quantization,
                    phase_bits=phase_bits,
                )
                for _ in range(num_layers)
            ]
        )
        if norm_type == "batch":
            self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim, affine=False) for _ in range(num_layers)])
        elif norm_type == "layer":
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim, elementwise_affine=True) for _ in range(num_layers)])
        elif norm_type == "none":
            self.norms = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
        if head_mode == "task_specific":
            self.heads = nn.ModuleList([nn.Linear(hidden_dim, n) for n in task_classes])
            self.head = None
        elif head_mode == "shared_global":
            self.heads = None
            self.head = nn.Linear(hidden_dim, global_classes)
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")
        self.freeze_electrical_runtime = False
        self.private_delta_lr_scale = 1.0
        self.private_delta_weight_decay = 1e-4

    def apply_shared_masks(self, masks):
        for layer, mask in zip(self.mzi_layers, masks):
            layer.set_shared_masks(mask["even"], mask["odd"])

    def set_electrical_eval(self):
        for norm in self.norms:
            norm.eval()

    def forward(self, x, task_id):
        if self.input_projection == "learned":
            x = self.act(self.proj_in(self.proj_low(x)))
        else:
            x = x.to(self.mzi_layers[0].theta_even.device)
        if self.photonic_readout == "per_mesh_intensity":
            for mzi, norm in zip(self.mzi_layers, self.norms):
                x = self.act(norm(mzi(x, task_id)))
        elif self.photonic_readout == "per_mesh_coherent":
            for mzi, norm in zip(self.mzi_layers, self.norms):
                field = mzi.propagate_field(x, task_id)
                # Homodyne-style coherent readout keeps signed interference
                # information for the electrical nonlinearity before re-encoding.
                x = self.act(norm(field.real))
        elif self.photonic_readout == "coherent_stack":
            field = x.to(torch.complex64 if x.dtype == torch.float32 else torch.complex128)
            for mzi in self.mzi_layers:
                field = mzi.propagate_field(field, task_id)
            x = self.act(self.norms[-1](SharedClementsMZILayer.detect(field)))
        else:
            raise ValueError(f"Unknown photonic_readout: {self.photonic_readout}")
        if self.head_mode == "shared_global":
            return self.head(x)
        return self.heads[task_id](x)


def mzi_shapes(hidden_dim, num_layers):
    return [
        {
            "even": (hidden_dim // 2, hidden_dim // 2),
            "odd": (hidden_dim // 2, (hidden_dim - 2) // 2),
        }
        for _ in range(num_layers)
    ]


def empty_masks(hidden_dim, num_layers, value=0.0):
    return [
        {
            "even": torch.full(shape["even"], value),
            "odd": torch.full(shape["odd"], value),
        }
        for shape in mzi_shapes(hidden_dim, num_layers)
    ]


def mask_units(hidden_dim, num_layers):
    units = []
    for li, shape in enumerate(mzi_shapes(hidden_dim, num_layers)):
        for kind in ("even", "odd"):
            rows, cols = shape[kind]
            for r in range(rows):
                for c in range(cols):
                    units.append((li, kind, r, c))
    return units


def set_unit(masks, unit, value):
    li, kind, r, c = unit
    masks[li][kind][r, c] = value


def count_shared(masks):
    total = 0
    shared = 0
    for mask in masks:
        for kind in ("even", "odd"):
            total += mask[kind].numel()
            shared += int(mask[kind].sum().item())
    return shared, total


def select_units_by_groups(groups, target, rng):
    order = list(range(len(groups)))
    rng.shuffle(order)
    selected = []
    for gi in order:
        group = groups[gi]
        if len(selected) + len(group) <= target:
            selected.extend(group)
        else:
            remaining = target - len(selected)
            if remaining > 0:
                group_copy = list(group)
                rng.shuffle(group_copy)
                selected.extend(group_copy[:remaining])
            break
    return selected


def make_random_node_masks(shared_ratio, hidden_dim, num_layers, seed):
    units = mask_units(hidden_dim, num_layers)
    target = int(round(shared_ratio * len(units)))
    rng = random.Random(seed)
    selected = rng.sample(units, target) if target > 0 else []
    masks = empty_masks(hidden_dim, num_layers, value=0.0)
    for unit in selected:
        set_unit(masks, unit, 1.0)
    return masks


def make_block_masks(shared_ratio, hidden_dim, num_layers, seed, block_h=6, block_w=6):
    shapes = mzi_shapes(hidden_dim, num_layers)
    groups = []
    for li, shape in enumerate(shapes):
        for kind in ("even", "odd"):
            rows, cols = shape[kind]
            bw = block_w if kind == "even" else max(2, block_w - 1)
            for r0 in range(0, rows, block_h):
                for c0 in range(0, cols, bw):
                    group = []
                    for r in range(r0, min(rows, r0 + block_h)):
                        for c in range(c0, min(cols, c0 + bw)):
                            group.append((li, kind, r, c))
                    groups.append(group)
    total = sum(len(g) for g in groups)
    target = int(round(shared_ratio * total))
    selected = select_units_by_groups(groups, target, random.Random(seed))
    masks = empty_masks(hidden_dim, num_layers, value=0.0)
    for unit in selected:
        set_unit(masks, unit, 1.0)
    return masks


def make_clements_depth_masks(shared_ratio, hidden_dim, num_layers):
    """Share a deterministic front-depth cascade of the Clements mesh.

    This is the coarse layer baseline: early optical meshes are shared across
    tasks, while the remaining downstream meshes keep task-private deltas. The
    boundary mesh is filled in a fixed physical order only to match the target
    MZI count, not to search for a favorable random slice.
    """
    units = []
    for li, shape in enumerate(mzi_shapes(hidden_dim, num_layers)):
        for kind in ("even", "odd"):
            rows, cols = shape[kind]
            for c in range(cols):
                for r in range(rows):
                    units.append((li, kind, r, c))
    target = int(round(shared_ratio * len(units)))
    masks = empty_masks(hidden_dim, num_layers, value=0.0)
    for unit in units[:target]:
        set_unit(masks, unit, 1.0)
    return masks


def stage_order_units(hidden_dim, num_layers):
    units = []
    for li, shape in enumerate(mzi_shapes(hidden_dim, num_layers)):
        max_stages = max(shape["even"][0], shape["odd"][0])
        for stage in range(max_stages):
            for kind in ("even", "odd"):
                rows, cols = shape[kind]
                if stage >= rows:
                    continue
                for c in range(cols):
                    units.append((li, kind, stage, c))
    return units


def make_clements_stage_masks(shared_ratio, hidden_dim, num_layers):
    """Share the first physical propagation columns of each Clements mesh."""
    units = stage_order_units(hidden_dim, num_layers)
    target = int(round(shared_ratio * len(units)))
    masks = empty_masks(hidden_dim, num_layers, value=0.0)
    for unit in units[:target]:
        set_unit(masks, unit, 1.0)
    return masks


def make_clements_slice_masks(shared_ratio, hidden_dim, num_layers, seed):
    shapes = mzi_shapes(hidden_dim, num_layers)
    groups = []
    for li, shape in enumerate(shapes):
        rows = shape["even"][0]
        for r in range(rows):
            group = []
            for c in range(shape["even"][1]):
                group.append((li, "even", r, c))
            for c in range(shape["odd"][1]):
                group.append((li, "odd", r, c))
            groups.append(group)
    total = sum(len(g) for g in groups)
    target = int(round(shared_ratio * total))
    selected = select_units_by_groups(groups, target, random.Random(seed))
    masks = empty_masks(hidden_dim, num_layers, value=0.0)
    for unit in selected:
        set_unit(masks, unit, 1.0)
    return masks


def make_stage_refined_node_masks(shared_ratio, scores, hidden_dim, num_layers, refine_fraction=0.35):
    """Start from a front-stage mask and do score-improving node swaps.

    This probes whether fine-grained MPU selection can improve a strong
    front-column sharing prior without changing the total replacement ratio.
    If no outside node has a higher commonness score than the weakest selected
    front-stage node, the original stage mask is kept.
    """
    masks = make_clements_stage_masks(shared_ratio, hidden_dim, num_layers)
    target, total = count_shared(masks)
    max_swaps = int(round(max(0.0, refine_fraction) * target))
    if target <= 0 or target >= total or max_swaps <= 0:
        return masks

    inside = []
    outside = []
    for li, layer_score in enumerate(scores):
        for kind in ("even", "odd"):
            score_tensor = layer_score[kind]
            rows, cols = score_tensor.shape
            for r in range(rows):
                for c in range(cols):
                    item = (float(score_tensor[r, c]), (li, kind, r, c))
                    if masks[li][kind][r, c].item() > 0:
                        inside.append(item)
                    else:
                        outside.append(item)

    inside.sort(key=lambda item: item[0])
    outside.sort(key=lambda item: item[0], reverse=True)
    swaps = 0
    for out_item, in_item in zip(outside, inside):
        if swaps >= max_swaps:
            break
        out_score, out_unit = out_item
        in_score, in_unit = in_item
        if out_score <= in_score:
            break
        set_unit(masks, in_unit, 0.0)
        set_unit(masks, out_unit, 1.0)
        swaps += 1
    return masks


def make_guided_node_masks(shared_ratio, scores, hidden_dim, num_layers, balance="global"):
    units = []
    for li, layer_score in enumerate(scores):
        for kind in ("even", "odd"):
            score_tensor = layer_score[kind]
            rows, cols = score_tensor.shape
            for r in range(rows):
                for c in range(cols):
                    units.append((float(score_tensor[r, c]), (li, kind, r, c)))
    target = int(round(shared_ratio * len(units)))
    if balance == "global":
        units.sort(key=lambda item: item[0], reverse=True)
        masks = empty_masks(hidden_dim, num_layers, value=0.0)
        for _, unit in units[:target]:
            set_unit(masks, unit, 1.0)
        return masks
    if balance not in ("layer", "kind"):
        raise ValueError(f"Unknown guided node balance: {balance}")

    groups = {}
    for score, unit in units:
        li, kind, _, _ = unit
        key = (li, kind) if balance == "kind" else li
        groups.setdefault(key, []).append((score, unit))
    masks = empty_masks(hidden_dim, num_layers, value=0.0)
    selected = 0
    remainders = []
    for key, group in groups.items():
        group.sort(key=lambda item: item[0], reverse=True)
        raw_quota = target * len(group) / max(1, len(units))
        quota = int(math.floor(raw_quota))
        remainders.append((raw_quota - quota, key))
        for _, unit in group[:quota]:
            set_unit(masks, unit, 1.0)
        selected += quota
    for _, key in sorted(remainders, reverse=True):
        if selected >= target:
            break
        current = []
        group = groups[key]
        for score, unit in group:
            li, kind, r, c = unit
            if masks[li][kind][r, c].item() == 0:
                current.append((score, unit))
        if current:
            _, unit = current[0]
            set_unit(masks, unit, 1.0)
            selected += 1
    return masks


def mask_profile(masks):
    shared, total = count_shared(masks)
    per_layer = []
    per_layer_kind = []
    for mask in masks:
        n = mask["even"].numel() + mask["odd"].numel()
        s = mask["even"].sum().item() + mask["odd"].sum().item()
        per_layer.append(float(s / n))
        per_layer_kind.append(
            {
                "even": float(mask["even"].sum().item() / max(1, mask["even"].numel())),
                "odd": float(mask["odd"].sum().item() / max(1, mask["odd"].numel())),
            }
        )
    return {
        "effective_shared_ratio": float(shared / total) if total else 0.0,
        "shared_mzi": int(shared),
        "total_mzi": int(total),
        "per_layer_shared_ratio": per_layer,
        "per_layer_kind_shared_ratio": per_layer_kind,
    }


def data_profile(specs, train_loaders, test_loaders, label_mode="local"):
    return [
        {
            "task_id": int(spec["id"]),
            "name": spec["name"],
            "label_range": [int(spec["lo"]), int(spec["hi"])],
            "classes": int(spec["classes"]),
            "label_mode": label_mode,
            "train_samples": int(len(train_loaders[spec["id"]].dataset)),
            "test_samples": int(len(test_loaders[spec["id"]].dataset)),
        }
        for spec in specs
    ]


def model_profile(args):
    shapes = mzi_shapes(args.hidden_dim, args.num_layers)
    mzi_per_layer = [int(shape["even"][0] * shape["even"][1] + shape["odd"][0] * shape["odd"][1]) for shape in shapes]
    total_mzi = int(sum(mzi_per_layer))
    return {
        "input_pixels": 784,
        "input_preprocess": args.input_preprocess,
        "input_dim": int(args.model_input_dim),
        "input_projection": args.input_projection,
        "input_rank": int(args.input_rank),
        "hidden_dim": int(args.hidden_dim),
        "num_clements_meshes": int(args.num_layers),
        "mzi_per_clements_mesh": mzi_per_layer,
        "total_mzi": total_mzi,
        "head_mode": args.head_mode,
        "global_classes": int(args.global_classes),
        "freeze_electrical_after_warmup": bool(args.freeze_electrical_after_warmup),
        "private_delta_lr_scale": float(args.private_delta_lr_scale),
        "private_delta_weight_decay": float(args.private_delta_weight_decay),
        "guided_node_balance": args.guided_node_balance,
        "guided_node_prior": args.guided_node_prior,
        "guided_node_refine_fraction": float(args.guided_node_refine_fraction),
        "clements_layer_mode": args.clements_layer_mode,
        "block_shape": [int(args.block_h), int(args.block_w)],
        "photonic_propagation": "complex_field_mzi",
        "mzi_unitary": "U(theta, phi) = i exp(i theta/2) [[exp(i phi) sin(theta/2), exp(i phi) cos(theta/2)], [cos(theta/2), -sin(theta/2)]]",
        "photonic_readout": args.photonic_readout,
        "norm_type": args.norm_type,
        "warmup_shared_ratio": float(args.warmup_shared_ratio),
        "phase_quantization": args.phase_quantization,
        "phase_bits": int(args.phase_bits),
        "theta_codebook": "2^bits levels k*pi/2^bits for k=0,...,2^bits-1 (pi excluded), matching the manuscript library grid",
        "phi_codebook": "2^bits uniform periodic levels on [0, 2pi)",
        "quantization_gradient": "straight-through estimator; every quantized forward pass uses hard codewords",
        "detector": "per_mesh_intensity uses physical square-law detection after each mesh; per_mesh_coherent and coherent_stack are ablations",
        "sharing_equation": "theta_task = theta_shared + (1 - shared_mask) * delta_task",
        "phase_sharing_equation": "phi_task = phi_shared + (1 - shared_mask) * delta_phi_task",
        "shared_mask_meaning": "1 means a design-time trainable cross-task common 2x2 unit with no task delta; 0 means task-private phase deltas are enabled",
        "mpu_deployment_semantics": "After training, shared_mask=1 units are fabricated/fixed as passive MPU transfer matrices using the learned common theta/phi. The training metric is multi-task accuracy only.",
    }


def pruning_score_profile(scores):
    profile = []
    for li, layer_score in enumerate(scores):
        item = {"layer": int(li)}
        for kind in ("even", "odd"):
            score = layer_score[kind].detach().cpu().float()
            item[kind] = {
                "mean": float(score.mean().item()),
                "std": float(score.std(unbiased=False).item()),
                "min": float(score.min().item()),
                "max": float(score.max().item()),
            }
        profile.append(item)
    return profile


def evaluate(model, loaders, task_ids, device):
    model.eval()
    accs = {}
    with torch.no_grad():
        for tid in task_ids:
            correct = 0
            total = 0
            for x, y in loaders[tid]:
                x, y = x.to(device), y.to(device)
                pred = model(x, tid).argmax(1)
                correct += (pred == y).sum().item()
                total += y.numel()
            accs[str(tid)] = correct / total if total else 0.0
    mean_acc = float(np.mean(list(accs.values()))) if accs else 0.0
    return mean_acc, accs


def compute_confusion_matrices(model, loaders, task_ids, device):
    model.eval()
    matrices = {}
    with torch.no_grad():
        for tid in task_ids:
            labels_seen = []
            preds_seen = []
            for x, y in loaders[tid]:
                x = x.to(device)
                pred = model(x, tid).argmax(1).detach().cpu().numpy()
                preds_seen.extend(int(v) for v in pred)
                labels_seen.extend(int(v) for v in y.numpy())
            labels = sorted(set(labels_seen) | set(preds_seen))
            index = {label: i for i, label in enumerate(labels)}
            cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
            for y_true, y_pred in zip(labels_seen, preds_seen):
                cm[index[y_true], index[y_pred]] += 1
            row_sum = cm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64), where=row_sum > 0)
            matrices[str(tid)] = {
                "labels": labels,
                "counts": cm.tolist(),
                "normalized": cm_norm.round(6).tolist(),
                "accuracy": float(np.trace(cm) / max(1, cm.sum())),
            }
    return matrices


def serialize_masks(masks):
    return [
        {
            "even": mask["even"].detach().cpu().to(torch.uint8).tolist(),
            "odd": mask["odd"].detach().cpu().to(torch.uint8).tolist(),
        }
        for mask in masks
    ]


def quantization_profile(model, masks):
    """Summarize the actually deployed shared 3-bit MZI codewords."""
    if not model.mzi_layers or model.mzi_layers[0].phase_quantization == "none":
        return {"enabled": False}
    total_shared = 0
    codewords = set()
    theta_errors = []
    phi_errors = []
    for layer, mask in zip(model.mzi_layers, masks):
        for kind in ("even", "odd"):
            theta = getattr(layer, f"theta_{kind}").detach()
            phi = getattr(layer, f"phi_{kind}").detach()
            shared = mask[kind].to(theta.device).bool()
            theta_q = layer.quantize_theta(theta, ste=False)
            phi_q = layer.quantize_phi(phi, ste=False)
            theta_levels = 1 << layer.phase_bits
            theta_code = torch.round(theta_q * theta_levels / math.pi).long().clamp(0, theta_levels - 1)
            phi_code = torch.remainder(
                torch.round(phi_q * (1 << layer.phase_bits) / (2 * math.pi)).long(),
                1 << layer.phase_bits,
            )
            total_shared += int(shared.sum().item())
            codewords.update(zip(theta_code[shared].cpu().tolist(), phi_code[shared].cpu().tolist()))
            theta_errors.extend((theta_q[shared] - theta[shared]).abs().cpu().tolist())
            phi_delta = torch.atan2(torch.sin(phi_q[shared] - phi[shared]), torch.cos(phi_q[shared] - phi[shared])).abs()
            phi_errors.extend(phi_delta.cpu().tolist())
    return {
        "enabled": True,
        "mode": model.mzi_layers[0].phase_quantization,
        "bits_per_phase": int(model.mzi_layers[0].phase_bits),
        "shared_mzi": total_shared,
        "unique_shared_codewords": len(codewords),
        "used_shared_codewords": [list(v) for v in sorted(codewords)],
        "mean_abs_theta_quantization_error": float(np.mean(theta_errors)) if theta_errors else 0.0,
        "mean_abs_phi_quantization_error": float(np.mean(phi_errors)) if phi_errors else 0.0,
    }


def train_model(
    model,
    train_loaders,
    test_loaders,
    task_ids,
    task_weights,
    epochs,
    base_lr,
    electrical_lr_scale,
    weight_decay,
    device,
    capture_best_state=False,
    track_train_accuracy=False,
):
    model.to(device)
    shared_optical_params = []
    private_delta_params = []
    electrical_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "mzi_layers" in name:
            if ".delta_" in name:
                private_delta_params.append(param)
            else:
                shared_optical_params.append(param)
        else:
            electrical_params.append(param)

    param_groups = []
    if shared_optical_params:
        param_groups.append({"params": shared_optical_params, "lr": base_lr, "weight_decay": weight_decay})
    if private_delta_params:
        param_groups.append(
            {
                "params": private_delta_params,
                "lr": base_lr * getattr(model, "private_delta_lr_scale", 1.0),
                "weight_decay": getattr(model, "private_delta_weight_decay", weight_decay),
            }
        )
    if electrical_params:
        param_groups.append({"params": electrical_params, "lr": base_lr * electrical_lr_scale, "weight_decay": weight_decay})
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()

    mean_hist = []
    train_loss_hist = []
    train_mean_hist = []
    train_task_hist = {str(tid): [] for tid in task_ids}
    task_hist = {str(tid): [] for tid in task_ids}
    best_state_dict = None
    best_seen = -1.0
    for epoch in range(epochs):
        model.train()
        if getattr(model, "freeze_electrical_runtime", False) and hasattr(model, "set_electrical_eval"):
            model.set_electrical_eval()
        loss_sum = 0.0
        loss_count = 0
        for tid in task_ids:
            weight = task_weights.get(str(tid), 1.0)
            for x, y in train_loaders[tid]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x, tid), y) * weight
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu().item()) * y.numel()
                loss_count += y.numel()
        scheduler.step()
        if track_train_accuracy:
            train_mean_acc, train_task_accs = evaluate(model, train_loaders, task_ids, device)
            train_mean_hist.append(train_mean_acc)
            for tid in task_ids:
                train_task_hist[str(tid)].append(train_task_accs[str(tid)])
        mean_acc, task_accs = evaluate(model, test_loaders, task_ids, device)
        mean_hist.append(mean_acc)
        train_loss_hist.append(loss_sum / loss_count if loss_count else 0.0)
        for tid in task_ids:
            task_hist[str(tid)].append(task_accs[str(tid)])
        if capture_best_state and mean_acc > best_seen:
            best_seen = mean_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch + 1:03d}/{epochs}: mean={mean_acc:.4f}")
    best_mean = max(mean_hist) if mean_hist else 0.0
    best_epoch = int(mean_hist.index(best_mean) + 1) if mean_hist else 0
    result = {
        "mean_hist": mean_hist,
        "train_loss_hist": train_loss_hist,
        "task_hist": task_hist,
        "final_mean": mean_hist[-1] if mean_hist else 0.0,
        "best_mean": best_mean,
        "best_epoch": best_epoch,
    }
    if capture_best_state:
        result["_best_state_dict"] = best_state_dict
    if track_train_accuracy:
        result["train_mean_hist"] = train_mean_hist
        result["train_task_hist"] = train_task_hist
    return result


def score_common_mzi(model, train_loaders, task_ids, max_batches, device, score_method="gradient", phase_bits=3):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    grads = [
        {
            "even": [],
            "odd": [],
        }
        for _ in model.mzi_layers
    ]
    fishers = [
        {
            "even": [],
            "odd": [],
        }
        for _ in model.mzi_layers
    ]

    for tid in task_ids:
        model.zero_grad(set_to_none=True)
        batches = 0
        for x, y in train_loaders[tid]:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x, tid), y)
            loss.backward()
            batches += 1
            if batches >= max_batches:
                break
        for li, layer in enumerate(model.mzi_layers):
            even_theta_grad = layer.theta_even.grad.detach()
            even_phi_grad = layer.phi_even.grad.detach()
            odd_theta_grad = layer.theta_odd.grad.detach()
            odd_phi_grad = layer.phi_odd.grad.detach()
            even_grad = even_theta_grad.abs() + even_phi_grad.abs()
            odd_grad = odd_theta_grad.abs() + odd_phi_grad.abs()
            even_fisher = even_theta_grad.pow(2) + even_phi_grad.pow(2)
            odd_fisher = odd_theta_grad.pow(2) + odd_phi_grad.pow(2)
            grads[li]["even"].append(even_grad.cpu().clone())
            grads[li]["odd"].append(odd_grad.cpu().clone())
            fishers[li]["even"].append(even_fisher.cpu().clone())
            fishers[li]["odd"].append(odd_fisher.cpu().clone())

    scores = []
    for layer_grads in grads:
        layer_score = {}
        for kind in ("even", "odd"):
            g = torch.stack(layer_grads[kind], dim=0)
            utility = g.mean(dim=0)
            conflict = g.std(dim=0, unbiased=False) / (utility + 1e-8)
            score = utility / (1.0 + conflict)
            layer_score[kind] = score
        scores.append(layer_score)
    if score_method == "gradient":
        return scores
    if score_method not in {"gradient_consensus", "damage_aware", "fisher_damage", "quantized_fisher_damage"}:
        raise ValueError(f"Unknown score_method: {score_method}")

    for li, layer in enumerate(model.mzi_layers):
        for kind in ("even", "odd"):
            if kind == "even":
                theta = layer.theta_even
                phi = layer.phi_even
                delta_theta = layer.delta_theta_even
                delta_phi = layer.delta_phi_even
            else:
                theta = layer.theta_odd
                phi = layer.phi_odd
                delta_theta = layer.delta_theta_odd
                delta_phi = layer.delta_phi_odd
            effective_theta = theta.detach().cpu().unsqueeze(0) + delta_theta.detach().cpu()[task_ids]
            effective_phi = phi.detach().cpu().unsqueeze(0) + delta_phi.detach().cpu()[task_ids]
            theta_c = torch.cos(effective_theta).mean(dim=0)
            theta_s = torch.sin(effective_theta).mean(dim=0)
            phi_c = torch.cos(effective_phi).mean(dim=0)
            phi_s = torch.sin(effective_phi).mean(dim=0)
            theta_agreement = torch.sqrt(theta_c * theta_c + theta_s * theta_s).clamp(0.0, 1.0)
            phi_agreement = torch.sqrt(phi_c * phi_c + phi_s * phi_s).clamp(0.0, 1.0)
            phase_agreement = 0.5 * (theta_agreement + phi_agreement)
            consensus_weight = 0.10 + 0.90 * phase_agreement.pow(2)
            if score_method == "gradient_consensus":
                scores[li][kind] = scores[li][kind] * consensus_weight
                continue

            theta_mean = torch.atan2(theta_s, theta_c)
            phi_mean = torch.atan2(phi_s, phi_c)
            if score_method == "quantized_fisher_damage":
                theta_levels = 1 << phase_bits
                phi_levels = 1 << phase_bits
                theta_step = math.pi / theta_levels
                theta_mean = torch.round(theta_mean / theta_step).clamp(0, theta_levels - 1) * theta_step
                phi_mean = torch.remainder(
                    torch.round(torch.remainder(phi_mean, 2 * math.pi) * phi_levels / (2 * math.pi)),
                    phi_levels,
                ) * (2 * math.pi / phi_levels)
            theta_cost = torch.atan2(
                torch.sin(effective_theta - theta_mean.unsqueeze(0)),
                torch.cos(effective_theta - theta_mean.unsqueeze(0)),
            ).pow(2).mean(dim=0)
            phi_cost = torch.atan2(
                torch.sin(effective_phi - phi_mean.unsqueeze(0)),
                torch.cos(effective_phi - phi_mean.unsqueeze(0)),
            ).pow(2).mean(dim=0)
            adapter_removal_cost = 0.5 * (theta_cost + phi_cost)
            if score_method in {"fisher_damage", "quantized_fisher_damage"}:
                fisher = torch.stack(fishers[li][kind], dim=0)
                utility = torch.sqrt(fisher.mean(dim=0) + 1e-12)
                utility_norm = utility / (utility.mean() + 1e-8)
                theta_dist = torch.atan2(
                    torch.sin(effective_theta - theta_mean.unsqueeze(0)),
                    torch.cos(effective_theta - theta_mean.unsqueeze(0)),
                ).pow(2)
                phi_dist = torch.atan2(
                    torch.sin(effective_phi - phi_mean.unsqueeze(0)),
                    torch.cos(effective_phi - phi_mean.unsqueeze(0)),
                ).pow(2)
                replacement_damage = (fisher * 0.5 * (theta_dist + phi_dist)).mean(dim=0)
                damage_norm = replacement_damage / (replacement_damage.mean() + 1e-12)
                scores[li][kind] = consensus_weight * (0.25 + utility_norm) / (0.25 + damage_norm)
                continue

            damage_weight = 1.0 / (1.0 + 4.0 * adapter_removal_cost)
            scores[li][kind] = scores[li][kind] * consensus_weight * damage_weight
    return scores


def freeze_electrical_parts(model):
    for name, param in model.named_parameters():
        if "mzi_layers" not in name:
            param.requires_grad = False
    if hasattr(model, "freeze_electrical_runtime"):
        model.freeze_electrical_runtime = True
    if hasattr(model, "set_electrical_eval"):
        model.set_electrical_eval()


def build_optical_model(args, task_classes, use_efficiency):
    model = SharedMZINet(
        task_classes=task_classes,
        input_rank=args.input_rank,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_efficiency=use_efficiency,
        efficiency_min=args.efficiency_min,
        head_mode=args.head_mode,
        global_classes=args.global_classes,
        photonic_readout=args.photonic_readout,
        norm_type=args.norm_type,
        input_dim=args.model_input_dim,
        input_projection=args.input_projection,
        phase_quantization=args.phase_quantization,
        phase_bits=args.phase_bits,
    )
    model.private_delta_lr_scale = args.private_delta_lr_scale
    model.private_delta_weight_decay = args.private_delta_weight_decay
    return model


def save_results(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def checkpoint_path(args, condition_name):
    checkpoint_dir = ROOT / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "warmup_checkpoint", None):
        warmup_path = Path(args.warmup_checkpoint)
        if not warmup_path.is_absolute():
            warmup_path = ROOT / warmup_path
        warmup_path.parent.mkdir(parents=True, exist_ok=True)
        return warmup_path
    stem = Path(args.output).stem
    return checkpoint_dir / f"{stem}_{condition_name}_warmup.pt"


def condition_checkpoint_path(args, condition_name, ratio, mode):
    checkpoint_dir = ROOT / args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.output).stem
    ratio_tag = f"r{int(round(ratio * 100)):03d}"
    mode_tag = mode.replace("/", "_")
    return checkpoint_dir / f"{stem}_{condition_name}_{mode_tag}_{ratio_tag}_final.pt"


def load_existing_payload(args):
    output_path = ROOT / args.output
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        print(f"Resuming from {output_path.relative_to(ROOT)}")
        return payload
    return None


def already_done(container, key):
    return key in container and isinstance(container[key], dict) and "final_mean" in container[key]


def run_condition(args, model, masks, train_loaders, test_loaders, task_ids, device):
    model.apply_shared_masks(masks)
    if args.freeze_electrical_after_warmup:
        freeze_electrical_parts(model)
    result = train_model(
        model,
        train_loaders,
        test_loaders,
        task_ids,
        args.task_weights,
        args.epochs,
        args.base_lr,
        args.electrical_lr_scale,
        args.weight_decay,
        device,
        track_train_accuracy=args.track_train_accuracy,
    )
    result["mask_profile"] = mask_profile(masks)
    result["quantization_profile"] = quantization_profile(model, masks)
    if args.save_mask_matrices:
        result["mask_matrices"] = serialize_masks(masks)
    if args.save_confusion_matrices:
        result["confusion_matrices"] = compute_confusion_matrices(model, test_loaders, task_ids, device)
    return result


def run_experiment(args):
    set_seed(args.seed)
    cuda_available = torch.cuda.is_available()
    if args.require_cuda and (args.cpu or not cuda_available):
        raise RuntimeError(
            "CUDA was required for this shard, but torch.cuda.is_available() is false. "
            "Check CUDA_VISIBLE_DEVICES, driver status, and GPU allocation before rerunning."
        )
    device = "cuda" if cuda_available and not args.cpu else "cpu"
    label_mode = "global" if args.head_mode == "shared_global" else "local"
    specs, train_loaders, test_loaders, preprocess_profile = get_task_loaders(
        args.task_set,
        args.train_samples,
        args.test_samples,
        args.batch_size,
        args.seed,
        download=args.download,
        label_mode=label_mode,
        input_preprocess=args.input_preprocess,
        input_rank=args.input_rank,
    )
    args.model_input_dim = int(preprocess_profile["input_dim"])
    args.input_projection = "direct" if args.input_preprocess == "pca" else "learned"
    if args.input_projection == "direct" and args.model_input_dim != args.hidden_dim:
        raise ValueError("--input-preprocess pca requires --input-rank == --hidden-dim for direct PCA-to-photonic input.")
    task_ids = [s["id"] for s in specs]
    task_classes = [s["classes"] for s in specs]
    args.task_weights = {str(tid): 1.0 for tid in task_ids}
    if args.task_set == "letters":
        args.task_weights = {"0": 1.15, "1": 1.0}

    output_path = ROOT / args.output
    payload = load_existing_payload(args)
    if payload is None:
        payload = {
            "settings": vars(args),
            "device": device,
            "task_specs": specs,
            "data_profile": data_profile(specs, train_loaders, test_loaders, label_mode),
            "preprocess_profile": preprocess_profile,
            "model_profile": model_profile(args),
            "baselines": {},
            "efficiency_conditions": {},
        }
    else:
        payload.setdefault("settings", vars(args))
        payload["device"] = device
        payload["task_specs"] = specs
        payload["data_profile"] = data_profile(specs, train_loaders, test_loaders, label_mode)
        payload["preprocess_profile"] = preprocess_profile
        payload["model_profile"] = model_profile(args)
        payload.setdefault("baselines", {})
        payload.setdefault("efficiency_conditions", {})

    print(f"Using device: {device}")
    if args.skip_electronic_baselines:
        print("Skipping electronic baselines (--skip-electronic-baselines)")
    else:
        if already_done(payload["baselines"], "electronic_shallow"):
            print("Skipping electronic shallow baseline (already done)")
        else:
            print("Training electronic shallow baseline")
            set_seed(args.seed + 10)
            shallow = MultiTaskMLP(
                [args.shallow_hidden],
                task_classes,
                dropout=0.0,
                head_mode=args.head_mode,
                global_classes=args.global_classes,
                input_dim=args.model_input_dim,
            )
            shallow_result = train_model(
                shallow,
                train_loaders,
                test_loaders,
                task_ids,
                args.task_weights,
                args.baseline_epochs,
                args.baseline_lr,
                1.0,
                args.weight_decay,
                device,
            )
            if args.save_confusion_matrices:
                shallow_result["confusion_matrices"] = compute_confusion_matrices(shallow, test_loaders, task_ids, device)
            payload["baselines"]["electronic_shallow"] = shallow_result
            save_results(output_path, payload)

        if already_done(payload["baselines"], "electronic_deep"):
            print("Skipping electronic deep baseline (already done)")
        else:
            print("Training electronic deep baseline")
            set_seed(args.seed + 20)
            deep = MultiTaskMLP(
                args.deep_hidden,
                task_classes,
                dropout=args.deep_dropout,
                head_mode=args.head_mode,
                global_classes=args.global_classes,
                input_dim=args.model_input_dim,
            )
            deep_result = train_model(
                deep,
                train_loaders,
                test_loaders,
                task_ids,
                args.task_weights,
                args.baseline_epochs,
                args.baseline_lr,
                1.0,
                args.weight_decay,
                device,
            )
            if args.save_confusion_matrices:
                deep_result["confusion_matrices"] = compute_confusion_matrices(deep, test_loaders, task_ids, device)
            payload["baselines"]["electronic_deep"] = deep_result
            save_results(output_path, payload)

    ratios = [r / 100.0 for r in args.ratios]
    for eff_name, use_eff in args.efficiency_conditions:
        print("\n" + "=" * 72)
        print(f"Efficiency condition: {eff_name}")
        print("=" * 72)
        condition = payload["efficiency_conditions"].setdefault(
            eff_name,
            {
                "use_efficiency": use_eff,
                "warmup": None,
                "optical_unfrozen": None,
                "curves": {},
            },
        )
        condition["use_efficiency"] = use_eff
        condition.setdefault("curves", {})

        set_seed(args.seed + 100 + (1 if use_eff else 0))
        warmup = build_optical_model(args, task_classes, use_eff)
        if args.warmup_shared_ratio >= 100.0:
            warmup_masks = empty_masks(args.hidden_dim, args.num_layers, value=1.0)
        elif args.warmup_shared_ratio <= 0.0:
            warmup_masks = empty_masks(args.hidden_dim, args.num_layers, value=0.0)
        else:
            warmup_masks = make_random_node_masks(
                args.warmup_shared_ratio / 100.0,
                args.hidden_dim,
                args.num_layers,
                args.seed + 777 + (1 if use_eff else 0),
            )
        warmup.apply_shared_masks(warmup_masks)
        ckpt = checkpoint_path(args, eff_name)
        if (args.resume or args.warmup_checkpoint) and ckpt.exists():
            warmup.load_state_dict(torch.load(ckpt, map_location=device))
            warmup.to(device)
            print(f"Loaded warmup checkpoint: {ckpt.relative_to(ROOT)}")
            if condition.get("warmup") is None or "final_mean" not in condition.get("warmup", {}):
                mean_acc, task_accs = evaluate(warmup, test_loaders, task_ids, device)
                warmup_result = {
                    "mean_hist": [mean_acc],
                    "task_hist": {str(k): [v] for k, v in task_accs.items()},
                    "final_mean": mean_acc,
                    "best_mean": mean_acc,
                    "best_epoch": 0,
                }
        elif args.pretrain_epochs > 0:
            print(f"Training {args.warmup_shared_ratio:.1f}% shared warmup")
            warmup_result = train_model(
                warmup,
                train_loaders,
                test_loaders,
                task_ids,
                args.task_weights,
                args.pretrain_epochs,
                args.base_lr,
                args.electrical_lr_scale,
                args.weight_decay,
                device,
                capture_best_state=args.use_best_warmup_state,
                track_train_accuracy=args.track_train_accuracy,
            )
            best_warmup_state = warmup_result.pop("_best_state_dict", None)
            if args.use_best_warmup_state and best_warmup_state is not None:
                warmup.load_state_dict(best_warmup_state)
                warmup.to(device)
                warmup_result["selected_state"] = "best_mean_checkpoint"
            else:
                warmup_result["selected_state"] = "final_epoch_checkpoint"
            torch.save(warmup.state_dict(), ckpt)
            print(f"Saved warmup checkpoint: {ckpt.relative_to(ROOT)}")
        else:
            mean_acc, task_accs = evaluate(warmup.to(device), test_loaders, task_ids, device)
            warmup_result = {
                "mean_hist": [mean_acc],
                "task_hist": {str(k): [v] for k, v in task_accs.items()},
                "final_mean": mean_acc,
                "best_mean": mean_acc,
                "best_epoch": 0,
                "selected_state": "initial_checkpoint",
            }
            torch.save(warmup.state_dict(), ckpt)
            print(f"Saved warmup checkpoint: {ckpt.relative_to(ROOT)}")
        if condition.get("warmup") is None or "final_mean" not in condition.get("warmup", {}):
            warmup_result["mask_profile"] = mask_profile(warmup_masks)
            if args.save_mask_matrices:
                warmup_result["mask_matrices"] = serialize_masks(warmup_masks)
            if args.save_confusion_matrices:
                warmup_result["confusion_matrices"] = compute_confusion_matrices(warmup, test_loaders, task_ids, device)
            condition["warmup"] = warmup_result
            save_results(output_path, payload)
        elif args.resume and ckpt.exists():
            print("Skipping warmup result write (already done)")

        if args.warmup_only:
            print(f"Warmup-only run complete. Checkpoint: {ckpt.relative_to(ROOT)}")
            print(f"\nDone. Results written to {output_path.relative_to(ROOT)}")
            return

        warmup_state = copy.deepcopy(warmup.state_dict())
        score_model = warmup
        unfrozen = None

        if args.skip_optical_unfrozen:
            print("Skipping pure photonic/unfrozen reference (--skip-optical-unfrozen)")
        elif condition.get("optical_unfrozen") and "final_mean" in condition["optical_unfrozen"]:
            print("Skipping pure photonic/unfrozen reference (already done)")
        else:
            print("Training pure photonic/unfrozen reference")
            set_seed(args.seed + 200 + (1 if use_eff else 0))
            unfrozen = build_optical_model(args, task_classes, use_eff)
            if args.init_from_warmup:
                unfrozen.load_state_dict(warmup_state)
            unfrozen_masks = empty_masks(args.hidden_dim, args.num_layers, value=0.0)
            condition["optical_unfrozen"] = run_condition(
                args, unfrozen, unfrozen_masks, train_loaders, test_loaders, task_ids, device
            )
            save_results(output_path, payload)
        if args.score_source == "unfrozen_probe":
            if unfrozen is not None:
                score_model = unfrozen
            else:
                print("Warning: unfrozen probe state is unavailable in this resumed shard; scoring from warmup instead.")

        print(f"Scoring MZI commonness for pruning-guided node masks from {args.score_source} with {args.score_method}")
        scores = score_common_mzi(
            score_model,
            train_loaders,
            task_ids,
            args.score_batches,
            device,
            score_method=args.score_method,
            phase_bits=args.phase_bits,
        )
        condition["pruning_score_source"] = args.score_source if score_model is not warmup else "warmup"
        condition["pruning_score_method"] = args.score_method
        condition["pruning_score_profile"] = pruning_score_profile(scores)
        save_results(output_path, payload)

        for ratio in ratios:
            ratio_key = f"{ratio:.2f}"
            condition["curves"].setdefault(ratio_key, {})
            if ratio <= 0.0 or ratio >= 1.0:
                boundary_name = "private_0pct" if ratio <= 0.0 else "shared_100pct"
                if all(already_done(condition["curves"][ratio_key], mode) for mode in args.modes):
                    print(f"Skipping boundary shared_ratio={ratio:.2f} efficiency={eff_name} (already done)")
                    continue
                print("\n" + "-" * 64)
                print(f"Boundary={boundary_name} shared_ratio={ratio:.2f} efficiency={eff_name}")
                print("-" * 64)
                set_seed(args.seed + int(ratio * 1000) + 997 + (1 if use_eff else 0))
                model = build_optical_model(args, task_classes, use_eff)
                if args.init_from_warmup:
                    model.load_state_dict(warmup_state)
                masks = empty_masks(args.hidden_dim, args.num_layers, value=1.0 if ratio >= 1.0 else 0.0)
                result = run_condition(args, model, masks, train_loaders, test_loaders, task_ids, device)
                result["boundary"] = boundary_name
                if args.save_condition_checkpoints:
                    ckpt_path = condition_checkpoint_path(args, eff_name, ratio, "boundary")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "masks": serialize_masks(masks),
                            "ratio": ratio,
                            "mode": "boundary",
                            "efficiency_condition": eff_name,
                            "model_profile": model_profile(args),
                            "task_specs": specs,
                        },
                        ckpt_path,
                    )
                    result["checkpoint_path"] = str(ckpt_path.relative_to(ROOT))
                for mode in args.modes:
                    if not already_done(condition["curves"][ratio_key], mode):
                        condition["curves"][ratio_key][mode] = copy.deepcopy(result)
                if ratio <= 0.0:
                    condition["optical_unfrozen"] = copy.deepcopy(result)
                save_results(output_path, payload)
                continue
            for mode in args.modes:
                if already_done(condition["curves"][ratio_key], mode):
                    print(f"Skipping Mode={mode} shared_ratio={ratio:.2f} efficiency={eff_name} (already done)")
                    continue
                print("\n" + "-" * 64)
                print(f"Mode={mode} shared_ratio={ratio:.2f} efficiency={eff_name}")
                print("-" * 64)
                set_seed(args.seed + int(ratio * 1000) + len(mode) * 13 + (1 if use_eff else 0))
                model = build_optical_model(args, task_classes, use_eff)
                if args.init_from_warmup:
                    model.load_state_dict(warmup_state)
                if mode == "node_random":
                    masks = make_random_node_masks(ratio, args.hidden_dim, args.num_layers, args.seed + 301)
                elif mode == "node_pruning":
                    if args.guided_node_prior == "stage_refine":
                        masks = make_stage_refined_node_masks(
                            ratio,
                            scores,
                            args.hidden_dim,
                            args.num_layers,
                            refine_fraction=args.guided_node_refine_fraction,
                        )
                    else:
                        masks = make_guided_node_masks(
                            ratio,
                            scores,
                            args.hidden_dim,
                            args.num_layers,
                            balance=args.guided_node_balance,
                        )
                elif mode == "block":
                    masks = make_block_masks(
                        ratio,
                        args.hidden_dim,
                        args.num_layers,
                        args.seed + 302,
                        block_h=args.block_h,
                        block_w=args.block_w,
                    )
                elif mode == "clements_layer":
                    if args.clements_layer_mode == "stage":
                        masks = make_clements_stage_masks(ratio, args.hidden_dim, args.num_layers)
                    elif args.clements_layer_mode == "depth":
                        masks = make_clements_depth_masks(ratio, args.hidden_dim, args.num_layers)
                    else:
                        masks = make_clements_slice_masks(ratio, args.hidden_dim, args.num_layers, args.seed + 303)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                result = run_condition(args, model, masks, train_loaders, test_loaders, task_ids, device)
                if args.save_condition_checkpoints:
                    ckpt_path = condition_checkpoint_path(args, eff_name, ratio, mode)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "masks": serialize_masks(masks),
                            "ratio": ratio,
                            "mode": mode,
                            "efficiency_condition": eff_name,
                            "model_profile": model_profile(args),
                            "task_specs": specs,
                        },
                        ckpt_path,
                    )
                    result["checkpoint_path"] = str(ckpt_path.relative_to(ROOT))
                condition["curves"][ratio_key][mode] = result
                save_results(output_path, payload)

    print(f"\nDone. Results written to {output_path.relative_to(ROOT)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run homogeneous main-figure shared-MPU-ratio experiments.")
    default_smoke_output = "results/main_figure_freeze_ratio_experiment_smoke.json"
    parser.add_argument("--preset", choices=["smoke", "main"], default="smoke")
    parser.add_argument("--output", default=default_smoke_output)
    parser.add_argument("--task-set", choices=["letters", "three"], default="letters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--checkpoint-dir", default="results/main_figure_checkpoints")
    parser.add_argument(
        "--warmup-checkpoint",
        default=None,
        help="Optional shared warmup checkpoint path. If present, shards load/save this exact warmup state.",
    )
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="Train or load only the warmup model, save its checkpoint/result, then exit before ratio conditions.",
    )
    parser.add_argument("--ratios", type=float, nargs="+", default=None, help="Percent values, e.g. 0 20 40 60 80 90")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["node_random", "node_pruning", "block", "clements_layer"],
    )
    parser.add_argument("--efficiency", choices=["both", "off", "on"], default="both")
    parser.add_argument(
        "--phase-quantization",
        choices=["none", "shared", "all"],
        default="none",
        help="none: continuous phases; shared: QAT only for shared MPU MZIs; all: QAT for every MZI.",
    )
    parser.add_argument("--phase-bits", type=int, default=3, help="Bits per theta and per phi phase control.")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--input-rank", type=int, default=None)
    parser.add_argument(
        "--input-preprocess",
        choices=["learned", "pca"],
        default="learned",
        help="learned: trainable electrical 784->rank->hidden projection; pca: fixed PCA features directly feed the photonic mesh.",
    )
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--baseline-epochs", type=int, default=None)
    parser.add_argument("--base-lr", type=float, default=0.0035)
    parser.add_argument("--baseline-lr", type=float, default=0.0015)
    parser.add_argument("--electrical-lr-scale", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--private-delta-lr-scale", type=float, default=1.0)
    parser.add_argument("--private-delta-weight-decay", type=float, default=1e-4)
    parser.add_argument("--score-batches", type=int, default=None)
    parser.add_argument("--efficiency-min", type=float, default=0.65)
    parser.add_argument(
        "--warmup-shared-ratio",
        type=float,
        default=100.0,
        help="Percent of MZIs shared during warmup. Use 0 or 20 when 100% shared warmup is too hard.",
    )
    parser.add_argument("--init-from-warmup", dest="init_from_warmup", action="store_true", default=True)
    parser.add_argument("--no-init-from-warmup", dest="init_from_warmup", action="store_false")
    parser.add_argument("--use-best-warmup-state", dest="use_best_warmup_state", action="store_true", default=True)
    parser.add_argument("--use-final-warmup-state", dest="use_best_warmup_state", action="store_false")
    parser.add_argument("--head-mode", choices=["task_specific", "shared_global"], default="task_specific")
    parser.add_argument("--global-classes", type=int, default=62)
    parser.add_argument(
        "--photonic-readout",
        choices=["per_mesh_coherent", "per_mesh_intensity", "coherent_stack"],
        default="per_mesh_intensity",
        help="per_mesh_intensity is the physical square-law O/E/O main path; per_mesh_coherent and coherent_stack are ablations.",
    )
    parser.add_argument(
        "--norm-type",
        choices=["layer", "batch", "none"],
        default="layer",
        help="Normalization after each optical mesh. LayerNorm avoids cross-task BatchNorm running-stat leakage.",
    )
    parser.add_argument("--freeze-electrical-after-warmup", action="store_true")
    parser.add_argument("--skip-electronic-baselines", action="store_true")
    parser.add_argument("--skip-optical-unfrozen", action="store_true")
    parser.add_argument("--score-source", choices=["warmup", "unfrozen_probe"], default="warmup")
    parser.add_argument(
        "--score-method",
        choices=["gradient", "gradient_consensus", "damage_aware", "fisher_damage", "quantized_fisher_damage"],
        default="gradient",
        help="damage_aware/fisher_damage prefer MZIs whose task-private phase adapters can be removed with low estimated loss.",
    )
    parser.add_argument(
        "--guided-node-balance",
        choices=["global", "layer", "kind"],
        default="global",
        help="Quota rule for pruning-guided node masks. 'kind' balances each Clements mesh and even/odd MZI family.",
    )
    parser.add_argument(
        "--guided-node-prior",
        choices=["none", "stage_refine"],
        default="none",
        help="Optional prior for node pruning. stage_refine starts from front physical columns and swaps in higher-score nodes.",
    )
    parser.add_argument(
        "--guided-node-refine-fraction",
        type=float,
        default=0.35,
        help="Maximum fraction of selected front-stage nodes that stage_refine may swap.",
    )
    parser.add_argument(
        "--clements-layer-mode",
        choices=["slice", "depth", "stage"],
        default="slice",
        help="Layer baseline: random intra-mesh slices, or deterministic shared front-depth cascade.",
    )
    parser.add_argument("--block-h", type=int, default=6, help="Physical block height for the block-sharing baseline.")
    parser.add_argument("--block-w", type=int, default=6, help="Physical block width for the block-sharing baseline.")
    parser.add_argument("--save-mask-matrices", action="store_true", help="Store full shared/private MZI masks for figure panels.")
    parser.add_argument(
        "--track-train-accuracy",
        action="store_true",
        help="Evaluate and store train accuracy curves each epoch for overfit/underfit diagnostics.",
    )
    parser.add_argument(
        "--save-condition-checkpoints",
        action="store_true",
        help="Save final model checkpoints for each ratio/mode condition for later test-set diagnostics.",
    )
    parser.add_argument(
        "--save-confusion-matrices",
        action="store_true",
        help="Store final test confusion matrices for baselines and optical conditions.",
    )
    parser.add_argument("--shallow-hidden", type=int, default=64)
    parser.add_argument("--deep-hidden", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--deep-dropout", type=float, default=0.10)
    args = parser.parse_args()

    if args.phase_bits < 1:
        parser.error("--phase-bits must be at least 1")
    if args.score_method == "quantized_fisher_damage" and args.phase_quantization == "none":
        parser.error("quantized_fisher_damage requires --phase-quantization shared or all")

    if args.preset == "smoke":
        args.output = args.output or "results/main_figure_freeze_ratio_experiment_smoke.json"
        args.hidden_dim = args.hidden_dim if args.hidden_dim is not None else 16
        args.num_layers = args.num_layers if args.num_layers is not None else 1
        args.input_rank = args.input_rank if args.input_rank is not None else 8
        args.train_samples = args.train_samples if args.train_samples is not None else 256
        args.test_samples = args.test_samples if args.test_samples is not None else 128
        args.batch_size = args.batch_size if args.batch_size is not None else 128
        args.epochs = args.epochs if args.epochs is not None else 2
        args.pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs is not None else 1
        args.baseline_epochs = args.baseline_epochs if args.baseline_epochs is not None else 2
        args.score_batches = args.score_batches if args.score_batches is not None else 1
        args.ratios = args.ratios or [0, 80]
    else:
        args.output = "results/main_figure_freeze_ratio_experiment_main.json" if args.output == default_smoke_output else args.output
        args.hidden_dim = args.hidden_dim if args.hidden_dim is not None else 72
        args.num_layers = args.num_layers if args.num_layers is not None else 3
        args.input_rank = args.input_rank if args.input_rank is not None else 32
        args.train_samples = args.train_samples if args.train_samples is not None else 10000
        args.test_samples = args.test_samples if args.test_samples is not None else 3000
        args.batch_size = args.batch_size if args.batch_size is not None else 512
        args.epochs = args.epochs if args.epochs is not None else 80
        args.pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs is not None else 35
        args.baseline_epochs = args.baseline_epochs if args.baseline_epochs is not None else 80
        args.score_batches = args.score_batches if args.score_batches is not None else 4
        args.ratios = args.ratios or [0, 20, 40, 60, 70, 80, 100]

    if args.train_samples == 0:
        args.train_samples = None
    if args.test_samples == 0:
        args.test_samples = None

    if args.efficiency == "both":
        args.efficiency_conditions = [("eta_off", False), ("eta_on", True)]
    elif args.efficiency == "on":
        args.efficiency_conditions = [("eta_on", True)]
    else:
        args.efficiency_conditions = [("eta_off", False)]
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
