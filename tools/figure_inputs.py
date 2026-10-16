"""Shared input selection for plots and MATLAB exports (no implicit seed mixing)."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("node_pruning", "clements_layer")


def pattern_ratios(cond, requested=None):
    available = []
    for key, modes in cond["curves"].items():
        if all(modes.get(mode, {}).get("mask_matrices") for mode in MODES):
            available.append(float(key) * 100)
    available.sort()
    preferred = [10, 30, 50, 70, 90] if requested is None else requested
    selected = [r for r in preferred if any(abs(r - a) < 1e-6 for a in available)]
    if requested is None and not selected:
        selected = [r for r in available if 0 < r < 100] or available
    return selected


def prepare_payload(payload, input_path, warmup_path=None):
    """Use a supplied warmup, or the exact matching published input's companion."""
    payload = copy.deepcopy(payload)
    conds = payload["efficiency_conditions"]
    cond = conds.get("eta_off", next(iter(conds.values())))
    if warmup_path is None and len((cond.get("warmup") or {}).get("mean_hist", [])) > 1:
        return payload
    if warmup_path is None:
        companions = {
            (ROOT / "data/main_10pct_sweep.json").resolve(): ROOT / "data/warmup_main.json",
            (ROOT / "data/replicate_20pct_sweep.json").resolve(): ROOT / "data/warmup_replicate.json",
        }
        warmup_path = companions.get(Path(input_path).resolve())
    if warmup_path is None:
        return payload
    source = Path(warmup_path)
    external = json.loads(source.read_text(encoding="utf-8-sig"))
    if external.get("settings", {}).get("seed") != payload.get("settings", {}).get("seed"):
        raise ValueError("Warmup seed differs from the selected sweep")
    external_conds = external["efficiency_conditions"]
    external_cond = external_conds.get("eta_off", next(iter(external_conds.values())))
    warmup = external_cond.get("warmup") or {}
    if len(warmup.get("mean_hist", [])) <= 1:
        raise ValueError("The supplied warmup JSON does not contain a full training history")
    cond["warmup"] = warmup
    cond["warmup_source"] = str(source)
    return payload
