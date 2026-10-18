"""Validate raw training results, export Fig. 5 source data, and plot panel d."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reproduction_data/fig5"
ORDER = ["Continuous", "2-bit QAT", "3-bit QAT", "3-bit PTQ",
         "4-bit QAT", "5-bit QAT", "Measured-error simulation",
         "Library + MZI voltage error"]
COLORS = ["#555F6B", "#B54159", "#2F6F9F", "#2F6F9F",
          "#70558F", "#A18AB8", "#8F3B61", "#57417D"]
MARKERS = ["o", "s", "D", "X", "^", "P", "*", "o"]
LABELS = ORDER[:6] + [
    "64-state codebook (90% transmission, 3.32-bit error)",
    "64-state codebook + 10 mV MZI drive error"]


def basename(source):
    return str(source).replace("\\", "/").split("/")[-1]


def result_at(payload, ratio):
    curves = payload["efficiency_conditions"]["eta_off"]["curves"]
    return next(v["node_pruning"] for k, v in curves.items()
                if np.isclose(float(k), ratio))


def validated_runs():
    runs = pd.read_csv(DATA / "quantization_runs.csv")
    expected_seeds = {"Continuous": 3, "2-bit QAT": 1, "3-bit QAT": 3,
                      "3-bit PTQ": 1, "4-bit QAT": 3, "5-bit QAT": 2,
                      ORDER[6]: 1, ORDER[7]: 1}
    assert not runs.duplicated(["condition", "sharing_ratio", "seed"]).any()
    with np.load(DATA / "device_runs/codebook_static_3bit_eps_2pow_minus332_losst90pct.npz") as z:
        m, u = z["matrices"].astype(complex), z["base_matrices"].astype(complex)
    assert np.allclose(np.linalg.norm(m, axis=(1, 2))**2 / 2, .90, atol=1e-7)
    assert np.linalg.svd(m, compute_uv=False).max() <= 1 + 1e-6
    residual = np.linalg.norm(m / np.sqrt(.9) - u, axis=(1, 2)) / np.linalg.norm(u, axis=(1, 2))
    assert np.allclose(residual, 2**(-3.32), atol=1e-7)
    checked = {}
    for index, row in runs.iterrows():
        device = row.condition in ORDER[6:]
        relative = Path("device_runs" if device else "nominal_runs") / basename(row.source)
        source = DATA / relative
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        assert payload["settings"]["seed"] == row.seed, source
        result = result_at(payload, row.sharing_ratio)
        value = result["final_mean"]
        if row.condition == ORDER[7]:
            paired = result["paired_mzi_voltage_error"]
            assert paired["test_noise_draws"] == 5
            assert not paired["additional_finetuning"]
            assert np.isclose(paired["voltage_rms_V"], .01)
            value = paired["mean_accuracy"]
            assert np.isclose(value, np.mean([d["mean_accuracy"] for d in paired["draw_results"]]))
        if device:
            meta = payload["static_332_t90pct_codebook"]
            assert meta["codebook_entries"] == 64
            assert np.isclose(meta["power_transmission_per_shared_mpu"], .9)
            assert np.isclose(meta["relative_matrix_error_frobenius"], 2**(-3.32))
        assert np.isclose(value * 100, row.accuracy_percent, atol=1e-6), (source, row.condition, row.sharing_ratio, value, row.accuracy_percent)
        runs.at[index, "source"] = relative.as_posix()
        checked[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    summary = runs.groupby(["condition", "sharing_ratio"], sort=False).accuracy_percent.agg(
        mean_accuracy_percent="mean", sample_sd_percent="std", training_seed_count="count").reset_index()
    for condition in ORDER:
        group = summary[summary.condition == condition].sort_values("sharing_ratio")
        assert np.allclose(group.sharing_ratio, [.2, .4, .6, .8, .9])
        assert (group.training_seed_count == expected_seeds[condition]).all()
    return runs, summary, checked


def export(runs, summary, checked, out=None):
    archive = Path(out) if out is not None else ROOT / "results/generated/fig5/source_data"
    out = archive
    out.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    (archive / "fig5.d").mkdir(parents=True, exist_ok=True)
    (out / "fig5.d").mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "fig5.d/fig5.d_runs.csv", index=False)
    summary.to_csv(out / "fig5.d/fig5.d_summary.csv", index=False)
    means = np.vstack([summary[summary.condition == c].sort_values("sharing_ratio").mean_accuracy_percent for c in ORDER])
    sd = np.vstack([summary[summary.condition == c].sort_values("sharing_ratio").sample_sd_percent for c in ORDER])
    counts = np.vstack([summary[summary.condition == c].sort_values("sharing_ratio").training_seed_count for c in ORDER])
    mat = {"panel_d_condition": np.array(ORDER, dtype=object),
           "panel_d_sharing_percent": np.array([20, 40, 60, 80, 90]),
           "panel_d_accuracy_percent": means,
           "panel_d_sample_sd_percent": sd, "panel_d_training_seed_count": counts}
    payload = json.loads((ROOT / "reproduction_data/main_10pct_sweep.json").read_text(encoding="utf-8-sig"))
    curves = payload["efficiency_conditions"]["eta_off"]["curves"]
    ratios = sorted(float(k) for k in curves)
    mat["panel_c_sharing_percent"] = np.array(ratios) * 100
    for mode, prefix in [("node_pruning", "node"), ("clements_layer", "stage")]:
        mat[f"panel_c_{prefix}_accuracy_percent"] = np.array([curves[f"{x:.2f}"][mode]["best_mean"]*100 for x in ratios])
        for tid in ("0", "1", "2"):
            cm = curves["0.90"][mode]["confusion_matrices"][tid]
            mat[f"panel_e_{prefix}_task{tid}_counts"] = np.asarray(cm["counts"])
            mat[f"panel_e_{prefix}_task{tid}_normalized_percent"] = np.asarray(cm["normalized"]) * 100
    savemat(archive / "fig5_source_data.mat", mat)
    savemat(archive / "fig5.d/fig5.d_source_data.mat",
            {key: value for key, value in mat.items() if key.startswith("panel_d_")})
    for name, target in [("Fig5.pdf", "fig5.pdf"), ("main_figure.png", "fig5.png")]:
        shutil.copy2(ROOT / "results" / name, archive / target)
    hashes = dict(checked)
    hashes["../main_10pct_sweep.json"] = hashlib.sha256((ROOT / "reproduction_data/main_10pct_sweep.json").read_bytes()).hexdigest()
    hashes["device_runs/codebook_static_3bit_eps_2pow_minus332_losst90pct.npz"] = hashlib.sha256((DATA / "device_runs/codebook_static_3bit_eps_2pow_minus332_losst90pct.npz").read_bytes()).hexdigest()
    (archive / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")


def draw(summary, output=None):
    plt.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    fig.subplots_adjust(left=.10, right=.98, top=.91, bottom=.30)
    for condition, color, marker, label in zip(ORDER, COLORS, MARKERS, LABELS):
        sub = summary[summary.condition == condition].sort_values("sharing_ratio")
        ls = "--" if condition == "3-bit PTQ" else ("-." if condition == ORDER[6] else (":" if condition == ORDER[7] else "-"))
        ax.plot(sub.sharing_ratio * 100, sub.mean_accuracy_percent, color=color,
                marker=marker, linestyle=ls, linewidth=1.7, markersize=5, label=label)
    ax.set(xlabel="MPU replacement ratio (%)", ylabel="Test accuracy (%)",
           title="64×64 Clements: operator quantization and device errors",
           xlim=(16, 94), ylim=(0, 100), xticks=[20, 40, 60, 80, 90])
    ax.grid(axis="y", color="#E5E8EF", linewidth=.7)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.20), ncol=2,
              frameon=False, fontsize=8.5)
    output = Path(output) if output is not None else ROOT / "results/generated/fig5"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "fig5d_quantization.pdf")
    fig.savefig(output / "fig5d_quantization.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--out-dir", help="Write generated plot/data under this directory without modifying archived inputs")
    args = parser.parse_args()
    runs, summary, checked = validated_runs()
    if not args.validate_only:
        out = Path(args.out_dir) if args.out_dir else None
        export(runs, summary, checked, out / "source_data" if out else None)
        draw(summary, out)
    print(f"Validated {len(runs)} run/ratio rows and {len(summary)} plotted means from {len(checked)} source JSONs.")


if __name__ == "__main__":
    main()
