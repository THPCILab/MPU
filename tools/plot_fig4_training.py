"""Validate and redraw archived Figure 4(b) training curves."""
import csv
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_data/fig4/fig4.b"
OUTPUT = ROOT / "results/fig4_training"
COLORS = ["#8c564b", "#9467bd", "#1f77b4", "#d62728"]


def load_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def decorate(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-120, 3140)
    ax.set_xticks(np.arange(0, 3001, 500))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training accuracy (%)")


def single_panel(ax, data):
    fields = ["mzi", "mzi_3bit", "mpu", "mpu_mzi"]
    for prefix, label, color in zip(fields, ["MZI", "MZI (3-bit)", "MPU", "MPU+MZI"], COLORS):
        ax.plot(data["epoch"], data[prefix+"_accuracy_percent"], color=color, label=label)
    decorate(ax)
    ax.set_ylim(14, 91)
    ax.set_title("Single-task training", pad=10)
    ax.legend(loc="lower right")
    peak = data["mpu_mzi_accuracy_percent"].argmax()
    delta = data["mpu_mzi_accuracy_percent"].max() - data["mpu_accuracy_percent"].max()
    ax.annotate(f"Peak accuracy +{delta:.1f} pp", xy=(data["epoch"][peak], data["mpu_mzi_accuracy_percent"][peak]),
                xytext=(2150, 69), color=COLORS[-1], fontsize=10,
                arrowprops=dict(arrowstyle="->", color=COLORS[-1], lw=1.2))


def dual_panel(ax, data, full_loss=False):
    ax.plot(data["epoch"], data["task1_accuracy_percent"], color="#6627b8", label="Task 1 Accuracy")
    ax.plot(data["epoch"], data["task2_accuracy_percent"], color="#0875ba", label="Task 2 Accuracy")
    decorate(ax)
    ax.set_ylim(0, 92)
    ax.set_title("Dual-task training", pad=10)
    right = ax.twinx()
    right.spines["top"].set_visible(False)
    right.plot(data["epoch"], data["task1_loss"], color="#1f77b4", linestyle="--", label="Task 1 Loss")
    right.plot(data["epoch"], data["task2_loss"], color="#ff7f0e", linestyle="--", label="Task 2 Loss")
    right.set_ylabel("Loss")
    if full_loss:
        right.set_ylim(0, max(data["task1_loss"].max(), data["task2_loss"].max())*1.05)
    else:
        right.set_ylim(0, 5)
    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = right.get_legend_handles_labels()
    ax.legend(handles+h2, labels+l2, loc="center right")
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    return right


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"{name}.{ext}", dpi=600, facecolor="white")
    plt.close(fig)


def validate():
    s, d = load_csv("fig4.b_single_task.csv"), load_csv("fig4.b_dual_task.csv")
    raw = json.loads((ROOT / "data/fig4/fig4.b_training_history.json").read_text(encoding="utf-8"))
    assert all(len(values) == 300 and np.isfinite(values).all() for values in [*s.values(), *d.values()])
    np.testing.assert_array_equal(s["epoch"], np.arange(10, 3001, 10))
    np.testing.assert_array_equal(d["epoch"], s["epoch"])
    for prefix, key in zip(["mzi", "mzi_3bit", "mpu", "mpu_mzi"],
                           ["mzi_2layer_history", "mzi_2layer_q3_history", "dpu_history", "dpu_mzi_single_history"]):
        np.testing.assert_array_equal(s[prefix+"_accuracy_percent"], np.asarray(raw[key]["acc"])*100)
        np.testing.assert_array_equal(s[prefix+"_loss"], raw[key]["loss"])
    for i in (1, 2):
        np.testing.assert_array_equal(d[f"task{i}_accuracy_percent"], np.asarray(raw["dpu_mzi_multi_history"][f"acc{i}"])*100)
        np.testing.assert_array_equal(d[f"task{i}_loss"], raw["dpu_mzi_multi_history"][f"loss{i}"])
    for table in (s, d):
        for column, values in table.items():
            if column.endswith("accuracy_percent"):
                assert ((values >= 0) & (values <= 100)).all()
            elif column.endswith("loss"):
                assert (values >= 0).all()
    print("PASS: Fig4.b: six training histories, 300 records each; CSVs match original JSON.")
    return s, d


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    s, d = validate()
    if args.validate_only:
        return
    OUTPUT = args.out_dir
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 11, "axes.titlesize": 11.5, "axes.titleweight": "normal",
                         "axes.labelsize": 11.5, "axes.linewidth": 1.0, "lines.linewidth": 1.65,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
                         "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 9.0))
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.07, top=0.96, hspace=0.34)
    single_panel(axs[0], s)
    dual_panel(axs[1], d)
    save(fig, "training_curves_combined")
    fig, ax = plt.subplots(figsize=(7.0, 4.5), layout="constrained")
    single_panel(ax, s)
    save(fig, "single_task_accuracy")
    for full in (False, True):
        fig, ax = plt.subplots(figsize=(7.4, 4.5), layout="constrained")
        dual_panel(ax, d, full_loss=full)
        save(fig, "dual_task_full_loss" if full else "dual_task_accuracy_loss")
    print("Verified all accuracy and loss fields; exported four PNG/PDF/SVG previews.")


if __name__ == "__main__":
    main()
