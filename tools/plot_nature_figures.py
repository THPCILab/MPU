import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, ListedColormap
from figure_inputs import pattern_ratios, prepare_payload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "reproduction_data/main_10pct_sweep.json"

NODE = "#6F3FB5" #
LAYER = "#2F5597"
DARK = "#30343B"
MID = "#8E97A8"
LIGHT = "#D7DCE5" #
GRID = "#E5E8EF"


def configure():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 12,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.25,
        }
    )


def load_payload(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def condition(payload, name="eta_off"):
    conditions = payload["efficiency_conditions"]
    if name in conditions:
        return conditions[name]
    return next(iter(conditions.values()))


def out_path(out_dir, stem, suffix):
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{suffix}"


def curve_rows(cond):
    rows = []
    for ratio_key in sorted(cond["curves"], key=lambda x: float(x)):
        modes = cond["curves"][ratio_key]
        node = modes.get("node_pruning")
        layer = modes.get("clements_layer")
        if not node or not layer:
            continue
        rows.append(
            {
                "ratio": 100.0 * float(ratio_key),
                "node": 100.0 * float(node["best_mean"]),
                "layer": 100.0 * float(layer["best_mean"]),
                "node_final": 100.0 * float(node["final_mean"]),
                "layer_final": 100.0 * float(layer["final_mean"]),
            }
        )
    return rows


def save(fig, out_stem):
    fig.savefig(out_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Wrote {out_stem.with_suffix('.png')}")
    print(f"Wrote {out_stem.with_suffix('.pdf')}")


def plot_main_curve(payload, out_dir):
    cond = condition(payload)
    rows = curve_rows(cond)
    xs = np.array([r["ratio"] for r in rows])
    node = np.array([r["node"] for r in rows])
    layer = np.array([r["layer"] for r in rows])
    if not rows:
        raise ValueError("Input has no ratios containing both comparison modes")
    upper = node[xs == 0][0] if np.any(xs == 0) else node.max()
    lower = node[xs == 100][0] if np.any(xs == 100) else min(node.min(), layer.min())

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    if np.any(xs == 0):
        ax.axhline(upper, color=DARK, linewidth=2.1, linestyle=(0, (2, 2)), zorder=1)
    if np.any(xs == 100):
        ax.axhline(lower, color=MID, linewidth=2.1, linestyle=(0, (5, 3)), zorder=1)
    ax.plot(xs, node, color=NODE, marker="^", markersize=8.5, linewidth=2.8, zorder=3)
    ax.plot(xs, layer, color=LAYER, marker="o", markersize=7.8, linewidth=2.8, zorder=3)

    ax.set_xlabel("Shared MPU ratio (%)", weight="bold")
    ax.set_ylabel("Average test accuracy (%)", weight="bold")
    ax.set_title("Accuracy Retention Under MPU Replacement", weight="bold", pad=10)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.set_xlim(-2, 102)
    ymin = max(0, min(lower, node.min(), layer.min()) - 4)
    ymax = min(100, max(upper, node.max(), layer.max()) + 1.8)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, axis="y", linestyle=":", color=GRID, linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        Line2D([0], [0], color=NODE, marker="^", linewidth=2.8, markersize=8.5, label="Neuron-level sharing"),
        Line2D([0], [0], color=LAYER, marker="o", linewidth=2.8, markersize=7.8, label="Layer-level sharing"),
        Line2D([0], [0], color=DARK, linestyle=(0, (2, 2)), linewidth=2.1, label="0% shared upper bound"),
        Line2D([0], [0], color=MID, linestyle=(0, (5, 3)), linewidth=2.1, label="100% shared lower bound"),
    ]
    handles = handles[:2] + ([handles[2]] if np.any(xs == 0) else []) + ([handles[3]] if np.any(xs == 100) else [])
    ax.legend(handles=handles, frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.04))
    x90 = 90.0
    diff_color = "#4A4F5A"
    if np.any(xs == 90):
        node90 = node[xs == 90][0]
        layer90 = layer[xs == 90][0]
        ax.vlines(x90, layer90, node90, color=diff_color, linewidth=1.9, linestyle=(0, (3, 2)), zorder=2)
        ax.scatter([x90, x90], [layer90, node90], color=[LAYER, NODE], s=68, zorder=4)
    # ax.annotate(
    #     "+7.26 pp \n at 90%",
    #     xy=(x90*0.7, (node90 + layer90) / 2),
    #     xytext=(99.2, (node90 + layer90) / 2 + 4.0),
    #     ha="right",
    #     va="center",
    #     color=diff_color,
    #     weight="bold",
    #     fontsize=13,
    #     #arrowprops=dict(arrowstyle="->", color=diff_color, lw=1.6, linestyle=(0, (3, 2)), shrinkA=4, shrinkB=3),
    # )
    save(fig, out_path(out_dir, "fig5c", "accuracy"))
    plt.close(fig)


def task_names(payload):
    return {str(s["id"]): s["name"] for s in payload["task_specs"]}


def class_labels(task_name, n):
    if task_name == "Digits":
        return [str(i) for i in range(n)]
    letters = [chr(ord("A") + i) for i in range(n)]
    if task_name == "Lowercase":
        return [c.lower() for c in letters]
    return letters


def plot_confusion(payload, out_dir):
    cond = condition(payload)
    if "0.90" not in cond["curves"]:
        print("Skipping 90% confusion matrices: ratio not present in this input.")
        return
    modes = [("node_pruning", "Neuron-level sharing", NODE), ("clements_layer", "Layer-level sharing", LAYER)]
    names = task_names(payload)
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.2), constrained_layout=True)
    im = None
    for row, (mode, label, color) in enumerate(modes):
        result = cond["curves"]["0.90"][mode]
        cms = result["confusion_matrices"]
        for col, task_id in enumerate(["0", "1", "2"]):
            ax = axes[row, col]
            item = cms[task_id]
            mat = np.array(item["normalized"], dtype=float) * 100.0
            im = ax.imshow(mat, cmap="PuBu", vmin=0, vmax=100, interpolation="nearest")
            task = names[task_id]
            acc = 100.0 * float(item["accuracy"])
            ax.set_title(f"{label}\n{task}, acc. {acc:.1f}%", color=color, weight="bold", fontsize=13.5)
            labels = class_labels(task, mat.shape[0])
            if mat.shape[0] <= 10:
                ticks = np.arange(mat.shape[0])
            else:
                ticks = np.arange(0, mat.shape[0], 5)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([labels[i] for i in ticks], rotation=0)
            ax.set_yticklabels([labels[i] for i in ticks])
            if row == 1:
                ax.set_xlabel("Predicted label", weight="bold")
            if col == 0:
                ax.set_ylabel("True label", weight="bold")
            for spine in ax.spines.values():
                spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.86, pad=0.015)
    cbar.set_label("Row-normalized samples (%)", weight="bold")
    fig.suptitle("90% Shared MPU Confusion Matrices", fontsize=18, weight="bold")
    save(fig, out_path(out_dir, "fig5e", "confusion"))
    plt.close(fig)


def plot_convergence(payload, out_dir):
    cond = condition(payload)
    if "0.90" not in cond["curves"]:
        print("Skipping 90% convergence: ratio not present in this input.")
        return
    warmup = cond.get("warmup") or {}
    node = cond["curves"]["0.90"]["node_pruning"]

    warmup_len = len(warmup.get("mean_hist", []))
    has_warmup_curve = warmup_len > 1
    warmup_x = np.arange(1, warmup_len + 1) if has_warmup_curve else np.array([0.0])
    finetune_x = np.arange(1, len(node["mean_hist"]) + 1)
    if has_warmup_curve:
        finetune_x = finetune_x + warmup_len
        transition_x = warmup_len + 0.5
    else:
        transition_x = 0.5

    fig, ax_acc = plt.subplots(figsize=(7.6, 5.1))
    ax_loss = ax_acc.twinx()

    task_colors = {"0": NODE, "1": LAYER, "2": "#6C778A"}
    task_labels = {"0": "Digits", "1": "Uppercase", "2": "Lowercase"}
    handles = []

    for tid in ["0", "1", "2"]:
        if warmup_len:
            warm_values = np.array(warmup["task_hist"][tid], dtype=float) * 100.0
            if has_warmup_curve:
                ax_acc.plot(
                    warmup_x,
                    warm_values,
                    color=task_colors[tid],
                    linewidth=2.2,
                    alpha=0.55,
                    linestyle='-',
                    #linestyle=(0, (2, 2)),
                )
            else:
                ax_acc.scatter(
                    warmup_x,
                    warm_values,
                    color=task_colors[tid],
                    marker="D",
                    s=34,
                    alpha=0.80,
                    zorder=4,
                )
        line = ax_acc.plot(
            finetune_x,
            np.array(node["task_hist"][tid]) * 100.0,
            color=task_colors[tid],
            linewidth=2.7,
            label=task_labels[tid],
        )[0]
        handles.append(line)

    if has_warmup_curve:
        ax_acc.axvspan(
            warmup_x.min(), warmup_x.max(),  # 覆盖整个 warmup 横坐标范围
            facecolor='#EAEAEC',             # 浅灰色背景
            alpha=0.3,                       # 透明度
            zorder=0                         # 放在曲线最下层
        )
    if has_warmup_curve and "train_loss_hist" in warmup:
        ax_loss.plot(
            warmup_x,
            warmup["train_loss_hist"],
            color=DARK,
            linewidth=2.0,
            linestyle=(0, (2, 2)),
            alpha=0.55,
        )
    loss_line = ax_loss.plot(
        finetune_x,
        node["train_loss_hist"],
        color=DARK,
        linewidth=2.4,
        linestyle=(0, (5, 3)),
        label="Training loss",
    )[0]
    handles.append(loss_line)

    ax_acc.axvline(transition_x, color="#4A4F5A", linewidth=1.5, linestyle=(0, (3, 2)), alpha=0.85)
    y0, y1 = ax_acc.get_ylim()
    # ax_acc.text(
    #     transition_x,
    #     y1 - 0.05 * (y1 - y0),
    #     "90% pattern imposed",
    #     ha="center",
    #     va="top",
    #     rotation=0,
    #     fontsize=14.0,
    #     color="#4A4F5A",
    #     weight="bold",
    # )
    if has_warmup_curve:
        ax_acc.text(warmup_x.mean(), y0 + 0.2 * (y1 - y0), "Warmup", ha="center", fontsize=16, color="#4A4F5A", weight="bold")
        ax_acc.text(warmup_x.mean()+100, y0 + 0.2 * (y1 - y0), "90% Neuron \n Sharing", ha="center", fontsize=16, color="#4A4F5A", weight="bold")
    else:
        ax_acc.text(0, y0 + 0.08 * (y1 - y0), "warmup\nstate", ha="center", fontsize=14.0, color="#4A4F5A", weight="bold")

    ax_acc.set_xlabel("Training epoch", weight="bold")
    ax_acc.set_ylabel("Test accuracy (%)", weight="bold")
    ax_loss.set_ylabel("Training loss", weight="bold", color=DARK)
    ax_acc.set_title("Warmup and 90% Neuron-Level Sharing Fine-Tuning", weight="bold")
    ax_acc.grid(True, axis="y", linestyle=":", color=GRID, linewidth=1.0)
    
    ax_acc.spines["top"].set_visible(False)
    ax_loss.spines["top"].set_visible(False)
    ax_acc.legend(handles=handles, frameon=False, loc="center right")
    ax_loss.tick_params(axis="y", colors=DARK)
    save(fig, out_path(out_dir, "sharing", "convergence"))
    plt.close(fig)


def draw_mesh(ax, mask_layer, shared_color="#9AA4B2", private_color=NODE):
    even = np.array(mask_layer["even"], dtype=float)
    odd = np.array(mask_layer["odd"], dtype=float)
    stages, num_even = even.shape
    dim = num_even * 2
    num_odd = odd.shape[1]
    mat = np.zeros((dim - 1, stages * 2), dtype=float)
    for stage in range(stages):
        x_even = 2 * stage
        for pair in range(num_even):
            mat[2 * pair, x_even] = 2.0 if even[stage, pair] > 0.5 else 1.0
        x_odd = 2 * stage + 1
        for pair in range(num_odd):
            mat[2 * pair + 1, x_odd] = 2.0 if odd[stage, pair] > 0.5 else 1.0
    cmap = ListedColormap(["#FFFFFF", private_color, shared_color])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest", rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#C7CCD6")
        spine.set_linewidth(0.65)


def plot_patterns(payload, out_dir, ratios=None, stem="sharing", main_title="Shared MPU Pattern Across Replacement Ratios"):
    cond = condition(payload)
    ratios = pattern_ratios(cond, ratios)
    if not ratios:
        print("Skipping sharing-pattern panel: no requested masks in this input.")
        return
    fig, axes = plt.subplots(len(ratios), 2, figsize=(11.5, max(4.5, 2.92 * len(ratios))), squeeze=False, constrained_layout=False)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.045, top=0.89, hspace=0.26, wspace=0.08)
    for row, ratio in enumerate(ratios):
        key = f"{ratio / 100:.2f}"
        for col, (mode, col_title) in enumerate([("node_pruning", "Neuron-level sharing"), ("clements_layer", "Layer-level sharing")]):
            ax = axes[row, col]
            draw_mesh(ax, cond["curves"][key][mode]["mask_matrices"][0])
            if row == 0:
                ax.set_title(col_title, weight="bold", fontsize=20, pad=9)
            if col == 0:
                ax.set_ylabel(f"{ratio}%\nMPU", rotation=0, labelpad=46, va="center", ha="right", weight="bold", fontsize=16)
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#9AA4B2", markeredgecolor="#6C778A", markersize=10, label="Shared MPU"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=NODE, markeredgecolor="#5F329B", markersize=10, label="Task-specific MZI"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.955), ncol=2, frameon=False, fontsize=16)
    fig.suptitle(main_title, fontsize=22, weight="bold", y=0.97,x=0.55)
    save(fig, out_path(out_dir, stem, "patterns"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", default="results/generated/fig5")
    parser.add_argument("--warmup-input", help="Optional matching full warmup JSON")
    args = parser.parse_args()
    configure()
    input_path = ROOT / args.input
    payload = prepare_payload(load_payload(input_path), input_path, args.warmup_input)
    out_dir = ROOT / args.out_dir
    plot_main_curve(payload, out_dir)
    plot_confusion(payload, out_dir)
    plot_convergence(payload, out_dir)
    plot_patterns(payload, out_dir)
    plot_patterns(payload, out_dir, ratios=[20, 40, 60, 80], stem="sharing_even", main_title="Shared MPU Pattern Across Even Replacement Ratios")


if __name__ == "__main__":
    main()
