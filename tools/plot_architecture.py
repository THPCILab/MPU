from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]

PRIVATE = "#7A3DB8"
SHARED = "#9AA4B2"
EDGE = "#5C667A"


def box(ax, xy, wh, text, fc="#F7F8FB", ec="#3C4A5F", lw=1.4, fontsize=10, weight="bold"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)


def arrow(ax, x0, y0, x1, y1, color="#344054"):
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=color, shrinkA=3, shrinkB=3),
    )


def draw_mesh(ax, x0, y0, width, height, modes=16, stages=16):
    wave = "#DDE3EE"
    stage_gap = width / (stages - 1)
    y_gap = height / (modes - 1)

    for m in range(modes):
        y = y0 + m * y_gap
        ax.plot([x0 - 0.05, x0 + width + 0.05], [y, y], color=wave, lw=0.75, zorder=0)

    for s in range(stages):
        x = x0 + s * stage_gap
        parity = s % 2
        for m in range(parity, modes - 1, 2):
            y_a = y0 + m * y_gap
            y_b = y0 + (m + 1) * y_gap
            is_shared = s < 6 or (s in (8, 10, 13) and m % 4 == 0)
            color = SHARED if is_shared else PRIVATE
            ax.plot([x - 0.025, x + 0.025], [y_a, y_b], color=EDGE, lw=0.65, alpha=0.55)
            ax.plot([x - 0.025, x + 0.025], [y_b, y_a], color=EDGE, lw=0.65, alpha=0.55)
            ax.scatter([x], [(y_a + y_b) / 2], s=22, color=color, edgecolors=EDGE, linewidths=0.35, zorder=3)

    ax.text(x0 + width / 2, y0 + height + 0.08, "single coherent 64x64 Clements mesh", ha="center", fontsize=10, weight="bold")
    ax.text(
        x0 + width / 2,
        y0 - 0.12,
        "64 optical modes, alternating even/odd 2x2 stages; one final detection",
        ha="center",
        fontsize=8.5,
        color="#475467",
    )
    ax.add_patch(Rectangle((x0 - 0.08, y0 - 0.05), width + 0.16, height + 0.10, fill=False, ec="#9AA4B2", lw=1.0))


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(14.0, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.035, 0.63), (0.12, 0.16), "EMNIST\n28x28", fc="#FFF8E8")
    box(ax, (0.20, 0.63), (0.13, 0.16), "fixed PCA\n64 features", fc="#ECFDF3")
    arrow(ax, 0.155, 0.71, 0.20, 0.71)

    draw_mesh(ax, 0.39, 0.32, 0.34, 0.47, modes=16, stages=16)
    arrow(ax, 0.33, 0.71, 0.39, 0.71)

    box(ax, (0.78, 0.66), (0.10, 0.11), "square-law\n|E|^2", fc="#EEF4FF")
    box(ax, (0.78, 0.49), (0.10, 0.11), "shared\nLN + GELU", fc="#F4F3FF")
    arrow(ax, 0.73, 0.71, 0.78, 0.71)
    arrow(ax, 0.83, 0.66, 0.83, 0.60)

    box(ax, (0.92, 0.75), (0.06, 0.07), "Digits\nhead", fc="#F0F9FF", fontsize=8.5)
    box(ax, (0.92, 0.62), (0.06, 0.07), "Upper\nhead", fc="#F0F9FF", fontsize=8.5)
    box(ax, (0.92, 0.49), (0.06, 0.07), "Lower\nhead", fc="#F0F9FF", fontsize=8.5)
    arrow(ax, 0.88, 0.545, 0.92, 0.785)
    arrow(ax, 0.88, 0.545, 0.92, 0.655)
    arrow(ax, 0.88, 0.545, 0.92, 0.525)

    ax.text(0.50, 0.94, "V27 Single-Mesh Multi-Task Photonic Network", ha="center", fontsize=17, weight="bold")
    ax.text(
        0.50,
        0.895,
        "Shared MPU pattern is learned during design: gray shared neurons use common theta/phi across tasks; purple private neurons keep task-specific phase deltas.",
        ha="center",
        fontsize=10,
        color="#475467",
    )

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SHARED, markeredgecolor=EDGE, markersize=8, label="shared MPU"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PRIVATE, markeredgecolor=EDGE, markersize=8, label="task-private MZI"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.50, 0.035), ncol=2, frameon=False)

    ax.text(
        0.39,
        0.14,
        "Implementation note: code uses one SharedClementsMZILayer with theta_even/phi_even in R^(32x32) and theta_odd/phi_odd in R^(32x31), total 2016 2x2 units.",
        ha="left",
        fontsize=9,
        color="#344054",
    )

    out = ROOT / "results" / "v27_single_mesh_architecture"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out.with_suffix('.png')}")
    print(f"Saved {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
