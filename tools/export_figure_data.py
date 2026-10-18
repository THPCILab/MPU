import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.io import savemat
from figure_inputs import pattern_ratios, prepare_payload


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "data/main_10pct_sweep.json"
DEFAULT_FIG_DIR = "results/generated/fig5"
DEFAULT_OUT_DIR = "results/generated/fig5/source_data"


def load_payload(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def condition(payload, name="eta_off"):
    conditions = payload["efficiency_conditions"]
    if name in conditions:
        return conditions[name]
    return next(iter(conditions.values()))


def task_names(payload):
    names = [""] * len(payload["task_specs"])
    for spec in payload["task_specs"]:
        names[int(spec["id"])] = spec["name"]
    return np.array(names, dtype=object)


def sorted_curve_rows(cond):
    rows = []
    for ratio_key in sorted(cond["curves"], key=lambda x: float(x)):
        modes = cond["curves"][ratio_key]
        node = modes.get("node_pruning")
        layer = modes.get("clements_layer")
        if node is None or layer is None:
            continue
        rows.append((float(ratio_key), node, layer))
    return rows


def copy_figures(fig_dir, out_dir, stem, target_stem=None):
    for ext in [".pdf", ".png"]:
        src = fig_dir / f"{stem}{ext}"
        if src.exists():
            shutil.copy2(src, out_dir / ((target_stem or stem) + ext))


def write_readme(path, title, body):
    text = f"# {title}\n\n{body.strip()}\n"
    path.write_text(text, encoding="utf-8")


def export_fig1(cond, fig_dir, root):
    folder = root / "fig5.c"
    folder.mkdir(parents=True, exist_ok=True)
    rows = sorted_curve_rows(cond)
    ratios = np.array([100.0 * r[0] for r in rows], dtype=float)
    node_best = np.array([100.0 * r[1]["best_mean"] for r in rows], dtype=float)
    layer_best = np.array([100.0 * r[2]["best_mean"] for r in rows], dtype=float)
    node_final = np.array([100.0 * r[1]["final_mean"] for r in rows], dtype=float)
    layer_final = np.array([100.0 * r[2]["final_mean"] for r in rows], dtype=float)
    optional = {}
    for ratio, name in [(0, "upper_bound_0pct_shared_percent"), (100, "lower_bound_100pct_shared_percent")]:
        indices = np.where(np.isclose(ratios, ratio))[0]
        if len(indices):
            optional[name] = np.array([[node_best[indices[0]]]])
    indices = np.where(np.isclose(ratios, 90))[0]
    if len(indices):
        i = indices[0]
        optional["gain_at_90pct_percentage_points"] = np.array([[node_best[i] - layer_best[i]]])
    savemat(
        folder / "fig5.c_source_data.mat",
        {
            "shared_mpu_ratio_percent": ratios,
            "node_level_best_accuracy_percent": node_best,
            "layer_level_best_accuracy_percent": layer_best,
            "node_level_final_accuracy_percent": node_final,
            "layer_level_final_accuracy_percent": layer_final,
            **optional,
        },
    )
    copy_figures(fig_dir, folder, "fig5c_accuracy", "fig5.c_accuracy")
    write_readme(
        folder / "README.md",
        "Fig. 5c: accuracy retention under MPU replacement",
        "Data are test accuracies in percent. Endpoint and 90% gain fields are included only when that ratio exists.",
    )


def export_fig2(payload, cond, fig_dir, root):
    if "0.90" not in cond["curves"]:
        print("Skipping 90% confusion export: ratio not present.")
        return
    folder = root / "fig5.e"
    folder.mkdir(parents=True, exist_ok=True)
    names = task_names(payload)
    data = {"task_names": names}
    for mode, prefix in [("node_pruning", "node_level"), ("clements_layer", "layer_level")]:
        result = cond["curves"]["0.90"][mode]
        for task_id in ["0", "1", "2"]:
            item = result["confusion_matrices"][task_id]
            data[f"{prefix}_task{task_id}_confusion_normalized_percent"] = np.array(item["normalized"], dtype=float) * 100.0
            data[f"{prefix}_task{task_id}_confusion_counts"] = np.array(item["counts"], dtype=float)
            data[f"{prefix}_task{task_id}_accuracy_percent"] = np.array([[100.0 * float(item["accuracy"])]])
    savemat(folder / "fig5.e_source_data.mat", data)
    copy_figures(fig_dir, folder, "fig5e_confusion", "fig5.e_confusion")
    write_readme(
        folder / "README.md",
        "Fig. 5e: 90% shared MPU confusion matrices",
        "Each normalized confusion matrix is row-normalized and stored in percent. "
        "Task IDs are 0=Digits, 1=Uppercase, 2=Lowercase.",
    )


def export_fig3(payload, cond, fig_dir, root):
    if "0.90" not in cond["curves"]:
        print("Skipping 90% convergence export: ratio not present.")
        return
    folder = root / "additional_analyses/convergence_90pct"
    folder.mkdir(parents=True, exist_ok=True)
    warmup = cond.get("warmup") or {}
    node = cond["curves"]["0.90"]["node_pruning"]
    warmup_len = len(warmup.get("mean_hist", []))
    warmup_epochs = np.arange(1, warmup_len + 1, dtype=float) if warmup_len > 1 else np.array([0.0], dtype=float)
    finetune_epochs = np.arange(1, len(node["mean_hist"]) + 1, dtype=float)
    if warmup_len > 1:
        finetune_epochs += warmup_len
    data = {
        "task_names": task_names(payload),
        "warmup_epoch": warmup_epochs,
        "finetune_epoch": finetune_epochs,
        "node_level_mean_test_accuracy_percent": np.array(node["mean_hist"], dtype=float) * 100.0,
        "node_level_training_loss": np.array(node["train_loss_hist"], dtype=float),
    }
    if warmup_len:
        data["warmup_mean_test_accuracy_percent"] = np.array(warmup["mean_hist"], dtype=float) * 100.0
        if cond.get("warmup_source"):
            data["warmup_source_json"] = np.array([cond["warmup_source"]], dtype=object)
        for task_id in ["0", "1", "2"]:
            data[f"warmup_task{task_id}_test_accuracy_percent"] = np.array(warmup["task_hist"][task_id], dtype=float) * 100.0
    if "train_loss_hist" in warmup:
        data["warmup_training_loss"] = np.array(warmup["train_loss_hist"], dtype=float)
    for task_id in ["0", "1", "2"]:
        data[f"node_level_task{task_id}_test_accuracy_percent"] = np.array(node["task_hist"][task_id], dtype=float) * 100.0
    if "train_mean_hist" in node:
        data["node_level_mean_train_accuracy_percent"] = np.array(node["train_mean_hist"], dtype=float) * 100.0
    savemat(folder / "figure_data.mat", data)
    copy_figures(fig_dir, folder, "sharing_convergence", "convergence_90pct")
    write_readme(
        folder / "README.md",
        "Additional analysis: warmup and 90% neuron-level sharing convergence",
        "The plotted left-axis data are per-task test accuracies in percent; "
        "the right-axis data are training loss. The full warmup segment is loaded from the "
        "standalone warmup JSON because the main sweep reused the saved warmup checkpoint.",
    )


def export_fig4(cond, fig_dir, root):
    folder = root / "additional_analyses/sharing_patterns"
    folder.mkdir(parents=True, exist_ok=True)
    ratios = pattern_ratios(cond)
    data = {"shared_mpu_ratio_percent": np.array(ratios, dtype=float)}
    for ratio in ratios:
        key = f"{ratio / 100:.2f}"
        for mode, prefix in [("node_pruning", "node_level"), ("clements_layer", "layer_level")]:
            mask = cond["curves"][key][mode]["mask_matrices"][0]
            data[f"{prefix}_{ratio}pct_even_stage_pair_mask"] = np.array(mask["even"], dtype=np.uint8)
            data[f"{prefix}_{ratio}pct_odd_stage_pair_mask"] = np.array(mask["odd"], dtype=np.uint8)
    savemat(folder / "figure_data.mat", data)
    copy_figures(fig_dir, folder, "sharing_patterns", "sharing_patterns")
    write_readme(
        folder / "README.md",
        "Additional analysis: archived MPU sharing patterns",
        "Mask value 1 means shared MPU and 0 means task-private MZI. "
        "Even and odd arrays are stored as physical stage-by-pair masks for the single Clements mesh.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--fig-dir", default=DEFAULT_FIG_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--warmup-input", help="Optional matching full warmup JSON")
    args = parser.parse_args()

    input_path = ROOT / args.input
    payload = prepare_payload(load_payload(input_path), input_path, args.warmup_input)
    cond = condition(payload)
    fig_dir = ROOT / args.fig_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    export_fig1(cond, fig_dir, out_dir)
    export_fig2(payload, cond, fig_dir, out_dir)
    export_fig3(payload, cond, fig_dir, out_dir)
    export_fig4(cond, fig_dir, out_dir)

    write_readme(
        out_dir / "README.md",
        "Manuscript Figure 5 source data",
        "Subdirectories use manuscript panel identifiers (fig5.a--fig5.e). "
        "The additional_analyses folder contains convergence and mask analyses "
        "that are not additional main-manuscript figures. "
        "Run tools/export_manuscript_source_data.py for the panel CSV files.",
    )
    print(f"Wrote data package to {out_dir}")


if __name__ == "__main__":
    main()
