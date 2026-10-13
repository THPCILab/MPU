import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "data/main_10pct_sweep.json"
DEFAULT_FIG_DIR = "results/analysis_panels"
DEFAULT_OUT_DIR = "paper_data"


def load_payload(path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def condition(payload, name="eta_off"):
    conditions = payload["efficiency_conditions"]
    if name in conditions:
        return conditions[name]
    return next(iter(conditions.values()))


def load_external_warmup_result():
    candidates = [
        ROOT / "data" / "warmup_main.json",
        ROOT / "data" / "warmup_replicate.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        payload = load_payload(path)
        warmup = condition(payload).get("warmup", {})
        if len(warmup.get("mean_hist", [])) > 1:
            return path, warmup
    return None, None


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


def copy_figures(fig_dir, out_dir, stem):
    for ext in [".pdf", ".png"]:
        src = fig_dir / f"{stem}{ext}"
        if src.exists():
            shutil.copy2(src, out_dir / src.name)


def write_readme(path, title, body):
    text = f"# {title}\n\n{body.strip()}\n"
    path.write_text(text, encoding="utf-8")


def export_fig1(cond, fig_dir, root):
    folder = root / "Fig1_Accuracy_Retention_Under_MPU_Replacement"
    folder.mkdir(parents=True, exist_ok=True)
    rows = sorted_curve_rows(cond)
    ratios = np.array([100.0 * r[0] for r in rows], dtype=float)
    node_best = np.array([100.0 * r[1]["best_mean"] for r in rows], dtype=float)
    layer_best = np.array([100.0 * r[2]["best_mean"] for r in rows], dtype=float)
    node_final = np.array([100.0 * r[1]["final_mean"] for r in rows], dtype=float)
    layer_final = np.array([100.0 * r[2]["final_mean"] for r in rows], dtype=float)
    idx90 = np.where(np.isclose(ratios, 90.0))[0][0]
    idx0 = np.where(np.isclose(ratios, 0.0))[0][0]
    idx100 = np.where(np.isclose(ratios, 100.0))[0][0]
    savemat(
        folder / "figure_data.mat",
        {
            "shared_mpu_ratio_percent": ratios,
            "node_level_best_accuracy_percent": node_best,
            "layer_level_best_accuracy_percent": layer_best,
            "node_level_final_accuracy_percent": node_final,
            "layer_level_final_accuracy_percent": layer_final,
            "upper_bound_0pct_shared_percent": np.array([[node_best[idx0]]]),
            "lower_bound_100pct_shared_percent": np.array([[node_best[idx100]]]),
            "gain_at_90pct_percentage_points": np.array([[node_best[idx90] - layer_best[idx90]]]),
        },
    )
    copy_figures(fig_dir, folder, "fig1_main_accuracy_curve")
    write_readme(
        folder / "README.md",
        "Fig. 1 Accuracy retention under MPU replacement",
        "Data are test accuracies in percent. The 90% node-layer gain is stored as "
        "`gain_at_90pct_percentage_points`.",
    )


def export_fig2(payload, cond, fig_dir, root):
    folder = root / "Fig2_90pct_Shared_MPU_Confusion_Matrices"
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
    savemat(folder / "figure_data.mat", data)
    copy_figures(fig_dir, folder, "fig2_confusion_90pct")
    write_readme(
        folder / "README.md",
        "Fig. 2 90% shared MPU confusion matrices",
        "Each normalized confusion matrix is row-normalized and stored in percent. "
        "Task IDs are 0=Digits, 1=Uppercase, 2=Lowercase.",
    )


def export_fig3(payload, cond, fig_dir, root):
    folder = root / "Fig3_90pct_Neuron_Level_Convergence"
    folder.mkdir(parents=True, exist_ok=True)
    warmup = cond.get("warmup") or {}
    external_warmup_path, external_warmup = load_external_warmup_result()
    if external_warmup is not None:
        warmup = external_warmup
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
        if external_warmup_path is not None:
            data["warmup_source_json"] = np.array([str(external_warmup_path.relative_to(ROOT))], dtype=object)
        for task_id in ["0", "1", "2"]:
            data[f"warmup_task{task_id}_test_accuracy_percent"] = np.array(warmup["task_hist"][task_id], dtype=float) * 100.0
    if "train_loss_hist" in warmup:
        data["warmup_training_loss"] = np.array(warmup["train_loss_hist"], dtype=float)
    for task_id in ["0", "1", "2"]:
        data[f"node_level_task{task_id}_test_accuracy_percent"] = np.array(node["task_hist"][task_id], dtype=float) * 100.0
    if "train_mean_hist" in node:
        data["node_level_mean_train_accuracy_percent"] = np.array(node["train_mean_hist"], dtype=float) * 100.0
    savemat(folder / "figure_data.mat", data)
    copy_figures(fig_dir, folder, "fig3_convergence_90pct")
    write_readme(
        folder / "README.md",
        "Fig. 3 warmup and 90% neuron-level sharing convergence",
        "The plotted left-axis data are per-task test accuracies in percent; "
        "the right-axis data are training loss. The full warmup segment is loaded from the "
        "standalone warmup JSON because the main sweep reused the saved warmup checkpoint.",
    )


def export_fig4(cond, fig_dir, root):
    folder = root / "Fig4_MPU_Sharing_Patterns"
    folder.mkdir(parents=True, exist_ok=True)
    ratios = [10, 30, 50, 70, 90]
    data = {"shared_mpu_ratio_percent": np.array(ratios, dtype=float)}
    for ratio in ratios:
        key = f"{ratio / 100:.2f}"
        for mode, prefix in [("node_pruning", "node_level"), ("clements_layer", "layer_level")]:
            mask = cond["curves"][key][mode]["mask_matrices"][0]
            data[f"{prefix}_{ratio}pct_even_stage_pair_mask"] = np.array(mask["even"], dtype=np.uint8)
            data[f"{prefix}_{ratio}pct_odd_stage_pair_mask"] = np.array(mask["odd"], dtype=np.uint8)
    savemat(folder / "figure_data.mat", data)
    copy_figures(fig_dir, folder, "fig4_sharing_patterns")
    write_readme(
        folder / "README.md",
        "Fig. 4 MPU sharing patterns",
        "Mask value 1 means shared MPU and 0 means task-private MZI. "
        "Even and odd arrays are stored as physical stage-by-pair masks for the single Clements mesh.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--fig-dir", default=DEFAULT_FIG_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    payload = load_payload(ROOT / args.input)
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
        "Figure source data",
        "The Fig1--Fig4 folders contain individual analysis plots, not manuscript figure "
        "numbers. Each contains a PDF/PNG and `figure_data.mat`. Manuscript Fig. 5 "
        "source data are in `Fig5_Sharing_and_Quantization/`.",
    )
    print(f"Wrote data package to {out_dir}")


if __name__ == "__main__":
    main()
