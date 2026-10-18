"""Export panel-labelled numerical sources without rerunning experiments."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, headers, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper_data")
    args = parser.parse_args()
    out = args.out_dir
    fig4 = json.loads((ROOT / "data/fig4/fig4.c_confusion_counts.json").read_text(encoding="utf-8"))
    counts, summary = [], []
    for item in fig4["matrices"]:
        matrix = item["counts"]
        n = sum(map(sum, matrix))
        correct = sum(matrix[i][i] for i in range(4))
        assert n == item["samples"] and correct == item["correct"]
        assert all(sum(row) == n // 4 for row in matrix)
        assert round(100 * correct / n, 2) == item["displayed_accuracy_percent"]
        for i, true_label in enumerate(item["labels"]):
            for j, predicted_label in enumerate(item["labels"]):
                counts.append([item["task"], item["split"], true_label, predicted_label, matrix[i][j]])
        summary.append([item["task"], item["split"], n, correct, 100 * correct / n,
                        item["displayed_accuracy_percent"]])
    write_csv(out / "fig4/fig4.c/fig4.c_confusion_counts.csv",
              ["task", "split", "true_label", "predicted_label", "count"], counts)
    write_csv(out / "fig4/fig4.c/fig4.c_confusion_summary.csv",
              ["task", "split", "samples", "correct", "accuracy_percent", "displayed_accuracy_percent"], summary)
    payload = json.loads((ROOT / "data/main_10pct_sweep.json").read_text(encoding="utf-8-sig"))
    curves = payload["efficiency_conditions"]["eta_off"]["curves"]
    sharing = []
    for key in sorted(curves, key=float):
        for mode in ("node_pruning", "clements_layer"):
            row = curves[key][mode]
            sharing.append([float(key) * 100, mode, row["best_mean"] * 100, row["final_mean"] * 100])
    write_csv(out / "fig5/fig5.c/fig5.c_accuracy.csv",
              ["shared_ratio_percent", "mode", "best_epoch_accuracy_percent", "final_epoch_accuracy_percent"], sharing)
    confusion = []
    tasks = {"0": "Digits", "1": "Uppercase", "2": "Lowercase"}
    for mode in ("node_pruning", "clements_layer"):
        for task_id, name in tasks.items():
            item = curves["0.90"][mode]["confusion_matrices"][task_id]
            labels = item["labels"]
            for i, true_label in enumerate(labels):
                for j, predicted_label in enumerate(labels):
                    confusion.append([90, mode, name, true_label, predicted_label,
                                      item["counts"][i][j], item["normalized"][i][j]])
    write_csv(out / "fig5/fig5.e/fig5.e_confusion.csv",
              ["shared_ratio_percent", "mode", "task", "true_label", "predicted_label", "count",
               "row_normalized_fraction"], confusion)
    print("Exported Fig4.c: %d counts; Fig5.c: %d points; Fig5.e: %d counts." %
          (len(counts), len(sharing), len(confusion)))


if __name__ == "__main__":
    main()
