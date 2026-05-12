import argparse
import json
from pathlib import Path


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def result_complete(result):
    return isinstance(result, dict) and "final_mean" in result and "mean_hist" in result


def merge_condition(dst, src):
    for key in ("pruning_score_profile", "pruning_score_source", "pruning_score_method"):
        if src.get(key) and key not in dst:
            dst[key] = src[key]
    if src.get("warmup") and result_complete(src["warmup"]):
        if dst.get("warmup") is None:
            dst["warmup"] = src["warmup"]
    if src.get("optical_unfrozen") and result_complete(src["optical_unfrozen"]):
        if dst.get("optical_unfrozen") is None:
            dst["optical_unfrozen"] = src["optical_unfrozen"]

    dst.setdefault("curves", {})
    for ratio, modes in src.get("curves", {}).items():
        dst["curves"].setdefault(ratio, {})
        for mode, result in modes.items():
            if result_complete(result):
                dst["curves"][ratio][mode] = result


def merge_payloads(paths):
    merged = None
    sources = []
    for path in paths:
        payload = load_json(path)
        sources.append(str(path))
        if merged is None:
            merged = {
                "settings": payload.get("settings", {}),
                "device": "merged_from_shards",
                "task_specs": payload.get("task_specs", []),
                "data_profile": payload.get("data_profile", []),
                "model_profile": payload.get("model_profile", {}),
                "baselines": {},
                "efficiency_conditions": {},
                "merged_sources": sources,
            }

        for key, value in payload.get("baselines", {}).items():
            if result_complete(value):
                merged["baselines"].setdefault(key, value)

        for cond_name, condition in payload.get("efficiency_conditions", {}).items():
            dst = merged["efficiency_conditions"].setdefault(
                cond_name,
                {
                    "use_efficiency": condition.get("use_efficiency"),
                    "warmup": None,
                    "optical_unfrozen": None,
                    "curves": {},
                },
            )
            merge_condition(dst, condition)
    if merged is None:
        raise ValueError("No input JSON files were provided")
    merged["merged_sources"] = sources
    ratio_values = set()
    for condition in merged.get("efficiency_conditions", {}).values():
        for ratio in condition.get("curves", {}):
            ratio_values.add(float(ratio) * 100.0)
    if ratio_values:
        merged.setdefault("settings", {})
        merged["settings"]["ratios"] = sorted(ratio_values)
    if merged.get("efficiency_conditions"):
        merged.setdefault("settings", {})
        names = sorted(merged["efficiency_conditions"])
        merged["settings"]["efficiency_conditions_merged"] = names
        merged["settings"]["efficiency"] = "both" if len(names) > 1 else names[0].replace("eta_", "")
        merged["settings"]["merged_from_shards"] = True
    return merged


def print_summary(payload):
    print("Baselines:")
    for key, value in payload.get("baselines", {}).items():
        print(f"  {key}: final={value.get('final_mean'):.4f} best={value.get('best_mean'):.4f}")
    print("Conditions:")
    for cond_name, condition in payload.get("efficiency_conditions", {}).items():
        print(f"  {cond_name}:")
        if condition.get("optical_unfrozen"):
            v = condition["optical_unfrozen"]
            print(f"    unfrozen: final={v.get('final_mean'):.4f} best={v.get('best_mean'):.4f}")
        for ratio, modes in sorted(condition.get("curves", {}).items(), key=lambda kv: float(kv[0])):
            print(f"    ratio {ratio}: {', '.join(sorted(modes))}")


def main():
    parser = argparse.ArgumentParser(description="Merge independent main-figure experiment shards.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = []
    for item in args.inputs:
        if any(ch in item for ch in "*?[]"):
            matches = sorted(Path().glob(item))
            paths.extend(matches)
        else:
            paths.append(Path(item))
    if not paths:
        raise FileNotFoundError(
            "No shard JSON files matched --inputs. Check results/main_figure_logs/*.log "
            "because the training shards probably failed before writing JSON."
        )
    existing = [p for p in paths if p.exists()]
    if len(existing) != len(paths):
        missing = [str(p) for p in paths if not p.exists()]
        raise FileNotFoundError(
            f"Missing shard files: {missing}. Check results/main_figure_logs/*.log "
            "for the corresponding shard traceback."
        )

    merged = merge_payloads(existing)
    save_json(Path(args.output), merged)
    print(f"Wrote {args.output}")
    print_summary(merged)


if __name__ == "__main__":
    main()
