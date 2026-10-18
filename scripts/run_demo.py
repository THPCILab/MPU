"""Offline CPU demo using public numerical sources and a small software test."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/demo")
    args = parser.parse_args()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / "reproduction_data").rglob("*.json")) + sorted((ROOT / "reproduction_data/fig5").rglob("*.csv")) + sorted((ROOT / "reproduction_data/fig5").rglob("*.npz"))
    sources = [p for p in sources if "replay" not in p.parts]
    before = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    started = time.perf_counter()
    steps = [
        ["tools/plot_fig5_quantization.py", "--out-dir", str(output / "quantization")],
        ["tools/check_quantization_model.py"],
    ]
    fixture = output / "synthetic_features.npz"
    if not fixture.exists():
        steps.append(["examples/train_custom_features.py", "--input", str(fixture),
                      "--make-demo", "--output", str(output / "training.json")])
    else:
        steps.append(["examples/train_custom_features.py", "--input", str(fixture),
                      "--output", str(output / "training.json")])
    for command in steps:
        subprocess.run([sys.executable] + command, cwd=ROOT, check=True)
    after = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if before != after:
        raise RuntimeError("Demo changed an archived numerical input")
    report = {"status": "PASS", "elapsed_seconds": round(time.perf_counter() - started, 3),
              "python": platform.python_version(), "platform": platform.platform(),
              "dependencies": {name: importlib.metadata.version(name) for name in
                               ("torch", "numpy", "scipy", "matplotlib", "pandas", "scikit-learn")},
              "archived_input_files_unchanged": len(before),
              "scope": "Archived-curve reconstruction plus numerical smoke/tiny synthetic training, not new research results"}
    (output / "demo_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS: offline CPU demo; archived inputs unchanged.")
    print("Outputs: " + str(output))


if __name__ == "__main__":
    main()
