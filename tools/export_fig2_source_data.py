"""Export final Figure 2 plotting tables."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reproduction_data/fig2/source_tables"
TABLES = [
    ("theta_phi_plot_points.csv", "fig2.b/fig2.b_theta_phi.csv", [
        ("device_index", "device_index"), ("theta_index", "theta_target_code"),
        ("phi_index", "phi_target_code"), ("theta_fit_code", "theta_fit_code"),
        ("phi_fit_code_display", "phi_fit_code"), ("joint_error_code", "joint_error_code"),
        ("effective_bits", "simulation_precision_bits")]),
    ("histogram_counts_from_supplied_image.csv", "fig2.b/fig2.b_precision_histogram.csv", [
        ("series_label_from_image", "series"), ("bin_left_bits", "bin_left_bits"),
        ("bin_right_bits", "bin_right_bits"), ("device_count", "device_count"),
        ("reported_mean_bits_from_image", "reported_mean_bits")]),
    ("bars_1550nm.csv", "fig2.c/fig2.c_power_1550nm.csv", [
        ("device_id", "device_id"), ("theta_pi", "theta_over_pi"), ("phi_pi", "phi_over_pi"),
        ("port", "port"), ("ideal_reference_db", "ideal_reference_db"),
        ("spectrum_1550nm_db", "power_1550nm_db"),
        ("illustrative_lower_db", "lower_db"), ("illustrative_upper_db", "upper_db")]),
    ("three_spectra.csv", "fig2.d/fig2.d_spectra.csv", [
        (c, c) for c in ["device_id", "wavelength_nm", "a11_db", "a11_smoothed_db",
                         "a12_db", "a12_smoothed_db", "a21_db", "a21_smoothed_db",
                         "a22_db", "a22_smoothed_db"]]),
]


def export(out):
    for name, relative, columns in TABLES:
        with (SOURCE / name).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow([new for old, new in columns])
            writer.writerows([row[old] for old, new in columns] for row in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "source_data/fig2")
    export(parser.parse_args().out_dir)

