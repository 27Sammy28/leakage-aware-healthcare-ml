#!/usr/bin/env python3
"""Copy generated figures, metrics, and reports into results/."""

from __future__ import annotations

from pathlib import Path
import shutil


def copy_if_exists(source: str | Path, destination_dir: str | Path) -> None:
    source = Path(source)
    destination_dir = Path(destination_dir)
    if source.exists() and source.is_file():
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_dir / source.name)


def main() -> None:
    result_dirs = [Path("results/figures"), Path("results/metrics"), Path("results/reports")]
    for directory in result_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    metric_files = [
        "benchmark_results.csv",
        "fold_metrics.csv",
        "bootstrap_ci_compact.csv",
        "bootstrap_ci_threshold_metrics.csv",
        "fixed_threshold_net_benefit.csv",
        "lr_hyperparameter_results.csv",
        "lr_sampling_results.csv",
        "clinical_audit_artifacts/decision_curve_net_benefit.csv",
        "clinical_audit_artifacts/decision_curve_net_benefit_compact.csv",
        "clinical_audit_artifacts/subgroup_calibration_performance.csv",
        "clinical_audit_artifacts/subgroup_lacve_compact.csv",
        "primary_audit_artifacts/calibration_and_ensemble_results.csv",
        "primary_audit_artifacts/calibration_and_ensemble_summary.csv",
        "primary_audit_artifacts/lacve_weights.csv",
        "stronger_baseline_artifacts/stronger_ensemble_results.csv",
        "stronger_baseline_artifacts/stronger_ensemble_summary.csv",
        "stronger_baseline_artifacts/lacve_plus_weights.csv",
        "uci_external_artifacts/uci_external_validation_results.csv",
        "uci_external_artifacts/uci_external_validation_summary.csv",
        "uci_external_artifacts/uci_leave_site_out_results.csv",
    ]
    for file_name in metric_files:
        copy_if_exists(file_name, "results/metrics")

    for figure in sorted(Path("figures").glob("*.png")):
        copy_if_exists(figure, "results/figures")
    for file_name in ["calibration_curve_etc.png", "shap_summary_plot.png"]:
        copy_if_exists(file_name, "results/figures")

    for file_name in ["main.pdf", "artifact_checklist.md", "NOTEBOOK_LOCATION.txt"]:
        copy_if_exists(file_name, "results/reports")

    Path("results/README.md").write_text(
        "# Results Directory\n\n"
        "Organized outputs for GitHub and manuscript review.\n\n"
        "- `figures/` contains generated PNG figures used for review and manuscript support.\n"
        "- `metrics/` contains CSV metric tables, calibration summaries, subgroup audits, and validation outputs.\n"
        "- `reports/` contains the compiled manuscript PDF and artifact/checklist notes.\n\n"
        "Run `python scripts/organize_results.py` after regenerating artifacts to refresh this directory.\n"
    )
    print("Organized outputs under results/.")


if __name__ == "__main__":
    main()
