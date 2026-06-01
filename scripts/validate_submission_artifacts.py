#!/usr/bin/env python3
"""Validate that submission-ready experiment artifacts are present.

This script does not rerun the ML pipeline. It checks whether the final rerun
exported the files needed to make manuscript tables and figures auditable.
"""

import csv
import sys
from pathlib import Path

REQUIRED_FILES = {
    "fold_metrics.csv": {
        "model", "sampling", "repeat", "fold", "accuracy", "precision",
        "recall", "f1", "roc_auc", "pr_auc", "brier"
    },
    "fold_predictions.csv": {
        "repeat", "fold", "fold_id", "model", "sampling", "sample_id",
        "y_true", "y_pred", "y_pred_proba"
    },
    "heldout_predictions.csv": {"model", "sampling", "sample_id", "y_true", "y_pred", "y_prob"},
    "confusion_matrices.csv": {"model", "tn", "fp", "fn", "tp"},
    "roc_points.csv": {"model", "fpr", "tpr"},
    "pr_points.csv": {"model", "precision", "recall"},
    "calibration_points.csv": {"model", "bin", "count", "mean_predicted_probability", "observed_event_fraction"},
    "bootstrap_ci.csv": {"model", "accuracy_mean", "accuracy_ci_low", "accuracy_ci_high", "roc_auc_mean", "roc_auc_ci_low", "roc_auc_ci_high"},
    "clinical_threshold_metrics.csv": {"model", "threshold_type", "threshold", "sensitivity", "specificity", "ppv", "npv"},
    "statistical_tests.csv": {"comparison", "metric", "n_pairs", "mean_difference", "median_difference", "p_value", "effect_direction"},
    "run_config.json": set(),
}

REQUIRED_NPY_FILES = ["train_idx.npy", "test_idx.npy"]

OPTIONAL_FILES = ["shap_values.csv", "shap_values.parquet", "final_model_artifact.joblib"]


def read_header(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            return set(next(reader))
        except StopIteration:
            return set()


def main() -> int:
    artifact_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts")
    failures = []

    if not artifact_dir.exists():
        failures.append(f"Missing artifact directory: {artifact_dir}")
    else:
        for filename, required_columns in REQUIRED_FILES.items():
            path = artifact_dir / filename
            if not path.exists():
                failures.append(f"Missing required file: {path}")
                continue
            if path.suffix == ".csv" and required_columns:
                header = read_header(path)
                missing = sorted(required_columns - header)
                if missing:
                    failures.append(f"{path} missing columns: {', '.join(missing)}")

        for filename in REQUIRED_NPY_FILES:
            path = artifact_dir / filename
            if not path.exists():
                failures.append(f"Missing required file: {path}")

        # NPY exports are optional when CSV prediction exports are present.
        npy_dir = artifact_dir / "npy"
        if npy_dir.exists():
            for model in ["svm", "svm_balanced", "lr", "lr_balanced", "rf", "rf_balanced", "etc", "xgbrf", "gb"]:
                for suffix in ["y_true_test.npy", "y_pred_test.npy", "y_proba_test.npy"]:
                    path = npy_dir / f"{model}_{suffix}"
                    if not path.exists():
                        failures.append(f"Missing optional NPY consistency file: {path}")

        if not any((artifact_dir / filename).exists() for filename in OPTIONAL_FILES):
            failures.append(
                "No SHAP/model artifact found. Add shap_values.csv, shap_values.parquet, "
                "or final_model_artifact.joblib if explainability claims are included."
            )

    if failures:
        print("Artifact validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Artifact validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
