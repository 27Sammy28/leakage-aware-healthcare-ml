#!/usr/bin/env python3
"""Build auditable submission artifacts from real model outputs.

Inputs must come from the final ML rerun, not from manuscript aggregate tables.
The script writes reviewer-auditable CSV/NPY artifacts, statistical tests,
bootstrap intervals, and clinical-threshold summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

MODELS = ["SVM", "LR", "RF", "ETC", "XGBRF", "GB"]
REQUIRED_PRED_COLUMNS = {"model", "sample_id", "y_true", "y_pred", "y_prob"}
REQUIRED_FOLD_COLUMNS = {
    "model", "sampling", "repeat", "fold", "accuracy", "precision",
    "recall", "f1", "roc_auc", "pr_auc", "brier"
}


def safe_name(model: str) -> str:
    return model.lower().replace(" ", "_").replace("+", "plus")


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    prob_sorted = y_prob[order]
    distinct = np.where(np.diff(prob_sorted))[0]
    threshold_idxs = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both positive and negative labels.")
    tpr = np.r_[0, tps / positives, 1]
    fpr = np.r_[0, fps / negatives, 1]
    return fpr, tpr


def precision_recall_points(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    prob_sorted = y_prob[order]
    distinct = np.where(np.diff(prob_sorted))[0]
    threshold_idxs = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    positives = np.sum(y_true == 1)
    if positives == 0:
        raise ValueError("PR-AUC requires positive labels.")
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positives
    return np.r_[1, precision], np.r_[0, recall]


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall = precision_recall_points(y_true, y_prob)
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict[str, float | int]:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    npv = tn / max(tn + fn, 1)
    ppv = precision
    result = {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1, "ppv": ppv, "npv": npv,
    }
    if y_prob is not None:
        fpr, tpr = roc_curve_points(y_true, y_prob)
        result["roc_auc"] = auc_trapezoid(fpr, tpr)
        result["pr_auc"] = average_precision(y_true, y_prob)
        result["brier"] = float(np.mean((y_prob - y_true) ** 2))
    return result


def calibration_rows(model: str, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict[str, float | int | str]]:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(np.sum(mask))
        if count == 0:
            continue
        rows.append({
            "model": model,
            "bin": bin_id,
            "count": count,
            "mean_predicted_probability": float(np.mean(y_prob[mask])),
            "observed_event_fraction": float(np.mean(y_true[mask])),
        })
    return rows


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, seed: int, n_bootstrap: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = {"accuracy": [], "roc_auc": []}
    n = y_true.size
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        metrics = binary_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        values["accuracy"].append(float(metrics["accuracy"]))
        values["roc_auc"].append(float(metrics["roc_auc"]))
    output = {}
    for metric, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        output[f"{metric}_mean"] = float(np.mean(arr))
        output[f"{metric}_ci_low"] = float(np.quantile(arr, 0.025))
        output[f"{metric}_ci_high"] = float(np.quantile(arr, 0.975))
    return output


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict[str, float | int]]:
    thresholds = np.unique(np.r_[0.0, y_prob, 0.5, 1.0])
    best_threshold = 0.5
    best_metrics = binary_metrics(y_true, (y_prob >= 0.5).astype(int), y_prob)
    for threshold in thresholds:
        metrics = binary_metrics(y_true, (y_prob >= threshold).astype(int), y_prob)
        if float(metrics["f1"]) > float(best_metrics["f1"]):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def require_columns(data: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(missing)}")


def ensure_models_present(models: Iterable[str]) -> None:
    missing = sorted(set(MODELS) - set(models))
    if missing:
        raise ValueError(f"Missing required model outputs for: {', '.join(missing)}")


def write_predictions_artifacts(predictions: pd.DataFrame, output_dir: Path, seed: int, n_bootstrap: int) -> None:
    ensure_models_present(predictions["model"].unique())
    npy_dir = output_dir / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)

    confusion_rows = []
    roc_rows = []
    pr_rows = []
    cal_rows = []
    bootstrap_rows = []
    threshold_rows = []

    for model in MODELS:
        model_data = predictions[predictions["model"] == model].copy()
        model_data = model_data.sort_values("sample_id")
        y_true = model_data["y_true"].astype(int).to_numpy()
        y_pred = model_data["y_pred"].astype(int).to_numpy()
        y_prob = model_data["y_prob"].astype(float).to_numpy()
        if y_true.size == 0:
            raise ValueError(f"No predictions found for {model}.")
        if not np.all(np.isin(y_true, [0, 1])) or not np.all(np.isin(y_pred, [0, 1])):
            raise ValueError(f"{model} has non-binary y_true or y_pred values.")
        if np.any((y_prob < 0) | (y_prob > 1)):
            raise ValueError(f"{model} has probabilities outside [0, 1].")

        prefix = safe_name(model)
        np.save(npy_dir / f"{prefix}_y_true_test.npy", y_true)
        np.save(npy_dir / f"{prefix}_y_pred_test.npy", y_pred)
        np.save(npy_dir / f"{prefix}_y_proba_test.npy", y_prob)

        metrics = binary_metrics(y_true, y_pred, y_prob)
        confusion_rows.append({"model": model, **metrics})

        fpr, tpr = roc_curve_points(y_true, y_prob)
        for x_value, y_value in zip(fpr, tpr):
            roc_rows.append({"model": model, "fpr": x_value, "tpr": y_value})

        precision, recall = precision_recall_points(y_true, y_prob)
        for x_value, y_value in zip(precision, recall):
            pr_rows.append({"model": model, "precision": x_value, "recall": y_value})

        cal_rows.extend(calibration_rows(model, y_true, y_prob))
        bootstrap_rows.append({"model": model, **bootstrap_ci(y_true, y_pred, y_prob, seed, n_bootstrap)})

        threshold_rows.append({"model": model, "threshold_type": "fixed_0.5", "threshold": 0.5, **metrics})
        best_threshold, best_metrics = best_f1_threshold(y_true, y_prob)
        threshold_rows.append({"model": model, "threshold_type": "max_f1", "threshold": best_threshold, **best_metrics})

    predictions.to_csv(output_dir / "heldout_predictions.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_dir / "confusion_matrices.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(output_dir / "roc_points.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(output_dir / "pr_points.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(output_dir / "calibration_points.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output_dir / "bootstrap_ci.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_dir / "clinical_threshold_metrics.csv", index=False)


def write_fold_artifacts(fold_metrics: pd.DataFrame, output_dir: Path) -> None:
    ensure_models_present(fold_metrics["model"].unique())
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)

    comparisons = [("ETC", "RF"), ("ETC", "GB"), ("ETC", "SVM")]
    metrics = ["roc_auc", "f1"]
    rows = []
    keys = ["repeat", "fold"]
    for left, right in comparisons:
        left_data = fold_metrics[fold_metrics["model"] == left]
        right_data = fold_metrics[fold_metrics["model"] == right]
        paired = left_data.merge(right_data, on=keys, suffixes=("_left", "_right"))
        if paired.empty:
            raise ValueError(f"No paired fold rows found for {left} vs {right}.")
        for metric in metrics:
            diff = paired[f"{metric}_left"] - paired[f"{metric}_right"]
            if np.allclose(diff, 0):
                statistic, p_value = 0.0, 1.0
            else:
                result = stats.wilcoxon(paired[f"{metric}_left"], paired[f"{metric}_right"], zero_method="wilcox")
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            rows.append({
                "comparison": f"{left} vs {right}",
                "metric": metric,
                "n_pairs": int(len(paired)),
                "mean_difference": float(np.mean(diff)),
                "median_difference": float(np.median(diff)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "effect_direction": "left_higher" if np.mean(diff) > 0 else "right_higher" if np.mean(diff) < 0 else "no_difference",
            })
    pd.DataFrame(rows).to_csv(output_dir / "statistical_tests.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create auditable BMC submission artifacts from real final-run outputs.")
    parser.add_argument("--predictions", required=True, help="CSV with model,sample_id,y_true,y_pred,y_prob.")
    parser.add_argument("--fold-metrics", required=True, help="CSV with one row per model/repeat/fold.")
    parser.add_argument("--train-idx", required=True, help="Numpy .npy file containing training indices.")
    parser.add_argument("--test-idx", required=True, help="Numpy .npy file containing held-out test indices.")
    parser.add_argument("--output-dir", default="artifacts", help="Output artifact directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for split/model rerun.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap resamples for CIs.")
    parser.add_argument("--dataset-checksum", default="", help="Optional dataset checksum from the final input file.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = Path(args.predictions)
    folds_path = Path(args.fold_metrics)
    predictions = pd.read_csv(predictions_path)
    fold_metrics = pd.read_csv(folds_path)
    require_columns(predictions, REQUIRED_PRED_COLUMNS, predictions_path)
    require_columns(fold_metrics, REQUIRED_FOLD_COLUMNS, folds_path)

    train_idx = np.load(args.train_idx)
    test_idx = np.load(args.test_idx)
    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    write_predictions_artifacts(predictions, output_dir, args.seed, args.n_bootstrap)
    write_fold_artifacts(fold_metrics, output_dir)

    config = {
        "random_seed": args.seed,
        "models": MODELS,
        "source_predictions": str(predictions_path),
        "source_fold_metrics": str(folds_path),
        "train_idx_file": "train_idx.npy",
        "test_idx_file": "test_idx.npy",
        "dataset_checksum": args.dataset_checksum,
        "bootstrap_resamples": args.n_bootstrap,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Submission artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
