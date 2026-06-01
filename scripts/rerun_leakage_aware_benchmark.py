#!/usr/bin/env python3
"""Rerun the leakage-aware heart-failure benchmark and export audit artifacts.

This script is intended for the final analysis environment where scikit-learn,
imblearn, and xgboost are available. It intentionally fails fast if those
packages are missing rather than silently generating partial results.

Example:
    python rerun_leakage_aware_benchmark.py \
        --data heart_failure_clinical_records_dataset.csv \
        --output-dir artifacts
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMN = "DEATH_EVENT"
FEATURE_COLUMNS = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time",
]


@dataclass(frozen=True)
class ExperimentConfig:
    random_seed: int = 42
    test_size: float = 0.20
    n_splits: int = 5
    n_repeats: int = 5
    bootstrap_repeats: int = 2000


def require_ml_packages():
    try:
        from imblearn.over_sampling import ADASYN, SMOTE
        from imblearn.pipeline import Pipeline
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            brier_score_loss,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.svm import SVC
        from xgboost import XGBRFClassifier
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing required ML package. Run this script in the original analysis "
            "environment with scikit-learn, imbalanced-learn, and xgboost installed. "
            f"Missing import: {exc.name}"
        ) from exc

    return {
        "ADASYN": ADASYN,
        "SMOTE": SMOTE,
        "Pipeline": Pipeline,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "RepeatedStratifiedKFold": RepeatedStratifiedKFold,
        "StratifiedKFold": StratifiedKFold,
        "SVC": SVC,
        "XGBRFClassifier": XGBRFClassifier,
        "MinMaxScaler": MinMaxScaler,
        "train_test_split": train_test_split,
        "accuracy_score": accuracy_score,
        "average_precision_score": average_precision_score,
        "brier_score_loss": brier_score_loss,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
    }


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path)
    missing = sorted(set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return data[FEATURE_COLUMNS].copy(), data[TARGET_COLUMN].astype(int).copy()


def safe_probability(estimator, x_values: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x_values)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x_values)
        return 1.0 / (1.0 + np.exp(-scores))
    return estimator.predict(x_values).astype(float)


def build_models(pkgs, seed: int):
    LogisticRegression = pkgs["LogisticRegression"]
    RandomForestClassifier = pkgs["RandomForestClassifier"]
    ExtraTreesClassifier = pkgs["ExtraTreesClassifier"]
    GradientBoostingClassifier = pkgs["GradientBoostingClassifier"]
    SVC = pkgs["SVC"]
    XGBRFClassifier = pkgs["XGBRFClassifier"]

    return {
        "SVM": SVC(kernel="rbf", probability=True, random_state=seed),
        "SVM_balanced": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "LR": LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed),
        "LR_balanced": LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1),
        "RF_balanced": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1),
        "ETC": ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1),
        "GB": GradientBoostingClassifier(random_state=seed),
        "XGBRF": XGBRFClassifier(
            n_estimators=500,
            subsample=0.8,
            colsample_bynode=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def build_samplers(pkgs, seed: int):
    return {
        "none": None,
        "SMOTE": pkgs["SMOTE"](random_state=seed),
        "ADASYN": pkgs["ADASYN"](random_state=seed),
    }


def metric_row(pkgs, y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "accuracy": pkgs["accuracy_score"](y_true, y_pred),
        "precision": pkgs["precision_score"](y_true, y_pred, zero_division=0),
        "recall": pkgs["recall_score"](y_true, y_pred, zero_division=0),
        "f1": pkgs["f1_score"](y_true, y_pred, zero_division=0),
        "roc_auc": pkgs["roc_auc_score"](y_true, y_prob),
        "pr_auc": pkgs["average_precision_score"](y_true, y_prob),
        "brier": pkgs["brier_score_loss"](y_true, y_prob),
    }


def calibration_rows(model: str, sampling: str, y_true, y_prob, n_bins: int = 10) -> list[dict]:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        rows.append({
            "model": model,
            "sampling": sampling,
            "bin": bin_id,
            "count": int(mask.sum()),
            "mean_predicted_probability": float(np.mean(y_prob[mask])),
            "observed_event_fraction": float(np.mean(y_true[mask])),
        })
    return rows


def curve_rows(model: str, sampling: str, y_true, y_prob, pkgs) -> tuple[list[dict], list[dict]]:
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    roc_rows = [{"model": model, "sampling": sampling, "fpr": float(x), "tpr": float(y)} for x, y in zip(fpr, tpr)]
    pr_rows = [
        {"model": model, "sampling": sampling, "precision": float(x), "recall": float(y)}
        for x, y in zip(precision, recall)
    ]
    return roc_rows, pr_rows


def make_pipeline(pkgs, sampler, model):
    steps = [("scale", pkgs["MinMaxScaler"]())]
    if sampler is not None:
        steps.append(("sample", sampler))
    steps.append(("model", model))
    return pkgs["Pipeline"](steps)


def run_experiment(data_path: Path, output_dir: Path, config: ExperimentConfig) -> None:
    pkgs = require_ml_packages()
    x_data, y_data = load_data(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "npy").mkdir(exist_ok=True)

    train_test_split = pkgs["train_test_split"]
    x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(
        x_data,
        y_data,
        np.arange(len(y_data)),
        test_size=config.test_size,
        stratify=y_data,
        random_state=config.random_seed,
    )
    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    models = build_models(pkgs, config.random_seed)
    samplers = build_samplers(pkgs, config.random_seed)
    cv = pkgs["RepeatedStratifiedKFold"](
        n_splits=config.n_splits,
        n_repeats=config.n_repeats,
        random_state=config.random_seed,
    )

    fold_rows = []
    fold_pred_rows = []
    heldout_rows = []
    confusion_rows = []
    roc_rows = []
    pr_rows = []
    cal_rows = []

    for model_name, model in models.items():
        for sampling_name, sampler in samplers.items():
            if "balanced" in model_name and sampling_name != "none":
                continue
            for fold_id, (inner_train_idx, valid_idx) in enumerate(cv.split(x_train, y_train)):
                pipe = make_pipeline(pkgs, sampler, model)
                x_inner_train = x_train.iloc[inner_train_idx]
                y_inner_train = y_train.iloc[inner_train_idx]
                x_valid = x_train.iloc[valid_idx]
                y_valid = y_train.iloc[valid_idx]
                pipe.fit(x_inner_train, y_inner_train)
                y_prob = safe_probability(pipe, x_valid)
                y_pred = (y_prob >= 0.5).astype(int)
                repeat = fold_id // config.n_splits
                fold = fold_id % config.n_splits
                fold_rows.append({
                    "model": model_name,
                    "sampling": sampling_name,
                    "repeat": repeat,
                    "fold": fold,
                    **metric_row(pkgs, y_valid, y_pred, y_prob),
                })
                for sample_id, truth, pred, prob in zip(train_idx[valid_idx], y_valid, y_pred, y_prob):
                    fold_pred_rows.append({
                        "repeat": repeat,
                        "fold": fold,
                        "fold_id": fold_id,
                        "model": model_name,
                        "sampling": sampling_name,
                        "sample_id": int(sample_id),
                        "y_true": int(truth),
                        "y_pred": int(pred),
                        "y_pred_proba": float(prob),
                    })

            pipe = make_pipeline(pkgs, sampler, model)
            pipe.fit(x_train, y_train)
            y_prob = safe_probability(pipe, x_test)
            y_pred = (y_prob >= 0.5).astype(int)
            for sample_id, truth, pred, prob in zip(test_idx, y_test, y_pred, y_prob):
                heldout_rows.append({
                    "model": model_name,
                    "sampling": sampling_name,
                    "sample_id": int(sample_id),
                    "y_true": int(truth),
                    "y_pred": int(pred),
                    "y_prob": float(prob),
                })
            tn, fp, fn, tp = pkgs["confusion_matrix"](y_test, y_pred, labels=[0, 1]).ravel()
            confusion_rows.append({
                "model": model_name,
                "sampling": sampling_name,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                **metric_row(pkgs, y_test, y_pred, y_prob),
            })
            model_roc_rows, model_pr_rows = curve_rows(model_name, sampling_name, y_test, y_prob, pkgs)
            roc_rows.extend(model_roc_rows)
            pr_rows.extend(model_pr_rows)
            cal_rows.extend(calibration_rows(model_name, sampling_name, y_test.to_numpy(), y_prob))

    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(fold_pred_rows).to_csv(output_dir / "fold_predictions.csv", index=False)
    pd.DataFrame(heldout_rows).to_csv(output_dir / "heldout_predictions.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(output_dir / "confusion_matrices.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(output_dir / "roc_points.csv", index=False)
    pd.DataFrame(pr_rows).to_csv(output_dir / "pr_points.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(output_dir / "calibration_points.csv", index=False)

    with (output_dir / "run_config.json").open("w") as handle:
        json.dump({
            "data_path": str(data_path),
            "target_column": TARGET_COLUMN,
            "feature_columns": FEATURE_COLUMNS,
            "random_seed": config.random_seed,
            "test_size": config.test_size,
            "n_splits": config.n_splits,
            "n_repeats": config.n_repeats,
            "models": list(models),
            "samplers": list(samplers),
        }, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Path to heart_failure_clinical_records_dataset.csv")
    parser.add_argument("--output-dir", default=Path("artifacts"), type=Path)
    parser.add_argument("--random-seed", default=42, type=int)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--n-repeats", default=5, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(random_seed=args.random_seed, n_splits=args.n_splits, n_repeats=args.n_repeats)
    run_experiment(args.data, args.output_dir, config)


if __name__ == "__main__":
    main()
