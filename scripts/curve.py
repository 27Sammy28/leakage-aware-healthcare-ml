import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def roc_curve_points(y_true, y_prob):
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    prob_sorted = y_prob[order]
    distinct = np.where(np.diff(prob_sorted))[0]
    threshold_idxs = np.r_[distinct, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    tpr = np.r_[0, tps / positives, 1]
    fpr = np.r_[0, fps / negatives, 1]
    return fpr, tpr


def precision_recall_points(y_true, y_prob):
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    prob_sorted = y_prob[order]
    distinct = np.where(np.diff(prob_sorted))[0]
    threshold_idxs = np.r_[distinct, y_sorted.size - 1]

    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    positives = np.sum(y_true == 1)

    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / positives
    return np.r_[1, precision], np.r_[0, recall]


def calibration_points(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    prob_true = []
    prob_pred = []

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if np.any(mask):
            prob_true.append(np.mean(y_true[mask]))
            prob_pred.append(np.mean(y_prob[mask]))

    return np.asarray(prob_true), np.asarray(prob_pred)


def average_precision(y_true, y_prob):
    precision, recall = precision_recall_points(y_true, y_prob)
    return np.sum((recall[1:] - recall[:-1]) * precision[1:])


def validate_inputs(y_true, y_prob):
    if y_true.size != y_prob.size:
        raise ValueError("y_true and y_prob must contain the same number of rows.")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true must contain only binary labels 0 and 1.")
    if np.any((y_prob < 0) | (y_prob > 1)):
        raise ValueError("y_prob must contain probabilities between 0 and 1.")
    if len(np.unique(y_true)) != 2:
        raise ValueError("Both classes must be present to compute ROC and PR curves.")


def build_figure(input_csv, y_true_column, y_prob_column, output_path):
    data = pd.read_csv(input_csv)
    y_true = data[y_true_column].astype(int).to_numpy()
    y_prob = data[y_prob_column].astype(float).to_numpy()
    validate_inputs(y_true, y_prob)

    fpr, tpr = roc_curve_points(y_true, y_prob)
    roc_auc = np.trapz(tpr, fpr)

    precision, recall = precision_recall_points(y_true, y_prob)
    pr_auc = average_precision(y_true, y_prob)

    prob_true, prob_pred = calibration_points(y_true, y_prob, n_bins=10)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(fpr, tpr, linewidth=2)
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve (AUC = {roc_auc:.3f})")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(recall, precision, linewidth=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision--Recall Curve (AP = {pr_auc:.3f})")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(prob_pred, prob_true, marker="o", linewidth=2)
    axes[2].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[2].set_xlabel("Mean Predicted Probability")
    axes[2].set_ylabel("Fraction of Positives")
    axes[2].set_title("Calibration Curve")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Model Performance Evaluation (Discrimination + Calibration)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ROC, precision-recall, and calibration curves from real model outputs."
    )
    parser.add_argument("input_csv", help="CSV file containing ground-truth labels and predicted probabilities.")
    parser.add_argument("--y-true-column", default="y_true", help="Column containing binary labels.")
    parser.add_argument("--y-prob-column", default="y_prob", help="Column containing predicted probabilities.")
    parser.add_argument("--output", default="figure6_curves.png", help="Output image path.")
    args = parser.parse_args()

    build_figure(args.input_csv, args.y_true_column, args.y_prob_column, args.output)


if __name__ == "__main__":
    main()
