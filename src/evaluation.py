"""Evaluation metrics, decision-curve, and subgroup audit helpers."""
import numpy as np
import pandas as pd
from .calibration import calibration_intercept_slope

def auc_score(y_true, scores):
    order = np.argsort(scores); ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
    pos = y_true == 1; npos = pos.sum(); nneg = len(y_true) - npos
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def average_precision(y_true, scores):
    order = np.argsort(-scores); sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1); precision = tp / (np.arange(len(y_true)) + 1)
    return float((precision * (sorted_true == 1)).sum() / max(1, (y_true == 1).sum()))

def classification_metrics(y_true, probabilities, threshold=0.5):
    pred = (probabilities >= threshold).astype(int)
    tn = int(((y_true == 0) & (pred == 0)).sum()); fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum()); tp = int(((y_true == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0; f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    intercept, slope = calibration_intercept_slope(y_true, probabilities)
    return {"accuracy": float((pred == y_true).mean()), "precision": precision, "recall": recall, "f1": f1, "balanced_accuracy": (recall + specificity) / 2, "roc_auc": auc_score(y_true, probabilities), "pr_auc": average_precision(y_true, probabilities), "brier": float(np.mean((probabilities - y_true) ** 2)), "calibration_intercept": intercept, "calibration_slope": slope, "tn": tn, "fp": fp, "fn": fn, "tp": tp}

def net_benefit(y_true, probabilities, threshold):
    pred = probabilities >= threshold
    tp = ((y_true == 1) & pred).sum(); fp = ((y_true == 0) & pred).sum()
    return float(tp / len(y_true) - fp / len(y_true) * (threshold / (1 - threshold)))

def decision_curve_rows(y_true, probability_by_model, thresholds):
    rows = []; prevalence = float(y_true.mean())
    for model, probabilities in probability_by_model.items():
        for threshold in thresholds:
            rows.append({"model": model, "threshold": threshold, "net_benefit": net_benefit(y_true, probabilities, threshold), "treat_all_net_benefit": prevalence - (1 - prevalence) * threshold / (1 - threshold), "treat_none_net_benefit": 0.0})
    return pd.DataFrame(rows)

def subgroup_rows(frame, y_true, probabilities, group_column, model):
    rows = []
    for group, index in frame.groupby(group_column, observed=False).groups.items():
        index = np.array(list(index))
        if len(index) < 50 or len(np.unique(y_true[index])) < 2: continue
        m = classification_metrics(y_true[index], probabilities[index])
        rows.append({"model": model, "subgroup_type": group_column, "subgroup": str(group), "n": len(index), "event_rate": float(y_true[index].mean()), "accuracy": m["accuracy"], "recall": m["recall"], "specificity": m["tn"] / max(m["tn"] + m["fp"], 1), "roc_auc": m["roc_auc"], "brier": m["brier"], "calibration_slope": m["calibration_slope"]})
    return rows

auc = auc_score
ap = average_precision
metrics = classification_metrics
