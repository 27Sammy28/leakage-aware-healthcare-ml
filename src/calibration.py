"""Calibration and calibrated-voting utilities."""
import numpy as np
from scipy.optimize import minimize
from .models import sigmoid

def calibration_intercept_slope(y_true, probabilities):
    eps = 1e-6
    p = np.clip(probabilities, eps, 1 - eps)
    logits = np.log(p / (1 - p))
    design = np.c_[np.ones(len(y_true)), logits]
    def objective_and_gradient(coef):
        scores = design @ coef
        fitted = sigmoid(scores)
        return np.mean(np.logaddexp(0, scores) - y_true * scores), design.T @ (fitted - y_true) / len(y_true)
    result = minimize(lambda c: objective_and_gradient(c)[0], np.array([0.0, 1.0]), jac=lambda c: objective_and_gradient(c)[1], method="BFGS")
    return float(result.x[0]), float(result.x[1])

def lacve_weights(metric_by_model, epsilon=1e-6):
    raw = {m: max(v["roc_auc"], epsilon) / (v["brier"] + abs(v["calibration_slope"] - 1.0) + epsilon) for m, v in metric_by_model.items()}
    total = sum(raw.values())
    return {m: v / total for m, v in raw.items()}

def weighted_vote(probability_by_model, weights):
    return sum(weights[model] * probability_by_model[model] for model in weights)

cal_slope = calibration_intercept_slope
