#!/usr/bin/env python3
"""Leakage-aware UCI Heart Disease external validation using NumPy/SciPy only."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

RNG_SEED = 42
NUMERIC = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
TARGET = "target"


def load_uci(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TARGET] = (df["num"].astype(int) > 0).astype(int)
    return df[["dataset", *NUMERIC, *CATEGORICAL, TARGET]].copy()


@dataclass
class Preprocessor:
    medians: dict
    means: np.ndarray
    stds: np.ndarray
    categories: dict

    @classmethod
    def fit(cls, df: pd.DataFrame):
        numeric = df[NUMERIC].copy()
        medians = {c: float(numeric[c].median()) for c in NUMERIC}
        numeric = numeric.fillna(medians)
        means = numeric.to_numpy(float).mean(axis=0)
        stds = numeric.to_numpy(float).std(axis=0)
        stds[stds == 0] = 1.0
        categories = {}
        for c in CATEGORICAL:
            vals = df[c].astype(object).where(df[c].notna(), "Missing").astype(str)
            categories[c] = sorted(vals.unique().tolist())
        return cls(medians, means, stds, categories)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        numeric = df[NUMERIC].copy().fillna(self.medians).to_numpy(float)
        numeric = (numeric - self.means) / self.stds
        parts = [numeric]
        for c in CATEGORICAL:
            vals = df[c].astype(object).where(df[c].notna(), "Missing").astype(str).to_numpy()
            cats = self.categories[c]
            onehot = np.zeros((len(df), len(cats)), dtype=float)
            index = {v: i for i, v in enumerate(cats)}
            for row, val in enumerate(vals):
                if val in index:
                    onehot[row, index[val]] = 1.0
            parts.append(onehot)
        return np.hstack(parts)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class LogisticRegressionNP:
    def __init__(self, c=1.0):
        self.c = c
        self.coef_ = None

    def fit(self, x, y):
        xb = np.c_[np.ones(len(x)), x]
        lam = 1.0 / self.c
        def objective(w):
            z = xb @ w
            loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * np.sum(w[1:] ** 2) / len(y)
            p = sigmoid(z)
            grad = xb.T @ (p - y) / len(y)
            grad[1:] += lam * w[1:] / len(y)
            return loss, grad
        res = minimize(lambda w: objective(w)[0], np.zeros(xb.shape[1]), jac=lambda w: objective(w)[1], method="L-BFGS-B", options={"maxiter": 500})
        self.coef_ = res.x
        return self

    def predict_proba(self, x):
        return sigmoid(np.c_[np.ones(len(x)), x] @ self.coef_)


class GaussianNB:
    def fit(self, x, y):
        self.classes_ = np.array([0, 1])
        self.prior_ = np.array([(y == k).mean() for k in self.classes_])
        self.mean_ = np.vstack([x[y == k].mean(axis=0) for k in self.classes_])
        self.var_ = np.vstack([x[y == k].var(axis=0) + 1e-6 for k in self.classes_])
        return self

    def predict_proba(self, x):
        logs = []
        for i, k in enumerate(self.classes_):
            logp = np.log(self.prior_[i] + 1e-12) - 0.5 * np.sum(np.log(2 * np.pi * self.var_[i]) + ((x - self.mean_[i]) ** 2) / self.var_[i], axis=1)
            logs.append(logp)
        logs = np.vstack(logs).T
        logs -= logs.max(axis=1, keepdims=True)
        probs = np.exp(logs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, 1]


class NearestCentroid:
    def fit(self, x, y):
        self.centroids_ = np.vstack([x[y == 0].mean(axis=0), x[y == 1].mean(axis=0)])
        return self

    def predict_proba(self, x):
        d0 = np.linalg.norm(x - self.centroids_[0], axis=1)
        d1 = np.linalg.norm(x - self.centroids_[1], axis=1)
        return sigmoid(d0 - d1)


class ShallowTree:
    def __init__(self, max_depth=3, min_leaf=20):
        self.max_depth = max_depth
        self.min_leaf = min_leaf

    def fit(self, x, y):
        self.tree_ = self._build(x, y, 0)
        return self

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 2 * p * (1 - p)

    def _build(self, x, y, depth):
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or y.mean() in (0, 1):
            return ("leaf", float(y.mean()))
        best = None
        parent = self._gini(y)
        for j in range(x.shape[1]):
            vals = np.unique(np.quantile(x[:, j], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
            for t in vals:
                left = x[:, j] <= t
                nl = left.sum(); nr = len(y) - nl
                if nl < self.min_leaf or nr < self.min_leaf:
                    continue
                gain = parent - (nl * self._gini(y[left]) + nr * self._gini(y[~left])) / len(y)
                if best is None or gain > best[0]:
                    best = (gain, j, float(t), left)
        if best is None or best[0] <= 1e-9:
            return ("leaf", float(y.mean()))
        _, j, t, left = best
        return ("node", j, t, self._build(x[left], y[left], depth + 1), self._build(x[~left], y[~left], depth + 1))

    def _pred_one(self, row, node):
        if node[0] == "leaf":
            return node[1]
        _, j, t, left, right = node
        return self._pred_one(row, left if row[j] <= t else right)

    def predict_proba(self, x):
        return np.array([self._pred_one(row, self.tree_) for row in x])


def auc_score(y, s):
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1; npos = pos.sum(); nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def average_precision(y, s):
    order = np.argsort(-s)
    yy = y[order]
    tp = np.cumsum(yy == 1)
    precision = tp / (np.arange(len(y)) + 1)
    return float((precision * (yy == 1)).sum() / max(1, (y == 1).sum()))


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float((pred == y).mean()), "precision": precision, "recall": recall, "specificity": specificity,
        "f1": f1, "balanced_accuracy": (recall + specificity) / 2, "roc_auc": float(auc_score(y, prob)),
        "pr_auc": float(average_precision(y, prob)), "brier": float(np.mean((prob - y) ** 2)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def stratified_folds(y, n_splits=5, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, item in enumerate(idx):
            folds[i % n_splits].append(item)
    return [np.array(sorted(fold)) for fold in folds]


def evaluate_split(df, train_idx, test_idx, model_name, model):
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    prep = Preprocessor.fit(train)
    x_train = prep.transform(train); y_train = train[TARGET].to_numpy(int)
    x_test = prep.transform(test); y_test = test[TARGET].to_numpy(int)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)
    return metrics(y_test, prob)


def model_factory():
    return {
        "Logistic Regression": LogisticRegressionNP(c=1.0),
        "Gaussian NB": GaussianNB(),
        "Nearest Centroid": NearestCentroid(),
        "Shallow Tree": ShallowTree(max_depth=3, min_leaf=20),
    }


def main():
    out = Path('uci_external_artifacts')
    out.mkdir(exist_ok=True)
    df = load_uci(Path('prism-uploads/heart_disease_uci.csv'))
    y = df[TARGET].to_numpy(int)
    rows = []
    folds = stratified_folds(y, 5)
    for fold_id, test_idx in enumerate(folds, 1):
        train_idx = np.setdiff1d(np.arange(len(df)), test_idx)
        for name in model_factory():
            row = evaluate_split(df, train_idx, test_idx, name, model_factory()[name])
            row.update({"evaluation": "5-fold CV", "fold": fold_id, "heldout_site": "", "model": name, "train_n": len(train_idx), "test_n": len(test_idx)})
            rows.append(row)
    for site in sorted(df['dataset'].unique()):
        test_idx = np.where(df['dataset'].to_numpy() == site)[0]
        train_idx = np.where(df['dataset'].to_numpy() != site)[0]
        for name in model_factory():
            row = evaluate_split(df, train_idx, test_idx, name, model_factory()[name])
            row.update({"evaluation": "leave-site-out", "fold": "", "heldout_site": site, "model": name, "train_n": len(train_idx), "test_n": len(test_idx)})
            rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(out / 'uci_external_validation_results.csv', index=False)
    summary = results.groupby(['evaluation','model'], dropna=False)[['accuracy','f1','balanced_accuracy','roc_auc','pr_auc','brier']].agg(['mean','std']).reset_index()
    summary.columns = ['_'.join([str(x) for x in col if x]) for col in summary.columns]
    summary.to_csv(out / 'uci_external_validation_summary.csv', index=False)
    site = results[results['evaluation']=='leave-site-out'].copy()
    site.to_csv(out / 'uci_leave_site_out_results.csv', index=False)
    metadata = {
        "dataset": "UCI Heart Disease combined sources",
        "n_records": int(len(df)),
        "target_definition": "num > 0 coded as cardiovascular disease present",
        "class_counts": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "site_counts": {str(k): int(v) for k, v in df['dataset'].value_counts().items()},
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "protocol": "Train-fold-only median imputation, standardization, and one-hot encoding; 5-fold stratified CV plus leave-one-source-out validation.",
    }
    (out / 'uci_external_validation_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print('\nLeave-site-out:')
    print(site[['heldout_site','model','test_n','accuracy','f1','balanced_accuracy','roc_auc','pr_auc','brier','tn','fp','fn','tp']].to_string(index=False))

if __name__ == '__main__':
    main()
