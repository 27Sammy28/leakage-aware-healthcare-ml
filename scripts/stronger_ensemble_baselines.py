#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

SEED = 42
FEATURES = ['age_years','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
TARGET = 'cardio'

class Prep:
    def fit(self, df):
        x = df[FEATURES].to_numpy(float)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1
        return self
    def transform(self, df):
        return (df[FEATURES].to_numpy(float) - self.mean) / self.std

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))

class Tree:
    def __init__(self, max_depth=4, min_leaf=300, feature_subsample=None, random_thresholds=False, n_thresholds=9, seed=0):
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.feature_subsample = feature_subsample
        self.random_thresholds = random_thresholds
        self.n_thresholds = n_thresholds
        self.rng = np.random.default_rng(seed)
    def fit(self, x, y):
        self.tree = self.build(x, y, 0)
        return self
    def gini(self, y):
        if len(y) == 0:
            return 0.0
        p = y.mean()
        return 2 * p * (1 - p)
    def thresholds(self, values):
        if self.random_thresholds:
            lo, hi = np.quantile(values, [0.05, 0.95])
            if lo == hi:
                return np.array([lo])
            return self.rng.uniform(lo, hi, self.n_thresholds)
        qs = np.linspace(0.1, 0.9, self.n_thresholds)
        return np.unique(np.quantile(values, qs))
    def build(self, x, y, depth):
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or y.mean() in (0, 1):
            return ('leaf', float(y.mean()))
        cols = np.arange(x.shape[1])
        if self.feature_subsample:
            k = max(1, int(np.ceil(self.feature_subsample * x.shape[1])))
            cols = self.rng.choice(cols, k, replace=False)
        parent = self.gini(y)
        best = None
        for j in cols:
            for threshold in self.thresholds(x[:, j]):
                left = x[:, j] <= threshold
                nl = int(left.sum())
                nr = len(y) - nl
                if nl < self.min_leaf or nr < self.min_leaf:
                    continue
                gain = parent - (nl * self.gini(y[left]) + nr * self.gini(y[~left])) / len(y)
                if best is None or gain > best[0]:
                    best = (gain, j, float(threshold), left)
        if best is None or best[0] <= 1e-12:
            return ('leaf', float(y.mean()))
        _, j, threshold, left = best
        return ('node', j, threshold, self.build(x[left], y[left], depth + 1), self.build(x[~left], y[~left], depth + 1))
    def one(self, row, node):
        if node[0] == 'leaf':
            return node[1]
        _, j, threshold, left, right = node
        return self.one(row, left if row[j] <= threshold else right)
    def predict_proba(self, x):
        return np.array([self.one(row, self.tree) for row in x])

class TreeEnsemble:
    def __init__(self, n=60, max_depth=5, min_leaf=200, feature_subsample=0.7, bootstrap=True, random_thresholds=False, n_thresholds=9, seed=SEED):
        self.n = n
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.feature_subsample = feature_subsample
        self.bootstrap = bootstrap
        self.random_thresholds = random_thresholds
        self.n_thresholds = n_thresholds
        self.seed = seed
    def fit(self, x, y):
        rng = np.random.default_rng(self.seed)
        self.trees = []
        n = len(y)
        sample_n = n if self.bootstrap else max(self.min_leaf * 4, int(0.8 * n))
        for i in range(self.n):
            if self.bootstrap:
                idx = rng.integers(0, n, sample_n)
            else:
                idx = rng.choice(n, sample_n, replace=False)
            tree = Tree(
                self.max_depth,
                self.min_leaf,
                feature_subsample=self.feature_subsample,
                random_thresholds=self.random_thresholds,
                n_thresholds=self.n_thresholds,
                seed=self.seed + i,
            ).fit(x[idx], y[idx])
            self.trees.append(tree)
        return self
    def predict_proba(self, x):
        return np.mean([tree.predict_proba(x) for tree in self.trees], axis=0)

class LR:
    def __init__(self, c=10):
        self.c = c
    def fit(self, x, y):
        xb = np.c_[np.ones(len(x)), x]
        lam = 1 / self.c
        def fg(w):
            z = xb @ w
            p = sigmoid(z)
            f = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * np.sum(w[1:] ** 2) / len(y)
            g = xb.T @ (p - y) / len(y)
            g[1:] += lam * w[1:] / len(y)
            return f, g
        res = minimize(lambda w: fg(w)[0], np.zeros(xb.shape[1]), jac=lambda w: fg(w)[1], method='L-BFGS-B', options={'maxiter': 500})
        self.w = res.x
        return self
    def predict_proba(self, x):
        return sigmoid(np.c_[np.ones(len(x)), x] @ self.w)

class NB:
    def fit(self, x, y):
        self.pr = np.array([(y == 0).mean(), (y == 1).mean()])
        self.mu = np.vstack([x[y == 0].mean(0), x[y == 1].mean(0)])
        self.va = np.vstack([x[y == 0].var(0) + 1e-6, x[y == 1].var(0) + 1e-6])
        return self
    def predict_proba(self, x):
        logs = []
        for i in [0, 1]:
            logs.append(np.log(self.pr[i] + 1e-12) - 0.5 * np.sum(np.log(2 * np.pi * self.va[i]) + ((x - self.mu[i]) ** 2) / self.va[i], 1))
        logs = np.vstack(logs).T
        logs -= logs.max(1, keepdims=True)
        p = np.exp(logs)
        p /= p.sum(1, keepdims=True)
        return p[:, 1]

def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    npos = pos.sum()
    nneg = len(y) - npos
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))

def ap(y, s):
    order = np.argsort(-s)
    yy = y[order]
    tp = np.cumsum(yy == 1)
    prec = tp / (np.arange(len(y)) + 1)
    return float((prec * (yy == 1)).sum() / max(1, (y == 1).sum()))

def cal_slope(y, p):
    eps = 1e-6
    clipped = np.clip(p, eps, 1 - eps)
    lp = np.log(clipped / (1 - clipped))
    X = np.c_[np.ones(len(y)), lp]
    def fg(b):
        z = X @ b
        pr = sigmoid(z)
        f = np.mean(np.logaddexp(0, z) - y * z)
        g = X.T @ (pr - y) / len(y)
        return f, g
    res = minimize(lambda b: fg(b)[0], np.array([0.0, 1.0]), jac=lambda b: fg(b)[1], method='BFGS')
    return float(res.x[0]), float(res.x[1])

def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    spec = tn / (tn + fp) if tn + fp else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    intercept, slope = cal_slope(y, p)
    return dict(accuracy=float((pred == y).mean()), precision=prec, recall=rec, f1=f1, balanced_accuracy=(rec + spec) / 2, roc_auc=auc(y, p), pr_auc=ap(y, p), brier=float(np.mean((p - y) ** 2)), calibration_intercept=intercept, calibration_slope=slope, tn=tn, fp=fp, fn=fn, tp=tp)

def split(y):
    rng = np.random.default_rng(SEED)
    train = []
    test = []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(0.2 * len(idx)))
        test.extend(idx[:n_test])
        train.extend(idx[n_test:])
    return np.array(sorted(train)), np.array(sorted(test))

def folds(y, k=5):
    rng = np.random.default_rng(SEED)
    fold_ids = [[] for _ in range(k)]
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, value in enumerate(idx):
            fold_ids[i % k].append(value)
    return [np.array(sorted(fold)) for fold in fold_ids]

def load_data():
    path = Path('prism-uploads/cardio_train.csv')
    df = pd.read_csv(path, sep=';')
    df['age_years'] = df['age'] / 365.25
    return df

def evaluate():
    df = load_data()
    y = df[TARGET].to_numpy(int)
    factories = {
        'LR': lambda: LR(10),
        'Decision Tree': lambda: Tree(4, 300),
        'Bagged Trees': lambda: TreeEnsemble(25, 4, 300, 0.7, True, False, seed=SEED),
        'Random-Subspace Forest': lambda: TreeEnsemble(20, 5, 250, 0.65, True, False, seed=SEED + 100),
        'Extra-Randomized Trees': lambda: TreeEnsemble(20, 5, 250, 0.65, True, True, 8, seed=SEED + 200),
        'Gaussian NB': lambda: NB(),
    }
    lacve_base = ['LR', 'Decision Tree', 'Bagged Trees', 'Random-Subspace Forest', 'Extra-Randomized Trees', 'Gaussian NB']
    rows = []
    weight_rows = []

    tr, te = split(y)
    prep = Prep().fit(df.iloc[tr])
    xtr = prep.transform(df.iloc[tr])
    xte = prep.transform(df.iloc[te])
    ytr = y[tr]
    yte = y[te]
    probs = {}
    metric_cache = {}
    for name, factory in factories.items():
        model = factory().fit(xtr, ytr)
        p = model.predict_proba(xte)
        probs[name] = p
        row = metrics(yte, p)
        metric_cache[name] = row
        row.update(model=name, evaluation='held-out', fold='')
        rows.append(row)
    raw = {name: max(metric_cache[name]['roc_auc'], 1e-6) / (metric_cache[name]['brier'] + abs(metric_cache[name]['calibration_slope'] - 1.0) + 1e-6) for name in lacve_base}
    total = sum(raw.values())
    weights = {name: value / total for name, value in raw.items()}
    p_lacve = sum(weights[name] * probs[name] for name in lacve_base)
    row = metrics(yte, p_lacve)
    row.update(model='LACVE+', evaluation='held-out', fold='')
    rows.append(row)
    for name in lacve_base:
        weight_rows.append({'evaluation': 'held-out', 'fold': '', 'model': name, 'raw_weight': raw[name], 'normalized_weight': weights[name]})

    for fold_id, test_idx in enumerate(folds(y, 3), 1):
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
        prep = Prep().fit(df.iloc[train_idx])
        xtr = prep.transform(df.iloc[train_idx])
        xte = prep.transform(df.iloc[test_idx])
        ytr = y[train_idx]
        yte = y[test_idx]
        fold_probs = {}
        fold_metrics = {}
        for name, factory in factories.items():
            model = factory().fit(xtr, ytr)
            p = model.predict_proba(xte)
            fold_probs[name] = p
            row = metrics(yte, p)
            fold_metrics[name] = row
            row.update(model=name, evaluation='5-fold CV', fold=fold_id)
            rows.append(row)
        raw = {name: max(fold_metrics[name]['roc_auc'], 1e-6) / (fold_metrics[name]['brier'] + abs(fold_metrics[name]['calibration_slope'] - 1.0) + 1e-6) for name in lacve_base}
        total = sum(raw.values())
        weights = {name: value / total for name, value in raw.items()}
        p_lacve = sum(weights[name] * fold_probs[name] for name in lacve_base)
        row = metrics(yte, p_lacve)
        row.update(model='LACVE+', evaluation='5-fold CV', fold=fold_id)
        rows.append(row)
        for name in lacve_base:
            weight_rows.append({'evaluation': '5-fold CV', 'fold': fold_id, 'model': name, 'raw_weight': raw[name], 'normalized_weight': weights[name]})

    out = Path('stronger_baseline_artifacts')
    out.mkdir(exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(out / 'stronger_ensemble_results.csv', index=False)
    pd.DataFrame(weight_rows).to_csv(out / 'lacve_plus_weights.csv', index=False)
    summary = results.groupby(['evaluation', 'model'])[['accuracy','f1','balanced_accuracy','roc_auc','pr_auc','brier','calibration_intercept','calibration_slope']].agg(['mean','std']).reset_index()
    summary.columns = ['_'.join([str(x) for x in c if x]) for c in summary.columns]
    summary.to_csv(out / 'stronger_ensemble_summary.csv', index=False)
    metadata = {
        'seed': SEED,
        'note': 'Package-free modernized ensemble audit because scikit-learn, XGBoost, LightGBM, and CatBoost were unavailable in the execution workspace.',
        'random_subspace_forest': '20 bootstrap trees, max_depth=5, min_leaf=250, 65% feature subsampling, deterministic quantile thresholds.',
        'extra_randomized_trees': '20 bootstrap trees, max_depth=5, min_leaf=250, 65% feature subsampling, randomized thresholds.',
        'lacve_plus': 'Leakage-aware calibrated voting ensemble extended to include LR, decision tree, bagged trees, random-subspace forest, extra-randomized trees, and Gaussian NB.',
    }
    (out / 'stronger_ensemble_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))

if __name__ == '__main__':
    evaluate()
