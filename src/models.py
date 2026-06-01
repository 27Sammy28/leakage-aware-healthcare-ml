"""Dependency-light model implementations used in the benchmark."""
import numpy as np
from scipy.optimize import minimize

def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40, 40)))

class LogisticRegressionGD:
    def __init__(self, c=10.0, max_iter=500):
        self.c = c
        self.max_iter = max_iter
    def fit(self, x_values, y_values):
        design = np.c_[np.ones(len(x_values)), x_values]
        penalty = 1.0 / self.c
        def objective_and_gradient(weights):
            scores = design @ weights
            probabilities = sigmoid(scores)
            objective = np.mean(np.logaddexp(0, scores) - y_values * scores) + 0.5 * penalty * np.sum(weights[1:] ** 2) / len(y_values)
            gradient = design.T @ (probabilities - y_values) / len(y_values)
            gradient[1:] += penalty * weights[1:] / len(y_values)
            return objective, gradient
        result = minimize(lambda w: objective_and_gradient(w)[0], np.zeros(design.shape[1]), jac=lambda w: objective_and_gradient(w)[1], method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.weights_ = result.x
        return self
    def predict_proba(self, x_values):
        return sigmoid(np.c_[np.ones(len(x_values)), x_values] @ self.weights_)

class ShallowDecisionTree:
    def __init__(self, max_depth=4, min_leaf=300, feature_subsample=None, seed=0):
        self.max_depth = max_depth; self.min_leaf = min_leaf; self.feature_subsample = feature_subsample; self.rng = np.random.default_rng(seed)
    def fit(self, x_values, y_values):
        self.tree_ = self._build(x_values, y_values, 0); return self
    @staticmethod
    def _gini(y_values):
        if len(y_values) == 0: return 0.0
        p = y_values.mean(); return float(2 * p * (1 - p))
    def _build(self, x_values, y_values, depth):
        if depth >= self.max_depth or len(y_values) < 2 * self.min_leaf or y_values.mean() in (0, 1):
            return ("leaf", float(y_values.mean()))
        columns = np.arange(x_values.shape[1])
        if self.feature_subsample:
            columns = self.rng.choice(columns, max(1, int(np.ceil(self.feature_subsample * x_values.shape[1]))), replace=False)
        parent = self._gini(y_values); best = None
        for col in columns:
            for threshold in np.unique(np.quantile(x_values[:, col], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])):
                left = x_values[:, col] <= threshold
                nl = left.sum(); nr = len(y_values) - nl
                if nl < self.min_leaf or nr < self.min_leaf: continue
                gain = parent - (nl * self._gini(y_values[left]) + nr * self._gini(y_values[~left])) / len(y_values)
                if best is None or gain > best[0]: best = (gain, col, float(threshold), left)
        if best is None or best[0] <= 1e-12: return ("leaf", float(y_values.mean()))
        _, col, threshold, left = best
        return ("node", col, threshold, self._build(x_values[left], y_values[left], depth + 1), self._build(x_values[~left], y_values[~left], depth + 1))
    def _predict_one(self, row, node):
        if node[0] == "leaf": return node[1]
        _, col, threshold, left_node, right_node = node
        return self._predict_one(row, left_node if row[col] <= threshold else right_node)
    def predict_proba(self, x_values):
        return np.array([self._predict_one(row, self.tree_) for row in x_values])

class BaggedTrees:
    def __init__(self, n=25, max_depth=4, min_leaf=300, seed=42):
        self.n = n; self.max_depth = max_depth; self.min_leaf = min_leaf; self.seed = seed
    def fit(self, x_values, y_values):
        rng = np.random.default_rng(self.seed); self.trees_ = []
        for tree_number in range(self.n):
            idx = rng.integers(0, len(y_values), len(y_values))
            self.trees_.append(ShallowDecisionTree(self.max_depth, self.min_leaf, feature_subsample=0.7, seed=self.seed + tree_number).fit(x_values[idx], y_values[idx]))
        return self
    def predict_proba(self, x_values):
        return np.mean([tree.predict_proba(x_values) for tree in self.trees_], axis=0)

class GaussianNB:
    def fit(self, x_values, y_values):
        self.prior_ = np.array([(y_values == 0).mean(), (y_values == 1).mean()])
        self.mean_ = np.vstack([x_values[y_values == 0].mean(axis=0), x_values[y_values == 1].mean(axis=0)])
        self.var_ = np.vstack([x_values[y_values == 0].var(axis=0) + 1e-6, x_values[y_values == 1].var(axis=0) + 1e-6])
        return self
    def predict_proba(self, x_values):
        logs = []
        for c in [0, 1]:
            logs.append(np.log(self.prior_[c] + 1e-12) - 0.5 * np.sum(np.log(2 * np.pi * self.var_[c]) + ((x_values - self.mean_[c]) ** 2) / self.var_[c], axis=1))
        logs = np.vstack(logs).T; logs -= logs.max(axis=1, keepdims=True)
        probs = np.exp(logs); probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, 1]

LR = LogisticRegressionGD
Tree = ShallowDecisionTree
NB = GaussianNB
