"""Leakage-safe split utilities."""
import numpy as np

SEED = 42

def stratified_train_test_split(y, test_size=0.20, seed=SEED):
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []
    for class_value in np.unique(y):
        class_indices = np.where(y == class_value)[0]
        rng.shuffle(class_indices)
        n_test = int(round(test_size * len(class_indices)))
        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])
    return np.array(sorted(train_indices)), np.array(sorted(test_indices))

def stratified_folds(y, n_splits=5, seed=SEED):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    for class_value in np.unique(y):
        class_indices = np.where(y == class_value)[0]
        rng.shuffle(class_indices)
        for position, index in enumerate(class_indices):
            folds[position % n_splits].append(int(index))
    return [np.array(sorted(fold)) for fold in folds]

split = stratified_train_test_split
folds = stratified_folds
