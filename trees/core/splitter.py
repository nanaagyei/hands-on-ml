"""
Finding the best split at a tree node.

Two implementations:
  - find_best_split_naive: clear, obviously correct, O(d · n²). Reference impl.
  - find_best_split:       sorted sweep with incremental Gini, O(d · n log n).
                           Used in production. Validated against the naive one.

Both return a Split dataclass with (feature, threshold, cost, n_left, n_right),
or None when no valid split exists (pure node, or every candidate would create
an empty child / violate min_samples_leaf).
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional
from core.impurity import gini, entropy, mse

@dataclass
class Split:
    """
    A split of a node into two children.
    """
    feature: int
    threshold: float
    cost: float
    n_left: int
    n_right: int


_IMPURITY_FUNCS = {
    "gini": gini,
    "entropy": entropy,
    "mse": mse
}


def _weighted_child_cost(y_left, y_right, impurity_fn):
    """
    Calculate the weighted cost of two children

    Args:
        y_left (array-like): The labels of the left child.
        y_right (array-like): The labels of the right child.
        impurity_fn (callable): The impurity function to use.

    Returns:
        float: The weighted cost of the two children.
    """
    n_left, n_right = len(y_left), len(y_right)
    n_total = n_left + n_right
    if n_total == 0:
        return 0.0
    return (n_left * impurity_fn(y_left) + n_right * impurity_fn(y_right)) / n_total


def find_best_split_naive(X, y, criterion="gini", min_samples_leaf=1):
    """
    Find the best split for a node using a naive approach.
    
    This implementation is clear, obviously correct, but O(d · n²).
    For every feature, try every midpoint between
    consecutive sorted unique values as a candidate threshold; pick the lowest-cost
    valid split.

    Why midpoints and not the unique values themselves?
    Because the split rule is `x <= t`. If t equals an actual feature value, you
    have an ambiguity about which side that value belongs on. Using midpoints
    between consecutive unique values avoids that and matches sklearn's convention.
    Args:
        X (array-like): The features of the data.
        y (array-like): The labels of the data.
        criterion (str): The criterion to use for the split.
        min_samples_leaf (int): The minimum number of samples in a leaf.

    Returns:
        Split: The best split for the node.
    """
    if criterion not in _IMPURITY_FUNCS:
        raise ValueError(f"Invalid criterion: {criterion}. Valid criteria are: {list(_IMPURITY_FUNCS.keys())}")
    impurity_fn = _IMPURITY_FUNCS[criterion]

    n_samples, n_features = X.shape
    best = None

    for k in range(n_features):
        column = X[:, k]
        unique_vals = np.unique(column)
        if len(unique_vals) < 2:
            continue # can't split on a single value

        # Try every midpoint between consecutive unique values as a candidate threshold
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        for t in thresholds:
            left_mask = column <= t
            n_left = int(left_mask.sum())
            n_right = n_samples - n_left

            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue

            cost = _weighted_child_cost(y[left_mask], y[~left_mask], impurity_fn)

            if best is None or cost < best.cost:
                best = Split(feature=k, threshold=float(t), cost=float(cost),
                             n_left=n_left, n_right=n_right)

    return best


def find_best_split(X, y, criterion="gini", min_samples_leaf=1):
    """
    Fast splitter using sorted sweep with incremental Gini counts.

    For binary or multiclass classification (K small), this is O(d · n log n)
    per call — the sort is the bottleneck. For the fraud dataset (K=2, n~228k,
    d=34) this is ~7M operations per call instead of ~10^12.

    Validated against find_best_split_naive in trees/tests/test_splitter.py.

    NOTE: currently classification only (gini, entropy). MSE/regression uses the
    same idea but with running sum and sum-of-squares instead of class counts;
    we'll add it when we build the regressor.
    """
    if criterion == "mse":
        # The sweep version for regression needs running moments, not counts.
        # Until we implement that, fall back to the naive version. It still
        # works, just slowly.
        return find_best_split_naive(X, y, criterion, min_samples_leaf)

    if criterion not in ("gini", "entropy"):
        raise ValueError(f"unknown criterion: {criterion}")

    n_samples, n_features = X.shape

    # Encode labels to dense 0..K-1 indices for array-based counting
    classes, y_encoded = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    total_counts = np.bincount(y_encoded, minlength=n_classes)

    best = None

    for k in range(n_features):
        column = X[:, k]

        # Sort this feature once. order maps sorted position -> original index.
        order = np.argsort(column, kind="stable")
        col_sorted = column[order]
        y_sorted = y_encoded[order]

        # Initially everything is on the right
        left_counts = np.zeros(n_classes, dtype=np.int64)
        right_counts = total_counts.copy()

        # Sweep: after step i, sample at sorted position i has moved to the left
        for i in range(n_samples - 1):
            c = y_sorted[i]
            left_counts[c] += 1
            right_counts[c] -= 1

            n_left = i + 1
            n_right = n_samples - n_left

            # Skip cuts that violate min_samples_leaf — and skip ties: if the next
            # feature value equals this one, there's no threshold between them.
            if n_left < min_samples_leaf or n_right < min_samples_leaf:
                continue
            if col_sorted[i] == col_sorted[i + 1]:
                continue

            if criterion == "gini":
                p_left = left_counts / n_left
                p_right = right_counts / n_right
                imp_left = 1.0 - np.sum(p_left ** 2)
                imp_right = 1.0 - np.sum(p_right ** 2)
            else:  # entropy
                p_left = left_counts / n_left
                p_right = right_counts / n_right
                # 0·log(0) := 0  →  filter zeros before log
                pl = p_left[p_left > 0]
                pr = p_right[p_right > 0]
                imp_left = -np.sum(pl * np.log2(pl))
                imp_right = -np.sum(pr * np.log2(pr))

            cost = (n_left / n_samples) * imp_left + (n_right / n_samples) * imp_right
            threshold = (col_sorted[i] + col_sorted[i + 1]) / 2.0

            if best is None or cost < best.cost:
                best = Split(feature=k, threshold=float(threshold),
                             cost=float(cost),
                             n_left=int(n_left), n_right=int(n_right))

    return best