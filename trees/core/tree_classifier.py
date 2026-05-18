"""
From-scratch decision tree classifier.

Mirrors sklearn's DecisionTreeClassifier API at a basic level: .fit(X, y),
.predict(X), .predict_proba(X). Hyperparameters cover pre-pruning only
(no post-pruning yet).

Implementation: recursive greedy partitioning. At each node, call the splitter
from splitter.py; if a useful split exists, recurse on left and right subsets;
otherwise return a leaf.
"""

import numpy as np

from .impurity import gini, entropy
from .splitter import find_best_split
from .node import Node


_IMPURITY_FUNCS = {"gini": gini, "entropy": entropy}


class DecisionTreeClassifier:
    """
    Parameters
    ----------
    criterion : {"gini", "entropy"}, default "gini"
        Impurity function for split scoring.

    max_depth : int or None, default None
        Maximum depth of the tree. None means no limit (grow until other
        stopping conditions kick in). The root is depth 0.

    min_samples_split : int, default 2
        A node with fewer samples than this becomes a leaf without trying
        to split. Statistical reliability: splitting on 3 samples is noise.

    min_samples_leaf : int, default 1
        Minimum samples per leaf. Splits creating a smaller leaf are
        rejected by the splitter.

    min_impurity_decrease : float, default 0.0
        A split is taken only if it reduces parent impurity by at least
        this much (weighted by the fraction of samples in the node).
        This is the gatekeeper between "best available split" (what the
        splitter returns) and "split worth taking" (what we actually do).

    random_state : int, default 42
        Forwarded to the splitter for tie-breaking determinism.
    """

    def __init__(self, criterion="gini", max_depth=None,
                 min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, random_state=42):
        if criterion not in _IMPURITY_FUNCS:
            raise ValueError(f"unknown criterion: {criterion}")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state

        # Set after fitting
        self.root_: Node = None
        self.classes_: np.ndarray = None         # sorted unique labels
        self.n_classes_: int = None
        self.n_features_: int = None
        self.tree_depth_: int = 0                # max depth actually reached
        self.n_leaves_: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.tree_depth_ = 0
        self.n_leaves_ = 0

        # Feature importance accumulator, normalized at the end of fit.
        # We use n_total as the denominator inside _build to weight each split
        # by the fraction of all training samples flowing through that node.
        self._n_total = len(y_encoded)
        self.feature_importances_ = np.zeros(self.n_features_, dtype=np.float64)

        self.root_ = self._build(X, y_encoded, depth=0)

        # Normalize so importances sum to 1 (matches sklearn convention).
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

        return self

    def predict(self, X):
        """Returns class labels (original dtype, not encoded indices)."""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        """Returns (n_samples, n_classes) probabilities."""
        X = np.asarray(X, dtype=np.float64)
        proba = np.empty((len(X), self.n_classes_), dtype=np.float64)
        for i, x in enumerate(X):
            proba[i] = self._traverse(self.root_, x)
        return proba

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def get_params(self):
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "random_state": self.random_state,
        }

    # ------------------------------------------------------------------
    # Recursive builder
    # ------------------------------------------------------------------

    def _build(self, X, y, depth):
        n_samples = len(y)
        impurity_fn = _IMPURITY_FUNCS[self.criterion]
        impurity = impurity_fn(y)

        # Compute leaf value lazily — only used if we end up as a leaf
        def make_leaf():
            value = np.bincount(y, minlength=self.n_classes_) / n_samples
            self.n_leaves_ += 1
            self.tree_depth_ = max(self.tree_depth_, depth)
            return Node(n_samples=n_samples, impurity=impurity, value=value)

        # Pre-pruning checks, ordered cheapest first
        if impurity == 0.0:
            return make_leaf()                                  # pure node
        if self.max_depth is not None and depth >= self.max_depth:
            return make_leaf()
        if n_samples < self.min_samples_split:
            return make_leaf()

        # Find best split — splitter already enforces min_samples_leaf
        split = find_best_split(
            X, y,
            criterion=self.criterion,
            min_samples_leaf=self.min_samples_leaf,
        )
        if split is None:
            return make_leaf()                                  # no valid split

        # Is the split actually an improvement?
        # We weight by the fraction of samples at this node (relative to root)
        # only if min_impurity_decrease is interpreted globally. Sklearn does
        # this weighting using n_node_samples / n_total — we'd need n_total
        # from the root. For now we use the simpler local definition:
        # ΔI = parent_impurity - cost_of_split.
        impurity_decrease = impurity - split.cost
        if impurity_decrease < self.min_impurity_decrease:
            return make_leaf()

        # Take the split. Partition the data and recurse.
        left_mask = X[:, split.feature] <= split.threshold
        right_mask = ~left_mask

        left_child = self._build(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build(X[right_mask], y[right_mask], depth + 1)

        # Record this split's contribution to feature importance.
        # Weighted by fraction of training samples at this node, multiplied
        # by the impurity decrease the split achieved.
        self.feature_importances_[split.feature] += (
            n_samples / self._n_total
        ) * impurity_decrease

        return Node(
            n_samples=n_samples,
            impurity=impurity,
            feature=split.feature,
            threshold=split.threshold,
            left=left_child,
            right=right_child,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _traverse(self, node, x):
        """Walk from node to leaf, return leaf's class probabilities."""
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def tree_to_text(self, feature_names=None, class_names=None, max_depth=None):
        """
        Render the tree as indented if-else rules. Useful for inspection and
        for explaining what the model learned in a write-up.

        Parameters
        ----------
        feature_names : list of str, optional
            Names for X columns. Defaults to "f0", "f1", ...
        class_names : list of str, optional
            Names for the original class labels. Defaults to str(self.classes_[k]).
        max_depth : int, optional
            Cap the printing depth. None means print the full tree.

        Returns
        -------
        str
        """
        if self.root_ is None:
            return "(unfit tree)"

        if feature_names is None:
            feature_names = [f"f{k}" for k in range(self.n_features_)]
        if class_names is None:
            class_names = [str(c) for c in self.classes_]

        lines = []

        def render(node, depth, prefix):
            indent = "  " * depth
            if node.is_leaf or (max_depth is not None and depth >= max_depth):
                # Leaf summary: class probabilities, n samples
                probs = node.value
                top_idx = int(np.argmax(probs))
                top_name = class_names[top_idx]
                prob_str = ", ".join(
                    f"{class_names[k]}={probs[k]:.2f}"
                    for k in range(len(probs))
                )
                lines.append(
                    f"{indent}{prefix}leaf: predict={top_name}  "
                    f"[{prob_str}]  n={node.n_samples}"
                )
                return

            name = feature_names[node.feature]
            lines.append(
                f"{indent}{prefix}if {name} <= {node.threshold:.4f}  "
                f"(n={node.n_samples}, impurity={node.impurity:.4f})"
            )
            render(node.left, depth + 1, "├─ ")
            render(node.right, depth + 1, "└─ ")

        render(self.root_, 0, "")
        return "\n".join(lines)