"""
Node type for binary decision trees.

A node is either:
  - Internal: has feature, threshold, left, right; is_leaf is False.
  - Leaf:     has prediction (class probabilities); left and right are None.

We use one class with optional fields rather than a sealed Internal/Leaf
hierarchy because it matches sklearn's tree representation and keeps the
recursive build code straightforward.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Node:
    # Filled for both leaves and internal nodes
    n_samples: int = 0
    impurity: float = 0.0

    # Filled only for leaves: shape (n_classes,), sums to 1
    value: Optional[np.ndarray] = None

    # Filled only for internal nodes
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None