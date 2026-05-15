"""
Impurity functions for decision trees splits.

A node's impurity measures how "mixed" its labels are.
A split is good when the weighted impurity of the children is lower
than the parent's impurity. That difference is the information gain.

Three functions:
  - gini:    classification, default in CART/sklearn. Cheap to compute.
  - entropy: classification, used by ID3/C4.5. Slightly more expensive (log).
  - mse:     regression. Just variance of the target.

All three take a 1-D array of labels/targets and return a scalar >= 0.
Pure node → 0.
"""

import numpy as np

def gini(y):
    """
    Gini impurity: G = 1 - Σ p_k²

    Interpretation: probability that two random samples from this node
    have different labels. Pure node → 0. Worst case for K classes → 1 - 1/K.
    
    Args:
        y (array-like): The labels to calculate the Gini impurity for.

    Returns:
        float: The Gini impurity of the labels.
    """

    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(np.square(probs))


def entropy(y):
    """
    Shannon entropy: H = -Σ p_k log₂(p_k)

    Interpretation: bits needed to encode the label of a random sample
    from this node. Pure node → 0. Uniform K classes → log₂(K).
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    probs = probs[probs > 0]                    # 0·log(0) := 0
    return -np.sum(probs * np.log2(probs))


def mse(y):
    """
    Regression impurity: variance of targets in the node.

    For squared-error regression trees, the constant prediction that
    minimizes loss in a node is the mean, and the resulting loss is
    exactly the variance — that's why MSE is the natural impurity.
    """
    if len(y) == 0:
        return 0.0
    return float(np.var(y))


    