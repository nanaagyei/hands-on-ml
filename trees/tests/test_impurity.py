"""
Validate impurity functions against hand-computed values and sklearn.
"""

import sys
from pathlib import Path

test_dir = Path(__file__).parent.resolve()
trees_dir = test_dir.parent
if str(trees_dir) not in sys.path:
    sys.path.insert(0, str(trees_dir))

import numpy as np
from core.impurity import gini, entropy, mse


def test_gini():
    print("Test: gini()")

    assert gini(np.array([1, 1, 1, 1])) == 0.0, "pure node must be 0"
    assert gini(np.array([])) == 0.0, "empty node must be 0"

    # 50/50 binary: 1 - (0.5² + 0.5²) = 0.5
    assert gini(np.array([0, 0, 1, 1])) == 0.5

    # 3 classes balanced: 1 - 3·(1/3)² = 2/3
    np.testing.assert_almost_equal(gini(np.array([0, 1, 2])), 2/3)

    # Mostly one class: 1 - (0.9² + 0.1²) = 0.18
    y = np.array([0]*9 + [1])
    np.testing.assert_almost_equal(gini(y), 0.18)

    print("  ✓ passed")


def test_entropy():
    print("Test: entropy()")

    assert entropy(np.array([5, 5, 5])) == 0.0, "pure node must be 0"
    assert entropy(np.array([])) == 0.0, "empty node must be 0"

    # 50/50 binary: exactly 1 bit
    assert entropy(np.array([0, 0, 1, 1])) == 1.0

    # Uniform K classes: log₂(K)
    np.testing.assert_almost_equal(entropy(np.array([0, 1, 2])), np.log2(3))
    np.testing.assert_almost_equal(entropy(np.array([0, 1, 2, 3])), 2.0)

    # No NaN from a class with 0 samples
    y = np.array([0]*9 + [1])
    h = entropy(y)
    assert np.isfinite(h) and h > 0, f"got {h}"

    print("  ✓ passed")


def test_mse():
    print("Test: mse()")

    assert mse(np.array([7, 7, 7])) == 0.0
    assert mse(np.array([])) == 0.0

    # Variance of [1,2,3,4,5] = 2.0
    np.testing.assert_almost_equal(mse(np.array([1, 2, 3, 4, 5])), 2.0)

    print("  ✓ passed")


def test_sklearn_agreement():
    """
    Sanity check against sklearn: train a depth-1 tree, compare its
    reported root impurity to ours.
    """
    print("Test: agreement with sklearn root impurity")

    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import load_breast_cancer
    except ImportError:
        print("  sklearn unavailable, skipping")
        return

    X, y = load_breast_cancer(return_X_y=True)

    for criterion, our_fn in [("gini", gini), ("entropy", entropy)]:
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=1,
                                     random_state=42)
        clf.fit(X, y)
        sk_root = clf.tree_.impurity[0]
        ours = our_fn(y)
        diff = abs(sk_root - ours)
        print(f"  {criterion}: ours={ours:.6f}  sklearn={sk_root:.6f}  "
              f"diff={diff:.2e}")
        assert diff < 1e-10, f"{criterion}: divergence {diff}"

    print("  ✓ passed")


if __name__ == "__main__":
    test_gini()
    test_entropy()
    test_mse()
    test_sklearn_agreement()
    print("=" * 40)
    print("All impurity tests passed.")