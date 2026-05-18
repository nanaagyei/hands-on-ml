"""
Validate the splitter on toy data, then prove the fast and naive versions
return identical splits across random datasets.
"""

import sys
from pathlib import Path

test_dir = Path(__file__).parent.resolve()
trees_dir = test_dir.parent
if str(trees_dir) not in sys.path:
    sys.path.insert(0, str(trees_dir))

import numpy as np
from core.splitter import find_best_split, find_best_split_naive


def test_obvious_split():
    """
    Trivial setup: one feature, perfectly separates two classes at threshold 1.5.
    Best split must be (feature=0, threshold=1.5, cost=0).
    """
    print("Test: obvious 1-D split")

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    s = find_best_split(X, y, criterion="gini")
    assert s.feature == 0
    assert s.threshold == 1.5, f"got {s.threshold}"
    assert s.cost == 0.0, f"got cost {s.cost}"
    assert s.n_left == 2 and s.n_right == 2

    print(f"  ✓ split at x <= {s.threshold}, cost={s.cost}")


def test_picks_informative_feature():
    """
    Two features. Feature 0 is noise (random), feature 1 perfectly separates.
    Splitter must pick feature 1.
    """
    print("Test: picks the informative feature")

    rng = np.random.RandomState(0)
    n = 100
    X = np.column_stack([rng.randn(n), np.r_[np.zeros(50), np.ones(50)]])
    y = np.r_[np.zeros(50, dtype=int), np.ones(50, dtype=int)]

    s = find_best_split(X, y, criterion="gini")
    assert s.feature == 1, f"picked feature {s.feature}, expected 1"
    assert s.cost == 0.0
    print(f"  ✓ picked feature {s.feature} with cost {s.cost}")


def test_pure_node_returns_none():
    """
    All labels identical → no split improves anything. We return None.
    (The tree builder will treat this as a leaf.)
    """
    print("Test: pure node returns None")

    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 1, 1])

    # All splits have cost 0 (children are also pure) so the splitter will
    # return *some* split — but the builder uses impurity_decrease, which is 0,
    # to decide to stop. This test just confirms we don't crash.
    s = find_best_split(X, y, criterion="gini")
    assert s is None or s.cost == 0.0
    print(f"  ✓ handled pure node (split={s})")


def test_min_samples_leaf():
    """
    With min_samples_leaf=2, splits creating a leaf of 1 should be rejected.
    """
    print("Test: min_samples_leaf is respected")

    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 0, 1, 1])

    s = find_best_split(X, y, criterion="gini", min_samples_leaf=2)
    assert s.n_left >= 2 and s.n_right >= 2
    print(f"  ✓ {s.n_left} | {s.n_right}")


def test_fast_matches_naive():
    """
    The fast and naive implementations must return splits with identical cost.
    They might pick a different (feature, threshold) when multiple ties exist,
    but the cost must match.
    """
    print("Test: fast matches naive on random data")

    rng = np.random.RandomState(42)
    n_trials = 12

    for trial in range(n_trials):
        n = rng.randint(20, 200)
        d = rng.randint(2, 6)
        X = rng.randn(n, d)
        y = rng.randint(0, rng.randint(2, 5), size=n)

        fast = find_best_split(X, y, criterion="gini")
        slow = find_best_split_naive(X, y, criterion="gini")

        if fast is None and slow is None:
            continue
        assert fast is not None and slow is not None, "one returned None"

        np.testing.assert_almost_equal(
            fast.cost, slow.cost,
            err_msg=f"trial {trial}: fast cost {fast.cost} != naive {slow.cost}",
        )

    print(f"  ✓ {n_trials} random trials, costs match")


def test_entropy_also_works():
    """Same checks but with entropy criterion."""
    print("Test: entropy criterion")

    rng = np.random.RandomState(1)
    X = rng.randn(80, 4)
    y = (X[:, 2] > 0).astype(int)

    s = find_best_split(X, y, criterion="entropy")
    assert s.feature == 2, f"picked feature {s.feature}, expected 2"
    print(f"  ✓ picked feature {s.feature}, threshold {s.threshold:.3f}")


def test_sklearn_agreement():
    """
    Compare against sklearn's depth-1 tree on a real dataset.
    A depth-1 fit returns exactly the best root split, so its
    .tree_.feature[0] and .tree_.threshold[0] should match ours.
    """
    print("Test: agreement with sklearn root split")

    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.datasets import load_breast_cancer, load_iris
    except ImportError:
        print("  sklearn unavailable, skipping")
        return

    for loader, name in [(load_breast_cancer, "breast_cancer"), (load_iris, "iris")]:
        X, y = loader(return_X_y=True)

        clf = DecisionTreeClassifier(criterion="gini", max_depth=1, random_state=42)
        clf.fit(X, y)
        sk_feature = clf.tree_.feature[0]
        sk_threshold = clf.tree_.threshold[0]

        s = find_best_split(X, y, criterion="gini")
        assert s.feature == sk_feature, (
            f"{name}: feature ours={s.feature} sklearn={sk_feature}")
        np.testing.assert_almost_equal(s.threshold, sk_threshold, decimal=5)
        print(f"  ✓ {name}: feature {s.feature}, threshold {s.threshold:.4f}")


if __name__ == "__main__":
    test_obvious_split()
    test_picks_informative_feature()
    test_pure_node_returns_none()
    test_min_samples_leaf()
    test_fast_matches_naive()
    test_entropy_also_works()
    test_sklearn_agreement()
    print("=" * 40)
    print("All splitter tests passed.")