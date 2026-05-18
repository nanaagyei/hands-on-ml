"""
End-to-end tests for the from-scratch decision tree classifier.

Strategy:
  1. Trivial datasets we can reason about by hand.
  2. Real datasets (iris, breast cancer) with parity against sklearn.
  3. Pre-pruning hyperparameters actually constrain the tree.
"""

import sys
from pathlib import Path

test_dir = Path(__file__).parent.resolve()
trees_dir = test_dir.parent
if str(trees_dir) not in sys.path:
    sys.path.insert(0, str(trees_dir))

import numpy as np
from core.tree_classifier import DecisionTreeClassifier


def test_perfect_separation():
    """A perfectly separable 1-D dataset should yield 100% train accuracy."""
    print("Test: perfect 1-D separation")

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier().fit(X, y)
    assert clf.score(X, y) == 1.0
    assert clf.tree_depth_ == 1, f"depth {clf.tree_depth_}"
    assert clf.n_leaves_ == 2
    print(f"  ✓ depth={clf.tree_depth_}, leaves={clf.n_leaves_}, acc=1.0")


def test_xor_needs_depth_2():
    """XOR: linear can't fit it, but depth-2 tree easily can."""
    print("Test: XOR requires depth 2")

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])

    clf_d1 = DecisionTreeClassifier(max_depth=1).fit(X, y)
    clf_d2 = DecisionTreeClassifier(max_depth=2).fit(X, y)

    assert clf_d1.score(X, y) < 1.0, "depth 1 shouldn't solve XOR"
    assert clf_d2.score(X, y) == 1.0, "depth 2 must solve XOR"
    print(f"  ✓ d1 acc={clf_d1.score(X, y):.2f}  d2 acc={clf_d2.score(X, y):.2f}")


def test_max_depth_caps_tree():
    """max_depth must be a hard cap on tree growth."""
    print("Test: max_depth is enforced")

    rng = np.random.RandomState(0)
    X = rng.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    for d in [1, 2, 3, 5]:
        clf = DecisionTreeClassifier(max_depth=d).fit(X, y)
        assert clf.tree_depth_ <= d, f"max_depth={d} but tree is {clf.tree_depth_}"
    print("  ✓ max_depth ∈ {1,2,3,5} all respected")


def test_min_samples_leaf():
    """Every leaf must have at least min_samples_leaf samples."""
    print("Test: min_samples_leaf is enforced")

    rng = np.random.RandomState(0)
    X = rng.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)

    clf = DecisionTreeClassifier(min_samples_leaf=10).fit(X, y)

    # Walk the tree, check every leaf
    def check_leaves(node):
        if node.is_leaf:
            assert node.n_samples >= 10, f"leaf with {node.n_samples} samples"
            return
        check_leaves(node.left)
        check_leaves(node.right)

    check_leaves(clf.root_)
    print(f"  ✓ all leaves >= 10 samples")


def test_predict_proba_sums_to_one():
    """Class probabilities at a leaf must sum to 1."""
    print("Test: predict_proba rows sum to 1")

    rng = np.random.RandomState(1)
    X = rng.randn(100, 4)
    y = rng.randint(0, 3, 100)

    clf = DecisionTreeClassifier(max_depth=4).fit(X, y)
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    print(f"  ✓ shape {proba.shape}, rows sum to 1")


def test_string_labels():
    """Should handle non-integer labels (strings, etc.)."""
    print("Test: string labels round-trip correctly")

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array(["cat", "cat", "dog", "dog"])

    clf = DecisionTreeClassifier().fit(X, y)
    preds = clf.predict(X)
    assert set(preds.tolist()) == {"cat", "dog"}
    assert (preds == y).all()
    print(f"  ✓ predictions: {preds.tolist()}")


def test_sklearn_parity_iris():
    """
    On iris, our tree (default hyperparameters) should match sklearn's
    train accuracy. Tree structure may differ on tie-breaks; accuracy
    must not.
    """
    print("Test: parity with sklearn on iris")

    try:
        from sklearn.tree import DecisionTreeClassifier as SkTree
        from sklearn.datasets import load_iris
    except ImportError:
        print("  sklearn unavailable, skipping")
        return

    X, y = load_iris(return_X_y=True)

    ours = DecisionTreeClassifier(criterion="gini", random_state=42).fit(X, y)
    theirs = SkTree(criterion="gini", random_state=42).fit(X, y)

    our_acc = ours.score(X, y)
    sk_acc = theirs.score(X, y)
    print(f"  ours: train acc={our_acc:.4f}, depth={ours.tree_depth_}, "
          f"leaves={ours.n_leaves_}")
    print(f"  sklearn: train acc={sk_acc:.4f}, depth={theirs.get_depth()}, "
          f"leaves={theirs.get_n_leaves()}")

    assert our_acc == sk_acc, f"acc diff: {our_acc} vs {sk_acc}"
    print(f"  ✓ train accuracy matches")


def test_sklearn_parity_breast_cancer():
    """Same parity check on a larger, more realistic dataset."""
    print("Test: parity with sklearn on breast cancer")

    try:
        from sklearn.tree import DecisionTreeClassifier as SkTree
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("  sklearn unavailable, skipping")
        return

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    for d in [3, 5, None]:
        ours = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_tr, y_tr)
        theirs = SkTree(max_depth=d, criterion="gini", random_state=42).fit(X_tr, y_tr)

        our_acc = ours.score(X_te, y_te)
        sk_acc = theirs.score(X_te, y_te)
        gap = abs(our_acc - sk_acc)
        print(f"  depth={d}: ours={our_acc:.4f}  sklearn={sk_acc:.4f}  "
              f"|diff|={gap:.4f}")

        # On a fully-grown tree (d=None), ties can lead to slightly different
        # structures and small accuracy diffs. Tolerate 3 percentage points.
        assert gap < 0.03, f"depth={d}: gap too large ({gap:.4f})"

    print(f"  ✓ test accuracy within 3 percentage points across depths")


if __name__ == "__main__":
    test_perfect_separation()
    test_xor_needs_depth_2()
    test_max_depth_caps_tree()
    test_min_samples_leaf()
    test_predict_proba_sums_to_one()
    test_string_labels()
    test_sklearn_parity_iris()
    test_sklearn_parity_breast_cancer()
    print("=" * 40)
    print("All tree classifier tests passed.")