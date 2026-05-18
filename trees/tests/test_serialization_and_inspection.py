"""
Verify the tree pickles cleanly and the inspection utilities work.
"""

import sys
import pickle
import io
from pathlib import Path

test_dir = Path(__file__).parent.resolve()
trees_dir = test_dir.parent
if str(trees_dir) not in sys.path:
    sys.path.insert(0, str(trees_dir))

import numpy as np
from core.tree_classifier import DecisionTreeClassifier


def test_pickle_round_trip():
    """Trained tree → pickle → unpickle → produces same predictions."""
    print("Test: pickle round-trip")

    try:
        from sklearn.datasets import load_breast_cancer
    except ImportError:
        print("  sklearn unavailable, skipping")
        return

    X, y = load_breast_cancer(return_X_y=True)
    clf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y)

    preds_before = clf.predict(X)
    importances_before = clf.feature_importances_.copy()

    buf = io.BytesIO()
    pickle.dump(clf, buf)
    buf.seek(0)
    clf_restored = pickle.load(buf)

    preds_after = clf_restored.predict(X)
    importances_after = clf_restored.feature_importances_

    assert (preds_before == preds_after).all(), "predictions changed after pickle"
    np.testing.assert_array_equal(importances_before, importances_after)
    print(f"  ✓ predictions and importances preserved")


def test_feature_importance_sums_to_one():
    """Importances must sum to 1 after normalization."""
    print("Test: feature importances normalize to 1")

    rng = np.random.RandomState(0)
    X = rng.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    clf = DecisionTreeClassifier(max_depth=5).fit(X, y)
    np.testing.assert_almost_equal(clf.feature_importances_.sum(), 1.0)

    # Features 0 and 1 are the real signal; importance should concentrate there.
    informative_share = clf.feature_importances_[[0, 1]].sum()
    assert informative_share > 0.85, (
        f"informative features got only {informative_share:.3f} of importance"
    )
    print(f"  ✓ sums to 1, features 0+1 hold "
          f"{100 * informative_share:.1f}% of importance")


def test_feature_importance_matches_sklearn():
    """
    On a clean dataset, our importance ranking should agree with sklearn's.
    Exact values may differ slightly due to tie-breaking; ranking shouldn't.
    """
    print("Test: importance ranking agrees with sklearn")

    try:
        from sklearn.tree import DecisionTreeClassifier as SkTree
        from sklearn.datasets import load_iris
    except ImportError:
        print("  sklearn unavailable, skipping")
        return

    X, y = load_iris(return_X_y=True)

    ours = DecisionTreeClassifier(random_state=42).fit(X, y)
    theirs = SkTree(random_state=42).fit(X, y)

    our_top = int(np.argmax(ours.feature_importances_))
    sk_top = int(np.argmax(theirs.feature_importances_))
    print(f"  our top feature: {our_top}  sklearn top: {sk_top}")
    assert our_top == sk_top, "top feature disagrees with sklearn"

    # Pearson correlation of full importance vectors should be very high
    corr = np.corrcoef(ours.feature_importances_,
                       theirs.feature_importances_)[0, 1]
    print(f"  importance vector correlation: {corr:.4f}")
    assert corr > 0.95, f"importances correlate only {corr:.3f}"
    print(f"  ✓ rankings agree, correlation {corr:.4f}")


def test_tree_to_text():
    """tree_to_text returns a non-empty string with the right structure."""
    print("Test: tree_to_text produces readable output")

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier().fit(X, y)
    text = clf.tree_to_text(feature_names=["x"], class_names=["A", "B"])

    assert "x <=" in text, "should mention the split feature"
    assert "leaf" in text, "should show leaf nodes"
    assert "predict=" in text, "should show predictions"
    print("  ✓ output:")
    for line in text.split("\n"):
        print(f"    {line}")


if __name__ == "__main__":
    test_pickle_round_trip()
    test_feature_importance_sums_to_one()
    test_feature_importance_matches_sklearn()
    test_tree_to_text()
    print("=" * 40)
    print("All serialization and inspection tests passed.")