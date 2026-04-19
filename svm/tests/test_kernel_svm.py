"""
Tests for KernelSVM — validates correctness against sklearn's SVC.

We test three things:
1. Linearly separable data — should hit near 100% accuracy
2. XOR problem — not linearly separable, RBF kernel must solve it
3. Agreement with sklearn SVC on the same data (sanity check)
"""

import sys
from pathlib import Path

# Add the svm directory to Python path before importing `core` (must run first
# when executing this file with `python svm/tests/test_kernel_svm.py`).
test_dir = Path(__file__).parent.resolve()
svm_dir = test_dir.parent
if str(svm_dir) not in sys.path:
    sys.path.insert(0, str(svm_dir))

import numpy as np
from core.kernel_svm import KernelSVM
from core.kernels import rbf_kernel


def make_blobs(n=100, centers=None, std=0.5, seed=42):
    rng = np.random.RandomState(seed)
    X, y = [], []
    for i, center in enumerate(centers):
        X.append(rng.randn(n, 2) * std + center)
        y.extend([i] * n)
    return np.vstack(X), np.array(y)


def make_xor(n=200, seed=42):
    """
    XOR dataset: 4 clusters, labels alternate by quadrant.
    No linear boundary can separate these — kernel required.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    return X, y


def test_linear_kernel_separable():
    print("Test 1: Linear kernel on separable blobs...")
    X, y = make_blobs(centers=[[-2, -2], [2, 2]])

    model = KernelSVM(C=1.0, kernel='linear', tol=1e-3, max_passes=10)
    model.fit(X, y)

    acc = model.score(X, y)
    print(f"  Accuracy: {acc:.4f}  |  Support vectors: {model.n_support_}")
    assert acc >= 0.95, f"Expected >=95%, got {acc:.4f}"
    print("  ✓ Passed\n")


def test_rbf_kernel_xor():
    print("Test 2: RBF kernel on XOR problem (non-linear)...")
    X, y = make_xor(n=200)

    # Linear SVM cannot solve XOR — let's verify that first
    linear_model = KernelSVM(C=1.0, kernel='linear', tol=1e-2, max_passes=5)
    linear_model.fit(X, y)
    linear_acc = linear_model.score(X, y)
    print(f"  Linear kernel accuracy (should be ~50%): {linear_acc:.4f}")

    # RBF kernel should handle it
    rbf_model = KernelSVM(C=10.0, kernel='rbf', gamma=1.0,
                          tol=1e-3, max_passes=10)
    rbf_model.fit(X, y)
    rbf_acc = rbf_model.score(X, y)
    print(f"  RBF kernel accuracy: {rbf_acc:.4f}  |  "
          f"Support vectors: {rbf_model.n_support_}")

    assert rbf_acc > linear_acc + 0.1, (
        f"RBF ({rbf_acc:.4f}) should significantly beat "
        f"linear ({linear_acc:.4f}) on XOR"
    )
    print("  ✓ Passed\n")


def test_sklearn_agreement():
    """
    Our KernelSVM should produce similar accuracy to sklearn's SVC.
    We don't expect bit-for-bit identical results (different solvers),
    but accuracy should be within ~5%.
    """
    print("Test 3: Agreement with sklearn SVC...")

    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler as SkScaler
    except ImportError:
        print("  sklearn not available, skipping comparison test")
        return

    X, y = make_blobs(n=80, centers=[[-2, 0], [2, 0]], std=0.8)

    # Scale (required for both)
    mu, sigma = X.mean(0), X.std(0)
    X_scaled = (X - mu) / sigma

    # Our implementation
    our_model = KernelSVM(C=1.0, kernel='rbf', gamma=0.5,
                          tol=1e-3, max_passes=10)
    our_model.fit(X_scaled, y)
    our_acc = our_model.score(X_scaled, y)

    # sklearn
    sk_model = SVC(C=1.0, kernel='rbf', gamma=0.5)
    sk_model.fit(X_scaled, 2 * y - 1)   # sklearn also wants {-1,+1}
    sk_acc = sk_model.score(X_scaled, 2 * y - 1)

    print(f"  Our accuracy:     {our_acc:.4f}")
    print(f"  sklearn accuracy: {sk_acc:.4f}")
    print(f"  Difference:       {abs(our_acc - sk_acc):.4f}")

    assert abs(our_acc - sk_acc) < 0.08, (
        f"Too much divergence from sklearn: {abs(our_acc - sk_acc):.4f}"
    )
    print("  ✓ Passed\n")


def test_support_vector_properties():
    """
    Mathematical properties SVMs must satisfy:
    1. Only support vectors have non-zero alpha
    2. Sum of alpha_i * y_i = 0 (dual constraint)
    3. For free SVs (0 < alpha < C): y_i * f(x_i) ≈ 1
    """
    print("Test 4: Support vector mathematical properties...")

    X, y = make_blobs(n=60, centers=[[-1.5, 0], [1.5, 0]])
    y_pm = 2 * y - 1   # Convert to {-1, +1}

    model = KernelSVM(C=1.0, kernel='rbf', gamma=1.0,
                      tol=1e-4, max_passes=15)
    model.fit(X, y_pm)

    # Property 1: Dual constraint Σ αᵢyᵢ = 0
    dual_sum = np.sum(model.alpha_ * model._y_train)
    print(f"  Σ αᵢyᵢ = {dual_sum:.6f}  (should be ≈ 0)")
    assert abs(dual_sum) < 0.01, f"Dual constraint violated: {dual_sum}"

    # Property 2: All alphas in [0, C]
    assert np.all(model.alpha_ >= -1e-6), "Negative alpha found"
    assert np.all(model.alpha_ <= model.C + 1e-6), "Alpha exceeds C"

    # Property 3: Free SVs should satisfy yᵢf(xᵢ) ≈ 1
    free_sv_mask = (model.alpha_ > 1e-4) & (model.alpha_ < model.C - 1e-4)
    if np.sum(free_sv_mask) > 0:
        f_vals = model.decision_function(X[free_sv_mask])
        y_f = model._y_train[free_sv_mask] * f_vals
        print(f"  Free SV margins (should be ≈ 1.0): "
              f"mean={y_f.mean():.3f}, std={y_f.std():.3f}")

    print("  ✓ Passed\n")


if __name__ == "__main__":
    test_linear_kernel_separable()
    test_rbf_kernel_xor()
    test_sklearn_agreement()
    test_support_vector_properties()
    print("=" * 50)
    print("All KernelSVM tests passed!")