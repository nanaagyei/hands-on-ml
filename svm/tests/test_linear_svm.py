import sys
from pathlib import Path

# Add the svm directory to Python path before importing `core` (must run first
# when executing this file with `python svm/tests/test_linear_svm.py`).
test_dir = Path(__file__).parent.resolve()
svm_dir = test_dir.parent
if str(svm_dir) not in sys.path:
    sys.path.insert(0, str(svm_dir))

import numpy as np

from core.linear_svm import LinearSVM
from core.kernels import rbf_kernel, linear_kernel, get_kernel

def test_linear_svm_separable():
    """
    Two clearly separated blobs. LinearSVM should nail this.
    """
    rng = np.random.RandomState(42)
    
    # Class +1: centered at (2, 2)
    X_pos = rng.randn(50, 2) + np.array([2, 2])
    # Class -1: centered at (-2, -2)
    X_neg = rng.randn(50, 2) + np.array([-2, -2])
    
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 50 + [-1] * 50)
    
    model = LinearSVM(C=1.0, n_epochs=500)
    model.fit(X, y)
    
    acc = model.score(X, y)
    print(f"Training accuracy on separable data: {acc:.4f}")
    print(f"Number of support vectors: {model.n_support_vectors_}")
    assert acc > 0.95, f"Expected >95% accuracy, got {acc:.4f}"
    print("✓ LinearSVM test passed\n")


def test_kernel_shapes():
    """Verify kernel matrices have correct shapes."""
    X1 = np.random.randn(10, 5)
    X2 = np.random.randn(7, 5)
    
    K_linear = linear_kernel(X1, X2)
    K_rbf = rbf_kernel(X1, X2, gamma=0.5)
    
    assert K_linear.shape == (10, 7), f"Linear kernel shape wrong: {K_linear.shape}"
    assert K_rbf.shape == (10, 7), f"RBF kernel shape wrong: {K_rbf.shape}"
    
    # RBF of a point with itself should be 1.0
    K_self = rbf_kernel(X1, X1, gamma=0.5)
    assert np.allclose(np.diag(K_self), 1.0), "RBF self-similarity should be 1.0"
    
    print("✓ Kernel shape tests passed\n")


if __name__ == "__main__":
    test_linear_svm_separable()
    test_kernel_shapes()
    print("All tests passed!")