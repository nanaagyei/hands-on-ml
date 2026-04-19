import numpy as np

def linear_kernel(X1, X2):
    """
    K(x, z) = x · z
    
    The simplest kernel — just a dot product in the original space.
    Equivalent to a linear decision boundary.
    Use when: data is linearly separable, very high-dimensional (text),
              or you have millions of samples (faster than RBF).
    
    X1: (n_samples_1, n_features)
    X2: (n_samples_2, n_features)
    Returns: (n_samples_1, n_samples_2) kernel matrix
    """
    return X1 @ X2.T

def polynomial_kernel(X1, X2, degree=3, gamma=1.0, coef0=1.0):
    """
    K(x, z) = (gamma * x·z + coef0)^degree
    
    Implicitly maps to a feature space of all monomials up to degree d.
    For degree=2: captures x1², x2², x1*x2, x1, x2, 1
    Use when: you know the relationship is polynomial (image pixels, etc.)
    
    Pitfall: high degree → numerical instability (values get huge or tiny)
    coef0 (r): controls influence of lower-degree terms
    """
    return (gamma * (X1 @ X2.T) + coef0) ** degree

def rbf_kernel(X1, X2, gamma=1.0):
    """
    K(x, z) = exp(-gamma * ||x - z||²)
    
    The workhorse. Infinite-dimensional feature space.
    
    Intuition: similarity function. Two identical points → K=1.
    Points infinitely far apart → K=0.
    gamma controls the "reach" of each training point:
        - High gamma: each point only influences its immediate neighbors
                      → complex, wiggly boundary → risk overfitting
        - Low gamma:  each point influences far neighbors
                      → smoother boundary → risk underfitting
    
    Implementation trick: ||x - z||² = ||x||² + ||z||² - 2x·z
    This is much faster than computing pairwise distances naively.
    
    X1: (n_samples_1, n_features)
    X2: (n_samples_2, n_features)
    Returns: (n_samples_1, n_samples_2) kernel matrix
    """
    # ||x||² for each row in X1: shape (n1,)
    X1_sq = np.sum(X1 ** 2, axis=1)
    # ||z||² for each row in X2: shape (n2,)
    X2_sq = np.sum(X2 ** 2, axis=1)
    
    # Squared distances via broadcasting: shape (n1, n2)
    # ||x - z||² = ||x||² + ||z||² - 2(x·z)
    sq_dists = X1_sq[:, np.newaxis] + X2_sq[np.newaxis, :] - 2 * (X1 @ X2.T)
    
    # Numerical safety: clamp negatives caused by floating point
    sq_dists = np.maximum(sq_dists, 0)
    
    return np.exp(-gamma * sq_dists)

def sigmoid_kernel(X1, X2, gamma=1.0, coef0=0.0):
    """
    K(x, z) = tanh(gamma * x·z + coef0)
    
    Borrowed from neural networks. Not always positive semi-definite,
    so technically not always a valid kernel. Use cautiously.
    Works well in practice for certain NLP tasks.
    """
    return np.tanh(gamma * (X1 @ X2.T) + coef0)

# Hyperparameters each kernel accepts (linear ignores gamma/degree/coef0).
_KERNEL_PARAM_KEYS = {
    'linear': frozenset(),
    'polynomial': frozenset({'degree', 'gamma', 'coef0'}),
    'rbf': frozenset({'gamma'}),
    'sigmoid': frozenset({'gamma', 'coef0'}),
}


def get_kernel(kernel_name, **kwargs):
    """
    Factory function. Maps string name to kernel function with params baked in.
    
    Usage:
        K = get_kernel('rbf', gamma=0.1)
        K_matrix = K(X_train, X_train)
    """

    kernels = {
        'linear': linear_kernel,
        'polynomial': polynomial_kernel,
        'rbf': rbf_kernel,
        'sigmoid': sigmoid_kernel,
    }

    if kernel_name not in kernels:
        raise ValueError(f"Unknown kernel: '{kernel_name}'. "
                         f"Available kernels: {list(kernels.keys())}")
    
    kernel_fn = kernels[kernel_name]
    allowed = _KERNEL_PARAM_KEYS[kernel_name]
    kernel_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    def kernel(X1, X2):
        return kernel_fn(X1, X2, **kernel_kwargs)
    
    kernel.__name__ = kernel_name
    return kernel