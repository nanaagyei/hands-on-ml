# test_scalers.py
"""Validate our scalers against sklearn."""

import numpy as np
from sklearn.preprocessing import StandardScaler as SklearnStandard
from sklearn.preprocessing import MinMaxScaler as SklearnMinMax
from preprocessing.scalers import StandardScaler, MinMaxScaler


def test_standard_scaler():
    """Test StandardScaler against sklearn."""
    np.random.seed(42)
    
    # Create data with different scales
    X = np.column_stack([
        np.random.randn(100) * 1000 + 5000,  # Large scale
        np.random.randn(100) * 0.1 + 0.5,    # Small scale
        np.random.randn(100) * 50 - 100      # Medium scale, negative
    ])
    
    # Our implementation
    our_scaler = StandardScaler()
    X_ours = our_scaler.fit_transform(X)
    
    # Sklearn
    sk_scaler = SklearnStandard()
    X_sklearn = sk_scaler.fit_transform(X)
    
    # Compare
    print("=== StandardScaler Test ===")
    print(f"Max difference in scaled data: {np.max(np.abs(X_ours - X_sklearn)):.2e}")
    print(f"Mean comparison - Ours: {our_scaler.mean_}, Sklearn: {sk_scaler.mean_}")
    print(f"Std comparison - Ours: {our_scaler.std_}, Sklearn: {sk_scaler.scale_}")
    
    # Test inverse transform
    X_recovered = our_scaler.inverse_transform(X_ours)
    print(f"Max recovery error: {np.max(np.abs(X_recovered - X)):.2e}")
    
    # Verify properties of scaled data
    print(f"\nScaled data properties:")
    print(f"  Means: {X_ours.mean(axis=0)}")  # Should be ~0
    print(f"  Stds:  {X_ours.std(axis=0)}")   # Should be ~1
    
    assert np.allclose(X_ours, X_sklearn), "StandardScaler mismatch!"
    print("✓ StandardScaler matches sklearn!\n")


def test_minmax_scaler():
    """Test MinMaxScaler against sklearn."""
    np.random.seed(42)
    
    X = np.column_stack([
        np.random.randn(100) * 1000 + 5000,
        np.random.randn(100) * 0.1 + 0.5,
        np.random.randn(100) * 50 - 100
    ])
    
    # Default [0, 1] range
    our_scaler = MinMaxScaler()
    X_ours = our_scaler.fit_transform(X)
    
    sk_scaler = SklearnMinMax()
    X_sklearn = sk_scaler.fit_transform(X)
    
    print("=== MinMaxScaler Test (default range) ===")
    print(f"Max difference: {np.max(np.abs(X_ours - X_sklearn)):.2e}")
    print(f"Min values: {X_ours.min(axis=0)}")  # Should be 0
    print(f"Max values: {X_ours.max(axis=0)}")  # Should be 1
    
    assert np.allclose(X_ours, X_sklearn), "MinMaxScaler mismatch!"
    print("✓ MinMaxScaler matches sklearn!\n")
    
    # Custom range [-1, 1]
    our_scaler2 = MinMaxScaler(feature_range=(-1, 1))
    X_ours2 = our_scaler2.fit_transform(X)
    
    sk_scaler2 = SklearnMinMax(feature_range=(-1, 1))
    X_sklearn2 = sk_scaler2.fit_transform(X)
    
    print("=== MinMaxScaler Test (range [-1, 1]) ===")
    print(f"Max difference: {np.max(np.abs(X_ours2 - X_sklearn2)):.2e}")
    print(f"Min values: {X_ours2.min(axis=0)}")  # Should be -1
    print(f"Max values: {X_ours2.max(axis=0)}")  # Should be 1
    
    assert np.allclose(X_ours2, X_sklearn2), "MinMaxScaler custom range mismatch!"
    print("✓ MinMaxScaler (custom range) matches sklearn!\n")


def test_fit_transform_separation():
    """
    Demonstrate why fit/transform separation matters.
    This is a critical concept!
    """
    print("=== Fit/Transform Separation Demo ===\n")
    
    # Simulate train/test split
    np.random.seed(42)
    X_train = np.array([[100], [200], [300], [400], [500]])
    X_test = np.array([[150], [600]])  # Note: 600 is outside training range!
    
    scaler = StandardScaler()
    
    # CORRECT: fit on train, transform both
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training data:")
    print(f"  Original: {X_train.flatten()}")
    print(f"  Scaled:   {X_train_scaled.flatten()}")
    print(f"  Mean={scaler.mean_[0]:.1f}, Std={scaler.std_[0]:.1f}")
    
    print(f"\nTest data (using TRAINING statistics):")
    print(f"  Original: {X_test.flatten()}")
    print(f"  Scaled:   {X_test_scaled.flatten()}")
    print(f"  Note: 600 scales to {X_test_scaled[1,0]:.2f} (outside [-1,1] - that's OK!)")
    
    # WRONG: fitting on test data separately
    wrong_scaler = StandardScaler()
    X_test_wrong = wrong_scaler.fit_transform(X_test)
    
    print(f"\n⚠️  WRONG - Fitting on test data separately:")
    print(f"  Test scaled: {X_test_wrong.flatten()}")
    print(f"  This uses different μ={wrong_scaler.mean_[0]:.1f}, σ={wrong_scaler.std_[0]:.1f}")
    print(f"  → Model trained on different scale than test data!")


def demonstrate_scaling_impact():
    """Show how scaling affects gradient descent convergence."""
    print("\n=== Scaling Impact on Gradient Descent ===\n")
    
    np.random.seed(42)
    
    # Create data with very different scales
    n_samples = 100
    X_unscaled = np.column_stack([
        np.random.randn(n_samples) * 1000,  # Feature 1: scale ~1000
        np.random.randn(n_samples) * 0.01   # Feature 2: scale ~0.01
    ])
    
    # True weights (similar magnitude)
    true_weights = np.array([2.0, 3.0])
    y = X_unscaled @ true_weights + np.random.randn(n_samples) * 0.1
    
    print(f"Feature scales: {X_unscaled.std(axis=0)}")
    print(f"True weights: {true_weights}")
    
    # Try gradient descent without scaling
    # (This will need very small learning rate)
    from core.optimizers import BatchGradientDescent
    from core.linear_regression import LinearRegressionGD as LR
    
    # This would struggle or diverge with normal learning rate
    print("\nWithout scaling:")
    print("  Gradient for feature 1 will be ~100,000x larger than feature 2")
    print("  Makes finding a good learning rate nearly impossible!")
    
    # With scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_unscaled)
    
    print(f"\nAfter scaling:")
    print(f"  Feature scales: {X_scaled.std(axis=0)}")
    print("  Now both gradients are comparable magnitude")
    print("  → Gradient descent converges smoothly!")


if __name__ == "__main__":
    test_standard_scaler()
    test_minmax_scaler()
    test_fit_transform_separation()
    demonstrate_scaling_impact()