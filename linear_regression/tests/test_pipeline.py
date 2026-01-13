"""Test Pipeline against sklearn."""

import numpy as np
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler as SklearnScaler
from sklearn.linear_model import Ridge as SklearnRidge

from preprocessing.pipeline import Pipeline
from preprocessing.scalers import StandardScaler
from core.linear_regression import LinearRegressionGD
from core.regularized import RidgeRegression


def test_basic_pipeline():
    """Test basic pipeline functionality."""
    print("=== Basic Pipeline Test ===\n")
    
    np.random.seed(42)
    
    # Generate data with different feature scales
    n_samples = 100
    X = np.column_stack([
        np.random.randn(n_samples) * 100 + 500,   # Scale ~100
        np.random.randn(n_samples) * 0.1 + 2,     # Scale ~0.1
        np.random.randn(n_samples) * 50 - 25      # Scale ~50
    ])
    
    true_weights = np.array([0.5, 2.0, -1.0])
    y = X @ true_weights + np.random.randn(n_samples) * 5
    
    # Split manually
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Our pipeline
    our_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegressionGD())
    ])
    
    our_pipe.fit(X_train, y_train)
    our_pred = our_pipe.predict(X_test)
    
    print(f"Pipeline structure:\n{our_pipe}\n")
    print(f"Predictions (first 5): {our_pred[:5]}")
    print(f"Actual (first 5):      {y_test[:5]}")
    
    # Verify we can access named steps
    print(f"\nAccessing named steps:")
    print(f"  Scaler mean: {our_pipe['scaler'].mean_}")
    print(f"  Model weights: {our_pipe['model'].coef_}")


def test_against_sklearn():
    """Compare our pipeline to sklearn's."""
    print("\n=== Pipeline vs Sklearn ===\n")
    
    np.random.seed(42)
    
    n_samples = 200
    X = np.column_stack([
        np.random.randn(n_samples) * 1000,
        np.random.randn(n_samples) * 0.01,
        np.random.randn(n_samples) * 100
    ])
    
    true_weights = np.array([1.0, 500.0, -2.0])
    y = X @ true_weights + np.random.randn(n_samples) * 10
    
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    # Our pipeline with Ridge
    our_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeRegression(alpha=1.0))
    ])
    our_pipe.fit(X_train, y_train)
    our_pred = our_pipe.predict(X_test)
    
    # Sklearn pipeline
    sk_pipe = SklearnPipeline([
        ('scaler', SklearnScaler()),
        ('ridge', SklearnRidge(alpha=1.0))
    ])
    sk_pipe.fit(X_train, y_train)
    sk_pred = sk_pipe.predict(X_test)
    
    # Compare predictions
    pred_diff = np.max(np.abs(our_pred - sk_pred))
    print(f"Max prediction difference: {pred_diff:.6f}")
    
    # Compare R² scores
    our_r2 = 1 - np.sum((y_test - our_pred)**2) / np.sum((y_test - y_test.mean())**2)
    sk_r2 = sk_pipe.score(X_test, y_test)
    
    print(f"Our R²:     {our_r2:.6f}")
    print(f"Sklearn R²: {sk_r2:.6f}")
    
    # Predictions should be very close
    if pred_diff < 0.01:
        print("✓ Pipeline matches sklearn!")
    else:
        print(f"⚠ Difference larger than expected: {pred_diff}")


def test_data_leakage_prevention():
    """
    Demonstrate how pipelines prevent data leakage.
    
    This is THE key benefit of pipelines!
    """
    print("\n=== Data Leakage Prevention Demo ===\n")
    
    np.random.seed(42)
    
    # Training data: values around 100
    X_train = np.array([[90], [95], [100], [105], [110]])
    y_train = np.array([9, 9.5, 10, 10.5, 11])
    
    # Test data: values around 200 (completely different distribution!)
    X_test = np.array([[190], [200], [210]])
    y_test = np.array([19, 20, 21])
    
    # With Pipeline (CORRECT)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegressionGD())
    ])
    pipe.fit(X_train, y_train)
    
    print("CORRECT (Pipeline):")
    print(f"  Scaler fitted on train: μ={pipe['scaler'].mean_[0]:.1f}, σ={pipe['scaler'].std_[0]:.1f}")
    
    # The scaler will use training statistics on test data
    # Test data will be scaled as: (200 - 100) / 7.07 ≈ 14.1 (way outside training range)
    X_test_scaled_correct = pipe['scaler'].transform(X_test)
    print(f"  Test data scaled: {X_test_scaled_correct.flatten()}")
    print(f"  (Values outside [-2, 2] because test distribution differs)")
    
    # Without Pipeline - WRONG approach
    print("\nWRONG (fitting scaler on test data):")
    wrong_scaler = StandardScaler()
    wrong_scaler.fit(X_test)  # BUG: fitting on test!
    X_test_scaled_wrong = wrong_scaler.transform(X_test)
    print(f"  Scaler fitted on test: μ={wrong_scaler.mean_[0]:.1f}, σ={wrong_scaler.std_[0]:.1f}")
    print(f"  Test data scaled: {X_test_scaled_wrong.flatten()}")
    print(f"  (Looks 'normal' but uses WRONG scaling - data leakage!)")


def test_multi_step_pipeline():
    """Test pipeline with multiple transformers."""
    print("\n=== Multi-Step Pipeline ===\n")
    
    # Let's create a simple polynomial feature transformer
    class PolynomialFeatures:
        """Add polynomial features (simplified)."""
        def __init__(self, degree=2):
            self.degree = degree
        
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            
            features = [X]
            for d in range(2, self.degree + 1):
                features.append(X ** d)
            
            return np.hstack(features)
        
        def fit_transform(self, X, y=None):
            return self.fit(X).transform(X)
    
    np.random.seed(42)
    
    # Nonlinear data
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = 0.5 * X.flatten()**2 - 2 * X.flatten() + 1 + np.random.randn(100) * 0.5
    
    # Pipeline: Scale → Add polynomial features → Scale again → Model
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('scaler', StandardScaler()),
        ('model', LinearRegressionGD())
    ])
    
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
    
    print(f"Multi-step pipeline: Poly(2) → Scale → LinearReg")
    print(f"R² score: {r2:.4f}")
    print(f"Model weights: {pipe['model'].coef_}")
    print("  (Should roughly match [1, -2, 0.5] for intercept, x, x²)")


def test_pipeline_params():
    """Test get_params for hyperparameter access."""
    print("\n=== Pipeline Parameters ===\n")
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeRegression(alpha=0.5))
    ])
    
    params = pipe.get_params(deep=True)
    
    print("Pipeline parameters:")
    for key, value in params.items():
        if key != 'steps':  # Skip the full steps list
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_basic_pipeline()
    test_against_sklearn()
    test_data_leakage_prevention()
    test_multi_step_pipeline()
    test_pipeline_params()