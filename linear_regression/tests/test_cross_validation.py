"""Test cross-validation against sklearn."""

from core.linear_regression import LinearRegression, LinearRegressionGD, LinearRegressionSGD
from core.regularized import RidgeRegression
from preprocessing.scalers import StandardScaler
from preprocessing.pipeline import Pipeline
from model_selection.cross_validation import KFold, cross_val_score, cross_val_predict, RepeatedKFold, LeaveOneOut
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler as SklearnScaler
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.model_selection import cross_val_predict as sklearn_cv_predict
from sklearn.model_selection import cross_val_score as sklearn_cv_score
from sklearn.model_selection import KFold as SklearnKFold
import numpy as np
import sys
from pathlib import Path

# Add the linear_regression directory to Python path
# This allows imports to work when running the test directly
test_dir = Path(__file__).parent.resolve()
# Go up from tests/ to linear_regression/
linear_regression_dir = test_dir.parent
if str(linear_regression_dir) not in sys.path:
    sys.path.insert(0, str(linear_regression_dir))


def test_kfold_splits():
    """Test that KFold generates correct splits."""
    print("=== KFold Split Test ===\n")

    X = np.arange(20).reshape(-1, 1)

    # Test without shuffle
    kf = KFold(n_splits=5, shuffle=False)

    print("5-Fold splits (no shuffle):")
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {i+1}: train={len(train_idx)} samples, val={val_idx}")

    # Verify each sample appears in validation exactly once
    all_val_indices = []
    for train_idx, val_idx in KFold(n_splits=5).split(X):
        all_val_indices.extend(val_idx)

    assert sorted(all_val_indices) == list(
        range(20)), "Each sample should be in val exactly once"
    print("\n✓ Each sample appears in validation exactly once!")

    # Compare with sklearn
    print("\nComparing with sklearn KFold:")
    our_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    sk_kf = SklearnKFold(n_splits=5, shuffle=True, random_state=42)

    for (our_train, our_val), (sk_train, sk_val) in zip(our_kf.split(X), sk_kf.split(X)):
        assert np.array_equal(
            sorted(our_train), sorted(sk_train)), "Train mismatch"
        assert np.array_equal(sorted(our_val), sorted(sk_val)), "Val mismatch"

    print("✓ KFold matches sklearn!\n")


def test_cross_val_score():
    """Test cross_val_score against sklearn."""
    print("=== cross_val_score Test ===\n")

    np.random.seed(42)

    # Generate data
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = X @ np.array([2, -1, 0.5]) + np.random.randn(n_samples) * 0.5

    # Our implementation with LinearRegression
    our_scores = cross_val_score(
        LinearRegression(), X, y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='r2'
    )

    # Sklearn
    from sklearn.linear_model import LinearRegression as SklearnLR
    sk_scores = sklearn_cv_score(
        SklearnLR(), X, y,
        cv=SklearnKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='r2'
    )

    print(f"Our scores:     {our_scores}")
    print(f"Sklearn scores: {sk_scores}")
    print(
        f"Our mean±std:     {our_scores.mean():.4f} ± {our_scores.std():.4f}")
    print(f"Sklearn mean±std: {sk_scores.mean():.4f} ± {sk_scores.std():.4f}")

    # Should be very close (small differences due to implementation details)
    assert np.allclose(our_scores, sk_scores,
                       atol=0.01), "Scores differ too much"
    print("✓ cross_val_score matches sklearn!\n")


def test_cv_with_pipeline():
    """
    Critical test: CV with Pipeline prevents data leakage.

    The scaler must be fit ONLY on training fold, not validation!
    """
    print("=== CV with Pipeline (Data Leakage Prevention) ===\n")

    np.random.seed(42)

    # Data with very different scales
    n_samples = 100
    X = np.column_stack([
        np.random.randn(n_samples) * 1000,
        np.random.randn(n_samples) * 0.01
    ])
    y = X @ np.array([0.001, 100]) + np.random.randn(n_samples) * 0.5

    # Our pipeline
    our_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeRegression(alpha=1.0))
    ])

    our_scores = cross_val_score(
        our_pipe, X, y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='r2'
    )

    # Sklearn pipeline
    sk_pipe = SklearnPipeline([
        ('scaler', SklearnScaler()),
        ('ridge', SklearnRidge(alpha=1.0))
    ])

    sk_scores = sklearn_cv_score(
        sk_pipe, X, y,
        cv=SklearnKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='r2'
    )

    print(
        f"Our Pipeline CV:     {our_scores.mean():.4f} ± {our_scores.std():.4f}")
    print(
        f"Sklearn Pipeline CV: {sk_scores.mean():.4f} ± {sk_scores.std():.4f}")

    # Demonstrate what happens with WRONG approach (leakage)
    print("\n⚠️  Demonstrating data leakage (WRONG approach):")

    # WRONG: Scale ALL data first, then CV
    wrong_scaler = StandardScaler()
    X_scaled_wrong = wrong_scaler.fit_transform(
        X)  # Leakage: val data influenced scaling!

    wrong_scores = cross_val_score(
        LinearRegression(), X_scaled_wrong, y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring='r2'
    )

    print(
        f"With leakage:    {wrong_scores.mean():.4f} ± {wrong_scores.std():.4f}")
    print(f"Without leakage: {our_scores.mean():.4f} ± {our_scores.std():.4f}")
    print("\n  Leakage often gives slightly better (but misleading) scores!")
    print("  Always use Pipeline inside CV to prevent this.\n")


def test_cross_val_predict():
    """Test cross_val_predict."""
    print("=== cross_val_predict Test ===\n")

    np.random.seed(42)

    n_samples = 50
    X = np.random.randn(n_samples, 2)
    y = X @ np.array([3, -2]) + np.random.randn(n_samples) * 0.3

    # Get CV predictions
    y_pred = cross_val_predict(
        LinearRegression(), X, y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42)
    )

    # Every sample should have a prediction
    assert len(y_pred) == len(y), "Should have prediction for each sample"

    # Calculate R² from CV predictions
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    cv_r2 = 1 - ss_res / ss_tot

    print(f"CV Predictions R²: {cv_r2:.4f}")
    print(f"Sample predictions vs actual:")
    for i in range(5):
        print(f"  y={y[i]:.3f}, ŷ={y_pred[i]:.3f}")

    print("\n✓ cross_val_predict working!\n")


def test_repeated_kfold():
    """Test RepeatedKFold for more robust estimates."""
    print("=== RepeatedKFold Test ===\n")

    np.random.seed(42)

    n_samples = 80
    X = np.random.randn(n_samples, 3)
    y = X @ np.array([1, 2, -1]) + np.random.randn(n_samples) * 0.5

    # Single 5-Fold
    single_scores = cross_val_score(
        LinearRegression(), X, y,
        cv=KFold(n_splits=5, shuffle=True, random_state=42)
    )

    # Repeated 5-Fold (5 folds × 3 repeats = 15 scores)
    repeated_scores = cross_val_score(
        LinearRegression(), X, y,
        cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    )

    print(
        f"Single 5-Fold:   {single_scores.mean():.4f} ± {single_scores.std():.4f} ({len(single_scores)} scores)")
    print(
        f"Repeated 5×3:    {repeated_scores.mean():.4f} ± {repeated_scores.std():.4f} ({len(repeated_scores)} scores)")
    print("\n  Repeated CV gives more stable estimates (lower variance in the mean)\n")


def test_leave_one_out():
    """Test Leave-One-Out CV."""
    print("=== Leave-One-Out Test ===\n")

    # Small dataset where LOO makes sense
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([2.1, 4.0, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.0])

    loo = LeaveOneOut()

    print(f"Dataset size: {len(X)}")
    print(f"LOO splits: {loo.get_n_splits(X)}")

    scores = cross_val_score(LinearRegression(), X, y, cv=loo, scoring='r2')

    print(f"LOO R² scores: {scores}")
    print(f"Mean R²: {scores.mean():.4f}")
    print("\n  LOO uses maximum training data but high variance in estimates\n")


def demonstrate_cv_interpretation():
    """Show how to interpret CV results."""
    print("=== Interpreting CV Results ===\n")

    np.random.seed(42)

    # Use a more challenging dataset with some noise features
    n_samples = 200
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    # Only first 5 features matter, rest are noise
    true_weights = np.array([3, -1, 2, 0.5, -0.5] + [0] * (n_features - 5))
    y = X @ true_weights + np.random.randn(n_samples) * 2

    # Compare different models
    models = {
        'Linear (Normal Eq)': LinearRegression(),
        'Linear (GD)': LinearRegressionGD(learning_rate=0.01, n_iterations=1000),
        'Linear (SGD)': LinearRegressionSGD(learning_rate=0.01),
        'Ridge(0.1)': RidgeRegression(alpha=0.1),
        'Ridge(1.0)': RidgeRegression(alpha=1.0),
        'Ridge(10)': RidgeRegression(alpha=10.0),
    }

    print("Model Comparison (10-Fold CV):\n")
    print(f"{'Model':<15} {'Mean R²':>10} {'Std':>10} {'95% CI':>20}")
    print("-" * 55)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        mean = scores.mean()
        std = scores.std()

        # 95% confidence interval (approximate)
        ci_low = mean - 1.96 * std / np.sqrt(len(scores))
        ci_high = mean + 1.96 * std / np.sqrt(len(scores))

        print(
            f"{name:<15} {mean:>10.4f} {std:>10.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    print("\nInterpretation:")
    print("  - Higher mean R² is better")
    print("  - Lower std means more consistent performance")
    print("  - Overlapping CIs → models may not be significantly different")


if __name__ == "__main__":
    test_kfold_splits()
    test_cross_val_score()
    test_cv_with_pipeline()
    test_cross_val_predict()
    test_repeated_kfold()
    test_leave_one_out()
    demonstrate_cv_interpretation()
