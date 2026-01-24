

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Split data into training and test sets.
    
    Why not use all data for training?
    - Need unbiased estimate of real-world performance
    - Model might memorize training data (overfit)
    - Test set simulates "new, unseen data"
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target values
    test_size : float, default=0.2
        Fraction of data for test set
    random_state : int, optional
        Random seed for reproducibility
    shuffle : bool, default=True
        Whether to shuffle before splitting
        
    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    # Create indices
    indices = np.arange(n_samples)
    
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)
    
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def temporal_train_test_split(X, y, time_column, test_size=0.2):
    """
    Split data temporally (older data for train, newer for test).
    
    Why temporal split?
    - More realistic for time-series data
    - Prevents "peeking into the future"
    - Better estimate of production performance
    
    For Ames data: Train on 2006-2008, test on 2009-2010
    """
    # Sort by time
    sort_idx = np.argsort(time_column)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Split
    n_samples = len(X)
    n_train = int(n_samples * (1 - test_size))
    
    return (X_sorted[:n_train], X_sorted[n_train:], 
            y_sorted[:n_train], y_sorted[n_train:])


# =============================================================================
# METRICS
# =============================================================================

def r2_score(y_true, y_pred):
    """
    R² (Coefficient of Determination).
    
    R² = 1 - (SS_res / SS_tot)
    
    Interpretation:
    - R² = 1.0: Perfect predictions
    - R² = 0.0: Model is as good as predicting mean
    - R² < 0.0: Model is WORSE than predicting mean!
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    
    return 1 - (ss_res / ss_tot)


def mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    
    Same units as target variable — more interpretable!
    RMSE of $25,000 means "on average, off by $25k"
    """
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """
    Mean Absolute Error.
    
    Less sensitive to outliers than RMSE.
    """
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.
    
    Expressed as percentage — easy to interpret!
    MAPE of 10% means "on average, off by 10%"
    
    ⚠️ Warning: Undefined when y_true contains zeros
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmsle(y_true, y_pred):
    """
    Root Mean Squared Logarithmic Error.
    
    Good for right-skewed targets (like house prices).
    Penalizes under-predictions more than over-predictions.
    
    Often used in Kaggle competitions for price prediction!
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Clip predictions to be non-negative
    y_pred = np.clip(y_pred, 0, None)
    
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


def evaluate_model(y_true, y_pred, prefix=''):
    """
    Compute all metrics and return as dictionary.
    """
    return {
        f'{prefix}R2': r2_score(y_true, y_pred),
        f'{prefix}RMSE': rmse(y_true, y_pred),
        f'{prefix}MAE': mae(y_true, y_pred),
        f'{prefix}MAPE': mape(y_true, y_pred),
        f'{prefix}RMSLE': rmsle(y_true, y_pred),
    }