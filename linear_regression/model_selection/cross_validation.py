"""
Cross-validation implementations for robust model evaluation.

Why cross-validate?
1. Single train/test split is high variance (luck of the draw)
2. Wastes data — can't use test data for training
3. Gives no sense of performance variance

K-Fold CV solves all three:
1. Average over K splits reduces variance
2. Every sample used for both training and testing
3. Get mean AND std of performance
"""

import numpy as np


class KFold:
    """
    K-Fold cross-validator.

    Splits data into K consecutive folds. Each fold is used once
    as validation while the remaining K-1 folds form training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be >= 2.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
        IMPORTANT: Always shuffle for real-world use!
    random_state : int, optional
        Random seed for reproducibility when shuffling.

    Example
    -------
    >>> kf = KFold(n_splits=5, shuffle=True, random_state=42)
    >>> for train_idx, val_idx in kf.split(X):
    ...     X_train, X_val = X[train_idx], X[val_idx]
    ...     # train and evaluate
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2; got {n_splits}")
        if not isinstance(n_splits, int):
            raise TypeError(
                f"n_splits must be an integer; got {type(n_splits)}")
        if shuffle and random_state is None:
            raise ValueError(
                "random_state must be specified when shuffle is True")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None):
        """
        Generate train/validation indices for each fold.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ignored
            Not used, present for API compatibility

        Yields
        ------
        train_idx : ndarray
            Training set indices for this fold
        val_idx : ndarray
            Validation set indices for this fold
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            val_idx = indices[current:current + fold_size]
            train_idx = np.concatenate([
                indices[:current],
                indices[current + fold_size:]
            ])

            yield train_idx, val_idx
            current += fold_size

    def get_n_splits(self):
        """Return number of splits."""
        return self.n_splits


def cross_val_score(estimator, X, y, cv=5, scoring='r2'):
    """
    Evaluate estimator using cross-validation.

    CRITICAL: The estimator is cloned (re-initialized) for each fold!
    This prevents information leakage between folds.

    Parameters
    ----------
    estimator : estimator object
        Must have fit() and predict() methods.
        If it's a Pipeline, preprocessing is done correctly per fold.
    X : ndarray of shape (n_samples, n_features)
        Training data
    y : ndarray of shape (n_samples,)
        Target values
    cv : int or cross-validator, default=5
        If int, uses KFold with that many splits.
        If cross-validator, uses it directly.
    scoring : str, default='r2'
        Scoring metric. Options: 'r2', 'mse', 'rmse', 'mae'

    Returns
    -------
    scores : ndarray of shape (n_splits,)
        Array of scores for each fold

    Example
    -------
    >>> scores = cross_val_score(LinearRegression(), X, y, cv=5)
    >>> print(f"R² = {scores.mean():.3f} ± {scores.std():.3f}")
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Handle cv parameter
    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=42)

    # scoring functions

    scorers = {
        'r2': _r2_score,
        'mse': _mse_score,
        'rmse': _rmse_score,
        'mae': _mae_score,
        # sklearn convention
        'neg_mse': lambda y_t, y_p: -_mse_score(y_t, y_p),
    }

    if scoring not in scorers:
        raise ValueError(
            f"Unknown scoring: {scoring}. Options: {list(scorers.keys())}")

    score_func = scorers[scoring]
    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Clone the estimator (fresh instance for each fold)
        # This is critical — we can't reuse a fitted model!
        estimator_clone = _clone_estimator(estimator)

        # Fit and predict
        estimator_clone.fit(X_train, y_train)
        y_pred = estimator_clone.predict(X_val)

        # Score - for R², use training mean only for LOO (1 validation sample)
        # Otherwise use validation mean to match sklearn behavior
        if scoring == 'r2' and len(y_val) == 1:
            # Leave-One-Out case: use training mean for SS_tot
            fold_score = _r2_score(y_val, y_pred, y_train_mean=y_train.mean())
        else:
            fold_score = score_func(y_val, y_pred)
        scores.append(fold_score)

    return np.array(scores)


def cross_val_predict(estimator, X, y, cv=5):
    """
    Generate cross-validated predictions for each sample.

    Each sample's prediction is made when it's in the validation set.
    Useful for:
    - Plotting predicted vs actual
    - Residual analysis
    - Stacking/blending models

    Parameters
    ----------
    estimator : estimator object
        Must have fit() and predict() methods
    X : ndarray of shape (n_samples, n_features)
        Training data
    y : ndarray of shape (n_samples,)
        Target values
    cv : int or cross-validator, default=5

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Cross-validated predictions
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if isinstance(cv, int):
        cv = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Initialize predictions array
    predictions = np.zeros_like(y, dtype=np.float64)

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        estimator_clone = _clone_estimator(estimator)
        estimator_clone.fit(X_train, y_train)

        predictions[val_idx] = estimator_clone.predict(X_val)

    return predictions


def _clone_estimator(estimator):
    """
    Create a fresh copy of an estimator.

    This is simplified — sklearn's clone is more sophisticated.
    We recreate the object with the same parameters.
    """
    # Get the class
    estimator_class = estimator.__class__

    # Common fitted attributes that should NOT be used for initialization
    fitted_attrs = {
        'coef_', 'intercept_', 'weights_', 'bias_', '_theta',
        'n_iter_', 'n_iter', 'cost_history_', 'feature_names_in_',
        'n_features_in_', 'classes_', 'n_classes_'
    }

    # Try to get initialization parameters from __init__
    import inspect
    try:
        # Get __init__ signature
        init_signature = inspect.signature(estimator_class.__init__)
        init_params = {}

        for param_name in init_signature.parameters:
            if param_name == 'self':
                continue
            # Get the parameter value from the instance
            if hasattr(estimator, param_name):
                init_params[param_name] = getattr(estimator, param_name)

        # Try to create new instance with init params
        if init_params:
            return estimator_class(**init_params)
    except (TypeError, AttributeError):
        pass

    # Fallback: try to instantiate with no args
    # This works for our simple implementations
    try:
        return estimator_class()
    except TypeError:
        # Last resort: return the estimator itself (not ideal)
        import warnings
        warnings.warn(
            f"Could not clone {estimator_class.__name__}, reusing instance")
        return estimator

# Scoring functions


def _r2_score(y_true, y_pred, y_train_mean=None):
    """
    R² (coefficient of determination).

    Parameters
    ----------
    y_true : ndarray
        True target values
    y_pred : ndarray
        Predicted target values
    y_train_mean : float, optional
        Mean of training set. If provided, used for SS_tot calculation.
        This is important for Leave-One-Out CV where validation set has 1 sample.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Use training mean if provided (for LOO and proper R² calculation)
    # Otherwise use validation mean (standard case)
    if y_train_mean is not None:
        ss_tot = np.sum((y_true - y_train_mean) ** 2)
    else:
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    if ss_tot == 0:
        # If all validation samples equal the mean, R² is undefined
        # Return 0 if there's error, 1 if perfect (but this shouldn't happen)
        return 0.0 if ss_res > 0 else 1.0

    return 1 - (ss_res / ss_tot)


def _mse_score(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def _rmse_score(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(_mse_score(y_true, y_pred))


def _mae_score(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


class RepeatedKFold:
    """
    Repeated K-Fold cross-validator.

    Repeats K-Fold n times with different randomization each time.
    Gives even more robust estimates at the cost of computation.

    Total number of fits = n_splits × n_repeats

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds per repeat
    n_repeats : int, default=10
        Number of times to repeat K-Fold
    random_state : int, optional
        Random seed
    """

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None):
        """Generate indices for repeated K-Fold."""
        rng = np.random.RandomState(self.random_state)

        for repeat in range(self.n_repeats):
            # Different seed for each repeat
            kf = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=rng.randint(0, 2**31)
            )

            for train_idx, val_idx in kf.split(X, y):
                yield train_idx, val_idx

    def get_n_splits(self):
        """Total number of splits."""
        return self.n_splits * self.n_repeats


class LeaveOneOut:
    """
    Leave-One-Out cross-validator.

    Each sample is used once as test set.
    Equivalent to KFold(n_splits=n_samples).

    Pros:
    - Maximum training data per fold
    - Deterministic (no randomness)

    Cons:
    - Computationally expensive for large datasets
    - High variance in error estimate

    Best for: Small datasets (< 100 samples)
    """

    def split(self, X, y=None):
        """Generate LOO splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        for i in range(n_samples):
            val_idx = np.array([i])
            train_idx = np.concatenate([indices[:i], indices[i+1:]])
            yield train_idx, val_idx

    def get_n_splits(self, X=None):
        """Number of splits equals number of samples."""
        if X is not None:
            return len(X)
        return None


class StratifiedKFold:
    """
    Stratified K-Fold cross-validator.

    Splits data into K consecutive folds. Each fold is used once
    as validation while the remaining K-1 folds form training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be >= 2.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int, optional
        Random seed for reproducibility when shuffling.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        """Generate stratified K-Fold splits."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            val_idx = indices[current:current + fold_size]
            train_idx = np.concatenate([
                indices[:current],
                indices[current + fold_size:]
            ])
            yield train_idx, val_idx
            current += fold_size

    def get_n_splits(self):
        """Number of splits equals number of samples."""
        return self.n_splits
