# src/models/online_learning.py
"""
Online Learning: Update model incrementally with new data.

STOCHASTIC GRADIENT DESCENT FOR ONLINE LEARNING:
─────────────────────────────────────────────────
Instead of computing gradient on full dataset,
update weights after each sample (or small batch).

This allows continuous learning without storing all data!
"""

import numpy as np


class OnlineRidgeRegression:
    """
    Ridge Regression with online (incremental) learning.
    
    Uses Stochastic Gradient Descent to update weights
    as new data arrives.
    
    Key Parameters:
    - learning_rate: How much to adjust weights per sample
    - decay: Reduce learning rate over time (for convergence)
    - alpha: L2 regularization strength
    
    Example
    -------
    >>> # Initial training (optional)
    >>> model = OnlineRidgeRegression(alpha=1.0)
    >>> model.partial_fit(X_initial, y_initial)
    >>> 
    >>> # As new data arrives...
    >>> for X_new, y_new in data_stream:
    ...     model.partial_fit(X_new, y_new)
    ...     predictions = model.predict(X_test)
    """
    
    def __init__(self, alpha=1.0, learning_rate=0.01, decay=0.0001, 
                 n_features=None, warm_start=True):
        """
        Parameters
        ----------
        alpha : float
            L2 regularization strength
        learning_rate : float
            Initial learning rate for SGD
        decay : float
            Learning rate decay: lr = lr_0 / (1 + decay * t)
        n_features : int, optional
            Number of features (if known ahead of time)
        warm_start : bool
            If True, retain weights between partial_fit calls
        """
        self.alpha = alpha
        self.learning_rate_init = learning_rate
        self.decay = decay
        self.n_features = n_features
        self.warm_start = warm_start
        
        # Model state
        self.weights_ = None
        self.n_samples_seen_ = 0
        self._is_initialized = False
    
    def _initialize_weights(self, n_features):
        """Initialize weights (zeros or small random)."""
        self.n_features = n_features
        self.weights_ = np.zeros(n_features + 1)  # +1 for bias
        self._is_initialized = True
    
    def _get_learning_rate(self):
        """Get current learning rate with decay."""
        return self.learning_rate_init / (1 + self.decay * self.n_samples_seen_)
    
    def partial_fit(self, X, y):
        """
        Incrementally fit on new data.
        
        This is the core online learning method!
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_features,)
            New training samples
        y : ndarray of shape (n_samples,) or scalar
            New target values
            
        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples, n_features = X.shape
        
        # Initialize if needed
        if not self._is_initialized:
            self._initialize_weights(n_features)
        
        # Add bias term
        X_bias = np.column_stack([np.ones(n_samples), X])
        
        # SGD update for each sample
        for i in range(n_samples):
            xi = X_bias[i]
            yi = y[i]
            
            # Prediction
            y_pred = np.dot(xi, self.weights_)
            
            # Error
            error = y_pred - yi
            
            # Gradient: ∂L/∂w = error * x + alpha * w (for L2 reg)
            gradient = error * xi
            gradient[1:] += self.alpha * self.weights_[1:]  # Don't regularize bias
            
            # Update
            lr = self._get_learning_rate()
            self.weights_ -= lr * gradient
            
            self.n_samples_seen_ += 1
        
        return self
    
    def fit(self, X, y, n_epochs=10):
        """
        Batch fit (multiple passes over data).
        
        For initial training when you have historical data.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if not self._is_initialized:
            self._initialize_weights(X.shape[1])
        
        for epoch in range(n_epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(len(X))
            self.partial_fit(X[indices], y[indices])
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self._is_initialized:
            raise RuntimeError("Model not fitted. Call partial_fit first.")
        
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_bias = np.column_stack([np.ones(len(X)), X])
        return np.dot(X_bias, self.weights_)
    
    def get_params(self):
        """Get model parameters."""
        return {
            'alpha': self.alpha,
            'learning_rate_init': self.learning_rate_init,
            'decay': self.decay,
            'n_samples_seen': self.n_samples_seen_,
        }


class IncrementalLearner:
    """
    Wrapper for incremental/online learning with monitoring.
    
    Features:
    - Periodic evaluation
    - Automatic retraining triggers
    - Learning rate adaptation
    """
    
    def __init__(self, base_model, evaluation_interval=100, 
                 performance_threshold=0.8):
        """
        Parameters
        ----------
        base_model : model with partial_fit
            The underlying online learning model
        evaluation_interval : int
            Evaluate performance every N samples
        performance_threshold : float
            Trigger alert if performance drops below
        """
        self.model = base_model
        self.evaluation_interval = evaluation_interval
        self.performance_threshold = performance_threshold
        
        # Buffers for evaluation
        self.X_buffer = []
        self.y_buffer = []
        
        # History
        self.performance_history = []
        self.n_updates = 0
    
    def update(self, X, y, X_holdout=None, y_holdout=None):
        """
        Update model with new data.
        
        Parameters
        ----------
        X, y : arrays
            New training data
        X_holdout, y_holdout : arrays, optional
            Holdout set for evaluation
        """
        # Update model
        self.model.partial_fit(X, y)
        self.n_updates += 1
        
        # Buffer for periodic evaluation
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        self.X_buffer.extend(X)
        self.y_buffer.extend(y.ravel())
        
        # Periodic evaluation
        if self.n_updates % self.evaluation_interval == 0:
            self._evaluate(X_holdout, y_holdout)
    
    def _evaluate(self, X_holdout=None, y_holdout=None):
        """Evaluate current model performance."""
        # Use holdout if provided, otherwise use recent buffer
        if X_holdout is not None and y_holdout is not None:
            X_eval = X_holdout
            y_eval = y_holdout
        else:
            # Use last N buffered samples
            n = min(200, len(self.X_buffer))
            X_eval = np.array(self.X_buffer[-n:])
            y_eval = np.array(self.y_buffer[-n:])
        
        if len(X_eval) < 10:
            return
        
        # Predict and score
        y_pred = self.model.predict(X_eval)
        
        ss_res = np.sum((y_eval - y_pred) ** 2)
        ss_tot = np.sum((y_eval - y_eval.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        self.performance_history.append({
            'n_updates': self.n_updates,
            'r2': r2,
            'n_samples': self.model.n_samples_seen_,
        })
        
        # Check for degradation
        if r2 < self.performance_threshold:
            print(f"Warning: Performance degraded below threshold ({r2:.4f} < {self.performance_threshold:.4f}).")
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def get_learning_curve(self):
        """Get performance over time."""
        return self.performance_history