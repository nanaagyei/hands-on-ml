"""
Feature scaling implementations.

Why fit/transform separation?
- fit(): Learn parameters from TRAINING data only
- transform(): Apply those parameters to ANY data

This prevents "data leakage" — using test data statistics
during training would give overly optimistic results. 
"""

import numpy as np

class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance.
    
    Formula: z = (x - μ) / σ
    
    After scaling:
    - Mean ≈ 0
    - Standard deviation ≈ 1
    
    Why use it?
    - Gradient descent converges faster
    - Features contribute equally to distance-based algorithms
    - Required for regularization to be fair across features
    
    Mathematical intuition:
    - Subtracting mean centers the data at origin
    - Dividing by std makes all features "speak the same language"
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.n_features_ = None
        self._is_fitted = False
    
    def fit(self, X):
        """
        Compute mean and std from training data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : returns fitted StandardScaler
        """

        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_ = X.shape[1]

        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0, ddof=0)

        # Handle zero variance features (constant columns)
        # Avoid division by zero — set std to 1 so feature stays at 0 after centering
        self.std_[self.std_ == 0] = 1.0
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Apply standardization using fitted parameters.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Standardized data
        """

        if not self._is_fitted:
            raise ValueError("This scaler is not fitted yet. Call 'fit' with appropriate data first.")
        
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X must have {self.n_features_} features, but has {X.shape[1]}.")
        
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Standardized data
        """

        return self.fit(X).transform(X)
    
    def inverse_transform(self, X):
        """
        Reverse the transformation applied by fit_transform.

        Useful for interpreting predictions!
        
        Formula: x = z * std + mean
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Standardized data
            
        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Original data
        """

        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X must have {self.n_features_} features, but has {X.shape[1]}.")
        
        return X * self.std_ + self.mean_
    
    def get_params(self, deep=True):
        """
        Get the parameters for this scaler.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """

        return {
            "mean_": self.mean_,
            "std_": self.std_,
            "n_features_": self.n_features_,
            "_is_fitted": self._is_fitted
        }

class MinMaxScaler:
    """
    Scale features to a given range.

    Formula: z = (x - min) / (max - min)

    After scaling:
    - Range is [0, 1]
    - Features contribute equally to distance-based algorithms
    - Required for regularization to be fair across features
    """

    def __init__(self, feature_range=(0, 1)):
        """
        Parameters
        ----------
        feature_range : tuple of floats (min, max), default=(0, 1)
            Desired range of transformed data.
        """

        self.feature_range = feature_range
        self.min_ = None         # minimum value per feature
        self.max_ = None         # maximum value per feature  
        self.data_range_ = None  # max - min per feature
        self.n_features_ = None
        self._is_fitted = False
        
        # Validate feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range[0] must be < feature_range[1]")
        
    def fit(self, X):
        """
        Compute min and max from training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
            
        Returns
        -------
        self : returns fitted MinMaxScaler
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.n_features_ = X.shape[1]
        
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.data_range_ = self.max_ - self.min_
        
        # Handle zero range (constant features)
        self.data_range_[self.data_range_ == 0] = 1.0
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Apply min-max scaling using fitted parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Scaled data
        """

        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, but got {X.shape[1]}.")
        
        # Scale to [0, 1]
        X_scaled = (X - self.min_) / self.data_range_
        
        # Scale to desired range
        range_min, range_max = self.feature_range
        return X_scaled * (range_max - range_min) + range_min
    
    def fit_transform(self, X):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_scaled : ndarray of shape (n_samples, n_features)
            Scaled data
        """

        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_scaled):
        """
        Reverse the transformation applied by fit_transform.

        Useful for interpreting predictions!
        
        Formula: x = z * (max - min) + min
        
        Parameters
        ----------
        X_scaled : ndarray of shape (n_samples, n_features)
            Scaled data

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Original data
        """

        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        X_scaled = np.asarray(X_scaled, dtype=np.float64)
        
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(-1, 1)
        
        if X_scaled.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, but got {X_scaled.shape[1]}.")
        
        # Scale back to original range
        range_min, range_max = self.feature_range
        X_01 = (X_scaled - range_min) / (range_max - range_min)

        return X_01 * self.data_range_ + self.min_
    
    def get_params(self, deep=True):
        """
        Get the parameters for this scaler.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'min': self.min_,
            'max': self.max_,
            'data_range': self.data_range_,
            'feature_range': self.feature_range,
            'n_features': self.n_features_
        }