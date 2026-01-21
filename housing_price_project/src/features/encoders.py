"""
Categorical Encoding implementations.

WHY ENCODE?
───────────
ML models need numbers. Categories like "Excellent", "Good", "Fair" 
must become numeric. But HOW you encode matters enormously!

ENCODING TYPES:
───────────────
1. Label Encoding: Category → Integer
   "Red"=0, "Green"=1, "Blue"=2
   Problem: Implies ordering (2 > 1 > 0) where none exists!
   Use for: Ordinal categories (quality ratings)

2. One-Hot Encoding: Category → Binary columns  
   "Red"   → [1, 0, 0]
   "Green" → [0, 1, 0]
   "Blue"  → [0, 0, 1]
   No false ordering
   Problem: Explodes dimensions for high-cardinality features

3. Target Encoding: Category → Mean target value
   "Neighborhood_A" → mean(price) for homes in A
   Great for high-cardinality
   Risk: Data leakage! Must be careful with CV
"""

import numpy as np
from collections import defaultdict


class LabelEncoder:
    """
    Encode categorical labels as integers.
    
    Maps each unique category to an integer 0, 1, 2, ...
    
    Best for: ORDINAL categories where order matters
    Example: Quality ratings (Poor < Fair < Good < Excellent)
    
    WARNING: Don't use for nominal categories!
    "Red"=0, "Blue"=1, "Green"=2 implies Blue > Red, which is meaningless.
    
    Example
    -------
    >>> le = LabelEncoder()
    >>> le.fit(['cat', 'dog', 'cat', 'bird'])
    >>> le.transform(['cat', 'dog', 'bird'])
    array([1, 2, 0])  # Alphabetical order by default
    """

    def __init__(self):
        self.classes_ = None
        self.class_to_index_ = None
        self._is_fitted = False
    

    def fit(self, y):
        """
        Learn the unique categories.
        
        Parameters
        ----------
        y : array-like
            Categorical values
            
        Returns
        -------
        self
        """
        y = np.asarray(y).ravel()
        
        # Get unique values, sorted for consistency
        self.classes_ = np.sort(np.unique(y[y != None].astype(str)))
        self.class_to_index_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        
        self._is_fitted = True
        return self
    
    def transform(self, y):
        """
        Transform categories to integers.
        
        Parameters
        ----------
        y : array-like
            Categorical values
            
        Returns
        -------
        y_encoded : ndarray of integers
        """
        if not self._is_fitted:
            raise RuntimeError("LabelEncoder not fitted. Call fit() first.")
        
        y = np.asarray(y).ravel()
        encoded = np.zeros(len(y), dtype=int)
        
        for i, val in enumerate(y):
            val_str = str(val)
            if val_str in self.class_to_index_:
                encoded[i] = self.class_to_index_[val_str]
            else:
                # Unknown category - could raise error or use special value
                raise ValueError(f"Unknown category: {val_str}. "
                               f"Known categories: {list(self.classes_)}")
        
        return encoded
    
    def fit_transform(self, y):
        """Fit and transform in one step."""
        return self.fit(y).transform(y)
    
    def inverse_transform(self, y_encoded):
        """
        Convert integers back to original categories.
        
        Useful for interpreting predictions!
        """
        if not self._is_fitted:
            raise RuntimeError("LabelEncoder not fitted.")
        
        y_encoded = np.asarray(y_encoded).ravel()
        return self.classes_[y_encoded]
    
    def get_mapping(self):
        """Return the category -> integer mapping."""
        return self.class_to_index_.copy()
    

class OrdinalEncoder:
    """
    Encode ordinal categories with a SPECIFIED order.
    
    Unlike LabelEncoder (alphabetical), you define the order.
    This is crucial for ordinal features like quality ratings!
    
    Example
    -------
    >>> # Quality: Poor < Fair < Good < Excellent
    >>> oe = OrdinalEncoder(order=['Poor', 'Fair', 'Good', 'Excellent'])
    >>> oe.fit_transform(['Good', 'Poor', 'Excellent'])
    array([2, 0, 3])  # Respects the meaningful order!
    """

    def __init__(self, order=None, unknown_value=-1):
        """
        Parameters
        ----------
        order : list
            Ordered list of categories (lowest to highest)
        unknown_value : int
            Value to use for unknown categories (-1 default)
        """
        self.order = order
        self.unknown_value = unknown_value
        self.class_to_index_ = None
        self._is_fitted = False
    
    def fit(self, y=None):
        """
        Create mapping from specified order.
        
        If order was provided at init, fit just validates.
        If not, learns order from data (like LabelEncoder).
        """
        if self.order is not None:
            self.class_to_index_ = {cat: idx for idx, cat in enumerate(self.order)}
        elif y is not None:
            y = np.asarray(y).ravel()
            unique = np.unique(y[y != None].astype(str))
            self.order = list(unique)
            self.class_to_index_ = {cat: idx for idx, cat in enumerate(self.order)}
        else:
            raise ValueError("Must provide 'order' at init or 'y' at fit")
        
        self._is_fitted = True
        return self
    
    def transform(self, y):
        """Transform categories to ordered integers."""
        if not self._is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        
        y = np.asarray(y).ravel()
        encoded = np.full(len(y), self.unknown_value, dtype=int)
        
        for i, val in enumerate(y):
            val_str = str(val)
            if val_str in self.class_to_index_:
                encoded[i] = self.class_to_index_[val_str]
        
        return encoded
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)


class OneHotEncoder:
    """
    Encode categorical features as binary (one-hot) vectors.
    
    Each category becomes a separate binary column.
    
    "Red"   → [1, 0, 0]
    "Green" → [0, 1, 0]  
    "Blue"  → [0, 0, 1]
    
    Best for: NOMINAL categories (no natural order)
    Examples: Color, Neighborhood, Building Type
    
    Parameters
    ----------
    drop_first : bool, default=False
        Drop first category to avoid multicollinearity.
        With 3 categories, you only need 2 binary columns!
        (If it's not Red or Green, it must be Blue)
    sparse : bool, default=False
        Return sparse matrix (memory efficient for many categories)
    handle_unknown : str, default='error'
        'error': Raise error for unknown categories
        'ignore': Encode as all zeros
    
    Example
    -------
    >>> ohe = OneHotEncoder(drop_first=True)
    >>> ohe.fit(['Red', 'Green', 'Blue'])
    >>> ohe.transform(['Red', 'Blue', 'Green'])
    array([[0, 0],    # Red (reference)
           [0, 1],    # Blue
           [1, 0]])   # Green
    """

    def __init__(self, drop_first=False, sparse=False, handle_unknown='error'):
        self.drop_first = drop_first
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        
        self.categories_ = None
        self.n_categories_ = None
        self._is_fitted = False
    
    def fit(self, y):
        """
        Learn the unique categories.
        """
        y = np.asarray(y).ravel()
        
        # Get unique categories (sorted for consistency)
        self.categories_ = np.sort(np.unique(y.astype(str)))
        self.n_categories_ = len(self.categories_)
        
        self._is_fitted = True
        return self
    
    def transform(self, y):
        """
        Transform categories to one-hot encoded matrix.
        
        Returns
        -------
        encoded : ndarray of shape (n_samples, n_categories)
            or (n_samples, n_categories - 1) if drop_first=True
        """
        if not self._is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        
        y = np.asarray(y).ravel()
        n_samples = len(y)
        
        # Number of output columns
        n_cols = self.n_categories_ - 1 if self.drop_first else self.n_categories_
        
        # Initialize output
        encoded = np.zeros((n_samples, n_cols), dtype=np.float64)
        
        # Category to column index
        start_idx = 1 if self.drop_first else 0
        cat_to_col = {cat: i for i, cat in enumerate(self.categories_[start_idx:])}
        
        for i, val in enumerate(y):
            val_str = str(val)
            
            if val_str in cat_to_col:
                encoded[i, cat_to_col[val_str]] = 1.0
            elif self.drop_first and val_str == self.categories_[0]:
                # First category is the reference (all zeros)
                pass
            elif self.handle_unknown == 'ignore':
                # Unknown category = all zeros
                pass
            else:
                raise ValueError(f"Unknown category: {val_str}")
        
        return encoded
    
    def fit_transform(self, y):
        return self.fit(y).transform(y)
    
    def get_feature_names(self, prefix=''):
        """
        Get names for the output columns.
        
        Useful for understanding what each column represents!
        """
        if not self._is_fitted:
            raise RuntimeError("Not fitted.")
        
        start_idx = 1 if self.drop_first else 0
        names = [f"{prefix}_{cat}" if prefix else cat 
                 for cat in self.categories_[start_idx:]]
        return names


class TargetEncoder:
    """
    Encode categories using target variable statistics.
    
    Each category is replaced with the mean (or other stat) of the 
    target variable for that category.
    
    "Neighborhood_A" → mean(price) for homes in Neighborhood_A
    
    Best for: HIGH-CARDINALITY categories (many unique values)
    Example: Neighborhood (25+ values), ZipCode (1000s of values)
    
    Why not One-Hot for high cardinality?
    - 25 neighborhoods → 25 columns
    - 1000 zipcodes → 1000 columns!
    - Target encoding: always 1 column
    
    ⚠️ DANGER: DATA LEAKAGE!
    ─────────────────────────
    If you compute mean(target) using ALL data (including test),
    you're leaking test information into features!
    
    SOLUTION: Fit ONLY on training data.
    Better: Use cross-validation within training (see fit method).
    
    Parameters
    ----------
    smoothing : float, default=1.0
        Smoothing factor for regularization.
        Blends category mean with global mean.
        Higher = more smoothing (less trust in category mean)
    min_samples : int, default=1
        Minimum samples in category to use category mean.
        Below this, uses global mean.
    
    Example
    -------
    >>> te = TargetEncoder(smoothing=10)
    >>> te.fit(X_train['Neighborhood'], y_train)
    >>> X_train['Neighborhood_encoded'] = te.transform(X_train['Neighborhood'])
    """
    
    def __init__(self, smoothing=1.0, min_samples=1):
        self.smoothing = smoothing
        self.min_samples = min_samples
        
        self.global_mean_ = None
        self.category_stats_ = None  # {category: (mean, count)}
        self.encoding_map_ = None    # {category: encoded_value}
        self._is_fitted = False
    
    def fit(self, X, y):
        """
        Learn target statistics for each category.
        
        Parameters
        ----------
        X : array-like
            Categorical feature values
        y : array-like
            Target variable values
        """
        X = np.asarray(X).ravel()
        y = np.asarray(y, dtype=float).ravel()
        
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        
        # Global mean
        self.global_mean_ = np.mean(y)
        
        # Category statistics
        self.category_stats_ = {}
        
        # Group by category
        categories = np.unique(X)
        for cat in categories:
            mask = X == cat
            cat_y = y[mask]
            
            self.category_stats_[str(cat)] = {
                'mean': np.mean(cat_y),
                'count': len(cat_y)
            }
        
        # Compute smoothed encodings
        # Formula: encoded = (count * cat_mean + smoothing * global_mean) / (count + smoothing)
        # This is Bayesian-inspired: with few samples, trust global more
        self.encoding_map_ = {}
        
        for cat, stats in self.category_stats_.items():
            count = stats['count']
            cat_mean = stats['mean']
            
            if count < self.min_samples:
                # Not enough samples, use global mean
                self.encoding_map_[cat] = self.global_mean_
            else:
                # Smoothed encoding
                smoothed = (count * cat_mean + self.smoothing * self.global_mean_) / \
                          (count + self.smoothing)
                self.encoding_map_[cat] = smoothed
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Transform categories to encoded values.
        """
        if not self._is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        
        X = np.asarray(X).ravel()
        encoded = np.zeros(len(X), dtype=np.float64)
        
        for i, val in enumerate(X):
            val_str = str(val)
            if val_str in self.encoding_map_:
                encoded[i] = self.encoding_map_[val_str]
            else:
                # Unknown category → use global mean
                encoded[i] = self.global_mean_
        
        return encoded
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
    
    def get_encoding_map(self):
        """Return the category → encoded value mapping."""
        return self.encoding_map_.copy()


class MultiColumnEncoder:
    """
    Apply different encoders to different columns.
    
    Real datasets have mixed column types:
    - Some need one-hot (nominal)
    - Some need ordinal (quality ratings)
    - Some need target encoding (high cardinality)
    
    Example
    -------
    >>> encoder = MultiColumnEncoder({
    ...     'Neighborhood': ('target', {'smoothing': 10}),
    ...     'House.Style': ('onehot', {'drop_first': True}),
    ...     'Overall.Qual': ('ordinal', {'order': list(range(1, 11))})
    ... })
    >>> encoder.fit(X_train, y_train)
    >>> X_encoded = encoder.transform(X_train)
    """
    
    def __init__(self, column_configs):
        """
        Parameters
        ----------
        column_configs : dict
            {column_name: (encoder_type, encoder_params)}
            encoder_type: 'label', 'ordinal', 'onehot', 'target'
        """
        self.column_configs = column_configs
        self.encoders_ = {}
        self.output_columns_ = []
        self._is_fitted = False
    
    def fit(self, X, y=None):
        """Fit all encoders."""
        import pandas as pd
        
        for col, config in self.column_configs.items():
            if col not in X.columns:
                print(f"Warning: Column '{col}' not found, skipping")
                continue
            
            encoder_type, params = config if isinstance(config, tuple) else (config, {})
            
            # Create encoder
            if encoder_type == 'label':
                encoder = LabelEncoder()
                encoder.fit(X[col])
            elif encoder_type == 'ordinal':
                encoder = OrdinalEncoder(**params)
                encoder.fit(X[col])
            elif encoder_type == 'onehot':
                encoder = OneHotEncoder(**params)
                encoder.fit(X[col])
            elif encoder_type == 'target':
                if y is None:
                    raise ValueError("Target encoding requires y")
                encoder = TargetEncoder(**params)
                encoder.fit(X[col], y)
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")
            
            self.encoders_[col] = (encoder_type, encoder)
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """Transform all columns."""
        import pandas as pd
        
        if not self._is_fitted:
            raise RuntimeError("Not fitted.")
        
        # Start with copy of X
        result = X.copy()
        
        for col, (encoder_type, encoder) in self.encoders_.items():
            if col not in X.columns:
                continue
            
            if encoder_type == 'onehot':
                # One-hot creates multiple columns
                encoded = encoder.transform(X[col])
                feature_names = encoder.get_feature_names(col)
                
                # Add new columns
                for i, name in enumerate(feature_names):
                    result[name] = encoded[:, i]
                
                # Drop original column
                result = result.drop(columns=[col])
            else:
                # Other encoders replace in place
                result[col] = encoder.transform(X[col])
        
        return result
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
