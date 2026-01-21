"""
Data Cleaning Module for Ames Housing Dataset.

PRINCIPLES:
1. Document every transformation
2. Make cleaning reproducible (save parameters)
3. Separate fit (learn from train) from transform (apply to any data)
4. Never modify raw data — create new cleaned version
"""
import numpy as np
from collections import defaultdict


class SimpleImputer:
    """
    Impute missing values using simple strategies.
    
    This is what sklearn.impute.SimpleImputer does under the hood.
    
    Strategies:
    - 'mean': Replace with column mean (numeric only)
    - 'median': Replace with column median (numeric only)  
    - 'most_frequent': Replace with mode (works for all types)
    - 'constant': Replace with a fixed value
    
    Parameters
    ----------
    strategy : str, default='mean'
        Imputation strategy
    fill_value : any, optional
        Value to use when strategy='constant'
    
    Example
    -------
    >>> imputer = SimpleImputer(strategy='median')
    >>> imputer.fit(X_train)
    >>> X_train_clean = imputer.transform(X_train)
    >>> X_test_clean = imputer.transform(X_test)  # Uses train statistics!
    """

    def __init__(self, strategy='mean', fill_value=None):
        valid_strategies = ['mean', 'median', 'most_frequent', 'constant']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of: {valid_strategies}")
        
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        self._is_fitted = False
    
    def fit(self, X):
        """
        Compute the imputation values from training data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data with potential missing values
            
        Returns
        -------
        self : returns fitted SimpleImputer
        """
        
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_features = X.shape[1]
        self.statistics_ = np.zeros(n_features)
        for col_idx in range(n_features):
            col = X[:, col_idx]
            
            # Get non-missing values
            # Handle both np.nan and None
            valid_mask = ~self._isnan(col)
            valid_values = col[valid_mask]
            
            if len(valid_values) == 0:
                # All values missing — use fill_value or 0
                self.statistics_[col_idx] = self.fill_value if self.fill_value is not None else 0
                continue
            
            if self.strategy == 'mean':
                self.statistics_[col_idx] = np.mean(valid_values.astype(float))
            
            elif self.strategy == 'median':
                self.statistics_[col_idx] = np.median(valid_values.astype(float))
            
            elif self.strategy == 'most_frequent':
                # Find mode (most common value)
                values, counts = np.unique(valid_values, return_counts=True)
                self.statistics_[col_idx] = values[np.argmax(counts)]
            
            elif self.strategy == 'constant':
                self.statistics_[col_idx] = self.fill_value
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Impute missing values in new data using the statistics from training data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to impute
            
        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            Imputed data
        """

        if not self._is_fitted:
            raise RuntimeError("Imputer not fitted. Call 'fit()' with training data first.")
        
        X = np.asarray(X, dtype=float)  # Convert to float for imputation
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_imputed = X.copy()
        
        for col_idx in range(X.shape[1]):
            col = X_imputed[:, col_idx]
            missing_mask = self._isnan(col)
            col[missing_mask] = self.statistics_[col_idx]
        
        return X_imputed
    

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_imputed : ndarray of shape (n_samples, n_features)
            Imputed data
        """

        return self.fit(X).transform(X)
    
    def _isnan(self, arr):
        """Check for NaN values, handling mixed types."""
        result = np.zeros(len(arr), dtype=bool)
        for i, val in enumerate(arr):
            if val is None:
                result[i] = True
            else:
                try:
                    result[i] = np.isnan(float(val))
                except (ValueError, TypeError):
                    result[i] = False
        return result
    
    def get_params(self):
        """Return fitted parameters for inspection/saving."""
        return {
            'strategy': self.strategy,
            'fill_value': self.fill_value,
            'statistics': self.statistics_
        }


class ColumnImputer:
    """
    Apply different imputation strategies to different columns.
    
    Real datasets need different strategies per column:
    - Numeric: mean/median
    - Categorical: most_frequent or constant
    - Domain-specific: custom rules
    
    Parameters
    ----------
    strategies : dict
        Mapping of column_index -> (strategy, fill_value)
        
    Example
    -------
    >>> imputer = ColumnImputer({
    ...     0: ('median', None),      # Column 0: median
    ...     1: ('constant', 'None'),  # Column 1: fill with 'None'
    ...     2: ('most_frequent', None) # Column 2: mode
    ... })
    """

    def __init__(self, strategies):
        """
        Parameters
        ----------
        strategies : dict
            {column_index: (strategy_name, fill_value)}
            or {column_index: strategy_name} for default fill_value
        """
        self.strategies = {}
        for col_idx, strategy in strategies.items():
            if isinstance(strategy, tuple):
                self.strategies[col_idx] = strategy
            else:
                self.strategies[col_idx] = (strategy, None)
        
        self.statistics_ = {}
        self._is_fitted = False
    
    def fit(self, X):
        """Compute imputation values for each column."""
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        for col_idx, (strategy, fill_value) in self.strategies.items():
            if col_idx >= X.shape[1]:
                continue
            
            col = X[:, col_idx]
            valid_mask = self._get_valid_mask(col)
            valid_values = col[valid_mask]
            
            if len(valid_values) == 0:
                self.statistics_[col_idx] = fill_value if fill_value is not None else 0
                continue
            
            if strategy == 'mean':
                self.statistics_[col_idx] = np.mean(valid_values.astype(float))
            elif strategy == 'median':
                self.statistics_[col_idx] = np.median(valid_values.astype(float))
            elif strategy == 'most_frequent':
                values, counts = np.unique(valid_values.astype(str), return_counts=True)
                self.statistics_[col_idx] = values[np.argmax(counts)]
            elif strategy == 'constant':
                self.statistics_[col_idx] = fill_value
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        
        self._is_fitted = True
        return self
    

    def transform(self, X):
        """Apply imputation."""
        if not self._is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_imputed = X.copy()
        
        for col_idx, fill_value in self.statistics_.items():
            if col_idx >= X_imputed.shape[1]:
                continue
            
            col = X_imputed[:, col_idx]
            missing_mask = ~self._get_valid_mask(col)
            
            # Handle type conversion carefully
            if np.any(missing_mask):
                X_imputed[missing_mask, col_idx] = fill_value
        
        return X_imputed
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    

    def _get_valid_mask(self, arr):
        """Identify non-missing values."""
        result = np.ones(len(arr), dtype=bool)
        for i, val in enumerate(arr):
            if val is None:
                result[i] = False
            elif isinstance(val, float) and np.isnan(val):
                result[i] = False
            elif isinstance(val, str) and val.lower() in ['nan', 'none', '']:
                result[i] = False
        return result

class OutlierHandler:
    """
    Handle outliers using various strategies.
    
    Methods:
    - 'clip': Cap values at boundaries (winsorization)
    - 'remove': Flag outliers for removal (returns mask)
    - 'nan': Replace outliers with NaN (then impute)
    
    Detection methods:
    - 'iqr': Interquartile range method
    - 'zscore': Standard deviation method
    - 'percentile': Hard percentile cutoffs
    """
    
    def __init__(self, method='iqr', threshold=1.5, strategy='clip'):
        """
        Parameters
        ----------
        method : str
            'iqr' (threshold = IQR multiplier, default 1.5)
            'zscore' (threshold = number of std devs, default 3)
            'percentile' (threshold = (lower_pct, upper_pct), e.g., (1, 99))
        threshold : float or tuple
            Detection threshold
        strategy : str
            'clip': Cap outliers at boundaries
            'remove': Return mask for removal
            'nan': Replace with NaN
        """
        self.method = method
        self.threshold = threshold
        self.strategy = strategy
        
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self._is_fitted = False
    
    def fit(self, X):
        """Learn outlier boundaries from training data."""
        X = np.asarray(X, dtype=float)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_features = X.shape[1]
        self.lower_bounds_ = np.zeros(n_features)
        self.upper_bounds_ = np.zeros(n_features)
        
        for col_idx in range(n_features):
            col = X[:, col_idx]
            valid = col[~np.isnan(col)]
            
            if len(valid) == 0:
                self.lower_bounds_[col_idx] = -np.inf
                self.upper_bounds_[col_idx] = np.inf
                continue
            
            if self.method == 'iqr':
                Q1 = np.percentile(valid, 25)
                Q3 = np.percentile(valid, 75)
                IQR = Q3 - Q1
                self.lower_bounds_[col_idx] = Q1 - self.threshold * IQR
                self.upper_bounds_[col_idx] = Q3 + self.threshold * IQR
            
            elif self.method == 'zscore':
                mean = np.mean(valid)
                std = np.std(valid)
                self.lower_bounds_[col_idx] = mean - self.threshold * std
                self.upper_bounds_[col_idx] = mean + self.threshold * std
            
            elif self.method == 'percentile':
                lower_pct, upper_pct = self.threshold
                self.lower_bounds_[col_idx] = np.percentile(valid, lower_pct)
                self.upper_bounds_[col_idx] = np.percentile(valid, upper_pct)
            
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """Apply outlier handling."""
        if not self._is_fitted:
            raise RuntimeError("Not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_handled = X.copy()
        
        if self.strategy == 'clip':
            for col_idx in range(X.shape[1]):
                X_handled[:, col_idx] = np.clip(
                    X_handled[:, col_idx],
                    self.lower_bounds_[col_idx],
                    self.upper_bounds_[col_idx]
                )
        
        elif self.strategy == 'nan':
            for col_idx in range(X.shape[1]):
                col = X_handled[:, col_idx]
                outlier_mask = (col < self.lower_bounds_[col_idx]) | \
                               (col > self.upper_bounds_[col_idx])
                col[outlier_mask] = np.nan
        
        elif self.strategy == 'remove':
            # Return boolean mask instead of transformed data
            outlier_mask = np.zeros(X.shape[0], dtype=bool)
            for col_idx in range(X.shape[1]):
                col = X[:, col_idx]
                outlier_mask |= (col < self.lower_bounds_[col_idx]) | \
                                (col > self.upper_bounds_[col_idx])
            return ~outlier_mask  # Return mask of rows to KEEP
        
        return X_handled
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    
    def get_bounds(self):
        """Return the computed boundaries."""
        return {
            'lower': self.lower_bounds_,
            'upper': self.upper_bounds_
        }


class DomainOutlierFilter:
    """
    Filter outliers based on domain-specific rules.
    
    This is where your insight about "Partial" sales comes in!
    Statistical methods can't catch domain violations.
    
    Example
    -------
    >>> filter = DomainOutlierFilter(rules=[
    ...     ('Sale.Condition', lambda x: x != 'Partial'),  # Remove partial sales
    ...     ('price', lambda x: x > 10000),  # Price must be > $10k
    ... ])
    >>> mask = filter.get_mask(df)
    >>> df_clean = df[mask]
    """
    
    def __init__(self, rules=None):
        """
        Parameters
        ----------
        rules : list of tuples
            Each tuple: (column_name, filter_function)
            filter_function returns True for rows to KEEP
        """
        self.rules = rules or []
        self.removal_stats_ = {}
    
    def add_rule(self, column, condition, description=None):
        """Add a filtering rule."""
        self.rules.append((column, condition, description or f"Filter on {column}"))
        return self
    
    def get_mask(self, df):
        """
        Apply all rules and return mask of rows to keep.
        
        Parameters
        ----------
        df : DataFrame (pandas or our custom)
            Data to filter
            
        Returns
        -------
        mask : boolean array
            True for rows to keep
        """
        import pandas as pd
        
        n_rows = len(df)
        mask = np.ones(n_rows, dtype=bool)
        
        self.removal_stats_ = {'original_rows': n_rows}
        
        for rule in self.rules:
            if len(rule) == 2:
                col, condition = rule
                desc = f"Filter on {col}"
            else:
                col, condition, desc = rule
            
            if col not in df.columns:
                print(f"  Warning: Column '{col}' not found, skipping rule")
                continue
            
            # Apply condition
            col_data = df[col].values if hasattr(df[col], 'values') else df[col]
            rule_mask = np.array([condition(x) for x in col_data])
            
            removed = np.sum(mask & ~rule_mask)
            self.removal_stats_[desc] = removed
            
            mask &= rule_mask
        
        self.removal_stats_['final_rows'] = np.sum(mask)
        self.removal_stats_['total_removed'] = n_rows - np.sum(mask)
        
        return mask
    
    def fit_transform(self, df):
        """Apply filter and return cleaned dataframe."""
        mask = self.get_mask(df)
        
        # Print summary
        print("\nDomain Filter Summary:")
        print("-" * 40)
        for key, value in self.removal_stats_.items():
            print(f"  {key}: {value}")
        
        if hasattr(df, 'iloc'):
            return df.iloc[mask]
        else:
            return df[mask]

class DataCleaningPipeline:
    """
    A reproducible data cleaning pipeline.
    
    Chains together multiple cleaning steps and tracks all transformations.
    Can be saved/loaded for production use.
    
    Example
    -------
    >>> pipeline = DataCleaningPipeline()
    >>> pipeline.add_step('remove_outliers', domain_filter)
    >>> pipeline.add_step('impute_numeric', numeric_imputer)
    >>> pipeline.add_step('impute_categorical', cat_imputer)
    >>> 
    >>> df_clean = pipeline.fit_transform(df_train)
    >>> df_test_clean = pipeline.transform(df_test)
    """
    
    def __init__(self):
        self.steps = []
        self.step_names = []
        self._is_fitted = False
        self.log_ = []
    
    def add_step(self, name, transformer, columns=None):
        """
        Add a cleaning step.
        
        Parameters
        ----------
        name : str
            Step name for logging
        transformer : object
            Must have fit() and transform() methods
        columns : list, optional
            Columns to apply this step to (None = all)
        """
        self.steps.append((name, transformer, columns))
        self.step_names.append(name)
        return self
    
    def fit(self, X, y=None):
        """Fit all transformers."""
        self.log_ = []
        
        for name, transformer, columns in self.steps:
            self.log_.append(f"Fitting: {name}")
            
            if columns is not None:
                # Fit only on specified columns
                X_subset = X[columns] if hasattr(X, '__getitem__') else X[:, columns]
                transformer.fit(X_subset)
            else:
                transformer.fit(X)
        
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """Apply all transformations."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        
        X_transformed = X.copy() if hasattr(X, 'copy') else np.array(X)
        
        for name, transformer, columns in self.steps:
            self.log_.append(f"Transforming: {name}")
            
            if columns is not None:
                X_subset = X_transformed[columns]
                X_transformed[columns] = transformer.transform(X_subset)
            else:
                X_transformed = transformer.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_log(self):
        """Return transformation log."""
        return self.log_




def create_ames_cleaning_config():
    """
    Create cleaning configuration specific to Ames Housing dataset.
    
    This encodes domain knowledge about the dataset!
    """
    
    config = {
        # Columns where NA means "None" (no feature present)
        'na_means_none': [
            'Alley',           # No alley access
            'Bsmt.Qual',       # No basement
            'Bsmt.Cond',
            'Bsmt.Exposure',
            'BsmtFin.Type.1',
            'BsmtFin.Type.2',
            'Fireplace.Qu',    # No fireplace
            'Garage.Type',     # No garage
            'Garage.Finish',
            'Garage.Qual',
            'Garage.Cond',
            'Pool.QC',         # No pool
            'Fence',           # No fence
            'Misc.Feature',    # No misc feature
        ],
        
        # Columns where NA means 0 (no feature present)
        'na_means_zero': [
            'Garage.Yr.Blt',   # No garage
            'Garage.Cars',
            'Garage.Area',
            'Bsmt.Full.Bath',  # No basement
            'Bsmt.Half.Bath',
            'BsmtFin.SF.1',
            'BsmtFin.SF.2',
            'Bsmt.Unf.SF',
            'Total.Bsmt.SF',
            'Mas.Vnr.Area',    # No masonry veneer
        ],
        
        # Numeric columns to impute with median
        'impute_median': [
            'Lot.Frontage',    # Impute with neighborhood median ideally
        ],
        
        # Categorical columns to impute with mode
        'impute_mode': [
            'Mas.Vnr.Type',
            'Electrical',
            'MS.Zoning',
        ],
        
        # Rows to filter out (domain rules)
        'filter_rules': [
            ('Sale.Condition', lambda x: x != 'Partial', 'Remove partial sales'),
            ('price', lambda x: x is not None and x > 0, 'Remove zero/null prices'),
        ],
        
        # Columns to drop (too much missing or not useful)
        'drop_columns': [
            'Order',           # Just row number
            'PID',             # Parcel ID (not a feature)
        ],
    }
    
    return config


def clean_ames_data(df, config=None):
    """
    Apply full cleaning pipeline to Ames Housing data.
    
    Parameters
    ----------
    df : pandas DataFrame
        Raw Ames housing data
    config : dict, optional
        Cleaning configuration (default: create_ames_cleaning_config())
    
    Returns
    -------
    df_clean : pandas DataFrame
        Cleaned data
    cleaning_report : dict
        Summary of cleaning operations
    """
    import pandas as pd
    
    if config is None:
        config = create_ames_cleaning_config()
    
    report = {
        'original_rows': len(df),
        'original_cols': len(df.columns),
        'steps': []
    }
    
    df_clean = df.copy()
    
    # Step 1: Drop unnecessary columns
    cols_to_drop = [c for c in config.get('drop_columns', []) if c in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        report['steps'].append(f"Dropped columns: {cols_to_drop}")
    
    # Step 2: Apply domain filters
    filter_rules = config.get('filter_rules', [])
    if filter_rules:
        domain_filter = DomainOutlierFilter(filter_rules)
        mask = domain_filter.get_mask(df_clean)
        n_removed = len(df_clean) - mask.sum()
        df_clean = df_clean[mask].reset_index(drop=True)
        report['steps'].append(f"Domain filter removed {n_removed} rows")
        report['domain_filter_stats'] = domain_filter.removal_stats_
    
    # Step 3: Fill NA with 'None' for specific columns
    for col in config.get('na_means_none', []):
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('None')
    report['steps'].append(f"Filled NA with 'None' for {len(config.get('na_means_none', []))} columns")
    
    # Step 4: Fill NA with 0 for specific columns
    for col in config.get('na_means_zero', []):
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    report['steps'].append(f"Filled NA with 0 for {len(config.get('na_means_zero', []))} columns")
    
    # Step 5: Impute with median
    for col in config.get('impute_median', []):
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            report['steps'].append(f"Imputed {col} with median={median_val:.2f}")
    
    # Step 6: Impute with mode
    for col in config.get('impute_mode', []):
        if col in df_clean.columns:
            mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
            df_clean[col] = df_clean[col].fillna(mode_val)
            report['steps'].append(f"Imputed {col} with mode='{mode_val}'")
    
    # Final stats
    report['final_rows'] = len(df_clean)
    report['final_cols'] = len(df_clean.columns)
    report['remaining_nulls'] = df_clean.isnull().sum().sum()
    
    return df_clean, report


def print_cleaning_report(report):
    """Pretty print the cleaning report."""
    print("\n" + "="*60)
    print("DATA CLEANING REPORT")
    print("="*60)
    print(f"\nOriginal: {report['original_rows']} rows × {report['original_cols']} columns")
    print(f"Final:    {report['final_rows']} rows × {report['final_cols']} columns")
    print(f"Rows removed: {report['original_rows'] - report['final_rows']}")
    print(f"Remaining nulls: {report['remaining_nulls']}")
    
    print("\nSteps performed:")
    for step in report['steps']:
        print(f"  • {step}")
    
    if 'domain_filter_stats' in report:
        print("\nDomain filter details:")
        for key, val in report['domain_filter_stats'].items():
            print(f"    {key}: {val}")
