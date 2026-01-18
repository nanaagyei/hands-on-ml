# src/data/loader.py
"""
Data loading utilities for the housing price project.

This module provides utilities for loading data from CSV files.
It is used to load the data for the housing price project.

The module provides the following functions:
- load_csv_manual: Load a CSV file without pandas (educational purposes).
- load_ames_housing: Load the Ames housing dataset with proper type handling.
- load_ames_pandas: Load the Ames housing dataset using pandas (recommended for production).

The module also provides a DataFrame class for learning purposes.
It is not for production use - this is educational!
Shows what pandas does under the hood.
"""

import numpy as np
from pathlib import Path


def load_csv_manual(filepath, delimiter=',', has_header=True, na_values=None):
    """
    Load CSV file without pandas (educational purposes).
    
    In production, use pandas. But understanding what happens
    underneath is valuable.
    
    Parameters
    ----------
    filepath : str or Path
        Path to CSV file
    delimiter : str
        Column separator
    has_header : bool
        Whether first row contains column names
    na_values : list
        Values to treat as missing (e.g., ['', 'NA', 'N/A'])
        
    Returns
    -------
    dict with:
        'data': numpy array (all string initially)
        'columns': list of column names
        'dtypes': dict mapping column to inferred dtype
    """
    na_values = na_values or ['', 'NA', 'N/A', 'nan', 'NaN', '.']
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if has_header:
        columns = [col.strip().strip('"') for col in lines[0].strip().split(delimiter)]
        data_lines = lines[1:]
    else:
        data_lines = lines
        columns = [f'col_{i}' for i in range(len(data_lines[0].split(delimiter)))]
    
    # Parse data
    rows = []
    for line in data_lines:
        # Handle quoted fields (simplified)
        row = [val.strip().strip('"') for val in line.strip().split(delimiter)]
        rows.append(row)
    
    # Convert to numpy array (object dtype initially)
    data = np.array(rows, dtype=object)
    
    # Replace NA values with None
    for na_val in na_values:
        data[data == na_val] = None
    
    return {
        'data': data,
        'columns': columns,
        'n_rows': len(rows),
        'n_cols': len(columns)
    }


# For practical use, we'll use a simple CSV parser
# But let's also create a pandas-like interface

class DataFrame:
    """
    Minimal DataFrame implementation for learning.
    
    NOT for production use - this is educational!
    Shows what pandas does under the hood.
    """
    
    def __init__(self, data=None, columns=None):
        """
        Parameters
        ----------
        data : ndarray or dict
            If ndarray: 2D array of data
            If dict: {column_name: array_of_values}
        columns : list, optional
            Column names (required if data is ndarray)
        """
        if isinstance(data, dict):
            self.columns = list(data.keys())
            # Stack columns into 2D array
            arrays = [np.asarray(data[col]) for col in self.columns]
            self._data = np.column_stack(arrays) if arrays else np.array([])
        elif isinstance(data, np.ndarray):
            self._data = data
            self.columns = columns if columns else [f'col_{i}' for i in range(data.shape[1])]
        else:
            self._data = np.array([])
            self.columns = []
        
        self._column_index = {col: i for i, col in enumerate(self.columns)}
    
    @property
    def shape(self):
        return self._data.shape
    
    @property
    def dtypes(self):
        """Infer dtypes for each column."""
        result = {}
        for col in self.columns:
            col_data = self[col]
            result[col] = self._infer_dtype(col_data)
        return result
    
    def _infer_dtype(self, arr):
        """Infer dtype of a column."""
        # Remove None/nan values for inference
        valid = [x for x in arr if x is not None and str(x) != 'nan']
        
        if len(valid) == 0:
            return 'unknown'
        
        # Try numeric
        try:
            floats = [float(x) for x in valid]
            # Check if all integers
            if all(f == int(f) for f in floats):
                return 'int64'
            return 'float64'
        except (ValueError, TypeError):
            return 'object'
    
    def __getitem__(self, key):
        """Get column or slice."""
        if isinstance(key, str):
            # Single column
            idx = self._column_index[key]
            return self._data[:, idx]
        elif isinstance(key, list):
            # Multiple columns
            indices = [self._column_index[k] for k in key]
            return DataFrame(self._data[:, indices], columns=key)
        elif isinstance(key, int):
            # Single row
            return self._data[key, :]
        elif isinstance(key, slice):
            # Row slice
            return DataFrame(self._data[key, :], columns=self.columns)
        else:
            raise KeyError(f"Invalid key: {key}")
    
    def __setitem__(self, key, value):
        """Set column values."""
        if isinstance(key, str):
            if key in self._column_index:
                idx = self._column_index[key]
                self._data[:, idx] = value
            else:
                # Add new column
                self.columns.append(key)
                self._column_index[key] = len(self.columns) - 1
                value = np.asarray(value).reshape(-1, 1)
                self._data = np.hstack([self._data, value])
    
    def head(self, n=5):
        """Return first n rows."""
        return DataFrame(self._data[:n, :], columns=self.columns)
    
    def tail(self, n=5):
        """Return last n rows."""
        return DataFrame(self._data[-n:, :], columns=self.columns)
    
    def describe(self):
        """Generate descriptive statistics for numeric columns."""
        stats = {}
        for col in self.columns:
            col_data = self[col]
            dtype = self._infer_dtype(col_data)
            
            if dtype in ['int64', 'float64']:
                # Convert to float, handling None
                valid = []
                for x in col_data:
                    try:
                        if x is not None:
                            valid.append(float(x))
                    except:
                        pass
                
                if valid:
                    valid = np.array(valid)
                    stats[col] = {
                        'count': len(valid),
                        'mean': np.mean(valid),
                        'std': np.std(valid),
                        'min': np.min(valid),
                        '25%': np.percentile(valid, 25),
                        '50%': np.percentile(valid, 50),
                        '75%': np.percentile(valid, 75),
                        'max': np.max(valid)
                    }
        return stats
    
    def isnull(self):
        """Return boolean array of null values."""
        result = np.zeros(self._data.shape, dtype=bool)
        for i in range(self._data.shape[0]):
            for j in range(self._data.shape[1]):
                val = self._data[i, j]
                result[i, j] = val is None or (isinstance(val, float) and np.isnan(val))
        return DataFrame(result, columns=self.columns)
    
    def null_counts(self):
        """Count nulls per column."""
        null_mask = self.isnull()
        counts = {}
        for col in self.columns:
            counts[col] = np.sum(null_mask[col])
        return counts
    
    def value_counts(self, column):
        """Count unique values in a column."""
        col_data = self[column]
        unique, counts = np.unique(
            [str(x) for x in col_data if x is not None],
            return_counts=True
        )
        return dict(zip(unique, counts))
    
    def to_numpy(self, dtype=None):
        """Convert to numpy array."""
        if dtype:
            return self._data.astype(dtype)
        return self._data.copy()
    
    def copy(self):
        """Return a copy."""
        return DataFrame(self._data.copy(), columns=self.columns.copy())
    
    def __repr__(self):
        """String representation."""
        lines = [f"DataFrame: {self.shape[0]} rows × {self.shape[1]} columns"]
        lines.append("Columns: " + ", ".join(self.columns[:5]) + 
                    (f"... (+{len(self.columns)-5} more)" if len(self.columns) > 5 else ""))
        return "\n".join(lines)
    
    def __len__(self):
        return self.shape[0]


def load_ames_housing(filepath):
    """
    Load Ames Housing dataset with proper type handling.
    
    This is the main entry point for our project.
    """
    # For this project, we'll use pandas for loading
    # (it handles CSV edge cases much better)
    # But we'll convert to our custom structures for learning
    
    import pandas as pd  # Use pandas for robust CSV parsing
    
    df_pandas = pd.read_csv(filepath)
    
    # Convert to our DataFrame for learning purposes
    data_dict = {col: df_pandas[col].values for col in df_pandas.columns}
    
    return DataFrame(data_dict)


# Also provide pandas loader for practical use
def load_ames_pandas(filepath):
    """Load Ames data using pandas (recommended for production)."""
    import pandas as pd
    return pd.read_csv(filepath)