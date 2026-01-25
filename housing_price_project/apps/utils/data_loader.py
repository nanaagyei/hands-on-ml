"""
Data loading utilities for Streamlit apps.
"""
import pandas as pd
import pickle
from pathlib import Path
import sys
import streamlit as st
from apps.config import PROCESSED_DATA, FEATURED_DATA, MODEL_PATH

# Ensure linear_regression module is available for unpickling
_current_file = Path(__file__).resolve()
_apps_dir = _current_file.parent.parent  # apps
_project_root = _apps_dir.parent  # housing_price_project
_main_root = _project_root.parent  # hands-on-ml

if str(_main_root) not in sys.path:
    sys.path.insert(0, str(_main_root))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


@st.cache_data(ttl=3600)
def load_house_data():
    """Load the cleaned house data."""
    try:
        df = pd.read_csv(PROCESSED_DATA)
        return df
    except Exception as e:
        st.error(f"Error loading house data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_featured_data():
    """Load the featured/engineered data."""
    try:
        df = pd.read_csv(FEATURED_DATA)
        return df
    except Exception as e:
        st.warning(f"Could not load featured data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def get_model_info():
    """Load model information."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        return {
            'model_type': model_data.get('model_name', 'Unknown'),
            'n_features': len(model_data.get('feature_names', [])),
            'feature_names': model_data.get('feature_names', []),
            'params': model_data.get('params', {}),
            'cv_score': model_data.get('cv_score', None),
            'test_metrics': model_data.get('test_metrics', {}),
        }
    except Exception as e:
        st.warning(f"Could not load model info: {e}")
        return {}


def get_neighborhoods(df):
    """Get unique neighborhoods from data."""
    if 'Neighborhood' in df.columns:
        return sorted(df['Neighborhood'].dropna().unique().tolist())
    return []


def get_house_types(df):
    """Get unique house types from data."""
    if 'House.Style' in df.columns:
        return sorted(df['House.Style'].dropna().unique().tolist())
    return []


def get_zoning_types(df):
    """Get unique zoning types from data."""
    if 'MS.Zoning' in df.columns:
        return sorted(df['MS.Zoning'].dropna().unique().tolist())
    return []


def filter_houses(df, filters):
    """Filter houses based on criteria."""
    filtered = df.copy()
    
    if filters.get('neighborhood') and filters['neighborhood'] != 'All':
        filtered = filtered[filtered['Neighborhood'] == filters['neighborhood']]
    
    if filters.get('min_price'):
        filtered = filtered[filtered['price'] >= filters['min_price']]
    
    if filters.get('max_price'):
        filtered = filtered[filtered['price'] <= filters['max_price']]
    
    if filters.get('min_year'):
        filtered = filtered[filtered['Year.Built'] >= filters['min_year']]
    
    if filters.get('max_year'):
        filtered = filtered[filtered['Year.Built'] <= filters['max_year']]
    
    if filters.get('house_style') and filters['house_style'] != 'All':
        filtered = filtered[filtered['House.Style'] == filters['house_style']]
    
    if filters.get('search_term'):
        search = filters['search_term'].lower()
        # Search in multiple columns
        mask = False
        for col in ['Neighborhood', 'MS.Zoning', 'House.Style', 'Exterior.1st']:
            if col in filtered.columns:
                mask |= filtered[col].astype(str).str.lower().str.contains(search, na=False)
        filtered = filtered[mask]
    
    return filtered


def get_similar_houses(df, house_id, n=5):
    """Get similar houses based on key features."""
    if df.empty:
        return pd.DataFrame()
    
    # Reset index if needed
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)
    
    # Convert house_id to int if it's not already
    try:
        house_id = int(house_id)
    except:
        pass
    
    if house_id not in df.index:
        return df.head(n)
    
    house = df.loc[house_id]
    
    # Calculate similarity based on key features
    similarity_cols = ['area', 'Overall.Qual', 'Year.Built', 'Bedroom.AbvGr', 'Full.Bath']
    available_cols = [col for col in similarity_cols if col in df.columns]
    
    if not available_cols:
        return df.head(n)
    
    # Calculate distance
    distances = []
    for idx, row in df.iterrows():
        if idx == house_id:
            continue
        distance = 0
        for col in available_cols:
            if pd.notna(house[col]) and pd.notna(row[col]):
                # Normalize by range
                col_range = df[col].max() - df[col].min()
                if col_range > 0:
                    distance += abs(house[col] - row[col]) / col_range
        distances.append((idx, distance))
    
    # Sort by distance and get top N
    distances.sort(key=lambda x: x[1])
    similar_indices = [idx for idx, _ in distances[:n]]
    
    return df.loc[similar_indices]


def get_neighborhood_stats(df):
    """Get statistics per neighborhood."""
    if 'Neighborhood' not in df.columns or 'price' not in df.columns:
        return pd.DataFrame()
    
    stats = df.groupby('Neighborhood')['price'].agg([
        'count', 'mean', 'median', 'min', 'max', 'std'
    ]).round(2)
    stats.columns = ['Count', 'Mean Price', 'Median Price', 'Min Price', 'Max Price', 'Std Dev']
    stats = stats.sort_values('Mean Price', ascending=False)
    
    return stats


@st.cache_data(ttl=3600)
def get_feature_defaults():
    """
    Get default feature values (medians) from featured training data.
    This provides better defaults than scaler mean for missing features.
    
    Returns
    -------
    dict : {feature_name: median_value}
    """
    try:
        featured_df = load_featured_data()
        if featured_df.empty:
            return {}
        
        # Exclude target columns
        feature_cols = [col for col in featured_df.columns if col not in ['price', 'log_price']]
        
        # Compute medians for each feature
        defaults = {}
        for col in feature_cols:
            median_val = featured_df[col].median()
            defaults[col] = float(median_val) if pd.notna(median_val) else 0.0
        
        return defaults
    except Exception as e:
        st.warning(f"Could not load feature defaults: {e}")
        return {}


@st.cache_data(ttl=3600)
def get_categorical_encodings():
    """
    Get categorical value encodings from featured data.
    Returns a dict with encoding information.
    """
    try:
        featured_df = load_featured_data()
        if featured_df.empty:
            return {}
        
        # Get the original cleaned data for categorical values
        cleaned_df = load_house_data()
        if cleaned_df.empty:
            return {}
        
        encodings = {}
        
        # For Neighborhood - it's target encoded (single numeric column)
        if 'Neighborhood' in cleaned_df.columns and 'Neighborhood' in featured_df.columns:
            # Create mapping from original to encoded value
            neighborhood_map = {}
            seen = set()
            for idx in cleaned_df.index:
                if idx in featured_df.index:
                    orig_val = cleaned_df.loc[idx, 'Neighborhood']
                    encoded_val = featured_df.loc[idx, 'Neighborhood']
                    if pd.notna(orig_val) and pd.notna(encoded_val):
                        key = str(orig_val)
                        if key not in seen:
                            neighborhood_map[key] = float(encoded_val)
                            seen.add(key)
            encodings['Neighborhood'] = {
                'type': 'target',
                'mapping': neighborhood_map
            }
        
        # For MS.Zoning - it's one-hot encoded (multiple binary columns)
        if 'MS.Zoning' in cleaned_df.columns:
            # Find all MS.Zoning columns in featured data
            zoning_cols = [col for col in featured_df.columns if col.startswith('MS.Zoning_')]
            if zoning_cols:
                # Create mapping from original value to column name
                # Match by checking if the original value appears in the column name
                zoning_map = {}
                for orig_val in cleaned_df['MS.Zoning'].dropna().unique():
                    orig_str = str(orig_val).strip()
                    # Try to find matching column - handle variations like "C (all)" -> "MS.Zoning_C (all)"
                    # Also handle "A (agr)" which might not exist in featured data
                    matching_col = None
                    for col in zoning_cols:
                        # Extract the value part after "MS.Zoning_"
                        col_value = col.replace('MS.Zoning_', '')
                        # Normalize both for comparison
                        orig_normalized = orig_str.replace(' ', '_').replace('(', '').replace(')', '').replace('&', '').upper()
                        col_normalized = col_value.replace(' ', '_').replace('(', '').replace(')', '').replace('&', '').upper()
                        # Try exact match first
                        if orig_normalized == col_normalized:
                            matching_col = col
                            break
                        # Try partial match
                        if orig_str in col_value or col_value.startswith(orig_str):
                            matching_col = col
                            break
                        # Special case: "A (agr)" might map to nothing, but we'll try to find closest match
                        # For now, if no match found, it will be None (handled by fallback)
                    if matching_col:
                        zoning_map[orig_str] = matching_col
                    # Note: "A (agr)" might not have a match - this is expected if it wasn't in training data
                encodings['MS.Zoning'] = {
                    'type': 'onehot',
                    'columns': zoning_cols,
                    'mapping': zoning_map
                }
        
        return encodings
    except Exception as e:
        st.warning(f"Could not load categorical encodings: {e}")
        return {}


def encode_categorical_feature(feature_name, feature_value, encodings=None):
    """
    Encode a categorical feature value.
    
    For target-encoded features: returns the numeric value
    For one-hot encoded features: returns a dict of {column_name: 1.0}
    
    Parameters
    ----------
    feature_name : str
        Name of the feature
    feature_value : str
        Raw categorical value
    encodings : dict, optional
        Pre-loaded encodings (will load if not provided)
        
    Returns
    -------
    encoded_value : float, dict, or None
        For target encoding: float value
        For one-hot encoding: dict of {column_name: 1.0}
        None if not found
    """
    if encodings is None:
        encodings = get_categorical_encodings()
    
    if feature_name in encodings:
        encoding_info = encodings[feature_name]
        encoding_type = encoding_info.get('type')
        mapping = encoding_info.get('mapping', {})
        
        if encoding_type == 'target':
            # Return the numeric encoded value
            return mapping.get(str(feature_value))
        elif encoding_type == 'onehot':
            # Return dict with the column name set to 1.0
            col_name = mapping.get(str(feature_value))
            if col_name:
                return {col_name: 1.0}
    
    return None
