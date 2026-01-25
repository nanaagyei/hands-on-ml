# src/serving/api.py
"""
Simple Flask API for house price predictions.

Run with: python -m src.serving.api
Access at: http://localhost:5000

Endpoints:
- GET  /health         : Health check
- GET  /model/info     : Model information
- POST /predict        : Single prediction
- POST /predict/batch  : Batch predictions
"""

from flask import Flask, request, jsonify
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root to path
_project_root = Path(__file__).parent.parent.parent  # housing_price_project
_main_root = _project_root.parent  # hands-on-ml

# Ensure both paths are available for imports
if str(_main_root) not in sys.path:
    sys.path.insert(0, str(_main_root))

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.serving.predictor import HousePricePredictor

app = Flask(__name__)

# Load model at startup
MODEL_PATH = Path("models/final_model.pkl")
FEATURED_DATA_PATH = _project_root / "data" / "features" / "ames_featured.csv"
predictor = None

# Cache for feature defaults (computed from featured training data)
_feature_defaults_cache = None


def get_feature_defaults_from_data():
    """
    Load feature medians from featured training data.
    These are better defaults than scaler mean because they're actual feature values.
    
    Returns
    -------
    dict : {feature_name: median_value} or None if unavailable
    """
    global _feature_defaults_cache
    
    if _feature_defaults_cache is not None:
        return _feature_defaults_cache
    
    try:
        import pandas as pd
        
        if not FEATURED_DATA_PATH.exists():
            return None
        
        featured_df = pd.read_csv(FEATURED_DATA_PATH)
        if featured_df.empty:
            return None
        
        # Exclude target columns
        feature_cols = [col for col in featured_df.columns if col not in ['price', 'log_price']]
        
        # Compute medians for each feature
        defaults = {}
        for col in feature_cols:
            median_val = featured_df[col].median()
            defaults[col] = float(median_val) if pd.notna(median_val) else 0.0
        
        _feature_defaults_cache = defaults
        return defaults
    except Exception:
        return None


def get_typical_house_features():
    """
    Get a complete feature vector for a typical house from training data.
    This provides a coherent baseline that can be overridden by user inputs.
    
    Returns
    -------
    dict : {feature_name: typical_value} or None if unavailable
    """
    try:
        import pandas as pd
        
        if not FEATURED_DATA_PATH.exists():
            return None
        
        featured_df = pd.read_csv(FEATURED_DATA_PATH)
        if featured_df.empty:
            return None
        
        # Exclude target columns
        feature_cols = [col for col in featured_df.columns if col not in ['price', 'log_price']]
        
        # Get the median house (middle row by price)
        median_price_idx = featured_df['price'].median()
        closest_idx = (featured_df['price'] - median_price_idx).abs().idxmin()
        
        # Get all features for this typical house
        typical_features = {}
        for col in feature_cols:
            val = featured_df.loc[closest_idx, col]
            typical_features[col] = float(val) if pd.notna(val) else 0.0
        
        return typical_features
    except Exception as e:
        print(f"Error loading typical house features: {e}")
        return None


def find_similar_houses_and_fill_features(user_features, model_feature_names, n_similar=10):
    """
    Find similar houses in training data based on provided features,
    then use their feature values to fill in missing features.
    
    This creates a more coherent feature vector by using actual houses
    that are similar to what the user described.
    
    Parameters
    ----------
    user_features : dict
        Features provided by the user
    model_feature_names : list
        All feature names expected by the model
    n_similar : int
        Number of similar houses to use for averaging
        
    Returns
    -------
    dict : Complete feature vector with missing features filled from similar houses
    """
    try:
        import pandas as pd
        
        if not FEATURED_DATA_PATH.exists():
            return None
        
        featured_df = pd.read_csv(FEATURED_DATA_PATH)
        if featured_df.empty:
            return None
        
        # Exclude target columns
        feature_cols = [col for col in featured_df.columns if col not in ['price', 'log_price']]
        
        # Find features that user provided and exist in training data
        # Exclude Neighborhood from similarity calculation (it's target-encoded and can be extreme)
        # We'll use it for filling but not for finding similar houses
        important_features = ['area', 'Overall.Qual', 'Year.Built', 'Lot.Area']
        provided_feature_names = [f for f in user_features.keys() if f in feature_cols]
        similarity_features = [f for f in provided_feature_names if f != 'Neighborhood']
        
        
        if not similarity_features:
            # If only Neighborhood was provided, use typical house
            return get_typical_house_features()
        
        # Calculate similarity based on provided features (excluding Neighborhood)
        # Use weighted distance - important features get more weight
        similarities = []
        
        # Pre-compute feature statistics for normalization
        feat_stats = {}
        for feat_name in similarity_features:
            if feat_name in featured_df.columns:
                feat_stats[feat_name] = {
                    'min': featured_df[feat_name].min(),
                    'max': featured_df[feat_name].max(),
                    'std': featured_df[feat_name].std(),
                    'weight': 2.0 if feat_name in important_features else 1.0
                }
        
        for idx, row in featured_df.iterrows():
            # Calculate weighted distance for each similarity feature (excluding Neighborhood)
            weighted_distances = []
            for feat_name in similarity_features:
                if feat_name not in feat_stats:
                    continue
                    
                if feat_name in row.index:
                    user_val = float(user_features[feat_name])
                    train_val = row[feat_name]
                    
                    # Handle NaN
                    if pd.isna(train_val):
                        continue
                    
                    train_val = float(train_val)
                    stats = feat_stats[feat_name]
                    
                    # Normalize by standard deviation (more robust than range)
                    if stats['std'] > 0:
                        normalized_dist = abs(user_val - train_val) / stats['std']
                    else:
                        normalized_dist = abs(user_val - train_val)
                    
                    # Apply weight (important features matter more)
                    weighted_dist = normalized_dist * stats['weight']
                    weighted_distances.append(weighted_dist)
            
            if weighted_distances:
                # Average weighted distance
                avg_distance = np.mean(weighted_distances)
                similarities.append((idx, avg_distance))
        
        if not similarities:
            # No similar houses found, use typical house
            return get_typical_house_features()
        
        # Sort by similarity (lower distance = more similar)
        similarities.sort(key=lambda x: x[1])
        
        # Get top n_similar houses
        top_indices = [idx for idx, _ in similarities[:n_similar]]
        
        # Average feature values from similar houses (use median for robustness)
        # Build complete_features for ALL model features, not just feature_cols
        complete_features = {}
        
        # First, process features that exist in both training data and model
        for feat_name in feature_cols:
            if feat_name in user_features:
                # For Neighborhood (target-encoded), always use similar house values
                # Target-encoded values can be extreme and misleading for similarity
                if feat_name == 'Neighborhood':
                    # Always get Neighborhood from similar houses (not user value)
                    # Use 95th percentile as max to avoid extreme values
                    feat_p95 = featured_df[feat_name].quantile(0.95)
                    feat_p5 = featured_df[feat_name].quantile(0.05)
                    feat_median = featured_df[feat_name].median()
                    
                    values = []
                    for idx in top_indices:
                        val = featured_df.loc[idx, feat_name]
                        if pd.notna(val):
                            val_float = float(val)
                            # Only use values within 5th-95th percentile (very conservative)
                            if feat_p5 <= val_float <= feat_p95:
                                values.append(val_float)
                    
                    if values:
                        # Use median for robustness
                        complete_features[feat_name] = float(np.median(values))
                        print(f"  Using Neighborhood from similar houses: {complete_features[feat_name]:.2f} (user provided: {user_features[feat_name]:.2f}, range: {feat_p5:.2f}-{feat_p95:.2f})")
                    else:
                        # Use overall median if no values in range
                        complete_features[feat_name] = float(feat_median) if pd.notna(feat_median) else 0.0
                else:
                    # For other features, use user-provided value (but check for extreme values)
                    user_val = float(user_features[feat_name])
                    # For extreme values, use similar house values instead
                    if abs(user_val) > 100000:
                        # Get values from similar houses
                        values = []
                        for idx in top_indices:
                            val = featured_df.loc[idx, feat_name]
                            if pd.notna(val):
                                val_float = float(val)
                                # Only include reasonable values (within 3 std devs of median)
                                feat_median = featured_df[feat_name].median()
                                feat_std = featured_df[feat_name].std()
                                if feat_std > 0:
                                    z_score = abs(val_float - feat_median) / feat_std
                                    if z_score < 3:  # Within 3 standard deviations
                                        values.append(val_float)
                                else:
                                    values.append(val_float)
                        
                        if values:
                            # Use median for robustness (less affected by outliers)
                            complete_features[feat_name] = float(np.median(values))
                        else:
                            # Fallback to overall median
                            median_val = featured_df[feat_name].median()
                            complete_features[feat_name] = float(median_val) if pd.notna(median_val) else 0.0
                    else:
                        complete_features[feat_name] = user_val
            else:
                # Get values from similar houses
                values = []
                for idx in top_indices:
                    val = featured_df.loc[idx, feat_name]
                    if pd.notna(val):
                        values.append(float(val))
                
                if values:
                    # Use median for robustness (especially for target-encoded features)
                    # Median is less affected by outliers than mean
                    complete_features[feat_name] = float(np.median(values))
                else:
                    # Fallback to overall median
                    median_val = featured_df[feat_name].median()
                    complete_features[feat_name] = float(median_val) if pd.notna(median_val) else 0.0
        
        # Now ensure ALL model features are present (some might not be in feature_cols)
        for feat_name in model_feature_names:
            if feat_name not in complete_features:
                # Feature not in training data - use median from similar houses if available
                if feat_name in featured_df.columns:
                    values = []
                    for idx in top_indices:
                        val = featured_df.loc[idx, feat_name]
                        if pd.notna(val):
                            val_float = float(val)
                            # Filter extreme values
                            feat_median = featured_df[feat_name].median()
                            feat_std = featured_df[feat_name].std()
                            if feat_std > 0:
                                z_score = abs(val_float - feat_median) / feat_std
                                if z_score < 3:
                                    values.append(val_float)
                            else:
                                values.append(val_float)
                    
                    if values:
                        complete_features[feat_name] = float(np.median(values))
                    else:
                        median_val = featured_df[feat_name].median()
                        complete_features[feat_name] = float(median_val) if pd.notna(median_val) else 0.0
                else:
                    # Feature doesn't exist in training data - use 0
                    complete_features[feat_name] = 0.0
        
        # Verify we have all features
        missing_in_complete = [f for f in model_feature_names if f not in complete_features]
        if missing_in_complete:
            for feat_name in missing_in_complete:
                complete_features[feat_name] = 0.0
        
        
        # Debug: Check for extreme values in complete_features and clamp aggressively
        # Use percentiles from training data to ensure all values are reasonable
        clamped_count = 0
        for feat_name in list(complete_features.keys()):
            if feat_name in featured_df.columns:
                feat_value = complete_features[feat_name]
                # Get percentiles from training data
                p95 = featured_df[feat_name].quantile(0.95)
                p5 = featured_df[feat_name].quantile(0.05)
                
                # Clamp to 95th percentile (very aggressive - only keep middle 90% of values)
                if feat_value > p95:
                    complete_features[feat_name] = float(p95)
                    clamped_count += 1
                elif feat_value < p5:
                    complete_features[feat_name] = float(p5)
                    clamped_count += 1
        
        
        return complete_features
        
    except Exception as e:
        return get_typical_house_features()  # Fallback


def load_model():
    """Load model (lazy loading)."""
    global predictor
    
    if predictor is None:
        try:
            predictor = HousePricePredictor.load(MODEL_PATH)
        except Exception as e:
            raise
    
    return predictor


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    })


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    model = load_model()
    return jsonify(model.get_model_info())


@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Predict price for a single house.
    
    Request body (JSON):
    {
        "features": {
            "Overall.Qual": 7,
            "area": 1500,
            ...
        }
    }
    
    Response:
    {
        "prediction": 175000.50,
        "confidence_interval": {
            "lower": 155000,
            "upper": 195000
        }
    }
    """
    try:
        model = load_model()
        data = request.get_json()
        
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" in request'}), 400
        
        features = data['features']
        
        # Convert to array with better error handling
        # Try to use feature medians from training data (best), then scaler mean, then zeros
        feature_defaults = get_feature_defaults_from_data()
        scaler_mean = None
        if hasattr(model.scaler, 'mean_') and model.scaler.mean_ is not None:
            scaler_mean = model.scaler.mean_.copy()
        
        # Count how many features will be provided by user
        num_provided = len([f for f in model.feature_names if f in features])
        missing_pct_estimate = (len(model.feature_names) - num_provided) / len(model.feature_names) * 100
        
        # Fill missing features by finding similar houses in training data
        # This creates a coherent feature vector based on houses similar to user's input
        complete_features = find_similar_houses_and_fill_features(features, model.feature_names, n_similar=5)
        
        # Initialize X with complete feature vector
        if complete_features:
            # Use features from similar houses (best option - coherent and similar to user input)
            # The complete_features dict already has all features filled from similar houses
            # and has already replaced extreme user values. We just need to ensure all model features are present.
            
            # Verify all model features are in complete_features
            for feat_name in model.feature_names:
                if feat_name not in complete_features:
                    # This shouldn't happen, but fill with 0 as fallback
                    complete_features[feat_name] = 0.0
            
            # Build X from complete_features (already has all features filled)
            X = np.array([complete_features.get(feat, 0.0) for feat in model.feature_names], dtype=np.float64)
            using_data_defaults = True
            
            # Count how many features came from user vs similar houses
            user_provided = [f for f in model.feature_names if f in features and abs(float(features.get(f, 0))) <= 100000]
            
            # Final safety check: clamp any remaining extreme values using training data percentiles
            # This ensures all values are within the range seen in training (BEFORE scaling)
            # This is critical - we need to clamp before scaling to prevent amplification
            if FEATURED_DATA_PATH.exists():
                try:
                    import pandas as pd
                    featured_df_check = pd.read_csv(FEATURED_DATA_PATH)
                    clamped_before_scale = 0
                    for i, feat_name in enumerate(model.feature_names):
                        if feat_name in featured_df_check.columns:
                            p95 = featured_df_check[feat_name].quantile(0.95)
                            p5 = featured_df_check[feat_name].quantile(0.05)
                            old_val = X[i]
                            X[i] = np.clip(X[i], float(p5), float(p95))
                            if old_val != X[i]:
                                clamped_before_scale += 1
                except Exception:
                    # Fallback to simple clipping
                    if np.any(np.abs(X) > 100000):
                        X = np.clip(X, -100000, 100000)
        elif feature_defaults:
            # Use feature medians from training data (fallback)
            matching_features = sum(1 for feat in model.feature_names if feat in feature_defaults)
            X = np.array([feature_defaults.get(feat, 0.0) for feat in model.feature_names], dtype=np.float64)
            using_data_defaults = True
        elif scaler_mean is not None:
            # Fallback to scaler mean
            X = scaler_mean.copy()
            using_data_defaults = False
        else:
            # Last resort: zeros
            X = np.zeros(len(model.feature_names))
            using_data_defaults = False
        
        # If we used complete_features, X is already fully populated
        # Only need to check for conversion errors if we didn't use complete_features
        conversion_errors = []
        missing_features = []
        
        if not complete_features:
            # Only do this conversion loop if we didn't use complete_features
            for i, feat in enumerate(model.feature_names):
                if feat in features:
                    try:
                        value = features[feat]
                        # Try to convert to float
                        if isinstance(value, str):
                            # Check if it's a numeric string
                            try:
                                X[i] = float(value)
                            except ValueError:
                                # It's a categorical string - need encoded value
                                conversion_errors.append(
                                    f"Feature '{feat}' has categorical value '{value}'. "
                                    f"Categorical features must be pre-encoded as numeric values. "
                                    f"Please provide the encoded numeric value for this feature."
                                )
                                # X[i] already set to default above
                                pass
                        else:
                            X[i] = float(value)
                    except (ValueError, TypeError) as e:
                        conversion_errors.append(
                            f"Could not convert feature '{feat}' value '{features[feat]}' to float: {str(e)}"
                        )
                        # X[i] already set to default above
                        pass
                else:
                    # Feature not provided - will use default
                    missing_features.append(feat)
                    # X[i] already set to default above
        else:
            # If we used complete_features, count what was actually from user vs similar houses
            user_provided = [f for f in model.feature_names if f in features and abs(float(features.get(f, 0))) <= 100000]
            missing_features = [f for f in model.feature_names if f not in user_provided]
        
        # Log missing features (but don't error - model will use scaler mean or 0)
        # Warn if too many features are missing (more than 50%)
        if missing_features and len(missing_features) >= len(model.feature_names) * 0.5:
            # Too many missing - prediction may be unreliable
            pass  # Logged below
        
        # If there are conversion errors, return helpful error message
        if conversion_errors:
            return jsonify({
                'error': 'Feature conversion errors',
                'details': conversion_errors,
                'hint': 'Categorical features (like Neighborhood, MS.Zoning, etc.) must be provided as encoded numeric values, not raw strings. The Streamlit app handles this encoding automatically.'
            }), 400
        
        # Calculate missing percentage (needed for warning later)
        missing_pct = len(missing_features) / len(model.feature_names) * 100 if missing_features else 0.0
        
        # Clamp extreme values in target-encoded features (like Neighborhood)
        # These can have very high values that cause unreasonable predictions
        # Use a more reasonable threshold based on training data statistics
        extreme_threshold = 200000  # Allow high target-encoded values but cap extreme outliers
        if np.any(np.abs(X) > extreme_threshold):
            # Clamp to reasonable range (keep most values, only clamp extreme outliers)
            X = np.clip(X, -extreme_threshold, extreme_threshold)
        
        # Predict
        prediction = model.predict(X.reshape(1, -1))[0]
        
        # Get confidence interval
        conf = model.predict_with_confidence(X.reshape(1, -1))
        
        # Add warning if too many features missing (only if not using complete_features)
        warning = None
        if not complete_features and missing_pct > 50:
            warning = f"Warning: {missing_pct:.1f}% of features are missing. Prediction may be unreliable."
        
        response = {
            'prediction': float(prediction),
            'confidence_interval': {
                'lower': float(conf['ci_lower'][0]),
                'upper': float(conf['ci_upper'][0])
            }
        }
        if warning:
            response['warning'] = warning
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict prices for multiple houses.
    
    Request body (JSON):
    {
        "data": [
            [feat1, feat2, ...],  // House 1
            [feat1, feat2, ...],  // House 2
            ...
        ]
    }
    
    Response:
    {
        "predictions": [175000.50, 220000.00, ...]
    }
    """
    try:
        model = load_model()
        data = request.get_json()
        
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" in request'}), 400
        
        X = np.array(data['data'])
        predictions = model.predict(X)
        
        return jsonify({
            'predictions': predictions.tolist(),
            'n_samples': len(predictions)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model/stats', methods=['GET'])
def prediction_stats():
    """Get prediction statistics from logs."""
    model = load_model()
    return jsonify(model.get_prediction_stats())


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)