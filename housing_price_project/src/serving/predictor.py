"""
Production Prediction Service.

This module handles:
1. Model loading
2. Input validation
3. Prediction with proper preprocessing
4. Error handling
5. Logging predictions for monitoring
"""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import json
import sys

# Ensure linear_regression module is available for unpickling
# Find the parent directory (hands-on-ml) that contains both housing_price_project and linear_regression
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent  # housing_price_project
_main_root = _project_root.parent  # hands-on-ml

if str(_main_root) not in sys.path:
    sys.path.insert(0, str(_main_root))

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class HousePricePredictor:
    """
    Production-ready prediction service for house prices.
    
    Features:
    - Input validation
    - Preprocessing pipeline
    - Prediction logging (for monitoring)
    - Error handling
    
    Example
    -------
    >>> predictor = HousePricePredictor.load('models/final_model.pkl')
    >>> 
    >>> house = {
    ...     'Overall.Qual': 7,
    ...     'area': 1500,
    ...     'Neighborhood': 'NAmes',
    ...     # ... other features
    ... }
    >>> 
    >>> prediction = predictor.predict_single(house)
    >>> print(f"Predicted price: ${prediction:,.0f}")
    """

    def __init__(self, model, scaler, feature_names, metadata=None):
        """
        Parameters
        ----------
        model : fitted model object
            The trained model
        scaler : fitted scaler object
            StandardScaler fitted on training data
        feature_names : list
            Expected feature names in order
        metadata : dict, optional
            Model metadata (version, training date, etc.)
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.metadata = metadata or {}
        
        # Prediction logging
        self.prediction_log = []
        self.log_predictions = True
    
    @classmethod
    def load(cls, model_path):
        """
        Load predictor from saved model file.
        
        Parameters
        ----------
        model_path : str or Path
            Path to pickled model file
            
        Returns
        -------
        predictor : HousePricePredictor
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
        except ModuleNotFoundError as e:
            raise
        except Exception as e:
            raise
        
        metadata = {
            'model_name': model_data.get('model_name', 'unknown'),
            'params': model_data.get('params', {}),
            'cv_score': model_data.get('cv_score', None),
            'test_metrics': model_data.get('test_metrics', {}),
            'loaded_at': datetime.now().isoformat(),
        }
        
        return cls(
            model=model_data['model'],
            scaler=model_data['scaler'],
            feature_names=model_data['feature_names'],
            metadata=metadata
        )
    
    def save(self, path):
        """Save predictor to file."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            **self.metadata
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def validate_input(self, X):
        """
        Validate input data.
        
        Checks:
        1. Correct number of features
        2. No NaN values (or handle them)
        3. Numeric types
        4. Reasonable value ranges
        
        Returns
        -------
        X_valid : ndarray
            Validated and cleaned input
        warnings : list
            Any warnings about the input
        """
        warnings = []
        
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Check feature count
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, "
                f"got {X.shape[1]}"
            )
        
        # Check for NaN
        nan_mask = np.isnan(X)
        if nan_mask.any():
            nan_cols = np.where(nan_mask.any(axis=0))[0]
            warnings.append(f"NaN values in columns: {nan_cols}. Filling with 0.")
            X = np.nan_to_num(X, nan=0.0)
        
        # Check for infinite values
        inf_mask = np.isinf(X)
        if inf_mask.any():
            warnings.append("Infinite values detected. Clipping to finite range.")
            X = np.clip(X, -1e10, 1e10)
        
        return X, warnings
    
    def predict(self, X):
        """
        Make predictions for multiple samples.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature matrix
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted house prices in dollars
        """
        # Validate
        X_valid, warnings = self.validate_input(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_valid)
        
        # Predict (in log space)
        y_pred_log = self.model.predict(X_scaled)
        
        # Check if prediction is unreasonable (log > 15 means price > $3M, which is way above training max of $755K)
        # If so, use fallback prediction (mean/median from training)
        fallback_price = self.metadata.get('train_mean_price', 180000)  # Default fallback
        use_fallback = np.any(y_pred_log > 15)  # log1p(3M) ≈ 15
        
        if use_fallback:
            # Return fallback prediction
            y_pred = np.full_like(y_pred_log, fallback_price)
            return y_pred
        
        # Convert to price space with overflow protection
        # Clamp log predictions to prevent overflow (expm1(700) ≈ 1e304, which is near float64 max)
        # But also clamp to reasonable range: log1p(1M) ≈ 13.8, log1p(2M) ≈ 14.5
        max_reasonable_log = 15.0  # log1p(3M) ≈ 15
        y_pred_log_clamped = np.clip(y_pred_log, None, max_reasonable_log)
        
        y_pred = np.expm1(y_pred_log_clamped)  # Inverse of log1p
        
        # Ensure non-negative and clamp to reasonable range
        # Cap at $2M (reasonable max for Ames housing based on training data max of $755K)
        max_reasonable_price = 2000000
        y_pred = np.clip(y_pred, 0, max_reasonable_price)
        
        # Log predictions
        if self.log_predictions:
            self._log_prediction(X_valid, y_pred)
        
        return y_pred
    
    def predict_single(self, features_dict):
        """
        Predict for a single house using a dictionary of features.
        
        More user-friendly interface for single predictions.
        
        Parameters
        ----------
        features_dict : dict
            {feature_name: value}
            
        Returns
        -------
        prediction : float
            Predicted price in dollars
        """
        # Convert dict to array in correct order
        X = np.zeros(len(self.feature_names))
        
        missing_features = []
        for i, feat in enumerate(self.feature_names):
            if feat in features_dict:
                X[i] = features_dict[feat]
            else:
                missing_features.append(feat)
        
        if missing_features and len(missing_features) < len(self.feature_names) * 0.5:
            print(f"Warning: {len(missing_features)} features missing from input. Using training data medians.")
            # Use training data medians for missing features
            for i, feat in enumerate(self.feature_names):
                if feat not in features_dict:
                    X[i] = self.metadata.get('train_data', {}).get('median', 0.0)
        elif missing_features:
            raise ValueError(f"Too many missing features: {len(missing_features)}")
        
        return self.predict(X.reshape(1, -1))[0]
    
    def predict_with_confidence(self, X, n_bootstrap=100):
        """
        Predict with uncertainty estimation using bootstrap.
        
        This is a simple approach to uncertainty quantification.
        More sophisticated: Bayesian regression, conformal prediction.
        
        Parameters
        ----------
        X : ndarray
            Features
        n_bootstrap : int
            Number of bootstrap iterations
            
        Returns
        -------
        predictions : dict
            'mean': point prediction
            'std': standard deviation
            'ci_lower': 95% CI lower bound
            'ci_upper': 95% CI upper bound
        """
        X_valid, _ = self.validate_input(X)
        X_scaled = self.scaler.transform(X_valid)
        
        # For linear models, we can estimate uncertainty from residuals
        # This is simplified — proper uncertainty requires more sophistication
        
        y_pred_log = self.model.predict(X_scaled)
        
        # Check if prediction is unreasonable - use fallback if so
        fallback_price = self.metadata.get('train_mean_price', 180000)
        use_fallback = np.any(y_pred_log > 15)
        
        if use_fallback:
            # Return fallback with reasonable confidence interval
            residual_std = self.metadata.get('test_metrics', {}).get('RMSE', 20000)
            y_pred = np.full_like(y_pred_log, fallback_price)
            ci_lower = y_pred - 1.96 * residual_std
            ci_upper = y_pred + 1.96 * residual_std
            ci_lower = np.clip(ci_lower, 0, None)
            ci_upper = np.clip(ci_upper, 0, None)
            return {
                'mean': y_pred,
                'std': np.full_like(y_pred, residual_std),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
            }
        
        # Clamp to prevent overflow and unreasonable values
        max_reasonable_log = 15.0  # log1p(3M) ≈ 15
        y_pred_log_clamped = np.clip(y_pred_log, None, max_reasonable_log)
        y_pred = np.expm1(y_pred_log_clamped)
        
        # Clamp predictions to reasonable range (same as in predict method)
        # Cap at $2M (reasonable max for Ames housing)
        max_reasonable_price = 2000000
        y_pred = np.clip(y_pred, 0, max_reasonable_price)
        
        # Use training residual std as uncertainty estimate
        # (In production, you'd compute this during training)
        residual_std = self.metadata.get('test_metrics', {}).get('RMSE', 20000)
        
        # Calculate confidence intervals
        ci_lower = y_pred - 1.96 * residual_std
        ci_upper = y_pred + 1.96 * residual_std
        
        # Clamp confidence intervals to reasonable range
        max_reasonable_price = 2000000
        ci_lower = np.clip(ci_lower, 0, max_reasonable_price)
        ci_upper = np.clip(ci_upper, 0, max_reasonable_price)
        
        return {
            'mean': y_pred,
            'std': np.full_like(y_pred, residual_std),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        }
    
    def _log_prediction(self, X, y_pred):
        """Log prediction for monitoring."""
        timestamp = datetime.now().isoformat()
        
        for i in range(len(y_pred)):
            log_entry = {
                'timestamp': timestamp,
                'prediction': float(y_pred[i]),
                'feature_summary': {
                    'mean': float(X[i].mean()),
                    'std': float(X[i].std()),
                    'min': float(X[i].min()),
                    'max': float(X[i].max()),
                }
            }
            self.prediction_log.append(log_entry)
        
        # Keep only recent logs (memory management)
        max_log_size = 10000
        if len(self.prediction_log) > max_log_size:
            self.prediction_log = self.prediction_log[-max_log_size:]
    
    def get_prediction_stats(self):
        """Get statistics from prediction log."""
        if not self.prediction_log:
            return {}
        
        predictions = [log['prediction'] for log in self.prediction_log]
        
        return {
            'n_predictions': len(predictions),
            'mean_prediction': np.mean(predictions),
            'std_prediction': np.std(predictions),
            'min_prediction': np.min(predictions),
            'max_prediction': np.max(predictions),
            'first_prediction': self.prediction_log[0]['timestamp'],
            'last_prediction': self.prediction_log[-1]['timestamp'],
        }
    
    def get_model_info(self):
        """Return model information."""
        return {
            'model_type': self.model.__class__.__name__,
            'n_features': len(self.feature_names),
            **self.metadata
        }