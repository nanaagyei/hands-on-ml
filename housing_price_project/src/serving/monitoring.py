"""
Model Monitoring: Detect when your model needs retraining.

WHY MONITOR?
────────────
The world changes! A model trained in 2023 might not work in 2026:
- Housing market conditions change
- New neighborhoods are built
- Economic factors shift

TYPES OF DRIFT:
───────────────
1. DATA DRIFT (Covariate Shift):
   Input distribution changes.
   Example: Suddenly seeing more luxury homes than during training.
   
2. CONCEPT DRIFT:
   Relationship between inputs and output changes.
   Example: Same house features now worth more due to market boom.
   
3. PREDICTION DRIFT:
   Model output distribution changes.
   May indicate either data drift or concept drift.
"""

import numpy as np
from collections import deque
from datetime import datetime, timedelta


class DriftDetector:
    """
    Detect distribution drift using statistical tests.
    
    Methods:
    1. Population Stability Index (PSI)
    2. Kolmogorov-Smirnov test
    3. Simple threshold-based monitoring
    
    Example
    -------
    >>> detector = DriftDetector(reference_data=X_train)
    >>> 
    >>> # As new predictions come in:
    >>> for batch in new_data_batches:
    ...     drift_report = detector.check_drift(batch)
    ...     if drift_report['drift_detected']:
    ...         alert("Data drift detected! Consider retraining.")
    """
    
    def __init__(self, reference_data, feature_names=None, 
                 psi_threshold=0.2, window_size=1000):
        """
        Parameters
        ----------
        reference_data : ndarray
            Training data distribution (baseline)
        feature_names : list, optional
            Names for interpretable reports
        psi_threshold : float
            PSI > threshold indicates significant drift
            0.1 = slight drift, 0.2 = moderate, 0.25+ = significant
        window_size : int
            Number of recent samples to compare
        """
        self.reference_data = np.asarray(reference_data)
        self.feature_names = feature_names or [f'feature_{i}' for i in range(reference_data.shape[1])]
        self.psi_threshold = psi_threshold
        self.window_size = window_size
        
        # Compute reference statistics
        self._compute_reference_stats()
        
        # Rolling window for incoming data
        self.recent_data = deque(maxlen=window_size)
        
        # Drift history
        self.drift_history = []
    
    def _compute_reference_stats(self):
        """Compute statistics from reference (training) data."""
        self.reference_stats = {
            'mean': np.mean(self.reference_data, axis=0),
            'std': np.std(self.reference_data, axis=0),
            'min': np.min(self.reference_data, axis=0),
            'max': np.max(self.reference_data, axis=0),
            'percentiles': {
                '25': np.percentile(self.reference_data, 25, axis=0),
                '50': np.percentile(self.reference_data, 50, axis=0),
                '75': np.percentile(self.reference_data, 75, axis=0),
            }
        }
        
        # Compute histograms for PSI (10 bins)
        self.reference_histograms = []
        self.bin_edges = []
        
        for col in range(self.reference_data.shape[1]):
            col_data = self.reference_data[:, col]
            hist, edges = np.histogram(col_data, bins=10)
            # Normalize to proportions
            hist = hist / hist.sum()
            # Avoid zero proportions (causes log issues)
            hist = np.clip(hist, 0.0001, None)
            
            self.reference_histograms.append(hist)
            self.bin_edges.append(edges)
    
    def add_data(self, X):
        """Add new data to monitoring window."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        for row in X:
            self.recent_data.append(row)
    
    def compute_psi(self, new_data, feature_idx):
        """
        Compute Population Stability Index for a feature.
        
        PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)
        
        Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 ≤ PSI < 0.2: Slight change, monitor
        - PSI ≥ 0.2: Significant change, investigate!
        """
        # Compute histogram of new data using same bins
        new_hist, _ = np.histogram(new_data, bins=self.bin_edges[feature_idx])
        new_hist = new_hist / new_hist.sum()
        new_hist = np.clip(new_hist, 0.0001, None)
        
        ref_hist = self.reference_histograms[feature_idx]
        
        # PSI formula
        psi = np.sum((new_hist - ref_hist) * np.log(new_hist / ref_hist))
        
        return psi
    
    def check_drift(self, X=None):
        """
        Check for data drift.
        
        Parameters
        ----------
        X : ndarray, optional
            New data to check. If None, uses accumulated recent_data.
            
        Returns
        -------
        report : dict
            Drift detection report
        """
        if X is not None:
            self.add_data(X)
        
        if len(self.recent_data) < 100:
            return {
                'drift_detected': False,
                'message': f'Insufficient data ({len(self.recent_data)} samples)',
                'psi_scores': {}
            }
        
        recent_array = np.array(list(self.recent_data))
        
        # Compute PSI for each feature
        psi_scores = {}
        drifted_features = []
        
        for i, feat_name in enumerate(self.feature_names):
            psi = self.compute_psi(recent_array[:, i], i)
            psi_scores[feat_name] = psi
            
            if psi >= self.psi_threshold:
                drifted_features.append((feat_name, psi))
        
        # Overall drift detection
        drift_detected = len(drifted_features) > 0
        
        # Compute mean shift for interpretability
        current_mean = np.mean(recent_array, axis=0)
        mean_shift = current_mean - self.reference_stats['mean']
        mean_shift_pct = mean_shift / (self.reference_stats['std'] + 1e-10)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_detected,
            'n_samples': len(self.recent_data),
            'psi_scores': psi_scores,
            'drifted_features': drifted_features,
            'mean_psi': np.mean(list(psi_scores.values())),
            'max_psi': max(psi_scores.values()) if psi_scores else 0,
            'mean_shift_std': dict(zip(self.feature_names, mean_shift_pct)),
        }
        
        # Log drift event
        self.drift_history.append({
            'timestamp': report['timestamp'],
            'drift_detected': drift_detected,
            'mean_psi': report['mean_psi'],
            'n_drifted': len(drifted_features)
        })
        
        return report
    
    def get_drift_summary(self):
        """Get summary of drift history."""
        if not self.drift_history:
            return {'message': 'No drift checks performed yet'}
        
        drift_events = [h for h in self.drift_history if h['drift_detected']]
        
        return {
            'total_checks': len(self.drift_history),
            'drift_events': len(drift_events),
            'drift_rate': len(drift_events) / len(self.drift_history),
            'last_check': self.drift_history[-1]['timestamp'],
            'recent_psi_trend': [h['mean_psi'] for h in self.drift_history[-10:]],
        }


class PredictionMonitor:
    """
    Monitor prediction distribution over time.
    
    Detects when model outputs change significantly,
    which may indicate problems even before ground truth is available.
    """
    
    def __init__(self, reference_predictions=None, alert_threshold=2.0):
        """
        Parameters
        ----------
        reference_predictions : array-like, optional
            Baseline predictions (e.g., from validation set)
        alert_threshold : float
            Number of std devs from reference to trigger alert
        """
        self.reference_predictions = reference_predictions
        self.alert_threshold = alert_threshold
        
        if reference_predictions is not None:
            self.ref_mean = np.mean(reference_predictions)
            self.ref_std = np.std(reference_predictions)
        else:
            self.ref_mean = None
            self.ref_std = None
        
        self.prediction_history = []
    
    def log_predictions(self, predictions, timestamps=None):
        """Log new predictions."""
        predictions = np.asarray(predictions).ravel()
        
        if timestamps is None:
            timestamps = [datetime.now().isoformat()] * len(predictions)
        
        for pred, ts in zip(predictions, timestamps):
            self.prediction_history.append({
                'timestamp': ts,
                'prediction': float(pred)
            })
    
    def check_prediction_drift(self, recent_n=100):
        """Check if recent predictions differ from reference."""
        if len(self.prediction_history) < recent_n:
            return {'status': 'insufficient_data'}
        
        recent_preds = [h['prediction'] for h in self.prediction_history[-recent_n:]]
        recent_mean = np.mean(recent_preds)
        recent_std = np.std(recent_preds)
        
        # Compare to reference
        if self.ref_mean is not None:
            z_score = (recent_mean - self.ref_mean) / (self.ref_std + 1e-10)
            alert = abs(z_score) > self.alert_threshold
        else:
            z_score = 0
            alert = False
        
        return {
            'recent_mean': recent_mean,
            'recent_std': recent_std,
            'reference_mean': self.ref_mean,
            'reference_std': self.ref_std,
            'z_score': z_score,
            'alert': alert,
            'message': f"Predictions shifted {z_score:.1f} std from reference" if alert else "Normal"
        }
    
    def get_prediction_trend(self, window='day'):
        """Get prediction trends over time."""
        if not self.prediction_history:
            return {}
        
        # Group by time period
        # Simplified: just return recent stats
        recent = self.prediction_history[-1000:]
        
        predictions = [h['prediction'] for h in recent]
        
        return {
            'n_predictions': len(predictions),
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'median': np.median(predictions),
        }


class PerformanceMonitor:
    """
    Monitor model performance when ground truth becomes available.
    
    In real estate, actual sale prices are known after closing.
    This allows us to track true model performance over time.
    """
    
    def __init__(self, performance_threshold=0.85, window_size=100):
        """
        Parameters
        ----------
        performance_threshold : float
            R² below this triggers alert
        window_size : int
            Rolling window for performance calculation
        """
        self.performance_threshold = performance_threshold
        self.window_size = window_size
        
        # Store predictions with actuals
        self.records = []
    
    def log_outcome(self, prediction, actual, timestamp=None):
        """Log a prediction with its actual outcome."""
        self.records.append({
            'timestamp': timestamp or datetime.now().isoformat(),
            'prediction': float(prediction),
            'actual': float(actual),
            'error': float(actual - prediction),
            'abs_error': float(abs(actual - prediction)),
            'pct_error': float(abs(actual - prediction) / actual * 100) if actual != 0 else 0,
        })
    
    def get_recent_performance(self, n=None):
        """Calculate performance on recent predictions."""
        n = n or self.window_size
        recent = self.records[-n:] if len(self.records) >= n else self.records
        
        if len(recent) < 10:
            return {'status': 'insufficient_data', 'n_records': len(recent)}
        
        predictions = np.array([r['prediction'] for r in recent])
        actuals = np.array([r['actual'] for r in recent])
        
        # Calculate metrics
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - actuals.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        mae = np.mean(np.abs(actuals - predictions))
        mape = np.mean([r['pct_error'] for r in recent])
        
        # Check for performance degradation
        alert = r2 < self.performance_threshold
        
        return {
            'n_samples': len(recent),
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'alert': alert,
            'message': f"Performance degraded (R²={r2:.3f})" if alert else "Normal",
        }
    
    def get_performance_trend(self):
        """Track performance over time."""
        if len(self.records) < self.window_size:
            return {'status': 'insufficient_data'}
        
        # Calculate rolling performance
        window_metrics = []
        
        for i in range(self.window_size, len(self.records), self.window_size // 2):
            window = self.records[i - self.window_size:i]
            
            preds = np.array([r['prediction'] for r in window])
            actuals = np.array([r['actual'] for r in window])
            
            ss_res = np.sum((actuals - preds) ** 2)
            ss_tot = np.sum((actuals - actuals.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            window_metrics.append({
                'end_index': i,
                'r2': r2,
                'timestamp': window[-1]['timestamp']
            })
        
        return {
            'trend': window_metrics,
            'current_r2': window_metrics[-1]['r2'] if window_metrics else None,
            'trend_direction': 'declining' if len(window_metrics) >= 2 and 
                              window_metrics[-1]['r2'] < window_metrics[-2]['r2'] else 'stable'
        }