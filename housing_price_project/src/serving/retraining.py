"""
Retraining Strategies: When and how to update your model.

DECISION FRAMEWORK:
───────────────────
1. SCHEDULED RETRAINING (Simple)
   - Retrain weekly/monthly regardless of performance
   - Works well for stable domains
   
2. PERFORMANCE-TRIGGERED
   - Monitor metrics, retrain when performance degrades
   - More efficient but requires ground truth
   
3. DRIFT-TRIGGERED  
   - Monitor input distribution, retrain when drift detected
   - Works without ground truth

4. CONTINUOUS (Online Learning)
   - Update incrementally with each new sample
   - Best for fast-changing domains
"""

import numpy as np
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import sys

# Ensure linear_regression module is available
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent.parent  # housing_price_project
_main_root = _project_root.parent  # hands-on-ml

if str(_main_root) not in sys.path:
    sys.path.insert(0, str(_main_root))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class RetrainingManager:
    """
    Manage model retraining decisions and execution.
    
    Example
    -------
    >>> manager = RetrainingManager(
    ...     model_path='models/final_model.pkl',
    ...     strategy='performance',
    ...     performance_threshold=0.85
    ... )
    >>> 
    >>> # Check if retraining needed
    >>> if manager.should_retrain():
    ...     manager.retrain(X_new, y_new)
    """
    
    def __init__(self, model_path, strategy='scheduled', 
                 retrain_interval_days=30,
                 performance_threshold=0.85,
                 drift_threshold=0.2):
        """
        Parameters
        ----------
        model_path : str
            Path to saved model
        strategy : str
            'scheduled', 'performance', 'drift', 'hybrid'
        retrain_interval_days : int
            For scheduled strategy
        performance_threshold : float
            For performance strategy
        drift_threshold : float
            For drift strategy
        """
        self.model_path = Path(model_path)
        self.strategy = strategy
        self.retrain_interval_days = retrain_interval_days
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold
        
        self.last_retrain = datetime.now()
        self.performance_history = []
        self.drift_history = []
        self.retrain_history = []
    
    def should_retrain(self, current_performance=None, drift_score=None):
        """
        Determine if retraining is needed.
        
        Parameters
        ----------
        current_performance : float, optional
            Current model R² or other metric
        drift_score : float, optional
            Current PSI or drift metric
            
        Returns
        -------
        should_retrain : bool
        reason : str
        """
        reasons = []
        
        # Check scheduled retraining
        if self.strategy in ['scheduled', 'hybrid']:
            days_since = (datetime.now() - self.last_retrain).days
            if days_since >= self.retrain_interval_days:
                reasons.append(f"Scheduled: {days_since} days since last retrain")
        
        # Check performance-based
        if self.strategy in ['performance', 'hybrid'] and current_performance is not None:
            if current_performance < self.performance_threshold:
                reasons.append(
                    f"Performance: {current_performance:.3f} < {self.performance_threshold}"
                )
        
        # Check drift-based
        if self.strategy in ['drift', 'hybrid'] and drift_score is not None:
            if drift_score > self.drift_threshold:
                reasons.append(
                    f"Drift: PSI {drift_score:.3f} > {self.drift_threshold}"
                )
        
        should = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "No retraining needed"
        
        return should, reason
    
    def retrain(self, X_train, y_train, X_val=None, y_val=None, 
                model_class=None, model_params=None):
        """
        Retrain the model with new data.
        
        Parameters
        ----------
        X_train, y_train : arrays
            Training data (should include historical + new)
        X_val, y_val : arrays, optional
            Validation data for evaluation
        model_class : class, optional
            Model class to use (default: same as current)
        model_params : dict, optional
            Model parameters (default: same as current)
        """
        from src.serving.predictor import HousePricePredictor
        from linear_regression.preprocessing.scalers import StandardScaler
        
        # Load current model for reference
        current = HousePricePredictor.load(self.model_path)
        
        # Use same model class/params if not specified
        if model_class is None:
            model_class = current.model.__class__
        if model_params is None:
            model_params = current.model.get_params()
        
        
        # Fit scaler on new training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Fit model
        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate if validation data provided
        if X_val is not None and y_val is not None:
            X_val_scaled = scaler.transform(X_val)
            y_pred = model.predict(X_val_scaled)
            
            # Convert from log space if needed
            y_pred_actual = np.expm1(y_pred)
            y_val_actual = np.expm1(y_val)
            
            ss_res = np.sum((y_val_actual - y_pred_actual) ** 2)
            ss_tot = np.sum((y_val_actual - y_val_actual.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
            
            rmse = np.sqrt(np.mean((y_val_actual - y_pred_actual) ** 2))
            
            print(f"\nValidation Results:")
            print(f"  R² = {r2:.4f}")
            print(f"  RMSE = ${rmse:,.0f}")
        
        # Save new model
        new_predictor = HousePricePredictor(
            model=model,
            scaler=scaler,
            feature_names=current.feature_names,
            metadata={
                'model_name': model_class.__name__,
                'params': model_params,
                'retrain_date': datetime.now().isoformat(),
                'n_training_samples': len(X_train),
            }
        )
        
        # Backup old model
        backup_path = self.model_path.parent / f"model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        current.save(backup_path)
        # Save new model
        new_predictor.save(self.model_path)
        
        # Update state
        self.last_retrain = datetime.now()
        self.retrain_history.append({
            'timestamp': self.last_retrain.isoformat(),
            'n_samples': len(X_train),
            'validation_r2': r2 if X_val is not None else None,
        })
        
        return new_predictor
    
    def get_retrain_history(self):
        """Get history of retraining events."""
        return self.retrain_history
