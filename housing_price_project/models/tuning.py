# src/models/tuning.py
"""
Hyperparameter Tuning utilities.

WHAT ARE HYPERPARAMETERS?
─────────────────────────
Parameters set BEFORE training, not learned from data:
- Learning rate
- Regularization strength (alpha)
- L1/L2 ratio (for Elastic Net)
- Number of iterations

HOW DO WE CHOOSE THEM?
──────────────────────
Grid Search: Try all combinations, pick best via CV
Random Search: Sample combinations randomly (often faster)
"""

import numpy as np
from itertools import product
import time


class GridSearchCV:
    """
    Exhaustive search over hyperparameter grid with cross-validation.
    
    Algorithm:
    1. Define parameter grid: {'alpha': [0.1, 1, 10], 'l1_ratio': [0.25, 0.5, 0.75]}
    2. For each combination:
       a. Run K-Fold cross-validation
       b. Record mean and std of scores
    3. Select best combination
    4. Refit on full training data with best params
    
    Example
    -------
    >>> param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    >>> grid = GridSearchCV(RidgeRegression, param_grid, cv=5)
    >>> grid.fit(X_train, y_train)
    >>> print(grid.best_params_)
    >>> best_model = grid.best_estimator_
    """
    
    def __init__(self, estimator_class, param_grid, cv=5, scoring='r2', 
                 verbose=1, return_train_score=False):
        """
        Parameters
        ----------
        estimator_class : class
            The model class (not instance!) e.g., RidgeRegression
        param_grid : dict
            {param_name: [values to try]}
        cv : int or CV splitter
            Number of folds or CV object
        scoring : str
            Metric to optimize ('r2', 'neg_mse', 'neg_rmse')
        verbose : int
            0 = silent, 1 = progress, 2 = detailed
        return_train_score : bool
            Whether to compute training scores (slower)
        """
        self.estimator_class = estimator_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.return_train_score = return_train_score
        
        # Results (populated after fit)
        self.cv_results_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
    
    def fit(self, X, y):
        """
        Run grid search.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,)
            Target values
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        all_combinations = list(product(*param_values))
        
        n_combinations = len(all_combinations)
        
        if self.verbose:
            print(f"Grid Search: {n_combinations} parameter combinations")
            print(f"Parameters: {param_names}")
            print(f"CV folds: {self.cv}")
            print("-" * 50)
        
        # Set up CV
        from sklearn.model_selection import KFold
        if isinstance(self.cv, int):
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        else:
            cv_splitter = self.cv
        
        # Results storage
        results = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
            'rank_test_score': [],
            'mean_fit_time': [],
        }
        
        if self.return_train_score:
            results['mean_train_score'] = []
            results['std_train_score'] = []
        
        # Try each combination
        best_score = -np.inf
        best_params = None
        
        for idx, param_combo in enumerate(all_combinations):
            params = dict(zip(param_names, param_combo))
            
            if self.verbose >= 2:
                print(f"\n[{idx+1}/{n_combinations}] Testing: {params}")
            
            # Cross-validation for this parameter combo
            test_scores = []
            train_scores = []
            fit_times = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Create and fit model
                start_time = time.time()
                model = self.estimator_class(**params)
                model.fit(X_train_fold, y_train_fold)
                fit_time = time.time() - start_time
                fit_times.append(fit_time)
                
                # Evaluate
                y_val_pred = model.predict(X_val_fold)
                test_score = self._compute_score(y_val_fold, y_val_pred)
                test_scores.append(test_score)
                
                if self.return_train_score:
                    y_train_pred = model.predict(X_train_fold)
                    train_score = self._compute_score(y_train_fold, y_train_pred)
                    train_scores.append(train_score)
            
            # Record results
            mean_test = np.mean(test_scores)
            std_test = np.std(test_scores)
            
            results['params'].append(params)
            results['mean_test_score'].append(mean_test)
            results['std_test_score'].append(std_test)
            results['mean_fit_time'].append(np.mean(fit_times))
            
            if self.return_train_score:
                results['mean_train_score'].append(np.mean(train_scores))
                results['std_train_score'].append(np.std(train_scores))
            
            # Track best
            if mean_test > best_score:
                best_score = mean_test
                best_params = params
            
            if self.verbose == 1:
                print(f"  [{idx+1}/{n_combinations}] {params} → "
                      f"{self.scoring}={mean_test:.4f} (±{std_test:.4f})")
        
        # Compute ranks
        scores = np.array(results['mean_test_score'])
        results['rank_test_score'] = (scores.argsort()[::-1].argsort() + 1).tolist()
        
        # Store results
        self.cv_results_ = results
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        # Refit best model on full training data
        if self.verbose:
            print("\n" + "="*50)
            print(f"Best params: {best_params}")
            print(f"Best CV score: {best_score:.4f}")
            print("Refitting on full training data...")
        
        self.best_estimator_ = self.estimator_class(**best_params)
        self.best_estimator_.fit(X, y)
        
        return self
    
    def _compute_score(self, y_true, y_pred):
        """Compute score based on scoring parameter."""
        if self.scoring == 'r2':
            return r2_score(y_true, y_pred)
        elif self.scoring == 'neg_mse':
            return -mse(y_true, y_pred)
        elif self.scoring == 'neg_rmse':
            return -rmse(y_true, y_pred)
        else:
            raise ValueError(f"Unknown scoring: {self.scoring}")
    
    def predict(self, X):
        """Predict using best estimator."""
        if self.best_estimator_ is None:
            raise RuntimeError("Not fitted. Call fit() first.")
        return self.best_estimator_.predict(X)
    
    def get_results_df(self):
        """Return results as DataFrame (if pandas available)."""
        try:
            import pandas as pd
            return pd.DataFrame(self.cv_results_).sort_values('rank_test_score')
        except ImportError:
            return self.cv_results_


class RandomizedSearchCV:
    """
    Randomized search over hyperparameters.
    
    Why Random Search?
    ─────────────────
    Grid search with many parameters is expensive:
    - 5 params × 10 values each = 100,000 combinations!
    
    Random search samples n_iter combinations randomly.
    Often finds good params with far fewer evaluations.
    
    Research shows: Random search is surprisingly effective!
    (See Bergstra & Bengio, 2012)
    """
    
    def __init__(self, estimator_class, param_distributions, n_iter=10, 
                 cv=5, scoring='r2', random_state=None, verbose=1):
        """
        Parameters
        ----------
        param_distributions : dict
            {param_name: distribution or list}
            For lists, samples uniformly
            For scipy distributions, samples from distribution
        n_iter : int
            Number of random combinations to try
        """
        self.estimator_class = estimator_class
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose
        
        self.cv_results_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
    
    def fit(self, X, y):
        """Run randomized search."""
        rng = np.random.RandomState(self.random_state)
        
        # Sample random parameter combinations
        param_list = []
        for _ in range(self.n_iter):
            params = {}
            for name, dist in self.param_distributions.items():
                if isinstance(dist, list):
                    params[name] = rng.choice(dist)
                elif hasattr(dist, 'rvs'):  # scipy distribution
                    params[name] = dist.rvs(random_state=rng)
                else:
                    params[name] = dist
            param_list.append(params)
        
        
        return self


# Helper functions
def r2_score(y_true, y_pred):
    """R² score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def mse(y_true, y_pred):
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mse(y_true, y_pred))