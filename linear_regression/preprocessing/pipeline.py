"""
Pipeline implementation for chaining transformers and estimators.

A pipeline chains multiple processing steps:
    [Transformer₁] → [Transformer₂] → ... → [Estimator]
    
Each transformer must implement:
    - fit(X, y=None): Learn parameters
    - transform(X): Apply transformation
    - fit_transform(X, y=None): Both in one step
    
The final estimator must implement:
    - fit(X, y): Train the model
    - predict(X): Make predictions

Data flows through like an assembly line:
    X → scale → encode → reduce dims → model → predictions
"""

import numpy as np


class Pipeline:
    """
    Chain transformers and a final estimator.
    
    Parameters
    ----------
    steps : list of tuples
        List of (name, transform/estimator) tuples.
        All but last must be transformers (have fit_transform).
        Last can be transformer or estimator (has fit + predict).
    
    Example
    -------
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('model', LinearRegression())
    ... ])
    >>> pipe.fit(X_train, y_train)
    >>> predictions = pipe.predict(X_test)
    """

    def __init__(self, steps):
        self.steps = steps
        self._validate_steps()
        
        # Create named access: pipe.named_steps['scaler']
        self.named_steps = {name: step for name, step in steps}
    
    def _validate_steps(self):
        """Validate pipeline steps."""
        if not self.steps:
            raise ValueError("Pipeline cannot be empty")
        
        names = [name for name, _ in self.steps]
        
        # Check for duplicate names
        if len(names) != len(set(names)):
            raise ValueError("Step names must be unique")
        
        # Check that all but last have transform method
        for name, step in self.steps[:-1]:
            if not hasattr(step, 'transform'):
                raise TypeError(
                    f"All intermediate steps must be transformers "
                    f"(implement transform). '{name}' doesn't."
                )
    
    @property
    def _transformers(self):
        """All steps except the last."""
        return self.steps[:-1]
    
    @property
    def _final_estimator(self):
        """The last step (transformer or estimator)."""
        return self.steps[-1][1]
    
    def fit(self, X, y=None):
        """
        Fit all transformers and the final estimator.
        
        Flow:
        1. For each transformer: fit_transform(X) → pass result to next
        2. For final estimator: fit(X_transformed, y)
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,), optional
            Target values (required if final step needs it)
            
        Returns
        -------
        self : fitted pipeline
        """
        X_transformed = X
        
        # Fit and transform through all transformers
        for name, transformer in self._transformers:
            # Use fit_transform for efficiency if available
            if hasattr(transformer, 'fit_transform'):
                X_transformed = transformer.fit_transform(X_transformed)
            else:
                transformer.fit(X_transformed)
                X_transformed = transformer.transform(X_transformed)
        
        # Fit the final estimator
        final = self._final_estimator
        if y is not None:
            final.fit(X_transformed, y)
        else:
            final.fit(X_transformed)
        
        return self
    
    def transform(self, X):
        """
        Apply transforms to the data.
        
        Only valid if final step is a transformer!
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_transformed : ndarray
            Transformed data
        """
        X_transformed = X
        
        # Transform through all steps including final
        for name, step in self.steps:
            if not hasattr(step, 'transform'):
                raise TypeError(
                    f"transform() only works if all steps are transformers. "
                    f"'{name}' has no transform method."
                )
            X_transformed = step.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        X_transformed = X
        
        # Fit and transform through all transformers
        for name, transformer in self._transformers:
            if hasattr(transformer, 'fit_transform'):
                X_transformed = transformer.fit_transform(X_transformed)
            else:
                transformer.fit(X_transformed)
                X_transformed = transformer.transform(X_transformed)
        
        # Handle final step
        final_name, final = self.steps[-1]
        if hasattr(final, 'fit_transform'):
            X_transformed = final.fit_transform(X_transformed, y)
        else:
            # Final step is an estimator, just fit it
            if y is not None:
                final.fit(X_transformed, y)
            else:
                final.fit(X_transformed)
            # Can't return transformed data if final is estimator
            # Return X_transformed before final step
            return X_transformed
        
        return X_transformed
    
    def predict(self, X):
        """
        Transform the data and predict using the final estimator.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        # Check that final step can predict
        if not hasattr(self._final_estimator, 'predict'):
            raise TypeError(
                "predict() requires final step to be an estimator "
                "with a predict method."
            )
        
        # Transform through all intermediate steps
        X_transformed = X
        for name, transformer in self._transformers:
            X_transformed = transformer.transform(X_transformed)
        
        # Predict with final estimator
        return self._final_estimator.predict(X_transformed)
    
    def fit_predict(self, X, y):
        """Fit the pipeline and predict on the same data."""
        self.fit(X, y)
        return self.predict(X)
    
    def score(self, X, y):
        """
        Score the pipeline (if final estimator has score method).
        
        Typically returns R² for regression, accuracy for classification.
        """
        if not hasattr(self._final_estimator, 'score'):
            raise TypeError("Final estimator has no score method.")
        
        X_transformed = X
        for name, transformer in self._transformers:
            X_transformed = transformer.transform(X_transformed)
        
        return self._final_estimator.score(X_transformed, y)
    
    def get_params(self, deep=True):
        """
        Get parameters of all steps.
        
        Useful for inspection and hyperparameter tuning.
        """
        params = {'steps': self.steps}
        
        if deep:
            for name, step in self.steps:
                if hasattr(step, 'get_params'):
                    step_params = step.get_params()
                    for key, value in step_params.items():
                        params[f'{name}__{key}'] = value
        
        return params
    
    def __repr__(self):
        step_reprs = [f"  ('{name}', {step.__class__.__name__})" 
                      for name, step in self.steps]
        return f"Pipeline([\n" + ",\n".join(step_reprs) + "\n])"
    
    def __getitem__(self, key):
        """
        Access steps by name or index.
        
        >>> pipe['scaler']  # by name
        >>> pipe[0]         # by index
        >>> pipe[:-1]       # slice (returns new Pipeline)
        """
        if isinstance(key, str):
            return self.named_steps[key]
        elif isinstance(key, int):
            return self.steps[key][1]
        elif isinstance(key, slice):
            return Pipeline(self.steps[key])
        else:
            raise KeyError(f"Invalid key type: {type(key)}")