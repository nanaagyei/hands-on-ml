"""
Regularized Linear Regression for Machine Learning.

This module contains the following regularized linear regression models:
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
"""

import numpy as np


class RidgeRegression:
    """
    Ridge Regression (L2 Regularization).

    Adds squared magnitude of weights to the cost function.
    This shrinks weights but never zeros them out.

    Cost = MSE + α * Σ(θⱼ²)

    Parameters:
        alpha (float): Regularization strength. α=0 is ordinary least squares.
        fit_intercept (bool): Whether to fit intercept term.

    Attributes:
        coef_ (ndarray): Feature weights
        intercept_ (float): Bias term
    """

    def __init__(self, alpha=0.1, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit the Ridge Regression model using Closed-Form Solution.

        The Ridge Regression closed-form solution:
        θ = (X^T X + αI)^(-1) X^T y

        Parameters:
            X (ndarray): Training data, shape (n_samples, n_features)
            y (ndarray): Target values, shape (n_samples,)

        Returns:
            self: Returns the instance itself
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if self.fit_intercept:
            # Center the data (subtract means)
            # This lets us solve without bias in the regularized system
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)

            X_centered = X - X_mean
            y_centered = y - y_mean

            # Solve for weights on centered data
            # θ = (X^T X + αI)^(-1) X^T y
            XtX = X_centered.T @ X_centered
            regularization = self.alpha * np.eye(n_features)
            Xty = X_centered.T @ y_centered

            self.coef_ = np.linalg.solve(XtX + regularization, Xty)

            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            # solve for weights without bias - straightforward
            XtX = X.T @ X
            regularization = self.alpha * np.eye(n_features)
            Xty = X.T @ y

            self.coef_ = np.linalg.solve(XtX + regularization, Xty)
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        """
        Make predictions using the learned parameters.

        Prediction: ŷ = X @ coef_ + intercept_

        Parameters:
            X (ndarray): Samples of shape (n_samples, n_features)

        Returns:
            ndarray: Predicted values of shape (n_samples, 1)
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination).

        R² = 1 - (SS_res / SS_tot)

        Where:
            SS_res = Σ(y - ŷ)² - residual sum of squares
            SS_tot = Σ(y - ȳ)² - total sum of squares

        Parameters:
            X (ndarray): Test samples
            y (ndarray): True values
        """
        y = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)

    def get_params(self):
        """
        Get the model parameters.

        Returns:
            dict: Model parameters
        """
        return {
            "coef_": self.coef_,
            "intercept_": self.intercept_
        }


class LassoRegression:
    """
    Lasso Regression (L1 Regularization).

    Adds absolute magnitude of weights to the cost function.
    Uses coordinate descent optimization to find the optimal weights.
    This can shrink weights to zero but never completely remove them - automatically performs feature selection.

    Cost = MSE + α * Σ(|θⱼ|)

    Parameters:
        alpha (float): Regularization strength. α=0 is ordinary least squares.
        max_iter (int): Maximum number of iterations. Default 1000.
        tol (float): Tolerance for convergence. Default 1e-4.
        fit_intercept (bool): Whether to fit intercept term.

    Attributes:
        coef_ (ndarray): Feature weights
        intercept_ (float): Bias term
    """

    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-4, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.n_iter = None

    def _soft_threshold(self, rho, lambda_):
        """
        Soft thresholding function for L1 regularization.
        S(rho, λ) = sign(rho) * max(|rho| - λ, 0)

        Returns:
            float: Thresholded value
        """

        return np.sign(rho) * max(abs(rho) - lambda_, 0)

    def fit(self, X, y):
        """
        Fit the Lasso Regression model using Coordinate Descent.
        For each feature j, update:
            1. Compute residual without feature j's contribution
            2. Compute what weight would be without regularization
            3. Apply soft-thresholding
            θⱼ = S(X^T (y - Xθ) + αθⱼ, α)

        Parameters:
            X (ndarray): Training data, shape (n_samples, n_features)
            y (ndarray): Target values, shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y
            X_mean = np.zeros(n_features)
            y_mean = 0.0

        # Precompute X^T X diagonal and X^T y
        # This speeds up coordinate descent significantly
        # ||X_j||^2 for each feature
        X_sq_sum = np.sum(X_centered ** 2, axis=0)

        # Initialize weights
        self.coef_ = np.zeros(n_features)

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()  # save old weights for convergence check

            for j in range(n_features):
                # Compute residual (what's left after all other features)
                # r_j = y - X @ coef + X_j * coef_j
                residual = y_centered - X_centered @ self.coef_ + \
                    X_centered[:, j] * self.coef_[j]

                # What would the weight be without regularization?
                # rho_j = X_j^T @ residual / ||X_j||^2
                rho_j = X_centered[:, j] @ residual / (X_sq_sum[j] + 1e-10)

                # Apply soft-thresholding
                # For Lasso cost: (1/(2*n)) * ||y - Xw||^2 + alpha * ||w||_1
                # The threshold needs to account for n_samples to match scikit-learn
                threshold = (self.alpha * n_samples) / (X_sq_sum[j] + 1e-10)
                self.coef_[j] = self._soft_threshold(rho_j, threshold)

            # Check convergence
            coef_change = np.max(np.abs(self.coef_ - coef_old))
            if coef_change < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter

        # Compute intercept
        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0

        return self

    def predict(self, X):
        """
        Make predictions using the learned parameters.

        Prediction: ŷ = X @ coef_ + intercept_

        Parameters:
            X (ndarray): Samples of shape (n_samples, n_features)

        Returns:
            ndarray: Predicted values of shape (n_samples, 1)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination).

        R² = 1 - (SS_res / SS_tot)

        Where:
            SS_res = Σ(y - ŷ)² - residual sum of squares
            SS_tot = Σ(y - ȳ)² - total sum of squares

        Parameters:
            X (ndarray): Test samples
            y (ndarray): True values
        """
        y = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)

    def get_params(self):
        """
        Get the model parameters.

        Returns:
            dict: Model parameters
        """
        return {
            "coef_": self.coef_,
            "intercept_": self.intercept_
        }


class ElasticNetRegression:
    """
    Elastic Net Regression (L1 + L2 Regularization).
    
    Combines L1 and L2 regularization.
    Cost = MSE + α * (L1 + L2)
    
    Parameters:
        alpha (float): Regularization strength. α=0 is ordinary least squares.
        l1_ratio (float): Mixing parameter between L1 and L2 regularization.
        fit_intercept (bool): Whether to fit intercept term.
        max_iter (int): Maximum number of iterations. Default 1000.
        tol (float): Tolerance for convergence. Default 1e-4.
    """
    def __init__(self, alpha=0.1, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.n_iter = None

    def _soft_threshold(self, rho, lambda_):
        """
        Soft thresholding function for Elastic Net regularization.
        S(rho, λ) = sign(rho) * max(|rho| - λ, 0)

        Returns:
            float: Thresholded value
        """
        return np.sign(rho) * max(abs(rho) - lambda_, 0)
    
    def fit(self, X, y):
        """
        Fit using coordinate descent with both L1 and L2 penalties.

        For each feature j, update:
            1. Compute residual without feature j's contribution
            2. Compute what weight would be without regularization
            3. Apply soft-thresholding
            θⱼ = S(X^T (y - Xθ) + αθⱼ, α)

        Parameters:
            X (ndarray): Training data, shape (n_samples, n_features)
            y (ndarray): Target values, shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape

        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X_centered = X - X_mean
            y_centered = y - y_mean
        else:
            X_centered = X
            y_centered = y
            X_mean = np.zeros(n_features)
            y_mean = 0.0
        
        X_sq_sum = np.sum(X_centered ** 2, axis=0)
        X_sq_sum_l1 = np.sum(np.abs(X_centered), axis=0)

        # L1 and L2 components
        l1_penalty = self.alpha * self.l1_ratio
        l2_penalty = self.alpha * (1 - self.l1_ratio)

        # Initialize weights
        self.coef_ = np.zeros(n_features)
        
        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy() # save old weights for convergence check
            
            for j in range(n_features):
                # Compute residual (what's left after all other features)
                # r_j = y - X @ coef + X_j * coef_j
                residual = y_centered - X_centered @ self.coef_ + X_centered[:, j] * self.coef_[j]
                
                # The denominator now includes L2 penalty
                denominator = X_sq_sum[j] + l2_penalty + 1e-10
                
                rho_j = X_centered[:, j] @ residual / denominator
                
                # Soft-threshold with L1 penalty
                threshold = (l1_penalty * X_sq_sum_l1[j]) / denominator
                self.coef_[j] = self._soft_threshold(rho_j, threshold)
            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                self.n_iter_ = iteration + 1
                break
        else:
            self.n_iter_ = self.max_iter
        
        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the learned parameters.

        Prediction: ŷ = X @ coef_ + intercept_

        Parameters:
            X (ndarray): Samples of shape (n_samples, n_features)

        Returns:
            ndarray: Predicted values of shape (n_samples, 1)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X @ self.coef_ + self.intercept_
    
    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination).

        R² = 1 - (SS_res / SS_tot)

        Where:
            SS_res = Σ(y - ŷ)² - residual sum of squares
            SS_tot = Σ(y - ȳ)² - total sum of squares

        Parameters:
            X (ndarray): Test samples
            y (ndarray): True values
        """
        y = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)
    
    def get_params(self):
        """
        Get the model parameters.

        Returns:
            dict: Model parameters
        """
        return {
            "coef_": self.coef_,
            "intercept_": self.intercept_
        }