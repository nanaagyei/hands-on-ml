"""
Linear regression from scratch using NumPy.
Replicating scikit-Learn's LinearRegression class and functionality.
"""

import numpy as np
from .optimizers import BatchGradientDescent, StochasticGradientDescent, MiniBatchGradientDescent

class SimpleLinearRegression:
    """
    A Simple Linear Regression using the Normal Equation.

    This is the closed-form solution - hence there's no iteration required.
    This is equivalent to scikit-Learn's LinearRegression class with default parameters.

    Attributes:
        weights_ : np.ndarray
            The weights of the linear regression model.
        bias_ : float
            The bias of the linear regression model.

    Methods:
        fit(X, y) -> None:
            Fit the model to the data.
        predict(X) -> np.ndarray:
            Predict the target values for the given input data.
        score(X, y) -> float:
            Calculate the R-squared score of the model.
    """

    def __init__(self):
        """
        Initialize the SimpleLinearRegression model
        """
        self.weights_ = None # weights of the linear regression model
        self.bias_ = None # bias of the linear regression model
        self._theta = None # combination of weights and bias for the linear regression model
    
    def fit(self, X, y) -> None:
        """
        Fit the model using the Normal Equation.
        
        The Normal Equation: θ = (X^T X)^(-1) X^T y
        
        Parameters:
            X (ndarray): Training data of shape (n_samples, n_features)
            y (ndarray): Target values of shape (n_samples,)
        
        Returns:
            self: Returns the instance itself
        """

        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1) # reshape X to (n_samples, 1) if it's a 1D array
        
        n_samples, n_features = X.shape # get the number of samples and features
        X_b = np.c_[np.ones((n_samples, 1)), X] # add a column of ones to X for the bias term

        

        self._theta = np.linalg.pinv(X_b) @ y

        self.bias_ = self._theta[0]
        self.weights_ = self._theta[1:]

        return self
    
    def predict(self, X) -> np.ndarray:
        """
        Make predictions using learned parameters.

        Prediction: ŷ = X @ weights + bias
        
        Parameters:
            X (ndarray): Samples of shape (n_samples, n_features)
        
        Returns:
            ndarray: Predicted values of shape (n_samples, 1)
        """

        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return X @ self.weights_ + self.bias_
    
    def score(self, X, y) -> float:
        """
        Calculate R² score (coefficient of determination).
        
        R² = 1 - (SS_res / SS_tot)
        
        Where:
            SS_res = Σ(y - ŷ)² - residual sum of squares
            SS_tot = Σ(y - ȳ)² - total sum of squares
        
        Parameters:
            X (ndarray): Test samples
            y (ndarray): True values
        
        Returns:
            float: R² score (1.0 is perfect, 0.0 is the baseline)
        """
        y = np.array(y)
        y_pred = self.predict(X)

        SS_res = np.sum((y - y_pred) ** 2)
        SS_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (SS_res / SS_tot)


class LinearRegression:
    """
    Linear Regression using the Normal Equation with Pseudoinverse.
    
    This implementation mirrors sklearn's LinearRegression.
    Uses SVD-based pseudoinverse for numerical stability.
    
    Parameters:
        fit_intercept (bool): Whether to calculate the intercept. 
                              Default True.
    
    Attributes:
        coef_ (ndarray): Learned coefficients (weights)
        intercept_ (float): Learned intercept (bias)
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y) -> None:
        """
        Fit linear model using pseudoinverse.
        
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
            X = np.c_[np.ones(n_samples, 1), X]
            theta = np.linalg.pinv(X) @ y
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            theta = np.linalg.pinv(X) @ y
            self.coef_ = theta
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
    
    def score(self, X, y) -> float:
        """
        Calculate R² score (coefficient of determination).
        
        R² = 1 - (SS_res / SS_tot)
        
        Where:
            SS_res = Σ(y - ŷ)² - residual sum of squares
            SS_tot = Σ(y - ȳ)² - total sum of squares
        
        Parameters:
            X (ndarray): Test samples
            y (ndarray): True values
        
        Returns:
            float: R² score (1.0 is perfect, 0.0 is the baseline)
        """
        y = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)

        SS_res = np.sum((y - y_pred) ** 2)
        SS_tot = np.sum((y - np.mean(y)) ** 2)

        if SS_tot == 0:
            return 1.0 if SS_res == 0 else 0.0

        return 1 - (SS_res / SS_tot)
    
    def get_params(self) -> dict:
        """
        Get the model parameters.
        
        Returns:
            dict: Model parameters
        """
        return {
            "coef_": self.coef_,
            "intercept_": self.intercept_
        }
    
    def set_params(self, **params) -> None:
        """
        Set the model parameters.
        
        Parameters:
            **params: Model parameters
        
        Returns:
            self: Returns the instance itself
        """
        if "coef_" in params:
            self.coef_ = params["coef_"]
        if "intercept_" in params:
            self.intercept_ = params["intercept_"]
        return self
    
    def __repr__(self) -> str:
        """
        Return a string representation of the model.
        
        Returns:
            str: String representation of the model
        """
        return f"LinearRegression(coef_={self.coef_}, intercept_={self.intercept_})"
    

class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent.
    
    This implementation mirrors sklearn's LinearRegression.
    Uses Gradient Descent to find the optimal parameters.
    
    Parameters:
        learning_rate (float): Learning rate. Default 0.01
        n_iterations (int): Number of iterations. Default 1000
        tolerance (float): Tolerance for convergence. Default 1e-6
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.coef_ = None
        self.intercept_ = None
        self.cost_history_ = []
    
    def fit(self, X, y):
        """
        Fit the model using gradient descent.
        
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

        X_b = np.c_[np.ones((n_samples, 1)), X]

        optimizer = BatchGradientDescent(
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations,
            tolerance=self.tolerance
        )

        theta = optimizer.optimize(X_b, y)

        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        self.cost_history_ = optimizer.cost_history

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
    
    def score(self, X, y) -> float:
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

class LinearRegressionSGD:
    """
    Linear Regression using Stochastic Gradient Descent.
    
    This implementation mirrors sklearn's LinearRegression.
    Uses Stochastic Gradient Descent to find the optimal parameters.
    
    Parameters:
        learning_rate (float): Learning rate. Default 0.01
        n_epochs (int): Number of epochs. Default 50
        shuffle (bool): Shuffle data each epoch. Default True
        random_state (int): Random state for reproducibility. Default None
    """

    def __init__(self, learning_rate=0.01, n_epochs=50, shuffle=True, random_state=None):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.cost_history_ = []
    
    def fit(self, X, y):
        """
        Fit the model using stochastic gradient descent.
        
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

        X_b = np.c_[np.ones(n_samples), X]

        optimizer = StochasticGradientDescent(
            learning_rate=self.learning_rate,
            n_epochs=self.n_epochs,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        theta = optimizer.optimize(X_b, y)
        self.intercept_ = theta[0]
        self.coef_ = theta[1:]
        self.cost_history_ = optimizer.cost_history

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
    
    def score(self, X, y) -> float:
        """
        Calculate R² score (coefficient of determination).
        
        R² = 1 - (SS_res / SS_tot)
        
        Where:
            SS_res = Σ(y - ŷ)² - residual sum of squares
            SS_tot = Σ(y - ȳ)² - total sum of squares
        
        Parameters:
            X (ndarray): Test samples
            y (ndarray): True values
        
        Returns:
            float: R² score (1.0 is perfect, 0.0 is the baseline)
        """

        y = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return (1 - (ss_res / ss_tot)) if ss_tot != 0 else (1.0 if ss_res == 0 else 0.0)
        