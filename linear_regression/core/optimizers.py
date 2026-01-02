"""
Gradient Descent Optimizers for Linear Regression.

This module contains the following optimizers:
- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-Batch Gradient Descent
"""

import numpy as np

class BatchGradientDescent:
    """
    Batch Gradient Descent optimizer.
    
    Uses ALL samples to compute gradient at each step.
    Guaranteed to converge to global minimum for convex functions (like MSE).
    
    Parameters:
        learning_rate (float): Step size. Default 0.01
        n_iterations (int): Maximum iterations. Default 1000
        tolerance (float): Stop if improvement < tolerance. Default 1e-6
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.cost_history = [] # to track cost function values
        self.theta_history = [] # keep track of theta values for visualization
    
    def optimize(self, X, y, theta_init=None):
        """
        Run gradient descent optimization.
        
        Parameters:
            X (ndarray): Feature matrix with bias column, shape (n_samples, n_features+1)
            y (ndarray): Target vector, shape (n_samples,)
            theta_init (ndarray): Initial parameters. Random if None.
        
        Returns:
            ndarray: Optimized parameters theta. Shape (n_features,)
        """

        n_samples, n_features = X.shape

        # initialize theta if not provided
        if theta_init is None:
            theta = np.random.randn(n_features) * 0.01
        else:
            theta = theta_init.copy()
        
        self.cost_history = []
        self.theta_history = [theta.copy()]

        for iteration in range(self.n_iterations):
            # Forward pass: compute predictions
            predictions = X @ theta

            # Compute error
            error = predictions - y

            # Compute gradient
            gradient = (1/ n_samples) * (X.T @ error)

            # Update parameters: theta
            theta = theta - self.learning_rate * gradient

            # Compute and store cost and theta values
            cost = (1/(2 * n_samples)) * np.sum(error ** 2)
            self.cost_history.append(cost)
            self.theta_history.append(theta.copy())

            # Check for convergence
            if iteration > 0 and len(self.cost_history) > 1:
                improvement = abs(self.cost_history[-1] - self.cost_history[-2])
                if 0 <= improvement < self.tolerance:
                    print(f"Converged after {iteration} iterations")
                    break

        return theta

class StochasticGradientDescent:
    """
    Stochastic Gradient Descent optimizer.
    
    Uses a single sample to compute gradient at each step.
    Faster than batch gradient descent, but less stable.
    
    Parameters:
        learning_rate (float): Step size. Default 0.01
        n_epochs (int): Number of passes through the data
        shuffle (bool): Shuffle data each epoch (important!)
        random_state (int): For reproducibility
    """

    def __init__(self, learning_rate=0.01, n_epochs=50, shuffle=True, random_state=None):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.random_state = random_state
        self.cost_history = []
    
    def optimize(self, X, y, theta_init=None):
        """
        Run stochastic gradient descent optimization.
        
        Parameters:
            X (ndarray): Feature matrix with bias column, shape (n_samples, n_features+1)
            y (ndarray): Target vector, shape (n_samples,)
            theta_init (ndarray): Initial parameters. Random if None.
        
        Returns:
            ndarray: Optimized parameters theta. Shape (n_features,)
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        # initialize theta if not provided
        if theta_init is None:
            theta = rng.standard_normal(n_features) * 0.01
        else:
            theta = theta_init.copy()
        
        self.cost_history = []

        for epoch in range(self.n_epochs):

            if self.shuffle:
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)
            
            for i in indices:
                xi = X[i:i+1] # get the i-th sample (1, n_features)
                yi = y[i:i+1] # get the i-th target value (1,)

                # Forward pass: compute predictions
                predictions = xi @ theta

                # Compute error for the i-th sample
                error = predictions - yi

                # Compute gradient
                gradient = xi.T @ error

                # Update parameters: theta
                theta = theta - self.learning_rate * gradient

            # Compute and store cost and theta values
            cost = (1/(2 * n_samples)) * np.sum((X @ theta - y) ** 2)
            self.cost_history.append(cost)

        return theta


class MiniBatchGradientDescent:
    """
    Mini-Batch Gradient Descent optimizer.
    
    Uses a small batch of samples to compute gradient at each step.
    Faster than stochastic gradient descent, but less stable.
    This is what is used in practice for deep learning models.
    
    Parameters:
        learning_rate (float): Step size. Default 0.01
        batch_size (int): Size of the mini-batch. Default 32
        n_epochs (int): Number of passes through the data
        shuffle (bool): Shuffle data each epoch (important!)
        random_state (int): For reproducibility
    """

    def __init__(self, learning_rate=0.01, batch_size=32, n_epochs=50, shuffle=True, random_state=True):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.random_state = random_state
        self.cost_history = []

    
    def optimize(self, X, y, theta_init=None):
        """
        Run mini-batch gradient descent optimization.
        
        Parameters:
            X (ndarray): Feature matrix with bias column, shape (n_samples, n_features+1)
            y (ndarray): Target vector, shape (n_samples,)
            theta_init (ndarray): Initial parameters. Random if None.
        
        Returns:
            ndarray: Optimized parameters theta. Shape (n_features,)
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        if theta_init is None:
            theta = rng.standard_normal(n_features) * 0.01
        else:
            theta = theta_init.copy()
        
        self.cost_history = []

        for epoch in range(self.n_epochs):

            if self.shuffle:
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)
            
            for start_index in range(0, n_samples, self.batch_size):
                end_index = min(start_index + self.batch_size, n_samples)
                batch_indices = indices[start_index:end_index]
            
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                predictions = X_batch @ theta

                error = predictions - y_batch
                gradient = (1 / len(batch_indices)) * (X_batch.T @ error)

                theta = theta - self.learning_rate * gradient
            
            epoch_cost = (1/ (2 * n_samples)) * np.sum((X @ theta - y) ** 2)
            self.cost_history.append(epoch_cost)

        return theta

class SGDWithSchedule:
    """
    Stochastic Gradient Descent with Learning Rate Schedule.

    Uses a learning rate schedule to adjust the learning rate at each step.
    
    Parameters:
        learning_rate (float): Step size. Default 0.01
        n_epochs (int): Number of passes through the data
        schedule (str): Learning rate schedule. Default 'constant'
        decay_rate (float): Decay rate for the learning rate. Default 0.01
        shuffle (bool): Shuffle data each epoch (important!)
        random_state (int): For reproducibility
    """
    
    def __init__(self, learning_rate=0.1, n_epochs=50, schedule='constant', decay_rate=0.01, shuffle=True, random_state=None):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.schedule = schedule
        self.decay_rate = decay_rate
        self.shuffle = shuffle
        self.random_state = random_state
        self.cost_history = []
        self.learning_rate_history = []
    
    
    def _get_learning_rate(self, epoch, iteration):
        """
        Get the learning rate for the current epoch and iteration.
        
        Parameters:
            epoch (int): Current epoch
            iteration (int): Current iteration
        
        Returns:
            float: Learning rate
        """
        t = epoch

        if self.schedule == 'constant':
            learning_rate = self.learning_rate
        elif self.schedule == 'inverse':
            learning_rate = self.learning_rate / (1 + self.decay_rate * t)
        elif self.schedule == 'exponential':
            learning_rate = self.learning_rate * np.exp(-self.decay_rate * t)
        elif self.schedule == 'polynomial':
            learning_rate = self.learning_rate * (1 - self.decay_rate * t) ** 2
        elif self.schedule == 'cosine':
            learning_rate = self.learning_rate * (1 + np.cos(np.pi * t / self.n_epochs)) / 2
        elif self.schedule == 'sigmoid':
            learning_rate = self.learning_rate / (1 + np.exp(-self.decay_rate * t))
        elif self.schedule == 'step':
            learning_rate = self.learning_rate * (1 - self.decay_rate * t)
        elif self.schedule == 'cyclic':
            learning_rate = self.learning_rate * (1 + np.sin(2 * np.pi * t / self.n_epochs)) / 2
        else:
            raise ValueError(f"Invalid learning rate schedule: {self.schedule}")

        return learning_rate
    
    def optimize(self, X, y, theta_init=None):
        """
        Run stochastic gradient descent with learning rate schedule.

        Parameters:
            X (ndarray): Feature matrix with bias column, shape (n_samples, n_features+1)
            y (ndarray): Target vector, shape (n_samples,)
            theta_init (ndarray): Initial parameters. Random if None.
        
        Returns:
            ndarray: Optimized parameters theta. Shape (n_features,)
        """
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        if theta_init is None:
            theta = rng.standard_normal(n_features) * 0.01
        else:
            theta = theta_init.copy()
        
        self.cost_history = []
        self.learning_rate_history = []

        for epoch in range(self.n_epochs):
            current_learning_rate = self._get_learning_rate(epoch, 0)
            self.learning_rate_history.append(current_learning_rate)

            if self.shuffle:
                indices = rng.permutation(n_samples)
            else:
                indices = np.arange(n_samples)
            
            for i in indices:
                xi = X[i:i+1] # get the i-th sample (1, n_features)
                yi = y[i:i+1] # get the i-th target value (1,)

                predictions = xi @ theta
                error = predictions - yi
                gradient = xi.T @ error
                theta = theta - current_learning_rate * gradient
            
            epoch_cost = (1/ (2 * n_samples)) * np.sum((X @ theta - y) ** 2)
            self.cost_history.append(epoch_cost)

        return theta