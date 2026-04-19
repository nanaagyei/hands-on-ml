import numpy as np


class LinearSVM:
    """
    Soft-margin linear SVM via subgradient descent on the primal.
    
    Optimizes:
        min_{w,b}  (1/2)||w||² + C * Σ max(0, 1 - yᵢ(w·xᵢ + b))
    
    The second term is the hinge loss — it's zero when a point is correctly
    classified AND outside the margin. It grows linearly for violations.
    
    Why subgradient and not gradient?
    The hinge loss max(0, 1-yf(x)) has a kink at yf(x)=1 — not differentiable
    there. We use the subgradient: gradient on either side, 0 at the kink.
    
    This is the workhorse for text classification (high-dim, sparse data).
    For non-linear problems, use KernelSVM with RBF kernel.
    """
    
    def __init__(self, C=1.0, learning_rate=0.001, n_epochs=1000,
                 tol=1e-4, random_state=42):
        """
        Parameters
        ----------
        C : float
            Regularization parameter. 
            Large C → small margin, fewer violations (risk overfitting).
            Small C → large margin, more violations (risk underfitting).
            Think of it as "how much do I care about misclassifications?"
        
        learning_rate : float
            Step size for gradient updates. 
            Too large → diverges. Too small → takes forever.
            For scaled data, 0.001 is a good starting point.
        
        n_epochs : int
            Full passes through the training data.
        
        tol : float
            Stop early if weight updates are smaller than this.
        """
        self.C = C
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.tol = tol
        self.random_state = random_state
        
        self.w_ = None       # Weight vector
        self.b_ = None       # Bias
        self.loss_history_ = []
        self.n_support_vectors_ = None
    
    def fit(self, X, y):
        """
        Train the SVM.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) with values in {-1, +1}
        
        IMPORTANT: y must be {-1, +1}, not {0, 1}.
        The math requires this. We handle the conversion internally.
        """
        X = np.asarray(X, dtype=np.float64)
        y = self._validate_labels(y)
        
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        
        # Initialize weights small and random, bias at 0
        self.w_ = rng.randn(n_features) * 0.01
        self.b_ = 0.0
        
        for epoch in range(self.n_epochs):
            w_old = self.w_.copy()
            
            # Shuffle data each epoch (crucial for SGD-style convergence)
            indices = rng.permutation(n_samples)
            
            epoch_loss = 0.0
            
            for i in indices:
                xi, yi = X[i], y[i]
                
                # Functional margin: how far is this point from the boundary,
                # scaled by the true label. If > 1, correctly outside margin.
                margin = yi * (np.dot(self.w_, xi) + self.b_)
                
                # Hinge loss for this sample
                hinge = max(0, 1 - margin)
                
                # Subgradient of the full objective:
                # d/dw [(1/2)||w||² + C * max(0, 1 - y(w·x+b))]
                if margin >= 1:
                    # Point correctly classified, outside margin
                    # Only the regularization term contributes
                    dw = self.w_
                    db = 0.0
                else:
                    # Point inside margin or misclassified
                    # Both terms contribute
                    dw = self.w_ - self.C * yi * xi
                    db = -self.C * yi
                
                # Update
                self.w_ -= self.learning_rate * dw
                self.b_ -= self.learning_rate * db
                
                # Full loss: regularization + hinge
                epoch_loss += 0.5 * np.dot(self.w_, self.w_) + self.C * hinge
            
            self.loss_history_.append(epoch_loss / n_samples)
            
            # Early stopping: if weights barely moved, we've converged
            weight_change = np.max(np.abs(self.w_ - w_old))
            if weight_change < self.tol:
                print(f"Converged at epoch {epoch+1}")
                break
        
        # Count support vectors: points inside or on the margin
        # These are the points that "support" the boundary
        margins = y * (X @ self.w_ + self.b_)
        self.n_support_vectors_ = int(np.sum(margins <= 1 + 1e-4))
        self.support_vector_indices_ = np.where(margins <= 1 + 1e-4)[0]
        
        return self
    
    def decision_function(self, X):
        """
        Raw score: w·x + b
        
        Positive → predicted class +1
        Negative → predicted class -1
        Magnitude → distance from decision boundary (confidence)
        """
        return X @ self.w_ + self.b_
    
    def predict(self, X):
        """Returns {-1, +1} predictions."""
        return np.sign(self.decision_function(X)).astype(int)
    
    def predict_proba_approx(self, X):
        """
        SVMs don't naturally give probabilities.
        This uses Platt scaling approximation: sigmoid of decision score.
        
        For real probability calibration, use sklearn's CalibratedClassifierCV.
        This is just a quick-and-dirty version.
        """
        scores = self.decision_function(X)
        # Sigmoid to squash to [0, 1]
        return 1 / (1 + np.exp(-scores))
    
    def _validate_labels(self, y):
        """Convert {0,1} to {-1,+1} if needed. Validate binary."""
        y = np.asarray(y)
        unique = np.unique(y)
        
        if set(unique) == {0, 1}:
            # Convert 0→-1, 1→+1
            return 2 * y - 1
        elif set(unique) == {-1, 1} or set(unique).issubset({-1, 0, 1}):
            return y.astype(np.float64)
        else:
            raise ValueError(f"Labels must be binary. Got: {unique}")
    
    def get_params(self):
        return {
            'C': self.C,
            'learning_rate': self.learning_rate,
            'n_epochs': self.n_epochs,
            'tol': self.tol,
            'random_state': self.random_state,
        }
    
    def score(self, X, y):
        """Accuracy score."""
        y = self._validate_labels(y)
        return np.mean(self.predict(X) == y)