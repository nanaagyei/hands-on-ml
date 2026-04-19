import numpy as np

from .kernels import rbf_kernel, linear_kernel, polynomial_kernel, get_kernel


class KernelSVM:
    """
    Kernel SVM via Sequential Minimal Optimization (SMO).

    Solves the dual problem:
        maximize  Σαᵢ - (1/2) ΣᵢΣⱼ αᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
        subject to  Σαᵢyᵢ = 0,   0 ≤ αᵢ ≤ C

    Prediction:
        f(x) = Σ αᵢyᵢK(xᵢ, x) + b
        ŷ = sign(f(x))

    Why the dual?
    1. The kernel trick only works in the dual — dot products become K(xᵢ,xⱼ)
    2. Complexity is O(n_support_vectors) at predict time, not O(n_features)
    3. The number of variables (n_samples) is fixed regardless of kernel dimension

    SMO reference: Platt (1998) "Sequential Minimal Optimization"
    """

    def __init__(self, C=1.0, kernel='rbf', gamma='scale', degree=3,
                 coef0=1.0, tol=1e-3, max_passes=5, random_state=42):
        """
        Parameters
        ----------
        C : float
            Soft-margin regularization. Same intuition as LinearSVM.

        kernel : str or callable
            'linear', 'rbf', 'poly', or a callable K(X1, X2) -> matrix

        gamma : float or 'scale' or 'auto'
            RBF/poly kernel bandwidth.
            'scale' → 1 / (n_features * X.var())   ← sklearn default
            'auto'  → 1 / n_features
            A float → use directly

        degree : int
            Polynomial kernel degree (ignored for other kernels).

        coef0 : float
            Polynomial kernel free term r in (γx·z + r)^d.

        tol : float
            KKT violation tolerance. Points with violation < tol are
            considered "satisfying" KKT conditions.

        max_passes : int
            Stop after this many passes where no α changed.
            This is the convergence criterion for the outer loop.
            Higher → more thorough but slower. 5 is standard.

        random_state : int
            For reproducible α₂ selection when the heuristic ties.
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.random_state = random_state

        # Set after fitting
        self.alpha_ = None          # Lagrange multipliers (n_samples,)
        self.b_ = None              # Bias term
        self.support_vectors_ = None       # X[sv_idx]
        self.support_vector_labels_ = None # y[sv_idx]
        self.support_vector_alphas_ = None # alpha[sv_idx]
        self.n_support_ = None
        self._kernel_fn = None      # Compiled kernel function
        self._X_train = None        # Needed for prediction
        self.objective_history_ = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Train via SMO.

        X : (n_samples, n_features)
        y : binary labels, will be converted to {-1, +1}
        """
        X = np.asarray(X, dtype=np.float64)
        y = self._validate_labels(y)
        n = len(y)

        rng = np.random.RandomState(self.random_state)

        # Compile kernel function with resolved gamma
        self._kernel_fn = self._build_kernel(X)

        # Precompute the full kernel matrix K[i,j] = K(xᵢ, xⱼ)
        # For large datasets this is expensive (O(n²) memory).
        # Production SVMs use caching strategies — we'll discuss this.
        print(f"  Computing {n}×{n} kernel matrix...")
        K = self._kernel_fn(X, X)
        print(f"  Kernel matrix: {K.shape}, "
              f"memory: {K.nbytes / 1e6:.1f} MB")

        # Initialize all αᵢ = 0, b = 0
        alpha = np.zeros(n)
        b = 0.0

        # Cache of decision function values f(xᵢ) = Σⱼ αⱼyⱼK(j,i) + b
        # We update this incrementally instead of recomputing from scratch
        # each time — this is what makes SMO practical.
        f_cache = np.zeros(n)   # Initially all 0 (since all α=0)

        passes = 0    # Consecutive passes with no α update
        iteration = 0

        print(f"  Starting SMO (max_passes={self.max_passes}, tol={self.tol})")

        while passes < self.max_passes:
            num_changed = 0

            for i in range(n):
                # -------------------------------------------------------
                # Step 1: Check KKT violation for point i
                # -------------------------------------------------------
                Ei = f_cache[i] - y[i]   # Prediction error

                # KKT conditions (with tolerance):
                #   αᵢ = 0  and  yᵢf(xᵢ) >= 1    → no violation
                #   αᵢ = C  and  yᵢf(xᵢ) <= 1    → no violation
                #   0 < αᵢ < C  and  yᵢf(xᵢ) = 1 → no violation
                #
                # Violation ↔ r_i = yᵢ * Eᵢ = yᵢf(xᵢ) - 1 is wrong sign
                r_i = y[i] * Ei

                kkt_violated = (
                    (r_i < -self.tol and alpha[i] < self.C) or
                    (r_i > self.tol  and alpha[i] > 0)
                )

                if not kkt_violated:
                    continue

                # -------------------------------------------------------
                # Step 2: Pick α₂ using the maximum step heuristic
                # Platt's heuristic: pick j that maximizes |Ei - Ej|
                # Large step → faster progress toward the optimum.
                # -------------------------------------------------------
                j = self._select_j(i, Ei, f_cache, y, n, rng)

                Ej = f_cache[j] - y[j]

                # Save old alphas before update
                alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                # -------------------------------------------------------
                # Step 3: Compute the bounds [L, H] for α_j
                # The box constraint 0 ≤ α ≤ C plus the linear constraint
                # αᵢyᵢ + αⱼyⱼ = const pins α_j to a line segment.
                # -------------------------------------------------------
                L, H = self._compute_LH(alpha[i], alpha[j], y[i], y[j])

                if L >= H:
                    continue   # No room to optimize, skip

                # -------------------------------------------------------
                # Step 4: Compute η (the curvature / second derivative)
                # η = K₁₁ + K₂₂ - 2K₁₂
                # This is the denominator of the analytic update.
                # η > 0 ensures we have a maximum (concave dual objective).
                # -------------------------------------------------------
                eta = K[i, i] + K[j, j] - 2 * K[i, j]

                if eta <= 1e-10:
                    # Degenerate case: K₁₁ + K₂₂ = 2K₁₂, meaning x_i = x_j
                    # in feature space. Skip to avoid divide-by-zero.
                    continue

                # -------------------------------------------------------
                # Step 5: Analytic update for α_j (unconstrained)
                # α_j^new = α_j^old + y_j * (E_i - E_j) / η
                # -------------------------------------------------------
                alpha[j] = alpha_j_old + y[j] * (Ei - Ej) / eta

                # Clip to [L, H]
                alpha[j] = np.clip(alpha[j], L, H)

                # If α_j barely moved, skip updating α_i too
                if abs(alpha[j] - alpha_j_old) < 1e-5:
                    alpha[j] = alpha_j_old  # Revert
                    continue

                # -------------------------------------------------------
                # Step 6: Update α_i using the linear constraint
                # αᵢyᵢ + αⱼyⱼ = const  →  αᵢ = αᵢ_old + yᵢyⱼ(αⱼ_old - αⱼ_new)
                # -------------------------------------------------------
                alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j])

                # -------------------------------------------------------
                # Step 7: Update bias b
                # b must satisfy f(xᵢ) = yᵢ for free support vectors (0 < α < C)
                # We compute two candidate values and pick appropriately.
                # -------------------------------------------------------
                delta_i = alpha[i] - alpha_i_old
                delta_j = alpha[j] - alpha_j_old

                b1 = (b - Ei
                      - y[i] * delta_i * K[i, i]
                      - y[j] * delta_j * K[i, j])

                b2 = (b - Ej
                      - y[i] * delta_i * K[i, j]
                      - y[j] * delta_j * K[j, j])

                if 0 < alpha[i] < self.C:
                    b = b1    # α_i is free → b1 is exact
                elif 0 < alpha[j] < self.C:
                    b = b2    # α_j is free → b2 is exact
                else:
                    b = (b1 + b2) / 2  # Both bound → average

                # -------------------------------------------------------
                # Step 8: Update f_cache incrementally
                # f(xₖ) = Σⱼ αⱼyⱼK(j,k) + b
                # When only α_i and α_j changed:
                # Δf(xₖ) = yᵢΔαᵢK(i,k) + yⱼΔαⱼK(j,k) + Δb
                # -------------------------------------------------------
                f_cache += (y[i] * delta_i * K[i, :]
                            + y[j] * delta_j * K[j, :])
                f_cache += b - (b - delta_i * y[i] * K[i, i]
                                   - delta_j * y[j] * K[j, i])
                # Simpler: just recompute fully after each update
                # (incremental is faster but more complex to get right)
                # Let's use the safe version:
                f_cache = (alpha * y) @ K + b

                num_changed += 1

            # Track objective value every 10 passes for diagnostics
            if iteration % 10 == 0:
                obj = self._dual_objective(alpha, y, K)
                self.objective_history_.append(obj)

            iteration += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0   # Reset — we made progress, keep going

        # ------------------------------------------------------------------
        # Post-training: extract support vectors
        # Support vectors are points with α > 0 (they sit on or inside margin)
        # ------------------------------------------------------------------
        sv_mask = alpha > 1e-5   # Numerical threshold

        self.alpha_ = alpha
        self.b_ = b
        self._X_train = X
        self._y_train = y

        self.support_vectors_ = X[sv_mask]
        self.support_vector_labels_ = y[sv_mask]
        self.support_vector_alphas_ = alpha[sv_mask]
        self.n_support_ = int(np.sum(sv_mask))

        print(f"  Training complete. {self.n_support_} support vectors "
              f"({100 * self.n_support_ / n:.1f}% of training set)")

        return self

    def decision_function(self, X):
        """
        f(x) = Σᵢ αᵢyᵢK(xᵢ, x) + b

        We only sum over support vectors (αᵢ > 0).
        This is why SVMs are efficient at prediction time —
        most training points drop out.

        Returns raw scores (not probabilities). Sign gives the class.
        """
        X = np.asarray(X, dtype=np.float64)

        # K_sv: shape (n_support_vectors, n_test_samples)
        K_sv = self._kernel_fn(self.support_vectors_, X)

        # Weighted sum: (αᵢ * yᵢ) @ K_sv → shape (n_test,)
        scores = (self.support_vector_alphas_ * self.support_vector_labels_) @ K_sv
        return scores + self.b_

    def predict(self, X):
        return np.sign(self.decision_function(X)).astype(int)

    def score(self, X, y):
        y = self._validate_labels(y)
        return np.mean(self.predict(X) == y)

    def get_params(self):
        return {
            'C': self.C, 'kernel': self.kernel, 'gamma': self.gamma,
            'degree': self.degree, 'coef0': self.coef0,
            'tol': self.tol, 'max_passes': self.max_passes,
            'random_state': self.random_state,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_j(self, i, Ei, f_cache, y, n, rng):
        """
        Heuristic for picking the second alpha (inner loop).

        Platt's heuristic: maximize |Ei - Ej| → largest optimization step.

        We look at all cached errors and pick the j that gives max |Ei - Ej|.
        Fall back to random if cache is all zeros (early in training).
        """
        errors = f_cache - y         # Eₖ for all k
        abs_diffs = np.abs(Ei - errors)
        abs_diffs[i] = -1            # Don't pick i itself

        j = int(np.argmax(abs_diffs))

        # If all errors are zero (very early in training), pick randomly
        if abs_diffs[j] < 1e-10:
            j = i
            while j == i:
                j = rng.randint(0, n)

        return j

    def _compute_LH(self, alpha_i, alpha_j, yi, yj):
        """
        Compute lower and upper bounds for the clipped α_j update.

        The constraint αᵢyᵢ + αⱼyⱼ = const (call it κ) combined with
        0 ≤ α ≤ C gives us a feasible interval [L, H] for α_j.

        Case 1: yᵢ ≠ yⱼ (opposite labels)
            κ = αⱼ - αᵢ (constant)
            L = max(0, αⱼ - αᵢ)
            H = min(C, C + αⱼ - αᵢ)

        Case 2: yᵢ = yⱼ (same labels)
            κ = αᵢ + αⱼ (constant)
            L = max(0, αᵢ + αⱼ - C)
            H = min(C, αᵢ + αⱼ)
        """
        if yi != yj:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_i + alpha_j - self.C)
            H = min(self.C, alpha_i + alpha_j)
        return L, H

    def _dual_objective(self, alpha, y, K):
        """
        Dual objective value: Σαᵢ - (1/2)ΣᵢΣⱼ αᵢαⱼyᵢyⱼK(i,j)
        We maximize this (equivalent to minimizing the primal).
        Useful for tracking convergence.
        """
        term1 = np.sum(alpha)
        # (alpha * y) @ K @ (alpha * y) = Σᵢ Σⱼ αᵢαⱼyᵢyⱼK(i,j)
        ay = alpha * y
        term2 = 0.5 * ay @ K @ ay
        return term1 - term2

    def _build_kernel(self, X):
        """
        Resolve gamma and return the compiled kernel function.
        """
        n_features = X.shape[1]

        if self.gamma == 'scale':
            gamma_val = 1.0 / (n_features * X.var()) if X.var() > 0 else 1.0
        elif self.gamma == 'auto':
            gamma_val = 1.0 / n_features
        else:
            gamma_val = float(self.gamma)

        self._gamma_val = gamma_val  # Store for inspection

        if callable(self.kernel):
            return self.kernel

        return get_kernel(
            self.kernel,
            gamma=gamma_val,
            degree=self.degree,
            coef0=self.coef0,
        )

    def _validate_labels(self, y):
        y = np.asarray(y)
        unique = np.unique(y)
        if set(unique) == {0, 1}:
            return 2 * y - 1
        elif set(unique).issubset({-1, 0, 1}):
            return y.astype(np.float64)
        else:
            raise ValueError(f"Labels must be binary. Got: {unique}")