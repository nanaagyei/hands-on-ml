"""
RBFSampler + SGD hinge classifier for large-n fraud training.

Lives in an importable module so pickle artifacts reference
``rbf_sampled_linear_svc.RBFSampledLinearSVC``, not ``__main__``.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier


class RBFSampledLinearSVC:
    """
    RBFSampler + linear hinge model without a float64 (n × d) Pipeline buffer.
    Chunked transform to float32, then SGDClassifier(hinge).
    """

    def __init__(
        self,
        gamma,
        n_components,
        C,
        *,
        class_weight="balanced",
        row_chunk=40_000,
        max_iter=1500,
        tol=1e-3,
        random_state=42,
    ):
        self.gamma = float(gamma)
        self.n_components = int(n_components)
        self.C = float(C)
        self.class_weight = class_weight
        self.row_chunk = int(row_chunk)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.sampler_ = RBFSampler(
            gamma=self.gamma,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self.sampler_.fit(X)
        n, d = X.shape[0], self.n_components
        Xt = np.empty((n, d), dtype=np.float32)
        t0 = time.time()
        for i in range(0, n, self.row_chunk):
            sl = slice(i, min(i + self.row_chunk, n))
            Xt[sl] = self.sampler_.transform(X[sl]).astype(
                np.float32, copy=False
            )
            if (i // self.row_chunk) % 4 == 0 or sl.stop >= n:
                pct = 100.0 * sl.stop / n
                print(
                    f"  RBF features: {sl.stop:,}/{n:,} rows "
                    f"({pct:.0f}%)  {time.time() - t0:.1f}s",
                    flush=True,
                )
        alpha = 1.0 / (self.C * float(n))
        self.clf_ = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=alpha,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            average=False,
        )
        print("  Training SGD hinge on RBF features...", flush=True)
        t1 = time.time()
        self.clf_.fit(Xt, y)
        print(f"  SGD fit done in {time.time() - t1:.1f}s", flush=True)
        return self

    def _transform_batches(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        for i in range(0, n, self.row_chunk):
            sl = slice(i, min(i + self.row_chunk, n))
            yield self.sampler_.transform(X[sl]).astype(np.float32, copy=False)

    def decision_function(self, X):
        return np.concatenate(
            [self.clf_.decision_function(b) for b in self._transform_batches(X)]
        )

    def predict(self, X):
        return np.concatenate(
            [self.clf_.predict(b) for b in self._transform_batches(X)]
        )
