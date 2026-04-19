"""
fraud_detection_project/src/serving/predictor.py

The serving layer has three jobs:
  1. Load the trained model artifact cleanly
  2. Accept a raw transaction, apply the same preprocessing as training
  3. Return a prediction + score + business-readable explanation

Two classes:
  FraudPredictor  — the core prediction engine
  ModelMonitor    — tracks score distributions to detect drift
"""

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from collections import deque
from datetime import datetime

import numpy as np


def _register_rbf_pickling_compat():
    """
    Model pickles may reference RBFSampledLinearSVC from either
    ``rbf_sampled_linear_svc`` (notebook imports the module) or legacy
    ``__main__`` (class defined inside 03_modeling_v2.py).
    """
    import __main__

    name = "rbf_sampled_linear_svc"
    if name not in sys.modules:
        mod_path = Path(__file__).resolve().parent / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, str(mod_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {mod_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    else:
        mod = sys.modules[name]
    setattr(__main__, "RBFSampledLinearSVC", mod.RBFSampledLinearSVC)


# ══════════════════════════════════════════════════════════════════════════════
# FRAUD PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

class FraudPredictor:
    """
    Wraps the trained SVM pipeline for production prediction.

    Responsibilities:
      - Load and validate the model artifact
      - Replicate training-time feature engineering
      - Apply the saved scaler (train statistics, not test)
      - Apply the decision threshold from the chosen operating point
      - Return structured prediction with confidence and explanation

    Usage (path is relative to cwd unless absolute):
        predictor = FraudPredictor.load(
            "fraud_detection_project/models/svm_fraud_model_v2.pkl")
        result = predictor.predict(transaction_dict)
        print(result['is_fraud'], result['risk_score'], result['explanation'])
    """

    # V1-V28 features expected from upstream (already PCA-transformed)
    V_FEATURES = [f'V{i}' for i in range(1, 29)]

    def __init__(self, model, scaler_mean, scaler_std,
                 feature_names, threshold, meta):
        self.model         = model
        self.scaler_mean   = np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_std    = np.asarray(scaler_std,  dtype=np.float64)
        self.feature_names = feature_names
        self.threshold     = threshold
        self.meta          = meta

        # Validate scaler dimensions match feature count
        assert len(self.scaler_mean) == len(self.feature_names), (
            f"Scaler mean dim {len(self.scaler_mean)} != "
            f"feature count {len(self.feature_names)}"
        )

        # Precompute feature index lookup for fast access
        self._feat_idx = {f: i for i, f in enumerate(feature_names)}

        print(f"FraudPredictor loaded:")
        print(f"  Features:  {len(feature_names)}")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Model:     {meta.get('model_name', 'unknown')}")
        print(f"  Test AUC-PR: {meta.get('test_auc_pr', 'N/A')}")

    @classmethod
    def load(cls, path):
        """Load from pickle artifact."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        _register_rbf_pickling_compat()
        with open(path, 'rb') as f:
            artifact = pickle.load(f)

        required = ['model', 'scaler_mean', 'scaler_std',
                    'feature_names', 'threshold', 'meta']
        for key in required:
            if key not in artifact:
                raise KeyError(f"Model artifact missing key: '{key}'")

        return cls(
            model        = artifact['model'],
            scaler_mean  = artifact['scaler_mean'],
            scaler_std   = artifact['scaler_std'],
            feature_names= artifact['feature_names'],
            threshold    = artifact['threshold'],
            meta         = artifact.get('meta', {}),
        )

    def predict(self, transaction: dict) -> dict:
        """
        Predict whether a single transaction is fraudulent.

        Parameters
        ----------
        transaction : dict with keys:
            Required:
                V1..V28  — PCA-transformed features from upstream
                Amount   — transaction amount in euros (raw, not log)
                Time     — seconds since start of monitoring window
            Optional (will use defaults if missing):
                any other keys are ignored

        Returns
        -------
        dict with:
            is_fraud       : bool
            risk_score     : float (raw decision function value)
            risk_score_norm: float in [0,1] via sigmoid squash
            threshold      : float (the operating threshold)
            confidence     : str ('high'/'medium'/'low')
            explanation    : str (human-readable reasoning)
            latency_ms     : float (prediction latency)
            timestamp      : str (ISO format)
        """
        t0 = time.perf_counter()

        # ── 1. Feature engineering (must match 02_data_cleaning.py exactly) ──
        features = self._engineer_features(transaction)

        # ── 2. Assemble feature vector in training order ──────────────────────
        X = self._assemble_vector(features)

        # ── 3. Scale using training statistics ───────────────────────────────
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        X_scaled = X_scaled.reshape(1, -1)

        # ── 4. Decision score ─────────────────────────────────────────────────
        score = self._get_score(X_scaled)

        # ── 5. Apply threshold ────────────────────────────────────────────────
        is_fraud = bool(score >= self.threshold)

        # ── 6. Normalize score to [0,1] via sigmoid for interpretability ──────
        # The raw score range varies by model. Sigmoid gives a consistent
        # "risk percentage" that non-technical users can understand.
        risk_norm = float(1.0 / (1.0 + np.exp(-score * 0.1)))

        # ── 7. Confidence band ────────────────────────────────────────────────
        # Distance from threshold tells us how confident the model is.
        # Close to threshold = uncertain. Far from threshold = confident.
        dist = abs(score - self.threshold)
        if dist > 5.0:
            confidence = 'high'
        elif dist > 2.0:
            confidence = 'medium'
        else:
            confidence = 'low'

        # ── 8. Explanation ────────────────────────────────────────────────────
        explanation = self._explain(transaction, score, is_fraud, features)

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            'is_fraud':        is_fraud,
            'risk_score':      float(score),
            'risk_score_norm': risk_norm,
            'threshold':       float(self.threshold),
            'confidence':      confidence,
            'explanation':     explanation,
            'amount':          float(transaction.get('Amount', 0)),
            'latency_ms':      round(latency_ms, 3),
            'timestamp':       datetime.utcnow().isoformat() + 'Z',
        }

    def predict_batch(self, transactions: list) -> list:
        """
        Predict a batch of transactions.
        More efficient than calling predict() in a loop for large batches
        because we vectorize the scaling and scoring steps.

        Each result dict matches ``predict()`` (including ``explanation``,
        ``latency_ms`` as amortized batch time per row, and ``timestamp``).
        """
        t0 = time.perf_counter()
        n  = len(transactions)

        # Build feature matrix
        X = np.zeros((n, len(self.feature_names)), dtype=np.float64)
        for i, txn in enumerate(transactions):
            features = self._engineer_features(txn)
            X[i]     = self._assemble_vector(features)

        # Scale
        X_scaled = (X - self.scaler_mean) / self.scaler_std

        # Score
        scores = self._get_score_batch(X_scaled)

        total_ms = (time.perf_counter() - t0) * 1000
        per_ms = total_ms / n if n else 0.0
        ts = datetime.utcnow().isoformat() + 'Z'

        # Assemble results (same keys as predict() for API / test parity)
        results = []
        for i, (txn, score) in enumerate(zip(transactions, scores)):
            is_fraud  = bool(score >= self.threshold)
            risk_norm = float(1.0 / (1.0 + np.exp(-score * 0.1)))
            dist      = abs(score - self.threshold)
            if dist > 5.0:
                confidence = 'high'
            elif dist > 2.0:
                confidence = 'medium'
            else:
                confidence = 'low'

            features = self._engineer_features(txn)
            explanation = self._explain(txn, float(score), is_fraud, features)

            results.append({
                'is_fraud':        is_fraud,
                'risk_score':      float(score),
                'risk_score_norm': risk_norm,
                'threshold':       float(self.threshold),
                'confidence':      confidence,
                'explanation':     explanation,
                'amount':          float(txn.get('Amount', 0)),
                'latency_ms':      round(per_ms, 3),
                'timestamp':       ts,
                'index':           i,
            })

        print(f"Batch of {n}: {total_ms:.1f}ms total, "
              f"{per_ms:.2f}ms per transaction")
        return results

    # ── Private helpers ──────────────────────────────────────────────────────

    def _engineer_features(self, txn: dict) -> dict:
        """
        Replicate feature engineering from 02_data_cleaning.py.
        This function must stay in sync with the training pipeline.
        If you change preprocessing at training time, update this too.
        """
        amount = float(txn.get('Amount', 0.0))
        time_s = float(txn.get('Time', 0.0))

        # log1p(Amount)
        amount_log = np.log1p(amount)

        # Cyclic hour encoding
        hour = int(time_s / 3600) % 24
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Amount to global median ratio
        # Global median from training data: €22.00 (from EDA)
        # In production this should be stored in the artifact, not hardcoded.
        TRAIN_MEDIAN_AMOUNT = 22.0
        amount_to_median = amount / (TRAIN_MEDIAN_AMOUNT + 1e-9)

        # Micro-transaction flag
        is_micro = float(amount < 1.0)

        # Normalized time
        # Max time from training: 172792 seconds
        TRAIN_MAX_TIME = 172792.0
        time_norm = time_s / TRAIN_MAX_TIME

        features = {}

        # V features (pass through)
        for v in self.V_FEATURES:
            features[v] = float(txn.get(v, 0.0))

        # Engineered features
        features['Amount_log']        = amount_log
        features['Hour_sin']          = hour_sin
        features['Hour_cos']          = hour_cos
        features['Amount_to_median']  = amount_to_median
        features['Is_micro']          = is_micro
        features['Time_norm']         = time_norm

        return features

    def _assemble_vector(self, features: dict) -> np.ndarray:
        """Build numpy array in the exact order of training feature_names."""
        vec = np.zeros(len(self.feature_names), dtype=np.float64)
        for feat, idx in self._feat_idx.items():
            vec[idx] = features.get(feat, 0.0)
        return vec

    def _get_score(self, X_scaled: np.ndarray) -> float:
        """
        Get decision score from whatever model type is stored.
        Handles SVC, LinearSVC, and SGDClassifier (RBFSampler pipeline).
        """
        if hasattr(self.model, 'decision_function'):
            return float(self.model.decision_function(X_scaled)[0])
        elif hasattr(self.model, 'predict_proba'):
            # Fallback: use log-odds of fraud probability
            prob = self.model.predict_proba(X_scaled)[0, 1]
            return float(np.log(prob / (1 - prob + 1e-10)))
        else:
            raise AttributeError(
                "Model has neither decision_function nor predict_proba"
            )

    def _get_score_batch(self, X_scaled: np.ndarray) -> np.ndarray:
        if hasattr(self.model, 'decision_function'):
            return self.model.decision_function(X_scaled).ravel()
        else:
            probs = self.model.predict_proba(X_scaled)[:, 1]
            return np.log(probs / (1 - probs + 1e-10))

    def _explain(self, txn, score, is_fraud, features) -> str:
        """
        Generate a human-readable explanation for the prediction.

        This is a rule-based explainer using feature values.
        In a production system you'd use SHAP values for feature attribution.
        That's covered in the monitoring section.
        """
        reasons = []
        amount  = float(txn.get('Amount', 0))

        # Amount-based signals
        if features.get('Is_micro', 0) == 1.0:
            reasons.append(f"micro-transaction (€{amount:.2f}) — "
                           f"common fraud testing pattern")
        elif amount > 1000:
            reasons.append(f"large amount (€{amount:.2f})")
        elif features.get('Amount_to_median', 1) > 10:
            reasons.append(f"unusually large relative to median "
                           f"(€{amount:.2f} vs €22 median)")

        # Time-based signals
        hour = int(float(txn.get('Time', 0)) / 3600) % 24
        if hour in [2, 3, 26 % 24, 28 % 24]:  # Peak fraud hours from EDA
            reasons.append(f"off-hours transaction (hour {hour:02d}:xx)")

        # V-feature signals — we only flag the top predictors
        v17 = float(txn.get('V17', 0))
        v14 = float(txn.get('V14', 0))
        v12 = float(txn.get('V12', 0))

        if v17 < -2.0:
            reasons.append(f"V17 anomaly ({v17:.2f}) — strong fraud signal")
        if v14 < -5.0:
            reasons.append(f"V14 anomaly ({v14:.2f}) — strong fraud signal")
        if abs(v12) > 3.0:
            reasons.append(f"V12 anomaly ({v12:.2f})")

        if not reasons:
            reasons.append("combination of behavioral features")

        verdict   = "FLAGGED AS FRAUD" if is_fraud else "Approved"
        score_str = f"risk score {score:.2f} vs threshold {self.threshold:.2f}"
        reason_str = "; ".join(reasons) if reasons else "model pattern match"

        return f"{verdict} | {score_str} | Signals: {reason_str}"


# ══════════════════════════════════════════════════════════════════════════════
# MODEL MONITOR
# ══════════════════════════════════════════════════════════════════════════════

class ModelMonitor:
    """
    Tracks prediction score distributions to detect model drift.

    Two types of drift we care about:
      1. Score drift: the distribution of decision scores is shifting
         → might mean the fraud pattern has changed
         → OR the transaction population has changed (new customer segment)

      2. Prediction rate drift: the fraction of transactions flagged as fraud
         is changing significantly
         → sudden spike: new fraud campaign, or data pipeline bug
         → sudden drop: fraudsters adapted to avoid the model

    We use the Population Stability Index (PSI) to quantify drift.
    PSI < 0.1  → no significant drift
    PSI < 0.2  → moderate drift, worth investigating
    PSI >= 0.2 → significant drift, consider retraining

    PSI formula:
      PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)

    This is the KL divergence between two binned distributions.
    """

    def __init__(self, reference_scores: np.ndarray,
                 window_size: int = 1000,
                 psi_threshold: float = 0.2,
                 n_bins: int = 10):
        """
        Parameters
        ----------
        reference_scores : scores from training/validation set
            This is the "expected" distribution we compare against.
        window_size : number of recent predictions to keep
        psi_threshold : PSI above this triggers a drift alert
        n_bins : number of bins for PSI computation
        """
        self.psi_threshold = psi_threshold
        self.n_bins        = n_bins
        self.window_size   = window_size

        # Compute reference bin edges and percentages from training scores
        self._ref_scores    = reference_scores
        self._bin_edges     = np.percentile(
            reference_scores,
            np.linspace(0, 100, n_bins + 1)
        )
        # Avoid duplicate edges (can happen with discrete features)
        self._bin_edges     = np.unique(self._bin_edges)

        # Reference bin percentages (with Laplace smoothing to avoid log(0))
        ref_counts, _       = np.histogram(reference_scores, bins=self._bin_edges)
        self._ref_pcts      = (ref_counts + 1) / (ref_counts.sum() + len(ref_counts))

        # Rolling window of recent scores and predictions
        self._score_window  = deque(maxlen=window_size)
        self._pred_window   = deque(maxlen=window_size)   # 0/1
        self._amount_window = deque(maxlen=window_size)

        # Metrics history for dashboard
        self.metrics_log    = []
        self._n_total       = 0
        self._n_fraud       = 0

        print(f"ModelMonitor initialized:")
        print(f"  Reference scores: {len(reference_scores):,}")
        print(f"  Score range: [{reference_scores.min():.2f}, "
              f"{reference_scores.max():.2f}]")
        print(f"  Window size: {window_size}")
        print(f"  PSI threshold: {psi_threshold}")

    def record(self, score: float, is_fraud: bool, amount: float = 0.0):
        """Record a single prediction for monitoring."""
        self._score_window.append(score)
        self._pred_window.append(int(is_fraud))
        self._amount_window.append(amount)
        self._n_total += 1
        self._n_fraud += int(is_fraud)

    def record_batch(self, scores, predictions, amounts=None):
        """Record a batch of predictions."""
        for i, (s, p) in enumerate(zip(scores, predictions)):
            a = amounts[i] if amounts is not None else 0.0
            self.record(float(s), bool(p), float(a))

    def compute_psi(self) -> float:
        """
        Compute PSI between reference distribution and current window.
        Returns 0.0 if window is too small for reliable estimate.
        """
        if len(self._score_window) < 100:
            return 0.0   # Not enough data yet

        window_scores = np.array(self._score_window)
        curr_counts, _ = np.histogram(window_scores, bins=self._bin_edges)

        # Laplace-smoothed current percentages
        curr_pcts = ((curr_counts + 1)
                     / (curr_counts.sum() + len(curr_counts)))

        # Only use bins where we have reference data
        n_bins_used = min(len(self._ref_pcts), len(curr_pcts))
        ref  = self._ref_pcts[:n_bins_used]
        curr = curr_pcts[:n_bins_used]

        # PSI = Σ (curr - ref) × ln(curr / ref)
        psi = float(np.sum((curr - ref) * np.log(curr / ref + 1e-10)))
        return psi

    def get_stats(self) -> dict:
        """
        Compute current monitoring statistics.
        Call this periodically (every N predictions or every hour).
        """
        window_scores = np.array(self._score_window) if self._score_window else np.array([])
        window_preds  = np.array(self._pred_window)  if self._pred_window  else np.array([])

        if len(window_scores) == 0:
            return {'status': 'no_data'}

        psi           = self.compute_psi()
        fraud_rate_window = float(window_preds.mean()) if len(window_preds) else 0.0
        fraud_rate_total  = self._n_fraud / max(self._n_total, 1)

        # Flag conditions
        alerts = []
        if psi >= self.psi_threshold:
            alerts.append({
                'type': 'score_drift',
                'severity': 'high' if psi > 0.25 else 'medium',
                'message': f"PSI={psi:.3f} exceeds threshold {self.psi_threshold}. "
                           f"Score distribution has shifted. Consider retraining.",
                'psi': psi
            })

        # Fraud rate spike: more than 3× expected rate in current window
        expected_fraud_rate = 0.00172  # from training data
        if fraud_rate_window > 3 * expected_fraud_rate:
            alerts.append({
                'type': 'fraud_spike',
                'severity': 'high',
                'message': f"Fraud rate in window: {100*fraud_rate_window:.2f}% "
                           f"(expected ~{100*expected_fraud_rate:.3f}%). "
                           f"Possible new fraud campaign or data bug.",
                'rate': fraud_rate_window
            })

        stats = {
            'timestamp':          datetime.utcnow().isoformat() + 'Z',
            'n_total':            self._n_total,
            'n_window':           len(window_scores),
            'psi':                round(psi, 4),
            'psi_status':         ('ok' if psi < 0.1
                                   else 'warn' if psi < 0.2
                                   else 'alert'),
            'fraud_rate_window':  round(fraud_rate_window, 5),
            'fraud_rate_total':   round(fraud_rate_total, 5),
            'score_mean_window':  round(float(window_scores.mean()), 3),
            'score_std_window':   round(float(window_scores.std()), 3),
            'score_p95_window':   round(float(np.percentile(window_scores, 95)), 3),
            'alerts':             alerts,
        }

        self.metrics_log.append(stats)
        return stats

    def status_report(self):
        """Print a formatted monitoring status report."""
        stats = self.get_stats()
        if stats.get('status') == 'no_data':
            print("Monitor: no data recorded yet.")
            return stats

        psi_icon = {'ok': '✓', 'warn': '⚠', 'alert': '✗'}
        icon = psi_icon.get(stats['psi_status'], '?')

        print(f"\n{'─'*50}")
        print(f"  MODEL MONITOR REPORT  "
              f"{stats['timestamp'][:19]}")
        print(f"{'─'*50}")
        print(f"  Total predictions:   {stats['n_total']:,}")
        print(f"  Window size:         {stats['n_window']:,}")
        print(f"  PSI:   {stats['psi']:.4f}  {icon}  "
              f"({stats['psi_status'].upper()})")
        print(f"  Score  mean={stats['score_mean_window']:.2f}  "
              f"std={stats['score_std_window']:.2f}  "
              f"p95={stats['score_p95_window']:.2f}")
        print(f"  Fraud rate (window): "
              f"{100*stats['fraud_rate_window']:.3f}%")

        if stats['alerts']:
            print(f"\n  ⚠ ALERTS ({len(stats['alerts'])}):")
            for alert in stats['alerts']:
                print(f"    [{alert['severity'].upper()}] {alert['message']}")
        else:
            print(f"\n  ✓ No alerts.")

        return stats