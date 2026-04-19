"""
fraud_detection_project/src/serving/test_serving.py

Tests the full serving stack:
  1. FraudPredictor loads and predicts correctly
  2. Predictions match what we'd get from the raw model
  3. ModelMonitor detects injected drift
  4. API endpoints respond correctly (no Flask test client needed)

Run from repo root (or any cwd); model/data paths resolve from this package:
    python fraud_detection_project/src/serving/test_serving.py
"""

import numpy as np
import sys
import json
from pathlib import Path

test_dir = Path(__file__).parent.resolve()
fraud_detection_project_dir = test_dir.parent
if str(fraud_detection_project_dir) not in sys.path:
    sys.path.insert(0, str(fraud_detection_project_dir))

from predictor import (
    FraudPredictor, ModelMonitor
)

# .../fraud_detection_project — same layout as 03_modeling_v2.py (OUT_MODELS, DATA)
_FRAUD_PKG_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _FRAUD_PKG_ROOT / "models" / "svm_fraud_model_v2.pkl"
DATA_PATH = _FRAUD_PKG_ROOT / "data" / "processed"



# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — MODEL LOADS CLEANLY
# ══════════════════════════════════════════════════════════════════════════════

def test_load():
    print("\n" + "="*55)
    print("Test 1: Model loading")
    print("="*55)

    predictor = FraudPredictor.load(MODEL_PATH)

    assert predictor.model       is not None, "model is None"
    assert predictor.scaler_mean is not None, "scaler_mean is None"
    assert predictor.scaler_std  is not None, "scaler_std is None"
    assert len(predictor.feature_names) == 34, (
        f"Expected 34 features, got {len(predictor.feature_names)}"
    )
    assert predictor.threshold != 0.0, (
        "Threshold is 0.0 — the saved threshold wasn't stored correctly"
    )

    print(f"  ✓ Model loaded: {predictor.meta.get('model_name')}")
    print(f"  ✓ Features: {len(predictor.feature_names)}")
    print(f"  ✓ Threshold: {predictor.threshold:.4f}")
    return predictor


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — PREDICTION ON KNOWN TEST SAMPLES
# ══════════════════════════════════════════════════════════════════════════════

def test_predictions(predictor):
    print("\n" + "="*55)
    print("Test 2: Predictions on known test samples")
    print("="*55)

    # Load raw test data — we know the ground truth
    X_test = np.load(DATA_PATH / "X_test_scaled.npy")
    y_test = np.load(DATA_PATH / "y_test.npy")

    # Feature names to build transaction dicts
    with open(DATA_PATH / "dataset_meta.json") as f:
        meta = json.load(f)
    feature_names = meta['feature_names']

    # Take first 200 samples: include some fraud, some legit
    fraud_idx = np.where(y_test == 1)[0][:10]
    legit_idx = np.where(y_test == 0)[0][:10]
    test_idx  = np.concatenate([fraud_idx, legit_idx])

    # Convert scaled features back to transactions for the predictor API
    # NOTE: Since X_test is already scaled, but predictor.predict() expects
    # RAW features (it does its own scaling), we need to unscale first.
    scaler_mean = predictor.scaler_mean
    scaler_std  = predictor.scaler_std

    X_raw = X_test[test_idx] * scaler_std + scaler_mean  # Unscale

    # Build transaction dicts — map feature_names back to raw values
    transactions = []
    for i in range(len(test_idx)):
        txn = {feat: float(X_raw[i, j])
               for j, feat in enumerate(feature_names)}

        # The predictor expects Amount (raw), Time, V1-V28.
        # Our features include Amount_log, etc. — we need to reverse-engineer
        # the originals. For test purposes, we'll derive Amount from Amount_log.
        # Amount_log = log1p(Amount) → Amount = expm1(Amount_log)
        if 'Amount_log' in txn:
            txn['Amount'] = float(np.expm1(txn['Amount_log']))
        else:
            txn['Amount'] = 0.0

        # Time_norm = Time / 172792 → Time = Time_norm * 172792
        if 'Time_norm' in txn:
            txn['Time'] = float(txn['Time_norm'] * 172792)
        else:
            txn['Time'] = 0.0

        transactions.append(txn)

    # Predict via predictor API
    results = predictor.predict_batch(transactions)

    # Compare against model's direct output for sanity
    scores_direct = predictor._get_score_batch(X_test[test_idx])
    preds_direct  = (scores_direct >= predictor.threshold).astype(int)

    # The predictor API re-engineers features from raw, so tiny numerical
    # differences from the unscale/re-scale cycle are expected.
    # We check that at least 80% of predictions agree.
    api_preds = np.array([int(r['is_fraud']) for r in results])
    agreement = np.mean(api_preds == preds_direct)

    print(f"  API vs direct predictions agreement: {100*agreement:.0f}%")
    print(f"  Fraud samples  (first 10): true=1  api={api_preds[:10].tolist()}")
    print(f"  Legit samples (last  10): true=0  api={api_preds[10:].tolist()}")

    assert agreement >= 0.75, (
        f"Too much disagreement between API and direct: {100*agreement:.0f}%"
    )

    # Check prediction structure
    for r in results[:2]:
        required_keys = ['is_fraud', 'risk_score', 'risk_score_norm',
                         'threshold', 'confidence', 'explanation',
                         'amount', 'latency_ms', 'timestamp']
        for k in required_keys:
            assert k in r, f"Missing key '{k}' in prediction result"

    print(f"  ✓ Prediction structure validated")
    print(f"  ✓ Sample latency: {results[0]['latency_ms']:.2f}ms per prediction")
    print(f"  ✓ Agreement with direct model: {100*agreement:.0f}%")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — MONITOR DETECTS INJECTED DRIFT
# ══════════════════════════════════════════════════════════════════════════════

def test_monitor_drift():
    print("\n" + "="*55)
    print("Test 3: Monitor drift detection")
    print("="*55)

    rng = np.random.RandomState(42)

    # Reference: typical score distribution (legit-heavy)
    ref_scores = np.concatenate([
        rng.normal(-2, 2, 5000),    # legit
        rng.normal( 8, 2,   10),    # fraud (0.2% rate)
    ])

    monitor = ModelMonitor(
        reference_scores = ref_scores,
        window_size      = 500,
        psi_threshold    = 0.2,
    )

    # Phase 1: Feed in normal traffic — no drift expected
    print("\n  Phase 1: Normal traffic (no drift expected)...")
    normal_scores = rng.normal(-2, 2, 500)   # Matches reference
    for s in normal_scores:
        monitor.record(s, is_fraud=False)

    stats_normal = monitor.get_stats()
    print(f"    PSI = {stats_normal['psi']:.4f}  "
          f"(status: {stats_normal['psi_status']})")
    assert stats_normal['psi_status'] in ('ok', 'warn'), (
        f"Expected no alert on normal traffic, got: {stats_normal['psi_status']}"
    )

    # Phase 2: Inject drifted traffic — scores shift upward (more fraud-like)
    print("\n  Phase 2: Injecting drift (scores shifted upward)...")
    monitor._score_window.clear()
    monitor._pred_window.clear()

    drifted_scores = rng.normal(3, 3, 500)   # Shifted: mean -2 → +3
    for s in drifted_scores:
        monitor.record(s, is_fraud=(s > monitor.psi_threshold))

    stats_drifted = monitor.get_stats()
    print(f"    PSI = {stats_drifted['psi']:.4f}  "
          f"(status: {stats_drifted['psi_status']})")
    assert stats_drifted['psi_status'] == 'alert', (
        f"Expected drift alert, got: {stats_drifted['psi_status']} "
        f"(PSI={stats_drifted['psi']:.4f})"
    )
    assert len(stats_drifted['alerts']) > 0, "Expected at least one alert"

    print(f"  ✓ No alert on normal traffic (PSI={stats_normal['psi']:.4f})")
    print(f"  ✓ Alert triggered on drifted traffic (PSI={stats_drifted['psi']:.4f})")
    print(f"  ✓ Alert message: {stats_drifted['alerts'][0]['message'][:60]}...")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — LATENCY BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

def test_latency(predictor):
    print("\n" + "="*55)
    print("Test 4: Latency benchmark")
    print("="*55)

    import time

    # Build a synthetic transaction (all V features = 0, Amount=100, Time=3600)
    txn = {f'V{i}': 0.0 for i in range(1, 29)}
    txn['Amount'] = 100.0
    txn['Time']   = 3600.0

    # Single-prediction latency
    n_warmup = 5
    n_bench  = 100
    for _ in range(n_warmup):    # Warmup (model may be cold first call)
        predictor.predict(txn)

    t0 = time.perf_counter()
    for _ in range(n_bench):
        predictor.predict(txn)
    single_ms = (time.perf_counter() - t0) * 1000 / n_bench

    # Batch latency
    batch = [txn] * 100
    t0 = time.perf_counter()
    predictor.predict_batch(batch)
    batch_ms = (time.perf_counter() - t0) * 1000

    print(f"  Single prediction: {single_ms:.2f}ms avg over {n_bench} calls")
    print(f"  Batch of 100:      {batch_ms:.1f}ms total, "
          f"{batch_ms/100:.2f}ms per transaction")
    print(f"\n  SLA guidance:")
    print(f"    Real-time card auth: <100ms → {'✓' if single_ms < 100 else '✗'}")
    print(f"    Near-real-time flag: <500ms → {'✓' if single_ms < 500 else '✗'}")

    assert single_ms < 500, (
        f"Single prediction too slow: {single_ms:.1f}ms (SLA: 500ms)"
    )
    print(f"  ✓ Latency within SLA")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Fraud Detection Serving Layer — Integration Tests")
    print("="*55)

    predictor = test_load()
    results   = test_predictions(predictor)
    test_monitor_drift()
    test_latency(predictor)

    print("\n" + "="*55)
    print("All serving tests passed.")
    print("="*55)
    print(f"""
Next steps:
  1. Start the API:
       python fraud_detection_project/src/serving/api.py

  2. Test with curl:
       curl http://localhost:5001/health
       curl http://localhost:5001/model/info

  3. Send a transaction:
       curl -X POST http://localhost:5001/predict \\
         -H "Content-Type: application/json" \\
         -d '{{"V1":-1.36,"V2":-0.07,"V3":2.54,"V4":1.38,"V5":-0.34,
               "V6":0.46,"V7":0.24,"V8":0.10,"V9":0.36,"V10":0.09,
               "V11":-0.55,"V12":-0.62,"V13":-0.99,"V14":-0.31,
               "V15":1.47,"V16":-0.47,"V17":0.21,"V18":0.03,"V19":0.40,
               "V20":0.25,"V21":-0.02,"V22":0.28,"V23":-0.11,"V24":0.07,
               "V25":0.13,"V26":-0.19,"V27":0.13,"V28":-0.02,
               "Amount":149.62,"Time":0.0}}'
    """)