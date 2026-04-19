"""
fraud_detection_project/src/serving/api.py

Lightweight Flask API for the fraud detection model.

Endpoints:
  GET  /health              — liveness check
  GET  /model/info          — model metadata
  POST /predict             — single transaction prediction
  POST /predict/batch       — batch prediction (up to 1000 transactions)
  GET  /monitor/status      — current drift monitoring stats
  POST /monitor/reset       — reset monitoring window

Design decisions:
  - Synchronous (no async) — SVM prediction is fast (<5ms per txn)
  - In-memory monitor — no external DB dependency for dev
  - Structured JSON errors — consistent error format for consumers
  - Request validation — fail fast with clear messages
"""

from flask import Flask, request, jsonify, send_from_directory, redirect
import numpy as np
import time
import os
from pathlib import Path

# ── Import our serving layer ────────────────────────────────────────────────
import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))

from fraud_detection_project.src.serving.predictor import (
    FraudPredictor, ModelMonitor
)

app = Flask(__name__)

# Built dashboard files: `npm run build` in fraud_detection_project/dashboard,
# then copy dist/* into this directory (see dashboard README).
DASHBOARD_STATIC = Path(__file__).resolve().parent / "static" / "dashboard"


def _maybe_cors(response):
    """Enable when UI runs on another origin without a dev proxy (set FRAUD_API_CORS=1)."""
    if os.environ.get("FRAUD_API_CORS", "").lower() in ("1", "true", "yes"):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.after_request
def _after_request(response):
    return _maybe_cors(response)


@app.route("/predict", methods=["OPTIONS"])
@app.route("/predict/batch", methods=["OPTIONS"])
@app.route("/monitor/reset", methods=["OPTIONS"])
def _cors_preflight():
    return "", 204


# ── Globals (loaded at startup) ─────────────────────────────────────────────
MODEL_PATH = Path("fraud_detection_project/models/svm_fraud_model_v2.pkl")
predictor: FraudPredictor = None
monitor:   ModelMonitor   = None

# ── Startup ─────────────────────────────────────────────────────────────────

def load_model():
    """Called once at startup. Loads model and initializes monitor."""
    global predictor, monitor

    print("Loading fraud detection model...")
    predictor = FraudPredictor.load(MODEL_PATH)

    # Initialize monitor with training score distribution
    # In production: load reference scores from the artifact or a database
    # Here: generate a plausible reference distribution for demo
    # Replace with: np.load("models/reference_scores.npy")
    rng = np.random.RandomState(42)
    # Approximate reference: mostly negative scores (legit), some positive (fraud)
    ref_scores = np.concatenate([
        rng.normal(-2, 3, 10000),    # legit: negative scores
        rng.normal(8,  3,    17),    # fraud: positive scores
    ])
    monitor = ModelMonitor(
        reference_scores=ref_scores,
        window_size=1000,
        psi_threshold=0.2,
    )
    print("API ready.")


# ── Helper: build error response ─────────────────────────────────────────────

def error_response(message: str, status: int = 400) -> tuple:
    return jsonify({'error': message, 'status': status}), status


# ── Helper: validate transaction fields ──────────────────────────────────────

def validate_transaction(txn: dict) -> str | None:
    """
    Returns an error message string if invalid, None if valid.

    Required: V1-V28, Amount, Time.
    V features must be finite floats (they're PCA components — no NaN allowed).
    Amount must be >= 0.
    Time must be >= 0.
    """
    v_features = [f'V{i}' for i in range(1, 29)]

    for v in v_features:
        if v not in txn:
            return f"Missing required feature: {v}"
        try:
            val = float(txn[v])
            if not np.isfinite(val):
                return f"Feature {v} must be finite (got {val})"
        except (TypeError, ValueError):
            return f"Feature {v} must be numeric (got {txn[v]!r})"

    if 'Amount' not in txn:
        return "Missing required field: Amount"
    try:
        amount = float(txn['Amount'])
        if amount < 0:
            return f"Amount must be >= 0 (got {amount})"
    except (TypeError, ValueError):
        return f"Amount must be numeric"

    if 'Time' not in txn:
        return "Missing required field: Time"
    try:
        time_val = float(txn['Time'])
        if time_val < 0:
            return f"Time must be >= 0 (got {time_val})"
    except (TypeError, ValueError):
        return "Time must be numeric"

    return None   # Valid


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    """
    Liveness probe. Returns 200 if the service is up.
    Used by load balancers, Kubernetes health checks, etc.
    """
    if predictor is None:
        return jsonify({'status': 'unhealthy',
                        'reason': 'model not loaded'}), 503
    return jsonify({
        'status':    'healthy',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })


@app.route('/model/info', methods=['GET'])
def model_info():
    """Returns model metadata — useful for debugging and audit trails."""
    if predictor is None:
        return error_response("Model not loaded", 503)

    return jsonify({
        'model_name':     predictor.meta.get('model_name', 'unknown'),
        'n_features':     len(predictor.feature_names),
        'feature_names':  predictor.feature_names,
        'threshold':      predictor.threshold,
        'threshold_label':predictor.meta.get('threshold_label', ''),
        'test_auc_pr':    predictor.meta.get('test_auc_pr', None),
        'test_recall':    predictor.meta.get('test_recall', None),
        'test_precision': predictor.meta.get('test_precision', None),
        'fraud_rate_train':predictor.meta.get('fraud_rate_train', None),
    })


@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Predict fraud for a single transaction.

    Request body (JSON):
    {
        "V1": -1.359807,
        "V2": -0.072781,
        ... (V3 through V28) ...
        "Amount": 149.62,
        "Time": 0.0
    }

    Response:
    {
        "is_fraud": false,
        "risk_score": -3.42,
        "risk_score_norm": 0.29,
        "threshold": 8.32,
        "confidence": "high",
        "explanation": "Approved | risk score -3.42 vs threshold 8.32 | ...",
        "amount": 149.62,
        "latency_ms": 1.23,
        "timestamp": "2024-01-15T10:30:00Z"
    }
    """
    if predictor is None:
        return error_response("Model not loaded", 503)

    data = request.get_json(silent=True)
    if data is None:
        return error_response("Request body must be valid JSON")

    # Validate
    err = validate_transaction(data)
    if err:
        return error_response(err)

    # Predict
    try:
        result = predictor.predict(data)
    except Exception as e:
        return error_response(f"Prediction failed: {str(e)}", 500)

    # Record for monitoring
    monitor.record(
        score    = result['risk_score'],
        is_fraud = result['is_fraud'],
        amount   = result['amount'],
    )

    return jsonify(result)


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict fraud for a batch of transactions.

    Request body (JSON):
    {
        "transactions": [
            {"V1": ..., "V2": ..., ..., "Amount": ..., "Time": ...},
            {"V1": ..., "V2": ..., ..., "Amount": ..., "Time": ...},
            ...
        ]
    }

    Response:
    {
        "results": [
            {"is_fraud": false, "risk_score": -3.42, ...},
            ...
        ],
        "n_transactions": 50,
        "n_flagged": 2,
        "total_latency_ms": 45.2
    }
    """
    if predictor is None:
        return error_response("Model not loaded", 503)

    data = request.get_json(silent=True)
    if data is None:
        return error_response("Request body must be valid JSON")

    if 'transactions' not in data:
        return error_response("Request must have a 'transactions' key")

    transactions = data['transactions']

    if not isinstance(transactions, list):
        return error_response("'transactions' must be a list")

    if len(transactions) == 0:
        return error_response("'transactions' list is empty")

    if len(transactions) > 1000:
        return error_response(
            "Batch size limited to 1000 transactions. "
            "Split into smaller batches."
        )

    # Validate each transaction
    for i, txn in enumerate(transactions):
        err = validate_transaction(txn)
        if err:
            return error_response(f"Transaction {i}: {err}")

    # Predict batch
    t0 = time.perf_counter()
    try:
        results = predictor.predict_batch(transactions)
    except Exception as e:
        return error_response(f"Batch prediction failed: {str(e)}", 500)
    total_ms = (time.perf_counter() - t0) * 1000

    # Record for monitoring
    scores = [r['risk_score'] for r in results]
    preds  = [r['is_fraud']   for r in results]
    amounts= [r['amount']     for r in results]
    monitor.record_batch(scores, preds, amounts)

    n_flagged = sum(int(r['is_fraud']) for r in results)

    return jsonify({
        'results':          results,
        'n_transactions':   len(results),
        'n_flagged':        n_flagged,
        'flag_rate':        round(n_flagged / len(results), 4),
        'total_latency_ms': round(total_ms, 1),
        'avg_latency_ms':   round(total_ms / len(results), 3),
    })


@app.route('/monitor/status', methods=['GET'])
def monitor_status():
    """
    Return current model monitoring statistics.
    Call this from your dashboard or alerting system.
    """
    if monitor is None:
        return error_response("Monitor not initialized", 503)

    stats = monitor.get_stats()
    if stats.get('status') == 'no_data':
        return jsonify({'status': 'no_data',
                        'message': 'No predictions recorded yet.'})

    # Convert numpy types for JSON serialization
    return jsonify({k: (float(v) if isinstance(v, (np.floating, np.integer))
                        else v)
                    for k, v in stats.items()})


@app.route('/monitor/reset', methods=['POST'])
def monitor_reset():
    """Reset the monitoring window. Useful after a planned retraining."""
    if monitor is None:
        return error_response("Monitor not initialized", 503)

    monitor._score_window.clear()
    monitor._pred_window.clear()
    monitor._amount_window.clear()

    return jsonify({
        'status': 'ok',
        'message': 'Monitoring window cleared.',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })


# ══════════════════════════════════════════════════════════════════════════════
# OPERATOR DASHBOARD (static SPA)
# ══════════════════════════════════════════════════════════════════════════════


@app.route("/dashboard")
def dashboard_redirect():
    if not DASHBOARD_STATIC.is_dir() or not (DASHBOARD_STATIC / "index.html").is_file():
        return jsonify({
            "error": "Dashboard not installed",
            "hint": "Build the UI in fraud_detection_project/dashboard and copy dist/* "
                     "to fraud_detection_project/src/serving/static/dashboard/",
        }), 404
    return redirect("/dashboard/", code=302)


@app.route("/dashboard/")
def dashboard_index():
    if not (DASHBOARD_STATIC / "index.html").is_file():
        return jsonify({
            "error": "Dashboard not installed",
            "hint": "Build the UI in fraud_detection_project/dashboard and copy dist/* "
                     "to fraud_detection_project/src/serving/static/dashboard/",
        }), 404
    return send_from_directory(DASHBOARD_STATIC, "index.html")


@app.route("/dashboard/assets/<path:filename>")
def dashboard_assets(filename):
    assets_dir = DASHBOARD_STATIC / "assets"
    path = (assets_dir / filename).resolve()
    if not str(path).startswith(str(assets_dir.resolve())):
        return error_response("Invalid path", 400)
    if not path.is_file():
        return error_response("Not found", 404)
    return send_from_directory(assets_dir, filename)


@app.route("/dashboard/samples/<path:filename>")
def dashboard_samples(filename):
    """Optional: ship sample JSON inside static dashboard for same-origin fetch."""
    samples_dir = DASHBOARD_STATIC / "samples"
    path = (samples_dir / filename).resolve()
    if not str(path).startswith(str(samples_dir.resolve())):
        return error_response("Invalid path", 400)
    if not path.is_file():
        return error_response("Not found", 404)
    return send_from_directory(samples_dir, filename)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5001))
    print(f"\nStarting fraud detection API on port {port}")
    print(f"Endpoints:")
    print(f"  GET  http://localhost:{port}/health")
    print(f"  GET  http://localhost:{port}/model/info")
    print(f"  POST http://localhost:{port}/predict")
    print(f"  POST http://localhost:{port}/predict/batch")
    print(f"  GET  http://localhost:{port}/monitor/status")
    print(f"  POST http://localhost:{port}/monitor/reset")
    if (DASHBOARD_STATIC / "index.html").is_file():
        print(f"  UI   http://localhost:{port}/dashboard/\n")
    else:
        print(f"  (Run dashboard build + copy dist to static/dashboard for /dashboard/)\n")
    app.run(port=port, debug=False)