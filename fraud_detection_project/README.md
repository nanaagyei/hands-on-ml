# Fraud Detection with Support Vector Machines

A production-grade fraud detection system built from scratch, demonstrating the full ML engineering lifecycle: math → implementation → training → serving → monitoring → UI.

## What This Project Covers

| Phase | What we built | Key concepts |
|---|---|---|
| Math | Derivation of SVM primal/dual | Lagrange multipliers, KKT conditions, kernel trick |
| Implementation | `LinearSVM` + `KernelSVM` via SMO | Subgradient descent, Sequential Minimal Optimization |
| Data | Credit card fraud EDA + preprocessing | Class imbalance, cyclic encoding, distribution mismatch |
| Training | LinearSVC + RBF approximation | Hyperparameter search, AUC-PR, threshold tuning |
| Serving | `FraudPredictor` + Flask API | Feature pipeline, PSI drift monitoring |
| Dashboard | Vite + React operator UI | Real-time scoring, batch upload, drift visualization |

---

## Project Structure

```
fraud_detection_project/
├── data/
│   ├── raw/
│   │   └── creditcard.csv             ← Kaggle: mlg-ulb/creditcardfraud
│   └── processed/
│       ├── X_train_scaled.npy
│       ├── X_test_scaled.npy
│       ├── X_dev.npy
│       ├── y_train.npy  y_test.npy  y_dev.npy
│       ├── scaler_mean.npy  scaler_std.npy
│       └── dataset_meta.json
│
├── notebooks/
│   ├── 01_eda.py                      ← Data exploration
│   ├── 02_data_cleaning.py            ← Feature engineering + preprocessing
│   ├── 03_modeling_v2.py              ← Training + hyperparameter search
│   └── figures/
│
├── src/
│   └── serving/
│       ├── predictor.py               ← FraudPredictor + ModelMonitor
│       ├── api.py                     ← Flask REST API
│       └── test_serving.py            ← Integration tests
│
├── models/
│   └── svm_fraud_model_v2.pkl         ← Trained model artifact
│
├── dashboard/                         ← Vite operator UI
│   ├── src/
│   ├── public/samples/
│   │   ├── legit.json
│   │   └── fraud.json
│   ├── vite.config.js
│   └── README.md
│
└── requirements-serving.txt
```

---

## The Dataset

**Credit Card Fraud Detection** — [Kaggle: mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions over 48 hours
- 492 fraudulent (0.173% — 578:1 imbalance)
- V1-V28: PCA-transformed anonymized features
- Amount, Time: raw features requiring engineering

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud \
  --path fraud_detection_project/data/raw/ --unzip
```

**Git / GitHub:** The repo root `.gitignore` excludes `data/raw/*.csv`, processed `*.npy`, and `models/*.pkl` so the Kaggle extract (~150MB), regenerated arrays, and pickle artifacts stay local. Clone the repo, download the CSV with the command above, then run preprocessing and training to produce those files.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r fraud_detection_project/requirements-serving.txt
```

### 2. Preprocess data

```bash
python fraud_detection_project/notebooks/02_data_cleaning.py
```

Outputs saved to `fraud_detection_project/data/processed/`.

### 3. Train the model

```bash
python fraud_detection_project/notebooks/03_modeling_v2.py
```

Trains LinearSVC (full dataset) and RBF approximation (RBFSampler + SGD).
Model saved to `fraud_detection_project/models/svm_fraud_model_v2.pkl`.

Expected results at the 80% recall operating point:

| Metric | Value |
|---|---|
| AUC-PR | 0.806 |
| AUC-ROC | 0.987 |
| Recall | 80.6% |
| Precision | 81.4% |
| False positives per 57k txns | 18 |

### 4. Run integration tests

```bash
python fraud_detection_project/src/serving/test_serving.py
```

### 5. Start the API

```bash
# From repo root
python fraud_detection_project/src/serving/api.py
```

API runs on `http://localhost:5001`.

### 6. Start the dashboard

```bash
cd fraud_detection_project/dashboard
npm install && npm run dev
```

Open `http://localhost:5173`.

---

## API Reference

### `GET /health`
Liveness probe. Returns 200 if model is loaded.

### `GET /model/info`
Model metadata: feature names, threshold, test metrics.

### `POST /predict`

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46,  "V7": 0.24, "V8": 0.10,
    "V9":  0.36, "V10": 0.09, "V11":-0.55,"V12":-0.62,
    "V13":-0.99, "V14":-0.31, "V15": 1.47,"V16":-0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40,"V20": 0.25,
    "V21":-0.02, "V22": 0.28, "V23":-0.11,"V24": 0.07,
    "V25": 0.13, "V26":-0.19, "V27": 0.13,"V28":-0.02,
    "Amount": 149.62, "Time": 0.0
  }'
```

Response:

```json
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
```

### `POST /predict/batch`

```json
{ "transactions": [ {...}, {...}, ... ] }
```

Max 1000 transactions per request.

### `GET /monitor/status`

Returns PSI drift score, fraud rate, and any active alerts.

### `POST /monitor/reset`

Clears the monitoring window after a planned retraining.

---

## Key Engineering Decisions

### Why not accuracy?

578:1 imbalance means a model that flags nothing gets 99.83% accuracy and catches zero frauds. We use **AUC-PR** as the primary metric and **business net value** for stakeholder communication.

### Why threshold tuning matters

The SVM's default threshold (score ≥ 0) produced 936 false positives per test batch. The tuned threshold (score ≥ 8.32) produces 18 false positives while maintaining 80% recall. That's the difference between a viable product and one that overwhelms the analyst team.

### Why the dev set search failed in v1

Hyperparameter search on a 12:1 subsample found parameters optimized for 12:1 imbalance. The production distribution is 578:1. C and gamma values that work at 12:1 don't transfer. **Always search on data with the same distribution as deployment.**

### Why RBFSampler instead of exact SVC

Exact `SVC(kernel='rbf')` on 227k samples would require a 415 GB kernel matrix. `RBFSampler` (Rahimi & Recht, 2007) approximates the RBF kernel via random Fourier features — giving 91.8% recall with 8-second training time.

### Why LinearSVC is competitive here

V1-V28 are already PCA-transformed. PCA finds linear projections that maximize variance. Fraud patterns that are hard to separate in raw transaction space are often approximately linearly separable after PCA. The LinearSVC result (AUC-ROC 0.9902) confirms this.

---

## From-Scratch SVM Implementation

`svm/core/` contains our NumPy implementations, independent of sklearn:

```python
from svm.core.linear_svm import LinearSVM
from svm.core.kernel_svm import KernelSVM

# Linear SVM via subgradient descent
model = LinearSVM(C=1.0, learning_rate=0.001, n_epochs=1000)
model.fit(X_train, y_train)

# Kernel SVM via SMO
model = KernelSVM(C=1.0, kernel='rbf', gamma='scale', max_passes=5)
model.fit(X_train, y_train)
scores = model.decision_function(X_test)
```

Validated against sklearn's `SVC` — 0.0% accuracy difference on test data.

---

## Monitoring

The `ModelMonitor` class tracks score distribution drift using **Population Stability Index (PSI)**:

| PSI | Interpretation | Action |
|---|---|---|
| < 0.10 | No drift | None |
| 0.10–0.20 | Moderate drift | Investigate |
| ≥ 0.20 | Significant drift | Retrain |

Access via `GET /monitor/status` or the dashboard's Monitoring tab.

---

## What's Next

This project used SVMs — maximum margin classifiers with kernel trick. The next module covers **tree-based methods**: Decision Trees, Random Forests, and Gradient Boosting. These handle:

- Feature interactions natively (no kernel engineering)
- Mixed feature types (categorical + numerical) without encoding tricks
- Non-linear boundaries without choosing a kernel
- Built-in feature importance

We'll apply them to the same fraud dataset for a direct comparison.