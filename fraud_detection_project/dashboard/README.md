# Fraud operations dashboard

Operator UI for the Flask serving API in `../src/serving/api.py`: model metadata, single and batch scoring, and drift monitoring.

## Prerequisites

- Node.js 20+ recommended
- Python dependencies: from repo root, `pip install -r fraud_detection_project/requirements-serving.txt`
- Trained model at `fraud_detection_project/models/svm_fraud_model_v2.pkl` (paths are relative to repo root when you start the API)

## Development (Vite + API on two ports)

1. Start the API from the **repository root** (loads the model and listens on port 5001 by default):

   ```bash
   cd /path/to/hands-on-ml
   python fraud_detection_project/src/serving/api.py
   ```

2. In another terminal, start the dashboard dev server:

   ```bash
   cd fraud_detection_project/dashboard
   npm install
   npm run dev
   ```

3. Open the URL Vite prints (usually `http://localhost:5173`). The dev server **proxies** `/api/*` to the Flask app and strips the `/api` prefix, so browser calls hit `http://127.0.0.1:5001/...` without CORS issues.

Optional: point the proxy elsewhere:

```bash
VITE_PROXY_TARGET=http://127.0.0.1:8080 npm run dev
```

If you run the UI on a different host than the API without the proxy, set `FRAUD_API_CORS=1` when starting Flask (see `api.py`) or set `VITE_API_BASE` to the full API origin.

## Production-style bundle (served by Flask)

Build static assets, then copy them next to the API so Flask can serve `/dashboard/`:

```bash
cd fraud_detection_project/dashboard
npm run build
npm run deploy:static
```

Then start the API from the repo root and open `http://localhost:5001/dashboard/` (or your `PORT`).

## Sample payloads

JSON fixtures for quick tests live in `public/samples/` (`legit.json`, `fraud.json`). They are copied into `dist/samples/` on build and are available in dev at `/samples/legit.json`.

## Environment variables

| Variable | Where | Purpose |
|----------|--------|---------|
| `VITE_API_BASE` | Vite (build-time) | Force API path prefix (default: `/api` in dev, `` in production build) |
| `VITE_PROXY_TARGET` | Vite dev | Flask origin for the dev proxy (default `http://127.0.0.1:5001`) |
| `PORT` | Flask | API port (default `5001`) |
| `FRAUD_API_CORS` | Flask | Set to `1` / `true` / `yes` to send permissive CORS headers |
