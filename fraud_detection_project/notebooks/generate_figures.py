"""
fraud_detection_project/notebooks/generate_figures.py

Generate static figures for fraud_detection notebooks (markdown image tags).

Run from anywhere:
    python fraud_detection_project/notebooks/generate_figures.py

Requires: numpy, matplotlib, scikit-learn (optional for model plots).
Optional: pandas — needed to read ``data/raw/creditcard.csv`` for feature-engineering figures.
Without CSV / pandas, those figures use calibrated synthetic data.

Outputs (default): fraud_detection_project/notebooks/figures/
  - margin_diagram.png
  - threshold_tradeoff.png
  - class_imbalance_bars.png
  - score_separation.png
  - feature_engineering_before_after.png  (Amount vs log1p)
  - cyclical_hour_fraud_rate.png
  - hour_sin_cos_circle.png
  - is_micro_fraud_rates.png
  - pr_curve_with_operating_points.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ── Paths (script-location–relative; no cwd assumption) ─────────────────────
FRAUD_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = FRAUD_ROOT / "data" / "processed"
META_PATH = DATA_DIR / "dataset_meta.json"
FIGURES_DIR = FRAUD_ROOT / "notebooks" / "figures"
RAW_CSV = FRAUD_ROOT / "data" / "raw" / "creditcard.csv"


def _load_creditcard_csv():
    """Return DataFrame with Amount, Time, Class if raw CSV exists; else None."""
    if not RAW_CSV.exists():
        return None
    try:
        import pandas as pd
    except ImportError:
        print("Note: pandas not installed — cannot load creditcard.csv for feature-engineering figures.", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(RAW_CSV)
        need = {"Amount", "Time", "Class"}
        if not need.issubset(df.columns):
            return None
        return df
    except Exception as e:
        print(f"Note: could not read creditcard.csv ({e}). Using synthetic data where needed.", file=sys.stderr)
        return None


def _synthetic_amounts(n: int = 80_000, seed: int = 42) -> np.ndarray:
    """Right-skewed nonnegative amounts roughly mimicking card txn."""
    rng = np.random.default_rng(seed)
    # Mostly modest spends + heavy tail (lognormal), clipped like typical caps
    x = rng.lognormal(mean=np.log(25), sigma=2.1, size=n)
    x = np.clip(x, 0.0, 25_691.0)
    return x


def _configure_matplotlib():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.facecolor": "#fafbfc",
            "figure.facecolor": "white",
        }
    )


def figure_margin_diagram(out_path: Path) -> None:
    """
    <!-- IMAGE: margin_diagram.png -->
    Conceptual 2D linear SVM: max-margin boundary, dashed margins, circled SVs.

    Support vectors are constructed *on* the margin lines (n·x = b0 ± margin) so
    rings align with geometry (no hand-placed off-margin circles).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    rng = np.random.default_rng(7)

    # Unit normal n; decision boundary n·x = b0; margin lines n·x = b0 ± margin
    n = np.array([0.65, 0.76], dtype=float)
    n /= np.linalg.norm(n)
    b0 = 0.08
    margin = 0.42
    b_lo = b0 - margin
    b_hi = b0 + margin
    pad = 0.1  # keep bulk points off the dashed lines

    def proj_value(x):
        return float(np.dot(n, x))

    def on_margin(b_line: float, x0: float) -> np.ndarray:
        """Point with given x0 lying exactly on n·x = b_line (2D)."""
        x1 = (b_line - n[0] * x0) / (n[1] + 1e-12)
        return np.array([x0, x1], dtype=float)

    # Bulk legit / fraud: strictly inside their half-spaces, away from margins
    legit = []
    while len(legit) < 40:
        x = rng.normal([-0.85, -0.55], 0.3, 2)
        p = proj_value(x)
        if p < b_lo - pad:
            legit.append(x)
    legit = np.array(legit)

    fraud = []
    while len(fraud) < 22:
        x = rng.normal([0.95, 0.75], 0.26, 2)
        p = proj_value(x)
        if p > b_hi + pad:
            fraud.append(x)
    fraud = np.array(fraud)

    # Support vectors: exact positions on the margin lines (same n·x for each side)
    sv_legit_x = np.array([-0.82, -0.28, 0.38])
    sv_fraud_x = np.array([-0.18, 0.52])
    sv_legit = np.stack([on_margin(b_lo, x0) for x0 in sv_legit_x])
    sv_fraud = np.stack([on_margin(b_hi, x0) for x0 in sv_fraud_x])

    xs = np.linspace(-1.6, 1.6, 200)

    def line_x2(b, x0s):
        return (b - n[0] * x0s) / (n[1] + 1e-12)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    ax.scatter(legit[:, 0], legit[:, 1], c="#2d6a8f", s=38, alpha=0.85, edgecolors="white", linewidths=0.6, label="Legit")
    ax.scatter(fraud[:, 0], fraud[:, 1], c="#c44536", s=44, alpha=0.88, marker="^", edgecolors="white", linewidths=0.6, label="Fraud")

    # SV markers sit exactly on the margins (slightly larger so they read as “on the edge”)
    ax.scatter(
        sv_legit[:, 0],
        sv_legit[:, 1],
        c="#2d6a8f",
        s=95,
        alpha=0.95,
        edgecolors="#111",
        linewidths=1.4,
        zorder=6,
        label="Support vectors",
    )
    ax.scatter(
        sv_fraud[:, 0],
        sv_fraud[:, 1],
        c="#c44536",
        s=105,
        alpha=0.95,
        marker="^",
        edgecolors="#111",
        linewidths=1.4,
        zorder=6,
    )

    ax.plot(xs, line_x2(b0, xs), color="#1a1a2e", linewidth=2.2, label="Decision boundary", zorder=4)
    ax.plot(xs, line_x2(b_lo, xs), "k--", linewidth=1.35, alpha=0.75, label="Margin", zorder=3)
    ax.plot(xs, line_x2(b_hi, xs), "k--", linewidth=1.35, alpha=0.75, zorder=3)

    # Circles centered on SVs only (centers satisfy n·x = b_lo or b_hi by construction)
    circle_r = 0.13
    for xy in np.vstack([sv_legit, sv_fraud]):
        ax.add_patch(
            Circle(
                (float(xy[0]), float(xy[1])),
                circle_r,
                fill=False,
                edgecolor="#111",
                linewidth=2.0,
                zorder=8,
            )
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel("Feature 1 (illustrative)")
    ax.set_ylabel("Feature 2 (illustrative)")
    ax.set_title("Linear SVM margin (conceptual)")
    ax.legend(loc="upper left", framealpha=0.92)
    fig.text(
        0.5,
        0.02,
        "Circled markers lie exactly on the dashed margins — support vectors.",
        ha="center",
        fontsize=9.5,
        color="#444",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _load_xy_if_available():
    x_train = DATA_DIR / "X_train_scaled.npy"
    y_train = DATA_DIR / "y_train.npy"
    x_test = DATA_DIR / "X_test_scaled.npy"
    y_test = DATA_DIR / "y_test.npy"
    if not all(p.exists() for p in (x_train, y_train, x_test, y_test)):
        return None
    return (
        np.load(x_train),
        np.load(y_train).astype(int),
        np.load(x_test),
        np.load(y_test).astype(int),
    )


def _load_predictor_test_scores():
    """
    Full test-set decision scores from the saved FraudPredictor artifact,
    using the same scaled-feature path as integration tests (fast batch score).
    Returns (scores, y_test, saved_threshold) or None.
    """
    model_path = (FRAUD_ROOT / "models" / "svm_fraud_model_v2.pkl").resolve()
    x_path = DATA_DIR / "X_test_scaled.npy"
    y_path = DATA_DIR / "y_test.npy"
    if not model_path.is_file() or not x_path.is_file() or not y_path.is_file():
        return None
    repo_root = FRAUD_ROOT.parent.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from fraud_detection_project.src.serving.predictor import FraudPredictor

        pred = FraudPredictor.load(model_path)
        X = np.load(x_path).astype(np.float64)
        y = np.load(y_path).astype(int)
        scores = pred._get_score_batch(X).astype(np.float64)
        return scores, y, float(pred.threshold)
    except Exception as e:
        print(f"Note: predictor scores unavailable ({e}). Operating-point PR plot uses fallback.", file=sys.stderr)
        return None


def _metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, t: float):
    pred = scores >= t
    TP = int(np.sum(pred & (y_true == 1)))
    FP = int(np.sum(pred & (y_true == 0)))
    FN = int(np.sum((~pred) & (y_true == 1)))
    prec = TP / (TP + FP + 1e-12)
    rec = TP / (TP + FN + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return prec, rec, f1, TP, FP, FN


def _pick_operating_points(y_true: np.ndarray, scores: np.ndarray):
    """
    Threshold sweep over score thresholds (equivalent to moving along the PR curve).

    - Aggressive: **maximum** threshold t such that recall ≥ 0.80 (tightest rule that still
      hits the recall target → fewer false positives than looser thresholds at same recall).
    - Balanced: threshold that maximizes F1 among bins that predict at least one fraud.
    - Conservative: **maximum** threshold t such that precision ≥ 0.90 and TP ≥ 1.
    """
    hi, lo = float(np.max(scores)), float(np.min(scores))
    ts = np.linspace(hi, lo, 15_000)

    rows: list[tuple[float, float, float, float, int, int, int]] = []
    for t in ts:
        prec, rec, f1, TP, FP, FN = _metrics_at_threshold(y_true, scores, t)
        if TP + FP == 0:
            continue
        rows.append((t, prec, rec, f1, TP, FP, FN))

    if not rows:
        return None, None, None

    agg_hit = [r for r in rows if r[2] >= 0.80 - 1e-6]
    agg_best = max(agg_hit, key=lambda r: r[0]) if agg_hit else None

    bal_best = max(rows, key=lambda r: r[3])

    cons_hit = [r for r in rows if r[1] >= 0.90 - 1e-6 and r[4] >= 1]
    cons_best = max(cons_hit, key=lambda r: r[0]) if cons_hit else None

    if agg_best is None:
        agg_loose = [r for r in rows if r[2] >= 0.65]
        agg_best = max(agg_loose, key=lambda r: r[0]) if agg_loose else bal_best

    if cons_best is None:
        cons_loose = [r for r in rows if r[1] >= 0.85 and r[4] >= 1]
        cons_best = max(cons_loose, key=lambda r: r[0]) if cons_loose else max(rows, key=lambda r: r[1])

    return agg_best, bal_best, cons_best


def figure_pr_curve_operating_points(out_path: Path) -> None:
    """
    <!-- IMAGE: pr_curve_with_operating_points.png -->
    Full test PR curve with aggressive / balanced / conservative operating points.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    bundle = _load_predictor_test_scores()
    if bundle is None:
        _figure_pr_operating_points_fallback(out_path)
        return

    scores, y_true, saved_thr = bundle
    prevalence = float(np.mean(y_true))

    prec_c, rec_c, _thr_c = precision_recall_curve(y_true, scores)
    auc_pr = average_precision_score(y_true, scores)

    agg, bal, cons = _pick_operating_points(y_true, scores)

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.plot(rec_c, prec_c, color="#1d3557", linewidth=2.4, label=f"Test PR curve (AUC-PR = {auc_pr:.3f})")
    ax.fill_between(rec_c, prec_c, alpha=0.08, color="#1d3557")
    ax.axhline(prevalence, color="#888", linestyle=":", linewidth=1.35, label=f"Random baseline (prevalence ≈ {prevalence:.4f})")

    styles = [
        (agg, "#e63946", "Aggressive\n(high recall)", "s"),
        (bal, "#2a9d8f", "Balanced\n(max F1)", "o"),
        (cons, "#bc6c25", "Conservative\n(high precision)", "^"),
    ]
    for row, color, label, m in styles:
        if row is None:
            continue
        t, p, r, f1, TP, FP, FN = row
        ax.scatter([r], [p], s=140, c=color, marker=m, edgecolors="white", linewidths=1.5, zorder=6)
        ax.annotate(
            f"{label}\nt≈{t:.2f}\nP={100*p:.1f}% R={100*r:.1f}%",
            (r, p),
            textcoords="offset points",
            xytext=(12, -6 if m != "^" else -36),
            fontsize=8.5,
            color=color,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.7),
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall tradeoff with business operating points")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.94)

    fig.text(
        0.5,
        0.02,
        f"Aggressive ≈ tightest score threshold with recall ≥ 80%. Artifact operating threshold: {saved_thr:.2f}. "
        "Conservative = strict threshold prioritizing precision.",
        ha="center",
        fontsize=8.8,
        color="#444",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _figure_pr_operating_points_fallback(out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.svm import LinearSVC

    xy = _load_xy_if_available()
    if xy is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Need model pickle + test arrays for operating-point PR plot.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    X_train, y_train, X_test, y_test = xy
    X_tr, y_tr = _subsample_for_speed(X_train, y_train, max_rows=35_000)
    clf = LinearSVC(class_weight="balanced", max_iter=4000, random_state=42, dual=False, C=0.1)
    clf.fit(X_tr, y_tr)
    scores = clf.decision_function(X_test.astype(np.float64))

    prec_c, rec_c, _ = precision_recall_curve(y_test, scores)
    auc_pr = average_precision_score(y_test, scores)
    agg, bal, cons = _pick_operating_points(y_test.astype(int), scores.astype(np.float64))
    prevalence = float(np.mean(y_test))

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.plot(rec_c, prec_c, color="#6a4c93", linewidth=2.2, label=f"PR curve — LinearSVC proxy (AUC-PR = {auc_pr:.3f})")
    ax.axhline(prevalence, color="#888", linestyle=":", linewidth=1.2, label=f"Baseline ≈ {prevalence:.4f}")

    styles = [
        (agg, "#e63946", "Aggressive", "s"),
        (bal, "#2a9d8f", "Balanced", "o"),
        (cons, "#bc6c25", "Conservative", "^"),
    ]
    for row, color, label, m in styles:
        if row is None:
            continue
        t, p, r, _, _, _, _ = row
        ax.scatter([r], [p], s=130, c=color, marker=m, edgecolors="white", linewidths=1.4, zorder=6)
        ax.annotate(f"{label}\nt≈{t:.2f}", (r, p), xytext=(10, 8), textcoords="offset points", fontsize=9, color=color)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Operating points (fallback — train svm_fraud_model_v2.pkl for production curve)")
    ax.legend(loc="lower left", fontsize=9)
    fig.text(0.5, 0.02, "Placeholder scores from LinearSVC — artifact missing or unloadable.", ha="center", fontsize=9, color="#666")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _subsample_for_speed(X, y, max_rows: int = 45_000, seed: int = 42):
    """Keep all positives; subsample negatives so PR / scores stay meaningful."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y).astype(int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_pos = len(pos)
    if n_pos == 0:
        return X, y
    n_neg_take = min(len(neg), max(max_rows - n_pos, n_pos * 200))
    neg_take = rng.choice(neg, size=n_neg_take, replace=False)
    idx = np.concatenate([pos, neg_take])
    rng.shuffle(idx)
    return X[idx], y[idx]


def figure_threshold_tradeoff(out_path: Path, xy_bundle) -> None:
    """
    <!-- IMAGE: threshold_tradeoff.png -->
    Precision–recall curve; each point corresponds to a score threshold.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, auc
    from sklearn.svm import LinearSVC

    if xy_bundle is None:
        _figure_threshold_synthetic(out_path)
        return

    X_train, y_train, X_test, y_test = xy_bundle
    X_tr, y_tr = _subsample_for_speed(X_train, y_train, max_rows=35_000)
    X_te, y_te = _subsample_for_speed(X_test, y_test, max_rows=25_000)

    clf = LinearSVC(
        class_weight="balanced",
        max_iter=3000,
        random_state=42,
        dual=False,
        C=0.1,
    )
    clf.fit(X_tr, y_tr)
    scores = clf.decision_function(X_te)

    precision, recall, _ = precision_recall_curve(y_te, scores)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.plot(recall, precision, color="#0b6e4f", linewidth=2.2, label=f"PR curve (AUC-PR ≈ {pr_auc:.3f})")
    ax.fill_between(recall, precision, alpha=0.12, color="#0b6e4f")
    ax.axhline(y_te.mean(), color="#888", linestyle=":", linewidth=1.2, label=f"Baseline (prevalence ≈ {100*y_te.mean():.3f}%)")
    ax.set_xlabel("Recall (fraud caught)")
    ax.set_ylabel("Precision (flagged that are fraud)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision–recall tradeoff vs. threshold")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.text(
        0.5,
        0.01,
        "Each point is a different decision threshold — choose from business costs, not from the curve alone.",
        ha="center",
        fontsize=9.5,
        color="#333",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _figure_threshold_synthetic(out_path: Path) -> None:
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    n = 8000
    prev = 0.002
    y = (rng.random(n) < prev).astype(int)
    scores = rng.standard_normal(n) + 1.4 * y + 0.35 * rng.standard_normal(n)

    from sklearn.metrics import precision_recall_curve, auc

    precision, recall, _ = precision_recall_curve(y, scores)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.plot(recall, precision, color="#6a4c93", linewidth=2.2, label=f"Synthetic PR (AUC-PR ≈ {pr_auc:.3f})")
    ax.fill_between(recall, precision, alpha=0.12, color="#6a4c93")
    ax.axhline(prev, color="#888", linestyle=":", linewidth=1.2, label=f"Random baseline ({100*prev:.2f}%)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall tradeoff (synthetic — add data for real curve)")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.text(0.5, 0.01, "Install processed arrays and re-run for a data-driven curve.", ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_class_imbalance_bars(out_path: Path) -> None:
    """Train / dev / test fraud rates from dataset_meta.json (no npy required)."""
    import matplotlib.pyplot as plt

    if not META_PATH.exists():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "dataset_meta.json not found", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    with open(META_PATH) as f:
        meta = json.load(f)

    labels: list[str] = []
    rates: list[float] = []
    if "fraud_rate_train" in meta:
        labels.append("Train")
        rates.append(float(meta["fraud_rate_train"]) * 100)
    if "fraud_rate_dev" in meta:
        labels.append("Dev")
        rates.append(float(meta["fraud_rate_dev"]) * 100)
    y_test_path = DATA_DIR / "y_test.npy"
    if y_test_path.exists():
        y_test = np.load(y_test_path)
        labels.append("Test")
        rates.append(100 * float(np.mean(y_test)))

    if not rates:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        ax.text(0.5, 0.5, "No fraud rates found in dataset_meta.json", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    colors = ["#2d6a8f", "#bc6c25", "#606c38"][: len(labels)]
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=1.0)
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Class imbalance by split")
    ymax = max(rates) * 1.35 if rates else 1
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, min(r + ymax * 0.02, ymax * 0.98), f"{r:.3f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, ymax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_feature_engineering_before_after(out_path: Path, df) -> None:
    """
    <!-- IMAGE: feature_engineering_before_after.png -->
    Amount histogram raw vs log1p — explains distance / kernel scaling.
    """
    import matplotlib.pyplot as plt

    if df is None:
        amounts = _synthetic_amounts()
        subtitle = "Synthetic skewed amounts (download Kaggle CSV for real distributions)."
    else:
        amounts = df["Amount"].to_numpy(dtype=np.float64)
        subtitle = "Kaggle creditcard.csv — Amount column."
        if len(amounts) > 120_000:
            rng = np.random.default_rng(42)
            amounts = amounts[rng.choice(len(amounts), size=120_000, replace=False)]

    log_amt = np.log1p(amounts)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.8))

    hi = float(np.percentile(amounts, 99.5))
    amt_max = float(np.max(amounts))
    # Histogram uses [0, hi] only; do NOT axvline at global max — it forces xlim ~max(Amount)
    # and squashes all bars into an unreadible sliver at the origin.
    ax0.hist(amounts, bins=72, range=(0, hi), color="#2d6a8f", edgecolor="white", linewidth=0.4, alpha=0.88)
    ax0.set_xlim(0, hi)
    ax0.set_xlabel("Amount (€)")
    ax0.set_ylabel("Count")
    ax0.set_title("Raw Amount")
    ax0.text(
        0.97,
        0.92,
        "Long right tail:\ndistances dominated\nby large spends.",
        transform=ax0.transAxes,
        fontsize=9.5,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#ccc"),
    )
    ax0.text(
        0.03,
        0.03,
        f"Histogram: 0–{hi:,.0f} € (99.5th pct)\nDataset max €{amt_max:,.0f}",
        transform=ax0.transAxes,
        fontsize=8.8,
        va="bottom",
        ha="left",
        color="#444",
    )

    ax1.hist(log_amt, bins=56, color="#606c38", edgecolor="white", linewidth=0.4, alpha=0.88)
    ax1.set_xlabel("log1p(Amount)")
    ax1.set_ylabel("Count")
    ax1.set_title("After log1p(Amount)")
    ax1.text(
        0.97,
        0.92,
        "≈ symmetric:\nkernel sees every\nfeature fairly.",
        transform=ax1.transAxes,
        fontsize=9.5,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#ccc"),
    )

    fig.suptitle("Amount: before and after log1p", fontsize=14, fontweight="600", y=1.02)
    fig.text(0.5, -0.02, subtitle, ha="center", fontsize=9.5, color="#555")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_cyclical_hour_fraud_rate(out_path: Path, df) -> None:
    """Fraud rate by clock hour bucket (48h span → hour indices 0–47)."""
    import matplotlib.pyplot as plt

    if df is None:
        rng = np.random.default_rng(17)
        hours = np.arange(48)
        fraud_rate = 0.003 + 0.012 * np.sin((hours - 22) * np.pi / 12) ** 2 + rng.uniform(-0.002, 0.002, 48)
        fraud_rate = np.clip(fraud_rate, 0.001, 0.06)
        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        ax.bar(hours, 100 * fraud_rate, color="#bc6c25", edgecolor="white", linewidth=0.5, alpha=0.9)
        ax.set_xlabel("Hour bucket (Time / 3600, 48-hour dataset)")
        ax.set_ylabel("Fraud rate (%)")
        ax.set_title("Illustrative hourly fraud pattern (synthetic — use CSV for real rates)")
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    t = df["Time"].to_numpy(dtype=np.float64)
    y = df["Class"].to_numpy(dtype=np.int64)
    hour_bucket = np.minimum((t / 3600.0).astype(np.int64), 47)

    fraud_rate = []
    counts = []
    for h in range(48):
        m = hour_bucket == h
        c = int(m.sum())
        counts.append(c)
        if c > 0:
            fraud_rate.append(float(np.mean(y[m])))
        else:
            fraud_rate.append(0.0)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(range(48), np.array(fraud_rate) * 100, color="#2d6a8f", edgecolor="white", linewidth=0.45, alpha=0.9)
    ax.set_xlabel("Hour index (floor(Time/3600), capped at 47)")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud rate by hour bucket — motivates cyclical encoding")
    ax.axhline(100 * np.mean(y), color="#888", linestyle=":", linewidth=1.2, label=f"Overall fraud rate ({100*np.mean(y):.3f}%)")
    ax.legend(loc="upper right", fontsize=9)
    fig.text(
        0.5,
        -0.02,
        "Modulo 24 maps buckets onto clock hour (e.g. buckets 24–47 → hours 0–23). Off-hours spikes justify Hour_sin / Hour_cos.",
        ha="center",
        fontsize=9,
        color="#444",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_hour_sin_cos_circle(out_path: Path, df) -> None:
    """Scatter Hour_sin vs Hour_cos; highlights adjacency of hour 23 and hour 0."""
    import matplotlib.pyplot as plt

    theta = np.linspace(0, 2 * np.pi, 100)
    fig, ax = plt.subplots(figsize=(6.8, 6.8))

    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    ax.plot(circle_x, circle_y, color="#bbb", linewidth=1.5, linestyle="--", label="Unit circle")
    ax.axhline(0, color="#ddd", linewidth=1)
    ax.axvline(0, color="#ddd", linewidth=1)

    label_hours = [0, 6, 12, 18, 23]
    for h in label_hours:
        sx = np.sin(2 * np.pi * h / 24)
        cy = np.cos(2 * np.pi * h / 24)
        ax.scatter([sx], [cy], s=120, c="#1a1a2e", zorder=5, edgecolors="white")
        ax.annotate(f"h={h}", (sx, cy), textcoords="offset points", xytext=(8, 6), fontsize=10, fontweight="bold")

    ax.annotate(
        "",
        xy=(np.sin(2 * np.pi * 0 / 24), np.cos(2 * np.pi * 0 / 24)),
        xytext=(np.sin(2 * np.pi * 23 / 24), np.cos(2 * np.pi * 23 / 24)),
        arrowprops=dict(arrowstyle="<->", color="#c44536", lw=2),
    )
    ax.text(
        0.05,
        -1.15,
        "Nearby on the circle → nearby in (sin, cos); hour 23 ≈ hour 0.",
        fontsize=10,
        color="#c44536",
        transform=ax.transAxes,
    )

    if df is not None and len(df) > 5:
        rng = np.random.default_rng(99)
        n = min(8000, len(df))
        idx = rng.choice(len(df), size=n, replace=False)
        sub = df.iloc[idx]
        hour = np.mod(sub["Time"].to_numpy(dtype=np.float64) / 3600.0, 24.0)
        hx = np.sin(2 * np.pi * hour / 24)
        hy = np.cos(2 * np.pi * hour / 24)
        col = np.where(sub["Class"].to_numpy() == 1, "#c44536", "#2d6a8f")
        ax.scatter(hx, hy, c=col, alpha=0.08, s=10, linewidths=0)
        ax.scatter([], [], c="#2d6a8f", s=36, label="Legit (sample)")
        ax.scatter([], [], c="#c44536", s=36, label="Fraud (sample)")
        ax.legend(loc="upper left", fontsize=9)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel("Hour_sin")
    ax.set_ylabel("Hour_cos")
    ax.set_title("Cyclical hour encoding (24h modulo)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_is_micro_fraud_rates(out_path: Path, df) -> None:
    """Fraud prevalence: Amount < €1 vs Amount ≥ €1."""
    import matplotlib.pyplot as plt

    if df is None:
        rng = np.random.default_rng(5)
        n = 55_000
        is_fraud = rng.random(n) < 0.002
        amt = rng.lognormal(4, 2.0, n)
        amt = np.clip(amt, 0.0, 25_691.0)
        ping = is_fraud & (rng.random(n) < 0.42)
        amt[ping] = rng.uniform(0.0, 0.99, size=int(ping.sum()))
        micro = amt < 1.0
        r_micro = float(np.mean(is_fraud[micro])) if micro.any() else 0.0
        r_not = float(np.mean(is_fraud[~micro])) if (~micro).any() else 0.0
        labels = ["Amount < €1\n(micro)", "Amount ≥ €1"]
        rates = [100 * r_micro, 100 * r_not]
        subtitle = "Synthetic — download creditcard.csv for dataset-specific rates."
    else:
        y = df["Class"].to_numpy(dtype=np.int64)
        amt = df["Amount"].to_numpy(dtype=np.float64)
        micro = amt < 1.0
        r_micro = float(np.mean(y[micro])) if micro.any() else 0.0
        r_not = float(np.mean(y[~micro])) if (~micro).any() else 0.0
        labels = ["Amount < €1\n(micro ping)", "Amount ≥ €1"]
        rates = [100 * r_micro, 100 * r_not]
        subtitle = "Share of rows labeled fraud (Class=1) within each amount bucket."

    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = ["#bc6c25", "#2d6a8f"]
    bars = ax.bar(labels, rates, color=colors, edgecolor="white", linewidth=1)
    ax.set_ylabel("% of transactions that are fraud")
    ax.set_title("Is_micro pattern — fraud probes often start tiny")
    ymax = max(rates) * 1.35 if rates else 1
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, min(r + ymax * 0.03, ymax * 0.97), f"{r:.2f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, ymax)
    fig.text(0.5, -0.06, subtitle, ha="center", fontsize=9.5, color="#444")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def figure_score_separation(out_path: Path, xy_bundle) -> None:
    """Overlapping histograms of decision scores for fraud vs legit (test)."""
    import matplotlib.pyplot as plt
    from sklearn.svm import LinearSVC

    if xy_bundle is None:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.text(0.5, 0.5, "Processed data not found — skipping score separation.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    X_train, y_train, X_test, y_test = xy_bundle
    X_tr, y_tr = _subsample_for_speed(X_train, y_train, max_rows=30_000)
    X_te, y_te = _subsample_for_speed(X_test, y_test, max_rows=20_000)

    clf = LinearSVC(class_weight="balanced", max_iter=3000, random_state=42, dual=False, C=0.1)
    clf.fit(X_tr, y_tr)
    s = clf.decision_function(X_te)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(s[y_te == 0], bins=45, density=True, alpha=0.65, color="#2d6a8f", label="Legit", edgecolor="white")
    ax.hist(s[y_te == 1], bins=25, density=True, alpha=0.72, color="#c44536", label="Fraud", edgecolor="white")
    thr = float(np.percentile(s[y_te == 1], 15))
    ax.axvline(thr, color="#111", linestyle="--", linewidth=1.5, label="Example threshold")
    ax.set_xlabel("Decision function score")
    ax.set_ylabel("Density")
    ax.set_title("Score separation (LinearSVC on subsample)")
    ax.legend(framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    _configure_matplotlib()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    xy = _load_xy_if_available()
    if xy is None:
        print("Note: processed .npy files not found — PR and score plots use fallbacks or partial data.")

    df_raw = _load_creditcard_csv()
    if df_raw is None:
        print("Note: creditcard.csv not found (or pandas missing) — hourly / micro plots use synthetic or partial overlays.")

    print(f"Writing figures to {FIGURES_DIR}/")
    figure_margin_diagram(FIGURES_DIR / "margin_diagram.png")
    print("  margin_diagram.png")

    figure_threshold_tradeoff(FIGURES_DIR / "threshold_tradeoff.png", xy)
    print("  threshold_tradeoff.png")

    figure_class_imbalance_bars(FIGURES_DIR / "class_imbalance_bars.png")
    print("  class_imbalance_bars.png")

    figure_score_separation(FIGURES_DIR / "score_separation.png", xy)
    print("  score_separation.png")

    figure_feature_engineering_before_after(FIGURES_DIR / "feature_engineering_before_after.png", df_raw)
    print("  feature_engineering_before_after.png")

    figure_cyclical_hour_fraud_rate(FIGURES_DIR / "cyclical_hour_fraud_rate.png", df_raw)
    print("  cyclical_hour_fraud_rate.png")

    figure_hour_sin_cos_circle(FIGURES_DIR / "hour_sin_cos_circle.png", df_raw)
    print("  hour_sin_cos_circle.png")

    figure_is_micro_fraud_rates(FIGURES_DIR / "is_micro_fraud_rates.png", df_raw)
    print("  is_micro_fraud_rates.png")

    figure_pr_curve_operating_points(FIGURES_DIR / "pr_curve_with_operating_points.png")
    print("  pr_curve_with_operating_points.png")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
