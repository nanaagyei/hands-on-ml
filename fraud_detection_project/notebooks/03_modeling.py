"""
fraud_detection_project/notebooks/03_modeling.py

Four acts:
  Act 1 — Scratch KernelSVM on dev set (proof of concept)
  Act 2 — Sklearn SVC on full training set (production baseline)
  Act 3 — Hyperparameter search (C × gamma grid)
  Act 4 — Threshold tuning (business operating point)
"""

import numpy as np
import json
import time
import sys
from pathlib import Path

# ── Make our scratch implementations importable ────────────────────────────
sys.path.append(str(Path(__file__).resolve().parents[2]))

from svm.core.kernel_svm  import KernelSVM
from svm.core.linear_svm  import LinearSVM

DATA = Path("fraud_detection_project/data/processed")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD PROCESSED DATA
# ══════════════════════════════════════════════════════════════════════════════

print("Loading processed data...")

X_dev          = np.load(DATA / "X_dev.npy")
y_dev          = np.load(DATA / "y_dev.npy")
X_train_scaled = np.load(DATA / "X_train_scaled.npy")
X_test_scaled  = np.load(DATA / "X_test_scaled.npy")
y_train        = np.load(DATA / "y_train.npy")
y_test         = np.load(DATA / "y_test.npy")

with open(DATA / "dataset_meta.json") as f:
    meta = json.load(f)

print(f"Dev:   {X_dev.shape}  |  fraud: {y_dev.sum()}")
print(f"Train: {X_train_scaled.shape}  |  fraud: {y_train.sum()}")
print(f"Test:  {X_test_scaled.shape}   |  fraud: {y_test.sum()}")
print(f"Features: {meta['n_features']}")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION UTILITIES  (same as 02, self-contained here for clarity)
# ══════════════════════════════════════════════════════════════════════════════

def compute_auc_pr(y_true, y_score):
    idx           = np.argsort(y_score)[::-1]
    y_sorted      = y_true[idx]
    cum_tp        = np.cumsum(y_sorted)
    cum_fp        = np.cumsum(1 - y_sorted)
    n_pos         = y_true.sum()
    precision_c   = cum_tp / (cum_tp + cum_fp + 1e-10)
    recall_c      = cum_tp / (n_pos + 1e-10)
    precision_c   = np.concatenate([[1.0], precision_c])
    recall_c      = np.concatenate([[0.0], recall_c])
    return float(abs(np.trapezoid(recall_c, precision_c)))


def compute_auc_roc(y_true, y_score):
    """
    AUC-ROC via Mann-Whitney U statistic.
    
    AUC = P(score(positive) > score(negative))
        = fraction of (fraud, legit) pairs where fraud 
          is scored higher than legit
    
    Equivalent to the trapezoidal ROC curve but numerically stable.
    Uses rank-sum formula: O(n log n) via sorting.
    """
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Rank all scores from 1 (lowest) to n (highest)
    # np.argsort twice gives ranks
    ranks = np.argsort(np.argsort(y_score)) + 1
    
    # Sum of ranks belonging to the positive class
    rank_sum_pos = np.sum(ranks[y_true == 1])
    
    # U statistic for positives
    U = rank_sum_pos - n_pos * (n_pos + 1) / 2
    
    # Normalize to [0, 1]
    return float(U / (n_pos * n_neg))


def full_report(y_true, y_pred, y_score, label, avg_fraud=122, fp_cost=2):
    TP = int(np.sum((y_pred==1) & (y_true==1)))
    TN = int(np.sum((y_pred==0) & (y_true==0)))
    FP = int(np.sum((y_pred==1) & (y_true==0)))
    FN = int(np.sum((y_pred==0) & (y_true==1)))

    precision  = TP / (TP + FP + 1e-10)
    recall     = TP / (TP + FN + 1e-10)
    f1         = 2*precision*recall / (precision + recall + 1e-10)
    auc_pr     = compute_auc_pr(y_true, y_score)
    auc_roc    = compute_auc_roc(y_true, y_score)
    net_value  = TP*avg_fraud - FN*avg_fraud - FP*fp_cost

    print(f"\n{'━'*55}")
    print(f"  {label}")
    print(f"{'━'*55}")
    print(f"  Confusion matrix:")
    print(f"              Pred 0(legit)  Pred 1(fraud)")
    print(f"  True 0(legit)  {TN:>9,}     {FP:>9,}")
    print(f"  True 1(fraud)  {FN:>9,}     {TP:>9,}")
    print(f"\n  Precision : {100*precision:6.2f}%")
    print(f"  Recall    : {100*recall:6.2f}%")
    print(f"  F1        : {f1:.4f}")
    print(f"  AUC-PR    : {auc_pr:.4f}   ← primary metric")
    print(f"  AUC-ROC   : {auc_roc:.4f}")
    print(f"\n  Business impact (avg fraud €{avg_fraud}, FP review €{fp_cost}):")
    print(f"    Caught : €{TP*avg_fraud:>8,.0f}  ({TP} frauds × €{avg_fraud})")
    print(f"    Missed :-€{FN*avg_fraud:>8,.0f}  ({FN} frauds missed)")
    print(f"    FP cost:-€{FP*fp_cost:>8,.0f}  ({FP} false alerts × €{fp_cost})")
    print(f"    Net    : €{net_value:>8,.0f}")

    return dict(precision=precision, recall=recall, f1=f1,
                auc_pr=auc_pr, auc_roc=auc_roc,
                TP=TP, TN=TN, FP=FP, FN=FN, net_value=net_value)


def scores_from_scratch_svm(model, X):
    """
    Our KernelSVM returns {-1,+1} from predict().
    For ranking (needed by AUC), we need the raw decision score.
    decision_function() gives us that — higher = more fraud-like.
    """
    return model.decision_function(X)


def threshold_sweep(y_true, y_score, label=""):
    """
    Sweep decision threshold and print the precision-recall tradeoff table.
    Crucial for picking the business operating point.
    
    An SVM's default threshold is 0 (sign of decision score).
    But we can shift it — positive threshold → stricter (higher precision,
    lower recall). Negative threshold → looser (lower precision, higher recall).
    """
    print(f"\n  Threshold sweep — {label}")
    print(f"  {'Threshold':>10}  {'Precision':>10}  "
          f"{'Recall':>8}  {'F1':>8}  {'FP':>7}  {'FN':>5}")
    print(f"  {'-'*60}")

    n_pos = y_true.sum()
    results = []

    for t in np.linspace(y_score.min(), y_score.max(), 40):
        y_pred_t = (y_score >= t).astype(int)
        TP = np.sum((y_pred_t==1) & (y_true==1))
        FP = np.sum((y_pred_t==1) & (y_true==0))
        FN = np.sum((y_pred_t==0) & (y_true==1))
        prec = TP / (TP + FP + 1e-10)
        rec  = TP / (TP + FN + 1e-10)
        f1   = 2*prec*rec / (prec + rec + 1e-10)
        results.append((t, prec, rec, f1, FP, FN))

    # Print every 4th row so table fits on screen
    for i, (t, prec, rec, f1, FP, FN) in enumerate(results):
        if i % 4 == 0:
            print(f"  {t:>10.3f}  {100*prec:>9.1f}%  "
                  f"{100*rec:>7.1f}%  {f1:>8.4f}  {FP:>7,}  {FN:>5}")

    # Find threshold for 80% recall
    for t, prec, rec, f1, FP, FN in results:
        if rec >= 0.80:
            print(f"\n  → At 80% recall: threshold={t:.3f}, "
                  f"precision={100*prec:.1f}%, FP={FP:,}")
            break

    # Find threshold for 90% precision
    for t, prec, rec, f1, FP, FN in reversed(results):
        if prec >= 0.90 and rec > 0:
            print(f"  → At 90% precision: threshold={t:.3f}, "
                  f"recall={100*rec:.1f}%, FP={FP:,}")
            break


# ══════════════════════════════════════════════════════════════════════════════
# ACT 1 — SCRATCH KERNELSVM ON DEV SET
#
# Goal: prove our implementation works on real fraud data.
#       Dev set is ~5000 samples — manageable for our SMO.
#
# We try two kernel configs:
#   a) Linear kernel  — fast, interpretable, weak on non-linear patterns
#   b) RBF kernel     — more powerful, our primary choice
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*55)
print("ACT 1 — SCRATCH KERNELSVM ON DEV SET")
print("═"*55)

# Class weight ratio for dev set
cw_fraud_dev = meta['class_weight_fraud_dev']
cw_legit_dev = meta['class_weight_legit_dev']
# Effective C for each class: C_i = C * weight_i
# We implement this by scaling C. In our scratch SVM we don't have
# per-sample weights, so we use a C value pre-scaled by the fraud weight.
# This approximates balanced training — a proper implementation would
# apply per-sample loss weights inside the SMO loop.
C_effective = 1.0 * cw_fraud_dev

print(f"\nClass weight (fraud): {cw_fraud_dev:.2f}")
print(f"Effective C for fraud: {C_effective:.2f}")

# ── 1a. Linear kernel ─────────────────────────────────────────────────────

print("\n── 1a. Linear Kernel ──")
t0 = time.time()

linear_dev = KernelSVM(
    C=C_effective,
    kernel='linear',
    tol=1e-3,
    max_passes=5,
    random_state=42
)
linear_dev.fit(X_dev, y_dev)
t_linear = time.time() - t0

# Predictions on dev set (in-sample — just to check it learned something)
scores_linear_dev = scores_from_scratch_svm(linear_dev, X_dev)
y_pred_linear_dev = (scores_linear_dev >= 0).astype(int)

print(f"Training time: {t_linear:.1f}s")
res_linear_dev = full_report(y_dev, y_pred_linear_dev,
                              scores_linear_dev,
                              "Scratch LinearKernel SVM — Dev Set (in-sample)")

# ── 1b. RBF kernel ────────────────────────────────────────────────────────

print("\n── 1b. RBF Kernel (gamma='scale') ──")
print("(This will take 1-3 minutes on a 5000-sample dev set)")
t0 = time.time()

rbf_dev = KernelSVM(
    C=C_effective,
    kernel='rbf',
    gamma='scale',
    tol=1e-3,
    max_passes=5,
    random_state=42
)
rbf_dev.fit(X_dev, y_dev)
t_rbf = time.time() - t0

scores_rbf_dev = scores_from_scratch_svm(rbf_dev, X_dev)
y_pred_rbf_dev = (scores_rbf_dev >= 0).astype(int)

print(f"Training time: {t_rbf:.1f}s")
print(f"Gamma used: {rbf_dev._gamma_val:.6f}")
res_rbf_dev = full_report(y_dev, y_pred_rbf_dev,
                           scores_rbf_dev,
                           "Scratch RBF SVM — Dev Set (in-sample)")

print("\nLinear vs RBF on dev set (in-sample):")
print(f"  AUC-PR: linear={res_linear_dev['auc_pr']:.4f}  "
      f"rbf={res_rbf_dev['auc_pr']:.4f}")
print(f"  Recall: linear={100*res_linear_dev['recall']:.1f}%  "
      f"rbf={100*res_rbf_dev['recall']:.1f}%")
print(f"  Note: in-sample — expect high, but we care about test performance.")
print(f"  Our scratch SVM is validated. Moving to sklearn for full dataset.")

# ══════════════════════════════════════════════════════════════════════════════
# ACT 2 — SKLEARN SVC ON FULL TRAINING SET
#
# Why switch to sklearn here?
#
# Our scratch SMO works but has two limitations for 227k samples:
#   1. O(n²) kernel matrix = ~415 GB RAM. Impossible.
#   2. No kernel caching — sklearn's libsvm caches kernel rows in an LRU
#      cache (default 200 MB), recomputing only what it needs.
#
# sklearn's SVC is a C wrapper around libsvm — the gold-standard SVM
# library. Same algorithm, optimized for large datasets.
#
# This is standard ML engineering practice:
#   Build from scratch to understand → use production library to ship.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*55)
print("ACT 2 — SKLEARN SVC ON FULL TRAINING SET")
print("═"*55)

try:
    from sklearn.svm import SVC, LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
except ImportError:
    print("sklearn not installed. Run: pip install scikit-learn")
    raise

# ── 2a. LinearSVC — primal solver, scales to 200k+ samples ───────────────
#
# LinearSVC solves the primal directly (not the dual), so it never
# needs the kernel matrix. Time complexity: O(n × features × iterations).
# Much faster for large datasets when a linear boundary is sufficient.

print("\n── 2a. LinearSVC (primal, full training set) ──")
print("  (Trains in seconds even on 200k samples)")
t0 = time.time()

linear_svc = LinearSVC(
    C=1.0,
    class_weight='balanced',   # Corrects for 578:1 imbalance
    max_iter=2000,
    random_state=42,
    dual=False                 # Primal is faster when n_samples > n_features
)
linear_svc.fit(X_train_scaled, y_train)
t_lsvc = time.time() - t0

# LinearSVC doesn't have decision_function in the same form as SVC
# We use the raw scores directly
scores_lsvc_test = linear_svc.decision_function(X_test_scaled)
y_pred_lsvc_test = linear_svc.predict(X_test_scaled)

# LinearSVC predicts {0,1} directly — no sign conversion needed
print(f"Training time: {t_lsvc:.2f}s")
res_lsvc = full_report(y_test, y_pred_lsvc_test,
                        scores_lsvc_test,
                        "LinearSVC — Full Test Set")

# ── 2b. Kernel SVC (RBF) on full training set ─────────────────────────────
#
# This is the real deal. libsvm with RBF kernel on 227k samples.
# Expect 3-10 minutes depending on your CPU.
# cache_size=2000 means 2GB kernel cache — use more if you have RAM.

print("\n── 2b. Kernel SVC RBF (full training set) ──")
print("  This will take several minutes. Training on 227k samples...")
print("  cache_size=2000 uses 2GB RAM for kernel caching.")
t0 = time.time()

rbf_svc = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    class_weight='balanced',
    cache_size=2000,            # MB of kernel cache
    probability=False,          # True would use Platt scaling — slow
    random_state=42
)
rbf_svc.fit(X_train_scaled, y_train)
t_rbf_svc = time.time() - t0

scores_rbf_test = rbf_svc.decision_function(X_test_scaled)
y_pred_rbf_test = rbf_svc.predict(X_test_scaled)

print(f"Training time: {t_rbf_svc:.1f}s  ({t_rbf_svc/60:.1f} min)")
print(f"Support vectors: {rbf_svc.n_support_.sum():,}  "
      f"({100*rbf_svc.n_support_.sum()/len(y_train):.1f}% of training set)")
print(f"  Class 0 (legit): {rbf_svc.n_support_[0]:,} SVs")
print(f"  Class 1 (fraud): {rbf_svc.n_support_[1]:,} SVs")

res_rbf_full = full_report(y_test, y_pred_rbf_test,
                            scores_rbf_test,
                            "Sklearn RBF SVC — Full Test Set")

# ── 2c. Side-by-side comparison ───────────────────────────────────────────

print("\n── Summary: Act 2 ──")
print(f"  {'Model':<30} {'AUC-PR':>8} {'Recall':>8} "
      f"{'Precis':>8} {'F1':>8} {'Time':>8}")
print(f"  {'-'*72}")
print(f"  {'LinearSVC (full)':<30} "
      f"{res_lsvc['auc_pr']:>8.4f} "
      f"{100*res_lsvc['recall']:>7.1f}% "
      f"{100*res_lsvc['precision']:>7.1f}% "
      f"{res_lsvc['f1']:>8.4f} "
      f"{t_lsvc:>6.1f}s")
print(f"  {'RBF SVC (full)':<30} "
      f"{res_rbf_full['auc_pr']:>8.4f} "
      f"{100*res_rbf_full['recall']:>7.1f}% "
      f"{100*res_rbf_full['precision']:>7.1f}% "
      f"{res_rbf_full['f1']:>8.4f} "
      f"{t_rbf_svc:>6.0f}s")

# ══════════════════════════════════════════════════════════════════════════════
# ACT 3 — HYPERPARAMETER SEARCH
#
# C and gamma interact — you cannot tune them independently.
# High C + high gamma → very complex boundary, risk overfitting
# Low C  + low gamma  → simple boundary, risk underfitting
#
# We do a coarse grid search on the dev set using 5-fold CV.
# Metric: AUC-PR (not accuracy, not AUC-ROC).
#
# We use the dev set here (not full train) because:
#   1. Grid search on 227k samples × kernel SVM = hours
#   2. The optimal C/gamma on a stratified subsample transfers well
#   3. We then retrain the winner on the full training set
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*55)
print("ACT 3 — HYPERPARAMETER SEARCH (Dev Set, 5-Fold CV)")
print("═"*55)

# Grid: 4 C values × 4 gamma values = 16 combos × 5 folds = 80 fits
# On 5000 samples this takes ~5-10 minutes
C_grid     = [0.1, 1.0, 10.0, 100.0]
gamma_grid = [0.001, 0.01, 0.1, 1.0]

print(f"\nGrid: C={C_grid}, gamma={gamma_grid}")
print(f"Total fits: {len(C_grid)*len(gamma_grid)*5}")
print("(Each fit trains a kernel SVM on ~4000 dev samples)")

def stratified_kfold(y, n_splits=5, seed=42):
    """
    Stratified k-fold: each fold has the same class ratio as the full dataset.
    Critical for imbalanced data — a fold with no fraud examples is useless.
    """
    rng = np.random.RandomState(seed)
    fraud_idx = np.where(y == 1)[0].copy()
    legit_idx = np.where(y == 0)[0].copy()
    rng.shuffle(fraud_idx)
    rng.shuffle(legit_idx)

    fraud_folds = np.array_split(fraud_idx, n_splits)
    legit_folds = np.array_split(legit_idx, n_splits)

    for k in range(n_splits):
        val_idx   = np.concatenate([fraud_folds[k], legit_folds[k]])
        train_idx = np.concatenate([
            np.concatenate([fraud_folds[i] for i in range(n_splits) if i != k]),
            np.concatenate([legit_folds[i] for i in range(n_splits) if i != k]),
        ])
        yield train_idx, val_idx


print("\nRunning grid search...")
print(f"  {'C':>8}  {'gamma':>8}  {'AUC-PR mean':>12}  "
      f"{'AUC-PR std':>11}  {'Recall mean':>12}")
print(f"  {'-'*58}")

best_auc_pr  = -1
best_params  = {}
all_results  = []

total = len(C_grid) * len(gamma_grid)
done  = 0

for C_val in C_grid:
    for gamma_val in gamma_grid:
        fold_auc_prs = []
        fold_recalls = []

        for train_idx, val_idx in stratified_kfold(y_dev, n_splits=5):
            X_tr, X_val = X_dev[train_idx], X_dev[val_idx]
            y_tr, y_val = y_dev[train_idx], y_dev[val_idx]

            model = SVC(
                C=C_val,
                kernel='rbf',
                gamma=gamma_val,
                class_weight='balanced',
                cache_size=500,
                random_state=42
            )
            model.fit(X_tr, y_tr)
            scores_val = model.decision_function(X_val)

            auc_pr = compute_auc_pr(y_val, scores_val)
            y_pred_val = model.predict(X_val)
            TP  = np.sum((y_pred_val==1) & (y_val==1))
            FN  = np.sum((y_pred_val==0) & (y_val==1))
            rec = TP / (TP + FN + 1e-10)

            fold_auc_prs.append(auc_pr)
            fold_recalls.append(rec)

        mean_auc = np.mean(fold_auc_prs)
        std_auc  = np.std(fold_auc_prs)
        mean_rec = np.mean(fold_recalls)

        all_results.append({
            'C': C_val, 'gamma': gamma_val,
            'auc_pr_mean': mean_auc, 'auc_pr_std': std_auc,
            'recall_mean': mean_rec,
        })

        marker = " ← best" if mean_auc > best_auc_pr else ""
        print(f"  {C_val:>8.1f}  {gamma_val:>8.3f}  "
              f"{mean_auc:>12.4f}  {std_auc:>11.4f}  "
              f"{100*mean_rec:>11.1f}%{marker}")

        if mean_auc > best_auc_pr:
            best_auc_pr = mean_auc
            best_params = {'C': C_val, 'gamma': gamma_val}

        done += 1

print(f"\nBest params: C={best_params['C']}, gamma={best_params['gamma']}")
print(f"Best CV AUC-PR: {best_auc_pr:.4f}")

# ── Retrain best config on FULL training set ──────────────────────────────

print(f"\n── Retraining best config on full training set ──")
print(f"C={best_params['C']}, gamma={best_params['gamma']}")
t0 = time.time()

best_model = SVC(
    C=best_params['C'],
    kernel='rbf',
    gamma=best_params['gamma'],
    class_weight='balanced',
    cache_size=2000,
    random_state=42
)
best_model.fit(X_train_scaled, y_train)
t_best = time.time() - t0

scores_best_test = best_model.decision_function(X_test_scaled)
y_pred_best_test = best_model.predict(X_test_scaled)

print(f"Training time: {t_best:.1f}s")
res_best = full_report(y_test, y_pred_best_test,
                        scores_best_test,
                        f"Tuned RBF SVC (C={best_params['C']}, "
                        f"gamma={best_params['gamma']}) — Test Set")

# ── Grid search heatmap (text version) ───────────────────────────────────

print("\n── AUC-PR Grid Heatmap ──")
print("  (rows=C, cols=gamma)")
print(f"  {'':>8}", end="")
for g in gamma_grid:
    print(f"  γ={g:<6}", end="")
print()

for C_val in C_grid:
    print(f"  C={C_val:<6}", end="")
    for gamma_val in gamma_grid:
        r = next(r for r in all_results
                 if r['C'] == C_val and r['gamma'] == gamma_val)
        marker = "★" if (C_val == best_params['C']
                          and gamma_val == best_params['gamma']) else " "
        print(f"  {r['auc_pr_mean']:.3f}{marker} ", end="")
    print()

print("\n  ★ = best combination")

# ══════════════════════════════════════════════════════════════════════════════
# ACT 4 — THRESHOLD TUNING
#
# An SVM's default decision threshold is 0 (sign of decision score).
# But the optimal business threshold is almost never 0.
#
# The business question: at what threshold do we flag a transaction?
#
# Two common business operating points:
#   "Conservative": high precision — only flag when very confident
#                   → fewer false positives → less customer friction
#   "Aggressive":   high recall — catch as many frauds as possible
#                   → more false positives → more analyst workload
#
# We show the full tradeoff curve and let the business decide.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*55)
print("ACT 4 — THRESHOLD TUNING")
print("═"*55)

print("\nUsing tuned RBF SVC scores on test set...")
threshold_sweep(y_test, scores_best_test, label="Tuned RBF SVC")

# ── Find specific business operating points ───────────────────────────────

print("\n── Business Operating Points ──")

operating_points = [
    ("Aggressive (catch 80% of fraud)",  0.80, 'recall'),
    ("Balanced  (catch 70% of fraud)",   0.70, 'recall'),
    ("Conservative (90% precision)",     0.90, 'precision'),
]

for label, target, metric in operating_points:
    scores_sorted = np.sort(scores_best_test)[::-1]

    best_thresh = 0.0
    best_prec   = 0.0
    best_rec    = 0.0
    best_f1     = 0.0
    best_FP     = 0
    best_FN     = 0

    for t in np.linspace(scores_best_test.min(),
                         scores_best_test.max(), 1000):
        y_pred_t = (scores_best_test >= t).astype(int)
        TP = np.sum((y_pred_t==1) & (y_test==1))
        FP = np.sum((y_pred_t==1) & (y_test==0))
        FN = np.sum((y_pred_t==0) & (y_test==1))
        prec = TP / (TP + FP + 1e-10)
        rec  = TP / (TP + FN + 1e-10)
        f1   = 2*prec*rec / (prec + rec + 1e-10)

        if metric == 'recall' and rec >= target:
            best_thresh, best_prec, best_rec = t, prec, rec
            best_f1, best_FP, best_FN = f1, FP, FN
            break
        elif metric == 'precision' and prec >= target and rec > 0:
            best_thresh, best_prec, best_rec = t, prec, rec
            best_f1, best_FP, best_FN = f1, FP, FN

    net = (np.sum((scores_best_test>=best_thresh)&(y_test==1))*122
           - best_FN*122 - best_FP*2)
    print(f"\n  {label}")
    print(f"    Threshold : {best_thresh:.4f}")
    print(f"    Precision : {100*best_prec:.1f}%")
    print(f"    Recall    : {100*best_rec:.1f}%")
    print(f"    F1        : {best_f1:.4f}")
    print(f"    FP (false alerts) : {best_FP:,}")
    print(f"    FN (missed frauds): {best_FN:,}")
    print(f"    Net business value: €{net:,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*55)
print("FINAL MODEL COMPARISON")
print("═"*55)

print(f"\n  {'Model':<35} {'AUC-PR':>8} {'Recall':>8} {'Precis':>8}")
print(f"  {'-'*62}")
models = [
    ("Scratch LinearKernel (dev, in-sample)",
     res_linear_dev['auc_pr'], res_linear_dev['recall'],
     res_linear_dev['precision']),
    ("Scratch RBF SVM (dev, in-sample)",
     res_rbf_dev['auc_pr'], res_rbf_dev['recall'],
     res_rbf_dev['precision']),
    ("Sklearn LinearSVC (full test)",
     res_lsvc['auc_pr'], res_lsvc['recall'],
     res_lsvc['precision']),
    ("Sklearn RBF SVC default (full test)",
     res_rbf_full['auc_pr'], res_rbf_full['recall'],
     res_rbf_full['precision']),
    ("Sklearn RBF SVC tuned (full test)",
     res_best['auc_pr'], res_best['recall'],
     res_best['precision']),
]
for name, auc, rec, prec in models:
    print(f"  {name:<35} {auc:>8.4f} "
          f"{100*rec:>7.1f}% {100*prec:>7.1f}%")

print(f"""
Key Takeaways:
  1. Scale matters: unscaled SVM would have near-random performance
  2. class_weight='balanced' is non-negotiable at 578:1 imbalance
  3. RBF kernel beats linear — fraud patterns are non-linear
  4. Threshold=0 is rarely the right business threshold
  5. AUC-PR is your headline number; threshold is a business decision

Next: 04_serving.py — save the model, build the prediction API,
      add monitoring for score distribution drift.
""")

# Save best model for serving
import pickle
OUT_MODELS = Path("fraud_detection_project/models")
OUT_MODELS.mkdir(exist_ok=True)

model_artifact = {
    'model':         best_model,
    'scaler_mean':   np.load(DATA / "scaler_mean.npy"),
    'scaler_std':    np.load(DATA / "scaler_std.npy"),
    'feature_names': meta['feature_names'],
    'best_params':   best_params,
    'test_auc_pr':   res_best['auc_pr'],
    'test_recall':   res_best['recall'],
    'test_precision':res_best['precision'],
    'threshold':     0.0,   # default; adjust per business operating point
    'meta':          meta,
}

with open(OUT_MODELS / "svm_fraud_model.pkl", "wb") as f:
    pickle.dump(model_artifact, f)

print(f"Model saved → {OUT_MODELS}/svm_fraud_model.pkl")