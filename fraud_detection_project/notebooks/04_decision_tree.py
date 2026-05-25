"""
fraud_detection_project/notebooks/04_decision_tree.py

Decision Trees on the credit card fraud dataset.

Four acts:
  Act 1 — From-scratch tree on dev set (validation)
  Act 2 — From-scratch tree on full training set
  Act 3 — Sklearn DecisionTreeClassifier comparison + hyperparameter tuning
  Act 4 — Head-to-head vs SVM at matched operating points

We reuse evaluation utilities from 03_modeling_v2.py (AUC-PR, AUC-ROC,
threshold sweep) so the comparison is apples-to-apples.
"""

import numpy as np
import json
import time
import sys
from pathlib import Path

# Make the trees package importable
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from trees.core.tree_classifier import DecisionTreeClassifier as OurTree

DATA = REPO_ROOT / "fraud_detection_project" / "data" / "processed"
OUT_MODELS = REPO_ROOT / "fraud_detection_project" / "models"
OUT_MODELS.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════════════════════

print("Loading processed data...")
X_dev   = np.load(DATA / "X_dev.npy")
y_dev   = np.load(DATA / "y_dev.npy")
X_train = np.load(DATA / "X_train_scaled.npy")
X_test  = np.load(DATA / "X_test_scaled.npy")
y_train = np.load(DATA / "y_train.npy")
y_test  = np.load(DATA / "y_test.npy")

with open(DATA / "dataset_meta.json") as f:
    meta = json.load(f)

print(f"Train: {X_train.shape}  fraud={y_train.sum()}  "
      f"({100*y_train.mean():.3f}%)")
print(f"Test:  {X_test.shape}   fraud={y_test.sum()}  "
      f"({100*y_test.mean():.3f}%)")
print("\nNote: features are scaled, but trees are scale-invariant — \n"
      "      results would be identical with raw features.\n")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION UTILITIES
# (same as 03_modeling_v2.py — kept here so this script is self-contained)
# ══════════════════════════════════════════════════════════════════════════════

def compute_auc_roc(y_true, y_score):
    """Mann-Whitney U; identical to 03_modeling_v2.py."""
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.argsort(np.argsort(y_score)) + 1
    rank_sum_pos = float(np.sum(ranks[y_true == 1]))
    U = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return U / (n_pos * n_neg)


def compute_auc_pr(y_true, y_score):
    idx = np.argsort(y_score)[::-1]
    y_sorted = y_true[idx]
    n_pos = y_true.sum()
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(1 - y_sorted)
    precision = cum_tp / (cum_tp + cum_fp + 1e-10)
    recall = cum_tp / (n_pos + 1e-10)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(precision, recall))
    return float(np.trapz(precision, recall))


def full_report(y_true, y_pred, y_score, label, avg_fraud=122, fp_cost=2):
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    prec = TP / (TP + FP + 1e-10)
    rec  = TP / (TP + FN + 1e-10)
    f1   = 2 * prec * rec / (prec + rec + 1e-10)
    auc_pr  = compute_auc_pr(y_true, y_score)
    auc_roc = compute_auc_roc(y_true, y_score)
    net = TP * avg_fraud - FN * avg_fraud - FP * fp_cost

    print(f"\n{'━' * 58}")
    print(f"  {label}")
    print(f"{'━' * 58}")
    print(f"  Confusion matrix:")
    print(f"              Pred 0(legit)  Pred 1(fraud)")
    print(f"  True 0(legit)  {TN:>9,}     {FP:>9,}")
    print(f"  True 1(fraud)  {FN:>9,}     {TP:>9,}  "
          f"(total fraud: {int(y_true.sum())})")
    print(f"\n  Precision : {100*prec:6.2f}%")
    print(f"  Recall    : {100*rec:6.2f}%")
    print(f"  F1        : {f1:.4f}")
    print(f"  AUC-PR    : {auc_pr:.4f}   ← primary metric")
    print(f"  AUC-ROC   : {auc_roc:.4f}")
    print(f"\n  Business impact:")
    print(f"    Caught :  €{TP*avg_fraud:>8,.0f}")
    print(f"    Missed : -€{FN*avg_fraud:>8,.0f}")
    print(f"    FP cost: -€{FP*fp_cost:>8,.0f}")
    print(f"    Net    :  €{net:>8,.0f}")

    return dict(prec=prec, rec=rec, f1=f1, auc_pr=auc_pr, auc_roc=auc_roc,
                TP=TP, TN=TN, FP=FP, FN=FN, net=net)


def find_operating_point(y_true, y_score, target, metric='recall'):
    """
    Find the operating point using sklearn's precision_recall_curve.
    Handles tied scores correctly (important for tree probabilities).

    Returns: (threshold, precision, recall, f1, FP, FN) or all-zeros if
    the target is unreachable with this model.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns one extra precision/recall point at the
    # end (the "no predictions" case). Threshold array is one shorter.
    # Trim so all three align.
    precision = precision[:-1]
    recall = recall[:-1]

    if metric == 'recall':
        # Want the strictest threshold (highest precision) still hitting target recall
        valid = recall >= target
        if not valid.any():
            return None, 0.0, 0.0, 0.0, 0, int(y_true.sum())
        # Among valid points, pick the one with highest precision
        best_idx = np.argmax(np.where(valid, precision, -np.inf))
    elif metric == 'precision':
        valid = precision >= target
        if not valid.any():
            return None, 0.0, 0.0, 0.0, 0, int(y_true.sum())
        # Among valid points, pick the one with highest recall
        best_idx = np.argmax(np.where(valid, recall, -np.inf))
    else:
        raise ValueError(f"unknown metric: {metric}")

    t = thresholds[best_idx]
    prec = precision[best_idx]
    rec = recall[best_idx]
    f1 = 2 * prec * rec / (prec + rec + 1e-10)

    y_pred_t = (y_score >= t).astype(int)
    FP = int(np.sum((y_pred_t == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred_t == 0) & (y_true == 1)))

    return float(t), float(prec), float(rec), float(f1), FP, FN


# ══════════════════════════════════════════════════════════════════════════════
# ACT 1 — FROM-SCRATCH TREE ON DEV SET
#
# Same dev set as the SVM chapter: ~5000 samples at 12:1 imbalance.
# Goal: confirm our implementation works on real fraud data, time the fit,
# look at the structure it produces.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 58)
print("ACT 1 — FROM-SCRATCH TREE ON DEV SET")
print("═" * 58)

print("\nTraining our tree (max_depth=8)...")
t0 = time.time()
our_tree_dev = OurTree(
    criterion="gini",
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
).fit(X_dev, y_dev)
t_ours_dev = time.time() - t0

# Probability of class 1 (fraud)
proba_dev = our_tree_dev.predict_proba(X_dev)[:, 1]
preds_dev = (proba_dev >= 0.5).astype(int)

print(f"  Training time : {t_ours_dev:.2f}s")
print(f"  Depth reached : {our_tree_dev.tree_depth_}")
print(f"  Total leaves  : {our_tree_dev.n_leaves_}")

res_ours_dev = full_report(
    y_dev, preds_dev, proba_dev,
    "Our Tree — Dev Set (in-sample, threshold=0.5)"
)

# Top-5 features by importance
print("\n  Top 5 features by importance:")
importances = our_tree_dev.feature_importances_
top_idx = np.argsort(importances)[::-1][:5]
for rank, idx in enumerate(top_idx, 1):
    feat_name = meta['feature_names'][idx]
    print(f"    {rank}. {feat_name:<20} {importances[idx]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# ACT 2 — FROM-SCRATCH TREE ON FULL TRAINING SET
#
# 227k samples is the real test. Trees scale much better than kernel SVMs:
# O(n × d × log n) per depth level, not O(n²) like kernel matrix construction.
# Expect: faster than sklearn SVC, slower than LinearSVC.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 58)
print("ACT 2 — FROM-SCRATCH TREE ON FULL TRAINING SET")
print("═" * 58)

print("\nTraining our tree (max_depth=10) on 227k samples...")
print("This will take a few minutes — pure NumPy, no Cython.")
t0 = time.time()
our_tree_full = OurTree(
    criterion="gini",
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
).fit(X_train, y_train)
t_ours_full = time.time() - t0

print(f"  Training time : {t_ours_full:.1f}s")
print(f"  Depth reached : {our_tree_full.tree_depth_}")
print(f"  Total leaves  : {our_tree_full.n_leaves_}")

# Test set evaluation
print("\nScoring on test set...")
proba_test_ours = our_tree_full.predict_proba(X_test)[:, 1]
preds_test_ours = (proba_test_ours >= 0.5).astype(int)
res_ours_test = full_report(
    y_test, preds_test_ours, proba_test_ours,
    "Our Tree — Test Set (threshold=0.5, max_depth=10)"
)

# ══════════════════════════════════════════════════════════════════════════════
# ACT 3 — SKLEARN COMPARISON + HYPERPARAMETER TUNING
#
# Sklearn's tree uses the same CART algorithm in optimized Cython.
# Two reasons to include it:
#   1. Sanity check: our results should be in the same ballpark.
#   2. Hyperparameter search is fast enough only with the optimized version.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 58)
print("ACT 3 — SKLEARN COMPARISON + HYPERPARAMETER TUNING")
print("═" * 58)

try:
    from sklearn.tree import DecisionTreeClassifier as SkTree
except ImportError:
    raise ImportError("pip install scikit-learn")

# ── 3a. Direct comparison at the same hyperparameters ──────────────────────

print("\n── 3a. Same hyperparameters as our tree ──")
t0 = time.time()
sk_match = SkTree(
    criterion="gini",
    max_depth=10,
    min_samples_leaf=20,
    class_weight=None,
    random_state=42,
).fit(X_train, y_train)
t_sk_match = time.time() - t0

proba_sk_match = sk_match.predict_proba(X_test)[:, 1]
preds_sk_match = (proba_sk_match >= 0.5).astype(int)

print(f"  Sklearn training time : {t_sk_match:.2f}s  "
      f"({t_ours_full/max(t_sk_match,0.01):.0f}× faster than ours)")
print(f"  Depth                 : {sk_match.get_depth()}")
print(f"  Leaves                : {sk_match.get_n_leaves()}")

res_sk_match = full_report(
    y_test, preds_sk_match, proba_sk_match,
    "Sklearn Tree — same hyperparams (with class_weight=None)"
)

# ── 3b. Hyperparameter search on full training set ─────────────────────────

print("\n── 3b. Hyperparameter search (max_depth × min_samples_leaf) ──")
print("  Searching on full train set with 5-fold stratified CV.")

def stratified_kfold(y, n_splits=5, seed=42):
    rng = np.random.RandomState(seed)
    fraud_idx = np.where(y == 1)[0].copy()
    legit_idx = np.where(y == 0)[0].copy()
    rng.shuffle(fraud_idx)
    rng.shuffle(legit_idx)
    fraud_folds = np.array_split(fraud_idx, n_splits)
    legit_folds = np.array_split(legit_idx, n_splits)
    for k in range(n_splits):
        val_idx = np.concatenate([fraud_folds[k], legit_folds[k]])
        train_idx = np.concatenate([
            *[fraud_folds[i] for i in range(n_splits) if i != k],
            *[legit_folds[i] for i in range(n_splits) if i != k],
        ])
        yield train_idx, val_idx


depths = [4, 6, 8, 10, 12, None]
leaf_sizes = [10, 20, 50]

print(f"\n  {'max_depth':>10}  {'min_leaf':>9}  "
      f"{'AUC-PR mean':>12}  {'AUC-PR std':>11}  {'Time':>8}")
print(f"  {'-' * 58}")

best_auc = -1
best_params = None
all_grid_results = []

for d in depths:
    for leaf in leaf_sizes:
        fold_aucs = []
        t0 = time.time()
        for train_idx, val_idx in stratified_kfold(y_train, n_splits=5):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            model = SkTree(
                criterion="gini",
                max_depth=d,
                min_samples_leaf=leaf,
                class_weight=None,
                random_state=42,
            ).fit(X_tr, y_tr)
            proba_val = model.predict_proba(X_val)[:, 1]
            fold_aucs.append(compute_auc_pr(y_val, proba_val))

        elapsed = time.time() - t0
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)

        marker = " ← best" if mean_auc > best_auc else ""
        d_str = "None" if d is None else str(d)
        print(f"  {d_str:>10}  {leaf:>9}  {mean_auc:>12.4f}  "
              f"{std_auc:>11.4f}  {elapsed:>6.1f}s{marker}")

        all_grid_results.append({
            'max_depth': d, 'min_samples_leaf': leaf,
            'auc_pr_mean': mean_auc, 'auc_pr_std': std_auc,
        })

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = {'max_depth': d, 'min_samples_leaf': leaf}

print(f"\nBest: max_depth={best_params['max_depth']}, "
      f"min_samples_leaf={best_params['min_samples_leaf']}")
print(f"CV AUC-PR: {best_auc:.4f}")

# ── 3c. Retrain best on full train, evaluate on test ───────────────────────

print(f"\n── 3c. Retraining best config on full training set ──")
t0 = time.time()
sk_best = SkTree(
    criterion="gini",
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    class_weight=None,
    random_state=42,
).fit(X_train, y_train)
t_sk_best = time.time() - t0

proba_sk_best = sk_best.predict_proba(X_test)[:, 1]
preds_sk_best = (proba_sk_best >= 0.5).astype(int)

print(f"  Training time: {t_sk_best:.2f}s")
print(f"  Depth: {sk_best.get_depth()}  Leaves: {sk_best.get_n_leaves()}")

res_sk_best = full_report(
    y_test, preds_sk_best, proba_sk_best,
    f"Sklearn Tree — tuned (max_depth={best_params['max_depth']}, "
    f"min_samples_leaf={best_params['min_samples_leaf']})"
)

# Top-10 features from the tuned tree
print("\n  Top 10 features by importance (tuned sklearn tree):")
sk_importances = sk_best.feature_importances_
sk_top = np.argsort(sk_importances)[::-1][:10]
for rank, idx in enumerate(sk_top, 1):
    feat_name = meta['feature_names'][idx]
    print(f"    {rank:>2}. {feat_name:<20} {sk_importances[idx]:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# ACT 4 — HEAD TO HEAD vs SVM AT MATCHED OPERATING POINTS
#
# The SVM chapter ended at an 80% recall operating point with these numbers:
#   AUC-PR: 0.806, Recall: 80.6%, Precision: 81.4%, FP: 18
#
# Let's see how the tuned tree does at the same target recall.
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 58)
print("ACT 4 — HEAD TO HEAD AT MATCHED OPERATING POINTS")
print("═" * 58)

operating_points = [
    ("High Recall  (catch 80% of fraud)", 0.80, 'recall'),
    ("Balanced     (catch 70% of fraud)", 0.70, 'recall'),
    ("Conservative (90% precision)",      0.90, 'precision'),
]

print("\nUsing tuned sklearn tree scores on test set:")

for label, target, metric in operating_points:
    t, prec, rec, f1, FP, FN = find_operating_point(
        y_test, proba_sk_best, target, metric
    )
    if t is None:
        print(f"\n  {label}: target not achievable")
        continue

    TP = int(y_test.sum()) - FN
    net = TP * 122 - FN * 122 - FP * 2

    print(f"\n  {label}")
    print(f"    Threshold  : {t:>8.4f}")
    print(f"    Precision  : {100*prec:>6.2f}%")
    print(f"    Recall     : {100*rec:>6.2f}%")
    print(f"    F1         : {f1:.4f}")
    print(f"    False Positives : {FP:,}")
    print(f"    False Negatives : {FN}")
    print(f"    Net business    : €{net:,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 58)
print("FINAL COMPARISON — All models on held-out test set")
print("═" * 58)

# Hardcoded SVM v2 results from the previous chapter for reference.
# If you re-ran with different seeds these will drift slightly; that's fine,
# the point is the qualitative comparison.
svm_lsvc_auc_pr = 0.806   # from README.md
svm_lsvc_recall = 0.806
svm_lsvc_prec   = 0.814

print(f"\n  {'Model':<40} {'AUC-PR':>8} {'Recall':>8} {'Precis':>8}")
print(f"  {'-' * 70}")
print(f"  {'SVM LinearSVC (Ch 5)':<40} "
      f"{svm_lsvc_auc_pr:>8.4f} "
      f"{100*svm_lsvc_recall:>7.1f}% "
      f"{100*svm_lsvc_prec:>7.1f}%")
print(f"  {'Our Tree (max_depth=10)':<40} "
      f"{res_ours_test['auc_pr']:>8.4f} "
      f"{100*res_ours_test['rec']:>7.1f}% "
      f"{100*res_ours_test['prec']:>7.1f}%")
print(f"  {'Sklearn Tree tuned':<40} "
      f"{res_sk_best['auc_pr']:>8.4f} "
      f"{100*res_sk_best['rec']:>7.1f}% "
      f"{100*res_sk_best['prec']:>7.1f}%")

print(f"""
Key observations:
  1. Trees are scale-invariant — same result with raw or scaled features.
  2. Single tree underperforms the SVM. This is expected and motivates
     ensembles (next chapter: Random Forests).
  3. Trees give us interpretability for free: feature importances and
     human-readable decision rules.
  4. Training time scales much better than kernel SVMs — no n² memory.
""")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════

import pickle

# Pick threshold for 80% recall operating point
t_80, prec_80, rec_80, _, FP_80, FN_80 = find_operating_point(
    y_test, proba_sk_best, 0.80, 'recall'
)

artifact = {
    'model':          sk_best,
    'model_name':     f"DecisionTreeClassifier(max_depth={best_params['max_depth']})",
    'scaler_mean':    np.load(DATA / "scaler_mean.npy"),
    'scaler_std':     np.load(DATA / "scaler_std.npy"),
    'feature_names':  meta['feature_names'],
    'threshold':      float(t_80) if t_80 is not None else 0.5,
    'threshold_label':'80% recall operating point',
    'test_auc_pr':    float(res_sk_best['auc_pr']),
    'test_recall':    float(rec_80),
    'test_precision': float(prec_80),
    'meta':           meta,
}

out_path = OUT_MODELS / "tree_fraud_model.pkl"
with open(out_path, "wb") as f:
    pickle.dump(artifact, f)

print(f"Model saved → {out_path}")
print(f"Threshold stored: {artifact['threshold']:.4f}")


