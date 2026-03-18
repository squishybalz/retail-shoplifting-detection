"""
Phase 3: Baselines + Full Ablation Table
Trains 4 baseline models on flattened Tier B features, combines with Phase 2 results.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve,
)
from xgboost import XGBClassifier
import re
import warnings
warnings.filterwarnings("ignore")

# -- Paths -------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\malho\Desktop\claudeagent\RETAILPROJECT")
FEATURE_DIR = BASE_DIR / "v10_Features"
RESULTS_DIR = BASE_DIR / "v10_Results"

# -- Load Phase 2 artifacts --------------------------------------------------
print("Loading Phase 2 results...")
with open(RESULTS_DIR / "fold_assignments.pkl", "rb") as f:
    fold_indices = pickle.load(f)

with open(RESULTS_DIR / "all_results.pkl", "rb") as f:
    phase2_results = pickle.load(f)

# -- Feature tier for baselines (Tier B = 72 features) ----------------------
TIER_B_COLS = list(range(0, 72))
PAD_LENGTH = 30  # Pad/truncate to 30 frames, flatten -> 30*72=2160

# -- Load all data -----------------------------------------------------------
def load_all_data() -> list[dict]:
    records = []
    for label in [0, 1]:
        label_dir = FEATURE_DIR / str(label)
        for npy_path in sorted(label_dir.glob("*.npy")):
            fname = npy_path.stem
            clip_name = re.sub(r"_id\d+$", "", fname)
            source_group = clip_name.rsplit("_", 1)[0]
            data = np.load(npy_path)
            records.append({
                "path": str(npy_path),
                "label": label,
                "clip_name": clip_name,
                "source_group": source_group,
                "seq_length": data.shape[0],
                "data": data,
            })
    return records


def flatten_features(records: list[dict], feature_cols: list[int], pad_len: int) -> np.ndarray:
    """Pad each track to pad_len frames, slice feature cols, flatten."""
    n_feat = len(feature_cols)
    X = np.zeros((len(records), pad_len * n_feat), dtype=np.float32)
    for i, rec in enumerate(records):
        seq = rec["data"][:, feature_cols]  # (T, F)
        T = min(seq.shape[0], pad_len)
        padded = np.zeros((pad_len, n_feat), dtype=np.float32)
        padded[:T] = seq[:T]
        X[i] = padded.flatten()
    return X


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """Find threshold that maximizes F1."""
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(best_thresh), float(f1_scores[best_idx])


def aggregate_clip_scores(clip_names: list, scores: np.ndarray, labels: np.ndarray):
    """MAX aggregation at clip level."""
    clip_scores = {}
    clip_labels = {}
    for name, score, label in zip(clip_names, scores, labels):
        if name not in clip_scores:
            clip_scores[name] = []
            clip_labels[name] = label
        clip_scores[name].append(score)
    names = sorted(clip_scores.keys())
    agg_scores = np.array([max(clip_scores[n]) for n in names])
    agg_labels = np.array([clip_labels[n] for n in names])
    return names, agg_scores, agg_labels


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0,
    }


# -- Main --------------------------------------------------------------------
print("Loading data...")
all_records = load_all_data()
print(f"Loaded {len(all_records)} tracks")

labels_all = np.array([r["label"] for r in all_records])
clip_names_all = [r["clip_name"] for r in all_records]

# Store baseline results: {model_name: {fold: metrics}}
baseline_results = {
    "Majority Class": {},
    "Logistic Regression": {},
    "Random Forest": {},
    "XGBoost (flat)": {},
}

for fold_num, (train_idx, val_idx) in enumerate(fold_indices, 1):
    print(f"\n--- Fold {fold_num}/5 ---")
    train_recs = [all_records[i] for i in train_idx]
    val_recs = [all_records[i] for i in val_idx]

    train_labels = np.array([r["label"] for r in train_recs])
    val_labels = np.array([r["label"] for r in val_recs])
    train_clips = [r["clip_name"] for r in train_recs]
    val_clips = [r["clip_name"] for r in val_recs]

    # Flatten features
    X_train = flatten_features(train_recs, TIER_B_COLS, PAD_LENGTH)
    X_val = flatten_features(val_recs, TIER_B_COLS, PAD_LENGTH)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # --- 1. Majority Class ---
    majority_class = int(np.argmax(np.bincount(train_labels)))
    maj_preds = np.full(len(val_labels), majority_class)
    maj_scores = np.full(len(val_labels), float(majority_class))
    # Clip aggregation
    cn, cs, cl = aggregate_clip_scores(val_clips, maj_scores, val_labels)
    cp = np.full(len(cl), majority_class)
    baseline_results["Majority Class"][fold_num] = compute_metrics(cl, cp, cs)
    print(f"  Majority Class: F1={baseline_results['Majority Class'][fold_num]['f1']:.4f}")

    # --- 2. Logistic Regression ---
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, train_labels)
    lr_scores = lr.predict_proba(X_val)[:, 1]
    cn, cs, cl = aggregate_clip_scores(val_clips, lr_scores, val_labels)
    thresh, _ = find_optimal_threshold(cl, cs)
    cp = (cs >= thresh).astype(int)
    baseline_results["Logistic Regression"][fold_num] = compute_metrics(cl, cp, cs)
    print(f"  Logistic Regression: F1={baseline_results['Logistic Regression'][fold_num]['f1']:.4f} (thresh={thresh:.3f})")

    # --- 3. Random Forest ---
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf.fit(X_train, train_labels)
    rf_scores = rf.predict_proba(X_val)[:, 1]
    cn, cs, cl = aggregate_clip_scores(val_clips, rf_scores, val_labels)
    thresh, _ = find_optimal_threshold(cl, cs)
    cp = (cs >= thresh).astype(int)
    baseline_results["Random Forest"][fold_num] = compute_metrics(cl, cp, cs)
    print(f"  Random Forest: F1={baseline_results['Random Forest'][fold_num]['f1']:.4f} (thresh={thresh:.3f})")

    # --- 4. XGBoost Direct ---
    n0 = np.sum(train_labels == 0)
    n1 = np.sum(train_labels == 1)
    xgb = XGBClassifier(
        n_estimators=150, max_depth=5, scale_pos_weight=n0 / max(n1, 1),
        eval_metric="logloss", verbosity=0, random_state=42,
    )
    xgb.fit(X_train, train_labels)
    xgb_scores = xgb.predict_proba(X_val)[:, 1]
    cn, cs, cl = aggregate_clip_scores(val_clips, xgb_scores, val_labels)
    thresh, _ = find_optimal_threshold(cl, cs)
    cp = (cs >= thresh).astype(int)
    baseline_results["XGBoost (flat)"][fold_num] = compute_metrics(cl, cp, cs)
    print(f"  XGBoost (flat): F1={baseline_results['XGBoost (flat)'][fold_num]['f1']:.4f} (thresh={thresh:.3f})")


# -- Build ablation table ---------------------------------------------------
print("\n" + "=" * 60)
print("BUILDING ABLATION TABLE")
print("=" * 60)

TIER_NAMES = {
    "A_base": ("A: 36 feat", 36),
    "B_velocity": ("B: 72 feat", 72),
    "C_spatial": ("C: 77 feat", 77),
    "D_full": ("D: 85 feat", 85),
}

rows = []

# Baseline rows (1-4)
for model_name in ["Majority Class", "Logistic Regression", "Random Forest", "XGBoost (flat)"]:
    fold_metrics = baseline_results[model_name]
    metrics_agg = {}
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        vals = [fold_metrics[f][metric] for f in fold_metrics]
        metrics_agg[metric] = f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"
    rows.append({
        "Model": model_name,
        "Features": 72,
        **metrics_agg,
    })

# LSTM and Hybrid rows (5-12)
for tier_key, (tier_label, n_feat) in TIER_NAMES.items():
    tier_res = phase2_results[tier_key]

    for model_type, metrics_key in [("LSTM Only", "lstm_metrics"), ("LSTM+XGB", "hybrid_metrics")]:
        metrics_agg = {}
        for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
            vals = [tier_res[f][metrics_key][metric] for f in tier_res]
            metrics_agg[metric] = f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"
        rows.append({
            "Model": f"{model_type} (Tier {tier_label})",
            "Features": n_feat,
            **metrics_agg,
        })

# Create DataFrame
df = pd.DataFrame(rows)
df.index = range(1, len(df) + 1)
df.index.name = "#"

# Save
df.to_csv(RESULTS_DIR / "ablation_table.csv")

# Save baseline results for Phase 4
with open(RESULTS_DIR / "baseline_results.pkl", "wb") as f:
    pickle.dump(baseline_results, f)

# Print table
print("\n" + df.to_string())

# -- Interpretation ----------------------------------------------------------
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

# Extract mean F1 for comparison
def get_mean_f1(tier_key: str, model_type: str) -> float:
    metrics_key = "lstm_metrics" if model_type == "lstm" else "hybrid_metrics"
    vals = [phase2_results[tier_key][f][metrics_key]["f1"] for f in phase2_results[tier_key]]
    return np.mean(vals)

def get_baseline_mean_f1(model_name: str) -> float:
    vals = [baseline_results[model_name][f]["f1"] for f in baseline_results[model_name]]
    return np.mean(vals)

best_baseline_f1 = max(get_baseline_mean_f1(m) for m in baseline_results)
best_baseline_name = max(baseline_results.keys(), key=lambda m: get_baseline_mean_f1(m))

tier_a_lstm = get_mean_f1("A_base", "lstm")
tier_b_lstm = get_mean_f1("B_velocity", "lstm")
tier_c_lstm = get_mean_f1("C_spatial", "lstm")
tier_d_lstm = get_mean_f1("D_full", "lstm")

tier_a_hyb = get_mean_f1("A_base", "hybrid")
tier_b_hyb = get_mean_f1("B_velocity", "hybrid")
tier_c_hyb = get_mean_f1("C_spatial", "hybrid")
tier_d_hyb = get_mean_f1("D_full", "hybrid")

all_f1s = {
    "A_lstm": tier_a_lstm, "A_hyb": tier_a_hyb,
    "B_lstm": tier_b_lstm, "B_hyb": tier_b_hyb,
    "C_lstm": tier_c_lstm, "C_hyb": tier_c_hyb,
    "D_lstm": tier_d_lstm, "D_hyb": tier_d_hyb,
}
best_overall = max(all_f1s, key=all_f1s.get)
best_f1 = all_f1s[best_overall]

print(f"\n1. Best performing model: {best_overall} with F1={best_f1:.4f}")
print(f"2. Best baseline: {best_baseline_name} with F1={best_baseline_f1:.4f}")
print(f"   -> LSTM/Hybrid beats best baseline by {(best_f1 - best_baseline_f1)*100:.1f}% F1")
print(f"3. Velocity features (B vs A): LSTM {tier_b_lstm:.4f} vs {tier_a_lstm:.4f} "
      f"({'helped' if tier_b_lstm > tier_a_lstm else 'did not help'})")
print(f"4. Spatial features (C vs B): LSTM {tier_c_lstm:.4f} vs {tier_b_lstm:.4f} "
      f"({'helped' if tier_c_lstm > tier_b_lstm else 'did not help'})")
print(f"5. Full features (D vs C): LSTM {tier_d_lstm:.4f} vs {tier_c_lstm:.4f} "
      f"({'overfitting likely' if tier_d_lstm < tier_c_lstm else 'still helping'})")
print(f"6. Hybrid vs LSTM-only (best tier): "
      f"{'Hybrid better' if max(tier_a_hyb, tier_b_hyb, tier_c_hyb, tier_d_hyb) > max(tier_a_lstm, tier_b_lstm, tier_c_lstm, tier_d_lstm) else 'LSTM-only better or comparable'}")

print("\nDone.")
