"""
Phase 4: Visualizations + Error Analysis
Generates 7 plots and error analysis report.
"""

import pickle
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

# -- Paths -------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\malho\Desktop\claudeagent\RETAILPROJECT")
FEATURE_DIR = BASE_DIR / "v10_Features"
RESULTS_DIR = BASE_DIR / "v10_Results"

plt.style.use("seaborn-v0_8-whitegrid")

# -- Load data ---------------------------------------------------------------
print("Loading results...")
with open(RESULTS_DIR / "all_results.pkl", "rb") as f:
    phase2_results = pickle.load(f)

with open(RESULTS_DIR / "baseline_results.pkl", "rb") as f:
    baseline_results = pickle.load(f)

ablation_df = pd.read_csv(RESULTS_DIR / "ablation_table.csv", index_col=0)

# -- Load feature files for data overview ------------------------------------
def load_feature_stats():
    lengths = {0: [], 1: []}
    counts = {0: 0, 1: 0}
    for label in [0, 1]:
        label_dir = FEATURE_DIR / str(label)
        for npy_path in label_dir.glob("*.npy"):
            data = np.load(npy_path)
            lengths[label].append(data.shape[0])
            counts[label] += 1
    return counts, lengths

counts, lengths = load_feature_stats()

# ============================================================================
# PLOT 1: Data Overview
# ============================================================================
print("Plot 1: Data Overview...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Left: class distribution
classes = ["Normal", "Shoplifting"]
class_counts = [counts[0], counts[1]]
bars = ax1.bar(classes, class_counts, color=["#4CAF50", "#F44336"], edgecolor="black")
for bar, count in zip(bars, class_counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             str(count), ha="center", fontweight="bold")
ax1.set_ylabel("Number of Tracks")
ax1.set_title("Class Distribution")

# Right: sequence length histogram
ax2.hist(lengths[0], bins=15, alpha=0.6, label="Normal", color="#4CAF50", edgecolor="black")
ax2.hist(lengths[1], bins=15, alpha=0.6, label="Shoplifting", color="#F44336", edgecolor="black")
ax2.set_xlabel("Sequence Length (frames)")
ax2.set_ylabel("Count")
ax2.set_title("Sequence Length Distribution")
ax2.legend()

plt.tight_layout()
plt.savefig(RESULTS_DIR / "plot1_data_overview.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# PLOT 2: Feature Tier Comparison (KEY PLOT)
# ============================================================================
print("Plot 2: Feature Tier Comparison...")
fig, ax = plt.subplots(figsize=(10, 5))

tier_names = ["A_base", "B_velocity", "C_spatial", "D_full"]
tier_labels = ["Tier A\n(36 feat)", "Tier B\n(72 feat)", "Tier C\n(77 feat)", "Tier D\n(85 feat)"]

lstm_f1_means = []
lstm_f1_stds = []
hyb_f1_means = []
hyb_f1_stds = []

for tier in tier_names:
    lstm_vals = [phase2_results[tier][f]["lstm_metrics"]["f1"] for f in phase2_results[tier]]
    hyb_vals = [phase2_results[tier][f]["hybrid_metrics"]["f1"] for f in phase2_results[tier]]
    lstm_f1_means.append(np.mean(lstm_vals))
    lstm_f1_stds.append(np.std(lstm_vals))
    hyb_f1_means.append(np.mean(hyb_vals))
    hyb_f1_stds.append(np.std(hyb_vals))

x = np.arange(len(tier_labels))
width = 0.35

bars1 = ax.bar(x - width / 2, lstm_f1_means, width, yerr=lstm_f1_stds,
               label="LSTM Only", color="#2196F3", capsize=5, edgecolor="black")
bars2 = ax.bar(x + width / 2, hyb_f1_means, width, yerr=hyb_f1_stds,
               label="LSTM+XGB Hybrid", color="#4CAF50", capsize=5, edgecolor="black")

ax.set_ylabel("F1 Score")
ax.set_title("Feature Tier Comparison: Which Features Matter?")
ax.set_xticks(x)
ax.set_xticklabels(tier_labels)
ax.legend()
ax.set_ylim(0.5, 1.0)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                f"{height:.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "plot2_tier_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# PLOT 3: Full Model Comparison
# ============================================================================
print("Plot 3: Full Model Comparison...")
fig, ax = plt.subplots(figsize=(12, 5))

# Parse F1 means from ablation table
model_names = ablation_df["Model"].tolist()
f1_means = []
f1_stds = []
for f1_str in ablation_df["f1"].tolist():
    parts = f1_str.split(" +/- ")
    f1_means.append(float(parts[0]))
    f1_stds.append(float(parts[1]))

# Sort descending
sorted_idx = np.argsort(f1_means)[::-1]
sorted_names = [model_names[i] for i in sorted_idx]
sorted_means = [f1_means[i] for i in sorted_idx]
sorted_stds = [f1_stds[i] for i in sorted_idx]

# Color by type
colors = []
for name in sorted_names:
    if "LSTM+XGB" in name:
        colors.append("#4CAF50")
    elif "LSTM" in name:
        colors.append("#2196F3")
    else:
        colors.append("#9E9E9E")

bars = ax.barh(range(len(sorted_names)), sorted_means, xerr=sorted_stds,
               color=colors, capsize=3, edgecolor="black")
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels(sorted_names, fontsize=8)
ax.set_xlabel("F1 Score")
ax.set_title("All Models Ranked by F1 Score")
ax.invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#9E9E9E", edgecolor="black", label="Baseline"),
    Patch(facecolor="#2196F3", edgecolor="black", label="LSTM Only"),
    Patch(facecolor="#4CAF50", edgecolor="black", label="LSTM+XGB Hybrid"),
]
ax.legend(handles=legend_elements, loc="lower right")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "plot3_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# PLOT 4: ROC Curves
# ============================================================================
print("Plot 4: ROC Curves...")
fig, ax = plt.subplots(figsize=(8, 8))

# Find best fold for Tier B hybrid (by AUC)
tier_b = phase2_results["B_velocity"]
best_fold_b = max(tier_b.keys(), key=lambda k: tier_b[k]["hybrid_metrics"]["auc"])

# Find best overall tier by mean hybrid AUC
tier_aucs = {}
for tier in tier_names:
    vals = [phase2_results[tier][f]["hybrid_metrics"]["auc"] for f in phase2_results[tier]]
    tier_aucs[tier] = np.mean(vals)
best_tier = max(tier_aucs, key=tier_aucs.get)

# Plot ROC curves
curves_to_plot = [
    ("B_velocity", best_fold_b, "LSTM Only (Tier B)", "lstm", "#2196F3", "-"),
    ("B_velocity", best_fold_b, "Hybrid (Tier B)", "hybrid", "#4CAF50", "-"),
]
if best_tier != "B_velocity":
    best_fold_other = max(phase2_results[best_tier].keys(),
                          key=lambda k: phase2_results[best_tier][k]["hybrid_metrics"]["auc"])
    curves_to_plot.append(
        (best_tier, best_fold_other, f"Hybrid ({best_tier})", "hybrid", "#FF9800", "--")
    )

for tier, fold, label, model_type, color, ls in curves_to_plot:
    r = phase2_results[tier][fold]
    scores_key = "clip_scores_lstm" if model_type == "lstm" else "clip_scores_xgb"
    y_true = np.array(r["clip_labels"])
    y_scores = np.array(r[scores_key])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linestyle=ls, lw=2, label=f"{label} (AUC={roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(RESULTS_DIR / "plot4_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# PLOT 5: Precision-Recall Curve
# ============================================================================
print("Plot 5: Precision-Recall Curve...")
fig, ax = plt.subplots(figsize=(8, 6))

# Use best hybrid model (best tier, best fold by F1)
best_tier_f1 = max(tier_names, key=lambda t: np.mean(
    [phase2_results[t][f]["hybrid_metrics"]["f1"] for f in phase2_results[t]]))
best_fold_pr = max(phase2_results[best_tier_f1].keys(),
                   key=lambda k: phase2_results[best_tier_f1][k]["hybrid_metrics"]["f1"])

r = phase2_results[best_tier_f1][best_fold_pr]
y_true = np.array(r["clip_labels"])
y_scores = np.array(r["clip_scores_xgb"])

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_f1 = f1_scores[best_idx]
best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
best_prec = precision[best_idx]
best_rec = recall[best_idx]

ax.plot(recall, precision, color="#4CAF50", lw=2)
ax.scatter([best_rec], [best_prec], color="red", s=100, zorder=5)
ax.annotate(f"Best F1={best_f1:.3f}\nat threshold={best_thresh:.3f}",
            xy=(best_rec, best_prec),
            xytext=(best_rec - 0.2, best_prec - 0.1),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10, color="red", fontweight="bold")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"Precision-Recall Curve (Best Hybrid: {best_tier_f1}, Fold {best_fold_pr})")
ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig(RESULTS_DIR / "plot5_pr_curve.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# PLOT 6: Confusion Matrix
# ============================================================================
print("Plot 6: Confusion Matrix...")
fig, ax = plt.subplots(figsize=(6, 5))

# Aggregate predictions across all folds for best tier hybrid
all_true = []
all_pred = []
for fold in phase2_results[best_tier_f1]:
    r = phase2_results[best_tier_f1][fold]
    for cr in r["clip_results"]:
        all_true.append(cr["true_label"])
        all_pred.append(cr["hybrid_pred"])

cm = confusion_matrix(all_true, all_pred)
total = cm.sum()

# Annotate with counts and percentages
annot = np.empty_like(cm, dtype=object)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f"{cm[i, j]}\n({cm[i, j] / total * 100:.1f}%)"

sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax,
            xticklabels=["Normal", "Shoplifting"],
            yticklabels=["Normal", "Shoplifting"])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (Hybrid {best_tier_f1}, All Folds)")

plt.tight_layout()
plt.savefig(RESULTS_DIR / "plot6_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# PLOT 7: Per-Fold Stability
# ============================================================================
print("Plot 7: Per-Fold Stability...")
fig, ax = plt.subplots(figsize=(8, 4))

fold_f1s = [phase2_results[best_tier_f1][f]["hybrid_metrics"]["f1"]
            for f in sorted(phase2_results[best_tier_f1].keys())]
folds = list(range(1, len(fold_f1s) + 1))
mean_f1 = np.mean(fold_f1s)

ax.bar(folds, fold_f1s, color="#4CAF50", edgecolor="black", alpha=0.8)
ax.axhline(y=mean_f1, color="red", linestyle="--", lw=2, label=f"Mean F1={mean_f1:.4f}")
for i, f1 in enumerate(fold_f1s):
    ax.text(folds[i], f1 + 0.005, f"{f1:.3f}", ha="center", fontsize=9)

ax.set_xlabel("Fold")
ax.set_ylabel("F1 Score")
ax.set_title(f"Per-Fold Stability (Hybrid {best_tier_f1})")
ax.set_xticks(folds)
ax.legend()
ax.set_ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "plot7_fold_stability.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================================
# ERROR ANALYSIS
# ============================================================================
print("\nError Analysis...")

lines = []
lines.append("=" * 60)
lines.append("ERROR ANALYSIS")
lines.append(f"Model: Hybrid {best_tier_f1}")
lines.append("=" * 60)

all_fp = []
all_fn = []

for fold in sorted(phase2_results[best_tier_f1].keys()):
    r = phase2_results[best_tier_f1][fold]
    for cr in r["clip_results"]:
        if cr["true_label"] == 0 and cr["hybrid_pred"] == 1:
            all_fp.append({"clip": cr["clip_name"], "score": cr["hybrid_score"], "fold": fold})
        elif cr["true_label"] == 1 and cr["hybrid_pred"] == 0:
            all_fn.append({"clip": cr["clip_name"], "score": cr["hybrid_score"], "fold": fold})

lines.append("\n=== FALSE POSITIVES (Normal -> predicted Shoplifting) ===")
for fp in sorted(all_fp, key=lambda x: -x["score"]):
    lines.append(f"  {fp['clip']} | score={fp['score']:.4f} | fold {fp['fold']}")

lines.append(f"\n=== FALSE NEGATIVES (Shoplifting -> predicted Normal) ===")
for fn in sorted(all_fn, key=lambda x: x["score"]):
    lines.append(f"  {fn['clip']} | score={fn['score']:.4f} | fold {fn['fold']}")

# Count unique clips
n_normal_clips = len(set(cr["clip_name"] for fold in phase2_results[best_tier_f1]
                         for cr in phase2_results[best_tier_f1][fold]["clip_results"]
                         if cr["true_label"] == 0))
n_shop_clips = len(set(cr["clip_name"] for fold in phase2_results[best_tier_f1]
                        for cr in phase2_results[best_tier_f1][fold]["clip_results"]
                        if cr["true_label"] == 1))

fp_unique = len(set(fp["clip"] for fp in all_fp))
fn_unique = len(set(fn["clip"] for fn in all_fn))

lines.append(f"\nSummary:")
lines.append(f"  FP: {fp_unique} unique normal clips misclassified across folds")
lines.append(f"  FN: {fn_unique} unique shoplifting clips misclassified across folds")

# Repeat offenders
from collections import Counter
fp_counter = Counter(fp["clip"] for fp in all_fp)
fn_counter = Counter(fn["clip"] for fn in all_fn)
repeat_fp = [clip for clip, count in fp_counter.items() if count >= 2]
repeat_fn = [clip for clip, count in fn_counter.items() if count >= 2]
lines.append(f"  Repeat FP offenders (misclassified 2+ times): {repeat_fp if repeat_fp else 'None'}")
lines.append(f"  Repeat FN offenders (misclassified 2+ times): {repeat_fn if repeat_fn else 'None'}")

error_text = "\n".join(lines)
print(error_text)

with open(RESULTS_DIR / "error_analysis.txt", "w") as f:
    f.write(error_text)

print("\nAll plots saved to v10_Results/")
print("Error analysis saved to v10_Results/error_analysis.txt")
print("Done.")
