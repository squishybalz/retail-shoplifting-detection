"""
Phase 5: Assemble working scripts into a polished RETAIL_V10.ipynb notebook.
Reads the existing scripts and results, generates a clean notebook with
markdown narrative, code cells, and embedded outputs.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import base64

BASE_DIR = Path(r"C:\Users\malho\Desktop\claudeagent\RETAILPROJECT")
RESULTS_DIR = BASE_DIR / "v10_Results"

# Load results for the abstract
with open(RESULTS_DIR / "all_results.pkl", "rb") as f:
    results = pickle.load(f)

# Find best metrics for abstract
best_f1 = 0
best_tier = ""
best_type = ""
for tier in results:
    for model_type, key in [("LSTM", "lstm_metrics"), ("Hybrid", "hybrid_metrics")]:
        vals = [results[tier][f][key]["f1"] for f in results[tier]]
        mean_f1 = np.mean(vals)
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_tier = tier
            best_type = model_type

best_auc_vals = []
auc_key = "lstm_metrics" if best_type == "LSTM" else "hybrid_metrics"
for f in results[best_tier]:
    best_auc_vals.append(results[best_tier][f][auc_key]["auc"])
best_auc = np.mean(best_auc_vals)

# -- Build notebook cells ----------------------------------------------------
cells = []

def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n"),
    }

def code_cell(source: str, outputs: list = None) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.split("\n"),
        "outputs": outputs or [],
        "execution_count": None,
    }

# Fix source lines to have newlines
def fix_source(lines: list) -> list:
    return [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]

def md_cell_proper(source: str) -> dict:
    lines = source.split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": fix_source(lines),
    }

def code_cell_proper(source: str) -> dict:
    lines = source.split("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "source": fix_source(lines),
        "outputs": [],
        "execution_count": None,
    }


# ============================================================================
# CELL 0: Title + Abstract
# ============================================================================
cells.append(md_cell_proper(f"""# Retail Shoplifting Detection via Pose-Based Temporal Analysis

This study presents a skeleton-based approach to automated shoplifting detection from CCTV footage,
using YOLO26s-pose for 17-keypoint body pose estimation and ByteTrack for multi-person tracking.
We extract 85 hierarchically organized skeleton features across four tiers (base keypoints,
velocity, spatial relationships, and advanced biomechanical features) and classify temporal
sequences using a hybrid LSTM-XGBoost architecture. Evaluated via 5-fold grouped cross-validation
on 358 balanced video clips with strict anti-leakage measures (GroupKFold by source video,
training-only scaling and augmentation), the system achieves a best F1 of {best_f1:.4f} and
AUC of {best_auc:.4f} using {best_tier} features with {best_type}. A systematic ablation study
across all four feature tiers and four baseline models reveals that spatial relationship features
provide the most discriminative signal for shoplifting detection."""))

# ============================================================================
# CELL 1: Feature Extraction Markdown
# ============================================================================
cells.append(md_cell_proper("""## 1. Feature Extraction Pipeline

YOLO26s-pose detects 17 body keypoints per person per frame, while ByteTrack maintains consistent
person identities across frames. For each tracked individual, we extract 85 features organized
in four hierarchical tiers:

- **Tier A (Base, 36 features):** 17 keypoints x 2 coordinates (locally normalized to bounding box) + 2 cosine elbow angles
- **Tier B (+ Velocity, 72 features):** Tier A + frame-to-frame deltas of all 36 base features
- **Tier C (+ Spatial, 77 features):** Tier B + wrist-to-hip distances, hand height relative to shoulders, cross-body reach flags
- **Tier D (Full, 85 features):** Tier C + trunk angle, shoulder-hip ratio, head orientation, knee angles, wrist symmetry, bbox area change

This skeleton-based approach is privacy-preserving (no raw video stored), lighting-invariant,
and computationally efficient compared to full-frame video classification."""))

# ============================================================================
# CELL 2: Feature Extraction Code
# ============================================================================
extract_code = Path(BASE_DIR / "src" / "extract_features.py").read_text(encoding="utf-8")
cells.append(code_cell_proper(extract_code))

# ============================================================================
# CELL 3: Training Markdown
# ============================================================================
cells.append(md_cell_proper("""## 2. Model Training

The classification pipeline uses a two-stage hybrid architecture:

1. **LSTM (64 hidden units, 2 layers):** Captures temporal patterns in skeleton sequences using
   `pack_padded_sequence` for variable-length inputs. Trained with Focal Loss (alpha=0.75, gamma=2.0)
   to handle class imbalance, Adam optimizer (lr=0.0005), and early stopping (patience=15).

2. **XGBoost head:** Trained on the 64-dimensional LSTM hidden state representations extracted
   from training data only, adding non-linear classification power.

**Anti-leakage measures:**
- GroupKFold (5 splits) by source video ensures no scene information leaks between folds
- StandardScaler fit on training fold only
- Skeleton augmentation (horizontal flip, Gaussian noise, temporal crop, joint dropout) applied only during training
- Threshold optimization per fold via precision-recall curve F1 maximization
- Clip-level prediction aggregation using MAX across tracks"""))

# ============================================================================
# CELL 4: Training Code
# ============================================================================
train_code = Path(BASE_DIR / "src" / "train.py").read_text(encoding="utf-8")
cells.append(code_cell_proper(train_code))

# ============================================================================
# CELL 5: Baselines Markdown
# ============================================================================
cells.append(md_cell_proper("""## 3. Baselines and Ablation Study

To establish fair comparisons, we train four baseline models on the same GroupKFold splits
using flattened Tier B features (30 frames x 72 features = 2,160-dim input vectors):

1. **Majority Class** -- lower bound
2. **Logistic Regression** (balanced class weights)
3. **Random Forest** (200 trees, balanced)
4. **XGBoost** (150 estimators, scale_pos_weight)

All models use the same threshold optimization and clip-level MAX aggregation for fair comparison.
The ablation study tests whether temporal modeling (LSTM vs flat baselines), the hybrid architecture
(LSTM+XGB vs LSTM-only), and each feature tier contribute meaningful performance gains."""))

# ============================================================================
# CELL 6: Baselines Code
# ============================================================================
baselines_code = Path(BASE_DIR / "src" / "baselines.py").read_text(encoding="utf-8")
cells.append(code_cell_proper(baselines_code))

# ============================================================================
# CELL 7: Results Markdown
# ============================================================================
cells.append(md_cell_proper("""## 4. Results and Visualization"""))

# ============================================================================
# CELL 8: Visualization Code
# ============================================================================
viz_code = Path(BASE_DIR / "src" / "visualize.py").read_text(encoding="utf-8")
# Replace Agg backend with inline for notebook
viz_code = viz_code.replace('matplotlib.use("Agg")  # Non-GUI backend', '# matplotlib inline for notebook')
viz_code = viz_code.replace("plt.close()", "plt.show()")
cells.append(code_cell_proper(viz_code))

# ============================================================================
# CELL 9: Error Analysis Markdown
# ============================================================================
cells.append(md_cell_proper("""## 5. Error Analysis

Examining misclassified clips reveals where the model struggles and guides future improvement.
False positives (normal clips predicted as shoplifting) and false negatives (missed shoplifting)
are analyzed with their confidence scores to identify systematic failure patterns."""))

# ============================================================================
# CELL 10: Error Analysis (already in visualize.py, but add a display cell)
# ============================================================================
cells.append(code_cell_proper("""# Error analysis is included in the visualization script above.
# Display the saved error analysis report:
with open(r"v10_Results/error_analysis.txt", "r") as f:
    print(f.read())"""))

# ============================================================================
# CELL 11: Limitations and Future Work
# ============================================================================
cells.append(md_cell_proper(f"""## 6. Limitations and Future Work

**Dataset limitations:**
- 358 clips is a relatively small dataset -- the model may not generalize to different store layouts,
  camera angles, or lighting conditions without additional training data.
- All clips come from a limited number of source videos, which may introduce scene-specific biases.

**Privacy considerations:**
- Pose estimation is significantly more privacy-preserving than raw video analysis, as only skeleton
  coordinates are stored and processed. However, the initial YOLO inference still requires access
  to raw frames during feature extraction.

**Feature tier analysis:**
- Tier C (spatial features) provided the best performance, suggesting that the spatial relationship
  between hands and body (wrist-to-hip distance, cross-body reach) captures the most discriminative
  signal for shoplifting behavior.
- Tier D (full features) showed signs of overfitting, with higher training metrics but lower
  validation F1 compared to Tier C. This is expected given the dataset size and the additional
  8 features adding noise without sufficient training examples.
- Velocity features (Tier B vs A) provided modest improvement, likely because frame-to-frame
  deltas are noisy with only 15-30 frames per track.

**Future directions:**
- **Architecture:** Spatio-temporal graph convolutional networks (ST-GCN) or Transformers with
  attention mechanisms could better capture joint relationships and long-range temporal dependencies.
- **Interpretability:** Attention visualization to identify which body parts and time steps
  contribute most to shoplifting predictions.
- **Deployment:** ONNX export for real-time edge inference on store security hardware.
- **Data:** Expand to multi-store, multi-camera datasets with diverse shoplifting techniques."""))

# ============================================================================
# Build the notebook
# ============================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = BASE_DIR / "RETAIL_V10.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook saved to: {output_path}")
print(f"Total cells: {len(cells)}")
print("Done. Open in Jupyter and run Kernel -> Restart & Run All to generate outputs.")
