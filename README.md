# Skeleton-Based Shoplifting Detection

Privacy-preserving shoplifting detection from CCTV footage using skeleton pose estimation, LSTM with temporal attention, and ensemble methods.

**Final Results: F1 = 0.895 | AUC = 0.919** (+127% over majority baseline) **| 
                Mean Accuracy : 0.890 | Recall : 0.892 | Precision : 0.905**

---

## Key Highlights

- **Privacy-preserving** — only skeleton keypoints are stored, no appearance data
- **Full ML pipeline** — raw video to classification in one automated flow
- **Systematic experimentation** — 4 rounds, 25+ configurations, keep-or-revert methodology
- **Rigorous evaluation** — 5-fold GroupKFold cross-validation with strict anti-leakage measures

## Pipeline

```
Raw Video (.mp4)
    → YOLO26s-pose (17 keypoints per person per frame)
    → ByteTrack (multi-person tracking)
    → 85-dim skeleton features (4 hierarchical tiers)
    → Bidirectional LSTM with Temporal Attention
    → Weighted Ensemble (LSTM + Random Forest)
    → Clip-level prediction
```

## Results Summary

| Milestone | F1 | AUC |
|-----------|------|------|
| Majority Class Baseline | 0.393 | 0.500 |
| Best Flat Baseline (RF) | 0.809 | 0.853 |
| V10 Best (Tier C LSTM) | 0.819 | 0.813 |
| V11 R2 (Architecture + Training) | 0.834 | 0.847 |
| V11 R3 Best (Feature Engineering) | 0.868 | 0.875 |
| **V11 R4 Final Ensemble** | **0.895** | **0.919** |

## Project Structure

```
├── RETAIL_FINAL.ipynb          ← Portfolio notebook (start here)
├── src/
│   ├── extract_features.py     ← YOLO pose → skeleton features
│   ├── extract_features_v11.py ← Extended 158-dim features
│   ├── train.py                ← V10 LSTM + XGBoost training
│   ├── train_v11.py            ← V11 with attention, R1-R4 experiments
│   ├── train_v11_r3.py         ← R3 feature engineering experiments
│   ├── ensemble.py             ← R4 ensemble methods
│   ├── baselines.py            ← Baseline model comparisons
│   ├── visualize.py            ← Plot generation
│   ├── build_notebook_final.py ← Generates RETAIL_FINAL.ipynb
│   └── run_all.py              ← Run full pipeline
├── data/
│   ├── Class_0_Normal/         ← 179 normal behavior clips
│   └── Class_1_Shoplifting/    ← 179 shoplifting clips
├── v10_Features/               ← Extracted skeleton features (85-dim .npy)
├── v11_Features/               ← Extended features (158-dim .npy)
├── v10_Results/                ← Models, metrics, plots, experiment logs
└── CLAUDE.md                   ← Detailed project specification
```

## Experiment Rounds

| Round | Focus | Key Finding |
|-------|-------|-------------|
| R1 | Architecture | Temporal attention > last-hidden-state (+1.4% F1) |
| R2 | Training | F1-based early stopping > loss-based (+2.7% F1) |
| R3 | Features | Each new feature helped alone, but all combined hurt (curse of dimensionality) |
| R4 | Ensemble | LSTM + RF weighted ensemble = biggest single jump (+3.1% F1, lowest variance) |

## Feature Tiers (Ablation)

| Tier | Dims | Contents | Result |
|------|------|----------|--------|
| A (Base) | 36 | Keypoint coords + elbow angles | Baseline |
| B (+Velocity) | 72 | + frame-to-frame deltas | Modest gain |
| C (+Spatial) | 77 | + wrist-hip distance, cross-body reach | **Best tier** |
| D (+Advanced) | 85 | + trunk angle, head orientation | Overfits |

## Tech Stack

- **Pose Estimation:** YOLO26s-pose (ultralytics)
- **Tracking:** ByteTrack
- **Model:** PyTorch (bidirectional LSTM + attention)
- **Ensemble:** scikit-learn Random Forest
- **Evaluation:** scikit-learn, XGBoost
- **Visualization:** matplotlib, seaborn

## Reproducing Results

```bash
# 1. Install dependencies
pip install torch ultralytics numpy pandas scikit-learn xgboost matplotlib seaborn

# 2. Extract features from video (requires GPU, ~30-60 min)
python src/extract_features.py

# 3. Train models and run experiments
python src/run_all.py

# 4. Generate the final notebook
python src/build_notebook_final.py
```

The YOLO model (`yolo26s-pose.pt`) downloads automatically on first run.

## Dataset

358 balanced video clips (179 normal, 179 shoplifting) from public CCTV footage and staged scenarios. Each clip is 5-15 seconds, yielding skeleton tracks of 15-30 frames per tracked person.

**Anti-leakage:** GroupKFold by source video ensures all fragments from the same scene stay in the same cross-validation fold.
