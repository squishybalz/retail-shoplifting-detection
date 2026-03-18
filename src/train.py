"""
Phase 2: Training Pipeline (Zero Leakage)
LSTM (64 hidden) -> XGBoost hybrid, 5-fold GroupKFold, 4 feature tiers
"""

import os
import re
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve,
)
from xgboost import XGBClassifier
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# -- Paths -------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\malho\Desktop\claudeagent\RETAILPROJECT")
FEATURE_DIR = BASE_DIR / "v10_Features"
RESULTS_DIR = BASE_DIR / "v10_Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -- Feature tier definitions ------------------------------------------------
FEATURE_TIERS = {
    "A_base":     list(range(0, 36)),
    "B_velocity": list(range(0, 72)),
    "C_spatial":  list(range(0, 77)),
    "D_full":     list(range(0, 85)),
}

# -- Data loading ------------------------------------------------------------
def load_all_data() -> list[dict]:
    """Scan all .npy files and return metadata list."""
    records = []
    for label in [0, 1]:
        label_dir = FEATURE_DIR / str(label)
        for npy_path in sorted(label_dir.glob("*.npy")):
            fname = npy_path.stem  # e.g. Normal_001_f1_id1
            # Remove _id{N} suffix to get clip_name
            clip_name = re.sub(r"_id\d+$", "", fname)
            # source_group: drop the last _part after rsplit
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


# -- Dataset -----------------------------------------------------------------
class SkeletonDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        feature_cols: list[int],
        scaler: StandardScaler,
        augment: bool = False,
    ):
        self.records = records
        self.feature_cols = feature_cols
        self.augment = augment
        self.scaler = scaler
        self.n_features = len(feature_cols)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        seq = rec["data"][:, self.feature_cols].copy()  # (T, F)

        # Scale
        seq = self.scaler.transform(seq)

        if self.augment:
            seq = self._augment(seq)

        label = rec["label"]
        return (
            torch.FloatTensor(seq),
            torch.FloatTensor([label]),
            seq.shape[0],
            rec["clip_name"],
        )

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        seq = seq.copy()
        n_feat = seq.shape[1]

        # 1. Horizontal Flip (50% chance)
        if np.random.rand() < 0.5 and n_feat >= 36:
            # Flip base x-coordinates (even indices 0,2,...,32)
            for i in range(0, min(34, n_feat), 2):
                seq[:, i] = -seq[:, i]  # negate (already scaled, so negate)

            # Swap left/right keypoint pairs in base (indices within 0-33)
            swap_pairs_base = [
                (10, 12), (11, 13),  # kpt5<->6
                (14, 16), (15, 17),  # kpt7<->8
                (18, 20), (19, 21),  # kpt9<->10
                (22, 24), (23, 25),  # kpt11<->12
                (26, 28), (27, 29),  # kpt13<->14
                (30, 32), (31, 33),  # kpt15<->16
            ]
            for a, b in swap_pairs_base:
                if a < n_feat and b < n_feat:
                    seq[:, [a, b]] = seq[:, [b, a]]

            # Swap elbow angles (34<->35)
            if n_feat > 35:
                seq[:, [34, 35]] = seq[:, [35, 34]]

            # Velocity features (indices 36-71)
            if n_feat >= 72:
                for i in range(36, 70, 2):
                    seq[:, i] = -seq[:, i]  # negate velocity x
                swap_pairs_vel = [(36 + a, 36 + b) for a, b in
                    [(10, 12), (11, 13), (14, 16), (15, 17),
                     (18, 20), (19, 21), (22, 24), (23, 25),
                     (26, 28), (27, 29), (30, 32), (31, 33)]]
                for a, b in swap_pairs_vel:
                    if a < n_feat and b < n_feat:
                        seq[:, [a, b]] = seq[:, [b, a]]

            # Spatial features (72-76)
            if n_feat >= 77:
                seq[:, [72, 73]] = seq[:, [73, 72]]  # wrist-hip left<->right
                seq[:, [75, 76]] = seq[:, [76, 75]]  # cross-body flags

            # Advanced features (77-84)
            if n_feat >= 85:
                seq[:, 79] = -seq[:, 79]  # negate head orientation x
                seq[:, [81, 82]] = seq[:, [82, 81]]  # swap knee angles

        # 2. Gaussian Noise (always)
        seq += np.random.normal(0, 0.01, seq.shape)

        # 3. Temporal Crop (30% chance, only if seq_length > 18)
        if np.random.rand() < 0.3 and seq.shape[0] > 18:
            crop_len = np.random.randint(3, 6)
            max_start = seq.shape[0] - crop_len - 1
            if max_start > 1:
                start = np.random.randint(1, max_start)
                seq = np.concatenate([seq[:start], seq[start + crop_len:]], axis=0)

        # 4. Joint Dropout (20% chance)
        if np.random.rand() < 0.2 and n_feat >= 36:
            n_drop = np.random.randint(1, 3)
            kpts = np.random.choice(17, n_drop, replace=False)
            for k in kpts:
                # Zero base coords
                if 2 * k + 1 < min(34, n_feat):
                    seq[:, 2 * k] = 0
                    seq[:, 2 * k + 1] = 0
                # Zero velocity coords
                if n_feat >= 72 and 36 + 2 * k + 1 < n_feat:
                    seq[:, 36 + 2 * k] = 0
                    seq[:, 36 + 2 * k + 1] = 0

        return seq


def collate_fn(batch):
    """Pad sequences to max length in batch, sort by length descending."""
    batch.sort(key=lambda x: x[2], reverse=True)
    seqs, labels, lengths, clip_names = zip(*batch)

    max_len = max(lengths)
    n_feat = seqs[0].shape[1]

    padded = torch.zeros(len(seqs), max_len, n_feat)
    for i, seq in enumerate(seqs):
        padded[i, :seq.shape[0]] = seq

    labels = torch.cat(labels)
    lengths = torch.LongTensor(lengths)
    return padded, labels, lengths, list(clip_names)


# -- Model -------------------------------------------------------------------
class RetailLSTM_V10(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.3,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths, return_hidden: bool = False):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # Get last valid hidden state for each sample
        idx = (lengths - 1).long().unsqueeze(1).unsqueeze(2)
        idx = idx.expand(-1, 1, output.size(2)).to(output.device)
        hidden = output.gather(1, idx).squeeze(1)  # (B, 64)

        if return_hidden:
            return hidden

        logits = self.fc(hidden).squeeze(1)
        return torch.sigmoid(logits)


# -- Loss --------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = targets * inputs + (1 - targets) * (1 - inputs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


# -- Threshold optimization --------------------------------------------------
def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """Find threshold that maximizes F1 on clip-level scores."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    return float(best_thresh), float(best_f1)


# -- Clip-level aggregation (MAX) -------------------------------------------
def aggregate_clip_scores(clip_names: list, scores: np.ndarray, labels: np.ndarray):
    """Aggregate track-level scores to clip-level using MAX."""
    clip_scores = {}
    clip_labels = {}
    for name, score, label in zip(clip_names, scores, labels):
        if name not in clip_scores:
            clip_scores[name] = []
            clip_labels[name] = label
        clip_scores[name].append(score)

    clip_names_unique = sorted(clip_scores.keys())
    agg_scores = np.array([max(clip_scores[n]) for n in clip_names_unique])
    agg_labels = np.array([clip_labels[n] for n in clip_names_unique])
    return clip_names_unique, agg_scores, agg_labels


# -- Training ----------------------------------------------------------------
def train_one_fold(
    train_records: list[dict],
    val_records: list[dict],
    feature_cols: list[int],
    fold_num: int,
    tier_name: str,
) -> dict:
    """Train LSTM + XGBoost for one fold, return metrics and predictions."""
    n_features = len(feature_cols)

    # Fit scaler on training data only
    all_train_frames = np.concatenate(
        [r["data"][:, feature_cols] for r in train_records], axis=0
    )
    scaler = StandardScaler()
    scaler.fit(all_train_frames)

    # Datasets
    train_ds = SkeletonDataset(train_records, feature_cols, scaler, augment=True)
    val_ds = SkeletonDataset(val_records, feature_cols, scaler, augment=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Model
    model = RetailLSTM_V10(input_size=n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = FocalLoss(alpha=0.75, gamma=2.0)

    # Training loop
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    max_epochs = 150
    early_stop_patience = 15

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        for seqs, labels, lengths, _ in train_loader:
            seqs, labels, lengths = seqs.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
            optimizer.zero_grad()
            preds = model(seqs, lengths)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for seqs, labels, lengths, _ in val_loader:
                seqs, labels, lengths = seqs.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
                preds = model(seqs, lengths)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                val_batches += 1

        avg_train = train_loss / max(train_batches, 1)
        avg_val = val_loss / max(val_batches, 1)
        scheduler.step(avg_val)

        if (epoch + 1) % 10 == 0:
            print(f"  [{tier_name} Fold {fold_num}] Epoch {epoch+1}: "
                  f"train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  [{tier_name} Fold {fold_num}] Early stopping at epoch {epoch+1}")
                break

    # Load best checkpoint
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    # --- LSTM-only evaluation ---
    val_scores_lstm = []
    val_labels_all = []
    val_clips_all = []
    with torch.no_grad():
        for seqs, labels, lengths, clip_names in val_loader:
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            preds = model(seqs, lengths)
            val_scores_lstm.extend(preds.cpu().numpy().tolist())
            val_labels_all.extend(labels.numpy().tolist())
            val_clips_all.extend(clip_names)

    val_scores_lstm = np.array(val_scores_lstm)
    val_labels_all = np.array(val_labels_all)

    # Clip-level aggregation for LSTM
    clip_names_u, clip_scores_lstm, clip_labels = aggregate_clip_scores(
        val_clips_all, val_scores_lstm, val_labels_all
    )
    thresh_lstm, _ = find_optimal_threshold(clip_labels, clip_scores_lstm)
    clip_preds_lstm = (clip_scores_lstm >= thresh_lstm).astype(int)

    lstm_metrics = {
        "accuracy": accuracy_score(clip_labels, clip_preds_lstm),
        "precision": precision_score(clip_labels, clip_preds_lstm, zero_division=0),
        "recall": recall_score(clip_labels, clip_preds_lstm, zero_division=0),
        "f1": f1_score(clip_labels, clip_preds_lstm, zero_division=0),
        "auc": roc_auc_score(clip_labels, clip_scores_lstm) if len(np.unique(clip_labels)) > 1 else 0.0,
        "threshold": thresh_lstm,
    }

    # --- XGBoost hybrid ---
    # Extract hidden states from TRAINING data (no augmentation)
    train_ds_no_aug = SkeletonDataset(train_records, feature_cols, scaler, augment=False)
    train_loader_no_aug = DataLoader(
        train_ds_no_aug, batch_size=64, shuffle=False, collate_fn=collate_fn
    )

    train_hiddens = []
    train_labels_xgb = []
    with torch.no_grad():
        for seqs, labels, lengths, _ in train_loader_no_aug:
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            h = model(seqs, lengths, return_hidden=True)
            train_hiddens.append(h.cpu().numpy())
            train_labels_xgb.extend(labels.numpy().tolist())

    train_hiddens = np.concatenate(train_hiddens, axis=0)
    train_labels_xgb = np.array(train_labels_xgb)

    # Count for scale_pos_weight
    n_class0 = np.sum(train_labels_xgb == 0)
    n_class1 = np.sum(train_labels_xgb == 1)
    spw = n_class0 / max(n_class1, 1)

    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=spw,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
    )
    xgb.fit(train_hiddens, train_labels_xgb)

    # Val hidden states
    val_hiddens = []
    with torch.no_grad():
        for seqs, labels, lengths, _ in val_loader:
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            h = model(seqs, lengths, return_hidden=True)
            val_hiddens.append(h.cpu().numpy())

    val_hiddens = np.concatenate(val_hiddens, axis=0)
    val_scores_xgb = xgb.predict_proba(val_hiddens)[:, 1]

    # Clip-level aggregation for hybrid
    clip_names_u2, clip_scores_xgb, clip_labels2 = aggregate_clip_scores(
        val_clips_all, val_scores_xgb, val_labels_all
    )
    thresh_xgb, _ = find_optimal_threshold(clip_labels2, clip_scores_xgb)
    clip_preds_xgb = (clip_scores_xgb >= thresh_xgb).astype(int)

    hybrid_metrics = {
        "accuracy": accuracy_score(clip_labels2, clip_preds_xgb),
        "precision": precision_score(clip_labels2, clip_preds_xgb, zero_division=0),
        "recall": recall_score(clip_labels2, clip_preds_xgb, zero_division=0),
        "f1": f1_score(clip_labels2, clip_preds_xgb, zero_division=0),
        "auc": roc_auc_score(clip_labels2, clip_scores_xgb) if len(np.unique(clip_labels2)) > 1 else 0.0,
        "threshold": thresh_xgb,
    }

    # Per-clip results for error analysis
    clip_results = []
    for i, name in enumerate(clip_names_u2):
        clip_results.append({
            "clip_name": name,
            "true_label": int(clip_labels2[i]),
            "lstm_score": float(clip_scores_lstm[i]) if i < len(clip_scores_lstm) else 0.0,
            "hybrid_score": float(clip_scores_xgb[i]),
            "hybrid_pred": int(clip_preds_xgb[i]),
            "lstm_pred": int(clip_preds_lstm[i]) if i < len(clip_preds_lstm) else 0,
        })

    return {
        "lstm_metrics": lstm_metrics,
        "hybrid_metrics": hybrid_metrics,
        "clip_results": clip_results,
        "clip_scores_lstm": clip_scores_lstm.tolist(),
        "clip_scores_xgb": clip_scores_xgb.tolist(),
        "clip_labels": clip_labels2.tolist(),
        "clip_names": clip_names_u2,
        "scaler": scaler,
        "model_state": best_state,
        "xgb_model": xgb,
    }


# -- Main --------------------------------------------------------------------
def main():
    print("Loading data...")
    all_records = load_all_data()
    print(f"Loaded {len(all_records)} tracks")

    # Extract arrays for GroupKFold
    groups = np.array([r["source_group"] for r in all_records])
    labels = np.array([r["label"] for r in all_records])

    gkf = GroupKFold(n_splits=5)
    fold_indices = list(gkf.split(np.zeros(len(all_records)), labels, groups))

    # Save fold assignments
    with open(RESULTS_DIR / "fold_assignments.pkl", "wb") as f:
        pickle.dump(fold_indices, f)

    # Print fold composition
    print("\nFold composition:")
    for i, (train_idx, val_idx) in enumerate(fold_indices):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        print(f"  Fold {i+1}: Train = {len(train_groups)} groups ({len(train_idx)} tracks) | "
              f"Val = {len(val_groups)} groups ({len(val_idx)} tracks)")

    # Train all tiers x folds
    all_results = {}

    for tier_name, feature_cols in FEATURE_TIERS.items():
        print(f"\n{'='*60}")
        print(f"TIER: {tier_name} ({len(feature_cols)} features)")
        print(f"{'='*60}")

        all_results[tier_name] = {}

        for fold_num, (train_idx, val_idx) in enumerate(fold_indices, 1):
            print(f"\n--- Fold {fold_num}/5 ---")
            train_recs = [all_records[i] for i in train_idx]
            val_recs = [all_records[i] for i in val_idx]

            fold_result = train_one_fold(
                train_recs, val_recs, feature_cols, fold_num, tier_name
            )
            all_results[tier_name][fold_num] = fold_result

            print(f"  LSTM:   F1={fold_result['lstm_metrics']['f1']:.4f} "
                  f"AUC={fold_result['lstm_metrics']['auc']:.4f} "
                  f"(thresh={fold_result['lstm_metrics']['threshold']:.3f})")
            print(f"  Hybrid: F1={fold_result['hybrid_metrics']['f1']:.4f} "
                  f"AUC={fold_result['hybrid_metrics']['auc']:.4f} "
                  f"(thresh={fold_result['hybrid_metrics']['threshold']:.3f})")

    # Save best Tier B model (best fold by hybrid F1)
    tier_b = all_results["B_velocity"]
    best_fold = max(tier_b.keys(), key=lambda k: tier_b[k]["hybrid_metrics"]["f1"])
    best_result = tier_b[best_fold]

    torch.save(best_result["model_state"], RESULTS_DIR / "lstm_v10.pth")
    with open(RESULTS_DIR / "xgboost_v10.pkl", "wb") as f:
        pickle.dump(best_result["xgb_model"], f)
    with open(RESULTS_DIR / "scaler_v10.pkl", "wb") as f:
        pickle.dump(best_result["scaler"], f)

    # Save all results (strip non-serializable objects for the main pickle)
    results_to_save = {}
    for tier in all_results:
        results_to_save[tier] = {}
        for fold in all_results[tier]:
            r = all_results[tier][fold]
            results_to_save[tier][fold] = {
                "lstm_metrics": r["lstm_metrics"],
                "hybrid_metrics": r["hybrid_metrics"],
                "clip_results": r["clip_results"],
                "clip_scores_lstm": r["clip_scores_lstm"],
                "clip_scores_xgb": r["clip_scores_xgb"],
                "clip_labels": r["clip_labels"],
                "clip_names": r["clip_names"],
            }

    with open(RESULTS_DIR / "all_results.pkl", "wb") as f:
        pickle.dump(results_to_save, f)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for tier_name in FEATURE_TIERS:
        tier_res = all_results[tier_name]
        lstm_f1s = [tier_res[f]["lstm_metrics"]["f1"] for f in tier_res]
        lstm_aucs = [tier_res[f]["lstm_metrics"]["auc"] for f in tier_res]
        hyb_f1s = [tier_res[f]["hybrid_metrics"]["f1"] for f in tier_res]
        hyb_aucs = [tier_res[f]["hybrid_metrics"]["auc"] for f in tier_res]

        n_feat = len(FEATURE_TIERS[tier_name])
        print(f"\nTIER {tier_name} ({n_feat} features):")
        print(f"  LSTM:   F1 = {np.mean(lstm_f1s):.4f} +/- {np.std(lstm_f1s):.4f} | "
              f"AUC = {np.mean(lstm_aucs):.4f} +/- {np.std(lstm_aucs):.4f}")
        print(f"  Hybrid: F1 = {np.mean(hyb_f1s):.4f} +/- {np.std(hyb_f1s):.4f} | "
              f"AUC = {np.mean(hyb_aucs):.4f} +/- {np.std(hyb_aucs):.4f}")

    print(f"\nBest Tier B model saved from fold {best_fold}")
    print("All results saved to v10_Results/all_results.pkl")
    print("Done.")


if __name__ == "__main__":
    main()
