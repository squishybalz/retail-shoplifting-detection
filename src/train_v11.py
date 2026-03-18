"""
V10 -> V11 Experiment Framework (Rounds 1-2)
Iterative improvement: ONE change at a time, measure, keep or revert.
Runs on Tier C (77 features) with LSTM-only (no XGBoost hybrid).
Uses same fold_assignments.pkl from V10 for fair comparison.
"""

import os, re, pickle, csv, time, math
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve,
)
import warnings
warnings.filterwarnings("ignore")

# -- Paths ------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\malho\Desktop\claudeagent\RETAILPROJECT")
FEATURE_DIR = BASE_DIR / "v10_Features"
RESULTS_DIR = BASE_DIR / "v10_Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- Experiment Config -------------------------------------------------------
@dataclass
class ExpConfig:
    exp_id: str = "1.0"
    description: str = "Baseline LSTM-only Tier C"
    # Model architecture
    hidden_size: int = 64
    num_layers: int = 2
    lstm_dropout: float = 0.3
    bidirectional: bool = False
    use_attention: bool = False
    use_layernorm: bool = False
    # Training
    lr: float = 0.0005
    batch_size: int = 32
    max_epochs: int = 150
    early_stop_patience: int = 15
    scheduler_type: str = "plateau"       # "plateau" or "cosine"
    early_stop_metric: str = "val_loss"   # "val_loss" or "val_f1"
    # Loss
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    # Features
    feature_cols: list = field(default_factory=lambda: list(range(0, 77)))

    def clone(self, **overrides):
        """Create a copy with optional overrides."""
        import copy
        cfg = copy.deepcopy(self)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


# -- Model -------------------------------------------------------------------
class RetailLSTM_V11(nn.Module):
    def __init__(self, cfg: ExpConfig):
        super().__init__()
        input_size = len(cfg.feature_cols)
        self.cfg = cfg

        self.layernorm = nn.LayerNorm(input_size) if cfg.use_layernorm else None

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.lstm_dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )

        lstm_out = cfg.hidden_size * (2 if cfg.bidirectional else 1)

        if cfg.use_attention:
            self.attn_weight = nn.Linear(lstm_out, 1)

        self.fc = nn.Linear(lstm_out, 1)

    def forward(self, x, lengths, return_hidden=False):
        if self.layernorm is not None:
            x = self.layernorm(x)

        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)  # (B, T, H)

        if self.cfg.use_attention:
            B, T, H = output.shape
            mask = torch.arange(T, device=output.device).unsqueeze(0) < lengths.unsqueeze(1)
            attn_scores = self.attn_weight(output).squeeze(-1)  # (B, T)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
            attn_weights = F.softmax(attn_scores, dim=1)        # (B, T)
            hidden = (output * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, H)
        else:
            idx = (lengths - 1).long().unsqueeze(1).unsqueeze(2)
            idx = idx.expand(-1, 1, output.size(2)).to(output.device)
            hidden = output.gather(1, idx).squeeze(1)

        if return_hidden:
            return hidden

        logits = self.fc(hidden).squeeze(1)
        return torch.sigmoid(logits)


# -- Loss --------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = targets * inputs + (1 - targets) * (1 - inputs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


# -- Dataset -----------------------------------------------------------------
class SkeletonDataset(Dataset):
    def __init__(self, records, feature_cols, scaler, augment=False):
        self.records = records
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.augment = augment
        self.n_features = len(feature_cols)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        seq = rec["data"][:, self.feature_cols].copy()
        seq = self.scaler.transform(seq)
        if self.augment:
            seq = self._augment(seq)
        return (
            torch.FloatTensor(seq),
            torch.FloatTensor([rec["label"]]),
            seq.shape[0],
            rec["clip_name"],
        )

    def _augment(self, seq):
        seq = seq.copy()
        n_feat = seq.shape[1]

        # 1. Horizontal Flip (50%)
        if np.random.rand() < 0.5 and n_feat >= 36:
            for i in range(0, min(34, n_feat), 2):
                seq[:, i] = -seq[:, i]
            swap_pairs = [
                (10, 12), (11, 13), (14, 16), (15, 17),
                (18, 20), (19, 21), (22, 24), (23, 25),
                (26, 28), (27, 29), (30, 32), (31, 33),
            ]
            for a, b in swap_pairs:
                if a < n_feat and b < n_feat:
                    seq[:, [a, b]] = seq[:, [b, a]]
            if n_feat > 35:
                seq[:, [34, 35]] = seq[:, [35, 34]]
            if n_feat >= 72:
                for i in range(36, 70, 2):
                    seq[:, i] = -seq[:, i]
                vel_swaps = [(36 + a, 36 + b) for a, b in swap_pairs]
                for a, b in vel_swaps:
                    if a < n_feat and b < n_feat:
                        seq[:, [a, b]] = seq[:, [b, a]]
            if n_feat >= 77:
                seq[:, [72, 73]] = seq[:, [73, 72]]
                seq[:, [75, 76]] = seq[:, [76, 75]]

        # 2. Gaussian Noise
        seq += np.random.normal(0, 0.01, seq.shape)

        # 3. Temporal Crop (30%, if len > 18)
        if np.random.rand() < 0.3 and seq.shape[0] > 18:
            crop_len = np.random.randint(3, 6)
            max_start = seq.shape[0] - crop_len - 1
            if max_start > 1:
                start = np.random.randint(1, max_start)
                seq = np.concatenate([seq[:start], seq[start + crop_len:]], axis=0)

        # 4. Joint Dropout (20%)
        if np.random.rand() < 0.2 and n_feat >= 36:
            n_drop = np.random.randint(1, 3)
            kpts = np.random.choice(17, n_drop, replace=False)
            for k in kpts:
                if 2 * k + 1 < min(34, n_feat):
                    seq[:, 2 * k] = 0
                    seq[:, 2 * k + 1] = 0
                if n_feat >= 72 and 36 + 2 * k + 1 < n_feat:
                    seq[:, 36 + 2 * k] = 0
                    seq[:, 36 + 2 * k + 1] = 0

        return seq


def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    seqs, labels, lengths, clip_names = zip(*batch)
    max_len = max(lengths)
    n_feat = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, n_feat)
    for i, seq in enumerate(seqs):
        padded[i, : seq.shape[0]] = seq
    labels = torch.cat(labels)
    lengths = torch.LongTensor(lengths)
    return padded, labels, lengths, list(clip_names)


# -- Utilities ---------------------------------------------------------------
def load_all_data():
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


def find_optimal_threshold(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1s = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(best_thresh), float(f1s[best_idx])


def aggregate_clip_scores(clip_names, scores, labels):
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


# -- Training ----------------------------------------------------------------
def train_one_fold(cfg, train_records, val_records, fold_num):
    feature_cols = cfg.feature_cols
    n_features = len(feature_cols)

    # Fit scaler on training data only
    all_train = np.concatenate(
        [r["data"][:, feature_cols] for r in train_records], axis=0
    )
    scaler = StandardScaler()
    scaler.fit(all_train)

    train_ds = SkeletonDataset(train_records, feature_cols, scaler, augment=True)
    val_ds = SkeletonDataset(val_records, feature_cols, scaler, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn
    )

    model = RetailLSTM_V11(cfg).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if cfg.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

    criterion = FocalLoss(
        alpha=cfg.focal_alpha,
        gamma=cfg.focal_gamma,
        label_smoothing=cfg.label_smoothing,
    )

    best_val_metric = float("-inf") if cfg.early_stop_metric == "val_f1" else float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(cfg.max_epochs):
        # -- Train --
        model.train()
        train_loss = 0.0
        n_batches = 0
        for seqs, labels, lengths, _ in train_loader:
            seqs, labels, lengths = (
                seqs.to(DEVICE),
                labels.to(DEVICE),
                lengths.to(DEVICE),
            )
            optimizer.zero_grad()
            preds = model(seqs, lengths)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

            if cfg.scheduler_type == "cosine":
                scheduler.step(epoch + n_batches / len(train_loader))

        # -- Validate --
        model.eval()
        val_loss = 0.0
        val_n = 0
        val_scores_list = []
        val_labels_list = []
        val_clips_list = []
        with torch.no_grad():
            for seqs, labels, lengths, clips in val_loader:
                seqs, labels, lengths = (
                    seqs.to(DEVICE),
                    labels.to(DEVICE),
                    lengths.to(DEVICE),
                )
                preds = model(seqs, lengths)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                val_n += 1
                val_scores_list.extend(preds.cpu().numpy().tolist())
                val_labels_list.extend(labels.cpu().numpy().tolist())
                val_clips_list.extend(clips)

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_n, 1)

        if cfg.scheduler_type == "plateau":
            scheduler.step(avg_val)

        # Early stopping metric
        if cfg.early_stop_metric == "val_f1":
            _, cs, cl = aggregate_clip_scores(
                val_clips_list,
                np.array(val_scores_list),
                np.array(val_labels_list),
            )
            _, current_f1 = find_optimal_threshold(cl, cs)
            current_metric = current_f1
            improved = current_metric > best_val_metric
        else:
            current_metric = avg_val
            improved = current_metric < best_val_metric

        if (epoch + 1) % 25 == 0:
            extra = f"val_f1={current_metric:.4f}" if cfg.early_stop_metric == "val_f1" else f"val_loss={avg_val:.4f}"
            print(f"  [Fold {fold_num}] Epoch {epoch+1}: train_loss={avg_train:.4f}, {extra}")

        if improved:
            best_val_metric = current_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"  [Fold {fold_num}] Early stop at epoch {epoch+1}")
                break

    # Load best checkpoint and evaluate
    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    val_scores = []
    val_labels = []
    val_clips = []
    with torch.no_grad():
        for seqs, labels, lengths, clips in val_loader:
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            preds = model(seqs, lengths)
            val_scores.extend(preds.cpu().numpy().tolist())
            val_labels.extend(labels.numpy().tolist())
            val_clips.extend(clips)

    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)

    clip_names_u, clip_scores, clip_labels = aggregate_clip_scores(
        val_clips, val_scores, val_labels
    )
    thresh, _ = find_optimal_threshold(clip_labels, clip_scores)
    clip_preds = (clip_scores >= thresh).astype(int)

    metrics = {
        "accuracy": accuracy_score(clip_labels, clip_preds),
        "precision": precision_score(clip_labels, clip_preds, zero_division=0),
        "recall": recall_score(clip_labels, clip_preds, zero_division=0),
        "f1": f1_score(clip_labels, clip_preds, zero_division=0),
        "auc": roc_auc_score(clip_labels, clip_scores)
        if len(np.unique(clip_labels)) > 1
        else 0.0,
        "threshold": thresh,
    }

    clip_results = []
    for i, name in enumerate(clip_names_u):
        clip_results.append({
            "clip_name": name,
            "true_label": int(clip_labels[i]),
            "score": float(clip_scores[i]),
            "pred": int(clip_preds[i]),
        })

    return {
        "metrics": metrics,
        "clip_results": clip_results,
        "model_state": best_state,
        "scaler": scaler,
    }


def run_experiment(cfg, all_records, fold_indices):
    """Run 5-fold CV with given config. Returns aggregate metrics."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {cfg.exp_id}: {cfg.description}")
    print(f"{'='*60}")

    fold_metrics = []
    all_clip_results = []
    best_fold_f1 = -1
    best_fold_state = None
    best_fold_scaler = None

    t0 = time.time()
    for fold_num, (train_idx, val_idx) in enumerate(fold_indices, 1):
        print(f"\n--- Fold {fold_num}/5 ---")
        train_recs = [all_records[i] for i in train_idx]
        val_recs = [all_records[i] for i in val_idx]

        result = train_one_fold(cfg, train_recs, val_recs, fold_num)
        fold_metrics.append(result["metrics"])
        for cr in result["clip_results"]:
            cr["fold"] = fold_num
        all_clip_results.extend(result["clip_results"])

        print(
            f"  F1={result['metrics']['f1']:.4f} "
            f"AUC={result['metrics']['auc']:.4f} "
            f"(thresh={result['metrics']['threshold']:.3f})"
        )

        if result["metrics"]["f1"] > best_fold_f1:
            best_fold_f1 = result["metrics"]["f1"]
            best_fold_state = result["model_state"]
            best_fold_scaler = result["scaler"]

    elapsed = time.time() - t0

    # Aggregate
    agg = {}
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        vals = [m[metric] for m in fold_metrics]
        agg[f"{metric}_mean"] = float(np.mean(vals))
        agg[f"{metric}_std"] = float(np.std(vals))

    fp = sum(1 for cr in all_clip_results if cr["true_label"] == 0 and cr["pred"] == 1)
    fn = sum(1 for cr in all_clip_results if cr["true_label"] == 1 and cr["pred"] == 0)
    agg["fp_count"] = fp
    agg["fn_count"] = fn

    print(
        f"\nRESULTS: F1 = {agg['f1_mean']:.4f} +/- {agg['f1_std']:.4f} | "
        f"AUC = {agg['auc_mean']:.4f} +/- {agg['auc_std']:.4f} | "
        f"FP={fp} FN={fn} | {elapsed:.0f}s"
    )

    return {
        "agg_metrics": agg,
        "fold_metrics": fold_metrics,
        "clip_results": all_clip_results,
        "best_model_state": best_fold_state,
        "best_scaler": best_fold_scaler,
    }


# -- Decision Logic ----------------------------------------------------------
def decide(baseline, experiment):
    b = baseline["agg_metrics"]
    e = experiment["agg_metrics"]

    delta_f1 = e["f1_mean"] - b["f1_mean"]
    delta_auc = e["auc_mean"] - b["auc_mean"]
    delta_std = e["f1_std"] - b["f1_std"]

    if delta_f1 > 0.005 and delta_std <= 0.02:
        return "KEEP", f"F1 +{delta_f1:.4f}, std ok ({delta_std:+.4f})"
    elif delta_f1 > 0.005 and delta_std > 0.02:
        return "KEEP_FLAG", f"F1 +{delta_f1:.4f}, but std +{delta_std:.4f} — needs investigation"
    elif delta_f1 > 0 and delta_auc > 0:
        return "KEEP", f"F1 +{delta_f1:.4f}, AUC +{delta_auc:.4f} (both improved)"
    else:
        return "REVERT", f"F1 {delta_f1:+.4f}, AUC {delta_auc:+.4f}"


# -- Logging -----------------------------------------------------------------
LOG_HEADER = [
    "exp_id", "change_description",
    "baseline_f1", "new_f1", "delta_f1",
    "baseline_auc", "new_auc", "delta_auc",
    "baseline_std", "new_std",
    "fp", "fn", "verdict",
]


def init_log(log_path):
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(LOG_HEADER)


def log_result(log_path, exp_id, desc, baseline, experiment, verdict):
    b = baseline["agg_metrics"]
    e = experiment["agg_metrics"]
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([
            exp_id, desc,
            f"{b['f1_mean']:.4f}", f"{e['f1_mean']:.4f}",
            f"{e['f1_mean'] - b['f1_mean']:+.4f}",
            f"{b['auc_mean']:.4f}", f"{e['auc_mean']:.4f}",
            f"{e['auc_mean'] - b['auc_mean']:+.4f}",
            f"{b['f1_std']:.4f}", f"{e['f1_std']:.4f}",
            e["fp_count"], e["fn_count"], verdict,
        ])


def print_comparison(baseline, experiment, exp_id):
    b = baseline["agg_metrics"]
    e = experiment["agg_metrics"]
    print(f"\n{'-'*50}")
    print(f"{'Metric':<12} {'Baseline':>10} {'Exp '+exp_id:>10} {'Delta':>10}")
    print(f"{'-'*50}")
    for m in ["f1", "auc", "accuracy", "precision", "recall"]:
        bv = b[f"{m}_mean"]
        ev = e[f"{m}_mean"]
        d = ev - bv
        arrow = "^" if d > 0 else "v" if d < 0 else "="
        print(f"{m:<12} {bv:>10.4f} {ev:>10.4f} {d:>+10.4f} {arrow}")
    print(f"{'f1_std':<12} {b['f1_std']:>10.4f} {e['f1_std']:>10.4f} {e['f1_std']-b['f1_std']:>+10.4f}")
    print(f"{'FP':<12} {b['fp_count']:>10d} {e['fp_count']:>10d} {e['fp_count']-b['fp_count']:>+10d}")
    print(f"{'FN':<12} {b['fn_count']:>10d} {e['fn_count']:>10d} {e['fn_count']-b['fn_count']:>+10d}")
    print(f"{'-'*50}")


# -- Consolidation ----------------------------------------------------------
def build_combined_config(base_cfg, kept_changes):
    """Merge all kept changes into a single config."""
    combined = base_cfg.clone(exp_id="combined", description="Combined kept changes")
    for cfg, _ in kept_changes:
        if cfg.hidden_size != 64:
            combined.hidden_size = cfg.hidden_size
        if cfg.bidirectional:
            combined.bidirectional = True
        if cfg.use_attention:
            combined.use_attention = True
        if cfg.use_layernorm:
            combined.use_layernorm = True
        if cfg.scheduler_type != "plateau":
            combined.scheduler_type = cfg.scheduler_type
        if cfg.label_smoothing > 0:
            combined.label_smoothing = cfg.label_smoothing
        if cfg.early_stop_metric != "val_loss":
            combined.early_stop_metric = cfg.early_stop_metric
        if cfg.focal_alpha != 0.75 or cfg.focal_gamma != 2.0:
            combined.focal_alpha = cfg.focal_alpha
            combined.focal_gamma = cfg.focal_gamma
    return combined


def save_error_analysis(clip_results, path, model_name):
    fps = [cr for cr in clip_results if cr["true_label"] == 0 and cr["pred"] == 1]
    fns = [cr for cr in clip_results if cr["true_label"] == 1 and cr["pred"] == 0]
    fps.sort(key=lambda x: -x["score"])
    fns.sort(key=lambda x: x["score"])

    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"ERROR ANALYSIS — {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"=== FALSE POSITIVES (Normal -> predicted Shoplifting) ===\n")
        for cr in fps:
            f.write(f"  {cr['clip_name']} | score={cr['score']:.4f} | fold {cr['fold']}\n")
        f.write(f"\n=== FALSE NEGATIVES (Shoplifting -> predicted Normal) ===\n")
        for cr in fns:
            f.write(f"  {cr['clip_name']} | score={cr['score']:.4f} | fold {cr['fold']}\n")
        f.write(f"\nSummary:\n")
        f.write(f"  FP: {len(fps)} unique normal clips misclassified\n")
        f.write(f"  FN: {len(fns)} unique shoplifting clips misclassified\n")


# -- Main --------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    log_path = RESULTS_DIR / "experiment_log.csv"
    init_log(log_path)

    print("Loading data...")
    all_records = load_all_data()
    print(f"Loaded {len(all_records)} tracks")

    with open(RESULTS_DIR / "fold_assignments.pkl", "rb") as f:
        fold_indices = pickle.load(f)
    print(f"Loaded {len(fold_indices)} fold assignments")

    # ===============================================================
    # BASELINE: LSTM-only Tier C (replicates V10 best)
    # ===============================================================
    baseline_cfg = ExpConfig()
    baseline_result = run_experiment(baseline_cfg, all_records, fold_indices)
    log_result(log_path, "1.0", "Baseline LSTM-only Tier C", baseline_result, baseline_result, "BASELINE")

    torch.save(baseline_result["best_model_state"], RESULTS_DIR / "lstm_v11_baseline.pth")

    current_best_cfg = baseline_cfg
    current_best_result = baseline_result

    # ===============================================================
    # ROUND 1: Architecture Tweaks
    # ===============================================================
    print("\n" + "#" * 60)
    print("# ROUND 1: ARCHITECTURE TWEAKS")
    print("#" * 60)

    r1_experiments = [
        baseline_cfg.clone(exp_id="1.2", description="Hidden size 64 -> 128", hidden_size=128),
        baseline_cfg.clone(exp_id="1.3", description="Temporal attention pooling", use_attention=True),
        baseline_cfg.clone(exp_id="1.4", description="LayerNorm before LSTM input", use_layernorm=True),
        baseline_cfg.clone(exp_id="1.5", description="Bidirectional LSTM", bidirectional=True),
    ]

    r1_kept = []
    for exp_cfg in r1_experiments:
        result = run_experiment(exp_cfg, all_records, fold_indices)
        print_comparison(baseline_result, result, exp_cfg.exp_id)
        verdict, reason = decide(baseline_result, result)
        log_result(log_path, exp_cfg.exp_id, exp_cfg.description, baseline_result, result, verdict)
        print(f"\n>>> VERDICT: {verdict} -- {reason}")

        if verdict.startswith("KEEP"):
            r1_kept.append((exp_cfg, result))

    # Consolidate Round 1
    if r1_kept:
        print(f"\n{'='*60}")
        print(f"ROUND 1 CONSOLIDATION: {len(r1_kept)} kept changes")
        for cfg, _ in r1_kept:
            print(f"  - {cfg.exp_id}: {cfg.description}")
        print(f"{'='*60}")

        combined_cfg = build_combined_config(baseline_cfg, r1_kept)
        combined_cfg.exp_id = "R1_combined"
        combined_cfg.description = "Round 1 all kept changes combined"
        combined_result = run_experiment(combined_cfg, all_records, fold_indices)
        print_comparison(baseline_result, combined_result, "R1_combined")
        verdict, reason = decide(baseline_result, combined_result)
        log_result(log_path, "R1_combined", combined_cfg.description, baseline_result, combined_result, verdict)
        print(f"\n>>> CONSOLIDATION VERDICT: {verdict} -- {reason}")

        if verdict.startswith("KEEP"):
            current_best_cfg = combined_cfg
            current_best_result = combined_result
            print(">>> R1 combined model is new baseline for Round 2")
        else:
            # Fall back to single best kept change
            best_single = max(r1_kept, key=lambda x: x[1]["agg_metrics"]["f1_mean"])
            v2, r2 = decide(baseline_result, best_single[1])
            if v2.startswith("KEEP"):
                current_best_cfg = best_single[0]
                current_best_result = best_single[1]
                print(f">>> Fallback: using single best ({best_single[0].exp_id}) as new baseline")
    else:
        print("\nNo Round 1 changes kept. Baseline unchanged.")

    # ===============================================================
    # ROUND 2: Training Optimization
    # ===============================================================
    print("\n" + "#" * 60)
    print("# ROUND 2: TRAINING OPTIMIZATION")
    print("#" * 60)

    r2_experiments = [
        current_best_cfg.clone(
            exp_id="2.1", description="CosineAnnealingWarmRestarts scheduler",
            scheduler_type="cosine"
        ),
        current_best_cfg.clone(
            exp_id="2.2", description="Label smoothing 0.05",
            label_smoothing=0.05
        ),
        current_best_cfg.clone(
            exp_id="2.3", description="Early stop on val F1 instead of val loss",
            early_stop_metric="val_f1"
        ),
    ]

    r2_kept = []
    for exp_cfg in r2_experiments:
        result = run_experiment(exp_cfg, all_records, fold_indices)
        print_comparison(current_best_result, result, exp_cfg.exp_id)
        verdict, reason = decide(current_best_result, result)
        log_result(log_path, exp_cfg.exp_id, exp_cfg.description, current_best_result, result, verdict)
        print(f"\n>>> VERDICT: {verdict} -- {reason}")

        if verdict.startswith("KEEP"):
            r2_kept.append((exp_cfg, result))

    # Exp 2.4: Focal loss grid search
    print(f"\n{'='*60}")
    print("EXP 2.4: FOCAL LOSS GRID SEARCH (12 combos)")
    print(f"{'='*60}")

    best_grid_cfg = None
    best_grid_result = None
    best_grid_f1 = current_best_result["agg_metrics"]["f1_mean"]

    for alpha in [0.5, 0.65, 0.75, 0.85]:
        for gamma in [1.0, 2.0, 3.0]:
            if alpha == 0.75 and gamma == 2.0:
                continue  # skip current default

            grid_cfg = current_best_cfg.clone(
                exp_id=f"2.4_a{alpha}_g{gamma}",
                description=f"Focal alpha={alpha}, gamma={gamma}",
                focal_alpha=alpha,
                focal_gamma=gamma,
            )
            result = run_experiment(grid_cfg, all_records, fold_indices)
            log_result(
                log_path, grid_cfg.exp_id, grid_cfg.description,
                current_best_result, result, "GRID_SEARCH"
            )

            if result["agg_metrics"]["f1_mean"] > best_grid_f1:
                best_grid_f1 = result["agg_metrics"]["f1_mean"]
                best_grid_cfg = grid_cfg
                best_grid_result = result

    if best_grid_result:
        print_comparison(current_best_result, best_grid_result, best_grid_cfg.exp_id)
        verdict, reason = decide(current_best_result, best_grid_result)
        log_result(
            log_path, f"2.4_best", f"Best grid: {best_grid_cfg.description}",
            current_best_result, best_grid_result, verdict
        )
        print(f"\nBest grid config: alpha={best_grid_cfg.focal_alpha}, gamma={best_grid_cfg.focal_gamma}")
        print(f">>> VERDICT: {verdict} -- {reason}")
        if verdict.startswith("KEEP"):
            r2_kept.append((best_grid_cfg, best_grid_result))
    else:
        print("\nNo grid search config beat current alpha=0.75, gamma=2.0")

    # Consolidate Round 2
    if r2_kept:
        print(f"\n{'='*60}")
        print(f"ROUND 2 CONSOLIDATION: {len(r2_kept)} kept changes")
        for cfg, _ in r2_kept:
            print(f"  - {cfg.exp_id}: {cfg.description}")
        print(f"{'='*60}")

        combined_cfg = build_combined_config(current_best_cfg, r2_kept)
        combined_cfg.exp_id = "R2_combined"
        combined_cfg.description = "Round 2 all kept changes combined"
        combined_result = run_experiment(combined_cfg, all_records, fold_indices)
        print_comparison(current_best_result, combined_result, "R2_combined")
        verdict, reason = decide(current_best_result, combined_result)
        log_result(log_path, "R2_combined", combined_cfg.description, current_best_result, combined_result, verdict)
        print(f"\n>>> CONSOLIDATION VERDICT: {verdict} -- {reason}")

        if verdict.startswith("KEEP"):
            current_best_cfg = combined_cfg
            current_best_result = combined_result
            print(">>> R2 combined model is new baseline")
        else:
            best_single = max(r2_kept, key=lambda x: x[1]["agg_metrics"]["f1_mean"])
            v2, r2 = decide(current_best_result, best_single[1])
            if v2.startswith("KEEP"):
                current_best_cfg = best_single[0]
                current_best_result = best_single[1]
                print(f">>> Fallback: using single best ({best_single[0].exp_id}) as new baseline")
    else:
        print("\nNo Round 2 changes kept.")

    # ===============================================================
    # SAVE FINAL BEST
    # ===============================================================
    torch.save(current_best_result["best_model_state"], RESULTS_DIR / "lstm_v11_best.pth")
    with open(RESULTS_DIR / "best_scaler_v11.pkl", "wb") as f:
        pickle.dump(current_best_result["best_scaler"], f)

    save_error_analysis(
        current_best_result["clip_results"],
        RESULTS_DIR / "error_analysis_v11.txt",
        f"V11 Best ({current_best_cfg.exp_id})",
    )

    with open(RESULTS_DIR / "v11_results.pkl", "wb") as f:
        pickle.dump({
            "best_config": {
                "exp_id": current_best_cfg.exp_id,
                "description": current_best_cfg.description,
                "hidden_size": current_best_cfg.hidden_size,
                "bidirectional": current_best_cfg.bidirectional,
                "use_attention": current_best_cfg.use_attention,
                "use_layernorm": current_best_cfg.use_layernorm,
                "scheduler_type": current_best_cfg.scheduler_type,
                "label_smoothing": current_best_cfg.label_smoothing,
                "early_stop_metric": current_best_cfg.early_stop_metric,
                "focal_alpha": current_best_cfg.focal_alpha,
                "focal_gamma": current_best_cfg.focal_gamma,
            },
            "best_metrics": current_best_result["agg_metrics"],
            "fold_metrics": current_best_result["fold_metrics"],
            "clip_results": current_best_result["clip_results"],
        }, f)

    # ===============================================================
    # FINAL SUMMARY
    # ===============================================================
    m = current_best_result["agg_metrics"]
    bm = baseline_result["agg_metrics"]
    print("\n" + "=" * 60)
    print("FINAL V11 SUMMARY (Rounds 1-2)")
    print("=" * 60)
    print(f"Best config: {current_best_cfg.exp_id} — {current_best_cfg.description}")
    print(f"  hidden={current_best_cfg.hidden_size}, bidir={current_best_cfg.bidirectional}, "
          f"attn={current_best_cfg.use_attention}, ln={current_best_cfg.use_layernorm}")
    print(f"  scheduler={current_best_cfg.scheduler_type}, smooth={current_best_cfg.label_smoothing}, "
          f"stop_on={current_best_cfg.early_stop_metric}")
    print(f"  focal: alpha={current_best_cfg.focal_alpha}, gamma={current_best_cfg.focal_gamma}")
    print(f"\n{'Metric':<12} {'V10 Baseline':>12} {'V11 Best':>12} {'Delta':>12}")
    print("-" * 50)
    for met in ["f1", "auc", "accuracy", "precision", "recall"]:
        bv = bm[f"{met}_mean"]
        ev = m[f"{met}_mean"]
        print(f"{met:<12} {bv:>12.4f} {ev:>12.4f} {ev-bv:>+12.4f}")
    print(f"{'f1_std':<12} {bm['f1_std']:>12.4f} {m['f1_std']:>12.4f} {m['f1_std']-bm['f1_std']:>+12.4f}")
    print(f"{'FP':<12} {bm['fp_count']:>12d} {m['fp_count']:>12d} {m['fp_count']-bm['fp_count']:>+12d}")
    print(f"{'FN':<12} {bm['fn_count']:>12d} {m['fn_count']:>12d} {m['fn_count']-bm['fn_count']:>+12d}")

    # Check stop criteria
    if m["f1_mean"] > 0.85 and m["f1_std"] < 0.05:
        print("\nSTOP CRITERIA MET: F1 > 0.85 with std < 0.05")
    else:
        print(f"\nTarget not yet reached (F1={m['f1_mean']:.4f}, std={m['f1_std']:.4f}). "
              f"Proceed to Round 3 (feature engineering).")

    print(f"\nExperiment log: {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
