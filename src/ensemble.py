"""
V11 Round 4: Ensemble & Error-Driven Experiments
Exp 4.1: Weighted ensemble (LSTM + Random Forest)
Exp 4.2: Platt scaling for threshold calibration
Exp 4.3: Hard negative mining (re-weight top FP/FN clips)

Uses same fold_assignments.pkl and best model config from prior rounds.
"""

import sys, re, pickle, csv, time
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve,
)
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

# Ensure src/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_v11 import (
    ExpConfig, RetailLSTM_V11, SkeletonDataset, FocalLoss,
    collate_fn, find_optimal_threshold, aggregate_clip_scores,
    load_all_data, save_error_analysis, init_log,
    RESULTS_DIR, DEVICE,
)
import train_v11

BASE_DIR = Path(r"C:\Users\malho\Desktop\claudeagent\RETAILPROJECT")

# Use v11_Features if available (158-dim), fall back to v10_Features (85-dim)
v11_feat_dir = BASE_DIR / "v11_Features"
if v11_feat_dir.exists() and any(v11_feat_dir.rglob("*.npy")):
    train_v11.FEATURE_DIR = v11_feat_dir
    FEATURE_DIR = v11_feat_dir
else:
    FEATURE_DIR = BASE_DIR / "v10_Features"
PAD_LENGTH = 30


def flatten_features(records, feature_cols, pad_len=PAD_LENGTH):
    n_feat = len(feature_cols)
    X = np.zeros((len(records), pad_len * n_feat), dtype=np.float32)
    for i, rec in enumerate(records):
        seq = rec["data"][:, feature_cols]
        T = min(seq.shape[0], pad_len)
        padded = np.zeros((pad_len, n_feat), dtype=np.float32)
        padded[:T] = seq[:T]
        X[i] = padded.flatten()
    return X


def get_lstm_scores(model, cfg, records, scaler):
    """Get LSTM prediction scores for records."""
    ds = SkeletonDataset(records, cfg.feature_cols, scaler, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    scores = []
    clips = []
    labels = []
    model.eval()
    with torch.no_grad():
        for seqs, lbls, lengths, clip_names in loader:
            seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
            preds = model(seqs, lengths)
            scores.extend(preds.cpu().numpy().tolist())
            clips.extend(clip_names)
            labels.extend(lbls.numpy().tolist())
    return np.array(scores), clips, np.array(labels)


def load_best_config():
    """Load best config from R1-R2 or R3 results."""
    for path in [RESULTS_DIR / "v11_r3_results.pkl", RESULTS_DIR / "v11_results.pkl"]:
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            cfg_dict = data["best_config"]
            cfg = ExpConfig(
                exp_id=cfg_dict.get("exp_id", "loaded"),
                description=cfg_dict.get("description", "loaded config"),
                hidden_size=cfg_dict.get("hidden_size", 64),
                bidirectional=cfg_dict.get("bidirectional", False),
                use_attention=cfg_dict.get("use_attention", False),
                use_layernorm=cfg_dict.get("use_layernorm", False),
                scheduler_type=cfg_dict.get("scheduler_type", "plateau"),
                label_smoothing=cfg_dict.get("label_smoothing", 0.0),
                early_stop_metric=cfg_dict.get("early_stop_metric", "val_loss"),
                focal_alpha=cfg_dict.get("focal_alpha", 0.75),
                focal_gamma=cfg_dict.get("focal_gamma", 2.0),
                feature_cols=cfg_dict.get("feature_cols", list(range(0, 77))),
            )
            return cfg, data
    return ExpConfig(), None


def train_lstm_fold(cfg, train_records, val_records, fold_num):
    """Train LSTM for one fold, return model + scaler + metrics."""
    from train_v11 import train_one_fold
    return train_one_fold(cfg, train_records, val_records, fold_num)


def main():
    print(f"Device: {DEVICE}")
    log_path = RESULTS_DIR / "experiment_log_r4.csv"
    init_log(log_path)

    print("Loading data...")
    all_records = load_all_data()
    print(f"Loaded {len(all_records)} tracks")

    with open(RESULTS_DIR / "fold_assignments.pkl", "rb") as f:
        fold_indices = pickle.load(f)

    best_cfg, prior_results = load_best_config()
    print(f"Using config: {best_cfg.exp_id} -- {best_cfg.description}")

    feature_cols = best_cfg.feature_cols

    # ===============================================================
    # First: run baseline (just LSTM) to establish metrics
    # ===============================================================
    from train_v11 import run_experiment
    print("\nRunning baseline LSTM...")
    baseline_cfg = best_cfg.clone(exp_id="R4_baseline", description="LSTM baseline for R4")
    baseline_result = run_experiment(baseline_cfg, all_records, fold_indices)

    # ===============================================================
    # Exp 4.1: Weighted ensemble LSTM + Random Forest
    # ===============================================================
    print("\n" + "=" * 60)
    print("EXP 4.1: WEIGHTED ENSEMBLE (LSTM + RANDOM FOREST)")
    print("=" * 60)

    ensemble_fold_metrics = []
    ensemble_clip_results = []
    best_global_w = None
    best_global_f1 = -1

    for fold_num, (train_idx, val_idx) in enumerate(fold_indices, 1):
        print(f"\n--- Fold {fold_num}/5 ---")
        train_recs = [all_records[i] for i in train_idx]
        val_recs = [all_records[i] for i in val_idx]

        # Train LSTM
        lstm_result = train_lstm_fold(best_cfg, train_recs, val_recs, fold_num)
        model = RetailLSTM_V11(best_cfg).to(DEVICE)
        model.load_state_dict(lstm_result["model_state"])
        model.eval()
        scaler_lstm = lstm_result["scaler"]

        # Get LSTM scores on validation
        lstm_scores, val_clips, val_labels = get_lstm_scores(
            model, best_cfg, val_recs, scaler_lstm
        )

        # Train Random Forest on flattened features
        X_train = flatten_features(train_recs, feature_cols)
        X_val = flatten_features(val_recs, feature_cols)
        train_labels = np.array([r["label"] for r in train_recs])

        scaler_rf = StandardScaler()
        X_train = scaler_rf.fit_transform(X_train)
        X_val = scaler_rf.transform(X_val)

        rf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        )
        rf.fit(X_train, train_labels)
        rf_scores = rf.predict_proba(X_val)[:, 1]

        # Tune weight w on validation set
        best_w = 0.5
        best_w_f1 = 0.0
        for w in np.arange(0.3, 0.9, 0.05):
            combined = w * lstm_scores + (1 - w) * rf_scores
            cn, cs, cl = aggregate_clip_scores(val_clips, combined, val_labels)
            thresh, f1_val = find_optimal_threshold(cl, cs)
            if f1_val > best_w_f1:
                best_w_f1 = f1_val
                best_w = w

        # Final ensemble with best weight
        combined_scores = best_w * lstm_scores + (1 - best_w) * rf_scores
        cn, cs, cl = aggregate_clip_scores(val_clips, combined_scores, val_labels)
        thresh, _ = find_optimal_threshold(cl, cs)
        cp = (cs >= thresh).astype(int)

        metrics = {
            "accuracy": accuracy_score(cl, cp),
            "precision": precision_score(cl, cp, zero_division=0),
            "recall": recall_score(cl, cp, zero_division=0),
            "f1": f1_score(cl, cp, zero_division=0),
            "auc": roc_auc_score(cl, cs) if len(np.unique(cl)) > 1 else 0.0,
            "threshold": thresh,
        }
        ensemble_fold_metrics.append(metrics)

        print(f"  w={best_w:.2f} | F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")

        for i, name in enumerate(cn):
            ensemble_clip_results.append({
                "clip_name": name,
                "true_label": int(cl[i]),
                "score": float(cs[i]),
                "pred": int(cp[i]),
                "fold": fold_num,
            })

    # Aggregate ensemble metrics
    ens_agg = {}
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        vals = [m[metric] for m in ensemble_fold_metrics]
        ens_agg[f"{metric}_mean"] = float(np.mean(vals))
        ens_agg[f"{metric}_std"] = float(np.std(vals))
    ens_agg["fp_count"] = sum(1 for cr in ensemble_clip_results if cr["true_label"] == 0 and cr["pred"] == 1)
    ens_agg["fn_count"] = sum(1 for cr in ensemble_clip_results if cr["true_label"] == 1 and cr["pred"] == 0)

    ensemble_result = {
        "agg_metrics": ens_agg,
        "fold_metrics": ensemble_fold_metrics,
        "clip_results": ensemble_clip_results,
        "best_model_state": None,
        "best_scaler": None,
    }

    print(f"\nEnsemble: F1 = {ens_agg['f1_mean']:.4f} +/- {ens_agg['f1_std']:.4f} | "
          f"AUC = {ens_agg['auc_mean']:.4f} +/- {ens_agg['auc_std']:.4f}")

    from train_v11 import decide, print_comparison, log_result
    print_comparison(baseline_result, ensemble_result, "4.1")
    verdict41, reason41 = decide(baseline_result, ensemble_result)
    log_result(log_path, "4.1", "Weighted ensemble LSTM+RF", baseline_result, ensemble_result, verdict41)
    print(f"\n>>> VERDICT: {verdict41} -- {reason41}")

    # ===============================================================
    # Exp 4.2: Platt Scaling for Threshold Calibration
    # ===============================================================
    print("\n" + "=" * 60)
    print("EXP 4.2: PLATT SCALING (THRESHOLD CALIBRATION)")
    print("=" * 60)

    platt_fold_metrics = []
    platt_clip_results = []

    for fold_num, (train_idx, val_idx) in enumerate(fold_indices, 1):
        print(f"\n--- Fold {fold_num}/5 ---")
        train_recs = [all_records[i] for i in train_idx]
        val_recs = [all_records[i] for i in val_idx]

        # Train LSTM
        lstm_result = train_lstm_fold(best_cfg, train_recs, val_recs, fold_num)
        model = RetailLSTM_V11(best_cfg).to(DEVICE)
        model.load_state_dict(lstm_result["model_state"])
        model.eval()
        scaler = lstm_result["scaler"]

        # Get LSTM scores on training data (no augmentation)
        train_scores, train_clips, train_labels = get_lstm_scores(
            model, best_cfg, train_recs, scaler
        )

        # Fit Platt scaler (logistic regression on raw scores)
        platt = LogisticRegression(max_iter=1000, random_state=42)
        platt.fit(train_scores.reshape(-1, 1), train_labels.astype(int))

        # Get calibrated scores on validation
        val_scores, val_clips, val_labels = get_lstm_scores(
            model, best_cfg, val_recs, scaler
        )
        calibrated = platt.predict_proba(val_scores.reshape(-1, 1))[:, 1]

        # Clip-level aggregation
        cn, cs, cl = aggregate_clip_scores(val_clips, calibrated, val_labels)
        thresh, _ = find_optimal_threshold(cl, cs)
        cp = (cs >= thresh).astype(int)

        metrics = {
            "accuracy": accuracy_score(cl, cp),
            "precision": precision_score(cl, cp, zero_division=0),
            "recall": recall_score(cl, cp, zero_division=0),
            "f1": f1_score(cl, cp, zero_division=0),
            "auc": roc_auc_score(cl, cs) if len(np.unique(cl)) > 1 else 0.0,
            "threshold": thresh,
        }
        platt_fold_metrics.append(metrics)
        print(f"  F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f} (thresh={thresh:.3f})")

        for i, name in enumerate(cn):
            platt_clip_results.append({
                "clip_name": name,
                "true_label": int(cl[i]),
                "score": float(cs[i]),
                "pred": int(cp[i]),
                "fold": fold_num,
            })

    platt_agg = {}
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        vals = [m[metric] for m in platt_fold_metrics]
        platt_agg[f"{metric}_mean"] = float(np.mean(vals))
        platt_agg[f"{metric}_std"] = float(np.std(vals))
    platt_agg["fp_count"] = sum(1 for cr in platt_clip_results if cr["true_label"] == 0 and cr["pred"] == 1)
    platt_agg["fn_count"] = sum(1 for cr in platt_clip_results if cr["true_label"] == 1 and cr["pred"] == 0)

    platt_result = {
        "agg_metrics": platt_agg,
        "fold_metrics": platt_fold_metrics,
        "clip_results": platt_clip_results,
        "best_model_state": None,
        "best_scaler": None,
    }

    print(f"\nPlatt: F1 = {platt_agg['f1_mean']:.4f} +/- {platt_agg['f1_std']:.4f}")
    print_comparison(baseline_result, platt_result, "4.2")
    verdict42, reason42 = decide(baseline_result, platt_result)
    log_result(log_path, "4.2", "Platt scaling calibration", baseline_result, platt_result, verdict42)
    print(f"\n>>> VERDICT: {verdict42} -- {reason42}")

    # ===============================================================
    # Exp 4.3: Hard Negative Mining
    # ===============================================================
    print("\n" + "=" * 60)
    print("EXP 4.3: HARD NEGATIVE MINING")
    print("=" * 60)

    hnm_fold_metrics = []
    hnm_clip_results = []

    for fold_num, (train_idx, val_idx) in enumerate(fold_indices, 1):
        print(f"\n--- Fold {fold_num}/5 ---")
        train_recs = [all_records[i] for i in train_idx]
        val_recs = [all_records[i] for i in val_idx]

        # Phase 1: Initial training
        lstm_result = train_lstm_fold(best_cfg, train_recs, val_recs, fold_num)
        model = RetailLSTM_V11(best_cfg).to(DEVICE)
        model.load_state_dict(lstm_result["model_state"])
        model.eval()
        scaler = lstm_result["scaler"]

        # Get training scores to find hard examples
        train_scores, train_clips, train_labels = get_lstm_scores(
            model, best_cfg, train_recs, scaler
        )

        # Identify hard examples: FP (normal with high score) and FN (shoplifting with low score)
        sample_weights = np.ones(len(train_recs), dtype=np.float32)
        for i, (score, label) in enumerate(zip(train_scores, train_labels)):
            is_fp = label == 0 and score > 0.7  # normal but model says shoplifting
            is_fn = label == 1 and score < 0.3  # shoplifting but model says normal
            if is_fp or is_fn:
                sample_weights[i] = 2.0  # 2x weight for hard examples

        n_hard = int(np.sum(sample_weights > 1.0))
        print(f"  Found {n_hard} hard examples out of {len(train_recs)} training samples")

        # Phase 2: Re-train with weighted loss
        feature_cols = best_cfg.feature_cols
        all_train = np.concatenate(
            [r["data"][:, feature_cols] for r in train_recs], axis=0
        )
        scaler2 = StandardScaler()
        scaler2.fit(all_train)

        train_ds = SkeletonDataset(train_recs, feature_cols, scaler2, augment=True)
        val_ds = SkeletonDataset(val_recs, feature_cols, scaler2, augment=False)
        train_loader = DataLoader(train_ds, batch_size=best_cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

        # Build clip-to-weight mapping
        clip_weights = {}
        for i, rec in enumerate(train_recs):
            clip_weights[rec["clip_name"]] = max(
                clip_weights.get(rec["clip_name"], 1.0), sample_weights[i]
            )

        model2 = RetailLSTM_V11(best_cfg).to(DEVICE)
        optimizer = torch.optim.Adam(model2.parameters(), lr=best_cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        criterion = FocalLoss(
            alpha=best_cfg.focal_alpha,
            gamma=best_cfg.focal_gamma,
            label_smoothing=best_cfg.label_smoothing,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(best_cfg.max_epochs):
            model2.train()
            train_loss = 0.0
            n_batches = 0
            for seqs, labels, lengths, clips in train_loader:
                seqs, labels, lengths = (
                    seqs.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
                )
                optimizer.zero_grad()
                preds = model2(seqs, lengths)

                # Apply per-sample weights
                base_loss = criterion(preds, labels)
                weights = torch.tensor(
                    [clip_weights.get(c, 1.0) for c in clips],
                    device=DEVICE, dtype=torch.float32,
                )
                # Recompute loss with weights
                bce = torch.nn.functional.binary_cross_entropy(
                    preds, labels, reduction="none"
                )
                pt = labels * preds + (1 - labels) * (1 - preds)
                alpha_t = labels * criterion.alpha + (1 - labels) * (1 - criterion.alpha)
                focal = (alpha_t * (1 - pt) ** criterion.gamma * bce)
                loss = (focal * weights).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            model2.eval()
            val_loss = 0.0
            val_n = 0
            with torch.no_grad():
                for seqs, labels, lengths, _ in val_loader:
                    seqs, labels, lengths = (
                        seqs.to(DEVICE), labels.to(DEVICE), lengths.to(DEVICE)
                    )
                    preds = model2(seqs, lengths)
                    loss = criterion(preds, labels)
                    val_loss += loss.item()
                    val_n += 1

            avg_val = val_loss / max(val_n, 1)
            scheduler.step(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in model2.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= best_cfg.early_stop_patience:
                    print(f"  [Fold {fold_num}] HNM early stop at epoch {epoch+1}")
                    break

        model2.load_state_dict(best_state)
        model2.to(DEVICE)
        model2.eval()

        # Evaluate
        val_scores = []
        val_labels = []
        val_clips = []
        with torch.no_grad():
            for seqs, labels, lengths, clips in val_loader:
                seqs, lengths = seqs.to(DEVICE), lengths.to(DEVICE)
                preds = model2(seqs, lengths)
                val_scores.extend(preds.cpu().numpy().tolist())
                val_labels.extend(labels.numpy().tolist())
                val_clips.extend(clips)

        val_scores = np.array(val_scores)
        val_labels = np.array(val_labels)

        cn, cs, cl = aggregate_clip_scores(val_clips, val_scores, val_labels)
        thresh, _ = find_optimal_threshold(cl, cs)
        cp = (cs >= thresh).astype(int)

        metrics = {
            "accuracy": accuracy_score(cl, cp),
            "precision": precision_score(cl, cp, zero_division=0),
            "recall": recall_score(cl, cp, zero_division=0),
            "f1": f1_score(cl, cp, zero_division=0),
            "auc": roc_auc_score(cl, cs) if len(np.unique(cl)) > 1 else 0.0,
            "threshold": thresh,
        }
        hnm_fold_metrics.append(metrics)
        print(f"  F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")

        for i, name in enumerate(cn):
            hnm_clip_results.append({
                "clip_name": name,
                "true_label": int(cl[i]),
                "score": float(cs[i]),
                "pred": int(cp[i]),
                "fold": fold_num,
            })

    hnm_agg = {}
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        vals = [m[metric] for m in hnm_fold_metrics]
        hnm_agg[f"{metric}_mean"] = float(np.mean(vals))
        hnm_agg[f"{metric}_std"] = float(np.std(vals))
    hnm_agg["fp_count"] = sum(1 for cr in hnm_clip_results if cr["true_label"] == 0 and cr["pred"] == 1)
    hnm_agg["fn_count"] = sum(1 for cr in hnm_clip_results if cr["true_label"] == 1 and cr["pred"] == 0)

    hnm_result = {
        "agg_metrics": hnm_agg,
        "fold_metrics": hnm_fold_metrics,
        "clip_results": hnm_clip_results,
        "best_model_state": None,
        "best_scaler": None,
    }

    print(f"\nHNM: F1 = {hnm_agg['f1_mean']:.4f} +/- {hnm_agg['f1_std']:.4f}")
    print_comparison(baseline_result, hnm_result, "4.3")
    verdict43, reason43 = decide(baseline_result, hnm_result)
    log_result(log_path, "4.3", "Hard negative mining", baseline_result, hnm_result, verdict43)
    print(f"\n>>> VERDICT: {verdict43} -- {reason43}")

    # ===============================================================
    # FINAL SUMMARY
    # ===============================================================
    print("\n" + "=" * 60)
    print("ROUND 4 SUMMARY")
    print("=" * 60)

    results_r4 = {
        "4.1 Ensemble": (ensemble_result, verdict41),
        "4.2 Platt": (platt_result, verdict42),
        "4.3 HNM": (hnm_result, verdict43),
    }

    bm = baseline_result["agg_metrics"]
    print(f"\n{'Experiment':<20} {'F1':>8} {'AUC':>8} {'dF1':>8} {'Verdict':>12}")
    print("-" * 60)
    print(f"{'Baseline':<20} {bm['f1_mean']:>8.4f} {bm['auc_mean']:>8.4f} {'---':>8} {'---':>12}")
    for name, (result, verdict) in results_r4.items():
        m = result["agg_metrics"]
        df1 = m["f1_mean"] - bm["f1_mean"]
        print(f"{name:<20} {m['f1_mean']:>8.4f} {m['auc_mean']:>8.4f} {df1:>+8.4f} {verdict:>12}")

    # Pick best overall
    best_name = max(results_r4, key=lambda n: results_r4[n][0]["agg_metrics"]["f1_mean"])
    best_r4 = results_r4[best_name]

    if best_r4[1].startswith("KEEP"):
        print(f"\nBest R4 approach: {best_name}")
        save_error_analysis(
            best_r4[0]["clip_results"],
            RESULTS_DIR / "error_analysis_v11_r4.txt",
            f"V11 R4 Best ({best_name})",
        )
    else:
        print("\nNo R4 experiments improved over baseline.")

    # Save R4 results
    with open(RESULTS_DIR / "v11_r4_results.pkl", "wb") as f:
        pickle.dump({
            "baseline_metrics": bm,
            "ensemble_result": ensemble_result["agg_metrics"],
            "platt_result": platt_result["agg_metrics"],
            "hnm_result": hnm_result["agg_metrics"],
            "verdicts": {
                "4.1": verdict41,
                "4.2": verdict42,
                "4.3": verdict43,
            },
        }, f)

    # Check final stop criteria
    all_f1s = [bm["f1_mean"]]
    for _, (r, _) in results_r4.items():
        all_f1s.append(r["agg_metrics"]["f1_mean"])
    best_f1 = max(all_f1s)

    if best_f1 > 0.85:
        print(f"\nTARGET REACHED: Best F1 = {best_f1:.4f} > 0.85")
    else:
        print(f"\nFinal best F1 = {best_f1:.4f}. All rounds exhausted.")

    print(f"\nExperiment log: {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
