"""
V11 Round 3: Feature Engineering Experiments
Requires v11_Features/ (158-dim) from extract_features_v11.py.
Loads best config from Rounds 1-2 as baseline, then tests new feature additions.
Uses same fold_assignments.pkl from V10.

Feature layout in v11_Features/*.npy (158 dims):
  [ 0: 85] V10 features
  [85:102] Keypoint confidences (17)    [Exp 3.1]
  [102:104] Soft cross-body (2)         [Exp 3.2 - replaces binary at 75-76]
  [104:140] Acceleration (36)           [Exp 3.3]
  [140:157] Motion magnitude (17)       [Exp 3.4]
  [157:158] Hand velocity correlation   [Exp 3.5]
"""

import sys, pickle, csv
from pathlib import Path

# Ensure src/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_v11 import (
    ExpConfig, run_experiment, load_all_data, decide, print_comparison,
    log_result, init_log, build_combined_config, save_error_analysis,
    RESULTS_DIR, DEVICE,
)
import train_v11
import torch
import warnings
warnings.filterwarnings("ignore")

# Override feature dir to use v11 extended features
BASE_DIR = Path(r"C:\Users\malho\Desktop\claudeagent\RETAILPROJECT")
train_v11.FEATURE_DIR = BASE_DIR / "v11_Features"

TIER_C_COLS = list(range(0, 77))  # V10 Tier C


def main():
    print(f"Device: {DEVICE}")
    log_path = RESULTS_DIR / "experiment_log_r3.csv"
    init_log(log_path)

    print("Loading V11 extended features...")
    all_records = load_all_data()
    print(f"Loaded {len(all_records)} tracks")
    if all_records:
        print(f"Feature dims: {all_records[0]['data'].shape[1]}")

    with open(RESULTS_DIR / "fold_assignments.pkl", "rb") as f:
        fold_indices = pickle.load(f)

    # Load best config from Rounds 1-2
    r12_results_path = RESULTS_DIR / "v11_results.pkl"
    if r12_results_path.exists():
        with open(r12_results_path, "rb") as f:
            r12 = pickle.load(f)
        best_r12_config = r12["best_config"]
        print(f"Using R1-R2 best config: {best_r12_config['exp_id']}")

        base_cfg = ExpConfig(
            exp_id="R12_baseline",
            description=f"R1-R2 best on V11 features ({best_r12_config['exp_id']})",
            hidden_size=best_r12_config["hidden_size"],
            bidirectional=best_r12_config["bidirectional"],
            use_attention=best_r12_config["use_attention"],
            use_layernorm=best_r12_config["use_layernorm"],
            scheduler_type=best_r12_config["scheduler_type"],
            label_smoothing=best_r12_config["label_smoothing"],
            early_stop_metric=best_r12_config["early_stop_metric"],
            focal_alpha=best_r12_config["focal_alpha"],
            focal_gamma=best_r12_config["focal_gamma"],
            feature_cols=TIER_C_COLS,
        )
    else:
        print("No R1-R2 results found; using V10 baseline config")
        base_cfg = ExpConfig(
            exp_id="R12_baseline",
            description="V10 baseline config on V11 features",
            feature_cols=TIER_C_COLS,
        )

    # Run baseline on V11 features with Tier C cols
    baseline_result = run_experiment(base_cfg, all_records, fold_indices)
    log_result(log_path, base_cfg.exp_id, base_cfg.description,
               baseline_result, baseline_result, "BASELINE")

    current_best_cfg = base_cfg
    current_best_result = baseline_result

    # ===================================================================
    # ROUND 3 EXPERIMENTS
    # ===================================================================
    print("\n" + "#" * 60)
    print("# ROUND 3: FEATURE ENGINEERING")
    print("#" * 60)

    r3_experiments = [
        # Exp 3.1: Tier C + keypoint confidences
        base_cfg.clone(
            exp_id="3.1",
            description="Add 17 keypoint confidence scores",
            feature_cols=TIER_C_COLS + list(range(85, 102)),
        ),
        # Exp 3.2: Tier C with soft cross-body replacing binary
        # Replace cols 75-76 (binary cross-body) with 102-103 (soft cross-body)
        base_cfg.clone(
            exp_id="3.2",
            description="Replace binary cross-body with soft continuous",
            feature_cols=list(range(0, 75)) + list(range(102, 104)),
        ),
        # Exp 3.3: Tier C + acceleration
        base_cfg.clone(
            exp_id="3.3",
            description="Add 36 acceleration features (2nd-order velocity)",
            feature_cols=TIER_C_COLS + list(range(104, 140)),
        ),
        # Exp 3.4: Tier C + motion magnitude
        base_cfg.clone(
            exp_id="3.4",
            description="Add 17 motion magnitude features",
            feature_cols=TIER_C_COLS + list(range(140, 157)),
        ),
        # Exp 3.5: Tier C + hand velocity correlation
        base_cfg.clone(
            exp_id="3.5",
            description="Add hand velocity correlation",
            feature_cols=TIER_C_COLS + [157],
        ),
    ]

    r3_kept = []
    for exp_cfg in r3_experiments:
        result = run_experiment(exp_cfg, all_records, fold_indices)
        print_comparison(baseline_result, result, exp_cfg.exp_id)
        verdict, reason = decide(baseline_result, result)
        log_result(log_path, exp_cfg.exp_id, exp_cfg.description,
                   baseline_result, result, verdict)
        print(f"\n>>> VERDICT: {verdict} -- {reason}")

        if verdict.startswith("KEEP"):
            r3_kept.append((exp_cfg, result))

    # Consolidate Round 3
    if r3_kept:
        print(f"\n{'='*60}")
        print(f"ROUND 3 CONSOLIDATION: {len(r3_kept)} kept changes")
        for cfg, _ in r3_kept:
            print(f"  - {cfg.exp_id}: {cfg.description}")
        print(f"{'='*60}")

        # Merge feature columns from all kept experiments
        all_cols = set(TIER_C_COLS)
        remove_cols = set()
        for cfg, _ in r3_kept:
            exp_cols = set(cfg.feature_cols)
            # Check for replacement experiments (3.2 removes 75-76)
            if cfg.exp_id == "3.2":
                remove_cols.update([75, 76])
            all_cols.update(exp_cols)

        all_cols -= remove_cols
        combined_cols = sorted(all_cols)

        combined_cfg = base_cfg.clone(
            exp_id="R3_combined",
            description="Round 3 all kept features combined",
            feature_cols=combined_cols,
        )

        combined_result = run_experiment(combined_cfg, all_records, fold_indices)
        print_comparison(baseline_result, combined_result, "R3_combined")
        verdict, reason = decide(baseline_result, combined_result)
        log_result(log_path, "R3_combined", combined_cfg.description,
                   baseline_result, combined_result, verdict)
        print(f"\n>>> CONSOLIDATION VERDICT: {verdict} -- {reason}")

        if verdict.startswith("KEEP"):
            current_best_cfg = combined_cfg
            current_best_result = combined_result
            print(">>> R3 combined model is new best")
        else:
            best_single = max(r3_kept, key=lambda x: x[1]["agg_metrics"]["f1_mean"])
            v, r = decide(baseline_result, best_single[1])
            if v.startswith("KEEP"):
                current_best_cfg = best_single[0]
                current_best_result = best_single[1]
                print(f">>> Fallback: using single best ({best_single[0].exp_id})")
    else:
        print("\nNo Round 3 changes kept.")

    # Save final R3 results
    torch.save(current_best_result["best_model_state"], RESULTS_DIR / "lstm_v11_r3_best.pth")
    with open(RESULTS_DIR / "best_scaler_v11_r3.pkl", "wb") as f:
        pickle.dump(current_best_result["best_scaler"], f)

    save_error_analysis(
        current_best_result["clip_results"],
        RESULTS_DIR / "error_analysis_v11_r3.txt",
        f"V11 R3 Best ({current_best_cfg.exp_id})",
    )

    with open(RESULTS_DIR / "v11_r3_results.pkl", "wb") as f:
        pickle.dump({
            "best_config": {
                "exp_id": current_best_cfg.exp_id,
                "description": current_best_cfg.description,
                "feature_cols": current_best_cfg.feature_cols,
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

    # Summary
    m = current_best_result["agg_metrics"]
    bm = baseline_result["agg_metrics"]
    print("\n" + "=" * 60)
    print("ROUND 3 SUMMARY")
    print("=" * 60)
    print(f"Best config: {current_best_cfg.exp_id}")
    print(f"Feature dims: {len(current_best_cfg.feature_cols)}")
    print(f"\n{'Metric':<12} {'R1-R2 Base':>12} {'R3 Best':>12} {'Delta':>12}")
    print("-" * 50)
    for met in ["f1", "auc", "accuracy", "precision", "recall"]:
        bv = bm[f"{met}_mean"]
        ev = m[f"{met}_mean"]
        print(f"{met:<12} {bv:>12.4f} {ev:>12.4f} {ev-bv:>+12.4f}")
    print(f"{'f1_std':<12} {bm['f1_std']:>12.4f} {m['f1_std']:>12.4f}")

    if m["f1_mean"] > 0.85 and m["f1_std"] < 0.05:
        print("\nSTOP CRITERIA MET: F1 > 0.85 with std < 0.05")
    else:
        print(f"\nTarget not yet reached (F1={m['f1_mean']:.4f}). Proceed to Round 4 (ensemble).")

    print(f"\nExperiment log: {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
