"""
Master runner for V11 experiments.
Executes rounds in order, with natural stopping between rounds.

Usage:
    python src/run_all.py              # Run Rounds 1-2 (no re-extraction needed)
    python src/run_all.py --round 3    # Run Round 3 only (needs v11 features)
    python src/run_all.py --round 4    # Run Round 4 only (ensemble experiments)
    python src/run_all.py --extract    # Run V11 feature extraction first
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(r"C:\Users\malho\Desktop\claudeagent\RETAILPROJECT")
SRC_DIR = BASE_DIR / "src"


def run_script(script_name, description):
    print(f"\n{'#'*60}")
    print(f"# {description}")
    print(f"# Running: {script_name}")
    print(f"{'#'*60}\n")

    result = subprocess.run(
        [sys.executable, str(SRC_DIR / script_name)],
        cwd=str(BASE_DIR),
    )
    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed with code {result.returncode}")
        return False
    return True


def main():
    args = sys.argv[1:]
    round_num = None
    do_extract = False

    for i, arg in enumerate(args):
        if arg == "--round" and i + 1 < len(args):
            round_num = int(args[i + 1])
        if arg == "--extract":
            do_extract = True

    if do_extract:
        if not run_script("extract_features_v11.py", "V11 Extended Feature Extraction (158 dims)"):
            return

    if round_num is None or round_num <= 2:
        # Check V10 features exist
        v10_feat_dir = BASE_DIR / "v10_Features"
        n_files = sum(1 for _ in v10_feat_dir.rglob("*.npy")) if v10_feat_dir.exists() else 0
        if n_files == 0:
            print("ERROR: v10_Features/ is empty. Run extract_features.py first.")
            print("  python src/extract_features.py")
            return

        if not run_script("train_v11.py", "Rounds 1-2: Architecture + Training Optimization"):
            return

        if round_num is not None:
            return

    if round_num is None or round_num == 3:
        # Check V11 features exist
        v11_feat_dir = BASE_DIR / "v11_Features"
        n_files = sum(1 for _ in v11_feat_dir.rglob("*.npy")) if v11_feat_dir.exists() else 0
        if n_files == 0:
            print("ERROR: v11_Features/ is empty. Run with --extract first.")
            print("  python src/run_all.py --extract --round 3")
            return

        if not run_script("train_v11_r3.py", "Round 3: Feature Engineering"):
            return

        if round_num is not None:
            return

    if round_num is None or round_num == 4:
        if not run_script("ensemble.py", "Round 4: Ensemble & Error-Driven"):
            return

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"\nResults in: {BASE_DIR / 'v10_Results'}")
    print("Key files:")
    print("  experiment_log.csv     -- Round 1-2 log")
    print("  experiment_log_r3.csv  -- Round 3 log")
    print("  experiment_log_r4.csv  -- Round 4 log")
    print("  v11_results.pkl        -- Best R1-R2 model config+metrics")
    print("  v11_r3_results.pkl     -- Best R3 model config+metrics")
    print("  v11_r4_results.pkl     -- R4 results")
    print("  lstm_v11_best.pth      -- Best model weights")


if __name__ == "__main__":
    main()
