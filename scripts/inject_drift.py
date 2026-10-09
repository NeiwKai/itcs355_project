# Run: python scripts/inject_drift.py --feature "volatile acidity" --mode shift --magnitude 0.3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd

from src import data


def main() -> int:
    ap = argparse.ArgumentParser(description="Inject drift into red wine quality dataset.")
    ap.add_argument("--source", type=Path, default=Path("data/raw/winequality-red.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/current.csv"))
    ap.add_argument(
        "--feature",
        default="volatile acidity",
        help="Feature column to apply drift to",
    )
    ap.add_argument("--mode", choices=["shift", "scale", "mix"], default="shift")
    ap.add_argument(
        "--magnitude",
        type=float,
        default=0.3,
        help="Shift/scale magnitude",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if not args.source.exists():
        print(f"Error: Source file '{args.source}' does not exist. Run 'make data' first.")
        return 1

    # Load raw data using project loader
    df = data.load_raw(args.source)

    # Normalize feature names (handle space/underscore mismatches)
    feature_map = {col.strip().lower().replace("_", " "): col for col in df.columns}
    target_key = args.feature.strip().lower().replace("_", " ")

    if target_key not in feature_map:
        print(f"Error: Feature '{args.feature}' not found in dataset columns:")
        print(f"Available columns: {list(df.columns)}")
        return 1

    actual_col = feature_map[target_key]
    before = df[actual_col].mean()

    if args.mode == "shift":
        df[actual_col] = df[actual_col] + args.magnitude
    elif args.mode == "scale":
        mean = df[actual_col].mean()
        df[actual_col] = mean + (df[actual_col] - mean) * args.magnitude
    else:  # mix mode — over-sample lower quality wines
        if data.TARGET in df.columns:
            weights = np.where(df[data.TARGET] <= 5, 4.0, 1.0)
        else:
            q_high = df[actual_col].quantile(0.75)
            weights = np.where(df[actual_col] >= q_high, 4.0, 1.0)

        df = df.sample(n=len(df), replace=True, weights=weights, random_state=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, lineterminator="\n")

    print(f"✓ Wrote {args.out} | mode={args.mode} | feature='{actual_col}'")
    print(f"  mean {before:.4f} -> {df[actual_col].mean():.4f}")
    print(f"\nNow run: python -m monitoring.drift --current {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
