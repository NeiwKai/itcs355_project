# Export a trained model to a local file
# Run: python -m scripts.export_model --C 1.0 --epsilon 0.1

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

from src import config, data, seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, required=True)
    ap.add_argument("--epsilon", type=float, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/model.joblib"))
    ap.add_argument("--seed", type=int, default=seeds.DEFAULT_SEED)
    args = ap.parse_args()

    cfg = config.load(strict=False)
    seed = seeds.set_all(args.seed)
    df = data.load_raw(cfg.raw_path)
    train_df, _, _ = data.split(df, seed=seed)

    model = make_pipeline(
        RobustScaler(),
        SVR(kernel="rbf", C=float(args.C), epsilon=float(args.epsilon))
    )
    model.fit(train_df[data.FEATURES], train_df[data.TARGET])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
