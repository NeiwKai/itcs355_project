# Run locally:      python -m src.train --C 1.0 --epsilon 0.1
# Run in Docker:    make reproduce

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
)

from src import config, data, seeds


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=config.REPO_ROOT,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ITCS355 Lab 1 — reproducible training")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=seeds.DEFAULT_SEED)
    p.add_argument("--experiment", default="itcs355-proj-train")
    p.add_argument("--run-name", default=None)
    p.add_argument("--metrics-out", type=Path, default=None,
                   help="Write final metrics as JSON. Used by `make verify`.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = config.load(strict=False)
    seed = seeds.set_all(args.seed)

    df = data.load_raw(cfg.raw_path)
    fingerprint = data.data_fingerprint(cfg.raw_path)
    train_df, val_df, test_df = data.split(df, seed=seed)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "C": args.C,
            "epsilon": args.epsilon,
            "seed": seed,
            "n_features": len(data.FEATURES)
        })

        mlflow.set_tags({
            "git_commit": git_commit(),
            "data_fingerprint": fingerprint,
            "n_train_rows": len(train_df),
            "n_val_rows": len(val_df),
            "n_test_rows": len(test_df)
        })

        model = make_pipeline(
            RobustScaler(),
            SVR(kernel="rbf", C=1.0, epsilon=0.1)
        )
        model.fit(train_df[data.FEATURES], train_df[data.TARGET])

        metrics: dict[str, float] = {}
        for name, part in (("val", val_df), ("test", test_df)):
            y_ground = part[data.TARGET]
            y_pred_cont = model.predict(part[data.FEATURES])

            # Post-processing: Round to discrete ratings and clip to valid guardrails (3 to 8)
            y_pred_discrete = np.clip(np.round(y_pred_cont), 3, 8).astype(int)

            # Log Regression Metrics
            metrics[f"{name}_mae"] = mean_absolute_error(y_ground, y_pred_cont)
            metrics[f"{name}_rmse"] = np.sqrt(mean_squared_error(y_ground, y_pred_cont))
            metrics[f"{name}_r2"] = r2_score(y_ground, y_pred_cont)

            # Log Discrete Metrics
            metrics[f"{name}_exact_auc"] = accuracy_score(y_ground, y_pred_discrete)
            metrics[f"{name}_within_1_auc"] = (np.abs(y_ground - y_pred_discrete) <= 1).mean()
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(
            model, name="itcs355-proj-6688101-6688132-model",
        )

        print(json.dumps({"seed": seed, "data_fingerprint": fingerprint, **metrics}, indent=2))
        if args.metrics_out:
            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_out.write_text(json.dumps(
                {"seed": seed, "data_fingerprint": fingerprint, **metrics}, indent=2))




if __name__ == '__main__':
    main()
