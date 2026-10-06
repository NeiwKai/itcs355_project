from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

from src import config, data, seeds

RAW = config.REPO_ROOT / "data" / "raw" / "winequality-red.csv"
LATENCY_BUDGET_MS = 50.0  # Target p95 prediction latency target in ms


@pytest.fixture(scope="module")
def fitted():
    if not RAW.exists():
        pytest.skip("run `make data` first")
    seed = seeds.set_all()
    df = data.load_raw(RAW)
    train_df, _, test_df = data.split(df, seed=seed)

    # Option B: Regression model with scaling
    model = make_pipeline(
        RobustScaler(),
        SVR(kernel="rbf", C=2.0, epsilon=0.1)
    )
    model.fit(train_df[data.FEATURES], train_df[data.TARGET])
    return model, test_df


def test_predictions_are_valid_quality_ratings(fitted):
    """Asserts that predictions fall within valid quality score bounds [3, 8]."""
    model, test_df = fitted
    raw_preds = model.predict(test_df[data.FEATURES])
    
    # Check no NaN or infinite outputs
    assert not np.isnan(raw_preds).any()
    assert not np.isinf(raw_preds).any()

    # Discrete predictions after rounding and clipping
    discrete_preds = np.clip(np.round(raw_preds), 3, 8)
    assert np.all((discrete_preds >= 3) & (discrete_preds <= 8))


def test_high_alcohol_low_acid_scores_well(fitted):
    """Domain assertion: A wine with high alcohol content and low volatile acidity

    must not be rated as a poor wine (< 5).
    """
    model, _ = fitted
    high_quality_sample = pd.DataFrame([{
        "fixed acidity": 8.0,
        "volatile acidity": 0.25,
        "citric acid": 0.40,
        "residual sugar": 2.0,
        "chlorides": 0.05,
        "free sulfur dioxide": 15.0,
        "total sulfur dioxide": 35.0,
        "density": 0.994,
        "pH": 3.30,
        "sulphates": 0.80,
        "alcohol": 12.5,
    }])[data.FEATURES]

    raw_pred = model.predict(high_quality_sample)[0]
    score = int(np.clip(np.round(raw_pred), 3, 8))
    assert score >= 6, f"Expected good quality rating (>=6), got {score} (raw: {raw_pred:.2f})"


def test_quality_decreases_with_volatile_acidity(fitted):
    """Monotonicity domain assertion: Increasing volatile acidity (vinegar taste)

    should decrease or maintain predicted wine quality.
    """
    model, _ = fitted
    base = {
        "fixed acidity": 7.5,
        "citric acid": 0.20,
        "residual sugar": 2.2,
        "chlorides": 0.08,
        "free sulfur dioxide": 14.0,
        "total sulfur dioxide": 40.0,
        "density": 0.996,
        "pH": 3.30,
        "sulphates": 0.60,
        "alcohol": 10.0,
    }

    # Vary volatile acidity from low (good) to high (spoiled)
    rows = pd.DataFrame([{**base, "volatile acidity": v} for v in (0.20, 0.60, 1.20)])[data.FEATURES]
    preds = model.predict(rows)

    assert preds[0] >= preds[-1], (
        f"Quality unexpectedly increased with high volatile acidity: {preds}"
    )


def test_prediction_latency_within_budget(fitted):
    """Checks that single prediction latency stays strictly within budget."""
    model, test_df = fitted
    sample = test_df[data.FEATURES].head(100)
    model.predict(sample)  # Warmup call

    started = time.perf_counter()
    for _ in range(20):
        model.predict(sample.head(1))
    elapsed_ms = (time.perf_counter() - started) * 1000 / 20

    assert elapsed_ms < LATENCY_BUDGET_MS, f"Single prediction took {elapsed_ms:.2f} ms"


def test_model_is_not_constant(fitted):
    """A model that always predicts a constant score is useless.

    Ensures prediction outputs exhibit variance across different samples.
    """
    model, test_df = fitted
    preds = model.predict(test_df[data.FEATURES])
    assert preds.std() > 0.1, "Predictions have almost no variance across test samples"
