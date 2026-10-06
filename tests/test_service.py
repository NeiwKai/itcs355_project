from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src import config, data, seeds  # noqa: E402

RAW = config.REPO_ROOT / "data" / "raw" / "winequality-red.csv"

# Valid sample matching red wine dataset features and PLAUSIBLE_RANGES
VALID = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


@pytest.fixture(scope="module")
def client():
    if not RAW.exists():
        pytest.skip("run `make data` first")
    model_path = config.REPO_ROOT / "reports" / "model.joblib"
    if not model_path.exists():
        import joblib
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import RobustScaler
        from sklearn.svm import SVR

        seed = seeds.set_all()
        df = data.load_raw(RAW)
        train_df, _, _ = data.split(df, seed=seed)

        # Build fallback model matching Option B SVR setup
        model = make_pipeline(
            RobustScaler(),
            SVR(kernel="rbf", C=2.0, epsilon=0.1)
        )
        model.fit(train_df[data.FEATURES], train_df[data.TARGET])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["MODEL_VERSION"] = "test-1"
    os.environ.pop("MODEL_REGISTRY_NAME", None)

    from service.app import app
    with TestClient(app) as c:
        yield c


def test_health_and_ready_are_different(client):
    """Liveness passes whenever the process is up. Readiness only when it can score."""
    assert client.get("/health").json()["status"] == "alive"
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_predict_returns_quality_score_and_version(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    
    # Verify quality rating is within [3, 8]
    assert 3 <= body["quality_score"] <= 8
    assert isinstance(body["quality_score"], int)
    assert isinstance(body["raw_score"], float)
    
    assert body["model_version"] == "test-1"
    assert r.headers["x-model-version"] == "test-1"
    assert r.headers["x-request-id"]


def test_out_of_range_input_is_rejected(client):
    # alcohol exceeds max bound (le=15.0)
    bad = {**VALID, "alcohol": 25.0}
    assert client.post("/predict", json=bad).status_code == 422


def test_unknown_field_is_rejected(client):
    """extra='forbid'. A silently ignored field is how a caller ends up sending a feature

    you never read while believing it matters.
    """
    assert client.post("/predict", json={**VALID, "surprise": 1}).status_code == 422


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in VALID.items() if k != "alcohol"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_batch_matches_singles(client):
    rows = [VALID, {**VALID, "alcohol": 12.0}]
    
    batch_res = client.post("/predict/batch", json={"rows": rows}).json()
    batch_quality = batch_res["quality_scores"]
    batch_raw = batch_res["raw_scores"]
    
    singles_res = [client.post("/predict", json=r).json() for r in rows]
    singles_quality = [r["quality_score"] for r in singles_res]
    singles_raw = [r["raw_score"] for r in singles_res]
    
    assert batch_quality == singles_quality
    assert batch_raw == pytest.approx(singles_raw, abs=1e-9)


def test_batch_size_limit_enforced(client):
    r = client.post("/predict/batch", json={"rows": [VALID] * 101})
    assert r.status_code == 422
