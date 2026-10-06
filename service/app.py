# Run locally:  uvicorn service.app:app --port 8080

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from service.schemas import BatchRequest, BatchResponse, PredictRequest, PredictResponse

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}',
)
log = logging.getLogger("service")

STATE: dict[str, Any] = {"model": None, "version": os.environ.get("MODEL_VERSION", "unknown")}


def _load_model():
    """Load once, at startup. Never per request.

    Loading per request is the commonest cause of a p99 that looks nothing like p50, and
    it is the first thing to check when your latency distribution has a long tail.
    """
    name = os.environ.get("MODEL_REGISTRY_NAME")
    version = os.environ.get("MODEL_VERSION")
    if name and version:
        import mlflow.sklearn  # imported lazily so tests can run without a registry

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        return mlflow.sklearn.load_model(f"models:/{name}/{version}")

    # Fallback for local development and tests only.
    from pathlib import Path

    import joblib

    path = Path(os.environ.get("MODEL_PATH", "reports/model.joblib"))
    if not path.exists():
        raise RuntimeError(
            "No model available. Set MODEL_REGISTRY_NAME and MODEL_VERSION, or MODEL_PATH."
        )
    return joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["model"] = _load_model()
        log.info('"model loaded, version=%s"', STATE["version"])
    except Exception as exc:  # readiness stays false; liveness still passes
        STATE["model"] = None
        log.error('"model load failed: %s"', exc)
    yield
    STATE["model"] = None


app = FastAPI(title="ITCS355 wine quality project", version="0.0.1", lifespan=lifespan)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    started = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-model-version"] = str(STATE["version"])
    log.info(
        '{"request_id":"%s","path":"%s","status":%d,"latency_ms":%.2f,"model_version":"%s"}',
        request_id, request.url.path, response.status_code, latency_ms, STATE["version"],
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness. The process is up. Says nothing about whether it can serve."""
    return {"status": "alive"}


@app.get("/ready")
def ready():
    """Readiness. The model is loaded and can score."""
    if STATE["model"] is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": "model not loaded"})
    return {"status": "ready", "model_version": STATE["version"]}


def _score(rows: list[dict]) -> tuple[list[int], list[float]]:
    """Predicts continuous regression quality scores and clips them to valid ratings [3, 8]."""
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    import pandas as pd

    from src.data import FEATURES

    # Convert request row dicts to DataFrame ensuring correct column names and order
    frame = pd.DataFrame(rows)
    
    # Ensure column names map correctly if aliases with spaces are passed
    if not all(col in frame.columns for col in FEATURES):
        # Convert Pydantic model attribute names (underscores) to CSV feature names (spaces)
        rename_map = {col.replace(" ", "_"): col for col in FEATURES}
        frame = frame.rename(columns=rename_map)

    frame = frame[FEATURES]
    
    # Option B Regression predictions
    raw_preds = STATE["model"].predict(frame)
    
    # Post-processing: Round to nearest integer and clip to valid guardrails [3, 8]
    discrete_scores = np.clip(np.round(raw_preds), a_min=3, a_max=8).astype(int).tolist()
    raw_scores = [float(p) for p in raw_preds]

    return discrete_scores, raw_scores


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    discrete_scores, raw_scores = _score([payload.model_dump(by_alias=True)])
    return PredictResponse(
        quality_score=discrete_scores[0],
        raw_score=raw_scores[0],
        model_version=str(STATE["version"]),
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest) -> BatchResponse:
    rows = [row.model_dump(by_alias=True) for row in payload.rows]
    discrete_scores, raw_scores = _score(rows)
    return BatchResponse(
        quality_scores=discrete_scores,
        raw_scores=raw_scores,
        model_version=str(STATE["version"]),
    )
