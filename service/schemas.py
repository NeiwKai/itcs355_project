"""Request and response contracts for Wine Quality Inference Service.

Validation lives here so a malformed request is rejected with 422 and a useful message,
rather than reaching the model and producing a confident number from nonsense.

The bounds mirror src/data.PLAUSIBLE_RANGES. Keep them in step.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    fixed_acidity: float = Field(..., ge=4.50, le=16.00, alias="fixed acidity")
    volatile_acidity: float = Field(..., ge=0.10, le=1.60, alias="volatile acidity")
    citric_acid: float = Field(..., ge=0.00, le=1.00, alias="citric acid")
    residual_sugar: float = Field(..., ge=0.50, le=15.50, alias="residual sugar")
    chlorides: float = Field(..., ge=0.010, le=0.615, alias="chlorides")
    free_sulfur_dioxide: float = Field(..., ge=0.0, le=75.0, alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., ge=5.0, le=290.0, alias="total sulfur dioxide")
    density: float = Field(..., ge=0.990, le=1.005, alias="density")
    pH: float = Field(..., ge=2.70, le=4.05, alias="pH")
    sulphates: float = Field(..., ge=0.30, le=2.00, alias="sulphates")
    alcohol: float = Field(..., ge=8.00, le=15.00, alias="alcohol")

    # Use 'pad' as the Python attribute and alias='_pad' for JSON payload parsing
    pad: Optional[str] = Field(default=None, alias="_pad")

    model_config = {
        "extra": "forbid",
        "populate_by_name": True,  # Allows accepting both alias (with spaces) and python attribute names
    }


class PredictResponse(BaseModel):
    quality_score: int = Field(..., ge=3, le=8, description="Predicted discrete quality rating (3-8)")
    raw_score: float = Field(..., description="Raw continuous regression model output")
    model_version: str


class BatchRequest(BaseModel):
    rows: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchResponse(BaseModel):
    quality_scores: list[int]
    raw_scores: list[float]
    model_version: str
