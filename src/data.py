from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "quality"

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

SCHEMA: dict[str, str] = {
    "fixed acidity": "float64",
    "volatile acidity": "float64",
    "citric acid": "float64",
    "residual sugar": "float64",
    "chlorides": "float64",
    "free sulfur dioxide": "float64",
    "total sulfur dioxide": "float64",
    "density": "float64",
    "pH": "float64",
    "sulphates": "float64",
    "alcohol": "float64",
    TARGET: "int64",
}

PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "fixed acidity": (4.50, 16.00),
    "volatile acidity": (0.10, 1.60),
    "citric acid": (0.00, 1.00),
    "residual sugar": (0.50, 15.50),
    "chlorides": (0.010, 0.615),
    "free sulfur dioxide": (0.0, 75.0),
    "total sulfur dioxide": (5.0, 290.0),
    "density": (0.990, 1.005),
    "pH": (2.70, 4.05),
    "sulphates": (0.30, 2.00),
    "alcohol": (8.00, 15.00),
}


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make data` first, or `dvc pull` if the remote is configured."
        )
    return pd.read_csv(path)


def data_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def split(
    df: pd.DataFrame,
    seed: int,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataframe into Train, Validation, and Test sets.
    
    Uses stratified sampling on the TARGET column to maintain quality rating distributions.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    # 1. First split off the test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_frac,
        random_state=seed,
        stratify=df[TARGET] if TARGET in df else None,
    )

    # 2. Adjust val_frac relative to the remaining train_val subset
    relative_val_frac = val_frac / (1.0 - test_frac)

    # 3. Split remaining into train and validation sets
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_frac,
        random_state=seed,
        stratify=train_val_df[TARGET] if TARGET in train_val_df else None,
    )

    parts = (train_df, val_df, test_df)
    if any(len(p) == 0 for p in parts):
        raise ValueError("A split partition is empty.")

    return parts
