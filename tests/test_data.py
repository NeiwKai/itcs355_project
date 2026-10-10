from __future__ import annotations

import pytest
import pandas as pd

from src import config, data

RAW = config.REPO_ROOT / "data" / "raw" / "winequality-red.csv"
CURRENT = config.REPO_ROOT / "data" / "current.csv"


@pytest.fixture(scope="module")
def df():
    if not RAW.exists():
        pytest.skip("data/raw/winequality-red.csv missing — run `make data` or `dvc pull` first")
    return data.load_raw(RAW)


# --- contract tests: about the data -------------------------------------------------

def test_schema_columns_present_and_typed(df):
    missing = set(data.SCHEMA) - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"
    unexpected = set(df.columns) - set(data.SCHEMA)
    assert not unexpected, f"unexpected columns: {sorted(unexpected)}"
    for col, expected in data.SCHEMA.items():
        assert str(df[col].dtype) == expected, f"{col}: expected {expected}, got {df[col].dtype}"


def test_no_nulls_in_required_columns(df):
    nulls = df[list(data.SCHEMA)].isna().sum()
    offenders = nulls[nulls > 0]
    assert offenders.empty, f"null values found: {offenders.to_dict()}"


def test_features_within_plausible_ranges(df):
    for col, (lo, hi) in data.PLAUSIBLE_RANGES.items():
        assert df[col].min() >= lo, f"{col} below plausible floor: {df[col].min()}"
        assert df[col].max() <= hi, f"{col} above plausible ceiling: {df[col].max()}"


def test_volatile_acidity_no_unphysical_drift_or_corruption(df):
    """CL Test (Incident Ref: 2026-10-10 Post-Mortem):

    Validates that volatile acidity strictly adheres to physical chemical bounds [0.00, 2.60].
    Fails immediately if upstream ingestion injects unphysical shifts or corrupted features.
    """
    # 1. Test dataset fixture
    lo, hi = data.PLAUSIBLE_RANGES.get("volatile acidity")
    # Ensure lower floor is strictly physical (>= 0.00)
    assert lo >= 0.00, f"Plausible range floor for volatile acidity must be >= 0.00, got {lo}"

    va = df["volatile acidity"]
    offenders = df[(va < lo) | (va > hi)]
    assert offenders.empty, (
        f"Data contract breach: Found {len(offenders)} rows with volatile acidity outside [{lo}, {hi}]. "
        f"Min = {va.min():.4f}, Max = {va.max():.4f}"
    )

    # 2. Also validate live inference dataset if present (data/current.csv)
    if CURRENT.exists():
        current_df = pd.read_csv(CURRENT)
        if "volatile acidity" in current_df.columns:
            cur_va = current_df["volatile acidity"]
            cur_offenders = current_df[(cur_va < lo) | (cur_va > hi)]
            assert cur_offenders.empty, (
                f"Production batch data contract breach in {CURRENT.name}: "
                f"Found {len(cur_offenders)} rows outside [{lo}, {hi}]. "
                f"Min = {cur_va.min():.4f}, Max = {cur_va.max():.4f}. "
                "Possible corrupted upstream pipeline or artificial drift injection."
            )


def test_target_is_valid_scale_and_not_degenerate(df):
    """Asserts that the target variable falls within the valid range [0, 10]

    and exhibits non-zero variance across the dataset.
    """
    target = df[data.TARGET]
    
    # 1. Assert no missing or NaN target values
    assert not target.isna().any(), "Target contains missing or NaN values."
    
    # 2. Assert target values are within valid sensory bounds [0, 10]
    values = set(target.unique().tolist())
    assert all(0 <= v <= 10 for v in values), f"Target has values outside [0, 10]: {values}"
    
    # 3. Assert target is integer-like rating scale
    assert pd.api.types.is_numeric_dtype(target), f"Target must be numeric, got {target.dtype}"
    
    # 4. Check for degeneracy: target must have more than 1 unique score and non-zero variance
    assert len(values) > 1, f"Target is degenerate with only 1 unique value: {values}"
    assert target.std() > 0.1, f"Target lacks sufficient variance (std = {target.std():.4f})"


# --- property tests: about the splitting logic --------------------------------------


def test_split_is_deterministic_given_seed(df):
    """Asserts that splitting with the same seed yields identical DataFrame indices."""
    a = data.split(df, seed=7)
    b = data.split(df, seed=7)
    for pa, pb in zip(a, b):
        assert pa.index.tolist() == pb.index.tolist()


def test_split_changes_with_seed(df):
    """Asserts that splitting with different seeds produces different row index orderings."""
    a, _, _ = data.split(df, seed=1)
    b, _, _ = data.split(df, seed=2)
    assert (
        a.index.tolist() != b.index.tolist()
    ), "seed has no effect — split is not random"


def test_every_row_lands_in_exactly_one_split(df):
    train, val, test = data.split(df, seed=13)
    total = len(train) + len(val) + len(test)
    assert total == len(df), f"rows lost or duplicated: {total} vs {len(df)}"


def test_data_fingerprint_is_stable():
    if not RAW.exists():
        pytest.skip("no raw data")
    assert data.data_fingerprint(RAW) == data.data_fingerprint(RAW)
