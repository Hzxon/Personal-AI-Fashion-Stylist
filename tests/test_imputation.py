"""
Unit tests for the O4U imputation pipeline.

Requirements: 32.2
"""

import json
import math
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.o4u_imputation_pipeline import run_pipeline, PHYSICAL_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_raw_json(df: pd.DataFrame, path: Path) -> None:
    """
    Write a DataFrame as a raw-format JSON file (list of records with 'id'
    and physical attribute columns).  The 'id' column is derived from
    'outfit_id' if present.
    """
    raw_cols = ["id"] + [c for c in PHYSICAL_COLS if c in df.columns]
    out = df.copy()
    if "outfit_id" in out.columns and "id" not in out.columns:
        out = out.rename(columns={"outfit_id": "id"})
    out["id"] = out["id"].astype(str)
    records = out[raw_cols].where(pd.notnull(out[raw_cols]), None).to_dict(orient="records")
    with open(path, "w") as f:
        json.dump(records, f)


def _write_manifest_json(df: pd.DataFrame, path: Path) -> None:
    """
    Write a DataFrame as a manifest-format JSON file (list of records with
    'id', 'score', 'binary_label').
    """
    out = df.copy()
    if "outfit_id" in out.columns and "id" not in out.columns:
        out = out.rename(columns={"outfit_id": "id"})
    out["id"] = out["id"].astype(str)
    manifest_cols = [c for c in ["id", "score", "binary_label"] if c in out.columns]
    records = out[manifest_cols].to_dict(orient="records")
    with open(path, "w") as f:
        json.dump(records, f)


def _run_pipeline_on_fixtures(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tmp_path: Path,
) -> tuple[list[dict], list[dict]]:
    """
    Persist fixtures to temp JSON files, run the imputation pipeline, and
    return the resulting train and val imputed manifests as lists of dicts.
    """
    train_raw = tmp_path / "train_raw.json"
    val_raw = tmp_path / "val_raw.json"
    train_manifest = tmp_path / "train_manifest.json"
    val_manifest = tmp_path / "val_manifest.json"
    train_out = tmp_path / "train_imputed.json"
    val_out = tmp_path / "val_imputed.json"

    _write_raw_json(train_df, train_raw)
    _write_raw_json(val_df, val_raw)
    _write_manifest_json(train_df, train_manifest)
    _write_manifest_json(val_df, val_manifest)

    run_pipeline(
        train_raw=str(train_raw),
        val_raw=str(val_raw),
        train_manifest=str(train_manifest),
        val_manifest=str(val_manifest),
        train_out=str(train_out),
        val_out=str(val_out),
        n_imputations=1,
    )

    with open(train_out) as f:
        train_records = json.load(f)
    with open(val_out) as f:
        val_records = json.load(f)

    return train_records, val_records


# ---------------------------------------------------------------------------
# Physical feature columns that must be NaN-free after imputation
# ---------------------------------------------------------------------------

# One-hot encoded body-figure columns produced from the fixture data
_BF_COLS = ["bf_apple", "bf_pear", "bf_hourglass", "bf_rectangle"]

# One-hot encoded categorical columns produced from the fixture data
_CAT_ONE_HOT_PREFIXES = [
    "skin_color_",
    "hair_style_",
    "hair_color_",
    "height_",
    "breasts_",
    "color_contrast_",
]

# Imputation indicator columns
_INDICATOR_COLS = [
    "height_was_imputed",
    "breasts_was_imputed",
    "skin_color_was_imputed",
    "color_contrast_was_imputed",
    "hair_style_was_imputed",
    "hair_color_was_imputed",
]


def _physical_cols_in_record(record: dict) -> list[str]:
    """Return all physical feature column names present in a manifest record."""
    cols = []
    for key in record:
        if key.startswith("bf_"):
            cols.append(key)
        elif any(key.startswith(p) for p in _CAT_ONE_HOT_PREFIXES):
            cols.append(key)
        elif key in _INDICATOR_COLS:
            cols.append(key)
    return cols


# ---------------------------------------------------------------------------
# Test 1: No NaN in physical feature columns after imputation
# ---------------------------------------------------------------------------

def test_no_nan_in_physical_columns_after_imputation(
    sample_train_df, sample_val_df, tmp_path
):
    """
    After running the imputation pipeline on sample_train_df / sample_val_df,
    the output manifest must contain no NaN (None) values in any physical
    feature column (body-figure one-hots, categorical one-hots, and
    _was_imputed indicators).

    Validates: Requirements 32.2, 11.1, 11.2, 12.1
    """
    train_records, val_records = _run_pipeline_on_fixtures(
        sample_train_df, sample_val_df, tmp_path
    )

    for split_name, records in [("train", train_records), ("val", val_records)]:
        for record in records:
            phys_cols = _physical_cols_in_record(record)
            assert phys_cols, (
                f"[{split_name}] Record {record.get('id')} has no physical feature columns"
            )
            for col in phys_cols:
                val = record[col]
                assert val is not None and not (
                    isinstance(val, float) and math.isnan(val)
                ), (
                    f"[{split_name}] Record {record.get('id')}: "
                    f"column '{col}' is NaN/None after imputation"
                )


# ---------------------------------------------------------------------------
# Test 2: _was_imputed indicators match original missingness
# ---------------------------------------------------------------------------

def test_was_imputed_indicators_match_original_missingness(
    sample_train_df, sample_val_df, tmp_path
):
    """
    For each row in the output manifest:
      - `{col}_was_imputed` == 1  iff the original value was NaN/None
      - `{col}_was_imputed` == 0  iff the original value was present

    The conftest fixture has:
      - rows 0, 3, 7 with NaN in `height`   → height_was_imputed == 1
      - rows 1, 5    with NaN in `skin_color` → skin_color_was_imputed == 1
      - all other indicator columns == 0 for all rows

    Validates: Requirements 32.2, 12.1, 12.2
    """
    # Build a lookup: outfit_id (str) → original missingness per indicator col
    indicator_source_cols = [
        "height", "breasts", "skin_color",
        "color_contrast", "hair_style", "hair_color",
    ]

    train_df = sample_train_df.copy()
    train_df["id"] = train_df["outfit_id"].astype(str)
    original_missing: dict[str, dict[str, int]] = {}
    for _, row in train_df.iterrows():
        row_id = str(row["id"])
        original_missing[row_id] = {
            f"{col}_was_imputed": int(pd.isna(row[col]))
            for col in indicator_source_cols
            if col in row.index
        }

    train_records, _ = _run_pipeline_on_fixtures(
        sample_train_df, sample_val_df, tmp_path
    )

    for record in train_records:
        row_id = str(record["id"])
        if row_id not in original_missing:
            continue  # skip rows not in our fixture (shouldn't happen)
        for indicator_col, expected in original_missing[row_id].items():
            if indicator_col not in record:
                continue  # column may not be present if pipeline skips it
            actual = record[indicator_col]
            assert actual == expected, (
                f"Row {row_id}: '{indicator_col}' expected {expected}, got {actual}"
            )


# ---------------------------------------------------------------------------
# Test 3: Lowercase normalization — "Pear" input → bf_pear column is set
# ---------------------------------------------------------------------------

def test_lowercase_normalization_pear(sample_val_df, tmp_path):
    """
    When a row has body_figure="Pear" (mixed case), the pipeline must
    lowercase it before encoding, so the output manifest contains
    bf_pear == 1 for that row (not a KeyError or bf_Pear column).

    Validates: Requirements 32.2, 2.1, 2.2, 2.4
    """
    # Build a small train DataFrame with mixed-case body_figure
    train_data = {
        "outfit_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "score": [1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5],
        "binary_label": [1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        "body_figure": [
            "Pear", "Apple", "hourglass", "rectangle",
            "pear", "apple", "Hourglass", "Rectangle",
            "pear", "apple",
        ],
        "skin_color": ["brown"] * 10,
        "hair_style": ["curly"] * 10,
        "hair_color": ["black"] * 10,
        "height": ["medium"] * 10,
        "breasts": ["medium"] * 10,
        "color_contrast": ["medium"] * 10,
    }
    train_df = pd.DataFrame(train_data)

    # Val DataFrame: one row with body_figure="Pear" (mixed case)
    val_data = {
        "outfit_id": [101],
        "score": [1.0],
        "binary_label": [1],
        "body_figure": ["Pear"],
        "skin_color": ["brown"],
        "hair_style": ["curly"],
        "hair_color": ["black"],
        "height": ["medium"],
        "breasts": ["medium"],
        "color_contrast": ["medium"],
    }
    val_df = pd.DataFrame(val_data)

    _, val_records = _run_pipeline_on_fixtures(train_df, val_df, tmp_path)

    assert len(val_records) == 1, "Expected exactly 1 val record"
    record = val_records[0]

    # bf_pear must exist (not bf_Pear) and be set to 1
    assert "bf_pear" in record, (
        f"'bf_pear' column missing from output; found keys: "
        f"{[k for k in record if k.startswith('bf_')]}"
    )
    assert record["bf_pear"] == 1, (
        f"Expected bf_pear == 1 for body_figure='Pear', got {record['bf_pear']}"
    )

    # Confirm no mixed-case bf_ columns leaked through
    mixed_case_bf = [k for k in record if k.startswith("bf_") and k != k.lower()]
    assert not mixed_case_bf, (
        f"Mixed-case bf_ columns found in output: {mixed_case_bf}"
    )
