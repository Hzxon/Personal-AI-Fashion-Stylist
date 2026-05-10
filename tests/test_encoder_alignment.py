"""
Encoder alignment test — verifies that phys_feature_cols.json saved by
build_encoder_artifacts contains column names consistent with those produced
by direct pd.get_dummies on the same DataFrame.

Validates: Requirements 32.5
"""

import json
import pandas as pd
import pytest

from scripts.o4u_imputation_pipeline import build_encoder_artifacts, CAT_COLS


def _make_sample_df() -> pd.DataFrame:
    """
    Small, fully-observed DataFrame with the same categorical columns used by
    the imputation pipeline.  No NaN values so pd.get_dummies is deterministic.
    """
    return pd.DataFrame(
        {
            "skin_color":     ["brown", "fair",     "dark",  "brown", "fair"],
            "hair_style":     ["curly", "straight", "curly", "straight", "curly"],
            "hair_color":     ["black", "brown",    "black", "brown",  "black"],
            "height":         ["short", "tall",     "medium","short",  "tall"],
            "breasts":        ["small", "medium",   "large", "small",  "medium"],
            "color_contrast": ["low",   "medium",   "high",  "low",    "medium"],
            "body_figure":    ["apple", "pear",     "hourglass", "rectangle", "apple"],
            # bf_ columns simulating MultiLabelBinarizer output
            "bf_apple":       [1, 0, 0, 0, 1],
            "bf_pear":        [0, 1, 0, 0, 0],
            "bf_hourglass":   [0, 0, 1, 0, 0],
            "bf_rectangle":   [0, 0, 0, 1, 0],
        }
    )


def test_encoder_artifact_round_trip(tmp_path):
    """
    1. Build a small sample DataFrame with categorical columns.
    2. One-hot encode it with pd.get_dummies (same way the pipeline does).
    3. Call build_encoder_artifacts to save phys_feature_cols.json and
       encoder_mapping.json to tmp_path.
    4. Load phys_feature_cols.json and verify the saved column names are a
       subset of (or equal to) the columns produced by pd.get_dummies.

    Validates: Requirements 32.5
    """
    df = _make_sample_df()

    # Replicate the pipeline's encoding step: get_dummies on CAT_COLS
    df_enc = pd.get_dummies(df, columns=CAT_COLS, prefix=CAT_COLS)

    # Collect bf_ columns (already present in the sample DataFrame)
    bf_cols = sorted(c for c in df_enc.columns if c.startswith("bf_"))

    # Build phys_feature_cols the same way _save_encoder_artifacts does:
    # bf_ columns + sorted one-hot columns per cat col
    one_hot_cols_flat = []
    for col in CAT_COLS:
        prefix = f"{col}_"
        one_hot_cols_flat.extend(sorted(c for c in df_enc.columns if c.startswith(prefix)))

    phys_feature_cols = bf_cols + one_hot_cols_flat

    # Save artifacts to tmp_path
    build_encoder_artifacts(
        df_train_enc=df_enc,
        cat_cols=CAT_COLS,
        bf_cols=bf_cols,
        phys_feature_cols=phys_feature_cols,
        save_dir=tmp_path,
    )

    # Verify files were created
    phys_path = tmp_path / "phys_feature_cols.json"
    enc_path = tmp_path / "encoder_mapping.json"
    assert phys_path.exists(), "phys_feature_cols.json was not created"
    assert enc_path.exists(), "encoder_mapping.json was not created"

    # Load the saved artifact
    with open(phys_path) as f:
        saved_cols = json.load(f)

    # The saved list must be a list of strings
    assert isinstance(saved_cols, list), "phys_feature_cols.json should contain a list"
    assert all(isinstance(c, str) for c in saved_cols), "All entries should be strings"

    # Every column in the saved artifact must exist in the get_dummies output
    dummies_cols = set(df_enc.columns)
    for col in saved_cols:
        assert col in dummies_cols, (
            f"Saved column '{col}' not found in pd.get_dummies output. "
            f"Available columns: {sorted(dummies_cols)}"
        )

    # The saved list must match the phys_feature_cols we passed in (round-trip)
    assert saved_cols == phys_feature_cols, (
        "Saved phys_feature_cols.json does not match the list passed to "
        "build_encoder_artifacts.\n"
        f"  Expected: {phys_feature_cols}\n"
        f"  Got:      {saved_cols}"
    )

    # Verify encoder_mapping.json structure
    with open(enc_path) as f:
        enc_mapping = json.load(f)

    assert "cat_cols" in enc_mapping, "encoder_mapping.json missing 'cat_cols'"
    assert "bf_classes" in enc_mapping, "encoder_mapping.json missing 'bf_classes'"
    assert "one_hot_cols" in enc_mapping, "encoder_mapping.json missing 'one_hot_cols'"

    # Each cat_col should have its one-hot columns listed
    for col in CAT_COLS:
        assert col in enc_mapping["one_hot_cols"], (
            f"encoder_mapping.json missing one_hot_cols entry for '{col}'"
        )
        saved_oh_cols = enc_mapping["one_hot_cols"][col]
        expected_oh_cols = sorted(c for c in df_enc.columns if c.startswith(f"{col}_"))
        assert saved_oh_cols == expected_oh_cols, (
            f"One-hot columns for '{col}' don't match.\n"
            f"  Expected: {expected_oh_cols}\n"
            f"  Got:      {saved_oh_cols}"
        )
