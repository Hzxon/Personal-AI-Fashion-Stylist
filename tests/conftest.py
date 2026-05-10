"""
Shared pytest fixtures for the O4U ML pipeline test suite.

Requirements: 32.1, 32.6
"""

import math
import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# sample_train_df
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def sample_train_df():
    """
    10-row DataFrame with physical columns, some NaN values, plus `score`
    and `binary_label` columns.

    NaN pattern:
      - rows 0, 3, 7 have NaN in `height`   (3 rows)
      - rows 1, 5    have NaN in `skin_color` (2 rows)
    """
    data = {
        "outfit_id": list(range(1, 11)),
        "score": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.0, 1.2, 2.2, 0.8],
        "binary_label": [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
        "body_figure": [
            "apple", "pear", "hourglass", "rectangle",
            "apple", "pear", "hourglass", "rectangle",
            "apple", "pear",
        ],
        "skin_color": [
            "brown", None, "fair", "dark",
            "brown", None, "fair", "dark",
            "brown", "fair",
        ],
        "hair_style": [
            "curly", "straight", "curly", "straight",
            "curly", "straight", "curly", "straight",
            "curly", "straight",
        ],
        "hair_color": [
            "black", "brown", "black", "brown",
            "black", "brown", "black", "brown",
            "black", "brown",
        ],
        "height": [
            None, 165.0, 170.0, None,
            158.0, 172.0, 160.0, None,
            168.0, 155.0,
        ],
        "breasts": [
            "small", "medium", "large", "small",
            "medium", "large", "small", "medium",
            "large", "small",
        ],
        "color_contrast": [
            "low", "medium", "high", "low",
            "medium", "high", "low", "medium",
            "high", "low",
        ],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# sample_val_df
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def sample_val_df():
    """
    5-row DataFrame with the same schema as sample_train_df.
    No NaN values — all physical attributes are observed.
    """
    data = {
        "outfit_id": list(range(101, 106)),
        "score": [1.0, 2.0, 0.5, 2.5, 1.5],
        "binary_label": [1, 1, 0, 1, 1],
        "body_figure": ["apple", "pear", "hourglass", "rectangle", "apple"],
        "skin_color": ["brown", "fair", "dark", "brown", "fair"],
        "hair_style": ["curly", "straight", "curly", "straight", "curly"],
        "hair_color": ["black", "brown", "black", "brown", "black"],
        "height": [165.0, 170.0, 158.0, 172.0, 160.0],
        "breasts": ["small", "medium", "large", "small", "medium"],
        "color_contrast": ["low", "medium", "high", "low", "medium"],
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# mock_pt_dir
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def mock_pt_dir(tmp_path):
    """
    Creates a temporary directory containing 5 small `.pt` files named
    `1.pt` through `5.pt`.  Each file holds a random tensor of shape
    `(3, 512)` saved with `torch.save`.

    Returns the Path to the temp directory.
    """
    for i in range(1, 6):
        tensor = torch.randn(3, 512)
        torch.save(tensor, tmp_path / f"{i}.pt")
    return tmp_path


# ---------------------------------------------------------------------------
# sample_phys_feature_cols
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_phys_feature_cols():
    """
    Ordered list of 20 physical feature column names matching the expected
    schema produced by the imputation pipeline (one-hot encoded body figure,
    categorical attributes, and `_was_imputed` indicator columns).
    """
    return [
        # Body figure one-hot (5 columns)
        "bf_apple",
        "bf_pear",
        "bf_hourglass",
        "bf_rectangle",
        "bf_unknown",
        # Skin color one-hot (4 columns)
        "skin_color_brown",
        "skin_color_dark",
        "skin_color_fair",
        "skin_color_unknown",
        # Hair style one-hot (3 columns)
        "hair_style_curly",
        "hair_style_straight",
        "hair_style_unknown",
        # Hair color one-hot (3 columns)
        "hair_color_black",
        "hair_color_brown",
        "hair_color_unknown",
        # Imputation indicator columns (5 columns)
        "height_was_imputed",
        "breasts_was_imputed",
        "skin_color_was_imputed",
        "color_contrast_was_imputed",
        "hair_style_was_imputed",
    ]
