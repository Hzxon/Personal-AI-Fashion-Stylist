"""
Unit tests for build_physical_vector in scripts/inference.py.

Requirements: 32.3
"""

import torch
import pytest

from scripts.inference import build_physical_vector


def _derive_cols(phys_feature_cols):
    """Helper: derive bf_cols and cat_cols from phys_feature_cols."""
    bf_cols = [c for c in phys_feature_cols if c.startswith("bf_")]
    # cat_cols are the base category names inferred from one-hot columns
    # (e.g. "skin_color_brown" → "skin_color"), excluding bf_ and _was_imputed cols
    seen = set()
    cat_cols = []
    for col in phys_feature_cols:
        if col.startswith("bf_") or col.endswith("_was_imputed"):
            continue
        # strip the last underscore-separated token to get the base category
        base = "_".join(col.split("_")[:-1])
        if base and base not in seen:
            seen.add(base)
            cat_cols.append(base)
    return bf_cols, cat_cols


def test_output_dimension(sample_phys_feature_cols):
    """Output tensor length must equal len(phys_feature_cols)."""
    bf_cols, cat_cols = _derive_cols(sample_phys_feature_cols)
    user_attrs = {"body_figure": "pear", "skin_color": "brown"}

    result = build_physical_vector(user_attrs, sample_phys_feature_cols, bf_cols, cat_cols)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (len(sample_phys_feature_cols),)


def test_body_figure_pear_sets_bf_pear(sample_phys_feature_cols):
    """body_figure='pear' must set bf_pear=1.0 at the correct index."""
    bf_cols, cat_cols = _derive_cols(sample_phys_feature_cols)
    user_attrs = {"body_figure": "pear"}

    result = build_physical_vector(user_attrs, sample_phys_feature_cols, bf_cols, cat_cols)

    pear_idx = sample_phys_feature_cols.index("bf_pear")
    assert result[pear_idx].item() == pytest.approx(1.0), (
        f"Expected bf_pear=1.0 at index {pear_idx}, got {result[pear_idx].item()}"
    )
    # All other bf_ columns should be 0
    for col in bf_cols:
        if col != "bf_pear":
            idx = sample_phys_feature_cols.index(col)
            assert result[idx].item() == pytest.approx(0.0), (
                f"Expected {col}=0.0, got {result[idx].item()}"
            )


def test_unknown_attributes_produce_zero_vector(sample_phys_feature_cols):
    """Unknown attribute keys must produce an all-zero vector without raising."""
    bf_cols, cat_cols = _derive_cols(sample_phys_feature_cols)
    user_attrs = {"body_figure": "nonexistent_shape", "skin_color": "ultraviolet"}

    # Must not raise
    result = build_physical_vector(user_attrs, sample_phys_feature_cols, bf_cols, cat_cols)

    assert result.sum().item() == pytest.approx(0.0), (
        "Expected all-zero vector for unknown attributes"
    )


def test_mixed_case_same_as_lowercase(sample_phys_feature_cols):
    """body_figure='Pear' must produce the same result as body_figure='pear'."""
    bf_cols, cat_cols = _derive_cols(sample_phys_feature_cols)

    result_lower = build_physical_vector(
        {"body_figure": "pear"}, sample_phys_feature_cols, bf_cols, cat_cols
    )
    result_mixed = build_physical_vector(
        {"body_figure": "Pear"}, sample_phys_feature_cols, bf_cols, cat_cols
    )

    assert torch.allclose(result_lower, result_mixed), (
        "Mixed-case 'Pear' should produce the same vector as lowercase 'pear'"
    )
