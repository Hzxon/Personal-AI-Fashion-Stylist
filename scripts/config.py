"""
Centralized path configuration for the O4U ML pipeline.

All scripts should import path constants from this module rather than
constructing paths inline. This ensures a single source of truth for
all file locations across the project.
"""

import pathlib

# ---------------------------------------------------------------------------
# Project root — resolved from this file's location (scripts/ -> parent)
# ---------------------------------------------------------------------------
try:
    PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
except Exception as exc:
    raise RuntimeError(
        f"Failed to resolve PROJECT_ROOT from {__file__!r}: {exc}"
    ) from exc

# Sanity-check: the resolved path must exist and be a directory.
if not PROJECT_ROOT.exists() or not PROJECT_ROOT.is_dir():
    raise RuntimeError(
        f"PROJECT_ROOT resolved to {PROJECT_ROOT!r}, which does not exist or "
        "is not a directory. Check that the scripts/ package is located inside "
        "the project root."
    )

# ---------------------------------------------------------------------------
# Top-level directories
# ---------------------------------------------------------------------------
DATA_RAW_DIR: pathlib.Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: pathlib.Path = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR: pathlib.Path = DATA_PROCESSED_DIR / "outfits"
SAVED_MODELS_DIR: pathlib.Path = PROJECT_ROOT / "saved_models"
PLOTS_DIR: pathlib.Path = PROJECT_ROOT / "plots"

# ---------------------------------------------------------------------------
# Raw label manifests
# ---------------------------------------------------------------------------
TRAIN_MANIFEST: pathlib.Path = (
    DATA_RAW_DIR / "Outfit4You" / "label" / "train.json"
)
VAL_MANIFEST: pathlib.Path = (
    DATA_RAW_DIR / "Outfit4You" / "label" / "val.json"
)
TEST_MANIFEST: pathlib.Path = (
    DATA_RAW_DIR / "Outfit4You" / "label" / "test.json"
)

# ---------------------------------------------------------------------------
# Imputed manifests (written by the imputation pipeline)
# ---------------------------------------------------------------------------
TRAIN_IMPUTED_MANIFEST: pathlib.Path = (
    DATA_PROCESSED_DIR / "train_imputed_manifest.json"
)
VAL_IMPUTED_MANIFEST: pathlib.Path = (
    DATA_PROCESSED_DIR / "val_imputed_manifest.json"
)
TEST_IMPUTED_MANIFEST: pathlib.Path = (
    DATA_PROCESSED_DIR / "test_imputed_manifest.json"
)
