"""
Public API for the scripts package.

Import the most-used symbols directly from `scripts` rather than
reaching into sub-modules.
"""

from scripts.data_utils import O4UHybridDataset, collate_fn
from scripts.models import HybridFashionModel, VisualBranch, GatedFusion, FiLMFusion, FeatureExtractor
from scripts.config import PROJECT_ROOT, SAVED_MODELS_DIR, FEATURES_DIR

__all__ = [
    # data_utils
    "O4UHybridDataset",
    "collate_fn",
    # models
    "HybridFashionModel",
    "VisualBranch",
    "GatedFusion",
    "FiLMFusion",
    "FeatureExtractor",
    # config
    "PROJECT_ROOT",
    "SAVED_MODELS_DIR",
    "FEATURES_DIR",
]
