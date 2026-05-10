"""
Test set evaluator for the O4U HybridFashionModel.

Loads the held-out test split and corresponding imputed manifest(s), runs the
trained model, and reports evaluation metrics.  Metric computation (ROC-AUC,
F1, accuracy, Brier score, MAE, bootstrap CIs, per-group metrics, reliability
diagram) will be added in tasks 12.2–12.8.  This module provides the scaffold
with data loading and model/artifact loading.

Usage:
    python scripts/evaluate_test_set.py [--n-imputations 1]
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Path configuration — try package import first, fall back for standalone use
# ---------------------------------------------------------------------------
try:
    from scripts.config import (
        DATA_PROCESSED_DIR,
        DATA_RAW_DIR,
        FEATURES_DIR,
        PLOTS_DIR,
        PROJECT_ROOT,
        SAVED_MODELS_DIR,
        TEST_IMPUTED_MANIFEST,
        TEST_MANIFEST,
        VAL_IMPUTED_MANIFEST,
    )
    from scripts.models import HybridFashionModel
except ImportError:
    _HERE = Path(__file__).resolve().parent.parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from scripts.config import (
        DATA_PROCESSED_DIR,
        DATA_RAW_DIR,
        FEATURES_DIR,
        PLOTS_DIR,
        PROJECT_ROOT,
        SAVED_MODELS_DIR,
        TEST_IMPUTED_MANIFEST,
        TEST_MANIFEST,
        VAL_IMPUTED_MANIFEST,
    )
    from scripts.models import HybridFashionModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TEST_JSON_PATH: Path = DATA_RAW_DIR / "Outfit4You" / "label" / "test.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(n_imputations: int = 1) -> list[pd.DataFrame]:
    """Load the test split and corresponding imputed manifest(s).

    Loads ``test.json`` from the raw label directory and merges it with the
    imputed physical feature manifest(s) produced by the imputation pipeline.

    Parameters
    ----------
    n_imputations:
        Number of imputed manifests to load.

        - If ``1``: loads ``test_imputed_manifest.json`` from
          ``DATA_PROCESSED_DIR``.  If that file does not exist yet (e.g. the
          imputation pipeline has not been run on the test split), falls back
          to ``val_imputed_manifest.json`` and logs a warning.
        - If ``> 1``: loads ``test_imputed_manifest_0.json``,
          ``test_imputed_manifest_1.json``, …,
          ``test_imputed_manifest_{n_imputations-1}.json`` from
          ``DATA_PROCESSED_DIR``.  Raises ``FileNotFoundError`` if any of the
          expected files is missing.

    Returns
    -------
    list[pd.DataFrame]
        A list of DataFrames, one per imputation.  Each DataFrame contains the
        raw test labels merged with the imputed physical features.  The list
        has length ``n_imputations``.

    Raises
    ------
    FileNotFoundError
        If ``test.json`` is missing, or if any required imputed manifest file
        is missing when ``n_imputations > 1``.
    ValueError
        If ``n_imputations`` is less than 1.
    """
    if n_imputations < 1:
        raise ValueError(
            f"n_imputations must be >= 1, got {n_imputations}"
        )

    # ------------------------------------------------------------------
    # 1. Load raw test labels (Requirement 4.1)
    # ------------------------------------------------------------------
    if not _TEST_JSON_PATH.exists():
        raise FileNotFoundError(
            f"test.json not found at {_TEST_JSON_PATH}. "
            "Ensure the raw data is present."
        )

    with _TEST_JSON_PATH.open("r", encoding="utf-8") as fh:
        test_raw = json.load(fh)

    # test.json is a list of dicts; convert to DataFrame
    test_df_raw = pd.DataFrame(test_raw)
    logger.info(
        "Loaded test.json: %d samples, columns: %s",
        len(test_df_raw),
        list(test_df_raw.columns),
    )

    # ------------------------------------------------------------------
    # 2. Load imputed manifest(s) and merge with raw labels
    # ------------------------------------------------------------------
    if n_imputations == 1:
        manifest_path = DATA_PROCESSED_DIR / "test_imputed_manifest.json"

        if not manifest_path.exists():
            # Fallback: use val_imputed_manifest.json (Requirement 4.1 note)
            logger.warning(
                "test_imputed_manifest.json not found at %s. "
                "Falling back to val_imputed_manifest.json. "
                "Run the imputation pipeline on the test split to generate "
                "test_imputed_manifest.json.",
                manifest_path,
            )
            manifest_path = VAL_IMPUTED_MANIFEST
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Neither test_imputed_manifest.json nor "
                    f"val_imputed_manifest.json found in {DATA_PROCESSED_DIR}. "
                    "Run the imputation pipeline first."
                )

        df = _load_and_merge_manifest(manifest_path, test_df_raw)
        logger.info(
            "Loaded imputed manifest: %s (%d rows)", manifest_path.name, len(df)
        )
        return [df]

    else:
        # Multiple imputation: load test_imputed_manifest_0.json, _1.json, …
        dfs: list[pd.DataFrame] = []
        for i in range(n_imputations):
            manifest_path = DATA_PROCESSED_DIR / f"test_imputed_manifest_{i}.json"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Missing imputed manifest for imputation {i}: "
                    f"{manifest_path}. "
                    f"Run the imputation pipeline with --n-imputations "
                    f"{n_imputations} to generate all {n_imputations} manifests."
                )
            df = _load_and_merge_manifest(manifest_path, test_df_raw)
            logger.info(
                "Loaded imputed manifest %d/%d: %s (%d rows)",
                i + 1,
                n_imputations,
                manifest_path.name,
                len(df),
            )
            dfs.append(df)
        return dfs


def _load_and_merge_manifest(
    manifest_path: Path, test_df_raw: pd.DataFrame
) -> pd.DataFrame:
    """Load an imputed manifest JSON and merge it with the raw test DataFrame.

    The imputed manifest is a list of dicts (one per outfit) containing the
    imputed physical feature columns.  It is merged with the raw test labels
    on ``outfit_id``.  If the manifest already contains all columns from the
    raw test DataFrame, the raw DataFrame is not merged (to avoid duplicates).

    Parameters
    ----------
    manifest_path:
        Path to the imputed manifest JSON file.
    test_df_raw:
        Raw test labels DataFrame loaded from ``test.json``.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing both raw labels and imputed features.
    """
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest_data = json.load(fh)

    manifest_df = pd.DataFrame(manifest_data)

    # Determine the outfit ID column name (may be "outfit_id" or "set_id")
    id_col = _detect_id_column(manifest_df, test_df_raw)

    if id_col is not None:
        # Merge on the shared ID column, avoiding duplicate columns
        raw_cols_to_merge = [
            c for c in test_df_raw.columns
            if c not in manifest_df.columns or c == id_col
        ]
        if len(raw_cols_to_merge) > 1:
            merged = manifest_df.merge(
                test_df_raw[raw_cols_to_merge],
                on=id_col,
                how="left",
            )
        else:
            # All raw columns already present in manifest; no merge needed
            merged = manifest_df
    else:
        # No shared ID column found; assume row-aligned and concatenate columns
        logger.warning(
            "No shared ID column found between manifest and test.json. "
            "Assuming row-aligned data and concatenating columns."
        )
        extra_cols = [c for c in test_df_raw.columns if c not in manifest_df.columns]
        if extra_cols:
            merged = pd.concat(
                [manifest_df.reset_index(drop=True),
                 test_df_raw[extra_cols].reset_index(drop=True)],
                axis=1,
            )
        else:
            merged = manifest_df

    return merged.reset_index(drop=True)


def _detect_id_column(
    manifest_df: pd.DataFrame, test_df_raw: pd.DataFrame
) -> Optional[str]:
    """Return the name of the shared outfit ID column, or None if not found."""
    candidates = ["outfit_id", "set_id", "id"]
    for col in candidates:
        if col in manifest_df.columns and col in test_df_raw.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# Model and artifact loading
# ---------------------------------------------------------------------------

def load_model_and_artifacts() -> tuple:
    """Load the trained model and all required artifacts from ``SAVED_MODELS_DIR``.

    Loads the following artifacts:

    - ``best_hybrid_model.pth``   — model checkpoint
    - ``phys_feature_cols.json``  — ordered list of physical feature column names
    - ``encoder_mapping.json``    — categorical encoding schema
    - ``thresholds.json``         — classification threshold (Requirement 4.5)
    - ``score_normalization.json``— score mean and std for inverse-transform

    Parameters
    ----------
    None

    Returns
    -------
    tuple of (model, phys_feature_cols, encoder_mapping, threshold, score_mean, score_std)
        - ``model``: loaded ``HybridFashionModel`` in eval mode on CPU
        - ``phys_feature_cols``: list[str] — ordered physical feature column names
        - ``encoder_mapping``: dict — categorical encoding schema
        - ``threshold``: float — classification threshold from ``thresholds.json``
        - ``score_mean``: float — regression score mean for inverse-transform
        - ``score_std``: float — regression score std for inverse-transform

    Raises
    ------
    FileNotFoundError
        If any required artifact file is missing.  The error message explicitly
        names the missing file (Requirement 4.2).
    """
    # ------------------------------------------------------------------
    # 1. Load JSON artifacts (Requirement 4.2)
    # ------------------------------------------------------------------
    phys_feature_cols = _load_json_artifact("phys_feature_cols.json")
    encoder_mapping = _load_json_artifact("encoder_mapping.json")
    thresholds = _load_json_artifact("thresholds.json")
    score_normalization = _load_json_artifact("score_normalization.json")

    threshold: float = float(thresholds["classification_threshold"])
    score_mean: float = float(score_normalization["mean"])
    score_std: float = float(score_normalization["std"])

    logger.info(
        "Artifacts loaded — phys_feature_cols: %d cols, threshold: %.4f, "
        "score_mean: %.4f, score_std: %.4f",
        len(phys_feature_cols),
        threshold,
        score_mean,
        score_std,
    )

    # ------------------------------------------------------------------
    # 2. Load model checkpoint (Requirement 4.2)
    # ------------------------------------------------------------------
    model_path = SAVED_MODELS_DIR / "best_hybrid_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing artifact: best_hybrid_model.pth — "
            f"run training to generate it. Expected path: {model_path}"
        )

    phys_input_dim = len(phys_feature_cols)
    model = HybridFashionModel(phys_input_dim=phys_input_dim)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

    # Checkpoint may be a raw state_dict or a dict with "model_state_dict" key
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        saved_epoch = checkpoint.get("epoch", "unknown")
        saved_metric = checkpoint.get("metric_value", "unknown")
        logger.info(
            "Loaded checkpoint from epoch %s (metric_value=%s)",
            saved_epoch,
            saved_metric,
        )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    logger.info(
        "Model loaded: HybridFashionModel(phys_input_dim=%d)", phys_input_dim
    )

    return model, phys_feature_cols, encoder_mapping, threshold, score_mean, score_std


def _load_json_artifact(filename: str) -> object:
    """Load a JSON artifact from ``SAVED_MODELS_DIR``.

    Parameters
    ----------
    filename:
        Name of the JSON file (e.g. ``"phys_feature_cols.json"``).

    Returns
    -------
    object
        Parsed JSON content (list, dict, etc.).

    Raises
    ------
    FileNotFoundError
        If the file does not exist, with a message that explicitly names the
        missing file (Requirement 4.2).
    """
    path = SAVED_MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing artifact: {filename} — "
            f"run training to generate it. Expected path: {path}"
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for a metric function.

    Resamples ``y_true`` and ``y_pred`` with replacement ``n_resamples`` times,
    computes ``metric_fn(y_true_resampled, y_pred_resampled)`` for each
    resample, and returns the lower and upper percentile bounds of the
    resulting bootstrap distribution.

    Parameters
    ----------
    y_true:
        Ground-truth labels, shape ``(N,)``.
    y_pred:
        Predicted values (probabilities or scores), shape ``(N,)``.
    metric_fn:
        A callable with signature ``metric_fn(y_true, y_pred) -> float``.
        If ``metric_fn`` raises an exception for a given resample (e.g. due
        to a degenerate sample with only one class), that resample is skipped
        and a warning is logged.
    n_resamples:
        Number of bootstrap resamples.  Defaults to 1,000.
    ci:
        Confidence level.  Defaults to 0.95 (95% CI), which corresponds to
        the 2.5th and 97.5th percentiles.
    seed:
        Optional integer seed for ``np.random.default_rng`` to make the
        bootstrap reproducible.  If ``None``, the RNG is seeded from the OS.

    Returns
    -------
    tuple[float, float]
        ``(ci_lower, ci_upper)`` — the lower and upper bounds of the
        confidence interval.

    Raises
    ------
    ValueError
        If fewer than 2 valid bootstrap samples could be computed (i.e. almost
        every resample raised an exception), making a CI meaningless.

    Notes
    -----
    Requirements 24.1, 24.2: 1,000 resamples; 2.5th and 97.5th percentiles
    for a 95% CI.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    rng = np.random.default_rng(seed)
    bootstrap_scores: list[float] = []

    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        try:
            score = metric_fn(y_true_boot, y_pred_boot)
            bootstrap_scores.append(float(score))
        except Exception as exc:
            logger.debug("Bootstrap resample skipped due to exception: %s", exc)

    if len(bootstrap_scores) < 2:
        raise ValueError(
            f"Only {len(bootstrap_scores)} valid bootstrap sample(s) out of "
            f"{n_resamples} resamples — cannot compute a meaningful CI. "
            "Check that metric_fn is compatible with the provided data."
        )

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)       # 2.5 for ci=0.95
    upper_pct = 100.0 * (1.0 - alpha / 2.0) # 97.5 for ci=0.95

    ci_lower = float(np.percentile(bootstrap_scores, lower_pct))
    ci_upper = float(np.percentile(bootstrap_scores, upper_pct))

    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_reg: np.ndarray,
    threshold: float = 0.5,
    n_resamples: int = 1000,
    ci: float = 0.95,
    bootstrap_seed: Optional[int] = None,
) -> dict:
    """Compute evaluation metrics with partial reporting on failure.

    Each metric is computed in its own ``try/except`` block so that a failure
    in one metric does not prevent the remaining metrics from being computed.
    A warning is logged for any metric that fails, and its value is set to
    ``None`` in the returned dict.

    Bootstrap confidence intervals (Requirements 24.1, 24.2) are computed for
    ROC-AUC, F1, accuracy, and Brier score using ``bootstrap_ci``.  If CI
    computation fails for a metric, ``ci_lower`` and ``ci_upper`` are set to
    ``None`` for that metric.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels, shape ``(N,)``.
    y_pred_proba:
        Predicted probabilities for the positive class, shape ``(N,)``.
    y_pred_reg:
        Predicted regression scores (inverse-transformed), shape ``(N,)``.
    threshold:
        Decision threshold applied to ``y_pred_proba`` to produce binary
        predictions for F1 and accuracy.  Defaults to ``0.5``.
    n_resamples:
        Number of bootstrap resamples for CI computation.  Defaults to 1,000.
    ci:
        Confidence level for bootstrap CIs.  Defaults to 0.95 (95% CI).
    bootstrap_seed:
        Optional seed for the bootstrap RNG.  Pass an integer for
        reproducible CI computation.

    Returns
    -------
    dict
        Keys: ``roc_auc``, ``f1``, ``accuracy``, ``brier_score``, ``mae``.
        For metrics that support bootstrap CIs (all except ``mae``), each
        value is a dict with keys ``value``, ``ci_lower``, ``ci_upper``.
        For ``mae``, the value is a plain ``float``.
        Any value (or sub-key) is ``None`` if computation failed.
    """
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        mean_absolute_error,
        roc_auc_score,
    )

    metrics: dict = {}
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    y_pred_reg = np.asarray(y_pred_reg)

    # ------------------------------------------------------------------
    # Helper: compute bootstrap CI for a metric, returning (lower, upper)
    # or (None, None) on failure.
    # ------------------------------------------------------------------
    def _ci(metric_fn) -> tuple:
        try:
            return bootstrap_ci(
                y_true, y_pred_proba, metric_fn,
                n_resamples=n_resamples, ci=ci, seed=bootstrap_seed,
            )
        except Exception as exc:
            logger.warning("Bootstrap CI computation failed: %s", exc)
            return None, None

    # ------------------------------------------------------------------
    # ROC-AUC (Requirements 24.1, 24.2)
    # ------------------------------------------------------------------
    try:
        roc_auc_val = float(roc_auc_score(y_true, y_pred_proba))
        ci_lower, ci_upper = _ci(roc_auc_score)
        metrics["roc_auc"] = {
            "value": roc_auc_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    except Exception as e:
        logging.warning(f"Failed to compute roc_auc: {e}")
        metrics["roc_auc"] = {"value": None, "ci_lower": None, "ci_upper": None}

    # ------------------------------------------------------------------
    # F1 (threshold applied to probabilities) (Requirements 24.1, 24.2)
    # ------------------------------------------------------------------
    try:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        f1_val = float(f1_score(y_true, y_pred_binary))

        def _f1_fn(yt, yp):
            return f1_score(yt, (yp >= threshold).astype(int))

        ci_lower, ci_upper = _ci(_f1_fn)
        metrics["f1"] = {
            "value": f1_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    except Exception as e:
        logging.warning(f"Failed to compute f1: {e}")
        metrics["f1"] = {"value": None, "ci_lower": None, "ci_upper": None}

    # ------------------------------------------------------------------
    # Accuracy (threshold applied to probabilities) (Requirements 24.1, 24.2)
    # ------------------------------------------------------------------
    try:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        acc_val = float(accuracy_score(y_true, y_pred_binary))

        def _acc_fn(yt, yp):
            return accuracy_score(yt, (yp >= threshold).astype(int))

        ci_lower, ci_upper = _ci(_acc_fn)
        metrics["accuracy"] = {
            "value": acc_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    except Exception as e:
        logging.warning(f"Failed to compute accuracy: {e}")
        metrics["accuracy"] = {"value": None, "ci_lower": None, "ci_upper": None}

    # ------------------------------------------------------------------
    # Brier score (Requirements 24.1, 24.2)
    # ------------------------------------------------------------------
    try:
        brier_val = float(brier_score_loss(y_true, y_pred_proba))
        ci_lower, ci_upper = _ci(brier_score_loss)
        metrics["brier_score"] = {
            "value": brier_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    except Exception as e:
        logging.warning(f"Failed to compute brier_score: {e}")
        metrics["brier_score"] = {"value": None, "ci_lower": None, "ci_upper": None}

    # ------------------------------------------------------------------
    # MAE (between regression predictions and true labels) — no CI
    # ------------------------------------------------------------------
    try:
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred_reg))
    except Exception as e:
        logging.warning(f"Failed to compute mae: {e}")
        metrics["mae"] = None

    return metrics


# ---------------------------------------------------------------------------
# Per-group metrics (Requirements 22.1, 22.2, 22.3)
# ---------------------------------------------------------------------------

def per_group_metrics(
    df: pd.DataFrame,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute accuracy and F1 broken down by ``body_figure`` group and by
    imputed-vs-observed rows.

    Parameters
    ----------
    df:
        DataFrame aligned with *y_pred_proba*.  Must contain:

        - ``binary_label`` column — ground-truth binary labels (0 or 1).
        - ``body_figure`` column — categorical body figure group per row.
        - Zero or more ``*_was_imputed`` columns — binary indicators (1 if the
          corresponding physical feature was imputed, 0 if observed).

    y_pred_proba:
        1-D array of predicted probabilities for the positive class, aligned
        row-for-row with *df*.
    threshold:
        Probability threshold applied to *y_pred_proba* to produce binary
        predictions.  Default ``0.5``.

    Returns
    -------
    dict
        A nested dictionary with the following structure::

            {
                "by_body_figure": {
                    "apple":     {"accuracy": 0.80, "f1": 0.75},
                    "hourglass": {"accuracy": 0.83, "f1": 0.79},
                    ...
                },
                "imputed_rows":  {"accuracy": 0.70, "f1": 0.65},
                "observed_rows": {"accuracy": 0.82, "f1": 0.78},
            }

        Groups with no samples are silently skipped.

    Notes
    -----
    - ``imputed_rows``  — rows where **any** ``*_was_imputed`` column equals 1.
    - ``observed_rows`` — rows where **all** ``*_was_imputed`` columns equal 0
      (or where no ``*_was_imputed`` columns exist, in which case every row is
      treated as observed).
    - Uses ``sklearn.metrics.accuracy_score`` and ``f1_score`` (with
      ``zero_division=0`` to handle degenerate groups gracefully).

    Requirements
    ------------
    22.1 — accuracy and F1 per unique ``body_figure`` value.
    22.2 — accuracy and F1 for imputed rows vs. observed rows.
    22.3 — results saved to ``saved_models/per_group_metrics.json`` via
           :func:`save_per_group_metrics`.
    """
    from sklearn.metrics import accuracy_score, f1_score

    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_true = df["binary_label"].to_numpy(dtype=int)

    result: dict = {}

    # ------------------------------------------------------------------
    # 1. Per body_figure group (Requirement 22.1)
    # ------------------------------------------------------------------
    by_body_figure: dict = {}
    for group_val in df["body_figure"].unique():
        mask = (df["body_figure"] == group_val).to_numpy()
        if mask.sum() == 0:
            continue  # skip empty groups
        group_true = y_true[mask]
        group_pred = y_pred[mask]
        by_body_figure[str(group_val)] = {
            "accuracy": float(accuracy_score(group_true, group_pred)),
            "f1": float(f1_score(group_true, group_pred, zero_division=0)),
        }
    result["by_body_figure"] = by_body_figure

    # ------------------------------------------------------------------
    # 2. Imputed vs. observed rows (Requirement 22.2)
    # ------------------------------------------------------------------
    imputed_cols = [c for c in df.columns if c.endswith("_was_imputed")]

    if imputed_cols:
        # imputed_rows: any _was_imputed column == 1
        imputed_mask = (df[imputed_cols].max(axis=1) == 1).to_numpy()
        # observed_rows: all _was_imputed columns == 0
        observed_mask = ~imputed_mask
    else:
        # No indicator columns present — treat all rows as observed
        imputed_mask = np.zeros(len(df), dtype=bool)
        observed_mask = np.ones(len(df), dtype=bool)

    if imputed_mask.sum() > 0:
        result["imputed_rows"] = {
            "accuracy": float(accuracy_score(y_true[imputed_mask], y_pred[imputed_mask])),
            "f1": float(f1_score(y_true[imputed_mask], y_pred[imputed_mask], zero_division=0)),
        }
    else:
        result["imputed_rows"] = {}

    if observed_mask.sum() > 0:
        result["observed_rows"] = {
            "accuracy": float(accuracy_score(y_true[observed_mask], y_pred[observed_mask])),
            "f1": float(f1_score(y_true[observed_mask], y_pred[observed_mask], zero_division=0)),
        }
    else:
        result["observed_rows"] = {}

    return result


def reliability_diagram(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
) -> dict:
    """Generate a reliability diagram and compute the Brier score.

    Computes a calibration curve using ``sklearn.calibration.calibration_curve``
    and plots the fraction of positives per predicted-probability bin against
    the mean predicted probability in each bin.  A diagonal "perfect
    calibration" reference line is included.

    If the data is sparse (fewer than 5 samples per bin on average),
    ``n_bins`` is adaptively reduced to ``max(2, len(y_true) // 5)``.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels, shape ``(N,)``.
    y_pred_proba:
        Predicted probabilities for the positive class, shape ``(N,)``.
    n_bins:
        Initial number of bins for the calibration curve.  Defaults to 10.
        Adaptively reduced when any bin has fewer than 5 samples.
    save_path:
        Optional path to save the figure.  If ``None``, the figure is saved to
        ``PLOTS_DIR / "reliability_diagram.png"``.

    Returns
    -------
    dict
        ``{"brier_score": float, "n_bins_used": int}``

    Notes
    -----
    - matplotlib is imported at module level with the Agg backend to support
      headless environments.
    - Requirements 23.1, 23.2, 23.3.
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_proba = np.asarray(y_pred_proba, dtype=np.float64)
    n = len(y_true)

    # ------------------------------------------------------------------
    # Adaptive bin count: if data is sparse (fewer than 5 samples per bin
    # on average), reduce n_bins to max(2, len(y_true) // 5).
    # (Requirement 23.2)
    # ------------------------------------------------------------------
    _MIN_SAMPLES_PER_BIN = 5

    if n < n_bins * _MIN_SAMPLES_PER_BIN:
        n_bins = max(2, n // _MIN_SAMPLES_PER_BIN)

    n_bins_used = n_bins

    # Compute calibration curve with the final bin count
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins_used, strategy="uniform"
    )

    # ------------------------------------------------------------------
    # Brier score (Requirement 23.1, 23.3)
    # ------------------------------------------------------------------
    brier = float(brier_score_loss(y_true, y_pred_proba))

    # ------------------------------------------------------------------
    # Plot (Requirement 23.2)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    # Bar chart of fraction of positives per bin
    bar_width = 1.0 / n_bins_used * 0.8
    ax.bar(
        mean_predicted_value,
        fraction_of_positives,
        width=bar_width,
        align="center",
        alpha=0.8,
        color="steelblue",
        label="Fraction of positives",
    )

    # Perfect calibration diagonal reference line
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Perfect calibration",
    )

    ax.set_title("Reliability Diagram")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left")
    ax.text(
        0.98,
        0.02,
        f"Brier score: {brier:.4f}\nn_bins: {n_bins_used}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save figure (Requirement 23.2)
    # ------------------------------------------------------------------
    if save_path is None:
        save_path = PLOTS_DIR / "reliability_diagram.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        "Reliability diagram saved to %s (n_bins_used=%d, brier_score=%.4f)",
        save_path,
        n_bins_used,
        brier,
    )

    return {"brier_score": brier, "n_bins_used": n_bins_used}


def pool_predictions(
    all_probas: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pool predicted probabilities across multiple imputation runs (Rubin's rules).

    When ``N > 1`` imputed manifests are used, each produces its own array of
    predicted probabilities.  This function averages them (Rubin's rules for
    prediction pooling) and returns the standard deviation as an uncertainty
    estimate.

    Parameters
    ----------
    all_probas:
        A list of 1-D ``np.ndarray`` objects, each of shape ``(N_samples,)``,
        containing predicted probabilities for the positive class — one array
        per imputation run.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(mean_proba, std_proba)`` where:

        - ``mean_proba`` — element-wise mean across all arrays, shape
          ``(N_samples,)``.  Implements Rubin's rules for prediction pooling
          (Requirement 13.2).
        - ``std_proba``  — element-wise standard deviation across all arrays,
          shape ``(N_samples,)``.  Serves as an uncertainty estimate
          (Requirement 13.3).

        When only one array is provided (``len(all_probas) == 1``), returns
        ``(all_probas[0], np.zeros_like(all_probas[0]))`` — the single
        prediction array with zero uncertainty.

    Raises
    ------
    ValueError
        If ``all_probas`` is empty.

    Notes
    -----
    Requirements 13.2, 13.3.
    """
    if len(all_probas) == 0:
        raise ValueError("all_probas must contain at least one array.")

    if len(all_probas) == 1:
        return all_probas[0], np.zeros_like(all_probas[0])

    stacked = np.stack(all_probas, axis=0)  # shape: (N_imputations, N_samples)
    mean_proba = np.mean(stacked, axis=0)   # shape: (N_samples,)
    std_proba = np.std(stacked, axis=0)     # shape: (N_samples,)

    return mean_proba, std_proba


def save_per_group_metrics(metrics: dict) -> None:
    """Save per-group metrics to ``saved_models/per_group_metrics.json``.

    Parameters
    ----------
    metrics:
        Dictionary returned by :func:`per_group_metrics`.

    Notes
    -----
    Requirement 22.3 — saves Per_Group_Metrics to
    ``saved_models/per_group_metrics.json``.
    """
    output_path = SAVED_MODELS_DIR / "per_group_metrics.json"
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Per-group metrics saved to %s", output_path)


# ---------------------------------------------------------------------------
# Save evaluation results (Requirement 4.4)
# ---------------------------------------------------------------------------

def save_evaluation_results(metrics: dict, n_imputations_pooled: int) -> None:
    """Save all evaluation metrics and CIs to ``test_evaluation_results.json``.

    Takes the metrics dictionary returned by :func:`compute_metrics`, adds the
    ``n_imputations_pooled`` field, and writes the combined result to
    ``SAVED_MODELS_DIR / "test_evaluation_results.json"`` with 2-space
    indentation.

    Parameters
    ----------
    metrics:
        Dictionary returned by :func:`compute_metrics`.  Expected keys:
        ``roc_auc``, ``f1``, ``accuracy``, ``brier_score`` (each a dict with
        ``value``, ``ci_lower``, ``ci_upper``), and ``mae`` (a plain float).
    n_imputations_pooled:
        Number of imputed manifests that were pooled to produce the
        predictions.  Stored as-is in the output JSON.

    Notes
    -----
    Requirement 4.4 — saves evaluation results (metrics, CIs, and
    ``n_imputations_pooled``) to
    ``saved_models/test_evaluation_results.json``.

    Example output::

        {
          "roc_auc": {"value": 0.81, "ci_lower": 0.79, "ci_upper": 0.83},
          "f1": {"value": 0.74, "ci_lower": 0.71, "ci_upper": 0.77},
          "accuracy": {"value": 0.76, "ci_lower": 0.74, "ci_upper": 0.78},
          "brier_score": {"value": 0.18, "ci_lower": 0.16, "ci_upper": 0.20},
          "mae": 0.31,
          "n_imputations_pooled": 1
        }
    """
    output: dict = dict(metrics)
    output["n_imputations_pooled"] = n_imputations_pooled

    output_path = SAVED_MODELS_DIR / "test_evaluation_results.json"
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    logger.info("Evaluation results saved to %s", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the evaluation script."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the O4U HybridFashionModel on the held-out test split. "
            "Loads test.json, imputed manifest(s), model checkpoint, and all "
            "artifacts from saved_models/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-imputations",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of imputed manifests to load. "
            "Use 1 for single imputation (default). "
            "Use N > 1 to pool predictions across N multiple-imputation runs "
            "(requires test_imputed_manifest_0.json … test_imputed_manifest_{N-1}.json)."
        ),
    )
    return parser


def _run_inference_on_df(
    df: pd.DataFrame,
    model: "HybridFashionModel",
    phys_feature_cols: list,
    score_mean: float,
    score_std: float,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference on a single DataFrame and return predictions.

    Parameters
    ----------
    df:
        DataFrame containing physical feature columns, ``score``, and
        ``binary_label`` columns.
    model:
        Loaded ``HybridFashionModel`` in eval mode.
    phys_feature_cols:
        Ordered list of physical feature column names.
    score_mean:
        Mean used for inverse-transforming regression output.
    score_std:
        Std used for inverse-transforming regression output.
    batch_size:
        Batch size for inference.  Defaults to 64.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(y_pred_proba, y_pred_reg)`` — predicted probabilities and
        inverse-transformed regression scores, each of shape ``(N,)``.
    """
    from torch.utils.data import DataLoader
    from scripts.data_utils import O4UHybridDataset, collate_fn

    # Filter phys_feature_cols to only those present in df (graceful degradation)
    available_cols = [c for c in phys_feature_cols if c in df.columns]
    if len(available_cols) < len(phys_feature_cols):
        missing = set(phys_feature_cols) - set(available_cols)
        logger.warning(
            "%d physical feature column(s) missing from test DataFrame; "
            "filling with zeros: %s",
            len(missing),
            sorted(missing),
        )
        for col in missing:
            df = df.copy()
            df[col] = 0.0

    dataset = O4UHybridDataset(
        df=df,
        features_dir=str(FEATURES_DIR),
        feature_cols=phys_feature_cols,
        cache_in_memory=True,
        on_missing="warn",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    all_probas: list[float] = []
    all_reg: list[float] = []

    model.eval()
    with torch.no_grad():
        for visual_padded, visual_mask, phys_vecs, _reg_labels, _bin_labels in loader:
            reg_out, class_out = model(visual_padded, visual_mask, phys_vecs)
            # Inverse-transform regression output (Requirement 8.3)
            reg_scores = reg_out.squeeze(-1) * score_std + score_mean
            probas = torch.sigmoid(class_out.squeeze(-1))
            all_probas.extend(probas.cpu().numpy().tolist())
            all_reg.extend(reg_scores.cpu().numpy().tolist())

    return np.array(all_probas, dtype=np.float64), np.array(all_reg, dtype=np.float64)


def main() -> None:
    """Entry point: evaluate the model on the test split and save results."""
    parser = build_parser()
    args = parser.parse_args()

    n_imputations: int = args.n_imputations

    print("=" * 60)
    print("O4U Test Set Evaluator")
    print("=" * 60)
    print(f"  Project root    : {PROJECT_ROOT}")
    print(f"  Test JSON       : {_TEST_JSON_PATH}")
    print(f"  Saved models    : {SAVED_MODELS_DIR}")
    print(f"  n_imputations   : {n_imputations}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    logger.info("Loading test data (n_imputations=%d)…", n_imputations)
    test_dfs = load_test_data(n_imputations=n_imputations)
    print(
        f"\nTest data loaded: {len(test_dfs)} imputation(s), "
        f"{len(test_dfs[0])} samples each."
    )
    print(f"Columns: {list(test_dfs[0].columns)}")

    # ------------------------------------------------------------------
    # Load model and artifacts
    # ------------------------------------------------------------------
    logger.info("Loading model and artifacts from %s…", SAVED_MODELS_DIR)
    model, phys_feature_cols, encoder_mapping, threshold, score_mean, score_std = (
        load_model_and_artifacts()
    )
    print(
        f"\nModel loaded: HybridFashionModel "
        f"(phys_input_dim={len(phys_feature_cols)})"
    )
    print(f"  Classification threshold : {threshold:.4f}")
    print(f"  Score normalization      : mean={score_mean:.4f}, std={score_std:.4f}")

    # ------------------------------------------------------------------
    # Run inference on each imputed DataFrame (Requirement 4.3, 13.2)
    # ------------------------------------------------------------------
    logger.info("Running inference on %d imputation(s)…", n_imputations)
    all_probas_list: list[np.ndarray] = []
    all_reg_list: list[np.ndarray] = []

    for i, df in enumerate(test_dfs):
        logger.info(
            "Inference on imputation %d/%d (%d samples)…",
            i + 1,
            n_imputations,
            len(df),
        )
        y_pred_proba, y_pred_reg = _run_inference_on_df(
            df=df,
            model=model,
            phys_feature_cols=phys_feature_cols,
            score_mean=score_mean,
            score_std=score_std,
        )
        all_probas_list.append(y_pred_proba)
        all_reg_list.append(y_pred_reg)

    # ------------------------------------------------------------------
    # Pool predictions across imputations (Requirement 13.2, 13.3)
    # ------------------------------------------------------------------
    y_pred_proba_pooled, pred_std = pool_predictions(all_probas_list)
    # Use mean regression predictions across imputations
    y_pred_reg_pooled = np.mean(np.stack(all_reg_list, axis=0), axis=0)

    if n_imputations > 1:
        logger.info(
            "Pooled %d imputation(s): mean prediction std = %.4f",
            n_imputations,
            float(pred_std.mean()),
        )

    # Ground-truth labels come from the first (or only) DataFrame
    df_primary = test_dfs[0]
    y_true = df_primary["binary_label"].to_numpy(dtype=int)

    print(
        f"\nInference complete: {len(y_true)} samples, "
        f"threshold={threshold:.4f}, n_imputations_pooled={n_imputations}"
    )

    # ------------------------------------------------------------------
    # Compute metrics with bootstrap CIs (Requirements 4.3, 24.1, 24.2)
    # ------------------------------------------------------------------
    logger.info("Computing evaluation metrics…")
    metrics = compute_metrics(
        y_true=y_true,
        y_pred_proba=y_pred_proba_pooled,
        y_pred_reg=y_pred_reg_pooled,
        threshold=threshold,
        n_resamples=1000,
        ci=0.95,
        bootstrap_seed=42,
    )

    print("\nEvaluation metrics:")
    for metric_name, metric_val in metrics.items():
        if isinstance(metric_val, dict):
            v = metric_val.get("value")
            lo = metric_val.get("ci_lower")
            hi = metric_val.get("ci_upper")
            if v is not None:
                ci_str = (
                    f"  95% CI [{lo:.4f}, {hi:.4f}]"
                    if lo is not None and hi is not None
                    else "  (CI unavailable)"
                )
                print(f"  {metric_name:15s}: {v:.4f}{ci_str}")
            else:
                print(f"  {metric_name:15s}: FAILED")
        else:
            if metric_val is not None:
                print(f"  {'mae':15s}: {metric_val:.4f}")
            else:
                print(f"  {'mae':15s}: FAILED")

    # ------------------------------------------------------------------
    # Save all evaluation results to JSON (Requirement 4.4)
    # ------------------------------------------------------------------
    logger.info("Saving evaluation results to test_evaluation_results.json…")
    save_evaluation_results(metrics=metrics, n_imputations_pooled=n_imputations)
    print(
        f"\nResults saved to: "
        f"{SAVED_MODELS_DIR / 'test_evaluation_results.json'}"
    )

    # ------------------------------------------------------------------
    # Per-group metrics (Requirements 22.1, 22.2, 22.3)
    # ------------------------------------------------------------------
    if "body_figure" in df_primary.columns:
        logger.info("Computing per-group metrics…")
        try:
            group_metrics = per_group_metrics(
                df=df_primary,
                y_pred_proba=y_pred_proba_pooled,
                threshold=threshold,
            )
            save_per_group_metrics(group_metrics)
            print(
                f"Per-group metrics saved to: "
                f"{SAVED_MODELS_DIR / 'per_group_metrics.json'}"
            )
        except Exception as exc:
            logger.warning("Per-group metrics computation failed: %s", exc)
    else:
        logger.warning(
            "Column 'body_figure' not found in test DataFrame; "
            "skipping per-group metrics."
        )

    # ------------------------------------------------------------------
    # Reliability diagram (Requirements 23.1, 23.2, 23.3)
    # ------------------------------------------------------------------
    logger.info("Generating reliability diagram…")
    try:
        reliability_diagram(
            y_true=y_true,
            y_pred_proba=y_pred_proba_pooled,
            n_bins=10,
            save_path=PLOTS_DIR / "reliability_diagram.png",
        )
        print(
            f"Reliability diagram saved to: "
            f"{PLOTS_DIR / 'reliability_diagram.png'}"
        )
    except Exception as exc:
        logger.warning("Reliability diagram generation failed: %s", exc)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
