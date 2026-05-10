"""
Binary label threshold sensitivity analysis for the O4U pipeline.

Evaluates binary label thresholds [0.5, 1.0, 1.5, 2.0] on the training data.
For each threshold:
  - Creates binary labels: binary_label = (score >= threshold).astype(int)
  - Computes positive rate (class balance) on train_df
  - Trains a simple logistic regression on physical features from train_df
  - Evaluates ROC-AUC on val_df

Saves results to saved_models/threshold_sensitivity.json.

Requirements: 10.1, 10.2, 10.3

Usage:
    python scripts/threshold_sensitivity.py [--train-manifest PATH] [--val-manifest PATH]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path configuration — import from scripts.config with fallback for direct
# execution (e.g. `python scripts/threshold_sensitivity.py`)
# ---------------------------------------------------------------------------
try:
    from scripts.config import (
        DATA_PROCESSED_DIR,
        SAVED_MODELS_DIR,
        TRAIN_IMPUTED_MANIFEST,
        VAL_IMPUTED_MANIFEST,
    )
except ImportError:
    # Fallback: resolve paths relative to this file's location
    _HERE = Path(__file__).resolve().parent.parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from scripts.config import (
        DATA_PROCESSED_DIR,
        SAVED_MODELS_DIR,
        TRAIN_IMPUTED_MANIFEST,
        VAL_IMPUTED_MANIFEST,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS: list[float] = [0.5, 1.0, 1.5, 2.0]

# Physical feature columns used for the logistic regression probe.
# These are the one-hot body-figure columns present in the imputed manifests.
# We use the bf_* columns (body figure binarized) plus one-hot encoded
# categorical columns derived from skin_color, hair_style, hair_color,
# height, breasts, and color_contrast.
_CATEGORICAL_COLS = [
    "skin_color",
    "hair_style",
    "hair_color",
    "height",
    "breasts",
    "color_contrast",
]


def _get_physical_features(
    df: pd.DataFrame, feature_cols: list[str] | None = None
) -> pd.DataFrame:
    """Extract and one-hot encode physical features from a manifest DataFrame.

    Uses body-figure binary columns (bf_*) already present in the manifest,
    plus one-hot encoded versions of the categorical physical columns.

    Parameters
    ----------
    df:
        DataFrame loaded from an imputed manifest.
    feature_cols:
        Optional list of column names to use.  When provided, the returned
        DataFrame is aligned to exactly these columns (missing columns are
        filled with 0).  Pass the columns derived from the training set to
        ensure train/val alignment.

    Returns
    -------
    pd.DataFrame
        Numeric feature matrix, one row per sample.
    """
    # Body-figure binary columns already present
    bf_cols = [c for c in df.columns if c.startswith("bf_")]

    # One-hot encode categorical columns that are present in df
    cat_frames = []
    for col in _CATEGORICAL_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(
                df[col].fillna("unknown").astype(str).str.lower(),
                prefix=col,
            )
            cat_frames.append(dummies)

    parts = [df[bf_cols].fillna(0).astype(float)]
    parts.extend(cat_frames)
    features = pd.concat(parts, axis=1)

    if feature_cols is not None:
        # Align to the training feature columns; add missing cols as zeros
        for col in feature_cols:
            if col not in features.columns:
                features[col] = 0.0
        features = features[feature_cols]

    return features


def analyze_threshold_sensitivity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Evaluate binary label thresholds and report class balance + ROC-AUC.

    For each threshold in *thresholds*:
      1. Creates binary labels on *train_df*: ``(score >= threshold).astype(int)``
      2. Computes the positive rate (fraction of positives) on *train_df*
      3. Trains a logistic regression on physical features from *train_df*
      4. Evaluates ROC-AUC on *val_df* using the same binary labels

    Parameters
    ----------
    train_df:
        Training DataFrame loaded from an imputed manifest.  Must contain
        ``score`` and physical feature columns.
    val_df:
        Validation DataFrame loaded from an imputed manifest.  Must contain
        ``score`` and the same physical feature columns as *train_df*.
    thresholds:
        List of threshold values to evaluate.  Defaults to
        ``[0.5, 1.0, 1.5, 2.0]``.

    Returns
    -------
    list[dict]
        One dict per threshold::

            [
                {"threshold": 0.5, "positive_rate": 0.85, "roc_auc": 0.72},
                {"threshold": 1.0, "positive_rate": 0.58, "roc_auc": 0.75},
                ...
            ]

        ``roc_auc`` is ``None`` when the validation set has only one class
        for a given threshold (ROC-AUC is undefined).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Build physical feature matrix from training data (defines column schema)
    X_train_raw = _get_physical_features(train_df)
    feature_cols = list(X_train_raw.columns)

    # Align val features to the same column schema
    X_val = _get_physical_features(val_df, feature_cols=feature_cols).values

    results: list[dict] = []

    for threshold in thresholds:
        # ----------------------------------------------------------------
        # 1. Create binary labels for this threshold
        # ----------------------------------------------------------------
        train_labels = (train_df["score"] >= threshold).astype(int).values
        val_labels = (val_df["score"] >= threshold).astype(int).values

        # ----------------------------------------------------------------
        # 2. Positive rate on training data
        # ----------------------------------------------------------------
        positive_rate = float(train_labels.mean())

        # ----------------------------------------------------------------
        # 3. Train logistic regression on physical features
        # ----------------------------------------------------------------
        X_train = X_train_raw.values

        # Guard: if only one class in training labels, skip model training
        n_pos_train = int(train_labels.sum())
        n_neg_train = int((1 - train_labels).sum())

        if n_pos_train == 0 or n_neg_train == 0:
            print(
                f"  [threshold={threshold}] Only one class in training labels "
                f"(pos={n_pos_train}, neg={n_neg_train}). Skipping ROC-AUC."
            )
            results.append(
                {
                    "threshold": float(threshold),
                    "positive_rate": positive_rate,
                    "roc_auc": None,
                }
            )
            continue

        clf = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            C=1.0,
        )
        clf.fit(X_train, train_labels)

        # ----------------------------------------------------------------
        # 4. Evaluate ROC-AUC on validation split
        # ----------------------------------------------------------------
        n_pos_val = int(val_labels.sum())
        n_neg_val = int((1 - val_labels).sum())

        if n_pos_val == 0 or n_neg_val == 0:
            print(
                f"  [threshold={threshold}] Only one class in validation labels "
                f"(pos={n_pos_val}, neg={n_neg_val}). ROC-AUC undefined."
            )
            roc_auc = None
        else:
            val_proba = clf.predict_proba(X_val)[:, 1]
            roc_auc = float(roc_auc_score(val_labels, val_proba))

        results.append(
            {
                "threshold": float(threshold),
                "positive_rate": positive_rate,
                "roc_auc": roc_auc,
            }
        )

    return results


def save_threshold_sensitivity(results: list[dict]) -> None:
    """Save threshold sensitivity analysis results to JSON.

    Writes *results* to ``SAVED_MODELS_DIR / "threshold_sensitivity.json"``.

    Parameters
    ----------
    results:
        List of dicts returned by :func:`analyze_threshold_sensitivity`.
    """
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SAVED_MODELS_DIR / "threshold_sensitivity.json"
    with save_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved threshold sensitivity results to: {save_path}")


def _select_best_threshold(results: list[dict]) -> dict:
    """Select the threshold with the highest ROC-AUC from the results.

    Ties are broken by preferring the threshold closest to 1.0 (the
    conventional default for this dataset).

    Parameters
    ----------
    results:
        List of dicts from :func:`analyze_threshold_sensitivity`.

    Returns
    -------
    dict
        The result entry with the highest ROC-AUC.
    """
    valid = [r for r in results if r["roc_auc"] is not None]
    if not valid:
        # Fall back to threshold=1.0 if no valid ROC-AUC was computed
        for r in results:
            if r["threshold"] == 1.0:
                return r
        return results[0]

    best = max(valid, key=lambda r: (r["roc_auc"], -abs(r["threshold"] - 1.0)))
    return best


def update_training_config(
    selected_threshold: float,
    justification: str,
) -> None:
    """Document the selected threshold and justification in training_config.json.

    If ``training_config.json`` already exists in ``SAVED_MODELS_DIR``, it is
    updated in-place.  Otherwise a minimal config is created.

    Parameters
    ----------
    selected_threshold:
        The binary label threshold chosen based on the sensitivity analysis.
    justification:
        Human-readable explanation for the threshold choice.
    """
    config_path = SAVED_MODELS_DIR / "training_config.json"

    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
    else:
        config = {}

    config["binary_label_threshold"] = float(selected_threshold)
    config["binary_label_threshold_justification"] = justification

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"Updated training_config.json with threshold={selected_threshold}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the threshold sensitivity script."""
    parser = argparse.ArgumentParser(
        description=(
            "Binary label threshold sensitivity analysis for the O4U pipeline. "
            "Evaluates thresholds [0.5, 1.0, 1.5, 2.0] and saves results to "
            "saved_models/threshold_sensitivity.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default=str(TRAIN_IMPUTED_MANIFEST),
        help="Path to the training imputed manifest JSON.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=str(VAL_IMPUTED_MANIFEST),
        help="Path to the validation imputed manifest JSON.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="List of binary label thresholds to evaluate.",
    )
    parser.add_argument(
        "--update-training-config",
        action="store_true",
        default=False,
        help=(
            "If set, update training_config.json with the selected threshold "
            "and justification."
        ),
    )
    return parser


def main() -> None:
    """Load manifests, run sensitivity analysis, print results, and save JSON."""
    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load manifests
    # ------------------------------------------------------------------
    train_manifest_path = Path(args.train_manifest)
    val_manifest_path = Path(args.val_manifest)

    if not train_manifest_path.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {train_manifest_path}\n"
            "Run the imputation pipeline first to generate it."
        )
    if not val_manifest_path.exists():
        raise FileNotFoundError(
            f"Validation manifest not found: {val_manifest_path}\n"
            "Run the imputation pipeline first to generate it."
        )

    print(f"Loading training manifest: {train_manifest_path}")
    with train_manifest_path.open("r", encoding="utf-8") as fh:
        train_data = json.load(fh)
    train_df = pd.DataFrame(train_data)

    print(f"Loading validation manifest: {val_manifest_path}")
    with val_manifest_path.open("r", encoding="utf-8") as fh:
        val_data = json.load(fh)
    val_df = pd.DataFrame(val_data)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    print(
        f"Score range — train: [{train_df['score'].min():.3f}, "
        f"{train_df['score'].max():.3f}], "
        f"val: [{val_df['score'].min():.3f}, {val_df['score'].max():.3f}]"
    )
    print()

    # ------------------------------------------------------------------
    # Run sensitivity analysis
    # ------------------------------------------------------------------
    print(f"Evaluating thresholds: {args.thresholds}")
    print("-" * 60)
    results = analyze_threshold_sensitivity(train_df, val_df, args.thresholds)

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    print()
    print(f"{'Threshold':>10}  {'Positive Rate':>14}  {'ROC-AUC':>10}")
    print("-" * 40)
    for r in results:
        roc_str = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "   N/A"
        print(f"{r['threshold']:>10.1f}  {r['positive_rate']:>14.4f}  {roc_str:>10}")
    print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    save_threshold_sensitivity(results)

    # ------------------------------------------------------------------
    # Select best threshold and optionally update training_config.json
    # ------------------------------------------------------------------
    best = _select_best_threshold(results)
    print(
        f"Selected threshold: {best['threshold']} "
        f"(positive_rate={best['positive_rate']:.4f}, "
        f"roc_auc={best['roc_auc']})"
    )

    if args.update_training_config:
        justification = (
            f"Threshold {best['threshold']} selected from sensitivity analysis "
            f"[0.5, 1.0, 1.5, 2.0] based on highest downstream logistic regression "
            f"ROC-AUC ({best['roc_auc']:.4f}) on the validation split. "
            f"Positive rate at this threshold: {best['positive_rate']:.4f}."
        )
        update_training_config(best["threshold"], justification)


if __name__ == "__main__":
    main()
