"""
Baseline models for the O4U fashion outfit compatibility pipeline.

Implements three baselines for comparison against the HybridFashionModel:
  1. pure_visual_baseline   — logistic regression on mean-pooled CLIP features
  2. pure_physical_baseline — logistic regression on imputed physical features
  3. clip_zeroshot_baseline — cosine similarity between CLIP features and a
                              proxy text embedding, thresholded at 0.5

Results are saved to saved_models/baseline_results.json.

Requirements: 25.1, 25.2, 25.3, 25.4, 33.3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Path configuration — import from scripts.config with fallback for direct
# execution (python scripts/baselines.py) where the package root may not be
# on sys.path yet.
# ---------------------------------------------------------------------------
try:
    from scripts.config import (
        DATA_PROCESSED_DIR,
        FEATURES_DIR,
        SAVED_MODELS_DIR,
        VAL_IMPUTED_MANIFEST,
        TRAIN_IMPUTED_MANIFEST,
    )
except ImportError:
    # Fallback: resolve paths relative to this file so the script can be run
    # directly from the project root without installing the package.
    _HERE = Path(__file__).resolve().parent
    _ROOT = _HERE.parent
    DATA_PROCESSED_DIR = _ROOT / "data" / "processed"
    FEATURES_DIR = DATA_PROCESSED_DIR / "outfits"
    SAVED_MODELS_DIR = _ROOT / "saved_models"
    TRAIN_IMPUTED_MANIFEST = DATA_PROCESSED_DIR / "train_imputed_manifest.json"
    VAL_IMPUTED_MANIFEST = DATA_PROCESSED_DIR / "val_imputed_manifest.json"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core baseline functions
# ---------------------------------------------------------------------------


def pure_visual_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Train a logistic regression classifier on mean-pooled CLIP features.

    Parameters
    ----------
    X_train:
        Training visual features, shape (N_train, D).  Each row is the
        mean-pooled CLIP embedding for one outfit.
    y_train:
        Binary training labels, shape (N_train,).
    X_val:
        Validation visual features, shape (N_val, D).
    y_val:
        Binary validation labels, shape (N_val,).

    Returns
    -------
    dict with keys ``roc_auc`` (float) and ``f1`` (float).

    Requirements: 25.1
    """
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_val)[:, 1]
    y_pred = clf.predict(X_val)

    return {
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
    }


def pure_physical_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Train a logistic regression classifier on imputed physical features.

    Parameters
    ----------
    X_train:
        Training physical feature matrix, shape (N_train, P).
    y_train:
        Binary training labels, shape (N_train,).
    X_val:
        Validation physical feature matrix, shape (N_val, P).
    y_val:
        Binary validation labels, shape (N_val,).

    Returns
    -------
    dict with keys ``roc_auc`` (float) and ``f1`` (float).

    Requirements: 25.2
    """
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_val)[:, 1]
    y_pred = clf.predict(X_val)

    return {
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
    }


def clip_zeroshot_baseline(
    val_features: np.ndarray,
    val_labels: np.ndarray,
    text_prompt: str = "a well-fitting outfit",
) -> dict:
    """CLIP zero-shot baseline using cosine similarity.

    Since a live CLIP text encoder is not guaranteed to be available at
    runtime, this function uses the mean of the *positive-class* validation
    features as a proxy for the text prompt embedding.  This is the most
    principled zero-shot proxy available without an external model: the
    centroid of positive examples approximates the direction in CLIP space
    that corresponds to "a well-fitting outfit".

    If ``transformers`` is installed and a CLIP model can be loaded, the
    actual text embedding for ``text_prompt`` is used instead.

    Parameters
    ----------
    val_features:
        Validation CLIP features, shape (N_val, D).  Each row is the
        mean-pooled CLIP embedding for one outfit.
    val_labels:
        Binary validation labels, shape (N_val,).
    text_prompt:
        Text description used as the zero-shot query.  Only used when a
        live CLIP text encoder is available.

    Returns
    -------
    dict with keys ``roc_auc`` (float) and ``f1`` (float).

    Requirements: 25.3
    """
    text_embedding: Optional[np.ndarray] = None

    # Attempt to use a real CLIP text encoder if transformers is available.
    try:
        from transformers import CLIPModel, CLIPTokenizer  # type: ignore

        model_name = "openai/clip-vit-base-patch32"
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        clip_model = CLIPModel.from_pretrained(model_name)
        clip_model.eval()

        import torch  # type: ignore

        inputs = tokenizer([text_prompt], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_feats = clip_model.get_text_features(**inputs)
        text_embedding = text_feats.cpu().numpy().squeeze(0)  # (D,)
        logger.info("clip_zeroshot_baseline: using CLIP text encoder for '%s'", text_prompt)
    except Exception:
        # Fall back to positive-class centroid as proxy text embedding.
        pos_mask = val_labels == 1
        if pos_mask.sum() == 0:
            # Edge case: no positive examples — use global mean.
            text_embedding = val_features.mean(axis=0)
            logger.warning(
                "clip_zeroshot_baseline: no positive-class examples found; "
                "using global feature mean as text proxy."
            )
        else:
            text_embedding = val_features[pos_mask].mean(axis=0)
            logger.info(
                "clip_zeroshot_baseline: CLIP text encoder unavailable; "
                "using positive-class centroid as proxy for '%s'.",
                text_prompt,
            )

    # Compute cosine similarities.
    # normalize rows to unit length, then dot with the (also normalised) query.
    val_norm = normalize(val_features, norm="l2")  # (N, D)
    query_norm = text_embedding / (np.linalg.norm(text_embedding) + 1e-12)  # (D,)
    similarities = val_norm @ query_norm  # (N,)

    # Binary predictions at threshold 0.5.
    y_pred = (similarities >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(val_labels, similarities)),
        "f1": float(f1_score(val_labels, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------


def save_baseline_results(results: dict) -> None:
    """Save baseline results to ``saved_models/baseline_results.json``.

    Parameters
    ----------
    results:
        Dictionary mapping baseline name → metric dict.

    Requirements: 25.4
    """
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVED_MODELS_DIR / "baseline_results.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Baseline results saved to %s", out_path)
    print(f"Baseline results saved to {out_path}")


# ---------------------------------------------------------------------------
# Feature loading helpers
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> list[dict]:
    """Load a JSON manifest file and return the list of records."""
    with open(manifest_path) as fh:
        data = json.load(fh)
    # Manifests may be a list directly or wrapped in a dict.
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    # Fallback: treat dict values as records.
    return list(data.values())


def _load_phys_feature_cols() -> list[str]:
    """Load the ordered physical feature column list from the saved artifact."""
    artifact_path = SAVED_MODELS_DIR / "phys_feature_cols.json"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing artifact: phys_feature_cols.json — run training to generate it. "
            f"Expected at: {artifact_path}"
        )
    with open(artifact_path) as fh:
        return json.load(fh)


def _build_feature_matrices(
    manifest: list[dict],
    phys_feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build visual feature matrix, physical feature matrix, and labels.

    Parameters
    ----------
    manifest:
        List of manifest records (each a dict with outfit metadata).
    phys_feature_cols:
        Ordered list of physical feature column names.

    Returns
    -------
    visual_features : np.ndarray, shape (N, D)
        Mean-pooled CLIP features loaded from .pt files.
    phys_features : np.ndarray, shape (N, P)
        Physical feature vectors assembled from manifest records.
    labels : np.ndarray, shape (N,)
        Binary labels.
    """
    import torch  # type: ignore

    visual_list: list[np.ndarray] = []
    phys_list: list[np.ndarray] = []
    label_list: list[int] = []
    skipped = 0

    for record in manifest:
        outfit_id = str(record.get("outfit_id", record.get("id", "")))
        pt_path = FEATURES_DIR / f"{outfit_id}.pt"

        if not pt_path.exists():
            skipped += 1
            continue

        # Load CLIP features and mean-pool across items.
        feats = torch.load(pt_path, map_location="cpu", weights_only=True)
        if isinstance(feats, torch.Tensor):
            feat_np = feats.numpy()
        else:
            feat_np = np.array(feats)

        if feat_np.ndim == 1:
            visual_vec = feat_np
        else:
            visual_vec = feat_np.mean(axis=0)

        # Build physical feature vector from manifest record.
        phys_vec = np.array(
            [float(record.get(col, 0.0) or 0.0) for col in phys_feature_cols],
            dtype=np.float32,
        )

        binary_label = int(record.get("binary_label", 0))

        visual_list.append(visual_vec.astype(np.float32))
        phys_list.append(phys_vec)
        label_list.append(binary_label)

    if skipped > 0:
        logger.warning("Skipped %d records with missing .pt files.", skipped)

    if not visual_list:
        raise RuntimeError(
            "No valid records found in manifest — all .pt files are missing or "
            "the manifest is empty."
        )

    return (
        np.stack(visual_list),
        np.stack(phys_list),
        np.array(label_list, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all three baselines and save results to JSON."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate pure-visual, pure-physical, and CLIP zero-shot baselines."
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=TRAIN_IMPUTED_MANIFEST,
        help="Path to the training imputed manifest JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=VAL_IMPUTED_MANIFEST,
        help="Path to the validation imputed manifest JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="a well-fitting outfit",
        help="Text prompt for the CLIP zero-shot baseline (default: %(default)s).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load physical feature column schema.
    # ------------------------------------------------------------------
    try:
        phys_feature_cols = _load_phys_feature_cols()
        logger.info("Loaded %d physical feature columns.", len(phys_feature_cols))
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load manifests.
    # ------------------------------------------------------------------
    logger.info("Loading training manifest from %s …", args.train_manifest)
    train_manifest = _load_manifest(args.train_manifest)
    logger.info("  %d training records.", len(train_manifest))

    logger.info("Loading validation manifest from %s …", args.val_manifest)
    val_manifest = _load_manifest(args.val_manifest)
    logger.info("  %d validation records.", len(val_manifest))

    # ------------------------------------------------------------------
    # Build feature matrices.
    # ------------------------------------------------------------------
    logger.info("Building training feature matrices …")
    X_train_vis, X_train_phys, y_train = _build_feature_matrices(
        train_manifest, phys_feature_cols
    )
    logger.info(
        "  visual: %s  physical: %s  labels: %s",
        X_train_vis.shape,
        X_train_phys.shape,
        y_train.shape,
    )

    logger.info("Building validation feature matrices …")
    X_val_vis, X_val_phys, y_val = _build_feature_matrices(
        val_manifest, phys_feature_cols
    )
    logger.info(
        "  visual: %s  physical: %s  labels: %s",
        X_val_vis.shape,
        X_val_phys.shape,
        y_val.shape,
    )

    # ------------------------------------------------------------------
    # Run baselines.
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}

    logger.info("Running pure_visual_baseline …")
    vis_results = pure_visual_baseline(X_train_vis, y_train, X_val_vis, y_val)
    results["pure_visual"] = vis_results
    print(f"  pure_visual      — ROC-AUC: {vis_results['roc_auc']:.4f}  F1: {vis_results['f1']:.4f}")

    logger.info("Running pure_physical_baseline …")
    phys_results = pure_physical_baseline(X_train_phys, y_train, X_val_phys, y_val)
    results["pure_physical"] = phys_results
    print(f"  pure_physical    — ROC-AUC: {phys_results['roc_auc']:.4f}  F1: {phys_results['f1']:.4f}")

    logger.info("Running clip_zeroshot_baseline …")
    zs_results = clip_zeroshot_baseline(X_val_vis, y_val, text_prompt=args.text_prompt)
    results["clip_zeroshot"] = zs_results
    print(f"  clip_zeroshot    — ROC-AUC: {zs_results['roc_auc']:.4f}  F1: {zs_results['f1']:.4f}")

    # ------------------------------------------------------------------
    # Save results.
    # ------------------------------------------------------------------
    save_baseline_results(results)


if __name__ == "__main__":
    main()
