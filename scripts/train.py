"""
Training script for the O4U HybridFashionModel.

Entry point for training the hybrid visual+physical outfit compatibility model.
Subsequent tasks will add: reproducibility seeding, data splits, multi-task loss,
LR scheduling, best-model selection, threshold tuning, per-group metrics, and
model card artifact saving.

Usage:
    python scripts/train.py [--seed 42] [--save-metric roc_auc] [--feature-set full] ...
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Allow running as `python3 scripts/train.py` from the project root
# as well as `python3 -m scripts.train`.
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
    TRAIN_IMPUTED_MANIFEST,
    TRAIN_MANIFEST,
    VAL_IMPUTED_MANIFEST,
    VAL_MANIFEST,
)


# ---------------------------------------------------------------------------
# Ablation study constants (Requirements 7.1, 7.2)
# ---------------------------------------------------------------------------

# NB02 EDA findings: these 6 traits showed low predictive signal and high
# missingness, making them candidates for exclusion in the ablated feature set.
ABLATED_TRAITS = [
    "skin_color",
    "hair_style",
    "hair_color",
    "height",
    "breasts",
    "color_contrast",
]


# ---------------------------------------------------------------------------
# Ablation study helpers (Requirements 7.1, 7.2, 7.3, 7.4)
# ---------------------------------------------------------------------------

def apply_feature_ablation(phys_feature_cols: list[str], feature_set: str) -> list[str]:
    """Filter physical feature columns according to the requested feature set.

    Parameters
    ----------
    phys_feature_cols:
        Ordered list of physical feature column names (e.g. from
        ``phys_feature_cols.json``).
    feature_set:
        One of ``"full"`` or ``"ablated"``.

        - ``"full"``    — returns *phys_feature_cols* unchanged.
        - ``"ablated"`` — returns a filtered list that excludes any column
          whose name starts with one of the :data:`ABLATED_TRAITS` prefixes
          (e.g. ``"skin_color_brown"``, ``"height_was_imputed"``).

    Returns
    -------
    list[str]
        The (possibly filtered) list of column names.

    Raises
    ------
    ValueError
        If *feature_set* is not ``"full"`` or ``"ablated"``.
    """
    if feature_set == "full":
        return phys_feature_cols
    elif feature_set == "ablated":
        return [
            col for col in phys_feature_cols
            if not any(col.startswith(trait) for trait in ABLATED_TRAITS)
        ]
    else:
        raise ValueError(
            f"Unknown feature_set {feature_set!r}. Expected 'full' or 'ablated'."
        )


def save_ablation_results(roc_auc: float, f1: float, feature_set: str) -> None:
    """Persist ablation study results to ``saved_models/ablation_results.json``.

    Only writes the file when *feature_set* is ``"ablated"``; does nothing for
    ``"full"`` runs.

    Parameters
    ----------
    roc_auc:
        Validation ROC-AUC achieved with the ablated feature set.
    f1:
        Validation F1 achieved with the ablated feature set.
    feature_set:
        The feature set mode used during training.  Must be ``"ablated"`` for
        the file to be written.
    """
    if feature_set != "ablated":
        return

    artifact = {"feature_set": feature_set, "roc_auc": roc_auc, "f1": f1}
    save_path = SAVED_MODELS_DIR / "ablation_results.json"
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the training script."""
    parser = argparse.ArgumentParser(
        description="Train the O4U HybridFashionModel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for full reproducibility (saved to training_config.json).",
    )

    # Checkpoint selection criterion
    parser.add_argument(
        "--save-metric",
        choices=["roc_auc", "f1", "val_loss"],
        default="roc_auc",
        help="Metric used to select the best model checkpoint.",
    )

    # Feature ablation
    parser.add_argument(
        "--feature-set",
        choices=["full", "ablated"],
        default="full",
        help=(
            "Feature set mode. 'full' uses all imputed physical features; "
            "'ablated' excludes the 6 EDA-flagged traits "
            "(skin_color, hair_style, hair_color, height, breasts, color_contrast)."
        ),
    )

    # LR schedule
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=3,
        help="Number of linear warmup epochs before cosine annealing begins.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Base learning rate (peak LR after warmup).",
    )

    # Data loading
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for training and validation DataLoaders.",
    )

    # Loss weights
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=(
            "Weight balancing regression (MSE) and classification (BCE) losses. "
            "total_loss = alpha * mse + (1 - alpha) * bce + consistency_weight * consistency."
        ),
    )
    parser.add_argument(
        "--consistency-weight",
        type=float,
        default=0.1,
        help="Weight for the regression-classification consistency loss term.",
    )

    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help=(
            "Number of epochs with no improvement in save-metric before training stops early. "
            "Set to 0 to disable early stopping."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Artifact loading helpers
# ---------------------------------------------------------------------------

def load_json_artifact(filename: str) -> object:
    """Load a JSON artifact from SAVED_MODELS_DIR.

    Raises
    ------
    FileNotFoundError
        If the artifact file does not exist.
    """
    path = SAVED_MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing artifact: {filename} — run the imputation pipeline to generate it. "
            f"Expected path: {path}"
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Reproducibility helpers (Requirements 21.1, 21.2, 21.3)
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed all sources of randomness and enforce deterministic algorithms.

    Calls:
      - ``random.seed(seed)``
      - ``np.random.seed(seed)``
      - ``torch.manual_seed(seed)``
      - ``torch.cuda.manual_seed_all(seed)``
      - ``torch.use_deterministic_algorithms(True)``

    Parameters
    ----------
    seed:
        Integer seed value (e.g. from ``--seed`` CLI argument).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def make_dataloader_generator(seed: int) -> torch.Generator:
    """Return a CPU ``torch.Generator`` seeded with *seed*.

    Pass the returned generator to every ``DataLoader`` via the
    ``generator=`` keyword argument to ensure reproducible data-loading
    order across runs.

    Parameters
    ----------
    seed:
        Integer seed value (e.g. from ``--seed`` CLI argument).

    Returns
    -------
    torch.Generator
        A freshly created, seeded CPU generator.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# Data splitting (Requirements 5.1, 5.2, 6.1, 6.2)
# ---------------------------------------------------------------------------

def make_splits(
    df: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (deep_train, calib, ensemble_train) DataFrames.

    Splitting strategy
    ------------------
    1. Hold out 20% of *df* as the calibration fold, stratified by
       ``binary_label``.  This fold is used exclusively for threshold tuning
       (Youden's J) and is never seen during gradient updates.
    2. From the remaining 80%, hold out 15% as the ensemble training fold,
       also stratified by ``binary_label``.  Ensemble classifiers are trained
       on features extracted from this fold using the *frozen* deep model.
    3. The remaining data becomes the deep-model training fold.

    No outfit ID appears in more than one fold (verified by assertion).

    Parameters
    ----------
    df:
        Full training DataFrame.  Must contain a ``binary_label`` column used
        for stratification and an ``outfit_id`` column (or index) used for
        overlap checking.  If ``outfit_id`` is not a column, the DataFrame
        index is used.
    seed:
        Integer random seed forwarded to ``train_test_split`` for
        reproducibility.

    Returns
    -------
    deep_train, calib, ensemble_train:
        Three non-overlapping DataFrames whose union equals *df*.

    Raises
    ------
    AssertionError
        If any outfit ID appears in more than one fold.
    """
    from sklearn.model_selection import train_test_split

    # ------------------------------------------------------------------
    # Step 1: split off 20% calibration, stratified by binary_label
    # ------------------------------------------------------------------
    remaining, calib = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df["binary_label"],
    )

    # ------------------------------------------------------------------
    # Step 2: from the remaining 80%, split off 15% as ensemble_train
    # ------------------------------------------------------------------
    deep_train, ensemble_train = train_test_split(
        remaining,
        test_size=0.15,
        random_state=seed,
        stratify=remaining["binary_label"],
    )

    # ------------------------------------------------------------------
    # Step 3: verify no outfit ID overlap between folds
    # ------------------------------------------------------------------
    id_col = "outfit_id" if "outfit_id" in df.columns else None
    if id_col is not None:
        ids_deep = set(deep_train[id_col])
        ids_calib = set(calib[id_col])
        ids_ens = set(ensemble_train[id_col])
    else:
        ids_deep = set(deep_train.index)
        ids_calib = set(calib.index)
        ids_ens = set(ensemble_train.index)

    assert ids_deep.isdisjoint(ids_calib), (
        f"Outfit ID overlap between deep_train and calib: "
        f"{ids_deep & ids_calib}"
    )
    assert ids_deep.isdisjoint(ids_ens), (
        f"Outfit ID overlap between deep_train and ensemble_train: "
        f"{ids_deep & ids_ens}"
    )
    assert ids_calib.isdisjoint(ids_ens), (
        f"Outfit ID overlap between calib and ensemble_train: "
        f"{ids_calib & ids_ens}"
    )

    return deep_train, calib, ensemble_train


# ---------------------------------------------------------------------------
# Score normalization (Requirements 8.1, 8.2)
# ---------------------------------------------------------------------------

def compute_score_normalization(deep_train_df: pd.DataFrame) -> tuple[float, float]:
    """Compute mean and std of the ``score`` column on the deep-train split.

    Parameters
    ----------
    deep_train_df:
        The deep-training fold DataFrame.  Must contain a ``score`` column.

    Returns
    -------
    (mean, std):
        Floating-point mean and standard deviation of the ``score`` column.

    Raises
    ------
    ValueError
        If the standard deviation of training scores is zero, which would make
        z-score normalization undefined.
    """
    scores = deep_train_df["score"].astype(float)
    mean = float(scores.mean())
    std = float(scores.std(ddof=1))  # sample std, consistent with z-score convention

    if std == 0.0:
        raise ValueError(
            "Training scores have zero standard deviation — cannot z-score normalize"
        )

    return mean, std


def save_score_normalization(mean: float, std: float) -> None:
    """Persist score normalization statistics to ``SAVED_MODELS_DIR``.

    Saves ``{"mean": mean, "std": std}`` as JSON to
    ``SAVED_MODELS_DIR / "score_normalization.json"``.

    Parameters
    ----------
    mean:
        Mean of training scores returned by :func:`compute_score_normalization`.
    std:
        Standard deviation of training scores returned by
        :func:`compute_score_normalization`.
    """
    artifact = {"mean": mean, "std": std}
    save_path = SAVED_MODELS_DIR / "score_normalization.json"
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)


# ---------------------------------------------------------------------------
# Multi-task loss (Requirements 8.1, 8.2, 9.2)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LR scheduling (Requirements 20.1, 20.2, 20.3)
# ---------------------------------------------------------------------------

def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Return a warmup + cosine annealing learning rate scheduler.

    Combines two schedulers via ``torch.optim.lr_scheduler.SequentialLR``:

    1. **LinearLR** (warmup phase): linearly scales the LR from
       ``start_factor * base_lr`` (≈ 0) up to ``base_lr`` over
       ``warmup_epochs`` steps.
    2. **CosineAnnealingLR** (post-warmup phase): decays the LR from
       ``base_lr`` down to 0 over the remaining ``total_epochs - warmup_epochs``
       steps.

    Parameters
    ----------
    optimizer:
        The optimizer whose learning rate will be scheduled.
    warmup_epochs:
        Number of epochs for the linear warmup phase (Requirement 20.1).
    total_epochs:
        Total number of training epochs.
    base_lr:
        Peak learning rate reached at the end of warmup (Requirement 20.1).

    Returns
    -------
    torch.optim.lr_scheduler.SequentialLR
        A single scheduler object that transitions automatically from warmup
        to cosine annealing at epoch ``warmup_epochs``.

    Notes
    -----
    Call ``scheduler.step()`` once per epoch *after* the optimizer step.
    Log the current LR via :func:`get_current_lr` at each epoch
    (Requirement 20.3).
    """
    # LinearLR scales the LR by a multiplicative factor that goes from
    # start_factor → end_factor over total_iters steps.
    # We want: epoch 0 → ~0, epoch warmup_epochs → base_lr.
    # Using a tiny start_factor (1/warmup_epochs) avoids division-by-zero
    # when warmup_epochs == 1 while still giving a near-zero initial LR.
    start_factor = 1.0 / max(warmup_epochs, 1)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=0.0,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return scheduler


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the current learning rate from the first param group.

    Parameters
    ----------
    optimizer:
        The optimizer to query.

    Returns
    -------
    float
        The ``lr`` value of ``optimizer.param_groups[0]``.
    """
    return float(optimizer.param_groups[0]["lr"])


# ---------------------------------------------------------------------------
# Validation metrics (Requirements 19.1, 19.2, 19.3)
# ---------------------------------------------------------------------------

def compute_val_metrics(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    score_mean: float = 0.0,
    score_std: float = 1.0,
) -> dict:
    """Compute validation metrics over the full validation DataLoader.

    Runs inference on every batch in *val_loader* (no gradient computation),
    collects regression outputs, classification outputs, regression labels, and
    binary labels, then computes:

    - ``roc_auc``  — ``sklearn.metrics.roc_auc_score`` on raw sigmoid probabilities
    - ``f1``       — ``sklearn.metrics.f1_score`` with predictions thresholded at
                     *threshold*
    - ``val_loss`` — multi-task loss computed with placeholder normalization
                     (``score_mean=0``, ``score_std=1``) unless caller passes real
                     normalization stats

    Parameters
    ----------
    model:
        The model to evaluate.  Must return ``(reg_out, class_out)`` from its
        ``forward`` method.
    val_loader:
        DataLoader yielding ``(visual_padded, visual_mask, phys_vecs,
        reg_labels, bin_labels)`` batches (the output of ``collate_fn``).
    device:
        Device to run inference on.
    threshold:
        Probability threshold applied to ``sigmoid(class_out)`` when computing
        F1.  Default ``0.5``.
    score_mean:
        Mean used for z-score normalization in the loss computation.  Pass the
        value from ``compute_score_normalization`` during training; defaults to
        ``0.0`` (no-op normalization) for convenience.
    score_std:
        Standard deviation used for z-score normalization.  Defaults to ``1.0``
        (no-op normalization).

    Returns
    -------
    dict
        Dictionary with keys ``"roc_auc"``, ``"f1"``, and ``"val_loss"``, each
        mapping to a Python float.

    Raises
    ------
    ValueError
        If all binary labels in the validation set belong to a single class,
        making ROC-AUC undefined.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    model.eval()

    all_reg_out: list[torch.Tensor] = []
    all_class_out: list[torch.Tensor] = []
    all_reg_labels: list[torch.Tensor] = []
    all_bin_labels: list[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # collate_fn returns: visual_padded, visual_mask, phys_vecs,
            #                     reg_labels, bin_labels
            visual_padded, visual_mask, phys_vecs, reg_labels, bin_labels = batch

            visual_padded = visual_padded.to(device)
            visual_mask = visual_mask.to(device)
            phys_vecs = phys_vecs.to(device)
            reg_labels = reg_labels.to(device)
            bin_labels = bin_labels.to(device)

            reg_out, class_out = model(visual_padded, visual_mask, phys_vecs)

            # Accumulate outputs for metric computation
            all_reg_out.append(reg_out.squeeze(-1).cpu())
            all_class_out.append(class_out.squeeze(-1).cpu())
            all_reg_labels.append(reg_labels.cpu())
            all_bin_labels.append(bin_labels.cpu())

            # Accumulate loss
            batch_loss = compute_multitask_loss(
                reg_out,
                class_out,
                reg_labels,
                bin_labels,
                score_mean=score_mean,
                score_std=score_std,
            )
            total_loss += batch_loss.item()
            n_batches += 1

    # Concatenate all collected tensors
    reg_out_all = torch.cat(all_reg_out).numpy()
    class_out_all = torch.cat(all_class_out).numpy()
    reg_labels_all = torch.cat(all_reg_labels).numpy()
    bin_labels_all = torch.cat(all_bin_labels).numpy()

    # Predicted probabilities from sigmoid
    proba = torch.sigmoid(torch.tensor(class_out_all)).numpy()

    # ROC-AUC (Requirement 19.1)
    roc_auc = float(roc_auc_score(bin_labels_all, proba))

    # F1 with threshold (Requirement 19.2)
    preds = (proba >= threshold).astype(int)
    f1 = float(f1_score(bin_labels_all, preds, zero_division=0))

    # Average validation loss
    val_loss = total_loss / max(n_batches, 1)

    return {"roc_auc": roc_auc, "f1": f1, "val_loss": val_loss}


# ---------------------------------------------------------------------------
# Per-group metrics (Requirements 22.1, 22.2)
# ---------------------------------------------------------------------------

def compute_per_group_metrics(
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


# ---------------------------------------------------------------------------
# Consistency fraction reporting (Requirement 9.3)
# ---------------------------------------------------------------------------

def compute_consistency_fraction(
    reg_out: np.ndarray, class_out: np.ndarray
) -> float:
    """Compute the fraction of samples where the regression and classification
    heads give inconsistent predictions.

    Inconsistency is defined as::

        sigmoid(class_out) >= 0.5  XOR  reg_out >= 1.0

    That is, one head predicts "compatible" while the other predicts
    "incompatible".

    Parameters
    ----------
    reg_out:
        Raw regression outputs (un-normalised), shape ``(N,)``.
    class_out:
        Raw classification logits (before sigmoid), shape ``(N,)``.

    Returns
    -------
    float
        Fraction of samples in ``[0.0, 1.0]`` where the two heads disagree.
        Returns ``0.0`` for empty arrays.
    """
    reg_out = np.asarray(reg_out, dtype=np.float64)
    class_out = np.asarray(class_out, dtype=np.float64)

    if reg_out.size == 0:
        return 0.0

    # sigmoid(class_out) >= 0.5  ⟺  class_out >= 0.0
    class_pred = (1.0 / (1.0 + np.exp(-class_out))) >= 0.5
    reg_pred = reg_out >= 1.0

    inconsistent = class_pred ^ reg_pred  # element-wise XOR
    return float(inconsistent.mean())


# ---------------------------------------------------------------------------
# Checkpoint saving (Requirements 19.1, 19.2)
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric_value: float,
    save_path: Path,
) -> None:
    """Save a model checkpoint to *save_path*.

    The checkpoint dictionary contains:

    - ``epoch``               — the epoch number (0-indexed)
    - ``model_state_dict``    — ``model.state_dict()``
    - ``optimizer_state_dict``— ``optimizer.state_dict()``
    - ``metric_value``        — the metric value that triggered this save

    Parameters
    ----------
    model:
        The model whose weights to save.
    optimizer:
        The optimizer whose state to save (enables training resumption).
    epoch:
        Current epoch index (0-indexed).
    metric_value:
        The metric value (e.g. ROC-AUC, F1, or val_loss) that triggered this
        checkpoint save.
    save_path:
        Full path (including filename) where the ``.pth`` file will be written.
        Parent directories are created if they do not exist.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric_value": metric_value,
    }
    torch.save(checkpoint, save_path)


# ---------------------------------------------------------------------------
# Best model tracker (Requirement 19.3)
# ---------------------------------------------------------------------------

class BestModelTracker:
    """Track the best validation metric and save checkpoints when it improves.

    Supports three save metrics:

    - ``"roc_auc"`` — higher is better (default per ``--save-metric``)
    - ``"f1"``      — higher is better
    - ``"val_loss"``— lower is better

    Parameters
    ----------
    save_metric:
        One of ``{"roc_auc", "f1", "val_loss"}``.  Determines which key from
        the metrics dict is used for comparison and which direction is
        considered an improvement.
    save_path:
        Full path (including filename) where the best checkpoint will be
        written.  Passed directly to :func:`save_checkpoint`.

    Raises
    ------
    ValueError
        If *save_metric* is not one of the supported values.

    Examples
    --------
    >>> tracker = BestModelTracker("roc_auc", Path("saved_models/best.pth"))
    >>> improved = tracker.update({"roc_auc": 0.82, "f1": 0.71, "val_loss": 0.45},
    ...                           model, optimizer, epoch=0)
    >>> improved
    True
    """

    _HIGHER_IS_BETTER = {"roc_auc", "f1"}
    _LOWER_IS_BETTER = {"val_loss"}
    _VALID_METRICS = _HIGHER_IS_BETTER | _LOWER_IS_BETTER

    def __init__(self, save_metric: str, save_path: Path) -> None:
        if save_metric not in self._VALID_METRICS:
            raise ValueError(
                f"save_metric must be one of {sorted(self._VALID_METRICS)}, "
                f"got {save_metric!r}"
            )
        self.save_metric = save_metric
        self.save_path = Path(save_path)

        # Initialise best value to the worst possible sentinel
        if save_metric in self._HIGHER_IS_BETTER:
            self.best_value: float = float("-inf")
        else:
            self.best_value = float("inf")

    def _is_improvement(self, current: float) -> bool:
        """Return True if *current* is better than the stored best value."""
        if self.save_metric in self._HIGHER_IS_BETTER:
            return current > self.best_value
        else:
            return current < self.best_value

    def update(
        self,
        metrics: dict,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> bool:
        """Conditionally save a checkpoint if the tracked metric improved.

        Parameters
        ----------
        metrics:
            Dictionary returned by :func:`compute_val_metrics`.  Must contain
            the key matching ``self.save_metric``.
        model:
            Model to checkpoint.
        optimizer:
            Optimizer to checkpoint.
        epoch:
            Current epoch index (0-indexed).

        Returns
        -------
        bool
            ``True`` if the metric improved and a checkpoint was saved;
            ``False`` otherwise.
        """
        current = float(metrics[self.save_metric])

        if self._is_improvement(current):
            self.best_value = current
            save_checkpoint(model, optimizer, epoch, current, self.save_path)
            return True

        return False


# ---------------------------------------------------------------------------
# Threshold tuning (Requirements 5.2, 5.3, 5.4)
# ---------------------------------------------------------------------------

def tune_threshold_youden(
    y_true: np.ndarray, y_proba: np.ndarray
) -> tuple[float, float]:
    """Find the optimal classification threshold using Youden's J statistic.

    Sweeps thresholds from 0.01 to 0.99 in steps of 0.01 (99 thresholds).
    For each threshold, computes sensitivity (TPR) and specificity (TNR), then
    Youden's J = sensitivity + specificity - 1.  Returns the threshold that
    maximises J.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels, shape ``(N,)``.
    y_proba:
        Predicted probabilities for the positive class, shape ``(N,)``.

    Returns
    -------
    (optimal_threshold, best_j_score):
        The threshold in [0.01, 0.99] that maximises Youden's J, and the
        corresponding J value.

    Notes
    -----
    Uses numpy for efficiency — no Python-level loop over samples.
    Ties in J are broken by taking the first (lowest) threshold.
    """
    thresholds = np.arange(0.01, 1.00, 0.01)  # 99 values: 0.01, 0.02, ..., 0.99

    y_true = np.asarray(y_true, dtype=np.int32)
    y_proba = np.asarray(y_proba, dtype=np.float64)

    # Broadcast: shape (99, N) — each row is predictions at one threshold
    preds = (y_proba[np.newaxis, :] >= thresholds[:, np.newaxis]).astype(np.int32)

    # True positives, false negatives, false positives, true negatives per threshold
    tp = (preds * y_true[np.newaxis, :]).sum(axis=1).astype(np.float64)
    fn = ((1 - preds) * y_true[np.newaxis, :]).sum(axis=1).astype(np.float64)
    fp = (preds * (1 - y_true)[np.newaxis, :]).sum(axis=1).astype(np.float64)
    tn = ((1 - preds) * (1 - y_true)[np.newaxis, :]).sum(axis=1).astype(np.float64)

    # Sensitivity (TPR) and specificity (TNR) — guard against zero denominators
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        specificity = np.where(tn + fp > 0, tn / (tn + fp), 0.0)

    j_scores = sensitivity + specificity - 1.0

    best_idx = int(np.argmax(j_scores))
    optimal_threshold = float(thresholds[best_idx])
    best_j_score = float(j_scores[best_idx])

    return optimal_threshold, best_j_score


def save_thresholds(threshold: float, calib_roc_auc: float) -> None:
    """Save the optimal classification threshold to ``thresholds.json``.

    Writes ``{"classification_threshold": threshold, "method": "youden_j",
    "calib_roc_auc": calib_roc_auc}`` to
    ``SAVED_MODELS_DIR / "thresholds.json"``.

    Parameters
    ----------
    threshold:
        Optimal threshold returned by :func:`tune_threshold_youden`.
    calib_roc_auc:
        ROC-AUC computed on the calibration split (used for provenance).
    """
    artifact = {
        "classification_threshold": threshold,
        "method": "youden_j",
        "calib_roc_auc": calib_roc_auc,
    }
    save_path = SAVED_MODELS_DIR / "thresholds.json"
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)


def compute_multitask_loss(
    reg_out: torch.Tensor,
    class_out: torch.Tensor,
    reg_labels: torch.Tensor,
    bin_labels: torch.Tensor,
    score_mean: float,
    score_std: float,
    alpha: float = 0.5,
    consistency_weight: float = 0.1,
) -> torch.Tensor:
    """Compute the combined multi-task training loss.

    The loss combines three terms:

    1. **MSE** — mean squared error between the regression output and the
       z-score-normalised regression labels.
    2. **BCE** — binary cross-entropy with logits between the classification
       output and the binary labels.
    3. **Consistency** — a soft penalty that discourages disagreement between
       the regression and classification heads (Requirement 9.1).

    The final loss is::

        alpha * mse + (1 - alpha) * bce + consistency_weight * consistency

    Parameters
    ----------
    reg_out:
        Regression head output, shape ``(B, 1)`` or ``(B,)``.
    class_out:
        Classification head output (logits), shape ``(B, 1)`` or ``(B,)``.
    reg_labels:
        Raw (un-normalised) regression targets, shape ``(B,)``.
    bin_labels:
        Binary classification targets (0 or 1), shape ``(B,)``.
    score_mean:
        Mean of training scores from :func:`compute_score_normalization`.
    score_std:
        Standard deviation of training scores from
        :func:`compute_score_normalization`.
    alpha:
        Weight for the MSE term.  The BCE term receives weight ``1 - alpha``.
        Default ``0.5``.
    consistency_weight:
        Weight for the consistency penalty term.  Default ``0.1``.

    Returns
    -------
    torch.Tensor
        Scalar loss tensor.
    """
    # Flatten to (B,) for loss functions
    reg_out = reg_out.squeeze(-1)
    class_out = class_out.squeeze(-1)

    # --- z-score normalise regression labels (Requirement 8.2) ---
    reg_labels_norm = (reg_labels - score_mean) / score_std

    # --- MSE loss on normalised targets ---
    mse = F.mse_loss(reg_out, reg_labels_norm)

    # --- BCE with logits loss ---
    bce = F.binary_cross_entropy_with_logits(class_out, bin_labels.float())

    # --- Consistency loss (Requirement 9.1, 9.2) ---
    # Penalise cases where:
    #   - reg_out >= 1.0 but sigmoid(class_out) < 0.5  (regressor says compatible,
    #     classifier disagrees)
    #   - reg_out < 1.0  but sigmoid(class_out) >= 0.5 (classifier says compatible,
    #     regressor disagrees)
    sig = torch.sigmoid(class_out)
    penalty_a = F.relu(reg_out - 1.0) * F.relu(0.5 - sig)
    penalty_b = F.relu(1.0 - reg_out) * F.relu(sig - 0.5)
    consistency = (penalty_a + penalty_b).mean()

    return alpha * mse + (1.0 - alpha) * bce + consistency_weight * consistency

def save_model_card(args, splits_info: dict, best_roc_auc: float, best_f1: float) -> None:
    """Save training_config.json and artifact_manifest.json (Requirement 30)."""
    import datetime

    timestamp = datetime.datetime.utcnow().isoformat() + "Z"

    training_config = {
        "seed": args.seed,
        "feature_set": args.feature_set,
        "binary_label_threshold": 1.0,
        "binary_label_threshold_justification": (
            "Score >= 1.0 yields ~58% positive rate (reasonably balanced). "
            "Validated by threshold sensitivity analysis."
        ),
        "model": {
            "visual_dim": 512,
            "phys_dim": 64,
            "hidden_dim": 256,
            "nhead": 8,
            "num_layers": 2,
        },
        "training": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "alpha": args.alpha,
            "consistency_weight": args.consistency_weight,
        },
        "splits": splits_info,
        "save_metric": args.save_metric,
        "best_val_roc_auc": best_roc_auc,
        "best_val_f1": best_f1,
        "timestamp": timestamp,
    }

    config_path = SAVED_MODELS_DIR / "training_config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(training_config, fh, indent=2)
    print(f"  Saved training_config.json")

    artifact_manifest = {
        "timestamp": timestamp,
        "artifacts": [
            {"filename": "best_hybrid_model.pth",    "purpose": "Best model checkpoint (selected by save_metric)"},
            {"filename": "phys_feature_cols.json",   "purpose": "Ordered physical feature column names for inference"},
            {"filename": "encoder_mapping.json",     "purpose": "Categorical encoding schema (one-hot column names)"},
            {"filename": "thresholds.json",          "purpose": "Optimal classification threshold (Youden's J on calib split)"},
            {"filename": "score_normalization.json", "purpose": "Score mean/std for z-score normalization and inverse-transform"},
            {"filename": "training_config.json",     "purpose": "Model card: hyperparameters, splits, metrics, timestamp"},
            {"filename": "artifact_manifest.json",   "purpose": "This file — lists all artifacts and their purposes"},
            {"filename": "training_history.json",    "purpose": "Per-epoch train/val loss, ROC-AUC, F1, LR for plotting"},
        ],
    }

    manifest_path = SAVED_MODELS_DIR / "artifact_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(artifact_manifest, fh, indent=2)
    print(f"  Saved artifact_manifest.json")


def main() -> None:
    """Full training loop: data loading, training, threshold tuning, artifact saving."""
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from scripts.data_utils import O4UHybridDataset, collate_fn, load_harmony_scores
    from scripts.models import HybridFashionModel

    parser = build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Reproducibility (Requirements 21.1, 21.2, 21.3)
    # ------------------------------------------------------------------
    seed_everything(args.seed)
    dl_generator = make_dataloader_generator(args.seed)

    # ------------------------------------------------------------------
    # Load encoder artifacts (Requirements 1.1, 1.2, 33.3)
    # ------------------------------------------------------------------
    phys_feature_cols = load_json_artifact("phys_feature_cols.json")
    encoder_mapping = load_json_artifact("encoder_mapping.json")

    # Apply feature ablation if requested (Requirement 7)
    active_feature_cols = apply_feature_ablation(phys_feature_cols, args.feature_set)

    print("=" * 60)
    print("O4U HybridFashionModel — Training")
    print("=" * 60)
    print(f"  seed={args.seed}  save_metric={args.save_metric}  feature_set={args.feature_set}")
    print(f"  epochs={args.epochs}  warmup={args.warmup_epochs}  lr={args.lr}  batch={args.batch_size}")
    print(f"  alpha={args.alpha}  consistency_weight={args.consistency_weight}")
    print(f"  Physical features: {len(active_feature_cols)} cols")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load imputed manifests and add binary labels
    # ------------------------------------------------------------------
    df_train_full = pd.read_json(TRAIN_IMPUTED_MANIFEST)
    df_val        = pd.read_json(VAL_IMPUTED_MANIFEST)

    BINARY_THRESHOLD = 1.0
    df_train_full["binary_label"] = (df_train_full["score"] >= BINARY_THRESHOLD).astype(int)
    df_val["binary_label"]        = (df_val["score"] >= BINARY_THRESHOLD).astype(int)

    # ------------------------------------------------------------------
    # Data splits: calibration / deep-train / ensemble-train (Requirements 5, 6)
    # ------------------------------------------------------------------
    deep_train, calib, ensemble_train = make_splits(df_train_full, seed=args.seed)
    print(f"  Splits — deep_train: {len(deep_train):,}  calib: {len(calib):,}  "
          f"ensemble_train: {len(ensemble_train):,}  val: {len(df_val):,}")

    splits_info = {
        "deep_train_n": len(deep_train),
        "calib_n": len(calib),
        "ensemble_train_n": len(ensemble_train),
        "val_n": len(df_val),
    }

    # ------------------------------------------------------------------
    # Score normalization (Requirements 8.1, 8.2)
    # ------------------------------------------------------------------
    score_mean, score_std = compute_score_normalization(deep_train)
    save_score_normalization(score_mean, score_std)
    print(f"  Score normalization — mean={score_mean:.4f}  std={score_std:.4f}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    # ------------------------------------------------------------------
    # DataLoaders (Requirements 21.3)
    # ------------------------------------------------------------------
    features_dir = str(FEATURES_DIR)

    # Load color harmony scores (optional — falls back to neutral if file missing)
    harmony_scores = load_harmony_scores(
        str(DATA_PROCESSED_DIR / "color_harmony_scores.json")
    )
    harmony_dim = 4 if harmony_scores else 0  # warm, cool, neutral, was_imputed
    if harmony_scores:
        print(f"  Color harmony scores loaded: {len(harmony_scores):,} outfits (+{harmony_dim} dims)")
    else:
        print("  Color harmony scores not found — skipping harmony features")

    train_dataset = O4UHybridDataset(deep_train, features_dir, active_feature_cols, harmony_scores=harmony_scores)
    val_dataset   = O4UHybridDataset(df_val,     features_dir, active_feature_cols, harmony_scores=harmony_scores)
    calib_dataset = O4UHybridDataset(calib,      features_dir, active_feature_cols, harmony_scores=harmony_scores)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, generator=dl_generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    calib_loader = DataLoader(
        calib_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ------------------------------------------------------------------
    # Model, optimizer, scheduler
    # ------------------------------------------------------------------
    phys_input_dim = len(active_feature_cols) + harmony_dim
    model = HybridFashionModel(phys_input_dim=phys_input_dim).to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = build_lr_scheduler(optimizer, args.warmup_epochs, args.epochs, args.lr)

    checkpoint_path = SAVED_MODELS_DIR / "best_hybrid_model.pth"
    tracker = BestModelTracker(args.save_metric, checkpoint_path)

    # Early stopping state
    epochs_no_improve = 0
    early_stop = args.patience > 0
    if early_stop:
        print(f"  Early stopping: patience={args.patience} epochs (metric: {args.save_metric})")
    else:
        print("  Early stopping: disabled")

    # ------------------------------------------------------------------
    # Training loop (Requirements 19, 20, 22, 9.3)
    # ------------------------------------------------------------------
    print("\nStarting training...\n")
    training_history = []  # list of per-epoch dicts for loss curve plotting
    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)

        for visual_padded, visual_mask, phys_vecs, reg_labels, bin_labels in pbar:
            visual_padded = visual_padded.to(device)
            visual_mask   = visual_mask.to(device)
            phys_vecs     = phys_vecs.to(device)
            reg_labels    = reg_labels.to(device)
            bin_labels    = bin_labels.to(device)

            optimizer.zero_grad()
            reg_out, class_out = model(visual_padded, visual_mask, phys_vecs)

            loss = compute_multitask_loss(
                reg_out, class_out, reg_labels, bin_labels,
                score_mean=score_mean, score_std=score_std,
                alpha=args.alpha, consistency_weight=args.consistency_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)

        # --- Validation ---
        val_metrics = compute_val_metrics(
            model, val_loader, device,
            threshold=0.5, score_mean=score_mean, score_std=score_std,
        )

        # --- Consistency fraction (Requirement 9.3) ---
        model.eval()
        all_reg, all_cls = [], []
        with torch.no_grad():
            for visual_padded, visual_mask, phys_vecs, reg_labels, bin_labels in val_loader:
                ro, co = model(
                    visual_padded.to(device),
                    visual_mask.to(device),
                    phys_vecs.to(device),
                )
                all_reg.append(ro.squeeze(-1).cpu().numpy())
                all_cls.append(co.squeeze(-1).cpu().numpy())
        consistency_frac = compute_consistency_fraction(
            np.concatenate(all_reg), np.concatenate(all_cls)
        )

        # --- LR step ---
        scheduler.step()
        current_lr = get_current_lr(optimizer)

        # --- Log ---
        improved = tracker.update(val_metrics, model, optimizer, epoch)
        improved_marker = " ← best" if improved else ""
        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"roc_auc={val_metrics['roc_auc']:.4f} | "
            f"f1={val_metrics['f1']:.4f} | "
            f"inconsistency={consistency_frac:.3f} | "
            f"lr={current_lr:.2e}"
            f"{improved_marker}"
        )

        # Record history for loss curve
        training_history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["val_loss"],
            "roc_auc": val_metrics["roc_auc"],
            "f1": val_metrics["f1"],
            "consistency_frac": consistency_frac,
            "lr": current_lr,
            "is_best": improved,
        })

        # --- Early stopping ---
        if early_stop:
            if improved:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(
                        f"\nEarly stopping triggered: no improvement in "
                        f"{args.save_metric} for {args.patience} consecutive epochs."
                    )
                    break

    # Save training history for plotting in notebooks
    history_path = SAVED_MODELS_DIR / "training_history.json"
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(training_history, fh, indent=2)
    print(f"\nTraining history saved to {history_path}")
    print(f"Training complete. Best {args.save_metric}: {tracker.best_value:.4f}")

    # ------------------------------------------------------------------
    # Threshold tuning on calibration split (Requirements 5.2, 5.3, 5.4)
    # ------------------------------------------------------------------
    print("\nTuning threshold on calibration split (Youden's J)...")

    # Load best checkpoint for threshold tuning
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    from sklearn.metrics import roc_auc_score
    calib_proba, calib_true = [], []
    with torch.no_grad():
        for visual_padded, visual_mask, phys_vecs, reg_labels, bin_labels in calib_loader:
            _, class_out = model(
                visual_padded.to(device),
                visual_mask.to(device),
                phys_vecs.to(device),
            )
            proba = torch.sigmoid(class_out.squeeze(-1)).cpu().numpy()
            calib_proba.extend(proba.tolist())
            calib_true.extend(bin_labels.numpy().tolist())

    calib_proba = np.array(calib_proba)
    calib_true  = np.array(calib_true, dtype=np.int32)
    calib_roc_auc = float(roc_auc_score(calib_true, calib_proba))

    optimal_threshold, best_j = tune_threshold_youden(calib_true, calib_proba)
    save_thresholds(optimal_threshold, calib_roc_auc)
    print(f"  Optimal threshold: {optimal_threshold:.4f}  (Youden's J={best_j:.4f})")
    print(f"  Calibration ROC-AUC: {calib_roc_auc:.4f}")

    # ------------------------------------------------------------------
    # Per-group metrics on validation split (Requirements 22.1, 22.2)
    # ------------------------------------------------------------------
    print("\nComputing per-group metrics on validation split...")
    val_proba, val_true_labels = [], []
    with torch.no_grad():
        for visual_padded, visual_mask, phys_vecs, reg_labels, bin_labels in val_loader:
            _, class_out = model(
                visual_padded.to(device),
                visual_mask.to(device),
                phys_vecs.to(device),
            )
            proba = torch.sigmoid(class_out.squeeze(-1)).cpu().numpy()
            val_proba.extend(proba.tolist())
            val_true_labels.extend(bin_labels.numpy().tolist())

    val_proba_arr = np.array(val_proba)

    if "body_figure" in df_val.columns:
        pg_metrics = compute_per_group_metrics(df_val, val_proba_arr, threshold=optimal_threshold)
        print(f"  Body figure groups: {list(pg_metrics.get('by_body_figure', {}).keys())}")
    else:
        pg_metrics = {}
        print("  body_figure column not found in val manifest — skipping per-group metrics")

    # ------------------------------------------------------------------
    # Ablation results (Requirement 7.3)
    # ------------------------------------------------------------------
    final_val_metrics = compute_val_metrics(
        model, val_loader, device,
        threshold=optimal_threshold, score_mean=score_mean, score_std=score_std,
    )
    save_ablation_results(final_val_metrics["roc_auc"], final_val_metrics["f1"], args.feature_set)

    # ------------------------------------------------------------------
    # Model card artifacts (Requirements 30.1, 30.2, 30.3)
    # ------------------------------------------------------------------
    print("\nSaving model card artifacts...")
    save_model_card(args, splits_info, tracker.best_value,
                    final_val_metrics["f1"] if args.save_metric != "f1" else tracker.best_value)

    print("\nAll artifacts saved:")
    for name in ["best_hybrid_model.pth", "score_normalization.json",
                 "thresholds.json", "training_config.json", "artifact_manifest.json"]:
        path = SAVED_MODELS_DIR / name
        status = "OK" if path.exists() else "MISSING"
        print(f"  {status:8s}  {name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
