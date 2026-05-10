import os
import glob
import random
import json
import torch
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

try:
    from scripts.config import SAVED_MODELS_DIR, FEATURES_DIR as _FEATURES_DIR, DATA_PROCESSED_DIR
    from scripts.models import HybridFashionModel
except ImportError:
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from config import SAVED_MODELS_DIR, FEATURES_DIR as _FEATURES_DIR, DATA_PROCESSED_DIR
    from models import HybridFashionModel


MODEL_PATH = str(SAVED_MODELS_DIR / "best_hybrid_model.pth")
FEATURES_DIR = str(_FEATURES_DIR)


def build_physical_vector(user_attrs, phys_feature_cols, bf_cols, cat_cols):
    """Build a one-hot physical feature vector from user-provided attributes.

    Args:
        user_attrs: dict with keys like "body_figure", "skin_color", "height", etc.
        phys_feature_cols: ordered list of feature column names (from training).
        bf_cols: list of bf_* column names.
        cat_cols: list of categorical column names.

    Returns:
        torch.Tensor of shape (phys_dim,)
    """
    # Normalize all string values to lowercase before any key lookup (Requirement 2.3)
    user_attrs = {k: v.lower() if isinstance(v, str) else v for k, v in user_attrs.items()}

    vec = np.zeros(len(phys_feature_cols), dtype=np.float32)
    col_to_idx = {col: i for i, col in enumerate(phys_feature_cols)}

    # Body figure (multi-label binarized)
    body_figures = user_attrs.get("body_figure", "unknown")
    if isinstance(body_figures, str):
        body_figures = [v.strip().lower() for v in body_figures.split(",")]
    for bf in body_figures:
        key = f"bf_{bf}"
        if key in col_to_idx:
            vec[col_to_idx[key]] = 1.0

    # Categorical one-hot
    for cat in cat_cols:
        val = user_attrs.get(cat, "")
        if val:
            key = f"{cat}_{val.strip().lower()}"
            if key in col_to_idx:
                vec[col_to_idx[key]] = 1.0

    return torch.tensor(vec, dtype=torch.float32)


def load_feature_artifacts():
    """Load phys_feature_cols and encoder_mapping from saved_models/.

    Returns:
        phys_feature_cols: ordered list of physical feature column names
        bf_cols: list of bf_* column names
        cat_cols: list of categorical column names
        phys_dim: length of phys_feature_cols

    Raises:
        FileNotFoundError: if either artifact file is missing
    """
    for filename in ("phys_feature_cols.json", "encoder_mapping.json"):
        artifact_path = SAVED_MODELS_DIR / filename
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Missing artifact: {filename} — run training to generate it"
            )

    phys_feature_cols_path = SAVED_MODELS_DIR / "phys_feature_cols.json"
    encoder_mapping_path = SAVED_MODELS_DIR / "encoder_mapping.json"

    with open(phys_feature_cols_path, "r") as f:
        phys_feature_cols = json.load(f)

    with open(encoder_mapping_path, "r") as f:
        encoder_mapping = json.load(f)

    cat_cols = encoder_mapping.get("cat_cols", [])
    bf_cols = [c for c in phys_feature_cols if c.startswith("bf_")]

    return phys_feature_cols, bf_cols, cat_cols, len(phys_feature_cols)


def digital_wardrobe_inference(user_attrs=None, num_outfits=5):
    """Run inference on random outfits for a given user's physical attributes.

    Args:
        user_attrs: dict with physical attributes, e.g.:
            {"body_figure": "pear", "skin_color": "brown", "height": "medium"}
            If None, defaults to {"body_figure": "unknown"}.
        num_outfits: number of random outfits to score.
    """
    if user_attrs is None:
        user_attrs = {"body_figure": "unknown"}

    print("--- Fashion Cognitive Inference System ---\n")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load feature columns from saved artifacts
    phys_feature_cols, bf_cols, cat_cols, phys_dim = load_feature_artifacts()

    # Build physical vector for this user
    phys_vec = build_physical_vector(user_attrs, phys_feature_cols, bf_cols, cat_cols)
    phys_batch = phys_vec.unsqueeze(0).to(device)
    print(f"User attributes: {user_attrs}")
    print(f"Physical vector dim: {phys_dim}")

    # Load model — handle both raw state_dict and checkpoint dict formats
    model = HybridFashionModel(phys_input_dim=phys_dim).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded.\n")

    # Load score normalization parameters for inverse-transform (Requirement 8.3)
    score_norm_path = SAVED_MODELS_DIR / "score_normalization.json"
    if not score_norm_path.exists():
        raise FileNotFoundError(
            "Missing artifact: score_normalization.json — run training to generate it"
        )
    with open(score_norm_path, "r") as f:
        score_norm = json.load(f)
    score_mean = score_norm["mean"]
    score_std = score_norm["std"]

    # Load val/test outfit ID allowlist
    val_test_ids_path = DATA_PROCESSED_DIR / "val_test_outfit_ids.json"
    if not val_test_ids_path.exists():
        raise FileNotFoundError(
            "val_test_outfit_ids.json not found in data/processed/ — "
            "run the imputation pipeline to generate it"
        )
    with open(val_test_ids_path, "r") as f:
        val_test_outfit_ids = set(json.load(f))

    # Sample random outfits restricted to val/test IDs
    pt_files = glob.glob(os.path.join(FEATURES_DIR, "*.pt"))
    pt_files = [
        f for f in pt_files
        if os.path.basename(f).replace(".pt", "") in val_test_outfit_ids
    ]
    if not pt_files:
        raise RuntimeError(
            "Outfit ID restriction produced an empty sample pool — "
            "check val_test_outfit_ids.json and the features directory"
        )
    sample_files = random.sample(pt_files, min(num_outfits, len(pt_files)))

    outfits = []
    outfit_ids = []
    for f in sample_files:
        outfit_id = os.path.basename(f).replace(".pt", "")
        feat = torch.load(f, map_location="cpu", weights_only=True)
        outfits.append(feat)
        outfit_ids.append(outfit_id)

    print(f"Scoring {len(outfits)} outfits...\n")

    # Pad and batch
    visual_padded = pad_sequence(outfits, batch_first=True, padding_value=0.0).to(device)
    lengths = torch.tensor([feat.size(0) for feat in outfits], dtype=torch.long)
    max_len = visual_padded.size(1)
    visual_mask = (torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)).to(device)

    # Expand physical vector for all outfits in batch
    phys_batch_expanded = phys_vec.unsqueeze(0).expand(len(outfits), -1).to(device)

    # Forward pass
    with torch.inference_mode():
        reg_out, class_out = model(visual_padded, visual_mask, phys_batch_expanded)
        probs = torch.sigmoid(class_out).squeeze().cpu().numpy()
        # Inverse-transform regression output from z-score back to original scale (Requirement 8.3)
        scores = reg_out.squeeze().cpu().numpy() * score_std + score_mean

    # Handle single-element case
    if len(outfits) == 1:
        probs = np.array([probs.item()])
        scores = np.array([scores.item()])

    # Sort by probability
    results = sorted(
        [{"id": outfit_ids[i], "prob": float(probs[i]), "score": float(scores[i])} for i in range(len(outfits))],
        key=lambda x: x["prob"],
        reverse=True,
    )

    print("--- RESULTS ---")
    for res in results:
        status = "GOOD FIT" if res["prob"] >= 0.5 else "BAD FIT"
        print(f"  Outfit {res['id']:<8} | {status} ({res['prob']*100:5.1f}%) | Score: {res['score']:.2f}")

    print(f"\nTop recommendation: Outfit #{results[0]['id']} ({results[0]['prob']*100:.1f}% compatible)")


if __name__ == "__main__":
    import sys

    # Parse CLI args as key=value pairs
    attrs = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            attrs[k] = v
        elif not attrs:
            attrs["body_figure"] = arg

    if not attrs:
        attrs = {"body_figure": "pear", "height": "medium"}

    digital_wardrobe_inference(user_attrs=attrs, num_outfits=5)
