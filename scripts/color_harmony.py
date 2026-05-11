"""
Color harmony preprocessing script for O4U.

Extracts dominant colors from garment images per outfit, classifies them as
warm / cool / neutral using HSV hue, then computes a 3-value harmony vector
(warm_fraction, cool_fraction, neutral_fraction) relative to the person's
skin_color recommendations from the CSV lookup table.

Output: data/processed/color_harmony_scores.json
    {
        "<outfit_id>": {
            "warm": 0.6,
            "cool": 0.1,
            "neutral": 0.3,
            "was_imputed": 0   # 1 if skin_color was unknown/missing
        },
        ...
    }

Usage:
    python3 -m scripts.color_harmony
    python3 -m scripts.color_harmony --n-colors 3 --img-size 64
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from scripts.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    PROJECT_ROOT,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
    TEST_MANIFEST,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
IMAGE_DIR = DATA_RAW_DIR / "Outfit4You" / "image"
CSV_PATH = Path("/Users/hazron/Downloads/recommendations.csv")
OUTPUT_PATH = DATA_PROCESSED_DIR / "color_harmony_scores.json"

# ---------------------------------------------------------------------------
# O4U skin_color → CSV Skin Tone mapping
# O4U values: yellow, dark, fair, brown, unknown
# CSV values: Very Fair, Fair, Medium, Olive, Brown, Very Dark
# ---------------------------------------------------------------------------
O4U_SKIN_TO_CSV = {
    "yellow":  "Medium",    # East Asian yellow-undertone maps to medium
    "fair":    "Fair",
    "brown":   "Brown",
    "dark":    "Very Dark",
    "unknown": None,        # will trigger was_imputed=1
}

# When a person has multiple skin_color labels (e.g. "yellow,brown"),
# we take the first recognized token.

# ---------------------------------------------------------------------------
# Color temperature classification via HSV hue
#
# Hue wheel (0-360):
#   Warm: reds, oranges, yellows         → [0,60) ∪ [300,360)
#   Cool: greens, blues, purples         → [120,300)
#   Neutral: yellow-greens, near-white,
#            near-black, low-saturation  → [60,120) or low saturation
#
# Saturation threshold: pixels with S < 0.15 are neutral regardless of hue.
# ---------------------------------------------------------------------------
WARM_HUE_RANGES = [(0, 60), (300, 360)]
COOL_HUE_RANGE  = (120, 300)
# [60,120) is yellow-green — treated as neutral (ambiguous)
SAT_NEUTRAL_THRESHOLD = 0.15


def classify_hue(h_deg: float, s: float) -> str:
    """Return 'warm', 'cool', or 'neutral' for a single HSV pixel."""
    if s < SAT_NEUTRAL_THRESHOLD:
        return "neutral"
    for lo, hi in WARM_HUE_RANGES:
        if lo <= h_deg < hi:
            return "warm"
    if COOL_HUE_RANGE[0] <= h_deg < COOL_HUE_RANGE[1]:
        return "cool"
    return "neutral"


def dominant_colors_temperature(img_path: str, n_colors: int = 3, img_size: int = 64):
    """
    Extract dominant color temperatures from a garment image.

    Returns (warm_frac, cool_frac, neutral_frac) as floats summing to 1.0,
    or None if the image cannot be loaded.
    """
    try:
        from PIL import Image
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(f"Requires Pillow and scikit-learn: {e}")

    try:
        img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
    except Exception:
        return None

    pixels = np.array(img).reshape(-1, 3).astype(np.float32) / 255.0

    # k-means to find dominant colors
    k = min(n_colors, len(pixels))
    km = KMeans(n_clusters=k, n_init=3, random_state=42)
    km.fit(pixels)

    centers = km.cluster_centers_          # (k, 3) RGB in [0,1]
    counts  = np.bincount(km.labels_, minlength=k)
    weights = counts / counts.sum()        # cluster weight by pixel count

    warm = cool = neutral = 0.0
    for center, w in zip(centers, weights):
        r, g, b = center
        # Convert RGB → HSV
        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin

        # Saturation
        s = (delta / cmax) if cmax > 0 else 0.0

        # Hue
        if delta == 0:
            h = 0.0
        elif cmax == r:
            h = 60.0 * (((g - b) / delta) % 6)
        elif cmax == g:
            h = 60.0 * (((b - r) / delta) + 2)
        else:
            h = 60.0 * (((r - g) / delta) + 4)

        temp = classify_hue(h, s)
        if temp == "warm":
            warm += w
        elif temp == "cool":
            cool += w
        else:
            neutral += w

    return float(warm), float(cool), float(neutral)


# ---------------------------------------------------------------------------
# CSV lookup: skin_tone → recommended color temperature
# ---------------------------------------------------------------------------

def build_skin_tone_lookup(csv_path: str) -> dict:
    """
    Build a dict: skin_tone_str → {"warm": float, "cool": float, "neutral": float}
    representing the recommended color temperature profile for that skin tone.

    Derived from the CSV's 'Recommended Clothing Colors' column:
      - "Earth Tones, Olive, Coral, Peach, Mustard, Warm Red" → warm profile
      - "Jewel Tones, Icy Blue, Lavender, Silver, Emerald"    → cool profile
      - "Soft Pinks, Plums, Teal, Neutral Beige"              → neutral/mixed profile
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Map recommendation string → temperature profile
    REC_TO_PROFILE = {
        "Earth Tones, Olive, Coral, Peach, Mustard, Warm Red": (0.7, 0.1, 0.2),
        "Jewel Tones, Icy Blue, Lavender, Silver, Emerald":    (0.1, 0.7, 0.2),
        "Soft Pinks, Plums, Teal, Neutral Beige":              (0.2, 0.3, 0.5),
    }

    lookup = {}
    for skin_tone in df["Skin Tone"].unique():
        # Take the most common recommendation for this skin tone
        subset = df[df["Skin Tone"] == skin_tone]
        most_common_rec = subset["Recommended Clothing Colors"].mode()[0]
        profile = REC_TO_PROFILE.get(most_common_rec, (0.33, 0.33, 0.34))
        lookup[skin_tone] = {
            "warm":    profile[0],
            "cool":    profile[1],
            "neutral": profile[2],
        }

    return lookup


def parse_o4u_skin_color(raw_value) -> str | None:
    """
    Parse O4U skin_color field (may be multi-label like 'yellow,brown')
    and return the first recognized CSV skin tone string, or None.
    """
    if not raw_value or str(raw_value).strip().lower() in ("", "nan", "none", "unknown"):
        return None

    tokens = [t.strip().lower() for t in str(raw_value).split(",")]
    for token in tokens:
        csv_tone = O4U_SKIN_TO_CSV.get(token)
        if csv_tone is not None:
            return csv_tone

    return None


def compute_harmony_score(
    outfit_warm: float,
    outfit_cool: float,
    outfit_neutral: float,
    recommended: dict,
) -> tuple[float, float, float]:
    """
    Compute harmony as dot-product similarity between outfit color distribution
    and recommended color profile, then re-normalize to sum to 1.

    Returns (harmony_warm, harmony_cool, harmony_neutral).
    """
    hw = outfit_warm    * recommended["warm"]
    hc = outfit_cool    * recommended["cool"]
    hn = outfit_neutral * recommended["neutral"]

    total = hw + hc + hn
    if total == 0:
        return (0.33, 0.33, 0.34)

    return (hw / total, hc / total, hn / total)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def collect_outfit_records(*manifest_paths) -> list[dict]:
    """Load and merge records from multiple manifest JSON files."""
    records = []
    seen_ids = set()
    for path in manifest_paths:
        p = Path(path)
        if not p.exists():
            print(f"  Warning: manifest not found: {path}")
            continue
        with open(p) as f:
            data = json.load(f)
        for rec in data:
            oid = str(rec["id"])
            if oid not in seen_ids:
                records.append(rec)
                seen_ids.add(oid)
    return records


def run(n_colors: int = 3, img_size: int = 64) -> None:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    print("=" * 60)
    print("O4U Color Harmony Preprocessing")
    print("=" * 60)

    # Build skin tone lookup from CSV
    print(f"  Loading CSV from: {CSV_PATH}")
    skin_tone_lookup = build_skin_tone_lookup(str(CSV_PATH))
    print(f"  Skin tone profiles: {list(skin_tone_lookup.keys())}")

    # Collect all outfit records across splits
    records = collect_outfit_records(
        TRAIN_MANIFEST,
        VAL_MANIFEST,
        TEST_MANIFEST,
    )
    print(f"  Total outfits to process: {len(records):,}")

    results = {}
    n_imputed = 0
    n_missing_images = 0

    for rec in tqdm(records, desc="Processing outfits"):
        outfit_id = str(rec["id"])

        # --- Collect garment images for this outfit ---
        item_keys = [f"item_{i}" for i in range(1, 10)]
        image_paths = []
        for k in item_keys:
            fname = rec.get(k, "")
            if fname:
                p = IMAGE_DIR / fname
                if p.exists():
                    image_paths.append(str(p))

        if not image_paths:
            n_missing_images += 1
            results[outfit_id] = {
                "warm": 0.33, "cool": 0.33, "neutral": 0.34, "was_imputed": 1
            }
            continue

        # --- Extract dominant color temperatures across all garment images ---
        all_warm = all_cool = all_neutral = 0.0
        valid = 0
        for img_path in image_paths:
            result = dominant_colors_temperature(img_path, n_colors=n_colors, img_size=img_size)
            if result is not None:
                w, c, n = result
                all_warm    += w
                all_cool    += c
                all_neutral += n
                valid += 1

        if valid == 0:
            results[outfit_id] = {
                "warm": 0.33, "cool": 0.33, "neutral": 0.34, "was_imputed": 1
            }
            n_missing_images += 1
            continue

        outfit_warm    = all_warm    / valid
        outfit_cool    = all_cool    / valid
        outfit_neutral = all_neutral / valid

        # --- Determine person's skin tone ---
        raw_skin = rec.get("skin_color", "")
        csv_tone = parse_o4u_skin_color(raw_skin)
        was_imputed = 0

        if csv_tone is None or csv_tone not in skin_tone_lookup:
            # Unknown skin tone — use neutral fallback
            harmony = (0.33, 0.33, 0.34)
            was_imputed = 1
            n_imputed += 1
        else:
            recommended = skin_tone_lookup[csv_tone]
            harmony = compute_harmony_score(
                outfit_warm, outfit_cool, outfit_neutral, recommended
            )

        results[outfit_id] = {
            "warm":        round(harmony[0], 6),
            "cool":        round(harmony[1], 6),
            "neutral":     round(harmony[2], 6),
            "was_imputed": was_imputed,
        }

    # Save output
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone.")
    print(f"  Outfits processed:      {len(results):,}")
    print(f"  Skin tone imputed:      {n_imputed:,} ({100*n_imputed/max(len(results),1):.1f}%)")
    print(f"  Missing images:         {n_missing_images:,}")
    print(f"  Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="O4U Color Harmony Preprocessing")
    parser.add_argument("--n-colors", type=int, default=3,
                        help="Number of dominant colors to extract per image (k-means k)")
    parser.add_argument("--img-size", type=int, default=64,
                        help="Resize images to this square size before k-means (smaller = faster)")
    args = parser.parse_args()
    run(n_colors=args.n_colors, img_size=args.img_size)
