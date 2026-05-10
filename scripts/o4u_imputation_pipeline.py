import argparse
import json
import numpy as np
import pandas as pd
import warnings
import os
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier

# Task 4.4: Import paths from scripts.config
from scripts.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    SAVED_MODELS_DIR,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
    TEST_MANIFEST,
    TRAIN_IMPUTED_MANIFEST,
    VAL_IMPUTED_MANIFEST,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PHYSICAL_COLS = [
    "body_figure", "skin_color", "hair_style",
    "hair_color", "height", "breasts", "color_contrast"
]

# Columns for which _was_imputed indicator columns are added (Requirement 12.1, 12.2)
IMPUTED_COLS = [
    "height", "breasts", "skin_color", "color_contrast", "hair_style", "hair_color",
    "height_was_imputed", "breasts_was_imputed", "skin_color_was_imputed",
    "color_contrast_was_imputed", "hair_style_was_imputed", "hair_color_was_imputed",
]

# Categorical columns that get one-hot encoded via pd.get_dummies
CAT_COLS = ["skin_color", "hair_style", "hair_color", "height", "breasts", "color_contrast"]


def load_and_normalize_raw(path: str) -> pd.DataFrame:
    """Load RAW JSON and normalize empty/null values to NaN, then lowercase all string values"""
    print(f"  → Loading Raw: {path}")
    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # Ensure ID is string for merging
    df['id'] = df['id'].astype(str)
    # Only keep ID and Physical columns from RAW
    keep_cols = ['id'] + PHYSICAL_COLS
    df = df[df.columns.intersection(keep_cols)]
    df[PHYSICAL_COLS] = df[PHYSICAL_COLS].replace({"": pd.NA, "null": pd.NA, None: pd.NA})
    # Lowercase all string values in PHYSICAL_COLS (Requirement 2.1, 2.4)
    for col in PHYSICAL_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.lower()
    return df


def process_body_figure(df_train, df_val):
    """Binarize body_figure consistently across train and val"""
    print("\n[STEP 2] Processing body_figure (Multi-Label Binarize)")

    for df in [df_train, df_val]:
        df["body_figure_clean"] = df["body_figure"].fillna("unknown")
        # Lowercase all body figure labels before fitting MultiLabelBinarizer (Requirement 2.2)
        df["_body_figure_list"] = df["body_figure_clean"].str.split(",").apply(
            lambda x: [v.strip().lower() for v in x if v.strip()]
        )

    # Fit on TRAIN labels only (prevent val leakage)
    mlb = MultiLabelBinarizer()
    mlb.fit(df_train["_body_figure_list"])

    print(f"  ✓ Found {len(mlb.classes_)} unique body figure classes")

    # Transform both
    def get_dummies(df):
        dummies = pd.DataFrame(
            mlb.transform(df["_body_figure_list"]),
            columns=[f"bf_{cls}" for cls in mlb.classes_],
            index=df.index
        )
        return pd.concat([df, dummies], axis=1).drop(columns=["_body_figure_list"])

    return get_dummies(df_train), get_dummies(df_val), mlb


def impute_categorical_group_mode(df_train, df_val, col, fallback_val):
    """Impute categorical column using group mode (by body_figure_clean)"""
    print(f"  → Imputing {col} via group mode")

    # Lowercase col values before group lookup (Requirement 2.1)
    for df in [df_train, df_val]:
        if df[col].dtype == object:
            df[col] = df[col].str.lower()

    group_modes = df_train.groupby("body_figure_clean")[col].apply(
        lambda x: x.dropna().mode()[0] if not x.dropna().mode().empty else fallback_val
    ).to_dict()
    # Ensure group_modes keys are lowercase (body_figure_clean is already lowercased via process_body_figure)
    group_modes = {k.lower(): v for k, v in group_modes.items()}

    for df in [df_train, df_val]:
        df[col] = df.apply(
            lambda row: group_modes.get(row["body_figure_clean"].lower(), fallback_val)
            if pd.isna(row[col]) else row[col], axis=1
        )
        df[col] = df[col].fillna(fallback_val)


def impute_mice(df_train, df_val, random_state: int = 42):
    """MICE Imputation for height & breasts, fitting on train only"""
    print(f"\n[STEP 6] MICE Imputation for height & breasts (seed={random_state})")

    target_cols = ["height", "breasts"]
    bf_cols = [c for c in df_train.columns if c.startswith("bf_")]
    cat_helper_cols = ["skin_color", "color_contrast"]

    # 1. Encode categorical helpers
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(df_train[cat_helper_cols].fillna("unknown"))

    def prep_mice_input(df):
        helper_encoded = enc.transform(df[cat_helper_cols].fillna("unknown"))
        helper_df = pd.DataFrame(helper_encoded, columns=cat_helper_cols, index=df.index)

        target_encoded = df[target_cols].copy()
        mappings = {}
        for col in target_cols:
            unique_vals = sorted(df_train[col].dropna().unique().tolist())
            val_to_idx = {v: i for i, v in enumerate(unique_vals)}
            idx_to_val = {i: v for i, v in enumerate(unique_vals)}
            target_encoded[col] = df[col].map(val_to_idx)
            mappings[col] = (val_to_idx, idx_to_val)

        return pd.concat([target_encoded, helper_df, df[bf_cols]], axis=1), mappings

    train_input, train_mappings = prep_mice_input(df_train)
    val_input, _ = prep_mice_input(df_val)

    try:
        imputer = IterativeImputer(
            estimator=RandomForestClassifier(n_estimators=100, random_state=random_state),
            max_iter=10,
            random_state=random_state
        )
    except Exception as e:
        raise ValueError(f"Failed to construct IterativeImputer with RandomForestClassifier estimator: {e}") from e
    imputer.fit(train_input)

    train_res = imputer.transform(train_input)
    val_res = imputer.transform(val_input)

    def decode_mice(res_arr, df_orig, mappings):
        res_df = pd.DataFrame(res_arr, columns=train_input.columns, index=df_orig.index)
        for col in target_cols:
            _, idx_to_val = mappings[col]
            decoded = res_df[col].round().clip(0, len(idx_to_val) - 1).astype(int).map(idx_to_val)
            df_orig[col] = decoded.fillna("unknown")
        return df_orig

    return decode_mice(train_res, df_train, train_mappings), decode_mice(val_res, df_val, train_mappings)


# ─────────────────────────────────────────────
# Task 4.1: Build and save encoder artifacts
# ─────────────────────────────────────────────
def build_encoder_artifacts(
    df_train_enc: pd.DataFrame,
    cat_cols: list,
    bf_cols: list,
    phys_feature_cols: list,
    save_dir: Path = None,
) -> None:
    """
    Build encoder_mapping.json and phys_feature_cols.json from the encoded DataFrame
    and save them to save_dir (defaults to SAVED_MODELS_DIR).

    Parameters
    ----------
    df_train_enc : pd.DataFrame
        The combined (train+val) encoded DataFrame — used to derive one-hot column names.
    cat_cols : list[str]
        Categorical columns that were one-hot encoded via pd.get_dummies.
    bf_cols : list[str]
        Body-figure binary columns (names include the 'bf_' prefix).
    phys_feature_cols : list[str]
        Ordered list of all physical feature column names (including _was_imputed columns).
    save_dir : Path, optional
        Directory to save artifacts. Defaults to SAVED_MODELS_DIR from scripts.config.
    """
    if save_dir is None:
        save_dir = SAVED_MODELS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # bf_classes: strip the 'bf_' prefix
    bf_classes = [col[len("bf_"):] for col in bf_cols]

    # one_hot_cols: for each cat col, collect the one-hot column names present in df_train_enc
    one_hot_cols = {}
    for col in cat_cols:
        prefix = f"{col}_"
        one_hot_cols[col] = sorted(
            [c for c in df_train_enc.columns if c.startswith(prefix)]
        )

    encoder_mapping = {
        "cat_cols": list(cat_cols),
        "bf_classes": bf_classes,
        "one_hot_cols": one_hot_cols,
    }

    encoder_mapping_path = save_dir / "encoder_mapping.json"
    with open(encoder_mapping_path, "w") as f:
        json.dump(encoder_mapping, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved encoder_mapping.json to: {encoder_mapping_path}")

    phys_feature_cols_path = save_dir / "phys_feature_cols.json"
    with open(phys_feature_cols_path, "w") as f:
        json.dump(list(phys_feature_cols), f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved phys_feature_cols.json to: {phys_feature_cols_path}")


# ─────────────────────────────────────────────
# Task 4.3: run_pipeline with n_imputations
# ─────────────────────────────────────────────
def run_pipeline(
    train_raw,
    val_raw,
    train_manifest,
    val_manifest,
    train_out,
    val_out,
    n_imputations: int = 1,
):
    print("🚀 Starting O4U Imputation Pipeline (Merging with Manifest)...")

    # Load Manifests (Base)
    print(f"  → Loading Manifests: {train_manifest}")
    df_train_base = pd.read_json(train_manifest)
    df_val_base = pd.read_json(val_manifest)
    df_train_base['id'] = df_train_base['id'].astype(str)
    df_val_base['id'] = df_val_base['id'].astype(str)

    # ── Task 4.2: Produce val_test_outfit_ids.json ────────────────────────────
    print("\n[STEP 0] Collecting val + test outfit IDs")
    val_ids = set(df_val_base['id'].tolist())

    test_raw_path = TEST_MANIFEST  # data/raw/Outfit4You/label/test.json
    test_ids: set = set()
    if test_raw_path.exists():
        print(f"  → Loading test raw: {test_raw_path}")
        with open(test_raw_path) as f:
            test_data = json.load(f)
        test_ids = {str(item['id']) for item in test_data if 'id' in item}
    else:
        # Also check processed test_manifest.json as fallback
        test_manifest_path = DATA_PROCESSED_DIR / "test_manifest.json"
        if test_manifest_path.exists():
            print(f"  → Loading test manifest: {test_manifest_path}")
            df_test_base = pd.read_json(test_manifest_path)
            df_test_base['id'] = df_test_base['id'].astype(str)
            test_ids = set(df_test_base['id'].tolist())

    val_test_ids = sorted(val_ids | test_ids)
    val_test_ids_path = DATA_PROCESSED_DIR / "val_test_outfit_ids.json"
    with open(val_test_ids_path, "w") as f:
        json.dump(val_test_ids, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved val_test_outfit_ids.json ({len(val_test_ids)} IDs) to: {val_test_ids_path}")
    # ─────────────────────────────────────────────────────────────────────────

    # Load Raw Physical Data
    df_train_raw = load_and_normalize_raw(train_raw)
    df_val_raw = load_and_normalize_raw(val_raw)

    # Belt-and-suspenders: ensure all PHYSICAL_COLS string values are lowercased (Requirement 2.4)
    for df_raw in [df_train_raw, df_val_raw]:
        for col in PHYSICAL_COLS:
            if col in df_raw.columns and df_raw[col].dtype == object:
                df_raw[col] = df_raw[col].str.lower()

    # Merge Physical data into base manifest
    # This ensures we only impute samples that were NOT filtered out
    df_train = pd.merge(df_train_base, df_train_raw, on='id', how='left')
    df_val = pd.merge(df_val_base, df_val_raw, on='id', how='left')

    # ── Imputation Indicator Columns (Requirement 12.1, 12.2, 12.3) ──────────
    # Record missingness BEFORE any imputation so indicators reflect original data.
    _indicator_cols = ["height", "breasts", "skin_color", "color_contrast", "hair_style", "hair_color"]
    for col in _indicator_cols:
        indicator = f"{col}_was_imputed"
        for df in [df_train, df_val]:
            df[indicator] = df[col].isna().astype(int)
    print(f"  ✓ Added _was_imputed indicators for: {_indicator_cols}")
    # ─────────────────────────────────────────────────────────────────────────

    # Steps
    df_train, df_val, mlb = process_body_figure(df_train, df_val)

    print("\n[STEP 3-4] Imputing color_contrast and skin_color")
    impute_categorical_group_mode(df_train, df_val, "color_contrast", "medium")
    impute_categorical_group_mode(df_train, df_val, "skin_color", "unknown")

    print("\n[STEP 5] Imputing hair features")
    for col in ["hair_style", "hair_color"]:
        df_train[col] = df_train[col].fillna("unknown")
        df_val[col] = df_val[col].fillna("unknown")

    # ── Task 4.3: Multiple imputation ────────────────────────────────────────
    if n_imputations > 1:
        print(f"\n[STEP 6] Multiple MICE Imputation (n={n_imputations})")
        seeds = list(range(42, 42 + n_imputations))

        # Determine output directory from train_out path
        train_out_path = Path(train_out)
        val_out_path = Path(val_out)
        out_dir = train_out_path.parent

        for i, seed in enumerate(seeds):
            # Work on copies so each run starts from the same pre-MICE state
            df_train_copy = df_train.copy()
            df_val_copy = df_val.copy()

            df_train_imp, df_val_imp = impute_mice(df_train_copy, df_val_copy, random_state=seed)

            # One-hot encode categorical columns on combined train+val (Task 4.1 requirement)
            bf_cols = [c for c in df_train_imp.columns if c.startswith("bf_")]
            combined = pd.concat([df_train_imp, df_val_imp], axis=0)
            combined_enc = pd.get_dummies(combined, columns=CAT_COLS, prefix=CAT_COLS)
            n_train = len(df_train_imp)
            df_train_enc = combined_enc.iloc[:n_train].copy()
            df_val_enc = combined_enc.iloc[n_train:].copy()

            # Save encoder artifacts only on first imputation run
            if i == 0:
                _save_encoder_artifacts(df_train_enc, combined_enc, bf_cols, _indicator_cols)

            # Save manifests
            train_out_i = out_dir / f"train_imputed_manifest_{i}.json"
            val_out_i = out_dir / f"val_imputed_manifest_{i}.json"
            _save_manifest(df_train_enc, train_out_i)
            _save_manifest(df_val_enc, val_out_i)

    else:
        # Single imputation (default behavior)
        df_train, df_val = impute_mice(df_train, df_val, random_state=42)

        # One-hot encode categorical columns on combined train+val (Task 4.1 requirement)
        bf_cols = [c for c in df_train.columns if c.startswith("bf_")]
        combined = pd.concat([df_train, df_val], axis=0)
        combined_enc = pd.get_dummies(combined, columns=CAT_COLS, prefix=CAT_COLS)
        n_train = len(df_train)
        df_train_enc = combined_enc.iloc[:n_train].copy()
        df_val_enc = combined_enc.iloc[n_train:].copy()

        # Task 4.1: Save encoder artifacts
        _save_encoder_artifacts(df_train_enc, combined_enc, bf_cols, _indicator_cols)

        # Cleanup and Save
        _save_manifest(df_train_enc, train_out)
        _save_manifest(df_val_enc, val_out)

    print("\n✅ Pipeline complete!")


def _save_encoder_artifacts(df_train_enc, combined_enc, bf_cols, indicator_cols):
    """Helper: derive phys_feature_cols and call build_encoder_artifacts."""
    # phys_feature_cols: bf_ columns + one-hot columns + _was_imputed columns
    one_hot_cols_flat = []
    for col in CAT_COLS:
        prefix = f"{col}_"
        one_hot_cols_flat.extend(sorted(c for c in combined_enc.columns if c.startswith(prefix)))

    was_imputed_cols = [f"{col}_was_imputed" for col in indicator_cols]
    phys_feature_cols = (
        sorted(bf_cols)
        + one_hot_cols_flat
        + [c for c in was_imputed_cols if c in combined_enc.columns]
    )

    build_encoder_artifacts(
        df_train_enc=combined_enc,
        cat_cols=CAT_COLS,
        bf_cols=sorted(bf_cols),
        phys_feature_cols=phys_feature_cols,
        save_dir=SAVED_MODELS_DIR,
    )


def _save_manifest(df, path):
    """Helper: drop body_figure_clean and save DataFrame as JSON manifest."""
    df = df.copy()
    if "body_figure_clean" in df.columns:
        df.drop(columns=["body_figure_clean"], inplace=True)
    df_json = df.where(pd.notnull(df), None)
    records = df_json.to_dict(orient="records")
    with open(path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved to: {path}")


if __name__ == "__main__":
    # Task 4.3: argparse with --n-imputations
    parser = argparse.ArgumentParser(description="O4U Imputation Pipeline")
    parser.add_argument(
        "--n-imputations",
        type=int,
        default=1,
        help="Number of MICE imputation runs (default: 1). When > 1, saves N manifests.",
    )
    args = parser.parse_args()

    # Task 4.4: Use path constants from scripts.config (no hardcoded strings)
    run_pipeline(
        train_raw=str(TRAIN_MANIFEST),
        val_raw=str(VAL_MANIFEST),
        train_manifest=str(DATA_PROCESSED_DIR / "train_manifest.json"),
        val_manifest=str(DATA_PROCESSED_DIR / "val_manifest.json"),
        train_out=str(TRAIN_IMPUTED_MANIFEST),
        val_out=str(VAL_IMPUTED_MANIFEST),
        n_imputations=args.n_imputations,
    )
