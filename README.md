# Outfit4You (O4U) — Fashion Outfit Compatibility Prediction

A hybrid deep learning pipeline that predicts whether a given outfit is compatible with a user's physical attributes. Combines **CLIP visual features** (Transformer + attention pooling) with **imputed physical attributes** (body figure, height, skin color, etc.) via a **FiLM fusion** module.

---

## Project Structure

```
O4U/
├── notebooks/
│   ├── 01_Data_Prep_and_CLIP_Extraction.ipynb  # Extract CLIP features (run once)
│   ├── 02_EDA.ipynb                             # Exploratory data analysis
│   ├── 03_Training.ipynb                        # Imputation + model training
│   ├── 04_Evaluation.ipynb                      # Test evaluation + inference demo
│   ├── 05_Bagging_Ensemble.ipynb                # RandomForest + BaggingClassifier
│   ├── 06_Stacking_Ensemble.ipynb               # Stacking ensemble (HGB + RF + MLP)
│   └── 07_Custom_Stacking_Ensemble.ipynb        # Custom stacking (XGBoost)
├── scripts/
│   ├── config.py                    # Centralized path configuration
│   ├── models.py                    # HybridFashionModel, VisualBranch, FiLMFusion
│   ├── data_utils.py                # O4UHybridDataset, collate_fn
│   ├── o4u_imputation_pipeline.py   # MICE imputation pipeline
│   ├── train.py                     # Training script (CLI)
│   ├── evaluate_test_set.py         # Test set evaluator
│   ├── baselines.py                 # Pure-visual, pure-physical, CLIP zero-shot
│   ├── inference.py                 # Live inference demo
│   └── threshold_sensitivity.py    # Binary label threshold analysis
├── tests/                           # pytest test suite
├── data/
│   ├── raw/Outfit4You/
│   │   ├── image/        # ⚠️ NOT in git — 139k raw .jpg images
│   │   └── label/        # train.json, val.json, test.json  ✅ in git
│   └── processed/
│       ├── outfits/      # ⚠️ NOT in git — 15k CLIP .pt feature files
│       └── *.json        # Imputed manifests  ✅ in git
├── saved_models/
│   ├── *.pth             # ⚠️ NOT in git — model weights (large)
│   └── *.json            # Artifacts (thresholds, config, etc.)  ✅ in git
└── plots/                # Generated charts  ✅ in git
```

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd O4U
```

### 2. Create a conda environment

```bash
conda create -n fashion_env python=3.10
conda activate fashion_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) instead of the version in `requirements.txt`.

### 4. Get the raw data

Download the **Outfit4You** dataset and place it at:

```
data/raw/Outfit4You/
├── image/        # all .jpg images
└── label/
    ├── train.json
    ├── val.json
    └── test.json
```

---

## Running the Pipeline

Run the notebooks **in order**:

| Step | Notebook | What it does | Run time |
|------|----------|--------------|----------|
| 1 | `01_Data_Prep_and_CLIP_Extraction.ipynb` | Extract CLIP features → `.pt` files | ~30 min (one-time) |
| 2 | `02_EDA.ipynb` | Score distribution, feature sparsity | Fast |
| 3 | `03_Training.ipynb` | Imputation + model training | Slow |
| 4 | `04_Evaluation.ipynb` | Test evaluation + inference demo | Medium |
| 5–7 | `05–07_*.ipynb` | Ensemble experiments (optional) | Slow |

### Or run training from the CLI

```bash
python3 -m scripts.train \
  --seed 42 \
  --save-metric roc_auc \
  --feature-set full \
  --epochs 50 \
  --warmup-epochs 3 \
  --lr 1e-4 \
  --batch-size 32 \
  --patience 10
```

---

## Model Architecture

```
Visual Branch (Transformer + Attention Pooling)
    ↓
FiLM Fusion ← Physical Branch (one-hot + MICE imputation)
    ↓
Shared MLP (LayerNorm)
    ↓
Regression Head    Classification Head
(compatibility     (binary: compatible
 score)             or not)
```

---

## Key Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9443 |
| F1 | 0.9191 |
| Accuracy | 90.24% |
| Brier Score | 0.0834 |

---

## Running Tests

```bash
python3 -m pytest tests/ -v
```

---

## Requirements

- Python 3.10+
- See `requirements.txt` for full package list
- GPU recommended for training (MPS on Apple Silicon, CUDA on NVIDIA)
