import json
import logging
import os
from typing import Optional
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# Neutral fallback used when harmony scores are missing for an outfit
_HARMONY_FALLBACK = [0.33, 0.33, 0.34, 1]  # warm, cool, neutral, was_imputed=1


def load_harmony_scores(path: str) -> dict:
    """Load color_harmony_scores.json. Returns empty dict if file not found."""
    if not os.path.exists(path):
        logger.warning(f"color_harmony_scores.json not found at {path!r} — harmony features will use neutral fallback")
        return {}
    with open(path) as f:
        return json.load(f)


class O4UHybridDataset(Dataset):
    def __init__(self, df, features_dir, feature_cols, cache_in_memory=True,
                 on_missing: str = "warn", harmony_scores: Optional[dict] = None):
        self.df = df.reset_index(drop=True)
        self.features_dir = features_dir
        self.feature_cols = feature_cols
        self.cache_in_memory = cache_in_memory
        self.on_missing = on_missing
        # harmony_scores: dict mapping outfit_id str → {warm, cool, neutral, was_imputed}
        self.harmony_scores = harmony_scores or {}
        self.cache = {}
        
        # Only cache if we are in the main process (to avoid worker overhead)
        # Note: In num_workers > 0, this logic can be tricky, 
        # so we recommend num_workers=0 for cached datasets.
        if self.cache_in_memory:
            print(f" Checking/Caching {len(self.df)} visual features...")
            for idx in range(len(self.df)):
                outfit_id = str(self.df.iloc[idx]['id'])
                pt_path = os.path.join(self.features_dir, f"{outfit_id}.pt")
                if os.path.exists(pt_path):
                    self.cache[outfit_id] = torch.load(pt_path, map_location='cpu')
                else:
                    if self.on_missing == "raise":
                        raise FileNotFoundError(f"Missing .pt file for outfit {outfit_id}: {pt_path}")
                    else:
                        logger.warning(f"Missing .pt for outfit {outfit_id}, substituting zero tensor")
                        self.cache[outfit_id] = torch.zeros((1, 512))
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        outfit_id = str(row['id'])
        
        if self.cache_in_memory and outfit_id in self.cache:
            visual_feat = self.cache[outfit_id]
        else:
            pt_path = os.path.join(self.features_dir, f"{outfit_id}.pt")
            if not os.path.exists(pt_path):
                if self.on_missing == "raise":
                    raise FileNotFoundError(f"Missing .pt file for outfit {outfit_id}: {pt_path}")
                else:
                    logger.warning(f"Missing .pt for outfit {outfit_id}, substituting zero tensor")
                    visual_feat = torch.zeros((1, 512))
            else:
                visual_feat = torch.load(pt_path, map_location='cpu')
        
        phys_vec = torch.tensor(row[self.feature_cols].values.astype(float), dtype=torch.float32)

        # Append color harmony vector [warm, cool, neutral, was_imputed]
        if self.harmony_scores:
            h = self.harmony_scores.get(outfit_id, None)
            if h is not None:
                harmony_vec = torch.tensor(
                    [h["warm"], h["cool"], h["neutral"], float(h["was_imputed"])],
                    dtype=torch.float32,
                )
            else:
                harmony_vec = torch.tensor(_HARMONY_FALLBACK, dtype=torch.float32)
            phys_vec = torch.cat([phys_vec, harmony_vec], dim=0)

        reg_label = torch.tensor(row['score'], dtype=torch.float32)
        bin_label = torch.tensor(row['binary_label'], dtype=torch.float32)

        return visual_feat, phys_vec, reg_label, bin_label

def collate_fn(batch):
    visual_list = [item[0] for item in batch]
    visual_padded = pad_sequence(visual_list, batch_first=True, padding_value=0.0)
    
    lengths = torch.tensor([feat.size(0) for feat in visual_list], dtype=torch.long)
    max_len = visual_padded.size(1)
    visual_mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    
    phys_vecs = torch.stack([item[1] for item in batch])
    reg_labels = torch.stack([item[2] for item in batch])
    bin_labels = torch.stack([item[3] for item in batch])
    
    return visual_padded, visual_mask, phys_vecs, reg_labels, bin_labels
