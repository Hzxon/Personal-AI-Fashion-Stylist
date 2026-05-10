"""
Unit tests for collate_fn from scripts/data_utils.py.

Requirements: 32.4
"""

import torch
import pytest
from scripts.data_utils import collate_fn


def make_batch(seq_lens, phys_dim=20, embed_dim=512):
    """
    Build a list of (visual_feat, phys_vec, reg_label, bin_label) tuples.

    Args:
        seq_lens: list of ints — number of items (sequence length) per sample
        phys_dim: dimension of the physical feature vector
        embed_dim: CLIP embedding dimension per item
    """
    batch = []
    for seq_len in seq_lens:
        visual_feat = torch.randn(seq_len, embed_dim)
        phys_vec = torch.randn(phys_dim)
        reg_label = torch.tensor(1.0)
        bin_label = torch.tensor(0.0)
        batch.append((visual_feat, phys_vec, reg_label, bin_label))
    return batch


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_batch_size_1_shapes():
    """Batch of 1 item with sequence length 3, phys_dim 20: verify output shapes."""
    seq_lens = [3]
    phys_dim = 20
    embed_dim = 512
    batch = make_batch(seq_lens, phys_dim=phys_dim, embed_dim=embed_dim)

    visual_padded, visual_mask, phys_vecs, reg_labels, bin_labels = collate_fn(batch)

    B = 1
    max_len = 3

    assert visual_padded.shape == (B, max_len, embed_dim), (
        f"Expected visual_padded shape ({B}, {max_len}, {embed_dim}), got {visual_padded.shape}"
    )
    assert visual_mask.shape == (B, max_len), (
        f"Expected visual_mask shape ({B}, {max_len}), got {visual_mask.shape}"
    )
    assert phys_vecs.shape == (B, phys_dim), (
        f"Expected phys_vecs shape ({B}, {phys_dim}), got {phys_vecs.shape}"
    )
    assert reg_labels.shape == (B,), f"Expected reg_labels shape ({B},), got {reg_labels.shape}"
    assert bin_labels.shape == (B,), f"Expected bin_labels shape ({B},), got {bin_labels.shape}"


def test_batch_size_2_shapes():
    """Batch of 2 items with different sequence lengths (3 and 5): verify output shapes."""
    seq_lens = [3, 5]
    phys_dim = 20
    embed_dim = 512
    batch = make_batch(seq_lens, phys_dim=phys_dim, embed_dim=embed_dim)

    visual_padded, visual_mask, phys_vecs, reg_labels, bin_labels = collate_fn(batch)

    B = 2
    max_len = 5  # max of [3, 5]

    assert visual_padded.shape == (B, max_len, embed_dim), (
        f"Expected visual_padded shape ({B}, {max_len}, {embed_dim}), got {visual_padded.shape}"
    )
    assert visual_mask.shape == (B, max_len), (
        f"Expected visual_mask shape ({B}, {max_len}), got {visual_mask.shape}"
    )
    assert phys_vecs.shape == (B, phys_dim), (
        f"Expected phys_vecs shape ({B}, {phys_dim}), got {phys_vecs.shape}"
    )
    assert reg_labels.shape == (B,), f"Expected reg_labels shape ({B},), got {reg_labels.shape}"
    assert bin_labels.shape == (B,), f"Expected bin_labels shape ({B},), got {bin_labels.shape}"


def test_batch_size_8_shapes():
    """Batch of 8 items with varying sequence lengths: verify output shapes."""
    seq_lens = [1, 2, 3, 4, 5, 6, 7, 8]
    phys_dim = 20
    embed_dim = 512
    batch = make_batch(seq_lens, phys_dim=phys_dim, embed_dim=embed_dim)

    visual_padded, visual_mask, phys_vecs, reg_labels, bin_labels = collate_fn(batch)

    B = 8
    max_len = 8  # max of seq_lens

    assert visual_padded.shape == (B, max_len, embed_dim), (
        f"Expected visual_padded shape ({B}, {max_len}, {embed_dim}), got {visual_padded.shape}"
    )
    assert visual_mask.shape == (B, max_len), (
        f"Expected visual_mask shape ({B}, {max_len}), got {visual_mask.shape}"
    )
    assert phys_vecs.shape == (B, phys_dim), (
        f"Expected phys_vecs shape ({B}, {phys_dim}), got {phys_vecs.shape}"
    )
    assert reg_labels.shape == (B,), f"Expected reg_labels shape ({B},), got {reg_labels.shape}"
    assert bin_labels.shape == (B,), f"Expected bin_labels shape ({B},), got {bin_labels.shape}"


# ---------------------------------------------------------------------------
# Mask correctness test
# ---------------------------------------------------------------------------

def test_mask_correctness():
    """
    For a batch with sequence lengths [2, 4, 3], verify that
    mask[i, j] == True iff j < seq_len[i].
    """
    seq_lens = [2, 4, 3]
    batch = make_batch(seq_lens)

    _, visual_mask, _, _, _ = collate_fn(batch)

    max_len = max(seq_lens)
    for i, seq_len in enumerate(seq_lens):
        for j in range(max_len):
            expected = j < seq_len
            actual = visual_mask[i, j].item()
            assert actual == expected, (
                f"mask[{i}, {j}] should be {expected} (seq_len={seq_len}), got {actual}"
            )


# ---------------------------------------------------------------------------
# Padding positions are zero test
# ---------------------------------------------------------------------------

def test_padding_positions_are_zero():
    """
    For a batch with sequence lengths [2, 5], verify that positions beyond
    seq_len are zero in visual_padded.
    """
    seq_lens = [2, 5]
    embed_dim = 512
    batch = make_batch(seq_lens, embed_dim=embed_dim)

    visual_padded, _, _, _, _ = collate_fn(batch)

    max_len = max(seq_lens)  # 5
    for i, seq_len in enumerate(seq_lens):
        # Positions [seq_len, max_len) should be all zeros
        if seq_len < max_len:
            padding_region = visual_padded[i, seq_len:, :]
            assert torch.all(padding_region == 0.0), (
                f"Sample {i} (seq_len={seq_len}): padding positions [{seq_len}:{max_len}] "
                f"should be zero but found non-zero values"
            )
        # Positions [0, seq_len) should be the original (non-zero) data —
        # we just verify the shape is correct here since values are random
        content_region = visual_padded[i, :seq_len, :]
        assert content_region.shape == (seq_len, embed_dim), (
            f"Sample {i}: content region shape mismatch"
        )
