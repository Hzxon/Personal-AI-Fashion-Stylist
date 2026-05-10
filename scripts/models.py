import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualBranch(nn.Module):
    def __init__(self, embed_dim=512, nhead=8, num_layers=2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            mask_check=False  # Prevents MPS-specific nested tensor errors
        )
        # Task 6.2: attention pooling query vector
        self.attn_query = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x, mask):
        x = self.proj(x)
        padding_mask = ~mask
        out = self.transformer(x, src_key_padding_mask=padding_mask)

        # Task 6.2: attention pooling (replaces mean pooling)
        scores = (out * self.attn_query).sum(-1)          # (B, T)
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        return (out * weights).sum(dim=1)                 # (B, D)


class GatedFusion(nn.Module):
    """DEPRECATED: Use FiLMFusion instead. Kept for backward compatibility."""

    def __init__(self, visual_dim=512, phys_dim=64, hidden_dim=256):
        super().__init__()
        self.fc_visual = nn.Linear(visual_dim, hidden_dim)
        self.fc_phys = nn.Linear(phys_dim, hidden_dim)
        self.norm_visual = nn.LayerNorm(hidden_dim)
        self.norm_phys = nn.LayerNorm(hidden_dim)
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, z_visual, z_phys):
        h_vis = self.norm_visual(self.fc_visual(z_visual))
        h_phys = self.norm_phys(self.fc_phys(z_phys))
        gate = self.gate_layer(torch.cat([h_vis, h_phys], dim=1))
        return gate * h_vis + (1 - gate) * h_phys


class FiLMFusion(nn.Module):
    """FiLM-style fusion of visual and physical feature vectors.

    Concatenates z_visual and z_phys and passes them through a shared MLP
    with LayerNorm stabilisation.  Output shape: (B, hidden_dim).
    """

    def __init__(self, visual_dim=512, phys_dim=64, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim + phys_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, z_visual, z_phys):
        return self.mlp(torch.cat([z_visual, z_phys], dim=1))


class HybridFashionModel(nn.Module):
    """Canonical architecture: Linear projection for physical features (imputed one-hot)."""

    def __init__(self, phys_input_dim, visual_dim=512, phys_dim=64, hidden_dim=256):
        super().__init__()
        self.visual_branch = VisualBranch(embed_dim=visual_dim)
        self.phys_input_proj = nn.Linear(phys_input_dim, phys_dim)
        # Task 6.5: dropout after phys_input_proj
        self.phys_dropout = nn.Dropout(0.3)
        self.phys_proj = nn.Sequential(
            nn.Linear(phys_dim, phys_dim),
            nn.LayerNorm(phys_dim),
            nn.ReLU(),
        )
        # Task 6.4: use FiLMFusion instead of GatedFusion
        self.fusion = FiLMFusion(visual_dim, phys_dim, hidden_dim)
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),   # Task 6.1: LayerNorm replaces BatchNorm1d
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.regression_head = nn.Linear(128, 1)
        self.classification_head = nn.Linear(128, 1)

    def forward(self, visual_feat, visual_mask, phys_vec):
        z_visual = self.visual_branch(visual_feat, visual_mask)
        # Task 6.5: apply phys_dropout before phys_proj
        z_phys = self.phys_proj(self.phys_dropout(self.phys_input_proj(phys_vec)))
        fused = self.fusion(z_visual, z_phys)
        shared = self.shared_mlp(fused)
        reg_out = self.regression_head(shared)
        class_out = self.classification_head(shared)
        return reg_out, class_out

    def consistency_loss(self, reg_out: torch.Tensor, class_out: torch.Tensor) -> torch.Tensor:
        """Penalize sign disagreement between regression and classification heads.

        Penalizes cases where sigmoid(class_out) >= 0.5 but reg_out < 1.0, or vice versa.
        Uses soft penalty via F.relu.
        """
        sig = torch.sigmoid(class_out)
        # Penalty when classifier says compatible but regressor says not
        penalty_a = F.relu(sig - 0.5) * F.relu(1.0 - reg_out)
        # Penalty when classifier says not compatible but regressor says compatible
        penalty_b = F.relu(0.5 - sig) * F.relu(reg_out - 1.0)
        return (penalty_a + penalty_b).mean()


class FeatureExtractor(nn.Module):
    """Extracts fused latent features (post-FiLMFusion) for ensemble input.

    Runs the visual branch and physical projection of the base HybridFashionModel,
    fuses them via FiLMFusion, and returns the fused representation.

    Returns:
        Tensor of shape (B, hidden_dim) — the fused feature vector.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, visual_feat, visual_mask, phys_vec):
        z_visual = self.base.visual_branch(visual_feat, visual_mask)
        # Task 6.5 dropout is baked into base.phys_dropout
        z_phys = self.base.phys_proj(
            self.base.phys_dropout(self.base.phys_input_proj(phys_vec))
        )
        # Task 6.7: FiLMFusion.forward(z_visual, z_phys) — same signature as GatedFusion
        fused = self.base.fusion(z_visual, z_phys)
        return fused  # (B, hidden_dim)
