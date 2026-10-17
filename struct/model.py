from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConfidenceMaskedBCELoss(nn.Module):
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        conf_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid_mask = ~torch.isnan(targets)
        if conf_weights is not None:
            valid_mask = valid_mask & (conf_weights > 1e-4)
        if not valid_mask.any():
            return logits.sum() * 0.0

        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        raw_loss = F.binary_cross_entropy_with_logits(
            logits, safe_targets, pos_weight=self.pos_weight, reduction="none"
        )
        if conf_weights is not None:
            w = torch.where(valid_mask, conf_weights.to(raw_loss.dtype), torch.zeros_like(raw_loss))
        else:
            w = valid_mask.to(raw_loss.dtype)

        return (raw_loss * w).sum() / w.sum().clamp_min(1e-6)


class MultimodalKneeEngine(nn.Module):
    """
    2.5D Position- & Plane-Conditioned Class-Specific Attention MIL:
    - Each slice token (from K series x S slices) is encoded by EfficientNet-B0 using
      2.5D tri-slice context [z-1, z, z+1].
    - Each token is concatenated with its 6-dim anatomical coordinate & sequence vector:
      [Fluid_Sensitive, Fat_Suppression, Sagittal, Coronal, Axial, z_pos].
    - Each of the 12 abnormality queries attends directly over all K*S valid tokens in the study
      (combating the medial vs. lateral averaging bottleneck) via Gated Multi-Head Class Attention.
    """

    def __init__(
        self,
        num_classes: int = 12,
        priors: Optional[np.ndarray] = None,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        att_dim: int = 256,
        dropout: float = 0.3,
        use_metadata: bool = True,
        text_dim: int = 6,
        channels_last: bool = False,
        freeze_early_stages: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_metadata = use_metadata
        self.freeze_early_stages = freeze_early_stages

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        vis_dim = self.backbone.num_features

        # Freeze stem and early stage 0/1 to preserve pretrained low-level edge/texture detectors
        if self.freeze_early_stages:
            for name, param in self.backbone.named_parameters():
                if name.startswith(("conv_stem", "bn1", "blocks.0.", "blocks.1.")):
                    param.requires_grad = False

        if self.use_metadata:
            self.meta_proj = nn.Sequential(
                nn.Linear(text_dim, 32),
                nn.LayerNorm(32),
                nn.SiLU(),
            )
            token_in_dim = vis_dim + 32
        else:
            self.meta_proj = None
            token_in_dim = vis_dim

        self.token_proj = nn.Sequential(
            nn.Linear(token_in_dim, att_dim),
            nn.LayerNorm(att_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Gated Class-Specific Attention over all N = K * S tokens in the study
        self.gate_V = nn.Sequential(nn.Linear(att_dim, att_dim), nn.Tanh())
        self.gate_U = nn.Sequential(nn.Linear(att_dim, att_dim), nn.Sigmoid())
        self.class_queries = nn.Parameter(torch.randn(num_classes, att_dim) * 0.05)

        self.class_head_norm = nn.LayerNorm(att_dim)
        self.class_classifiers = nn.Linear(att_dim, num_classes)

        if priors is not None:
            clipped_priors = np.clip(priors, 1e-4, 0.999)
            init_bias = -np.log((1.0 - clipped_priors) / clipped_priors)
            self.class_classifiers.bias.data.copy_(torch.tensor(init_bias, dtype=torch.float32))
            nn.init.xavier_normal_(self.class_classifiers.weight, gain=0.1)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep BatchNorm running stats frozen in eval mode so small study batch sizes never corrupt BN stats
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        series_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if images.ndim == 5:
            images = images.unsqueeze(1)
            if metadata is not None and metadata.ndim == 2:
                metadata = metadata.unsqueeze(1).unsqueeze(2)

        b, k, s, c, h, w = images.shape
        if c == 1:
            images = images.expand(b, k, s, 3, h, w)

        flat_x = images.reshape(b * k * s, 3, h, w).contiguous()
        if not torch.is_autocast_enabled():
            target_dtype = next(self.backbone.parameters()).dtype
            if flat_x.dtype != target_dtype:
                flat_x = flat_x.to(dtype=target_dtype)

        vis_tokens = self.backbone(flat_x).reshape(b, k * s, -1)

        if self.use_metadata and self.meta_proj is not None and metadata is not None:
            if metadata.ndim == 3:
                # Backward compatibility if (B, K, 5) is passed instead of (B, K, S, 6)
                z_grid = torch.linspace(-1.0, 1.0, s, device=metadata.device, dtype=metadata.dtype)
                z_exp = z_grid.view(1, 1, s, 1).expand(b, k, s, 1)
                m_exp = metadata.unsqueeze(2).expand(b, k, s, -1)
                metadata = torch.cat([m_exp, z_exp], dim=-1)
            meta_tokens = self.meta_proj(metadata.reshape(b * k * s, -1)).reshape(b, k * s, -1)
            tokens = torch.cat([vis_tokens, meta_tokens], dim=-1)
        else:
            tokens = vis_tokens

        # Project tokens: (B, N, D) where N = K * S
        h_tokens = self.token_proj(tokens)
        gated_tokens = self.gate_V(h_tokens) * self.gate_U(h_tokens)  # (B, N, D)

        # Class-specific attention scores over all N = K * S tokens: (B, 12, N)
        attn_logits = torch.einsum("bnd,cd->bcn", gated_tokens, self.class_queries) / (h_tokens.shape[-1] ** 0.5)

        if series_mask is not None:
            # Expand (B, K) mask to (B, 1, K * S)
            token_mask = series_mask.unsqueeze(-1).expand(b, k, s).reshape(b, 1, k * s)
            attn_logits = attn_logits.masked_fill(~token_mask, -1e4)

        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, 12, N)

        # Combine attention-pooled representation with top-2 max-activated tokens per class
        attn_pooled = torch.einsum("bcn,bnd->bcd", attn_weights, h_tokens)  # (B, 12, D)
        study_class_feats = self.class_head_norm(attn_pooled)

        logits = (study_class_feats * self.class_classifiers.weight).sum(dim=-1) + self.class_classifiers.bias
        return logits
