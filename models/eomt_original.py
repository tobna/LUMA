# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

"""
eomt_original.py — Original EoMT formulation, written in master style.

Key differences vs eomt.py (injected-attention variant):
  - No CrossAttentionSingleInput / Attention / LayerScale injection.
  - Queries are concatenated into the token sequence at the (L - num_blocks)-th
    block and attend to image tokens via the backbone's own self-attention,
    optionally gated by a boolean mask derived from predicted mask logits.
  - Works with any ViT-style backbone that exposes the standard timm API via
    encoder.backbone (i.e. the master ViT wrapper).
  - Follows master conventions: nn.Embedding queries, same _predict / ScaleBlock
    upscaler, same forward output tuple.
"""

from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.scale_block import ScaleBlock


class EoMT_Original(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        num_q: int,
        num_blocks: int = 4,
        masked_attn_enabled: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        embed_dim = encoder.backbone.embed_dim

        self.q = nn.Embedding(num_q, embed_dim)

        self.class_head = nn.Linear(embed_dim, num_classes + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        patch_size = encoder.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
        self.upscale = nn.Sequential(
            *[ScaleBlock(embed_dim) for _ in range(num_upscale)]
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _predict(self, x: torch.Tensor):
        """Compute mask & class logits. x = [queries | prefix | image tokens]."""
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        img = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        img = img.transpose(1, 2).reshape(
            img.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(img)
        )

        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask: torch.Tensor, prob: float):
        """Randomly drop mask constraints during training for regularisation."""
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens :
            ][random_queries] = True
        return attn_mask

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the timm attention block with an optional boolean attention mask."""
        B, N, C = x.shape
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        bb = self.encoder.backbone

        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        x = bb.patch_embed(x)
        x = bb._pos_embed(x)
        x = bb.patch_drop(x)
        x = bb.norm_pre(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(bb.blocks):
            if i == len(bb.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= len(bb.blocks) - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(bb.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = torch.ones(
                    x.shape[0], x.shape[1], x.shape[1],
                    dtype=torch.bool, device=x.device,
                )
                interpolated = F.interpolate(
                    mask_logits,
                    bb.patch_embed.grid_size,
                    mode="bilinear",
                )
                interpolated = interpolated.view(
                    interpolated.size(0), interpolated.size(1), -1
                )
                attn_mask[
                    :, : self.num_q, self.num_q + bb.num_prefix_tokens :
                ] = (interpolated > 0)
                attn_mask = self._disable_attn_mask(
                    attn_mask,
                    self.attn_mask_probs[i - len(bb.blocks) + self.num_blocks],
                )

            x = x + block.drop_path1(
                block.ls1(self._attn(block.attn, block.norm1(x), attn_mask))
            )
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

        mask_logits, class_logits = self._predict(bb.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return mask_logits_per_layer, class_logits_per_layer
