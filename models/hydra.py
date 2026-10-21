# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import torch.nn.functional as F

from typing import Optional
import timm
import torch
import os
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn

from models.backbones.utils import vit_sizes

class Hydra(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name="hydra_vit_small_patch16",
        multiplier: int = 1,
        ls_init = 1e-6,
        vit_size: str = "S",
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            no_embed_class=True,
            init_values=ls_init,
            num_classes=0
        )
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            print("checkpoint found at {}".format(ckpt_path))
            checkpoint = torch.load(ckpt_path, weights_only=False)
            
            state_dict = None
            if 'model_state' in checkpoint:
                state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint['model_state'].items()}
                src_len = state_dict['pos_embed'].shape[1] + self.backbone.num_prefix_tokens
                tgt_len = self.backbone.pos_embed.shape[1] + self.backbone.num_prefix_tokens
                state_dict['pos_embed'] = resize_pos_embed(
                    state_dict['pos_embed'],
                    self.backbone.pos_embed
                )
                rescale_attn_out(state_dict, src_len, tgt_len)
            
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False) if state_dict else self.backbone.load_state_dict(checkpoint, strict=False)
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")

        self.embed_dim = self.backbone.embed_dim
        sizes = vit_sizes[vit_size]
        self.num_heads = [sizes['num_heads']]
        self.depths = [sizes['depth']]
        self.patch_size = self.backbone.patch_embed.patch_size
        self.grid_size = self.backbone.patch_embed.grid_size
        self.num_prefix_tokens = self.backbone.num_prefix_tokens
        self.blocks = self.backbone.blocks
        self.norm = self.backbone.norm
        self.attn_multiplier = multiplier

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(
            1, -1, 1, 1
        )
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        return x
    
    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i):
        if i >= len(self.blocks) - eomt_obj.num_blocks:
            # --- EoMT Interaction Logic ---
            
            # 1. Combine q and x for normalization
            xq = torch.cat(((q if q.dim() == 3 else q[None, :, :].expand(x.shape[0], -1, -1)), x), dim=1)
            
            # 2. Pre-Attention Norm
            pre_attn = block.norm1(xq)
            
            # 3. Split normalized tokens
            x_norm, q_norm = pre_attn[:, eomt_obj.num_q:, :], pre_attn[:, : eomt_obj.num_q, :]
            
            # 4. EoMT Predictions
            attn_mask = eomt_obj.predict_and_mask(x_norm, q_norm, i - len(self.blocks))

            # 5. EoMT Cross-Attention
            after_eomt = torch.cat((q_norm, x_norm), dim=1)
            new_x = eomt_obj.attn[i - len(self.blocks)](self.norm(after_eomt), num_queries=eomt_obj.num_q, attn_mask=attn_mask)
            
            # 6. Update x and q with Cross-Attention result
            ls_val = eomt_obj.ls_list[i - len(self.blocks)](new_x)
            xq = xq + eomt_obj.dp(ls_val)
            x, q = xq[:, eomt_obj.num_q :, :], xq[:, : eomt_obj.num_q, :]
            
            # 7. Standard Block Attention (Self-Attention)
            # Hydra/ViT blocks usually have: norm1 -> attn -> ls1 -> drop_path1
            # We apply this to x
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            
            # 8. Standard Block MLP
            # Hydra/ViT blocks usually have: norm2 -> mlp -> ls2 -> drop_path2
            # Apply to x
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
            
            # Apply to q (only MLP part, no self-attention for query tokens in this design)
            # Query self-attention (Mask2Former order: cross-attn, self-attn, MLP).
            q = eomt_obj.apply_query_self_attn(q, i - len(self.blocks))
            q = q + block.drop_path2(block.ls2(block.mlp(block.norm2(q))))
            
        else:
            # Standard execution for earlier blocks
            x = block(x)

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        x = self.norm(x)
        return x, q


def rescale_attn_out(state_dict, src_len, tgt_len):
    """Fold the 1/N that ``HydraAttention`` is missing into the loaded weights.

    ``models/backbones/hydra.py`` computes ``(k * v).sum(dim=-2) * q`` with no
    normalisation by the token count, so the attention branch scales linearly with
    the sequence length. Weights pretrained at 224 (N=197) run at 512 (N=1025)
    therefore drive the residual stream ~5x too hard, and it compounds with depth:
    measured on the real IN21k checkpoint, block 0's ``|sum(k*v)|`` grows 4.69x
    (token ratio 5.20x) and block 3 grows 100x. That is the ADE20K semantic
    0.349@224 -> 0.075@512 collapse.

    ``attn.out`` is the linear layer right after the sum, so pre-scaling its weight
    by ``src_len / tgt_len`` restores the pretraining scale. The bias is a
    per-channel constant that does not scale with N, so it is left alone.
    """
    if src_len == tgt_len:
        return
    scale = src_len / tgt_len
    for key in state_dict:
        if key.endswith("attn.out.weight"):
            state_dict[key] = state_dict[key] * scale
    print(f"Rescaling Hydra attn.out weights by {scale:.4f} (seq len {src_len} -> {tgt_len})")


def resize_pos_embed(posemb, posemb_new):
    # posemb: [1, L_old, C], posemb_new: [1, L_new, C]
    L_old = posemb.shape[1]
    L_new = posemb_new.shape[1]

    if L_old == L_new:
        return posemb

    # infer how many extra tokens (e.g. CLS) there are
    num_tokens = infer_num_tokens(posemb, posemb_new)

    extra_tokens = posemb[:, :num_tokens]
    posemb_grid = posemb[:, num_tokens:]          # [1, old_grid, C]

    gs_old = int((posemb_grid.shape[1]) ** 0.5)   # e.g. 14 for 196 tokens
    gs_new = int((L_new - num_tokens) ** 0.5)     # e.g. 32 for 1024 tokens

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new),
                                mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    
    return torch.cat([extra_tokens, posemb_grid], dim=1)

def infer_num_tokens(posemb_ckpt, posemb_model):
    # posemb_ckpt: [1, L_ckpt, C]; posemb_model: [1, L_model, C]
    Lc = posemb_ckpt.shape[1]
    Lm = posemb_model.shape[1]

    # Try common cases: 0 or 1 extra tokens
    for n in (1, 0):
        if (Lc - n) > 0 and int((Lc - n) ** 0.5) ** 2 == (Lc - n):
            return n
    # Fallback: assume same #extra as model
    for n in (1, 0):
        if (Lm - n) > 0 and int((Lm - n) ** 0.5) ** 2 == (Lm - n):
            return n
    raise ValueError(f"Cannot infer num_tokens from shapes {posemb_ckpt.shape} and {posemb_model.shape}")
