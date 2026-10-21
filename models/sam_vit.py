# ---------------------------------------------------------------
# EoMT-compatible wrapper for timm's VisionTransformerSAM.
# Mirrors the interface of models.vit.ViT so EoMT can interleave
# queries into the last `num_blocks` blocks uniformly with the rest
# of the backbone catalog.
# ---------------------------------------------------------------
"""
Why this exists
---------------
SAM's ViT in timm (`VisionTransformerSAM` from
`timm.models.vision_transformer_sam`) differs structurally from the standard
`VisionTransformer`:

* **NHWC layout.** ``patch_embed`` returns ``(B, H, W, C)`` and every block
  expects / returns the same. ``models.vit.ViT`` assumes NLC ``(B, L, C)``
  everywhere.
* **No ``_pos_embed`` method.** Positional embeddings are added inline inside
  ``forward_features`` via ``resample_abs_pos_embed_nhwc``, so the plain ViT
  pre-block sequence breaks.
* **Window + global attention.** Most blocks run window attention with a
  fixed 14x14 window; a handful of "global-attention indexes" run full-grid
  attention. **We preserve this structure exactly** -- the pretrained
  features depend on it. The EoMT cross-attention is a *separate* module that
  runs on top, not in place of, the block's attention.
* **No LayerScale, no CLS / reg tokens.** SAM checkpoints contain no
  ``ls1.gamma`` / ``ls2.gamma`` and no prefix tokens. Passing
  ``init_values=1e-6`` (the default in ``models.vit.ViT``) would create
  parameters that don't exist in the checkpoint.
* **``pos_embed`` and per-head ``rel_pos_{h,w}`` are sized for the pretrain
  image size (1024 for SA-1B).** They get dynamically resampled at forward
  time, but only if we build the model at the pretrain size; resampling at
  construction time isn't part of timm's load path.

We therefore:

1. Build with ``img_size=1024`` (pretrain shape) so the state-dict loads
   cleanly. Smaller / larger inputs at forward time are handled by timm's
   ``resample_abs_pos_embed_nhwc`` + the internal ``get_rel_pos``
   interpolation.
2. Pass ``init_values=None`` and ignore the user's ``ls_init``.
3. **Every block is called natively** via ``block(x_nhwc)``. Whether the
   block is a window-attention or global-attention block, SAM's own
   ``Block.forward`` runs as it was trained.
4. For the last ``num_blocks`` blocks (the EoMT-injection region), we
   sandwich a separate EoMT cross-attention update *before* the block's
   native forward. The cross-attention sees the queries + flattened feature
   tokens; the block then runs untouched on the feature tokens only. This
   matches how ``models.vit.ViT::run_block`` separates ``eomt_obj.attn[idx]``
   from ``block.attn``.
5. Add a fresh ``LayerNorm`` for ``self.norm`` (EoMT uses it before its
   cross-attention and at the end of ``post_blocks``). SAM has no token-space
   final norm we can reuse -- ``neck`` is a Conv2d + LayerNorm2d pair in
   spatial form, not usable here. The new LN is small (~2 * embed_dim params)
   and trained from scratch.
6. Drop ``self.backbone.neck`` and ``self.backbone.head`` to keep DDP happy
   about unused parameters.
"""
from typing import Optional
import os

import timm
import torch
import torch.nn as nn

from timm.models.vision_transformer_sam import resample_abs_pos_embed_nhwc

from models.backbones.utils import vit_sizes


# SAM ViT was pretrained at this image size; we keep params at this shape so
# the state-dict load is clean. Forward-time resampling handles other sizes.
_SAM_PRETRAIN_IMG_SIZE = 1024


class SAMViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size: int = 16,
        backbone_name: str = "samvit_large_patch16.sa1b",
        multiplier: int = 1,
        ls_init: Optional[float] = None,   # ignored; SAM has no LayerScale
        vit_size: str = "L",
        pretrained: bool = False,
        emb_class: bool = False,            # ignored; SAM has no CLS/REG tokens
        ckpt_path: Optional[str] = None,
        global_pool: Optional[str] = None,  # ignored
    ):
        super().__init__()

        # Build the backbone at the pretrain img_size so pos_embed / rel_pos
        # parameter shapes match the checkpoint. Forward-time resampling
        # handles smaller / larger runtime inputs.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            img_size=_SAM_PRETRAIN_IMG_SIZE,
            patch_size=patch_size,
            num_classes=0,
            init_values=None,
        )

        # Drop the SAM neck (Conv2d + LayerNorm2d pair, spatial) and head --
        # we replace them with our own EoMT decoder. Leaving them in would
        # waste memory and trigger DDP unused-param warnings.
        if hasattr(self.backbone, "neck"):
            self.backbone.neck = nn.Identity()
        if hasattr(self.backbone, "head"):
            self.backbone.head = nn.Identity()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        ps = patch_size if isinstance(patch_size, int) else patch_size[0]
        self.runtime_img_size = tuple(img_size)
        self.runtime_grid_size = (img_size[0] // ps, img_size[1] // ps)

        if ckpt_path is not None and os.path.isfile(ckpt_path) and not pretrained:
            print(f"checkpoint found at {ckpt_path}")
            checkpoint = torch.load(ckpt_path, weights_only=False)
            missing, unexpected = self.backbone.load_state_dict(checkpoint, strict=False)
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")

        # ----- EoMT-compatible interface -----
        self.embed_dim = self.backbone.embed_dim
        sizes = vit_sizes[vit_size]
        self.num_heads = [sizes["num_heads"]]
        self.depths = [sizes["depth"]]
        self.patch_size = self.backbone.patch_embed.patch_size
        # Runtime grid size -- LUMA._predict uses this to reshape NLC -> BCHW.
        self.grid_size = self.runtime_grid_size
        self.num_prefix_tokens = 0  # SAM has no CLS / REG tokens
        self.blocks = self.backbone.blocks
        # Fresh final LN -- SAM has no token-space final norm we can reuse.
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.attn_multiplier = multiplier

        pixel_mean = torch.tensor(self.backbone.default_cfg["mean"]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(self.backbone.default_cfg["std"]).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        # Cached spatial dims of the current forward (set in pre_block, read
        # in run_block when reshaping between NLC and NHWC).
        self._cached_HW = None

    # ------------------------------------------------------------------
    # EoMT calls these:
    # ------------------------------------------------------------------
    def pre_block(self, x: torch.Tensor) -> torch.Tensor:
        """``patch_embed`` + dynamic ``pos_embed`` resample. Returns NHWC."""
        x = self.backbone.patch_embed(x)  # (B, H, W, C)
        if self.backbone.pos_embed is not None:
            x = x + resample_abs_pos_embed_nhwc(self.backbone.pos_embed, x.shape[1:3])
        x = self.backbone.pos_drop(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        self._cached_HW = (int(x.shape[1]), int(x.shape[2]))
        return x  # NHWC

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i):
        """Per-block forward; mirrors the Swin pattern in ``models.swin.Swin``.

        Early blocks (``i < depth - num_blocks``):
            Native SAM ``block(x_nhwc)``. Window-attention or global-attention
            blocks run exactly as pretrained.

        EoMT-injection blocks (``i >= depth - num_blocks``):
            1. Concat queries + (flattened) features so they share
               ``block.norm1`` (gives EoMT a normalized view of both for the
               prediction heads).
            2. Run EoMT cross-attention ``eomt_obj.attn[idx]`` on the
               concatenated tensor; residual back into ``xq``.
            3. Split ``xq`` back into ``(q, x)``.
            4. Run the **native SAM block** ``block(x_nhwc)`` on features
               only. Window / global attention is untouched.
            5. Update queries via the shared block MLP (or per-block MLP
               depending on ``eomt_obj.query_mlp``).

        The block's own ``Attention`` never sees query tokens -- only the
        separate EoMT cross-attention module does.
        """
        depth = len(self.blocks)
        in_injection = (i >= depth - eomt_obj.num_blocks)

        # All blocks consume NHWC (pre_block returns NHWC; every branch below
        # also returns NHWC). x.dim() == 4 here.
        B, H, W, C = x.shape

        if not in_injection:
            return block(x), q

        # ---- EoMT injection: cross-attend queries to features, then native block ----

        x_nlc = x.reshape(B, H * W, C)
        q_batched = (q if q.dim() == 3 else q[None, :, :].expand(B, -1, -1))

        # 1. Shared norm1 view for EoMT prediction heads.
        xq = torch.cat((q_batched, x_nlc), dim=1)  # (B, num_q + L, C)
        pre_attn = block.norm1(xq)
        q_tok = pre_attn[:, :eomt_obj.num_q, :]    # (B, num_q, C)
        x_tok = pre_attn[:, eomt_obj.num_q:, :]    # (B, L, C)

        attn_mask = eomt_obj.predict_and_mask(x_tok, q_tok, i - depth)

        # 2. EoMT cross-attention; residual on pre-norm xq (matches Swin).
        after_eomt = torch.cat((q_tok, x_tok), dim=1)
        idx = i - depth  # negative index into eomt_obj.attn / ls_list
        new_x = eomt_obj.attn[idx](self.norm(after_eomt), num_queries=eomt_obj.num_q, attn_mask=attn_mask)
        xq = xq + eomt_obj.dp(eomt_obj.ls_list[idx](new_x))

        # 3. Split back into queries and features.
        q_batched = xq[:, :eomt_obj.num_q, :]      # (B, num_q, C)
        x_nlc     = xq[:, eomt_obj.num_q:, :]      # (B, L, C)

        # 4. Native SAM block on features only (NHWC). Window / global
        #    attention runs untouched -- the block has never seen the queries.
        x_nhwc = x_nlc.reshape(B, H, W, C)
        x_nhwc = block(x_nhwc)

        # 5. Query MLP update -- mirror the feature MLP residual inside
        #    SAM Block.forward exactly: shortcut + drop_path2(ls2(mlp(norm2(.)))).
        #    For SAM with init_values=None, ``block.ls2`` is ``nn.Identity()``,
        #    so the call is a no-op; we keep it explicit so the query path stays
        #    structurally identical to the feature path (and stays correct if
        #    a future SAM variant ships with LayerScale enabled).
        # Query self-attention first (Mask2Former order: cross-attn, self-attn, MLP).
        q_batched = eomt_obj.apply_query_self_attn(q_batched, idx)

        mode = getattr(eomt_obj, "query_mlp", "shared")
        if mode == "shared":
            q_batched = q_batched + block.drop_path2(
                block.ls2(block.mlp(block.norm2(q_batched)))
            )
        elif mode == "separate":
            eomt_idx = i - (depth - eomt_obj.num_blocks)
            q_batched = q_batched + block.drop_path2(
                block.ls2(eomt_obj.q_mlps[eomt_idx](q_batched))
            )
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown query_mlp mode: {mode!r}")

        return x_nhwc, q_batched

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        # All blocks return NHWC; flatten to NLC for LUMA._predict, which
        # reshapes NLC -> BCHW via encoder.grid_size.
        if x.dim() == 4:
            B, H, W, C = x.shape
            x = x.reshape(B, H * W, C)
        x = self.norm(x)
        return x, q
