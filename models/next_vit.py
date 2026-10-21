# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import Optional
import timm
import torch
import torch.nn as nn
from models.backbones.next_vit import to_doubletuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_nextvit_final_grid(input_resolution):
    final_size = (input_resolution + 31) // 32
    return (final_size, final_size)


class NextViT(nn.Module):
    """NextViT wrapper for LUMA.

    Channel-count plumbing for NextViT-S (depths=[…, 4]; last stage emits
    [768, 768, 768, 1024]):

      • Queries (``eomt.q``) are sized to ``self.embed_dim``, which we derive
        from the first EoMT-region block's input channels. For NextViT-S with
        ``num_blocks=2`` that's 768 — the second-to-last block takes 768 in.
      • The last block transitions 768 → 1024 via its internal ``patch_embed``.
        Its ``block.mlp`` runs at 1024 channels. So for the q-update inside
        that block, we project queries 768 → 1024 *before* feeding them to
        ``block.mlp``. The projection is pre-allocated per EoMT-region block
        in ``self.eomt_q_projs`` (Identity when dims already match).
      • Post-blocks ``x`` and the final ``q`` come out at 1024 channels;
        ``self.projection`` and ``self.q_projection`` project both back to
        ``embed_dim`` (768) for EoMT's prediction heads.

    All modules are allocated in ``__init__`` so they're in the optimizer,
    saved to checkpoints, and synced under DDP from step 0. There is no
    lazy-on-first-forward module creation anywhere — that pattern broke DDP
    and skipped the optimizer for the affected weights.
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        num_blocks: int,
        num_q: int,
        backbone_name: str = "nextvit_small_cus",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            num_classes=0,
            img_size=img_size,
        )
        self.backbone = load_swin_ckpt_ignore_attn_mask(self.backbone, ckpt_path)

        # --------------------------------------------------------------
        # Block bookkeeping
        # --------------------------------------------------------------
        all_blocks = list(self.backbone.features)
        if num_blocks > len(all_blocks):
            raise ValueError(
                f"num_blocks={num_blocks} > len(backbone.features)={len(all_blocks)}"
            )
        eomt_blocks = all_blocks[-num_blocks:]

        # ``embed_dim`` for LUMA.q is the input-channel count of the FIRST
        # EoMT-region block — that's the channel dim where queries get
        # concat'd to image features. Reading it from the backbone instead of
        # hardcoding 768 means this wrapper still works if anyone swaps in a
        # different NextViT variant.
        self.embed_dim = eomt_blocks[0].in_channels
        self.num_blocks = num_blocks
        self.num_q = num_q

        # Final block's output dim — where x lands at post_blocks time.
        # For NextViT-S this is 1024.
        last_block_out = eomt_blocks[-1].out_channels

        # Per-EoMT-block q-projection. Each maps ``embed_dim → block.mlp.conv1.in_channels``
        # (which equals block.out_channels — the dim block.mlp operates at).
        # Identity when no projection is needed. The wrapper holds these as a
        # ModuleList so every Linear is registered before optimizer init.
        self.eomt_q_projs = nn.ModuleList()
        for blk in eomt_blocks:
            mlp_in = blk.mlp.conv1.in_channels
            if mlp_in == self.embed_dim:
                self.eomt_q_projs.append(nn.Identity())
            else:
                proj = nn.Linear(self.embed_dim, mlp_in)
                nn.init.kaiming_normal_(proj.weight)
                nn.init.constant_(proj.bias, 0)
                self.eomt_q_projs.append(proj)

        # Post-blocks projections: backbone last-block-out → embed_dim. One
        # for x (token grid), one for q. Identity when dims already match.
        if last_block_out == self.embed_dim:
            self.projection = nn.Identity()
            self.q_projection = nn.Identity()
        else:
            x_proj = nn.Linear(last_block_out, self.embed_dim)
            nn.init.kaiming_normal_(x_proj.weight)
            nn.init.constant_(x_proj.bias, 0)
            self.projection = x_proj
            q_proj = nn.Linear(last_block_out, self.embed_dim)
            nn.init.kaiming_normal_(q_proj.weight)
            nn.init.constant_(q_proj.bias, 0)
            self.q_projection = q_proj

        # Token-space LayerNorm at embed_dim. ``xq`` entering every EoMT-region
        # block has last-dim == embed_dim (queries are embed_dim and x is
        # embed_dim before block(x) runs), so one shared LN suffices.
        self.token_norm = nn.LayerNorm(self.embed_dim)

        # Encoder-final BN reused from the backbone (pretrained).
        self.norm = self.backbone.norm

        # --------------------------------------------------------------
        # Standard wrapper attributes used by EoMT / LLRD
        # --------------------------------------------------------------
        self.patch_size = to_doubletuple(32)
        self.grid_size = get_nextvit_final_grid(img_size[0])
        self.num_prefix_tokens = 0
        self.attn_multiplier = multiplier

        self.depths = self.backbone.depths
        self.num_heads = []
        self.blocks = nn.ModuleList()
        for block in self.backbone.features:
            self.blocks.append(block)
            self.num_heads.append(block.mhca.groups)

        # Pixel-normalization buffers
        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        # Spatial state cache (used by run_block)
        self.B, self.C, self.H, self.W = 0, 0, 0, 0

    def pre_block(self, x: torch.Tensor):
        """Run the stem (ConvBNReLU sequence) — emits (B, stem_out_ch, H/4, W/4)."""
        x = self.backbone.stem(x)
        self.B, self.C, self.H, self.W = x.shape
        return x

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i):
        """Block forward with optional EoMT integration in the last
        ``num_blocks`` calls."""
        in_eomt_region = i >= len(self.blocks) - eomt_obj.num_blocks

        if in_eomt_region:
            k = i - (len(self.blocks) - eomt_obj.num_blocks)

            # Spatial → tokens
            if x.dim() == 4:
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)  # (B, N, C)
                self.grid_size = (H, W)

            # Pre-attn LayerNorm on concatenated query+feature tokens.
            xq = torch.cat(
                ((q if q.dim() == 3 else q[None, :, :].expand(x.shape[0], -1, -1)), x), dim=1
            )
            pre_attn = self.token_norm(xq)
            x_tok = pre_attn[:, eomt_obj.num_q:, :]
            q_tok = pre_attn[:, :eomt_obj.num_q, :]

            attn_mask = eomt_obj.predict_and_mask(x_tok, q_tok, i - len(self.blocks))

            # Cross-attention via EoMT's attn module.
            after_eomt = torch.cat((q_tok, x_tok), dim=1)
            attn_idx = i - len(self.blocks)
            new_x = eomt_obj.attn[attn_idx](after_eomt, num_queries=eomt_obj.num_q, attn_mask=attn_mask)
            xq = xq + eomt_obj.dp(eomt_obj.ls_list[attn_idx](new_x))
            x_new = xq[:, eomt_obj.num_q:, :]
            q_new = xq[:, :eomt_obj.num_q, :]

            # Image features: reshape back to spatial and run the full block.
            B, N, C = x_new.shape
            H, W = self.grid_size
            x_spatial = x_new.transpose(1, 2).reshape(B, C, H, W)
            x_spatial = block(x_spatial)

            self.B, self.C, self.H, self.W = x_spatial.shape
            self.grid_size = (self.H, self.W)

            # Query self-attention (Mask2Former order: cross-attn, self-attn, MLP).
            q_new = eomt_obj.apply_query_self_attn(q_new, attn_idx)

            # Query update: mirror block.mlp's channel mixing. Project q to
            # the block's mlp input dim (Identity when no transition), then
            # run the same `norm → mlp → drop_path → residual` path the
            # block applies to x internally.
            q_proj = self.eomt_q_projs[k](q_new)                       # (B, num_q, mlp_in)
            q_reshaped = q_proj.transpose(1, 2).unsqueeze(-1)          # (B, mlp_in, num_q, 1)
            # NCB has ``block.norm``; NTB has ``block.norm2`` for the mlp branch.
            # Both are BatchNorm2d *shared* with the image path (``block(x_spatial)``
            # above ran the same module), so a train-mode query pass gives them a
            # second momentum-0.1 update per step off query statistics — the
            # ``efficient_net`` bug, and here the queries even land last. Freeze the
            # norm for the query pass so train and eval agree; see CLAUDE.md and
            # ``models/efficient_net.py``.
            q_norm = block.norm if hasattr(block, "norm") else block.norm2
            bn_was_training = q_norm.training
            q_norm.eval()
            q_normed = q_norm(q_reshaped)
            q_norm.train(bn_was_training)
            q_out = block.mlp(q_normed).squeeze(-1).transpose(1, 2)    # (B, num_q, mlp_in)
            q = q_proj + block.mlp_path_dropout(q_out)                  # residual

            return x_spatial, q

        # Non-EoMT region — plain block forward, queries pass through.
        x = block(x)
        if x.dim() == 4:
            self.B, self.C, self.H, self.W = x.shape
            self.grid_size = (self.H, self.W)
        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        """Apply the backbone's final BN, then project both x and q from the
        backbone's last-block output dim back to ``embed_dim`` so EoMT's
        prediction heads (sized at embed_dim) work."""
        x = self.norm(x)
        # (B, C, H, W) → (B, H, W, C) → Linear → (B, H, W, embed_dim) → (B, embed_dim, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.projection(x)
        x = x.permute(0, 3, 1, 2)
        q = self.q_projection(q)
        return x, q


def load_swin_ckpt_ignore_attn_mask(model, ckpt_path):
    if ckpt_path is None:
        return model
    # 1) Load checkpoint state dict
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if 'model_state' in checkpoint:
        state_dict = {
            k[7:] if k.startswith('module.') else k: v
            for k, v in checkpoint['model_state'].items()
        }
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Drop classification-head weights from the checkpoint — they don't exist on
    # this model (timm was built with num_classes=0) and their shape wouldn't
    # match anyway. Same defensive filter as efficient_net.py / resnet.py / etc.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}

    # 2) Load with strict=False to tolerate any remaining non-critical diffs
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {ckpt_path}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model


def pad2d_concat(b, a):
    P = a.shape[2] + b.shape[2]
    B, C = a.shape[:2]
    out = a.new_zeros(B, C, P, P)
    out[..., :a.shape[2], :a.shape[2]] = a
    out[..., a.shape[2]:P, a.shape[2]:P] = b
    return out


def pad2d_unconcat(out, a_size, b_size=None, clone=True):
    # out: [B,C,P,P], where P = a_size + b_size
    if b_size is None:
        b_size = out.shape[-1] - a_size  # assumes square P x P

    a = out[..., :a_size, :a_size]
    b = out[..., a_size:a_size + b_size, a_size:a_size + b_size]

    if clone:
        a = a.clone()
        b = b.clone()
    return b, a  # matches your original arg order (b, a)
