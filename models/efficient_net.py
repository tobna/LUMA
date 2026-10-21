from typing import Optional
import timm
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn

from models.backbones.convViT import to_doubletuple


def get_efficientvit_final_grid(input_resolution: int):
    final_size = (input_resolution + 31) // 32
    return (final_size, final_size)


class EfficientNet(nn.Module):
    """EfficientViT-B* wrapper for LUMA.

    Mirrors the conventions every other transformer-backbone wrapper in this
    repo follows (see ``vit.py``, ``wave.py``, ``swin.py``):

    * Per-EoMT-block pre-attention LayerNorm (``self.eomt_pre_norms[k]``) —
      the same role ``block.norm1`` plays inside a timm ViT block. We have to
      allocate fresh ones because EfficientViT is post-norm BatchNorm and has
      no usable pre-attention LayerNorm to reuse.
    * One shared encoder-final LayerNorm (``self.norm``) used for both the
      KV side of the cross-attention and ``post_blocks``. Matches how
      ``vit.py`` reuses ``backbone.norm``.
    * Queries are updated through the block's own FFN sub-layer with the
      spatial-mixing op skipped (the ``wave.py`` ``_pvt2ffn_no_dwconv``
      pattern): pretrained 1x1 convs are reused; the 3x3 depthwise conv is
      not applied to queries because they have no meaningful spatial
      neighbourhood.
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        num_blocks: int,
        num_q: int,
        backbone_name: str = "efficient_vit_b2",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            num_classes=0,
        )
        # Load and unwrap from EfficientViTCls(backbone, head) → EfficientViTBackbone.
        self.backbone = load_swin_ckpt_ignore_attn_mask(self.backbone, ckpt_path).backbone

        self.embed_dim = self.backbone.width_list[-1]
        self.patch_size = to_doubletuple(32)
        self.grid_size = get_efficientvit_final_grid(img_size[0])
        self.num_prefix_tokens = 0
        self.attn_multiplier = multiplier

        self.blocks = nn.ModuleList()
        self.depths = []
        self.num_heads = []
        self.patch_embed_dict = nn.ModuleDict()
        self.patch_embed_dict["0"] = self.backbone.input_stem

        # Flatten EfficientViT stages into a single block list so LLRD and the
        # EoMT loop can iterate by index. EfficientViT has no head count per
        # block; we synthesize a dummy num_heads matching the rough scale
        # transformer-block count of the later stages.
        for stage_idx, stage in enumerate(self.backbone.stages):
            stage_depth = len(stage)
            self.depths.append(stage_depth)
            stage_heads = 8 if stage_idx >= 2 else 1  # dummy
            for block in stage:
                self.blocks.append(block)
                self.num_heads.append(stage_heads)

        # ------------------------------------------------------------------
        # EoMT integration: norms and shape contract
        # ------------------------------------------------------------------
        # ``num_blocks`` and ``num_q`` must match the corresponding values on
        # the parent ``EoMT`` module. Duplicate them in the YAML (the rest of
        # the codebase uses ``link_arguments`` for cross-module values; we
        # avoid that here to keep ``main.py`` untouched).
        self.num_blocks = num_blocks
        self.num_q = num_q

        # Per-EoMT-block pre-attention LayerNorm. Fresh, trainable, one per
        # block in the EoMT region. Plays the role of ``block.norm1`` in ViT.
        self.eomt_pre_norms = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(num_blocks)]
        )
        # Shared encoder-final LayerNorm. Used for cross-attn KV and
        # ``post_blocks``. Plays the role of ``self.norm = backbone.norm`` in
        # ViT (which is also reused by both call sites).
        self.norm = nn.LayerNorm(self.embed_dim)

        # Sanity-check: each EoMT-region block must be an EfficientViTBlock
        # (has ``local_module.main`` = MBConv). Standard configs put the
        # last ``num_blocks=2`` blocks inside the final stage of B2 (which
        # contains 5 EfficientViTBlocks after one downsample MBConv), so this
        # holds. Bumping ``num_blocks`` past the final-stage transformer count
        # would otherwise silently land on a ResidualBlock(MBConv) downsample
        # with a non-Identity shortcut — we want a loud error there.
        eomt_blocks = list(self.blocks)[-num_blocks:]
        for k, b in enumerate(eomt_blocks):
            assert hasattr(b, "local_module") and hasattr(b.local_module, "main"), (
                f"EoMT block index {k} (global idx {len(self.blocks) - num_blocks + k}, "
                f"type={type(b).__name__}) is not an EfficientViTBlock — "
                f"num_blocks={num_blocks} likely spans past the final stage's "
                f"transformer blocks. Reduce num_blocks or extend run_block to "
                f"handle this block layout."
            )

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        # Stem only; the EfficientViT stages are flattened into ``self.blocks``.
        return self.patch_embed_dict["0"](x)

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i: int):
        """Run one block in the EoMT-aware forward.

        Image features flow through the block spatially. For the last
        ``num_blocks`` blocks (the EoMT region):

        1. concat queries with image-feature tokens
        2. apply per-block LN → split → intermediate ``_predict``
        3. cross-attend on ``self.norm(after_eomt)`` (shared encoder-final
           LN, matching ``vit.py``'s KV-side norm)
        4. residual-add cross-attn output to the **un-normed** ``xq``
        5. image features: run ``block(x)`` (full block: context + local)
        6. queries: mirror ``block.local_module`` with the spatial 3x3
           DWConv skipped (channel-mixing-only — the wave pattern). Residual
           added because ``block.local_module.shortcut`` is Identity.
        """
        assert x.dim() == 4, (
            f"EfficientNet.run_block expects spatial input (B,C,H,W); got {tuple(x.shape)}. "
            f"pre_block returns 4D and run_block returns 4D — the loop should stay spatial."
        )
        B, C, H, W = x.shape
        # Record grid_size *before* the block so that ``_predict`` inside the
        # EoMT region reshapes ``x_tok`` (which lives in the block's input
        # token space) with the correct H, W.
        self.grid_size = (H, W)

        in_eomt = i >= len(self.blocks) - eomt_obj.num_blocks
        if in_eomt:
            k = i - (len(self.blocks) - eomt_obj.num_blocks)

            # 1) flatten image features to tokens; prepend the (expanded) queries
            x_tokens = x.flatten(2).transpose(1, 2)              # (B, H*W, C)
            xq = torch.cat(
                ((q if q.dim() == 3 else q[None, :, :].expand(B, -1, -1)), x_tokens), dim=1
            )                                                     # (B, num_q + N, C)

            # 2) per-block pre-attn LayerNorm — used only to feed the prediction
            #    heads and as input to the cross-attn (after slicing/recombining).
            #    The un-normed ``xq`` remains the residual stream (no double-norm
            #    on the residual path).
            pre_attn = self.eomt_pre_norms[k](xq)
            q_tok = pre_attn[:, : eomt_obj.num_q, :]
            x_tok = pre_attn[:, eomt_obj.num_q :, :]

            attn_mask = eomt_obj.predict_and_mask(x_tok, q_tok, i - len(self.blocks))

            # 3) cross-attention. We mirror ``vit.py`` exactly here: the KV-side
            #    of the cross-attn sees ``self.norm(after_eomt)``, where
            #    ``after_eomt`` is built from the already-LN'd ``pre_attn``
            #    slices. This is two LayerNorms (per-block + encoder-final) in
            #    sequence, intentional and matches the working ViT path.
            after_eomt = torch.cat((q_tok, x_tok), dim=1)
            attn_idx = i - len(self.blocks)                       # negative index into EoMT's lists
            new_x = eomt_obj.attn[attn_idx](
                self.norm(after_eomt), num_queries=eomt_obj.num_q, attn_mask=attn_mask
            )
            xq = xq + eomt_obj.dp(eomt_obj.ls_list[attn_idx](new_x))

            q_new = xq[:, : eomt_obj.num_q, :]
            x_tokens = xq[:, eomt_obj.num_q :, :]

            # 4) queries: mirror block.local_module (= ResidualBlock(MBConv,
            #    Identity)) but skip the spatial 3x3 DWConv. This is the
            #    direct EfficientViT analogue of wave.py's _pvt2ffn_no_dwconv:
            #    pretrained 1x1 convs (inverted_conv, point_conv) are reused;
            #    the depthwise conv has no meaning on an unordered query set
            #    and is dropped.
            #
            #    ``point_conv.norm`` is a BatchNorm2d *shared* with the image
            #    path (``block(x)`` below runs the same module), so leaving it
            #    in train mode here gives it two momentum-0.1 updates per step
            #    — one from the queries, one from the features — and the fixed
            #    point of that is ~47% query statistics. Training never notices
            #    (batch stats), evaluation gets image features rescaled by an
            #    order of magnitude: ADE20K semantic 512 trained to the same
            #    loss as efficientmod (CE 0.60) and validated at 0.039 mIoU vs
            #    0.402. Freezing the norm for the query pass makes the two
            #    modes agree — queries then see a fixed affine built from the
            #    feature statistics, which the following LayerNorms absorb.
            # Query self-attention (Mask2Former order: cross-attn, self-attn, MLP).
            q_new = eomt_obj.apply_query_self_attn(q_new, attn_idx)
            mb = block.local_module.main                          # MBConv
            q_in = q_new.transpose(1, 2).unsqueeze(-1)            # (B, C, num_q, 1)
            y = mb.inverted_conv(q_in)                            # 1x1 + Hardswish (pretrained)
            # mb.depth_conv intentionally skipped
            bn_was_training = mb.point_conv.norm.training
            mb.point_conv.norm.eval()
            y = mb.point_conv(y)                                  # 1x1 + BN (pretrained, frozen)
            mb.point_conv.norm.train(bn_was_training)
            y = y.squeeze(-1).transpose(1, 2)                     # (B, num_q, C)
            # block.local_module.shortcut is Identity → plain residual.
            q = q_new + y

            # 5) image features: full block forward in spatial form.
            x = x_tokens.transpose(1, 2).reshape(B, C, H, W)
            x = block(x)

        else:
            # Non-EoMT region — vanilla block forward, queries pass through.
            x = block(x)

        # Block output may have different spatial dims (downsample blocks);
        # refresh grid_size for the next iteration's ``_predict`` reshape.
        self.grid_size = tuple(x.shape[-2:])
        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        # Final encoder-side normalization on the feature tokens. Apply LN
        # over the channel dim (token form), then reshape back to spatial so
        # ``LUMA._predict``'s ``len(x.shape) == 4`` branch fires and the
        # encoder-grid reshape is skipped.
        B, C, H, W = x.shape
        x_tokens = self.norm(x.flatten(2).transpose(1, 2))        # (B, N, C)
        x = x_tokens.transpose(1, 2).reshape(B, C, H, W)
        return x, q


def load_swin_ckpt_ignore_attn_mask(model, ckpt_path):
    if ckpt_path is None:
        return model
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state" in checkpoint:
        state_dict = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in checkpoint["model_state"].items()
        }
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # The classification head is discarded immediately after this load (see the
    # ``.backbone`` unwrap in ``EfficientNet.__init__``), and its out_features
    # (1000 for the freshly-built timm head vs. 10450 for the IN21k checkpoint)
    # won't match anyway. Drop head.* keys to keep load_state_dict happy.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {ckpt_path}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model
