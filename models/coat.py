from typing import Optional

import timm
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import models.backbones.coat  # noqa: F401


class coat(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name: str = "coat_lite_small",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            return_interm_layers=False,
            num_classes=0,
        )

        if ckpt_path is not None:
            self.backbone = load_coat_ckpt(self.backbone, ckpt_path)

        self.num_prefix_tokens = 0
        self.attn_multiplier = multiplier

        # Derive architecture dims directly from the created backbone
        serial_depths = [
            len(self.backbone.serial_blocks1),
            len(self.backbone.serial_blocks2),
            len(self.backbone.serial_blocks3),
            len(self.backbone.serial_blocks4),
        ]
        parallel_depth = len(self.backbone.parallel_blocks) if self.backbone.parallel_blocks else 0
        num_heads = self.backbone.serial_blocks1[0].factoratt_crpe.num_heads

        num_stages = len(serial_depths) + (1 if parallel_depth > 0 else 0)
        self.num_heads = [num_heads] * num_stages
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = (4, 4)
        self.depths = list(serial_depths) + ([parallel_depth] if parallel_depth > 0 else [])

        H, W = img_size
        self.grid_size = (
            H // (4 * 2 ** (len(serial_depths) - 1)),
            W // (4 * 2 ** (len(serial_depths) - 1)),
        )

        self._serial_blocks = [
            blk
            for stage in [
                self.backbone.serial_blocks1,
                self.backbone.serial_blocks2,
                self.backbone.serial_blocks3,
                self.backbone.serial_blocks4,
            ]
            for blk in stage
        ]
        self._parallel_blocks = list(self.backbone.parallel_blocks) if parallel_depth > 0 else []
        self.blocks = self._serial_blocks + self._parallel_blocks

        self._stage_ends = []
        s = 0
        for d in serial_depths:
            s += d
            self._stage_ends.append(s)

        self.norm = self.backbone.norm4

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std  = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        x1, (H1, W1) = self.backbone.patch_embed1(x)
        x1 = self.backbone.insert_cls(x1, self.backbone.cls_token1)
        self._sizes = [(H1, W1), None, None, None]
        self._B = x.shape[0]
        self._par_x2 = self._par_x3 = self._par_x4 = None
        return x1

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i: int):
        """Block forward with EoMT cross-attention injected in the LAST
        ``num_blocks`` *parallel* blocks (not the serial blocks).

        Rationale: ``self.blocks = serial_blocks + parallel_blocks`` is in
        forward-execution order, so for LLRD's "deepest index gets the least
        discount" assumption to be meaningful, the EoMT-attending blocks
        have to sit at the highest indices. The parallel stack is also where
        the deepest pretrained features live, which is the natural place to
        attach query-side cross-attention. We hook the cross-attn onto the
        ``self._par_x4`` stream (the finest-scale token stream the parallel
        block fuses across scales) *before* calling the parallel block, so
        the block's multi-scale mixing sees the query-modified x4. Query
        updates use the parallel block's own ``norm24 / mlp4`` (the x4 MLP
        sub-layer), mirroring the ``vit.py`` symmetry.
        """
        B = x.shape[0]
        num_serial = len(self._serial_blocks)
        num_parallel = len(self._parallel_blocks)
        total = num_serial + num_parallel
        eomt_window_start = total - eomt_obj.num_blocks

        # === Parallel-block path (hosts the EoMT region) ===
        if i >= num_serial:
            par_idx = i - num_serial
            block_p = self._parallel_blocks[par_idx]
            in_eomt = i >= eomt_window_start

            if in_eomt:
                # x4 stream — finest scale, deepest semantic features.
                # It carries a cls token; queries don't touch it, so we
                # split and reattach around the cross-attention.
                x4 = self._par_x4
                cls4 = x4[:, :1, :]
                img4 = x4[:, 1:, :]

                if q.dim() == 2:
                    q_batched = q[None].expand(B, -1, -1)
                else:
                    q_batched = q

                # Pre-attention norm via the parallel block's own x4-side
                # pre-conv-attention norm. Same role ``block.norm1`` plays
                # in a ViT block: this LN was trained to feed attention.
                xq = torch.cat([q_batched, img4], dim=1)
                pre_attn = block_p.norm14(xq)
                q_tok = pre_attn[:, : eomt_obj.num_q, :]
                x_tok = pre_attn[:, eomt_obj.num_q :, :]

                attn_mask = eomt_obj.predict_and_mask(x_tok, q_tok, i - total)

                # EoMT cross-attention. Attn-list is indexed from the END of
                # self.blocks (negative index), so attn[i - total] picks up
                # the correct per-EoMT-block attn module regardless of where
                # the EoMT region lives.
                after_eomt = torch.cat([q_tok, x_tok], dim=1)
                attn_idx = i - total
                new_x = eomt_obj.attn[attn_idx](
                    self.norm(after_eomt), num_queries=eomt_obj.num_q, attn_mask=attn_mask
                )
                xq = xq + eomt_obj.dp(eomt_obj.ls_list[attn_idx](new_x))

                q_batched = xq[:, : eomt_obj.num_q, :]
                img4_new = xq[:, eomt_obj.num_q :, :]

                # Re-attach cls and seed the parallel block's x4 input
                # with the query-modified stream. The block's cross-scale
                # mixing then sees these modifications.
                self._par_x4 = torch.cat([cls4, img4_new], dim=1)

                # Run the parallel block normally (now with updated x4).
                x1, x2, x3, x4 = block_p(
                    x, self._par_x2, self._par_x3, self._par_x4,
                    sizes=self._sizes,
                )
                self._par_x2, self._par_x3, self._par_x4 = x2, x3, x4
                x = x4

                # Query self-attention (Mask2Former order: cross-attn, self-attn, MLP).
                q_batched = eomt_obj.apply_query_self_attn(q_batched, attn_idx)

                # Query MLP — mirror the block's x4 MLP sub-layer
                # (norm24 → mlp4 → drop_path + residual). Same shape the
                # block applies internally to x4.
                q = q_batched + block_p.drop_path(
                    block_p.mlp4(block_p.norm24(q_batched))
                )
            else:
                # Non-EoMT parallel block — vanilla multi-scale forward.
                x1, x2, x3, x4 = block_p(
                    x, self._par_x2, self._par_x3, self._par_x4,
                    sizes=self._sizes,
                )
                self._par_x2, self._par_x3, self._par_x4 = x2, x3, x4
                x = x4

            return x, q

        # === Serial-block path (no EoMT cross-attention here anymore) ===
        stage_idx = sum(i >= end for end in self._stage_ends)
        stage_idx = min(stage_idx, 3)
        H, W = self._sizes[stage_idx]

        x = block(x, size=(H, W))

        # Stage-boundary patch_embeds (unchanged): when a serial stage ends,
        # apply the next stage's patch_embed and seed the corresponding x2/x3/x4
        # cache so the parallel blocks downstream have them ready.
        idx_1based = i + 1
        next_embeds = [
            (self._stage_ends[0], self.backbone.patch_embed2, self.backbone.cls_token2, 1),
            (self._stage_ends[1], self.backbone.patch_embed3, self.backbone.cls_token3, 2),
            (self._stage_ends[2], self.backbone.patch_embed4, self.backbone.cls_token4, 3),
        ]
        for stage_end, pembed, cls_tok_param, next_idx in next_embeds:
            if idx_1based == stage_end:
                x_nocls = self.backbone.remove_cls(x)
                x_nocls = x_nocls.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                x_new, (Hn, Wn) = pembed(x_nocls)
                x = self.backbone.insert_cls(x_new, cls_tok_param)
                self._sizes[next_idx] = (Hn, Wn)
                cache = [self._par_x2, self._par_x3, self._par_x4]
                cache[next_idx - 1] = x
                self._par_x2, self._par_x3, self._par_x4 = cache
                break

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        x = self.backbone.remove_cls(x)
        x = self.norm(x)
        return x, q


def load_coat_ckpt(model, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state" in checkpoint:
        state_dict = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in checkpoint["model_state"].items()
        }
    else:
        state_dict = checkpoint

    state_dict = {
        k[9:] if k.startswith("backbone.") else k: v
        for k, v in state_dict.items()
    }
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("CoaT — Missing keys :", missing)
    print("CoaT — Unexpected keys:", unexpected)
    return model
