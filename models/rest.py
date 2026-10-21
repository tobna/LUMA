from typing import Optional

import timm
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import models.backbones.rest  # noqa: F401


class rest(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name: str = "rest_base",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            num_classes=0,
        )

        if ckpt_path is not None:
            self.backbone = load_rest_ckpt(self.backbone, ckpt_path)

        # ResT overall stride is 32 (BasicStem stride=4, three patch_embed stride=2 each)
        self.patch_size = (32, 32)
        H, W = img_size
        self.grid_size = (H // 32, W // 32)
        self.num_prefix_tokens = 0
        self.attn_multiplier = multiplier

        # Only the last (stage4) blocks are exposed to EoMT — they operate on
        # flat (B, N, C) tokens at 1/32 resolution with sr_ratio=1.
        self.embed_dim = self.backbone.embed_dim
        self.num_heads = [self.backbone.stage4[0].attn.num_heads]
        self.depths = [len(self.backbone.stage4)]
        self.blocks = list(self.backbone.stage4)
        self.norm = self.backbone.norm

        # Track spatial dims of stage-4 tokens (H/32, W/32)
        self._H = H // 32
        self._W = W // 32

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std  = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        """Run stem + stages 1-3 + patch_embeds to reach flat stage-4 tokens."""
        B = x.shape[0]

        # BasicStem: (B,3,H,W) -> (B, C0, H/4, W/4)
        x = self.backbone.stem(x)
        _, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, N1, C0)

        # Stage 1
        for blk in self.backbone.stage1:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # Stage 2
        x, (H, W) = self.backbone.patch_embed_2(x)
        for blk in self.backbone.stage2:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # Stage 3
        x, (H, W) = self.backbone.patch_embed_3(x)
        for blk in self.backbone.stage3:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # Stage 4 patch embed -> flat tokens (B, N4, embed_dim)
        x, (H, W) = self.backbone.patch_embed_4(x)
        self._H, self._W = H, W
        return x

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i: int):
        B = x.shape[0]
        H, W = self._H, self._W

        if i >= len(self.blocks) - eomt_obj.num_blocks:
            eomt_idx = i - (len(self.blocks) - eomt_obj.num_blocks)

            if q.dim() == 2:
                q_batched = q[None].expand(B, -1, -1)
            else:
                q_batched = q

            xq = torch.cat([q_batched, x], dim=1)
            pre_attn = block.norm1(xq)
            q_tok = pre_attn[:, : eomt_obj.num_q, :]
            x_tok = pre_attn[:, eomt_obj.num_q :, :]

            attn_mask = eomt_obj.predict_and_mask(x_tok, q_tok, eomt_idx)

            after_eomt = torch.cat([q_tok, x_tok], dim=1)
            new_x = eomt_obj.attn[eomt_idx](self.norm(after_eomt), num_queries=eomt_obj.num_q, attn_mask=attn_mask)
            xq = xq + eomt_obj.dp(eomt_obj.ls_list[eomt_idx](new_x))

            q_batched = xq[:, : eomt_obj.num_q, :]
            x        = xq[:, eomt_obj.num_q :, :]

            # ResTBlock.forward requires (x, H, W); sr_ratio=1 in stage4 so no reshape needed
            x = x + block.drop_path1(block.attn(block.norm1(x), H, W))
            x = x + block.drop_path2(block.mlp(block.norm2(x)))
            # Query self-attention (Mask2Former order: cross-attn, self-attn, MLP).
            q_batched = eomt_obj.apply_query_self_attn(q_batched, eomt_idx)
            q_batched = q_batched + block.drop_path2(block.mlp(block.norm2(q_batched)))
            q = q_batched

        else:
            x = block(x, H, W)

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        x = self.norm(x)
        return x, q


def load_rest_ckpt(model, ckpt_path: str):
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
    print("ResT — Missing keys :", missing)
    print("ResT — Unexpected keys:", unexpected)
    return model
