from typing import Optional

import timm
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import models.backbones.wave  # noqa: F401
from models.backbones.wave import WaveViT


def _pvt2ffn_no_dwconv(mlp, x: torch.Tensor) -> torch.Tensor:
    """Run PVT2FFN on tokens that have no valid spatial layout (e.g. query tokens).
    Bypasses the DWConv which requires x to be reshapeable to (B, C, H, W)."""
    x = mlp.fc1(x)
    x = mlp.act(x)
    x = mlp.fc2(x)
    return x


class wave(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name: str = "wavevit_b",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone: WaveViT = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            num_classes=0,
        )

        if ckpt_path is not None:
            self.backbone = load_wavevit_ckpt(self.backbone, ckpt_path)

        # WaveViT Stem: stride-2 conv x2 -> overall stride 4; 3 more stride-2
        # DownSamples across stages -> total stride = 4 * 2^3 = 32
        self.patch_size = (32, 32)
        H, W = img_size
        self.grid_size = (H // 32, W // 32)
        self.num_prefix_tokens = 0
        self.attn_multiplier = multiplier

        self.embed_dim = self.backbone.embed_dim
        self.num_heads = [self.backbone.block4[0].attn.num_heads]
        self.depths = [len(self.backbone.block4)]

        # Expose only stage-4 blocks to EoMT
        self.blocks = list(self.backbone.block4)
        self.norm = self.backbone.norm4

        self._H = H // 32
        self._W = W // 32

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std  = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        """Run stages 1-3 (patch embeds + WaveBlocks + inter-stage norms) to produce
        flat (B, N4, embed_dims[3]) stage-4 tokens."""
        B = x.shape[0]

        for i in range(self.backbone.num_stages - 1):  # stages 0,1,2
            patch_embed = getattr(self.backbone, f"patch_embed{i + 1}")
            block       = getattr(self.backbone, f"block{i + 1}")
            norm        = getattr(self.backbone, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # back to (B, C, H, W)

        # Stage-4 patch embed -> flat tokens
        patch_embed4 = getattr(self.backbone, "patch_embed4")
        x, H, W = patch_embed4(x)
        self._H, self._W = H, W
        return x  # (B, H/32 * W/32, embed_dim)

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

            # Stage-4 uses std_att (Attention), both attn and mlp take (x, H, W)
            x = x + block.drop_path(block.attn(block.norm1(x), H, W))
            x = x + block.drop_path(block.mlp(block.norm2(x), H, W))

            # PVT2FFN contains a DWConv that reshapes to (B, C, H, W) — this is only
            # valid for image tokens. For query tokens, bypass DWConv and run fc1/act/fc2.
            # Query self-attention (Mask2Former order: cross-attn, self-attn, MLP).
            q_batched = eomt_obj.apply_query_self_attn(q_batched, eomt_idx)
            q_batched = q_batched + block.drop_path(
                _pvt2ffn_no_dwconv(block.mlp, block.norm2(q_batched))
            )
            q = q_batched

        else:
            x = block(x, H, W)

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        x = self.norm(x)
        return x, q


def load_wavevit_ckpt(model: WaveViT, ckpt_path: str) -> WaveViT:
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
    state_dict = {k: v for k, v in state_dict.items()
                  if not k.startswith("head.") and not k.startswith("aux_head.")}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("WaveViT — Missing keys :", missing)
    print("WaveViT — Unexpected keys:", unexpected)
    return model
