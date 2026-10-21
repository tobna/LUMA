from typing import Optional

import timm
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import models.backbones.resnet  # noqa: F401
from models.backbones.resnet import ResNet


class ResNetSeg(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name: str = "resnet50",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone: ResNet = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            num_classes=0,
        )

        if ckpt_path is not None:
            self.backbone = load_resnet_ckpt(self.backbone, ckpt_path)

        # layer4 output: 2048 channels for resnet50, stride 32
        self.embed_dim = self.backbone.layer4[-1].conv3.out_channels
        self.patch_size = (32, 32)
        H, W = img_size
        self.grid_size = (H // 32, W // 32)
        self.num_prefix_tokens = 0
        self.attn_multiplier = multiplier

        # Wrap each layer4 bottleneck so EoMT can call norm1/norm2/mlp on them
        self.backbone.layer4 = nn.ModuleList(
            [_BottleneckEoMTWrapper(blk, self.embed_dim) for blk in self.backbone.layer4]
        )

        self.depths = [len(self.backbone.layer4)]
        self.num_heads = [8]  # 2048 / 256 per head
        self.blocks = list(self.backbone.layer4)
        self.norm = nn.LayerNorm(self.embed_dim)

        self._H = H // 32
        self._W = W // 32

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor) -> torch.Tensor:
        """Run stem + layer1-3, return spatial map ready for layer4."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        # (B, 1024, H/16, W/16) — layer4 blocks will stride to H/32
        return x

    def run_block(self, block, eomt_obj, x, q, i: int):
        """x is (B,C,H,W) spatial for first block, (B,N,C) tokens thereafter."""
        B = x.shape[0]
        total = len(self.blocks)

        # Re-inflate to spatial if we received tokens
        if x.dim() == 3:
            x = x.transpose(1, 2).reshape(B, self.embed_dim, self._H, self._W)

        # Run the CNN bottleneck block on spatial input
        x_spatial = block.bottleneck(x)  # (B, C_out, H', W')
        _, _, H, W = x_spatial.shape
        self._H, self._W = H, W

        # Flatten to tokens
        x_tok = x_spatial.flatten(2).transpose(1, 2)  # (B, N, C)

        if i >= total - eomt_obj.num_blocks:
            eomt_idx = i - (total - eomt_obj.num_blocks)

            if q.dim() == 2:
                q_batched = q[None].expand(B, -1, -1)
            else:
                q_batched = q

            xq = torch.cat([q_batched, x_tok], dim=1)
            pre_attn = block.norm1(xq)
            q_tok = pre_attn[:, : eomt_obj.num_q, :]
            x_tok = pre_attn[:, eomt_obj.num_q :, :]

            attn_mask = eomt_obj.predict_and_mask(x_tok, q_tok, eomt_idx)

            after_eomt = torch.cat([q_tok, x_tok], dim=1)
            new_x = eomt_obj.attn[eomt_idx](self.norm(after_eomt), num_queries=eomt_obj.num_q, attn_mask=attn_mask)
            xq = xq + eomt_obj.dp(eomt_obj.ls_list[eomt_idx](new_x))

            q_batched = xq[:, : eomt_obj.num_q, :]
            x_tok = xq[:, eomt_obj.num_q :, :]

            # Query self-attention (Mask2Former order: cross-attn, self-attn, MLP).
            q_batched = eomt_obj.apply_query_self_attn(q_batched, eomt_idx)
            q_batched = block.mlp(block.norm2(q_batched)) + q_batched
            q = q_batched

        return x_tok, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        x = self.norm(x)
        return x, q


class _BottleneckEoMTWrapper(nn.Module):
    """Wraps a timm Bottleneck and exposes norm1/norm2/mlp for EoMT query updates."""

    def __init__(self, bottleneck: nn.Module, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.bottleneck = bottleneck
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x):
        return self.bottleneck(x)


def load_resnet_ckpt(model: ResNet, ckpt_path: str) -> ResNet:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    state_dict = {k[9:] if k.startswith("backbone.") else k: v for k, v in state_dict.items()}
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("ResNet \u2014 Missing keys :", missing)
    print("ResNet \u2014 Unexpected keys:", unexpected)
    return model
