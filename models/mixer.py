from typing import Optional

import timm
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import models.backbones.mixer  # noqa: F401


mixer_sizes = {
    "S16": {"backbone_name": "mixer_s16", "patch_size": 16, "num_blocks": 8,  "embed_dim": 512},
    "S32": {"backbone_name": "mixer_s32", "patch_size": 32, "num_blocks": 8,  "embed_dim": 512},
    "B16": {"backbone_name": "mixer_b16", "patch_size": 16, "num_blocks": 12, "embed_dim": 768},
    "B32": {"backbone_name": "mixer_b32", "patch_size": 32, "num_blocks": 12, "embed_dim": 768},
    "L16": {"backbone_name": "mixer_l16", "patch_size": 16, "num_blocks": 24, "embed_dim": 1024},
    "L32": {"backbone_name": "mixer_l32", "patch_size": 32, "num_blocks": 24, "embed_dim": 1024},
}


class mixer(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name: str = "mixer_b16",
        mixer_size: str = "B16",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        sizes = mixer_sizes[mixer_size]

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size[0],
            num_classes=0,
        )

        if ckpt_path is not None:
            self.backbone = load_mixer_ckpt(self.backbone, ckpt_path)

        self.embed_dim = sizes["embed_dim"]
        # MLP-Mixer has no attention heads; EoMT cross-attn uses embed_dim with 1 head
        self.num_heads = [1]
        self.depths = [sizes["num_blocks"]]
        self.patch_size = (sizes["patch_size"], sizes["patch_size"])
        self.num_prefix_tokens = 0
        self.attn_multiplier = multiplier

        H, W = img_size
        p = sizes["patch_size"]
        self.grid_size = (H // p, W // p)

        self.blocks = list(self.backbone.blocks)
        self.norm = self.backbone.norm

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std  = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        return self.backbone.stem(x)

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i: int):
        B = x.shape[0]

        if i >= len(self.blocks) - eomt_obj.num_blocks:
            eomt_idx = i - (len(self.blocks) - eomt_obj.num_blocks)

            if q.dim() == 2:
                q_batched = q[None].expand(B, -1, -1)
            else:
                q_batched = q

            # MixerBlock has no norm1/attn; use norm2 for pre-attn normalisation
            xq = torch.cat([q_batched, x], dim=1)
            pre_attn = block.norm2(xq)
            q_tok = pre_attn[:, : eomt_obj.num_q, :]
            x_tok = pre_attn[:, eomt_obj.num_q :, :]

            attn_mask = eomt_obj.predict_and_mask(x_tok, q_tok, eomt_idx)

            after_eomt = torch.cat([q_tok, x_tok], dim=1)
            new_x = eomt_obj.attn[eomt_idx](self.norm(after_eomt), num_queries=eomt_obj.num_q, attn_mask=attn_mask)
            xq = xq + eomt_obj.dp(eomt_obj.ls_list[eomt_idx](new_x))

            q_batched = xq[:, : eomt_obj.num_q, :]
            x        = xq[:, eomt_obj.num_q :, :]

            # Token-mixing MLP operates on the full sequence length — run on image tokens only
            x = x + block.drop_path(block.mlp_tokens(block.norm1(x).transpose(1, 2)).transpose(1, 2))
            x = x + block.drop_path(block.mlp_channels(block.norm2(x)))

            # Mirror channel MLP for query tokens (no token-mixing across queries)
            # Query self-attention (Mask2Former order: cross-attn, self-attn, MLP).
            q_batched = eomt_obj.apply_query_self_attn(q_batched, eomt_idx)
            q_batched = q_batched + block.drop_path(block.mlp_channels(block.norm2(q_batched)))
            q = q_batched

        else:
            x = block(x)

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        x = self.norm(x)
        return x, q


def load_mixer_ckpt(model, ckpt_path: str):
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
    print("Mixer — Missing keys :", missing)
    print("Mixer — Unexpected keys:", unexpected)
    return model
