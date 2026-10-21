from typing import Optional

import timm
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import models.backbones.nystrom  # noqa: F401
from models.vit import resize_pos_embed
from models.backbones.utils import vit_sizes


nystrom_sizes = {
    "Ti_64": {"backbone_name": "nystrom64_vit_tiny_patch16",  "vit_size": "Ti"},
    "Ti_32": {"backbone_name": "nystrom32_vit_tiny_patch16",  "vit_size": "Ti"},
    "S_64":  {"backbone_name": "nystrom64_vit_small_patch16", "vit_size": "S"},
    "S_32":  {"backbone_name": "nystrom32_vit_small_patch16", "vit_size": "S"},
    "B_64":  {"backbone_name": "nystrom64_vit_base_patch16",  "vit_size": "B"},
    "B_32":  {"backbone_name": "nystrom32_vit_base_patch16",  "vit_size": "B"},
    "L_64":  {"backbone_name": "nystrom64_vit_large_patch16", "vit_size": "L"},
    "L_32":  {"backbone_name": "nystrom32_vit_large_patch16", "vit_size": "L"},
}


class nystrom(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name: str = "nystrom64_vit_base_patch16",
        nystrom_size: str = "B_64",
        multiplier: int = 1,
        ls_init: Optional[float] = 1e-6,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            init_values=ls_init,
            num_classes=0,
            no_embed_class=True
        )

        if ckpt_path is not None:
            self.backbone = load_nystrom_ckpt(self.backbone, ckpt_path)

        sizes = vit_sizes[nystrom_sizes[nystrom_size]["vit_size"]]

        self.embed_dim = self.backbone.embed_dim
        self.num_heads = [sizes["num_heads"]]
        self.depths = [sizes["depth"]]
        self.patch_size = self.backbone.patch_embed.patch_size
        self.grid_size = self.backbone.patch_embed.grid_size
        self.num_prefix_tokens = self.backbone.num_prefix_tokens
        self.attn_multiplier = multiplier

        self.blocks = self.backbone.blocks
        self.norm = self.backbone.norm

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std  = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        if self.backbone.patch_drop is not None:
            x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        return x

    def _ls1(self, block, x):
        if hasattr(block, "ls1") and block.ls1 is not None:
            return block.ls1(x)
        return x

    def _ls2(self, block, x):
        if hasattr(block, "ls2") and block.ls2 is not None:
            return block.ls2(x)
        return x

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i: int):
        B = x.shape[0]

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

            x = x + block.drop_path1(self._ls1(block, block.attn(block.norm1(x))))
            x = x + block.drop_path2(self._ls2(block, block.mlp(block.norm2(x))))
            # Query self-attention (Mask2Former order: cross-attn, self-attn, MLP).
            q_batched = eomt_obj.apply_query_self_attn(q_batched, eomt_idx)
            q_batched = q_batched + block.drop_path2(self._ls2(block, block.mlp(block.norm2(q_batched))))
            q = q_batched

        else:
            x = block(x)

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        x = self.norm(x)
        return x, q


def load_nystrom_ckpt(model, ckpt_path: str):
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

    # Strip backbone. prefix and head. keys, then resize pos_embed inline
    # before anything touches the state_dict — matching the vit.py pattern.
    state_dict = {
        k[9:] if k.startswith("backbone.") else k: v
        for k, v in state_dict.items()
    }
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
    state_dict["pos_embed"] = resize_pos_embed(
        state_dict["pos_embed"], model.pos_embed
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("NystromViT — Missing keys :", missing)
    print("NystromViT — Unexpected keys:", unexpected)
    return model
