from typing import Optional

import timm
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import models.backbones.poly_sa  # noqa: F401
import torch.nn.functional as F
from models.backbones.utils import vit_sizes
from models.vit import resize_pos_embed


polysa_sizes = {
    "Ti": {"backbone_name": "polysa_vit_tiny_patch16",  "vit_size": "Ti"},
    "S":  {"backbone_name": "polysa_vit_small_patch16", "vit_size": "S"},
    "B":  {"backbone_name": "polysa_vit_base_patch16",  "vit_size": "B"},
    "L":  {"backbone_name": "polysa_vit_large_patch16", "vit_size": "L"},
}


class poly_sa(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name: str = "polysa_vit_base_patch16",
        polysa_size: str = "B",
        multiplier: int = 1,
        ls_init: Optional[float] = 1e-6,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size[0],
            init_values=ls_init,
            no_embed_class=True,
            num_classes=0,
        )

        if ckpt_path is not None:
            self.backbone = load_polysa_ckpt(self.backbone, ckpt_path)

        sizes = vit_sizes[polysa_sizes[polysa_size]["vit_size"]]

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

            # PolySelfAttention.p1/p2 are sized (num_heads, seq_len) where seq_len
            # is the number of image tokens. Call attn on image tokens only to avoid
            # a mismatch with the einsum over p1/p2.
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

def resize_poly_param(param, num_extra_tokens, new_seq_len):
    """Bicubic-interpolate a PolySA p1/p2 parameter from its checkpoint
    seq_len to new_seq_len.  param: [H, L_old]"""
    H, L_old = param.shape
    if L_old == new_seq_len:
        return param
    extra = param[:, :num_extra_tokens]          # [H, num_extra]
    grid  = param[:, num_extra_tokens:]           # [H, gs_old^2]
    gs_old = int(grid.shape[1] ** 0.5)
    gs_new = int((new_seq_len - num_extra_tokens) ** 0.5)
    grid = F.interpolate(
        grid.reshape(1, H, gs_old, gs_old),
        size=(gs_new, gs_new),
        mode="bicubic",
        align_corners=False,
    ).reshape(H, gs_new * gs_new)
    return torch.cat([extra, grid], dim=-1)       # [H, new_seq_len]

def load_polysa_ckpt(model, ckpt_path: str):
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
    # before load_state_dict — matching the vit.py pattern.
    state_dict = {
        k[9:] if k.startswith("backbone.") else k: v
        for k, v in state_dict.items()
    }
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
    state_dict["pos_embed"] = resize_pos_embed(
        state_dict["pos_embed"], model.pos_embed
    )
# 2. p1 / p2 per block — bicubic interpolation matching set_image_res logic
    num_extra = model.num_prefix_tokens
    new_seq_len = model.seq_len
    for i in range(len(model.blocks)):
        for param_name in ("p1", "p2"):
            key = f"blocks.{i}.attn.{param_name}"
            if key in state_dict:
                state_dict[key] = resize_poly_param(
                    state_dict[key], num_extra, new_seq_len
                )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("PolySAViT — Missing keys :", missing)
    print("PolySAViT — Unexpected keys:", unexpected)
    return model