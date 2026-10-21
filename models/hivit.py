from typing import Optional

import timm
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import models.backbones.hivit  # noqa: F401
from models.backbones.hivit import HiViTBlock


class hivit(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name: str = "hivit_base_patch16",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        assert img_size[0] == img_size[1] == 224, (
            "HiViT does not support resizing beyond 224x224 (see backbone _set_input_strand)."
        )

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size[0],
            num_classes=0,
        )

        if ckpt_path is not None:
            self.backbone = load_hivit_ckpt(self.backbone, ckpt_path)

        self.num_prefix_tokens = 0
        self.attn_multiplier = multiplier
        self.embed_dim = self.backbone.num_features
        self.patch_size = (16, 16)
        self.grid_size = (
            img_size[0] // 16,
            img_size[1] // 16,
        )

        # HiViT has 3 stages: two hierarchical stem stages + one flat main stage.
        # Main-stage blocks have attention (with_attn=True).
        self._main_blocks = [b for b in self.backbone.blocks if isinstance(b, HiViTBlock) and b.with_attn]
        self._all_backbone_blocks = list(self.backbone.blocks)

        self.depths = [len(self._main_blocks)]
        self.num_heads = [self._main_blocks[0].attn.num_heads]

        # EoMT iterates self.blocks — expose only main-stage blocks
        self.blocks = self._main_blocks
        self.norm = self.backbone.fc_norm

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std  = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        # Run stem stages (hierarchical local blocks + PatchMerge) to produce
        # flat (B, num_patches, embed_dim) tokens ready for the main stage.
        x = self.backbone.patch_embed(x)

        for blk in self._all_backbone_blocks:
            if isinstance(blk, HiViTBlock) and blk.with_attn:
                break
            x = blk(x)

        # After last PatchMerge x is (B, num_patches, inner, inner, C); collapse inner patches
        x = x[..., 0, 0, :]   # (B, num_patches, embed_dim)

        if self.backbone.ape:
            x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        return x

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i: int):
        B = x.shape[0]
        num_main = len(self._main_blocks)

        rpe_index = None
        if self.backbone.rpe:
            rpe_index = self.backbone.relative_position_index.view(-1)

        if i >= num_main - eomt_obj.num_blocks:
            eomt_idx = i - (num_main - eomt_obj.num_blocks)

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
            x       = xq[:, eomt_obj.num_q :, :]

            x = x + block.drop_path(block.attn(block.norm1(x), rpe_index))
            x = x + block.drop_path(block.mlp(block.norm2(x)))
            # Query self-attention (Mask2Former order: cross-attn, self-attn, MLP).
            q_batched = eomt_obj.apply_query_self_attn(q_batched, eomt_idx)
            q_batched = q_batched + block.drop_path(block.mlp(block.norm2(q_batched)))
            q = q_batched

        else:
            x = block(x, rpe_index)

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        x = self.norm(x)
        return x, q


def load_hivit_ckpt(model, ckpt_path: str):
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
    print("HiViT — Missing keys :", missing)
    print("HiViT — Unexpected keys:", unexpected)
    return model
