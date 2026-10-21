"""Checkpoint loading through ``--model.ckpt_path``.

  * default                          -- everything is restored (evaluation).
  * `load_ckpt_class_head: false`    -- everything transfers except the final
                                        class `nn.Linear` (fine-tuning on a
                                        dataset with other classes).
  * `load_ckpt_backbone_only: true`  -- only `network.encoder.*` transfers.
"""

import tempfile
from pathlib import Path

import torch

from models.luma import LUMA
from models.vit import ViT
from training.mask_classification_panoptic import MaskClassificationPanoptic

IMG_SIZE = (64, 64)
COCO_CLASSES, ADE_CLASSES = 133, 150


def _module(num_classes: int, seed: int, **load_kwargs) -> MaskClassificationPanoptic:
    torch.manual_seed(seed)
    # Small random-init ViT -- no download, no pretrained weights.
    encoder = ViT(
        img_size=IMG_SIZE,
        backbone_name="vit_small_patch16_384",
        patch_size=16,
        pretrained=False,
        vit_size="S",
    )
    network = LUMA(
        encoder=encoder,
        num_classes=num_classes,
        num_q=4,
        num_blocks=2,
        self_a=False,
        query_only_residual=True,
        thread_queries=True,
        query_self_attn=True,
        masked_cross_attn=True,
    )
    return MaskClassificationPanoptic(
        network=network,
        img_size=IMG_SIZE,
        num_classes=num_classes,
        stuff_classes=[0, 1, 2],
        attn_mask_annealing_enabled=False,
        **load_kwargs,
    )


def _same(a: dict, b: dict, key: str) -> bool:
    return key in a and key in b and torch.equal(a[key], b[key])


def test_ckpt_transfer() -> None:
    src = _module(COCO_CLASSES, seed=0)
    src_sd = src.state_dict()

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "coco.ckpt"
        torch.save(src_sd, ckpt)

        # --- variant 1: backbone only -------------------------------------
        bb = _module(
            ADE_CLASSES, seed=1, ckpt_path=str(ckpt), load_ckpt_backbone_only=True
        ).state_dict()
        enc = [k for k in src_sd if k.startswith("network.encoder.")]
        assert enc, "no encoder keys -- test is vacuous"
        assert all(_same(src_sd, bb, k) for k in enc), "encoder did not transfer"
        for k in ("network.q", "network.mask_head.0.weight"):
            assert not _same(src_sd, bb, k), f"{k} transferred but should be fresh"
        assert bb["network.class_head.weight"].shape[0] == ADE_CLASSES + 1

        # --- variant 2: everything but the final class Linear ---------------
        full = _module(
            ADE_CLASSES, seed=1, ckpt_path=str(ckpt), load_ckpt_class_head=False
        ).state_dict()
        transferable = [k for k in src_sd if "class_head" not in k and "empty_weight" not in k]
        assert all(_same(src_sd, full, k) for k in transferable), (
            "non-class-head weights did not transfer"
        )
        assert full["network.class_head.weight"].shape[0] == ADE_CLASSES + 1

        # --- default: same class count, load the lot ------------------------
        same = _module(COCO_CLASSES, seed=1, ckpt_path=str(ckpt)).state_dict()
        assert all(_same(src_sd, same, k) for k in src_sd if "empty_weight" not in k), (
            "default load is no longer a full restore"
        )
