# ---------------------------------------------------------------
# Linear-probe module for ImageNet classification.
#
# Trains a single nn.Linear head on top of a *frozen* backbone
# encoder.  The encoder is the same object used in EoMT segmentation,
# so you can directly compare ImageNet linear-probe accuracy with
# segmentation mIoU for the same backbone.
#
# Metric keys follow the same "metrics/" prefix as the segmentation
# training, so Wandb panels and JSONL files have a consistent structure.
# ---------------------------------------------------------------

import math
from typing import Optional

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric.utilities import rank_zero_info
from torch.optim import AdamW
from torchmetrics.classification import MulticlassAccuracy

bold_green = "\033[1;32m"
reset = "\033[0m"


# ---------------------------------------------------------------------------
# Plain encoder forward — no EoMT queries / cross-attention
# ---------------------------------------------------------------------------

class _PlainForwardSentinel:
    """Stand-in for eomt_obj that keeps every encoder.run_block() call on the
    plain backbone path.

    With ``num_blocks = 0`` the guard ``i >= len(blocks) - num_blocks``
    becomes ``i >= len(blocks)``, which is never true for valid indices, so
    the EoMT branch is skipped unconditionally.
    """
    num_blocks = 0


_SENTINEL = _PlainForwardSentinel()


def _run_encoder_plain(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run any EoMT-compatible encoder wrapper without query / cross-attention.

    Works for ViT, Swin, ResNetSeg, and every other wrapper that exposes the
    ``pre_block / blocks / run_block / post_blocks`` interface used by LUMA.
    PatchMerging (Swin) and the ResNet spatial↔token conversion are handled
    automatically through the existing ``run_block`` implementations.

    Args:
        encoder: Any project encoder wrapper with the EoMT interface.
        x:       Normalised image tensor [B, 3, H, W].

    Returns:
        Spatial feature map [B, embed_dim, H_grid, W_grid].
    """
    x = encoder.pre_block(x)
    q = None
    for i, block in enumerate(encoder.blocks):
        x, q = encoder.run_block(block, _SENTINEL, x, q, i)
    x, _ = encoder.post_blocks(x, q, _SENTINEL)

    # x: [B, N_all, C] — strip any prefix tokens (e.g. CLS)
    patch_tokens = x[:, encoder.num_prefix_tokens:]   # [B, N_patch, C]
    B, N, C = patch_tokens.shape
    h, w = encoder.grid_size
    assert h * w == N, (
        f"encoder.grid_size {encoder.grid_size} gives {h * w} tokens but "
        f"got {N}. Check that img_size is linked correctly from the data module."
    )
    return patch_tokens.reshape(B, h, w, C).permute(0, 3, 1, 2)  # [B, C, h, w]


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class LinearProbeImageNet(lightning.LightningModule):
    """Frozen-backbone linear probe on ImageNet.

    A single ``nn.Linear(embed_dim, num_classes)`` head is trained on top of
    globally-averaged backbone features.  The backbone is frozen; only the
    linear head is optimised.

    The monitor metric is ``metrics/val_acc_top1``.  Top-5 accuracy is also
    logged as ``metrics/val_acc_top5``.

    Usage — pass the *segmentation* config to ``main_linear_probe.py`` with
    ``-c``; the script automatically extracts the encoder section and injects
    it here.  All other hyperparameters (lr, epochs, data path) can be
    overridden on the CLI::

        python main_linear_probe.py fit \\
            -c configs/ade20k/semantic/vit_base.yaml \\
            --data.init_args.path /ds/ImageNet \\
            --model.lr 1e-3 \\
            --trainer.max_epochs 90
    """

    def __init__(
        self,
        encoder_class: str,
        encoder_init: dict,
        num_classes: int = 1000,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        warmup_steps: int = 500,
        ckpt_path: Optional[str] = None,
    ):
        """
        Args:
            encoder_class: Dotted import path of the encoder class, e.g.
                           ``"models.vit.ViT"``.  Injected automatically by
                           ``main_linear_probe.py`` from the segmentation config.
            encoder_init:  Keyword arguments forwarded to the encoder constructor.
                           Injected automatically; ``img_size`` is overridden to
                           the LP training resolution (224×224 by default).
            num_classes:   Number of output classes.  Linked automatically from
                           ``data.init_args.num_classes`` (1000 for ImageNet).
            lr:            Peak LR for the linear head (AdamW).
            weight_decay:  Weight decay (default 0 — standard for linear probe).
            warmup_steps:  Linear LR warmup steps before cosine decay.
            ckpt_path:     Optional path to a **segmentation** Lightning
                           checkpoint.  When given, the encoder is re-initialised
                           from that checkpoint.  Handles both plain and
                           ``torch.compile``-wrapped state-dict keys (strips
                           ``_orig_mod.network.encoder.`` and
                           ``network.encoder.`` prefixes automatically).
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder = self._build_encoder(encoder_class, encoder_init)
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        # Linear classification head.
        self.head = nn.Linear(self.encoder.embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

        # Freeze encoder completely.
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        if ckpt_path is not None:
            self._load_encoder_from_seg_ckpt(ckpt_path)

        self.log = torch.compiler.disable(self.log)  # type: ignore

        # Metrics — same "metrics/" namespace as segmentation training.
        self.acc_top1 = MulticlassAccuracy(num_classes=num_classes, top_k=1)
        self.acc_top5 = MulticlassAccuracy(num_classes=num_classes, top_k=5)

    # ------------------------------------------------------------------
    # Encoder construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_encoder(class_path: str, init_args: dict) -> nn.Module:
        """Instantiate the encoder from a dotted class path + kwargs dict.

        This mirrors how jsonargparse resolves ``class_path + init_args`` but
        done explicitly so that the Lightning CLI never needs to perform
        subclass dispatch on an ``nn.Module``-typed parameter (which breaks
        with older jsonargparse versions in the container).
        """
        import importlib
        module_path, cls_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), cls_name)
        return cls(**init_args)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _load_encoder_from_seg_ckpt(self, ckpt_path: str) -> None:
        """Re-initialise encoder from a full segmentation Lightning checkpoint.

        Strips ``_orig_mod.network.encoder.`` and ``network.encoder.``
        prefixes so both plain and torch.compile checkpoints work.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("state_dict", ckpt)

        prefixes = ("_orig_mod.network.encoder.", "network.encoder.")
        encoder_sd: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            for pfx in prefixes:
                if k.startswith(pfx):
                    encoder_sd[k[len(pfx):]] = v
                    break

        if not encoder_sd:
            sample = list(state_dict.keys())[:10]
            raise ValueError(
                f"No encoder keys found in '{ckpt_path}'.\n"
                f"Expected keys prefixed with 'network.encoder.' or "
                f"'_orig_mod.network.encoder.'.\nFirst 10 keys: {sample}"
            )

        missing, unexpected = self.encoder.load_state_dict(encoder_sd, strict=False)
        if missing:
            raise ValueError(
                f"Missing encoder keys from '{ckpt_path}': {missing}"
            )
        rank_zero_info(
            f"LinearProbeImageNet: loaded encoder from {ckpt_path} "
            f"({len(encoder_sd)} tensors, {len(unexpected)} unexpected ignored)"
        )

    # ------------------------------------------------------------------
    # Always keep encoder in eval mode
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        super().train(mode)
        # Encoder is frozen — keep BN running stats and Dropout in eval state.
        self.encoder.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            imgs: uint8 tensor [B, 3, H, W] in [0, 255].

        Returns:
            Class logits [B, num_classes].
        """
        x = imgs.float() / 255.0
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        # No grad through the frozen encoder saves activation memory.
        with torch.no_grad():
            spatial = _run_encoder_plain(self.encoder, x)  # [B, C, h, w]

        # Global average pool over the spatial grid → [B, C]
        pooled = spatial.mean(dim=(-2, -1))
        return self.head(pooled)  # [B, num_classes]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        imgs, labels = batch  # [B, 3, H, W], [B]
        logits = self(imgs)
        loss = F.cross_entropy(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx=0):
        imgs, labels = batch
        logits = self(imgs)
        self.acc_top1.update(logits, labels)
        self.acc_top5.update(logits, labels)

    def on_validation_epoch_end(self):
        top1 = self.acc_top1.compute()
        top5 = self.acc_top5.compute()
        self.acc_top1.reset()
        self.acc_top5.reset()
        self.log("metrics/val_acc_top1", top1)
        self.log("metrics/val_acc_top5", top5)

    def on_validation_end(self):
        if not self.trainer.sanity_checking:
            top1 = self.trainer.callback_metrics.get("metrics/val_acc_top1", 0.0)
            top5 = self.trainer.callback_metrics.get("metrics/val_acc_top5", 0.0)
            rank_zero_info(
                f"{bold_green}Linear-probe  "
                f"Top-1: {float(top1) * 100:.2f}%  "
                f"Top-5: {float(top5) * 100:.2f}%"
                f"{reset}"
            )

    # ------------------------------------------------------------------
    # Optimiser + scheduler (head only)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = AdamW(
            self.head.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup = self.warmup_steps

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return step / warmup
            adj = step - warmup
            decay_steps = max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * adj / decay_steps))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Checkpoint saving
    # ------------------------------------------------------------------

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v
            for k, v in checkpoint["state_dict"].items()
        }
