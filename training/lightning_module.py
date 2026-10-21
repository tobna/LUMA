# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from:
# - the torchmetrics library by the PyTorch Lightning team
# - the Mask2Former repository by Facebook, Inc. and its affiliates
# All used under the Apache 2.0 License.
# ---------------------------------------------------------------

import math
from typing import List, Optional, cast
import lightning
from lightning.fabric.utilities import rank_zero_info
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.detection import PanopticQuality, MeanAveragePrecision
from torchmetrics.functional.detection._panoptic_quality_common import (
    _prepocess_inputs,
    _Color,
    _get_color_areas,
    _calculate_iou,
)
import wandb
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import io
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import interpolate
from torchvision.transforms.v2.functional import pad

from training.two_stage_warmup_cosine_schedule import TwoStageWarmupCosineSchedule

bold_green = "\033[1;32m"
reset = "\033[0m"


class LightningModule(lightning.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]],
        attn_mask_annealing_end_steps: Optional[list[int]],
        lr: float,
        llrd: float,
        weight_decay: float,
        poly_power: float,
        warmup_steps: tuple[int, int],
        backbone_freeze_steps: int = 0,
        backbone_freeze_lr_scale: float = 0.0,
        warmup_epochs: Optional[List[int]] = None,
        backbone_freeze_epochs: Optional[int] = None,
        frozen_layers: Optional[List[str]] = None,
        ckpt_path=None,
        load_ckpt_class_head=True,
        load_ckpt_backbone_only: bool = False,
        no_decay_on_norm_bias: bool = False,
    ):
        """
        Args:
            load_ckpt_backbone_only: Take only ``network.encoder.*`` from
                ``ckpt_path`` and randomly initialise everything else (queries,
                cross-attention, LayerScale, mask/class heads). This is the
                "backbone transfer" arm of a COCO -> ADE20K finetune; the
                "full transfer" arm is ``load_ckpt_class_head: false``, which
                keeps every weight except the final class ``nn.Linear``
                (whose output width is ``num_classes + 1`` and so cannot
                transfer between datasets).

            no_decay_on_norm_bias: Exclude 1-D parameters (biases, LayerNorm /
                LayerScale weights) and the positional / cls / register /
                distillation tokens from weight decay, which is what every
                standard ViT recipe (timm, DeiT, MAE, Mask2Former) does.

                Defaults to ``False``, which applies ``weight_decay`` uniformly
                to every parameter — the behaviour every run trained so far
                used. Turn it on only for a fresh sweep; it is not a bug fix you
                can retrofit onto an in-flight comparison, since it changes the
                optimiser for the baseline arm too.
        """
        super().__init__()

        self.network = network
        self.img_size = img_size
        self.num_classes = num_classes
        self.attn_mask_annealing_enabled = attn_mask_annealing_enabled
        self.attn_mask_annealing_start_steps = attn_mask_annealing_start_steps
        self.attn_mask_annealing_end_steps = attn_mask_annealing_end_steps
        self.lr = lr
        self.llrd = llrd
        self.weight_decay = weight_decay
        self.poly_power = poly_power
        self.warmup_steps = warmup_steps
        self.backbone_freeze_steps = backbone_freeze_steps
        self.backbone_freeze_lr_scale = backbone_freeze_lr_scale
        self.warmup_epochs = warmup_epochs
        self.backbone_freeze_epochs = backbone_freeze_epochs

        self.strict_loading = False
        self._frozen_layers = frozen_layers or []
        self.no_decay_on_norm_bias = no_decay_on_norm_bias

        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)

            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]

            # `torch.compile` prefixes every key with `_orig_mod.`; a ckpt from a
            # run without `compile_disabled` is otherwise unloadable here.
            ckpt = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}

            def wanted(key: str) -> bool:
                """Should `key` come from the checkpoint (rather than fresh init)?"""
                if "criterion.empty_weight" in key:
                    return False
                if not load_ckpt_class_head and (
                    "class_head" in key or "class_predictor" in key
                ):
                    return False
                if load_ckpt_backbone_only and not key.startswith("network.encoder."):
                    return False
                return True

            ckpt = {k: v for k, v in ckpt.items() if wanted(k)}

            incompatible_keys = self.load_state_dict(ckpt, strict=False)

            missing_keys = [k for k in incompatible_keys.missing_keys if wanted(k)]
            if missing_keys:
                raise ValueError(f"Missing keys: {missing_keys}")

            if incompatible_keys.unexpected_keys:
                raise ValueError(
                    f"Unexpected keys: {incompatible_keys.unexpected_keys}"
                )

        self.log = torch.compiler.disable(self.log)  # type: ignore

        if self._frozen_layers:
            self._apply_frozen_layers(self._frozen_layers)

    def _apply_frozen_layers(self, patterns: List[str]) -> None:
        """Freeze all parameters whose fully-qualified name matches any pattern.

        A parameter matches when its name equals a pattern *or* starts with
        ``"<pattern>."`` — i.e. prefix-based subtree matching.  Example::

            frozen_layers:
              - "network.encoder.patch_embed"   # entire patch-embedding subtree
              - "network.encoder.blocks.0"      # first transformer block only
        """
        frozen_tensors, frozen_params = 0, 0
        for name, param in self.named_parameters():
            if any(
                name == pat or name.startswith(pat + ".")
                for pat in patterns
            ):
                param.requires_grad_(False)
                frozen_tensors += 1
                frozen_params += param.numel()
        if frozen_tensors:
            rank_zero_info(
                f"frozen_layers: {frozen_tensors} tensor(s) frozen "
                f"({frozen_params:,} params) matching {patterns}"
            )

    def _log_llrd_summary(
        self,
        *,
        backbone_blocks: int,
        block_param_counts: List[int],
        pretrained_nonblock_count: int,
        fresh_nonblock_count: int,
        new_module_count: int,
        sample_pretrained_nonblock: Optional[str],
        sample_fresh_nonblock: Optional[str],
        sample_new_module: Optional[str],
    ) -> None:
        """One-shot LLRD bucket summary, printed at fit-start (rank-zero).

        Lets you eyeball — on every run, before training starts — exactly
        which LR each part of the model is getting. Catches regressions
        from backbone refactors / wrapper edits / config typos before
        they cost you a sweep.
        """
        lines = [
            f"LLRD param-group summary "
            f"(backbone_blocks={backbone_blocks}, llrd={self.llrd}, "
            f"base_lr={self.lr:.2e})"
        ]

        # Per-block lines, condensed: contiguous runs of equal-count blocks
        # collapse into one row. For a flat ViT the counts are uniform across
        # all N blocks; printing 24 identical rows is noise.
        i = 0
        while i < backbone_blocks:
            j = i
            while (
                j + 1 < backbone_blocks
                and block_param_counts[j + 1] == block_param_counts[i]
            ):
                j += 1
            mult_lo = self.llrd ** (backbone_blocks - 1 - j)  # deepest in run
            mult_hi = self.llrd ** (backbone_blocks - 1 - i)  # shallowest in run
            label = (
                f"encoder.blocks[{i}]"
                if i == j
                else f"encoder.blocks[{i}..{j}]"
            )
            if i == j:
                lines.append(
                    f"  {label:<32s}  ({block_param_counts[i]:>11,d} params)  "
                    f"lr={self.lr * mult_lo:.2e}  (×{mult_lo:.4f})"
                )
            else:
                lines.append(
                    f"  {label:<32s}  ({block_param_counts[i] * (j - i + 1):>11,d} params, "
                    f"{block_param_counts[i]:,} each)  "
                    f"lr={self.lr * mult_hi:.2e}..{self.lr * mult_lo:.2e}  "
                    f"(×{mult_hi:.4f}..{mult_lo:.4f})"
                )
            i = j + 1

        if pretrained_nonblock_count:
            mult = self.llrd ** (backbone_blocks - 1) if backbone_blocks > 0 else 1.0
            lines.append(
                f"  {'pretrained non-block':<32s}  ({pretrained_nonblock_count:>11,d} params)  "
                f"lr={self.lr * mult:.2e}  (×{mult:.4f})  e.g. {sample_pretrained_nonblock}"
            )
        if fresh_nonblock_count:
            lines.append(
                f"  {'fresh wrapper non-block':<32s}  ({fresh_nonblock_count:>11,d} params)  "
                f"lr={self.lr:.2e}  (×1.0000)  e.g. {sample_fresh_nonblock}"
            )
        if new_module_count:
            lines.append(
                f"  {'EoMT / new modules':<32s}  ({new_module_count:>11,d} params)  "
                f"lr={self.lr:.2e}  (×1.0000)  e.g. {sample_new_module}"
            )

        rank_zero_info("\n".join(lines))

    def configure_optimizers(self):
        encoder = self.network.encoder
        backbone_blocks = len(encoder.blocks)

        # --------------------------------------------------------------
        # LLRD parameter-group assignment
        # --------------------------------------------------------------
        # Each parameter falls into exactly one bucket:
        #
        #   (1) encoder.blocks[i].*               LR = base_lr * llrd^(N-1-i)
        #       The wrapper's flat `self.blocks` list is the canonical
        #       global ordering — used by the EoMT loop and by
        #       `backbone_blocks` above. We look the index up by parameter
        #       `id()` rather than by parsing the param name, which makes
        #       this robust to every naming convention in the repo:
        #       flat ViT (`backbone.blocks.<i>`), hierarchical multi-stage
        #       (`backbone.layers.<L>.blocks.<bi>`), and single-stage
        #       transformer-on-CNN (`backbone.stage4.<i>` / `block4.<i>` /
        #       `layer4.<i>`). The old `name.split('.')`-based parser
        #       gave per-stage indices for the hierarchical cases (so
        #       LLRD was inverted) and matched nothing for the
        #       single-stage cases (so LLRD silently did not apply).
        #
        #   (2) pretrained encoder params NOT in encoder.blocks      LR = base_lr * llrd^(N-1)
        #       (patch_embed, pos_embed, cls_token, aliased final norm,
        #       earlier CNN stages for ResT/Wave/ResNet, …). Treated as
        #       block index 0 → maximum LLRD discount. Matches the
        #       behaviour the old parser accidentally produced for
        #       patch_embed / pos_embed / cls_token via the stale-state
        #       `block_i == 0` clause, so config llrd values keep their
        #       calibration. Identified by the `.backbone.` substring in
        #       the param's fully-qualified name — every wrapper in the
        #       repo nests pretrained weights under `self.backbone`.
        #
        #   (3) fresh encoder params NOT in encoder.blocks           LR = base_lr
        #       (the new `eomt_pre_norms.*` and fresh `self.norm` we
        #       added to `efficient_net`; any future wrapper-owned
        #       random-init modules that live at the top level of the
        #       encoder wrapper). Full LR — they have nothing pretrained
        #       to protect.
        #
        #   (4) everything outside encoder                           LR = base_lr
        #       EoMT heads, queries, attn, ls_list, etc.
        # --------------------------------------------------------------

        # (1)-builder: id-based lookup table {param_id: global_block_index}
        param_to_block_idx: dict[int, int] = {}
        for global_idx, blk in enumerate(encoder.blocks):
            for _, p in blk.named_parameters(recurse=True):
                param_to_block_idx[id(p)] = global_idx

        # set of every parameter id under the encoder (block or not)
        encoder_param_ids = {id(p) for _, p in encoder.named_parameters()}

        backbone_param_groups: list[dict] = []
        other_param_groups: list[dict] = []

        # bookkeeping for the one-shot LLRD summary log
        block_param_counts = [0] * backbone_blocks
        pretrained_nonblock_count = 0
        fresh_nonblock_count = 0
        new_module_count = 0
        sample_pretrained_nonblock: Optional[str] = None
        sample_fresh_nonblock: Optional[str] = None
        sample_new_module: Optional[str] = None

        max_discount = self.llrd ** (backbone_blocks - 1) if backbone_blocks > 0 else 1.0

        # Tokens that are conceptually embeddings, not weights — decayed only
        # when no_decay_on_norm_bias is off. 1-D params (biases, LayerNorm and
        # LayerScale gammas) are caught by the param.ndim check instead.
        no_decay_names = ("pos_embed", "cls_token", "reg_token", "dist_token")
        no_decay_count = 0

        def _weight_decay_for(name, param):
            nonlocal no_decay_count
            if not self.no_decay_on_norm_bias:
                return self.weight_decay
            if param.ndim <= 1 or any(n in name for n in no_decay_names):
                no_decay_count += param.numel()
                return 0.0
            return self.weight_decay

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # skip frozen parameters entirely

            pid = id(param)
            if pid in encoder_param_ids:
                if pid in param_to_block_idx:
                    bi = param_to_block_idx[pid]
                    lr = self.lr * (self.llrd ** (backbone_blocks - 1 - bi))
                    block_param_counts[bi] += param.numel()
                elif ".backbone." in name:
                    # bucket (2): pretrained encoder param outside blocks
                    lr = self.lr * max_discount
                    pretrained_nonblock_count += param.numel()
                    if sample_pretrained_nonblock is None:
                        sample_pretrained_nonblock = name
                else:
                    # bucket (3): fresh wrapper-owned encoder param
                    lr = self.lr
                    fresh_nonblock_count += param.numel()
                    if sample_fresh_nonblock is None:
                        sample_fresh_nonblock = name
                backbone_param_groups.append(
                    {"params": [param], "lr": lr, "name": name,
                     "weight_decay": _weight_decay_for(name, param)}
                )
            else:
                # bucket (4): EoMT-injected modules — heads, queries, attn, ls
                other_param_groups.append(
                    {"params": [param], "lr": self.lr, "name": name,
                     "weight_decay": _weight_decay_for(name, param)}
                )
                new_module_count += param.numel()
                if sample_new_module is None:
                    sample_new_module = name

        self._log_llrd_summary(
            backbone_blocks=backbone_blocks,
            block_param_counts=block_param_counts,
            pretrained_nonblock_count=pretrained_nonblock_count,
            fresh_nonblock_count=fresh_nonblock_count,
            new_module_count=new_module_count,
            sample_pretrained_nonblock=sample_pretrained_nonblock,
            sample_fresh_nonblock=sample_fresh_nonblock,
            sample_new_module=sample_new_module,
        )
        if self.no_decay_on_norm_bias:
            rank_zero_info(
                f"no_decay_on_norm_bias: weight_decay=0 on {no_decay_count:,} params "
                f"(1-D tensors + {'/'.join(no_decay_names)})"
            )

        param_groups = backbone_param_groups + other_param_groups
        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)

        # Resolve epoch-based warmup/freeze overrides to steps
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        resolved_warmup = (
            [e * steps_per_epoch for e in self.warmup_epochs]
            if self.warmup_epochs is not None
            else self.warmup_steps
        )
        resolved_freeze = (
            self.backbone_freeze_epochs * steps_per_epoch
            if self.backbone_freeze_epochs is not None
            else self.backbone_freeze_steps
        )

        # Build indices for the 3 representative LR curves logged per epoch:
        #   lr/backbone_max  — param group with the highest backbone LR (least LLRD discount)
        #   lr/backbone_min  — param group with the lowest backbone LR (most LLRD discount)
        #   lr/new_modules   — first non-backbone param group
        # Uses LR values directly so it works for any backbone architecture.
        self._lr_log_indices: dict[str, int] = {}
        if backbone_param_groups:
            backbone_lrs = [pg["lr"] for pg in backbone_param_groups]
            self._lr_log_indices["lr/backbone_max"] = backbone_lrs.index(max(backbone_lrs))
            self._lr_log_indices["lr/backbone_min"] = backbone_lrs.index(min(backbone_lrs))
        if other_param_groups:
            self._lr_log_indices["lr/new_modules"] = len(backbone_param_groups)

        scheduler = TwoStageWarmupCosineSchedule(
            optimizer,
            num_backbone_params=len(backbone_param_groups),
            warmup_steps=resolved_warmup,
            total_steps=self.trainer.estimated_stepping_batches,
            backbone_freeze_steps=resolved_freeze,
            backbone_freeze_lr_scale=self.backbone_freeze_lr_scale,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, imgs):
        x = imgs / 255.0

        return self.network(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch

        mask_logits_per_block, class_logits_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        total_loss = self.criterion.loss_total(losses_all_blocks, self.log)

        # Add Switch load-balancing auxiliary loss
        switch_aux = getattr(self.network, 'switch_aux_loss', None)
        if switch_aux is not None:
            self.log("train/switch_aux_loss", switch_aux, on_step=True, on_epoch=False)
            total_loss = total_loss + switch_aux

        return total_loss

    def validation_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "val")

    def mask_annealing(self, start_iter, current_iter, final_iter):
        device = self.device
        dtype = self.network.attn_mask_probs[0].dtype
        if current_iter < start_iter:
            return torch.ones(1, device=device, dtype=dtype)
        elif current_iter >= final_iter:
            return torch.zeros(1, device=device, dtype=dtype)
        else:
            progress = (current_iter - start_iter) / (final_iter - start_iter)
            progress = torch.tensor(progress, device=device, dtype=dtype)
            return (1.0 - progress).pow(self.poly_power)

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx=None,
        dataloader_idx=None,
    ):
        if self.attn_mask_annealing_enabled:
            for i in range(self.network.num_blocks):
                self.network.attn_mask_probs[i] = self.mask_annealing(
                    self.attn_mask_annealing_start_steps[i],
                    self.global_step,
                    self.attn_mask_annealing_end_steps[i],
                )

            for i, attn_mask_prob in enumerate(self.network.attn_mask_probs):
                self.log(
                    f"attn_mask_prob_{i}",
                    attn_mask_prob,
                    on_step=True,
                )

    def _log_lrs(self):
        optimizer = self.optimizers()
        for label, idx in getattr(self, "_lr_log_indices", {}).items():
            self.log(label, optimizer.param_groups[idx]["lr"])

    def on_train_start(self):
        self._log_lrs()

    def on_train_epoch_end(self):
        self._log_lrs()

    def init_metrics_semantic(self, ignore_idx, num_blocks):
        self.metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,
                )
                for _ in range(num_blocks)
            ]
        )

    def init_metrics_instance(self, num_blocks):
        self.metrics = nn.ModuleList(
            [MeanAveragePrecision(iou_type="segm") for _ in range(num_blocks)]
        )

    def init_metrics_panoptic(self, thing_classes, stuff_classes, num_blocks):
        self.metrics = nn.ModuleList(
            [
                PanopticQuality(
                    thing_classes,
                    stuff_classes + [self.num_classes],
                    return_sq_and_rq=True,
                    return_per_class=True,
                )
                for _ in range(num_blocks)
            ]
        )

    @torch.compiler.disable
    def update_metrics_semantic(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        block_idx,
    ):
        for i in range(len(preds)):
            self.metrics[block_idx].update(preds[i][None, ...], targets[i][None, ...])

    @torch.compiler.disable
    def update_metrics_instance(
        self,
        preds: list[dict],
        targets: list[dict],
        block_idx,
    ):
        self.metrics[block_idx].update(preds, targets)

    @torch.compiler.disable
    def update_metrics_panoptic(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        is_crowds: list[torch.Tensor],
        block_idx,
    ):
        for i in range(len(preds)):
            metric = self.metrics[block_idx]
            flatten_pred = _prepocess_inputs(
                metric.things,
                metric.stuffs,
                preds[i][None, ...],
                metric.void_color,
                metric.allow_unknown_preds_category,
            )[0]
            flatten_target = _prepocess_inputs(
                metric.things,
                metric.stuffs,
                targets[i][None, ...],
                metric.void_color,
                True,
            )[0]

            pred_areas = cast(
                dict[_Color, torch.Tensor], _get_color_areas(flatten_pred)
            )
            target_areas = cast(
                dict[_Color, torch.Tensor], _get_color_areas(flatten_target)
            )
            intersection_matrix = torch.transpose(
                torch.stack((flatten_pred, flatten_target), -1), -1, -2
            )
            intersection_areas = cast(
                dict[tuple[_Color, _Color], torch.Tensor],
                _get_color_areas(intersection_matrix),
            )

            pred_segment_matched = set()
            target_segment_matched = set()
            for pred_color, target_color in intersection_areas:
                if is_crowds[i][target_color[1]]:
                    continue
                if target_color == metric.void_color:
                    continue
                if pred_color[0] != target_color[0]:
                    continue
                iou = _calculate_iou(
                    pred_color,
                    target_color,
                    pred_areas,
                    target_areas,
                    intersection_areas,
                    metric.void_color,
                )
                continuous_id = metric.cat_id_to_continuous_id[target_color[0]]
                if iou > 0.5:
                    pred_segment_matched.add(pred_color)
                    target_segment_matched.add(target_color)
                    metric.iou_sum[continuous_id] += iou
                    metric.true_positives[continuous_id] += 1

            false_negative_colors = set(target_areas) - target_segment_matched
            false_positive_colors = set(pred_areas) - pred_segment_matched

            false_negative_colors.discard(metric.void_color)
            false_positive_colors.discard(metric.void_color)

            for target_color in list(false_negative_colors):
                void_target_area = intersection_areas.get(
                    (metric.void_color, target_color), 0
                )
                if void_target_area / target_areas[target_color] > 0.5:
                    false_negative_colors.discard(target_color)

            crowd_by_cat_id = {}
            for false_negative_color in false_negative_colors:
                if is_crowds[i][false_negative_color[1]]:
                    crowd_by_cat_id[false_negative_color[0]] = false_negative_color[1]
                    continue

                continuous_id = metric.cat_id_to_continuous_id[false_negative_color[0]]
                metric.false_negatives[continuous_id] += 1

            for pred_color in list(false_positive_colors):
                pred_void_crowd_area = intersection_areas.get(
                    (pred_color, metric.void_color), 0
                )

                if pred_color[0] in crowd_by_cat_id:
                    crowd_color = (pred_color[0], crowd_by_cat_id[pred_color[0]])
                    pred_void_crowd_area += intersection_areas.get(
                        (pred_color, crowd_color), 0
                    )

                if pred_void_crowd_area / pred_areas[pred_color] > 0.5:
                    false_positive_colors.discard(pred_color)

            for false_positive_color in false_positive_colors:
                continuous_id = metric.cat_id_to_continuous_id[false_positive_color[0]]
                metric.false_positives[continuous_id] += 1

    def block_postfix(self, block_idx):
        if not self.network.masked_attn_enabled:
            return ""
        return (
            f"_block_{-len(self.metrics) + block_idx + 1}"
            if block_idx != self.network.num_blocks
            else ""
        )

    def _on_eval_epoch_end_semantic(self, log_prefix, log_per_class=False):
        for i, metric in enumerate(self.metrics):  # type: ignore
            iou_per_class = metric.compute()
            metric.reset()

            block_postfix = self.block_postfix(i)
            if log_per_class:
                for class_idx, iou in enumerate(iou_per_class):
                    self.log(
                        f"metrics/{log_prefix}_iou_class_{class_idx}{block_postfix}",
                        iou,
                    )

            iou_all = float(iou_per_class.mean())
            self.log(
                f"metrics/{log_prefix}_iou_all{block_postfix}",
                iou_all,
            )

    def _on_eval_epoch_end_instance(self, log_prefix):
        for i, metric in enumerate(self.metrics):  # type: ignore
            results = metric.compute()
            metric.reset()

            block_postfix = self.block_postfix(i)
            self.log(
                f"metrics/{log_prefix}_ap_all{block_postfix}",
                results["map"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_small_all{block_postfix}",
                results["map_small"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_medium_all{block_postfix}",
                results["map_medium"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_large_all{block_postfix}",
                results["map_large"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_50_all{block_postfix}",
                results["map_50"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_75_all{block_postfix}",
                results["map_75"],
            )

    def _on_eval_epoch_end_panoptic(self, log_prefix, log_per_class=False):
        for i, metric in enumerate(self.metrics):  # type: ignore
            result = metric.compute()[:-1]
            metric.reset()

            pq, sq, rq = result[:, 0], result[:, 1], result[:, 2]

            block_postfix = self.block_postfix(i)
            if log_per_class:
                for class_idx in range(len(pq)):
                    self.log(
                        f"metrics/{log_prefix}_pq_class_{class_idx}{block_postfix}",
                        pq[class_idx],
                    )
                    self.log(
                        f"metrics/{log_prefix}_sq_class_{class_idx}{block_postfix}",
                        sq[class_idx],
                    )
                    self.log(
                        f"metrics/{log_prefix}_rq_class_{class_idx}{block_postfix}",
                        rq[class_idx],
                    )

            self.log(
                f"metrics/{log_prefix}_pq_all{block_postfix}",
                pq.mean(),
            )
            self.log(f"metrics/{log_prefix}_sq_all{block_postfix}", sq.mean())
            self.log(f"metrics/{log_prefix}_rq_all{block_postfix}", rq.mean())

            num_things = len(metric.things)
            pq_things, sq_things, rq_things = (
                result[:num_things, 0],
                result[:num_things, 1],
                result[:num_things, 2],
            )
            pq_stuff, sq_stuff, rq_stuff = (
                result[num_things:, 0],
                result[num_things:, 1],
                result[num_things:, 2],
            )

            self.log(
                f"metrics/{log_prefix}_pq_things{block_postfix}",
                pq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_sq_things{block_postfix}",
                sq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_rq_things{block_postfix}",
                rq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_pq_stuff{block_postfix}",
                pq_stuff.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_sq_stuff{block_postfix}",
                sq_stuff.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_rq_stuff{block_postfix}",
                rq_stuff.mean(),
            )

    def _on_eval_end_semantic(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}mIoU: {self.trainer.callback_metrics[f'metrics/{log_prefix}_iou_all'] * 100:.1f}{reset}"
            )

    def _on_eval_end_instance(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}mAP All: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_all'] * 100:.1f} | "
                f"mAP Small: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_small_all'] * 100:.1f} | "
                f"mAP Medium: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_medium_all'] * 100:.1f} | "
                f"mAP Large: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_large_all'] * 100:.1f}{reset}"
            )

    def _on_eval_end_panoptic(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}PQ All: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_all'] * 100:.1f} | "
                f"PQ Things: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_things'] * 100:.1f} | "
                f"PQ Stuff: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_stuff'] * 100:.1f}{reset}"
            )

    @torch.compiler.disable
    def plot_semantic(
        self,
        img,
        target,
        logits,
        log_prefix,
        block_idx,
        batch_idx,
        cmap="tab20",
    ):
        fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)

        axes[0].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axes[0].axis("off")

        target = target.cpu().numpy()
        unique_classes = np.unique(target)

        preds = torch.argmax(logits, dim=0).cpu().numpy()
        unique_classes = np.unique(np.concatenate((unique_classes, np.unique(preds))))

        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))  # type: ignore

        if self.ignore_idx in unique_classes:
            colors[unique_classes == self.ignore_idx] = [0, 0, 0, 1]  # type: ignore

        custom_cmap = mcolors.ListedColormap(colors)  # type: ignore
        norm = mcolors.Normalize(0, num_classes - 1)

        axes[1].imshow(
            np.digitize(target, unique_classes) - 1,
            cmap=custom_cmap,
            norm=norm,
            interpolation="nearest",
        )
        axes[1].axis("off")

        if preds is not None:
            axes[2].imshow(
                np.digitize(preds, unique_classes, right=True),
                cmap=custom_cmap,
                norm=norm,
                interpolation="nearest",
            )
            axes[2].axis("off")

        patches = [
            Line2D([0], [0], color=colors[i], lw=4, label=str(unique_classes[i]))
            for i in range(num_classes)
        ]

        fig.legend(handles=patches, loc="upper left")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="black")
        plt.close(fig)
        buf.seek(0)

        block_postfix = self.block_postfix(block_idx)
        name = f"{log_prefix}_pred_{batch_idx}{block_postfix}"
        self.trainer.logger.experiment.log({name: [wandb.Image(Image.open(buf))]})

    @torch.compiler.disable
    def scale_img_size_semantic(self, size: tuple[int, int]):
        factor = max(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def window_imgs_semantic(self, imgs):
        crops, origins = [], []

        for i in range(len(imgs)):
            img = imgs[i]
            new_h, new_w = self.scale_img_size_semantic(img.shape[-2:])
            pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
            resized_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).to(img.device)
            )

            num_crops = math.ceil(max(resized_img.shape[-2:]) / min(self.img_size))
            overlap = num_crops * min(self.img_size) - max(resized_img.shape[-2:])
            overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

            for j in range(num_crops):
                start = int(j * (min(self.img_size) - overlap_per_crop))
                end = start + min(self.img_size)
                if resized_img.shape[-2] > resized_img.shape[-1]:
                    crop = resized_img[:, start:end, :]
                else:
                    crop = resized_img[:, :, start:end]

                crops.append(crop)
                origins.append((i, start, end))

        return torch.stack(crops), origins

    def revert_window_logits_semantic(self, crop_logits, origins, img_sizes):
        logit_sums, logit_counts = [], []
        for size in img_sizes:
            h, w = self.scale_img_size_semantic(size)
            logit_sums.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )
            logit_counts.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )

        for crop_i, (img_i, start, end) in enumerate(origins):
            if img_sizes[img_i][0] > img_sizes[img_i][1]:
                logit_sums[img_i][:, start:end, :] += crop_logits[crop_i]
                logit_counts[img_i][:, start:end, :] += 1
            else:
                logit_sums[img_i][:, :, start:end] += crop_logits[crop_i]
                logit_counts[img_i][:, :, start:end] += 1

        return [
            interpolate(
                (sums / counts)[None, ...],
                img_sizes[i],
                mode="bilinear",
            )[0]
            for i, (sums, counts) in enumerate(zip(logit_sums, logit_counts))
        ]

    @staticmethod
    def to_per_pixel_logits_semantic(
        mask_logits: torch.Tensor, class_logits: torch.Tensor
    ):
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_semantic(
        targets: list[dict],
        ignore_idx,
    ):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = torch.full(
                target["masks"].shape[-2:],
                ignore_idx,
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[mask] = target["labels"][i]

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def scale_img_size_instance_panoptic(self, size: tuple[int, int]):
        factor = min(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def resize_and_pad_imgs_instance_panoptic(self, imgs):
        transformed_imgs = []

        for img in imgs:
            new_h, new_w = self.scale_img_size_instance_panoptic(img.shape[-2:])

            pil_img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).to(img.device)
            )

            pad_h = max(0, self.img_size[-2] - resized_img.shape[-2])
            pad_w = max(0, self.img_size[-1] - resized_img.shape[-1])
            padding = [0, 0, pad_w, pad_h]

            padded_img = pad(resized_img, padding)

            transformed_imgs.append(padded_img)

        return torch.stack(transformed_imgs)

    @torch.compiler.disable
    def revert_resize_and_pad_logits_instance_panoptic(
        self, transformed_logits, img_sizes
    ):
        logits = []
        for i in range(len(transformed_logits)):
            scaled_size = self.scale_img_size_instance_panoptic(img_sizes[i])
            logits_i = transformed_logits[i][:, : scaled_size[0], : scaled_size[1]]
            logits_i = interpolate(
                logits_i[None, ...],
                img_sizes[i],
                mode="bilinear",
            )[0]
            logits.append(logits_i)

        return logits

    def to_per_pixel_preds_panoptic(
        self, mask_logits_list, class_logits, stuff_classes, mask_thresh, overlap_thresh
    ):
        scores, classes = class_logits.softmax(dim=-1).max(-1)
        preds_list = []

        for i in range(len(mask_logits_list)):
            preds = -torch.ones(
                (*mask_logits_list[i].shape[-2:], 2),
                dtype=torch.long,
                device=class_logits.device,
            )
            preds[:, :, 0] = self.num_classes

            keep = classes[i].ne(class_logits.shape[-1] - 1) & (scores[i] > mask_thresh)
            if not keep.any():
                preds_list.append(preds)
                continue

            masks = mask_logits_list[i].sigmoid()
            segments = -torch.ones(
                *masks.shape[-2:],
                dtype=torch.long,
                device=class_logits.device,
            )

            mask_ids = (scores[i][keep][..., None, None] * masks[keep]).argmax(0)
            stuff_segment_ids, segment_id = {}, 0
            segment_and_class_ids = []

            for k, class_id in enumerate(classes[i][keep].tolist()):
                orig_mask = masks[keep][k] >= 0.5
                new_mask = mask_ids == k
                final_mask = orig_mask & new_mask

                orig_area = orig_mask.sum().item()
                new_area = new_mask.sum().item()
                final_area = final_mask.sum().item()
                if (
                    orig_area == 0
                    or new_area == 0
                    or final_area == 0
                    or new_area / orig_area < overlap_thresh
                ):
                    continue

                if class_id in stuff_classes:
                    if class_id in stuff_segment_ids:
                        segments[final_mask] = stuff_segment_ids[class_id]
                        continue
                    else:
                        stuff_segment_ids[class_id] = segment_id

                segments[final_mask] = segment_id
                segment_and_class_ids.append((segment_id, class_id))

                segment_id += 1

            for segment_id, class_id in segment_and_class_ids:
                segment_mask = segments == segment_id
                preds[:, :, 0] = torch.where(segment_mask, class_id, preds[:, :, 0])
                preds[:, :, 1] = torch.where(segment_mask, segment_id, preds[:, :, 1])

            preds_list.append(preds)

        return preds_list

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_panoptic(targets: list[dict]):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = -torch.ones(
                (*target["masks"].shape[-2:], 2),
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[:, :, 0] = torch.where(
                    mask, target["labels"][i], per_pixel_target[:, :, 0]
                )

                per_pixel_target[:, :, 1] = torch.where(
                    mask,
                    torch.tensor(i, device=target["masks"].device),
                    per_pixel_target[:, :, 1],
                )

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
        }
