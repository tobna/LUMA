import math
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class TwoStageWarmupCosineSchedule(LRScheduler):
    """
    A learning rate scheduler that implements a two-stage warmup followed by
    cosine decay, with an optional extended backbone freeze phase.

    Parameter groups are handled in two categories:
    1. Backbone parameters (indices < num_backbone_params): delayed warmup with
       an optional constant-LR freeze phase before the linear ramp begins.
    2. New-module parameters (indices >= num_backbone_params): immediate linear
       warmup from step 0, then cosine decay.

    Timeline for backbone LR:
      [0, module_warmup + backbone_freeze_steps)   → base_lr × backbone_freeze_lr_scale
      [freeze_end, freeze_end + backbone_warmup)   → linear ramp: freeze_scale → 1.0
      [warmup_end, total_steps)                    → cosine decay

    With backbone_freeze_steps=0 and backbone_freeze_lr_scale=0.0 the behaviour
    is identical to the original two-stage scheduler.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        num_backbone_params: int,
        warmup_steps: tuple[int, int],
        total_steps: int,
        backbone_freeze_steps: int = 0,
        backbone_freeze_lr_scale: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: The optimizer to schedule.
            num_backbone_params: Number of parameter groups that belong to the
                backbone. These groups occupy indices [0, num_backbone_params).
            warmup_steps: (module_warmup, backbone_warmup) — warmup lengths in
                steps for new-module and backbone groups respectively.
            total_steps: Total number of training steps.
            backbone_freeze_steps: Extra steps the backbone stays at
                backbone_freeze_lr_scale *after* module_warmup completes.
                Total freeze = module_warmup + backbone_freeze_steps.
            backbone_freeze_lr_scale: LR multiplier during the freeze phase.
                0.0 = fully frozen; 0.01 = 1 % of target LR.
            last_epoch: Index of the last epoch (PyTorch convention).
        """
        self.num_backbone_params = num_backbone_params
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.backbone_freeze_steps = backbone_freeze_steps
        self.backbone_freeze_lr_scale = backbone_freeze_lr_scale
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        module_warmup, backbone_warmup = self.warmup_steps
        freeze_end = module_warmup + self.backbone_freeze_steps
        warmup_end = freeze_end + backbone_warmup

        lrs = []
        for i, base_lr in enumerate(self.base_lrs):
            is_new_module = i >= self.num_backbone_params

            if is_new_module:
                # --- New-module parameters: immediate warmup then cosine decay ---
                if module_warmup > 0 and step < module_warmup:
                    lr = base_lr * (step / module_warmup)
                else:
                    adj = max(0, step - module_warmup)
                    decay_steps = max(1, self.total_steps - module_warmup)
                    lr = base_lr * 0.5 * (1 + math.cos(math.pi * adj / decay_steps))
            else:
                # --- Backbone parameters: freeze → linear ramp → cosine decay ---
                if step < freeze_end:
                    lr = base_lr * self.backbone_freeze_lr_scale
                elif step < warmup_end:
                    t = step - freeze_end
                    scale = (
                        self.backbone_freeze_lr_scale
                        + (1.0 - self.backbone_freeze_lr_scale) * (t / backbone_warmup)
                    )
                    lr = base_lr * scale
                else:
                    adj = max(0, step - warmup_end)
                    decay_steps = max(1, self.total_steps - warmup_end)
                    lr = base_lr * 0.5 * (1 + math.cos(math.pi * adj / decay_steps))

            lrs.append(lr)

        return lrs
