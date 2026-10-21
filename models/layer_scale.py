import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """LayerScale on tensors with channels in last-dim.

    ``gamma`` is initialised to ``init_values``, as in timm. Until 2026-08-12
    this class ignored ``init_values`` twice over: it built the parameter from
    ``torch.empty`` (uninitialised memory) and then ``reset_parameters``
    overwrote it with ones. EoMT's cross-attention blocks are therefore grafted
    onto a pretrained residual stream at *full* strength unless a caller passes
    a small ``init_values`` -- see ``LUMA(layer_scale_init=...)``, which still
    defaults to 1.0 so existing runs reproduce.
    """

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
