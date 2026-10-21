"""Every backbone wrapper must run the full LUMA recipe correctly.

Encoders are built from the real ADE20K semantic configs, one per wrapper class,
so the tests cannot drift from what training constructs. Weights are irrelevant
here (see ``conftest.py``); the wiring is what is under test.
"""

import importlib
from pathlib import Path

import pytest
import torch
import yaml

from models.luma import LUMA

NUM_Q = 8
NUM_BLOCKS = 2
NUM_CLASSES = 4
BATCH = 2

CONFIGS = Path("configs/ade20k/semantic")
# The 224px backbone configs keep this cheap; SAM has no backbone config.
PATHS = sorted((CONFIGS / "backbones").glob("*_224.yaml")) + [CONFIGS / "pretrained_l/sam_large.yaml"]


def encoder_specs() -> list[tuple[str, dict]]:
    """(label, encoder block) for every distinct backbone wrapper in the configs."""
    seen, specs = set(), []
    for p in PATHS:
        enc = yaml.safe_load(p.read_text())["model"]["init_args"]["network"]["init_args"]["encoder"]
        if enc["class_path"] in seen:  # e.g. swin_small / swin_tiny share a wrapper
            continue
        seen.add(enc["class_path"])
        specs.append((p.stem, enc))
    return specs


SPECS = encoder_specs()
IDS = [label for label, _ in SPECS]
ENCODERS = [enc for _, enc in SPECS]


def build(enc: dict, *, nb: int = NUM_BLOCKS, mask: bool = True) -> tuple[LUMA, int]:
    mod_path, cls_name = enc["class_path"].rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_path), cls_name)
    kwargs = dict(enc.get("init_args") or {})
    kwargs.pop("ckpt_path", None)
    kwargs.pop("pretrained", None)
    # YAML 1.1 parses `1e-6` as a string; jsonargparse coerces it during training,
    # plain yaml.safe_load does not.
    for k, v in kwargs.items():
        if isinstance(v, str):
            try:
                kwargs[k] = float(v)
            except ValueError:
                pass
    # Some wrappers allocate per-block modules themselves; keep them in sync.
    for k, v in (("num_blocks", nb), ("num_q", NUM_Q)):
        if k in kwargs:
            kwargs[k] = v
    img = kwargs.pop("img_size", None) or (224, 224)

    torch.manual_seed(0)
    model = LUMA(
        encoder=cls(img_size=tuple(img), **kwargs),
        num_classes=NUM_CLASSES,
        num_q=NUM_Q,
        num_blocks=nb,
        self_a=False,
        query_only_residual=True,
        thread_queries=True,
        query_self_attn=True,
        masked_cross_attn=mask,
    )
    return model, img[0]


def grad_sum(module) -> float:
    return sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)


@pytest.mark.parametrize("enc", ENCODERS, ids=IDS)
def test_recipe_wiring(enc):
    """Masked cross-attention receives a real mask, and every LUMA block gets gradients."""
    m, side = build(enc)
    m.train()

    seen = []
    handles = [
        a.register_forward_pre_hook(lambda mod, args, kwargs: seen.append(kwargs.get("attn_mask")), with_kwargs=True)
        for a in m.attn
    ]
    try:
        mask_logits, class_logits = m(torch.randn(BATCH, 3, side, side))
    finally:
        for h in handles:
            h.remove()

    nb = m.num_blocks
    assert len(mask_logits) == nb + 1, f"expected {nb + 1} predictions, got {len(mask_logits)}"
    assert len(seen) == nb, f"{nb} cross-attentions expected, {len(seen)} ran"
    for i, am in enumerate(seen):
        assert am is not None, f"attn[{i}] got no attn_mask"
        assert am.dtype == torch.bool and am.shape[:2] == (BATCH, NUM_Q), f"attn[{i}] mask {tuple(am.shape)} {am.dtype}"
        assert not bool(am.all()), f"attn[{i}] mask blocks nothing"

    # A scalar off every prediction, so deep supervision is exercised too.
    loss = sum(t.float().mean() for t in mask_logits) + sum(t.float().mean() for t in class_logits)
    loss.backward()

    # Without query threading, block 0's update is discarded and these are exactly zero.
    dead = [
        name
        for i in range(nb)
        for name, mod in [(f"q_self_attn[{i}]", m.q_self_attn[i]), (f"attn[{i}]", m.attn[i])]
        if grad_sum(mod) == 0.0
    ]
    assert not dead, "zero gradient on " + ", ".join(dead)


def running_stats(model) -> dict[str, torch.Tensor]:
    return {
        name: buf.clone()
        for name, buf in model.encoder.named_buffers()
        if name.endswith(("running_mean", "running_var"))
    }


def accumulate(enc: dict, nb: int) -> dict[str, torch.Tensor]:
    model, side = build(enc, nb=nb, mask=False)
    # nb=1 and nb=2 draw a different number of random numbers, so live dropout
    # would desynchronise the two runs and read as drift.
    for mod in model.modules():
        for attr in ("p", "drop_prob"):
            if isinstance(getattr(mod, attr, None), float):
                setattr(mod, attr, 0.0)
    model.train()
    torch.manual_seed(1)
    with torch.no_grad():
        for _ in range(3):
            model(torch.randn(BATCH, 3, side, side))
    return running_stats(model)


@pytest.mark.parametrize("enc", ENCODERS, ids=IDS)
def test_no_bn_drift(enc):
    """The query pass must not update BatchNorm statistics shared with the image path.

    With ``query_only_residual`` the image path is identical for any number of
    LUMA blocks, so a running-stat buffer that differs between 1 and 2 blocks
    was moved by the queries.
    """
    one = accumulate(enc, 1)
    if not one:
        pytest.skip("no running stats")
    two = accumulate(enc, 2)
    moved = [n for n in one if not torch.allclose(one[n], two[n], atol=1e-6)]
    assert not moved, f"{len(moved)}/{len(one)} buffers moved by the query pass, e.g. {moved[0]}"
