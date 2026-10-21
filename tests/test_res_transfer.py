"""Loading an IN21k backbone at a larger ``img_size`` must preserve what the
pretrained attention computes.

Three wrappers rewrite sequence-length-dependent weights while loading:

* ``linformer`` contracts ``proj_k``/``proj_v`` over the sequence, so the resized
  projections must keep their column sums.
* ``hydra`` sums ``k*v`` over the sequence with no 1/N, so ``attn.out`` must be
  rescaled by the token ratio.
* ``synthfr`` must resize ``random_attn`` and ``pos_embed`` as 2-D grids, not
  along the 1-D raster.

Each test plants a known pattern in a fake 224 checkpoint, loads it at 512 through
the real wrapper, and checks that property.
"""

import tempfile
from pathlib import Path

import torch

SRC, DST = 224, 512
P = 16
GS_OLD, GS_NEW = SRC // P, DST // P          # 14, 32
SEQ_OLD, SEQ_NEW = GS_OLD**2 + 1, GS_NEW**2 + 1   # 197, 1025


def transfer(cls, backbone_name, plant):
    """Build at 224, plant a pattern in its state dict, reload it at 512."""
    src = cls(img_size=(SRC, SRC), backbone_name=backbone_name, ls_init=1e-6,
              vit_size="S", ckpt_path=None)
    sd = {k: v.clone() for k, v in src.backbone.state_dict().items()}
    plant(sd)
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "fake_in21k.pt"
        torch.save({"model_state": sd}, ckpt)
        dst = cls(img_size=(DST, DST), backbone_name=backbone_name, ls_init=1e-6,
                  vit_size="S", ckpt_path=str(ckpt))
    return sd, dst.backbone.state_dict()


def test_linformer():
    from models.linformer import Linformer

    def plant(sd):
        for k, v in sd.items():
            if "proj_k" in k or "proj_v" in k:
                v[1:] = 1.0          # every grid row contributes exactly 1 per column
                v[0] = 0.0

    src, dst = transfer(Linformer, "linformer_vit_small_patch16", plant)
    keys = [k for k in dst if "proj_k" in k or "proj_v" in k]
    assert keys, "no linformer projections found"
    for k in keys:
        assert dst[k].shape[0] == SEQ_NEW, (k, dst[k].shape)
        # attn does einsum("bnd,nk->bkd"), i.e. it sums the projection over the
        # sequence -- the column sums are what reaches the softmax.
        got, want = dst[k][1:].sum(0), src[k][1:].sum(0)
        assert torch.allclose(got, want, rtol=2e-2), (
            f"{k}: column sums {got.mean():.1f} vs {want.mean():.1f} "
            f"(unrescaled resize would give {want.mean() * (SEQ_NEW - 1) / (SEQ_OLD - 1):.1f})"
        )


def test_hydra():
    from models.hydra import Hydra

    src, dst = transfer(Hydra, "hydra_vit_small_patch16", lambda sd: None)
    keys = [k for k in dst if k.endswith("attn.out.weight")]
    assert keys, "no hydra attn.out found"
    want = SEQ_OLD / SEQ_NEW
    for k in keys:
        # HydraAttention does (k*v).sum(dim=-2) * q with no 1/N, so out.weight
        # carries the missing normalisation.
        ratio = dst[k] / src[k]
        assert torch.allclose(ratio, torch.full_like(ratio, want), rtol=1e-5), (
            f"{k}: scaled by {ratio.mean():.4f}, expected {want:.4f}")
        b = k[: -len("weight")] + "bias"
        assert torch.equal(dst[b], src[b]), f"{b} must not be rescaled"


def test_synthfr():
    from models.synthFR import SynthFR

    def rows(n):
        """A pattern that depends on the grid *row* only."""
        return torch.arange(n, dtype=torch.float32).view(n, 1).expand(n, n).reshape(-1)

    def plant(sd):
        sd["pos_embed"][0, :, :] = rows(GS_OLD)[:, None]
        for k, v in sd.items():
            if "random_attn_1" in k:                       # [H, L, f]
                v[:, 1:1 + GS_OLD**2, :] = rows(GS_OLD)[None, :, None]
            elif "random_attn_2" in k:                     # [H, f, L]
                v[:, :, 1:1 + GS_OLD**2] = rows(GS_OLD)[None, None, :]

    src, dst = transfer(SynthFR, "synthesizer_fr_vit_small_patch16", plant)

    def assert_row_only(grid, what):
        """1-D interpolation over the raster mixes rows; 2-D interpolation cannot."""
        g = grid.reshape(GS_NEW, GS_NEW)
        spread = (g - g[:, :1]).abs().max()
        assert spread < 1e-3, f"{what}: varies along a row by {spread:.3f}, rows got blended"
        assert torch.all(g[1:, 0] > g[:-1, 0]), f"{what}: row order not preserved"
        assert abs(g[0, 0]) < 0.5 and abs(g[-1, 0] - (GS_OLD - 1)) < 0.5, (
            f"{what}: range {g[0,0]:.2f}..{g[-1,0]:.2f}, expected 0..{GS_OLD - 1}")

    assert_row_only(dst["pos_embed"][0, :, 0], "pos_embed")

    keys = [k for k in dst if "random_attn" in k]
    assert keys, "no synthfr random_attn found"
    for k in keys:
        v = dst[k]
        seq = v[0, 0, :] if "random_attn_2" in k else v[0, :, 0]
        assert_row_only(seq[1:1 + GS_NEW**2], k)
        pad = seq[1 + GS_NEW**2:]
        assert torch.all(pad == 0), f"{k}: padding must stay zero, got {pad}"
