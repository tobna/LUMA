"""LUMA(predict_q_norm=...): an ablation flag, off in the paper recipe."""

import torch

from models.luma import LUMA
from models.vit import ViT


def _luma(**kw):
    torch.manual_seed(0)
    # Small random-init ViT — no download, no pretrained weights.
    encoder = ViT(
        img_size=(64, 64),
        backbone_name="vit_small_patch16_384",
        patch_size=16,
        pretrained=False,
        vit_size="S",
    )
    return LUMA(
        encoder=encoder, num_classes=5, num_q=4, num_blocks=2,
        thread_queries=True, self_a=False, **kw
    )


def test_off_is_identical():
    """Default must reproduce the pre-change forward bit-for-bit."""
    a, b = _luma(), _luma(predict_q_norm=False)
    assert a.pred_q_norm is None
    a.eval(), b.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        ma, _ = a(x)
        mb, _ = b(x)
    assert torch.equal(ma[-1], mb[-1]), "predict_q_norm=False changed the forward"


def test_on_feeds_raw_queries_to_every_prediction():
    """The head norm must run at all num_blocks+1 sites, on *raw* queries.

    The discriminating check is the first one: at block 0 the raw query is
    exactly the `self.q` parameter, so if vit.py handed `_predict` the
    block.norm1'd slice instead, this fails.
    """
    m = _luma(predict_q_norm=True)
    assert m.pred_q_norm is not None

    seen = []
    m.pred_q_norm.register_forward_pre_hook(
        lambda mod, args: seen.append(args[0].detach())
    )

    m.train()
    m(torch.randn(2, 3, 64, 64))

    assert len(seen) == m.num_blocks + 1, (
        f"expected {m.num_blocks + 1} prediction sites, got {len(seen)}"
    )
    assert torch.allclose(seen[0], m.q.detach()[None].expand(2, -1, -1)), (
        "first _predict did not get the raw query — vit.py passed the normalised slice"
    )
    for i, q in enumerate(seen):
        assert q.shape == (2, m.num_q, m.encoder.embed_dim), f"call {i}: {q.shape}"


def test_double_norm_guard():
    """predict_q_norm + final_query_norm would normalise the final query twice."""
    try:
        _luma(predict_q_norm=True, final_query_norm=True)
    except ValueError as e:
        assert "final_query_norm" in str(e)
    else:
        raise AssertionError("predict_q_norm + final_query_norm should raise")
