import pytest
import timm


@pytest.fixture(autouse=True)
def offline_timm(monkeypatch):
    """Random-init every backbone, so the tests need no downloads and no weights.

    Most wrappers hardcode ``pretrained=ckpt_path is None``, so dropping the
    checkpoint alone would make them fetch ImageNet weights from the hub.
    """
    real = timm.create_model

    def create_model(*args, **kwargs):
        kwargs["pretrained"] = False
        return real(*args, **kwargs)

    monkeypatch.setattr(timm, "create_model", create_model)
