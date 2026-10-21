"""
inference.py — Post-training efficiency metric capture for EoMT / EoMT_Original.

Usage:
    # Easiest: reuse the training config that produced the checkpoint.
    # All model/encoder/img_size/num_classes values are pulled from the YAML;
    # only the checkpoint is mandatory on top. Any CLI flag overrides the config.
    # NOTE: the checkpoint is a POSITIONAL argument. There is no --ckpt_path flag.
    python inference.py \\
        --config configs/ade20k/semantic/eomt/eomt_large_512.yaml \\
        /path/to/eomt_orig.ckpt \\
        --batch_size 1 --device cuda --amp \\
        [--output metrics.json]

    # EoMT variant (backbone_kwargs forwarded to ViT encoder):
    python inference.py \\
        /path/to/eomt.ckpt \\
        --backbone EoMT \\
        --img_size 512 512 \\
        --num_classes 150 \\
        --num_q 100 \\
        --num_blocks 4 \\
        --batch_size 1 \\
        --device cuda \\
        [--all_metrics] [--amp] [--output metrics.json] \\
        --backbone_kwargs backbone_name=vit_base_patch14_dinov2.lvd142m vit_size=B emb_class=True

    # EoMT_Original variant (backbone_kwargs forwarded to ViT encoder):
    python inference.py \\
        /path/to/eomt_orig.ckpt \\
        --backbone eomt_original \\
        --img_size 512 512 \\
        --num_classes 150 \\
        --num_q 100 \\
        --num_blocks 4 \\
        --batch_size 1 \\
        --device cuda \\
        [--all_metrics] [--amp] [--output metrics.json] \\
        --backbone_kwargs backbone_name=vit_large_patch14_reg4_dinov2 vit_size=L emb_class=True

    # ResNet variant (backbone_kwargs: backbone_name=resnet50, multiplier=1):
    python inference.py \\
        /path/to/semantic_resnet50_512.ckpt \\
        --backbone resnet \\
        --img_size 512 512 \\
        --num_classes 150 \\
        --num_q 100 \\
        --num_blocks 4 \\
        --batch_size 1 \\
        --device cuda \\
        [--all_metrics] [--amp] [--output metrics.json] \\
        --backbone_kwargs backbone_name=resnet50 multiplier=1

    # Other backbones (ViT, Swin, etc.) — backbone_kwargs forwarded directly:
    python inference.py \\
        /path/to/checkpoint.ckpt \\
        --backbone ViT \\
        --img_size 512 512 \\
        --num_classes 150 \\
        --num_q 100 \\
        --batch_size 1 \\
        --device cuda \\
        [--all_metrics] [--amp] [--output metrics.json] \\
        --backbone_kwargs backbone_name=vit_base_patch16_224 vit_size=B emb_class=True

Notes:
  - Both EoMT and EoMT_Original take an inner ViT encoder; backbone_kwargs are
    forwarded to ViT.__init__ (backbone_name, vit_size, emb_class, ...).
  - ResNet uses ResNetSeg as its encoder, instantiated with backbone_name and
    multiplier from backbone_kwargs. The full model is then EoMT(encoder=ResNetSeg).
  - Use eomt_large_512.yaml (configs/ade20k/semantic/eomt/eomt_large_512.yaml) as the
    canonical reference for backbone_name / vit_size / emb_class per checkpoint.
  - All Lightning wrappers are bypassed — only the raw nn.Module is loaded.
  - For EoMT / EoMT_Original, 'num_blocks' passed via --backbone_kwargs is
    automatically extracted and NOT forwarded to ViT.__init__.
  - Architecture flags on the EoMT wrapper itself (query_only_residual,
    attend_queries, thread_queries, query_self_attn, cross_attn_encoder_norm,
    query_pos_embed, layer_scale_init, query_mlp) are read from the training
    config's network.init_args automatically and can be overridden with
    --eomt_kwargs KEY=VALUE. Several of them change the forward pass without
    changing a single state-dict key, so running without --config (or without
    the matching --eomt_kwargs) silently benchmarks the *default* architecture.
"""

import argparse
import json
import importlib
import inspect
import os
import socket

from pathlib import Path

import torch
import torch.nn as nn
import yaml

# ── project imports ──────────────────────────────────────────────────
from models.luma import LUMA
from models.eomt_original import EoMT_Original
from models.resnet import ResNetSeg
from metrics import calculate_metrics

# Backbones that wrap a ViT encoder rather than being standalone timm models.
_EOMT_ORIGINAL_ALIASES = {"eomt_original", "eomtoriginal", "eomt-original"}
_EOMT_ALIASES = {"eomt"}

# ResNet aliases — backbone_kwargs forwarded directly to ResNetSeg.__init__.
_RESNET_ALIASES = {"resnet", "resnetseg", "resnet_seg"}

# Keys in backbone_kwargs that belong to the EoMT wrapper, NOT to ViT.__init__.
_EOMT_ONLY_KWARGS = {"num_blocks"}

# Keys consumed by ResNetSeg that must NOT be forwarded to build_backbone.
_RESNET_CONSTRUCTOR_KEYS = {"backbone_name", "multiplier", "ckpt_path"}


# ── checkpoint loading (Lightning-free) ──────────────────────────────────────────


def _strip_lightning_prefix(state_dict: dict) -> dict:
    """Remove 'network.' prefix that LightningModule wraps weights under."""
    return {(k[len("network.") :] if k.startswith("network.") else k): v for k, v in state_dict.items()}


def _remove_compiled_prefix(state_dict: dict) -> dict:
    """torch.compile wraps params under '_orig_mod.*'; unwrap that."""
    return {k.replace("._orig_mod", "").replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def _clean_state_dict(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = _strip_lightning_prefix(state_dict)
    state_dict = _remove_compiled_prefix(state_dict)
    state_dict = {k: v for k, v in state_dict.items() if "criterion.empty_weight" not in k}
    return state_dict


def _check_eomt_kwargs(cls, eomt_kwargs: dict) -> dict:
    """Reject kwargs the target class does not accept, loudly.

    Silently dropping one of these is the whole reason this function exists:
    the parameter-free architecture flags (``query_only_residual``,
    ``attend_queries``, ``thread_queries``, ``cross_attn_encoder_norm``)
    change the forward pass without changing a single state-dict key, so a
    dropped flag benchmarks the *baseline* architecture with no missing/
    unexpected-key warning to notice.
    """
    accepted = set(inspect.signature(cls).parameters)
    unknown = sorted(set(eomt_kwargs) - accepted)
    if unknown:
        raise ValueError(
            f"{cls.__name__} does not accept {unknown}. "
            f"Accepted: {sorted(accepted - {'encoder', 'num_classes', 'num_q'})}"
        )
    return eomt_kwargs


def load_eomt_from_checkpoint(
    ckpt_path: str,
    backbone: nn.Module,
    num_classes: int,
    num_q: int,
    num_blocks: int = 4,
    self_a: bool = False,
    masked_attn_enabled: bool = True,
    device: torch.device = torch.device("cpu"),
    eomt_kwargs: dict | None = None,
) -> LUMA:
    """Instantiate LUMA and load weights from a checkpoint without Lightning.

    ``eomt_kwargs`` carries every remaining ``network.init_args`` entry from the
    training config (query_only_residual, attend_queries, thread_queries,
    layer_scale_init, …). Without it the model is rebuilt as the default
    architecture regardless of what was trained.
    """
    eomt_kwargs = _check_eomt_kwargs(LUMA, dict(eomt_kwargs or {}))
    model = LUMA(
        encoder=backbone,
        num_classes=num_classes,
        num_q=num_q,
        num_blocks=num_blocks,
        self_a=self_a,
        masked_attn_enabled=masked_attn_enabled,
        **eomt_kwargs,
    )
    state_dict = _clean_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARNING] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARNING] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return model.to(device)


def load_eomt_original_from_checkpoint(
    ckpt_path: str,
    backbone: nn.Module,
    num_classes: int,
    num_q: int,
    num_blocks: int = 4,
    masked_attn_enabled: bool = True,
    device: torch.device = torch.device("cpu"),
    eomt_kwargs: dict | None = None,
) -> EoMT_Original:
    """Instantiate EoMT_Original and load weights without Lightning."""
    eomt_kwargs = _check_eomt_kwargs(EoMT_Original, dict(eomt_kwargs or {}))
    model = EoMT_Original(
        encoder=backbone,
        num_classes=num_classes,
        num_q=num_q,
        num_blocks=num_blocks,
        masked_attn_enabled=masked_attn_enabled,
        **eomt_kwargs,
    )
    state_dict = _clean_state_dict(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARNING] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARNING] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return model.to(device)


# ── dummy args shim ──────────────────────────────────────────────────────────────


class _MetricArgs:
    """
    Minimal args namespace consumed by calculate_metrics / throughput / etc.
    Mirrors only the fields actually read inside metrics.py.
    """

    def __init__(
        self,
        batch_size: int,
        cuda: bool,
        amp: bool,
        eval_amp: bool,
        compile_model: bool = False,
        tqdm: bool = True,
        max_grad_norm: float = 0.0,
        amp_dtype: torch.dtype = torch.float16,
    ):
        self.batch_size = batch_size
        self.cuda = cuda
        self.amp = amp
        self.eval_amp = eval_amp
        self.compile_model = compile_model
        self.tqdm = tqdm
        self.max_grad_norm = max_grad_norm
        self.amp_dtype = amp_dtype


def build_backbone(
    model_name: str,
    img_size: tuple[int, int],
    **kwargs,
) -> nn.Module:
    """
    Instantiate a standalone backbone by its project class name (case-insensitive).

    NOTE: Do NOT call this for EoMT, EoMT_Original, or ResNet — those are full
    models that wrap an encoder. Use the dedicated loaders instead.
    """
    models_dir = Path(__file__).resolve().parent / "models"
    _model_modules = []
    if models_dir.exists() and models_dir.is_dir():
        for p in sorted(models_dir.iterdir()):
            if p.is_file() and p.suffix == ".py" and p.name != "__init__.py":
                _model_modules.append(f"models.{p.stem}")
        for p in sorted(models_dir.iterdir()):
            if p.is_dir() and (p / "__init__.py").exists():
                _model_modules.append(f"models.{p.name}")
    else:
        raise RuntimeError(f"models directory not found: {models_dir}")

    # print(f"kwargs: {kwargs}", flush=True)
    if "ls_init" in kwargs and isinstance(kwargs["ls_init"], str):
        print(f"converting ls_init {kwargs['ls_init']} to float")
        kwargs["ls_init"] = float(kwargs["ls_init"])

    for module_path in _model_modules:
        mod = importlib.import_module(module_path)
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if name.lower() == model_name.lower() and issubclass(obj, nn.Module):
                print(f"[inference] Found '{name}' in {module_path}")
                return obj(img_size=img_size, **kwargs)

    all_classes = []
    for module_path in _model_modules:
        mod = importlib.import_module(module_path)
        all_classes += [
            name
            for name, obj in inspect.getmembers(mod, inspect.isclass)
            if issubclass(obj, nn.Module) and obj.__module__ == module_path
        ]

    raise ValueError(f"Backbone class '{model_name}' not found.\nAvailable: {sorted(all_classes)}")


def _disable_fused_attention(model: nn.Module) -> int:
    """
    Force timm attention blocks to use the explicit PyTorch
    softmax(QKᵀ/√d)·V path instead of `F.scaled_dot_product_attention`
    (which dispatches to FlashAttention / mem-efficient kernels).

    Walks every submodule and flips `fused_attn = False` wherever the
    attribute exists (timm's `Attention`, `AttentionPoolLatent`, EVA blocks,
    etc.). Returns the number of modules that were toggled.
    """
    n = 0
    for m in model.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
            n += 1
    return n


def run_inferency(
    ckpt_path: str,
    backbone_name: str,
    img_size: tuple[int, int],
    num_classes: int,
    num_q: int,
    batch_size: int,
    device_str: str = "cuda",
    all_metrics: bool = True,
    amp: bool = False,
    num_blocks: int = 4,
    self_a: bool = False,
    masked_attn_enabled: bool = True,
    output_path: str | None = None,
    backbone_kwargs: dict | None = None,
    eomt_kwargs: dict | None = None,
    bf16: bool = False,
    disable_flash_attn: bool = False,
    compile_model: bool = True,
    max_batch_size: int = 10000,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    bkw = dict(backbone_kwargs or {})
    backbone_lower = backbone_name.lower()
    use_eomt_original = backbone_lower in _EOMT_ORIGINAL_ALIASES
    use_eomt = backbone_lower in _EOMT_ALIASES
    use_resnet = backbone_lower in _RESNET_ALIASES
    print(f"[inference] Running on device: {device}")

    # 1. build encoder ─────────────────────────────────────────────────────────
    if use_eomt_original or use_eomt:
        # Both EoMT variants wrap a ViT encoder.
        inner_label = bkw.get("backbone_name", "ViT")
        variant = "EoMT_Original" if use_eomt_original else "EoMT"
        print(f"[inference] Building ViT encoder for {variant} (backbone_name={inner_label})")

        eomt_bkw = {k: bkw.pop(k) for k in list(bkw) if k in _EOMT_ONLY_KWARGS}
        if "num_blocks" in eomt_bkw:
            num_blocks = eomt_bkw["num_blocks"]

        backbone = build_backbone("ViT", img_size, **bkw)

    elif use_resnet:
        # ResNetSeg is itself the encoder; pass supported kwargs directly.
        timm_name = bkw.get("backbone_name", "resnet50")
        multiplier = bkw.get("multiplier", 1)
        resnet_ckpt = bkw.get("ckpt_path", None)
        print(f"[inference] Building ResNetSeg encoder (backbone_name={timm_name}, multiplier={multiplier})")
        backbone = ResNetSeg(
            img_size=img_size,
            backbone_name=timm_name,
            multiplier=multiplier,
            ckpt_path=resnet_ckpt,
        )
        inner_label = timm_name

    else:
        print(f"[inference] Building backbone: {backbone_name}")
        backbone = build_backbone(backbone_name, img_size, **bkw)

    # 2. load model (no Lightning) ──────────────────────────────────────────
    print(f"[inference] Loading checkpoint: {ckpt_path}")
    if use_eomt_original:
        model = load_eomt_original_from_checkpoint(
            ckpt_path=ckpt_path,
            backbone=backbone,
            num_classes=num_classes,
            num_q=num_q,
            num_blocks=num_blocks,
            masked_attn_enabled=masked_attn_enabled,
            device=device,
            eomt_kwargs=eomt_kwargs,
        )
        label = f"EoMT_Original + {inner_label}"
    elif use_eomt:
        model = load_eomt_from_checkpoint(
            ckpt_path=ckpt_path,
            backbone=backbone,
            num_classes=num_classes,
            num_q=num_q,
            num_blocks=num_blocks,
            self_a=self_a,
            masked_attn_enabled=masked_attn_enabled,
            device=device,
            eomt_kwargs=eomt_kwargs,
        )
        label = f"EoMT + {inner_label}"
    elif use_resnet:
        model = load_eomt_from_checkpoint(
            ckpt_path=ckpt_path,
            backbone=backbone,
            num_classes=num_classes,
            num_q=num_q,
            num_blocks=num_blocks,
            self_a=self_a,
            masked_attn_enabled=masked_attn_enabled,
            device=device,
            eomt_kwargs=eomt_kwargs,
        )
        label = f"EoMT + ResNetSeg({inner_label})"
    else:
        model = load_eomt_from_checkpoint(
            ckpt_path=ckpt_path,
            backbone=backbone,
            num_classes=num_classes,
            num_q=num_q,
            num_blocks=num_blocks,
            self_a=self_a,
            masked_attn_enabled=masked_attn_enabled,
            device=device,
            eomt_kwargs=eomt_kwargs,
        )
        label = backbone_name
    model.eval()
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.masked_attn_enabled = False  # deployed, masking-free inference path

    # Optionally disable timm's fused SDPA path (FlashAttention / mem-efficient)
    # and fall back to the explicit PyTorch softmax(QKᵀ/√d)·V implementation.
    if disable_flash_attn:
        n_toggled = _disable_fused_attention(model)
        print(f"[inference] Flash/SDPA disabled on {n_toggled} attention modules — using vanilla PyTorch attention.")

    # torch.compile the model for throughput measurements (paper convention).
    # FLOPs/MACs are computed on `_orig_mod` inside metrics.py, so the compile
    # wrapper doesn't affect those numbers — it only speeds up the timed runs.
    if compile_model:
        print("[inference] Compiling model with torch.compile() ...")
        model = torch.compile(model)

    # 3. synthetic input (B, 3, H, W) ───────────────────────────────────────
    dummy_input = torch.randn(
        batch_size,
        3,
        img_size[0],
        img_size[1],
        device=device,
    )

    # 4. assemble metric args ──────────────────────────────────────────────
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    metric_args = _MetricArgs(
        batch_size=batch_size, cuda=use_cuda, amp=amp, eval_amp=amp, amp_dtype=amp_dtype, compile_model=compile_model
    )

    # 5. run calculate_metrics ───────────────────────────────────────────
    print("[inference] Calculating metrics…")
    metrics = calculate_metrics(
        args=metric_args,
        model=model,
        rank=0,
        input=dummy_input,
        device=device if use_cuda else None,
        did_training=False,
        world_size=1,
        all_metrics=all_metrics,
        n_ims=batch_size,
        key_start="inference/",
        max_batch_size=max_batch_size,
    )

    # 6. pretty-print results ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  EoMT Efficiency Metrics  |  backbone: {label}")
    print("=" * 60)

    serialisable = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_k, sub_v in value.items():
                print(f"      {sub_k}: {sub_v}")
            serialisable[key] = value
        elif isinstance(value, torch.Tensor):
            v = value.item()
            print(f"  {key}: {v}")
            serialisable[key] = v
        else:
            print(f"  {key}: {value}")
            serialisable[key] = value

    print("=" * 60 + "\n")

    # 7. optional JSON dump ────────────────────────────────────────────
    if output_path:
        # Record the SLURM node name (falls back to hostname outside SLURM).
        serialisable["node"] = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
        Path(output_path).write_text(json.dumps(serialisable, indent=2))
        print(f"[inference] Metrics saved → {output_path}")

    return metrics


# ── CLI ─────────────────────────────────────────────────────────────────────────


def _parse_backbone_kwargs(raw: list[str] | None) -> dict:
    if not raw:
        return {}
    kwargs = {}
    for item in raw:
        k, _, v = item.partition("=")
        if v.lower() in ("true", "false"):
            kwargs[k] = v.lower() == "true"
        else:
            try:
                kwargs[k] = int(v)
            except ValueError:
                try:
                    kwargs[k] = float(v)
                except ValueError:
                    kwargs[k] = v
    return kwargs


# ── training-config → inference-args mapping ─────────────────────────────────────


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _infer_config_from_ckpt(ckpt_path: str) -> str | None:
    """
    Try to recover the training YAML that produced this checkpoint.

    `main.py` mirrors `configs/<X>.yaml` → `<output_root>/<X>/<name>.ckpt`, so
    the path components of the checkpoint's parent directory should contain
    `<X>` as a contiguous slice. We enumerate every (start, end) window over
    those components and test `<repo>/configs/<window>.yaml`. This tolerates:
      - configs at any depth (`configs/<dataset>/<task>/<backbone>.yaml`, or
        deeper ones under `pretrained/`, `size_vs_res/`, …),
      - extra per-run subfolders below the config-mirrored dir,
      - unknown output roots (we don't need to know where the mirror starts).

    Selection rule:
      1. Longest matching window wins (most specific).
      2. On ties, prefer the window whose end is closest to the ckpt file
         (i.e. innermost — the per-run-subfolder case sits *above* the file).
      3. If still ambiguous, return None and let the caller require --config.
    """
    if not ckpt_path:
        return None
    configs_root = Path(__file__).resolve().parent / "configs"
    if not configs_root.is_dir():
        return None

    parent = Path(ckpt_path).resolve().parent
    parts = parent.parts
    if not parts:
        return None

    candidates: list[tuple[int, int, Path]] = []  # (length, end_idx, path)
    for start in range(len(parts)):
        for end in range(start, len(parts)):
            window = parts[start : end + 1]
            # Drop trivially-empty windows (e.g. just the POSIX root '/').
            if all(p in ("", "/") for p in window):
                continue
            candidate = configs_root.joinpath(*window).with_suffix(".yaml")
            if candidate.is_file():
                candidates.append((end - start + 1, end, candidate))

    if not candidates:
        return None

    # Sort by (length desc, end desc) → most specific, innermost.
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best_len, best_end, best_path = candidates[0]
    ties = [c for c in candidates if c[0] == best_len and c[1] == best_end]
    if len(ties) > 1:
        print(
            "[inference] Could not unambiguously infer --config from ckpt path; "
            f"candidates: {[str(c[2]) for c in ties]}. Pass --config explicitly."
        )
        return None
    return str(best_path)


def _class_tail(class_path: str) -> str:
    """'models.eomt_original.EoMT_Original' → 'EoMT_Original'."""
    return class_path.rsplit(".", 1)[-1] if class_path else ""


def _backbone_from_train_cfg(network_cls: str, encoder_cls: str) -> str:
    """Translate the training network/encoder class_path pair into inference --backbone."""
    network_tail = _class_tail(network_cls).lower()
    encoder_tail = _class_tail(encoder_cls)
    if network_tail == "eomt_original":
        return "eomt_original"
    if encoder_tail.lower() in {"resnetseg", "resnet_seg"}:
        return "resnet"
    return encoder_tail  # ViT, Swin, Coat, ...


def _resolve_data_arg(data_cfg: dict, key: str):
    """
    Read `key` from data.init_args; fall back to the data class's __init__
    default. main.py links both num_classes and img_size from the datamodule
    via parser.link_arguments, so a config that omits them trained with the
    class default — not with inference.py's argparse default.
    """
    init = (data_cfg or {}).get("init_args") or {}
    if key in init:
        return init[key]

    cls_path = (data_cfg or {}).get("class_path")
    if not cls_path:
        return None
    try:
        mod_path, cls_name = cls_path.rsplit(".", 1)
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        param = inspect.signature(cls).parameters.get(key)
        if param is not None and param.default is not inspect.Parameter.empty:
            return param.default
    except Exception as e:
        # Loud: a missing dep (pycocotools outside the container) would otherwise
        # silently hand back inference.py's own default for a value that training
        # took from the datamodule.
        print(f"[inference] could not read data default '{key}' from {cls_path}: {e}", flush=True)
        return None
    return None


# network.init_args keys that reach the model through a dedicated argparse flag
# rather than through eomt_kwargs. Everything else in network.init_args is an
# architecture flag and must be forwarded verbatim — see _EOMT_ARCH_NOTE.
_NETWORK_KEYS_HANDLED = {"encoder", "num_classes", "img_size", "num_q", "num_blocks", "self_a", "masked_attn_enabled"}


def _train_config_to_inference(cfg: dict) -> tuple[dict, dict, dict]:
    """
    Extract inference-relevant fields from a training YAML.

    Returns (argparse_defaults, backbone_kwargs, eomt_kwargs).

    ``eomt_kwargs`` is every ``network.init_args`` key that is not already
    covered by a dedicated flag. Forwarding the whole remainder — rather than a
    hand-maintained list — is deliberate: a new EoMT architecture flag is
    honoured here the day it is added to the config schema, instead of being
    silently dropped until someone notices the benchmark numbers are the
    baseline's.
    """
    defaults: dict = {}
    bkw: dict = {}
    eomt_kw: dict = {}

    model_init = ((cfg.get("model") or {}).get("init_args")) or {}
    network_block = model_init.get("network") or {}
    network_cls = network_block.get("class_path", "")
    network_init = network_block.get("init_args") or {}

    encoder_block = network_init.get("encoder") or {}
    encoder_cls = encoder_block.get("class_path", "")
    encoder_init = encoder_block.get("init_args") or {}

    # ── backbone selector ───────────────────────────────────────────────────
    backbone_sel = _backbone_from_train_cfg(network_cls, encoder_cls)
    if backbone_sel:
        defaults["backbone"] = backbone_sel

    # ── EoMT wrapper fields ─────────────────────────────────────────────────
    if "num_q" in network_init:
        defaults["num_q"] = int(network_init["num_q"])
    if "num_blocks" in network_init:
        defaults["num_blocks"] = int(network_init["num_blocks"])
    if "self_a" in network_init:
        defaults["self_a"] = bool(network_init["self_a"])
    if "masked_attn_enabled" in network_init:
        defaults["masked_attn"] = bool(network_init["masked_attn_enabled"])

    # ── every remaining architecture flag (query_only_residual, attend_queries,
    #    thread_queries, cross_attn_encoder_norm, layer_scale_init, query_mlp, …)
    for k, v in network_init.items():
        if k not in _NETWORK_KEYS_HANDLED:
            eomt_kw[k] = v

    # ── data ────────────────────────────────────────────────────────────────
    data_cfg = cfg.get("data") or {}
    img_size = _resolve_data_arg(data_cfg, "img_size")
    if img_size is not None:
        defaults["img_size"] = list(img_size)
    nc = _resolve_data_arg(data_cfg, "num_classes")
    if nc is not None:
        defaults["num_classes"] = int(nc)

    # ── backbone_kwargs from encoder.init_args (img_size is passed separately) ─
    for k, v in encoder_init.items():
        if k == "img_size":
            continue
        bkw[k] = v

    return defaults, bkw, eomt_kw


def _parse_args(argv=None):
    # 1) Pre-parse for --config only; lets us inject YAML values as defaults
    #    so explicit CLI flags still win.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", default=None)
    pre.add_argument("ckpt_path", nargs="?", default=None)
    pre_args, _ = pre.parse_known_args(argv)

    # If --config wasn't passed but we have a ckpt path, try to recover the
    # training YAML from the checkpoint's location (configs/<X>.yaml ↔
    # <output_root>/<X>/<name>.ckpt mirror established in main.py).
    inferred_config = None
    if not pre_args.config and pre_args.ckpt_path:
        inferred_config = _infer_config_from_ckpt(pre_args.ckpt_path)
        if inferred_config:
            print(f"[inference] --config not set; inferred {inferred_config} from ckpt path", flush=True)
        else:
            print(
                f"[inference] --config not set; failed to infer config from ckpt path {pre_args.ckpt_path}", flush=True
            )

    cfg_path = pre_args.config or inferred_config
    cfg_defaults: dict = {}
    cfg_bkw: dict = {}
    cfg_eomt_kw: dict = {}
    if cfg_path:
        cfg = _load_yaml(cfg_path)
        cfg_defaults, cfg_bkw, cfg_eomt_kw = _train_config_to_inference(cfg)

    # 2) Main parser — nothing is required at the argparse level; we validate
    #    after merging config + CLI so YAML-supplied values count as "provided".
    p = argparse.ArgumentParser(
        description="Post-training EoMT / EoMT_Original / ResNet efficiency metric capture (Lightning-free).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        "-c",
        default=None,
        help=(
            "Path to a training YAML (e.g. configs/ade20k/semantic/eomt/eomt_large_512.yaml). "
            "Network class, encoder kwargs, img_size and num_classes are taken from it. "
            "Any CLI flag overrides the config."
        ),
    )
    p.add_argument("ckpt_path", default=None)
    p.add_argument(
        "--backbone",
        default=None,
        help=(
            "Backbone selector (case-insensitive). "
            "Use 'EoMT' or 'eomt_original' for those full models — "
            "backbone_kwargs are then forwarded to the inner ViT encoder. "
            "Use 'resnet' to use ResNetSeg as encoder — backbone_kwargs: "
            "backbone_name=resnet50 multiplier=1. "
            "For other standalone backbones (ViT, Swin, …) the class is "
            "instantiated directly with backbone_kwargs."
        ),
    )
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--num_q", type=int, default=None, help="Number of EoMT query slots.")
    p.add_argument("--img_size", nargs=2, type=int, default=[512, 512], metavar=("H", "W"))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num_blocks", type=int, default=4)
    p.add_argument("--self_a", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--masked_attn", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--all_metrics", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use bfloat16 instead of float16 for AMP autocast (requires --amp). Preferred on A100/H100.",
    )
    p.add_argument(
        "--compile",
        dest="compile_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Wrap the model in torch.compile() before timing throughput. "
            "Matches the EoMT paper's measurement protocol. FLOPs/MACs are "
            "always computed on the un-compiled module."
        ),
    )
    p.add_argument(
        "--disable-flash-attn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable timm's fused scaled_dot_product_attention path (which "
            "dispatches to FlashAttention / mem-efficient SDPA) and fall back "
            "to the explicit PyTorch softmax(QKᵀ/√d)·V implementation. "
            "Affects every submodule that exposes `fused_attn` (ViT blocks, "
            "EVA, AttentionPoolLatent, etc.)."
        ),
    )
    p.add_argument(
        "--output",
        default=None,
        help="Path to write JSON metrics output. Defaults to '<checkpoint_dir>/metrics.json' if not provided.",
    )
    p.add_argument(
        "--backbone_kwargs",
        nargs="*",
        metavar="KEY=VALUE",
        help=(
            "kwargs forwarded to the backbone constructor. "
            "Merged on top of any kwargs derived from --config (per-key override). "
            "EoMT/eomt_original: backbone_name=<timm_name> vit_size=<B|L> emb_class=<True|False>. "
            "resnet: backbone_name=resnet50 multiplier=1. "
            "num_blocks=N passed here overrides --num_blocks (EoMT only)."
        ),
    )
    p.add_argument(
        "--eomt_kwargs",
        nargs="*",
        metavar="KEY=VALUE",
        help=(
            "Architecture kwargs forwarded to the EoMT / EoMT_Original constructor "
            "(query_only_residual, attend_queries, thread_queries, query_self_attn, "
            "cross_attn_encoder_norm, query_pos_embed, layer_scale_init, query_mlp, …). "
            "Taken from the training config's network.init_args automatically; use this "
            "only to override. An unknown key is an error, not a silent no-op — several "
            "of these flags change the forward pass without changing any state-dict key."
        ),
    )
    p.add_argument("--max-batch-size", type=int, default=10_000, help="Maximum batch size to try out.")

    # 3) Inject config-derived values as defaults — explicit CLI flags still win.
    if cfg_defaults:
        p.set_defaults(**cfg_defaults)

    args = p.parse_args(argv)

    # 4) Merge backbone_kwargs: config first, CLI key=value pairs override per key.
    cli_bkw = _parse_backbone_kwargs(args.backbone_kwargs)
    merged = dict(cfg_bkw)
    merged.update(cli_bkw)
    args.backbone_kwargs_dict = merged

    merged_eomt = dict(cfg_eomt_kw)
    merged_eomt.update(_parse_backbone_kwargs(args.eomt_kwargs))
    args.eomt_kwargs_dict = merged_eomt
    if merged_eomt:
        print(f"[inference] EoMT architecture kwargs: {merged_eomt}")

    # 5) Validate required fields (after config + CLI merge).
    missing = [name for name in ("ckpt_path", "backbone", "num_classes", "num_q") if getattr(args, name) in (None, "")]
    if missing:
        p.error("Missing required values (provide via --config or CLI): " + ", ".join(f"--{m}" for m in missing))

    return args


if __name__ == "__main__":
    args = _parse_args()

    # Default --output to <checkpoint_dir>/metrics.json when not provided.
    output_path = args.output
    if output_path is None and args.ckpt_path:
        output_path = str(
            Path(args.ckpt_path).resolve().parent
            / ("metrics-no-flash-attn.json" if args.disable_flash_attn else "metrics.json")
        )
        print(f"[inference] --output not set; defaulting to {output_path}")

    run_inferency(
        ckpt_path=args.ckpt_path,
        backbone_name=args.backbone,
        img_size=tuple(args.img_size),
        num_classes=args.num_classes,
        num_q=args.num_q,
        batch_size=args.batch_size,
        device_str=args.device,
        all_metrics=args.all_metrics,
        amp=args.amp,
        num_blocks=args.num_blocks,
        self_a=args.self_a,
        masked_attn_enabled=args.masked_attn,
        output_path=output_path,
        backbone_kwargs=args.backbone_kwargs_dict,
        eomt_kwargs=args.eomt_kwargs_dict,
        bf16=args.bf16,
        disable_flash_attn=args.disable_flash_attn,
        compile_model=args.compile_model,
        max_batch_size=args.max_batch_size,
    )
