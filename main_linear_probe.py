# ---------------------------------------------------------------
# Entry point for ImageNet linear-probe training.
#
# Pass the *same* segmentation config you use for backbone training.
# The script extracts ONLY the encoder definition; all other settings
# (trainer hyperparams, data path, lr, etc.) come from the defaults file:
#
#   configs/linear_probe_defaults.yaml   ← edit this for your ImageNet path,
#                                          epochs, …
#
# Usage:
#   python main_linear_probe.py fit \
#       -c configs/ade20k/semantic/vit_base.yaml
#
# Any -c file that does NOT contain a segmentation encoder is passed
# through to LightningCLI unchanged — use this for per-run overrides:
#   python main_linear_probe.py fit \
#       -c configs/ade20k/semantic/vit_base.yaml \
#       -c my_lp_overrides.yaml \
#       --trainer.max_epochs 30
#
# Config load order (later values win):
#   1. configs/linear_probe_defaults.yaml  — project defaults
#   2. Extracted encoder + dynamic run name  — injected programmatically
#   3. Non-seg -c override files  — per-run overrides
#   4. CLI --key value flags  — highest priority
#
# Experiment name = {seg_logger_name}_lp
#   override with: --trainer.logger.init_args.name my_name
# Output dir = {output_dir}/{seg_subdir}/linear_probe/{timestamp}/
#   (--output_dir, default: outputs)
# ---------------------------------------------------------------

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import yaml
import jsonargparse._typehints as _t
import logging
import torch
import warnings
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
from lightning.pytorch.loggers import Logger as _LightningLogger
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from lightning.pytorch.utilities import rank_zero_only

from training.linear_probe_imagenet import LinearProbeImageNet
from datasets.lightning_data_module import LightningDataModule

_DEFAULTS_FILE = Path(__file__).parent / "configs" / "linear_probe_defaults.yaml"


# ---------------------------------------------------------------------------
# Shared helpers (identical to main.py)
# ---------------------------------------------------------------------------

def _to_scalar(v):
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return str(v)
    return v


class JsonlLogger(_LightningLogger):
    """Append-only JSONL metrics logger (identical to main.py)."""

    def __init__(self, save_dir, filename="train.jsonl"):
        super().__init__()
        self._save_dir = str(save_dir)
        os.makedirs(self._save_dir, exist_ok=True)
        self._path = os.path.join(self._save_dir, filename)

    @property
    def name(self):
        return "jsonl"

    @property
    def version(self):
        return ""

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._save_dir

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        record = {"step": step}
        for k, v in metrics.items():
            record[k] = _to_scalar(v)
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")

    @rank_zero_only
    def log_hyperparams(self, params, *args, **kwargs):
        return


_orig_single = _t.raise_unexpected_value


def _raise_single(*args, exception=None, **kwargs):
    if isinstance(exception, Exception):
        raise exception
    return _orig_single(*args, exception=exception, **kwargs)


_orig_union = _t.raise_union_unexpected_value


def _raise_union(subtypes, val, vals):
    for e in reversed(vals):
        if isinstance(e, Exception):
            raise e
    return _orig_union(subtypes, val, vals)


_t.raise_unexpected_value = _raise_single
_t.raise_union_unexpected_value = _raise_union



def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    if not self._should_check_val_epoch():
        return False
    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True
    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True
    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (self.batch_idx + 1) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (self.batch_idx + 1) % self.trainer.val_check_batch == 0
        else:
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()
    return is_val_check_batch


# ---------------------------------------------------------------------------
# LightningCLI subclass
# ---------------------------------------------------------------------------

class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, jsonl_save_dir=None, **kwargs):
        self._jsonl_save_dir = jsonl_save_dir
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        warnings.filterwarnings(
            "ignore",
            message=r".*It is recommended to use .* when logging on epoch level.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*functools.partial will be a method descriptor.*",
        )
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        # num_classes (1000 for ImageNet) → linear head output size.
        # With subclass_mode_model=False, model args have no init_args wrapper.
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.num_classes",
        )

    @staticmethod
    def _collect_git_state():
        def _run(args):
            try:
                out = subprocess.run(
                    ["git", *args],
                    capture_output=True, text=True, timeout=5, check=False,
                )
                return out.stdout.strip() or None if out.returncode == 0 else None
            except (OSError, subprocess.SubprocessError):
                return None
        status = _run(["status", "--porcelain"])
        return {
            "commit": _run(["rev-parse", "HEAD"]),
            "branch": _run(["rev-parse", "--abbrev-ref", "HEAD"]),
            "describe": _run(["describe", "--tags", "--always", "--dirty"]),
            "remote": _run(["config", "--get", "remote.origin.url"]),
            "dirty": bool(status) if status is not None else None,
            "status": status,
        }

    @staticmethod
    @rank_zero_only
    def _dump_hyperparams(save_dir, config):
        if save_dir is None:
            return
        os.makedirs(save_dir, exist_ok=True)
        try:
            cfg_dict = config.as_dict()
        except AttributeError:
            cfg_dict = {"config": str(config)}
        payload = {
            "invocation": {
                "argv": sys.argv,
                "command": shlex.join(sys.argv),
                "cwd": os.getcwd(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            "git": LightningCLI._collect_git_state(),
            "config": cfg_dict,
        }
        with open(os.path.join(save_dir, "hyperparams.json"), "w") as f:
            json.dump(payload, f, indent=2, default=str, sort_keys=False)

    def fit(self, model, **kwargs):
        try:
            if hasattr(self.trainer.logger.experiment, "log_code"):
                from gitignore_parser import parse_gitignore
                self.trainer.logger.experiment.log_code(
                    ".",
                    include_fn=lambda p: p.endswith(".py") or p.endswith(".yaml"),
                    exclude_fn=parse_gitignore(".gitignore"),
                )
        except Exception:
            pass

        if self._jsonl_save_dir is not None:
            self.trainer.loggers.append(JsonlLogger(save_dir=self._jsonl_save_dir))
            self._dump_hyperparams(self._jsonl_save_dir, self.config)

        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )
        self.trainer.fit(model, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_run_timestamp() -> str:
    slurm_start = os.environ.get("SLURM_JOB_START_TIME")
    if slurm_start:
        try:
            return datetime.fromtimestamp(int(slurm_start)).strftime("%Y-%m-%d-%H-%M-%S")
        except (ValueError, OSError):
            pass
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def _config_to_subdir(config_path: str) -> str | None:
    parts = list(Path(config_path).with_suffix("").parts)
    if parts and parts[0] == "configs":
        parts = parts[1:]
    return "/".join(parts) if parts else None


def _extract_encoder(cfg: dict) -> dict | None:
    """Return the encoder sub-config from a segmentation YAML dict, or None."""
    try:
        return cfg["model"]["init_args"]["network"]["init_args"]["encoder"]
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Pre-processing: argparse extracts -c configs before LightningCLI sees argv
# ---------------------------------------------------------------------------

def _preprocess_argv(argv: list[str]):
    """Use argparse to collect -c config files, detect segmentation ones,
    and route them correctly.

    Uses ``argparse.ArgumentParser(add_help=False)`` + ``parse_known_args``
    (the same pattern as ``_resolve_run_info`` in main.py) so all the
    edge-cases of argument parsing (``-c=file``, ``--config=file``, etc.)
    are handled correctly without manual string iteration.

    Segmentation configs (those containing
    ``model.init_args.network.init_args.encoder``) are consumed entirely:
    encoder and run name are extracted, everything else is discarded.

    Non-segmentation -c configs (e.g. per-run override files) are kept and
    will be forwarded to LightningCLI as additional -c arguments.

    Returns:
        passthrough_configs:  List of non-seg config paths for LightningCLI.
        remaining_argv:       All non-config argv tokens (subcommand + flags).
        encoder_cfg:          Extracted encoder dict, or None.
        seg_name:             WandbLogger name from the seg config, or None.
        seg_subdir:           Output subdir derived from the seg config, or None.
        name_override:        --trainer.logger.init_args.name value, or None.
        output_dir:           --output_dir value (default: outputs).
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", action="append", default=[])
    # Also capture the logger-name override so we can derive lp_name before
    # LightningCLI starts (needed for ModelCheckpoint filename and output dir).
    pre.add_argument(
        "--trainer.logger.init_args.name", dest="logger_name", default=None
    )
    pre.add_argument("--output_dir", default="outputs")

    known, remaining_argv = pre.parse_known_args(argv)

    encoder_cfg = None
    seg_name = None
    seg_subdir = None
    passthrough_configs: list[str] = []

    for config_path in known.config:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            enc = _extract_encoder(cfg)
        except (OSError, yaml.YAMLError):
            enc = None

        if enc is not None:
            # Segmentation config — extract what we need, drop the file.
            encoder_cfg = enc
            try:
                seg_name = cfg["trainer"]["logger"]["init_args"]["name"]
            except (KeyError, TypeError):
                pass
            seg_subdir = _config_to_subdir(config_path)
        else:
            # Non-segmentation config (LP override file) — pass through.
            passthrough_configs.append(config_path)

    return (
        passthrough_configs,
        remaining_argv,
        encoder_cfg,
        seg_name,
        seg_subdir,
        known.logger_name,
        known.output_dir,
    )


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------

def cli_main(name: str, subdir: str, run_ts: str, output_dir: str = "outputs"):
    save_dir = f"{output_dir}/{subdir}/{run_ts}/"
    LightningCLI(
        LinearProbeImageNet,
        LightningDataModule,
        jsonl_save_dir=save_dir,
        subclass_mode_model=False,  # encoder injected as str+dict, no nn.Module dispatch
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=True,
        trainer_defaults={
            "precision": "bf16-mixed",
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=3),
                ModelCheckpoint(
                    dirpath=save_dir,
                    filename=name,
                    monitor="metrics/val_acc_top1",
                    mode="max",
                    save_top_k=1,
                    save_last=False,
                ),
            ],
            "devices": "auto",
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    (
        passthrough_configs,
        remaining_argv,
        encoder_cfg,
        seg_name,
        seg_subdir,
        name_override,
        output_dir,
    ) = _preprocess_argv(sys.argv[1:])

    if encoder_cfg is None:
        print(
            "ERROR: No encoder found.\n"
            "Pass a segmentation config that contains "
            "model.init_args.network.init_args.encoder, e.g.:\n\n"
            "  python main_linear_probe.py fit \\\n"
            "      -c configs/ade20k/semantic/vit_base.yaml",
            file=sys.stderr,
        )
        sys.exit(1)

    # Derive LP experiment name and output subdir.
    lp_name = name_override or (f"{seg_name}_lp" if seg_name else "linear_probe")
    lp_subdir = f"{seg_subdir}/linear_probe" if seg_subdir else lp_name
    run_ts = _get_run_timestamp()

    print(f"Backbone:         {seg_subdir}")
    print(f"Experiment name:  {lp_name}")
    print(
        f"Output directory: "
        f"{output_dir}/{lp_subdir}/{run_ts}/"
    )

    # ── Resolve LP image size ────────────────────────────────────────────────
    # The seg encoder was trained at a different resolution (e.g. 256×256).
    # Override img_size in the injected encoder_init so the encoder's positional
    # embeddings match the LP data module.
    # Read from defaults file; fall back to 224×224.
    # If you change --data.init_args.img_size on the CLI, also pass
    # --model.encoder_init.img_size to keep them in sync.
    lp_img_size: list[int] = [224, 224]
    if _DEFAULTS_FILE.exists():
        try:
            with open(_DEFAULTS_FILE) as _f:
                _defs = yaml.safe_load(_f) or {}
            _from_defaults = (
                _defs.get("data", {}).get("init_args", {}).get("img_size")
            )
            if _from_defaults:
                lp_img_size = list(_from_defaults)
        except Exception:
            pass

    # ── Build the minimal programmatic injection ─────────────────────────────
    # Contains only things that CAN'T go in a static file:
    #   - encoder_class / encoder_init  (extracted from the seg config at runtime)
    #   - logger name                   (derived from the seg config name at runtime)
    # Everything else (lr, epochs, imagenet path, …) lives in
    # configs/linear_probe_defaults.yaml and is loaded as the first -c.
    #
    # With subclass_mode_model=False, model config keys sit directly under
    # "model" — no "class_path" / "init_args" wrapper needed.
    encoder_init = dict(encoder_cfg.get("init_args") or {})
    encoder_init["img_size"] = lp_img_size  # already overridden above
    injected_cfg: dict = {
        "trainer": {
            "logger": {
                "init_args": {
                    "name": lp_name,
                },
            },
        },
        "model": {
            "encoder_class": encoder_cfg["class_path"],
            "encoder_init": encoder_init,
        },
    }
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(injected_cfg, tmp)
    tmp.close()

    # ── Assemble sys.argv for LightningCLI ───────────────────────────────────
    # Load order (earlier = lower priority):
    #   1. linear_probe_defaults.yaml  — project defaults
    #   2. injected_cfg (tmp file)     — encoder + dynamic run name
    #   3. passthrough_configs         — per-run -c override files
    #   4. remaining_argv              — subcommand + CLI flags (highest priority)
    config_args: list[str] = []
    if _DEFAULTS_FILE.exists():
        config_args += ["-c", str(_DEFAULTS_FILE)]
    config_args += ["-c", tmp.name]
    for cfg_path in passthrough_configs:
        config_args += ["-c", cfg_path]

    # remaining_argv[0] is the subcommand (fit/validate/…).  Config files must
    # come AFTER the subcommand — same as how the user calls main.py:
    #   python main.py fit -c config.yaml
    # Putting -c before the subcommand causes LightningCLI to reject any
    # subcommand-scoped options (e.g. trainer.logger.init_args.resume).
    subcmd = remaining_argv[:1]   # ["fit"] (or empty)
    rest   = remaining_argv[1:]   # extra CLI flags / overrides
    sys.argv = [sys.argv[0]] + subcmd + config_args + rest

    try:
        cli_main(lp_name, lp_subdir, run_ts, output_dir)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
