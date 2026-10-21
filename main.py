# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from PyTorch Lightning,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------


import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from types import MethodType
from gitignore_parser import parse_gitignore
import logging
import torch
import warnings
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import Logger as _LightningLogger
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from lightning.pytorch.utilities import rank_zero_only

from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule


def _to_scalar(v):
    """Coerce tensor / numpy scalar metric values to JSON-friendly types."""
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return str(v)
    return v


class JsonlLogger(_LightningLogger):
    """Append-only JSONL metrics logger.

    Mirrors every ``log_metrics`` dispatch that the trainer sends to its
    other loggers (e.g. WandbLogger) by appending one JSON object per call
    to ``<save_dir>/<filename>``. Rank-zero only under DDP.
    """

    def __init__(self, save_dir, filename="train.jsonl"):
        super().__init__()
        self._save_dir = str(save_dir)
        self._filename = filename
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
        # Hyperparams aren't part of the per-step metrics stream; skip.
        return



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
        is_val_check_batch = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (
                self.batch_idx + 1
            ) % self.trainer.val_check_batch == 0
        else:
            # added below to check val based on global steps instead of batches in case of iteration based val check and gradient accumulation
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()

    return is_val_check_batch


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, jsonl_save_dir=None, **kwargs):
        self._jsonl_save_dir = jsonl_save_dir
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.capture_scalar_outputs = True
        # Falling back to eager is fine, but it must not be silent: a backbone
        # that never compiles just runs slower with no signal otherwise.
        torch._dynamo.config.suppress_errors = True
        logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
        warnings.filterwarnings(
            "ignore",
            message=r".*It is recommended to use .* when logging on epoch level in distributed setting to accumulate the metric across devices.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^The ``compute`` method of metric PanopticQuality was called before the ``update`` method.*",
        )
        warnings.filterwarnings(
            "ignore", message=r"^Grad strides do not match bucket view strides.*"
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*functools.partial will be a method descriptor in future Python versions*",
        )

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile_disabled", action="store_true")
        parser.add_argument("--checkpoint_monitor", type=str, default="metrics/val_iou_all")
        parser.add_argument("--output_dir", type=str, default="outputs")
        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )

        parser.link_arguments(
            "data.init_args.stuff_classes", "model.init_args.stuff_classes"
        )

        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.network.init_args.img_size"
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.encoder.init_args.img_size",
        )

        # parser.link_arguments(
        #     "model.init_args.ckpt_path",
        #     "model.init_args.network.init_args.encoder.init_args.ckpt_path",
        # )

    @staticmethod
    def _collect_git_state():
        """Best-effort snapshot of the working-tree git state.

        Returns a dict with ``commit``, ``branch``, ``dirty``, ``status``,
        ``remote``, and ``describe``. Any field that can't be resolved is
        ``None``. Never raises — this is metadata, not a precondition.
        """
        def _run(args):
            try:
                out = subprocess.run(
                    ["git", *args],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if out.returncode != 0:
                    return None
                return out.stdout.strip() or None
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
        """Write the fully-resolved run config to ``<save_dir>/hyperparams.json``.

        Captures every parameter LightningCLI knows about *after* linking
        (data → model), CLI overrides, and defaults have been applied —
        i.e. the exact set of values the run will use. Also records the
        invocation (start command, cwd, timestamp) and git state of the
        working tree so each run is reproducible from this single file.
        Rank-zero only.
        """
        if save_dir is None:
            return
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "hyperparams.json")
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
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str, sort_keys=False)

    def fit(self, model, **kwargs):
        if hasattr(self.trainer.logger.experiment, "log_code"):
            is_gitignored = parse_gitignore(".gitignore")
            def include_fn(path):
                return path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        if self._jsonl_save_dir is not None:
            # Attach alongside the user-configured logger (e.g. WandbLogger)
            # so both receive the same log_metrics() dispatches.
            self.trainer.loggers.append(JsonlLogger(save_dir=self._jsonl_save_dir))
            # Document every parameter this run is using.
            self._dump_hyperparams(self._jsonl_save_dir, self.config)

        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        if not self.config[self.config["subcommand"]]["compile_disabled"]:
            model = torch.compile(model)

        self.trainer.fit(model, **kwargs)


def _get_run_timestamp() -> str:
    """Return a yyyy-mm-dd-HH-MM-SS string that is identical across all DDP ranks.

    Under SLURM, ``SLURM_JOB_START_TIME`` is a Unix epoch integer set
    identically on every task in the same job, so all ranks derive the same
    string regardless of minor scheduling skew.  Outside SLURM (local runs,
    plain ``torchrun``) all worker processes start within the same second, so
    ``datetime.now()`` at one-second resolution is consistent in practice.
    """
    slurm_start = os.environ.get("SLURM_JOB_START_TIME")
    if slurm_start:
        try:
            return datetime.fromtimestamp(int(slurm_start)).strftime(
                "%Y-%m-%d-%H-%M-%S"
            )
        except (ValueError, OSError):
            pass
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


class _QuietValProgressBar(TQDMProgressBar):
    """One persisted log line per bar instead of one per validation step.

    Lightning gives the validation bar ``position=1`` during fit, and tqdm
    implements a non-zero position by emitting real newlines on every refresh —
    in a non-tty log that is a line per step. ``pos=0`` puts it on the train
    bar's single ``\r``-rewritten line; ``leave=True`` makes it end that line
    with a newline when it closes, so the finished bar stays and the train bar
    resumes below it.
    """

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        try:
            bar.pos, bar.leave = 0, True
        except Exception:  # never let cosmetics kill a run
            pass
        return bar


def cli_main(name, subdir, run_ts, checkpoint_monitor="metrics/val_iou_all", output_dir="outputs"):
    save_dir = f"{output_dir}/{subdir}/{run_ts}/"
    LightningCLI(
        LightningModule,
        LightningDataModule,
        jsonl_save_dir=save_dir,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=True,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=3),
                _QuietValProgressBar(),
                ModelCheckpoint(dirpath=save_dir,
                                filename=f"{name}",
                                monitor=checkpoint_monitor,
                                mode="max",               # "min" for loss, "max" for accuracy
                                save_top_k=1,             # save best checkpoint
                                save_last=False)

            ],
            "devices": 'auto',
            "gradient_clip_val": 0.01,
            "gradient_clip_algorithm": "norm",
        },
    )


def _config_to_subdir(config_path):
    """Mirror a config file's location as an output subdirectory.

    Strips a leading ``configs/`` segment and the ``.yaml``/``.yml`` suffix.
    Returns ``None`` if nothing usable remains.

    Example:
        ``configs/ade20k/semantic/size_vs_res/eva02/eva02_base_384.yaml``
        → ``ade20k/semantic/size_vs_res/eva02/eva02_base_384``
    """
    parts = list(Path(config_path).with_suffix("").parts)
    if parts and parts[0] == "configs":
        parts = parts[1:]
    if not parts:
        return None
    return "/".join(parts)


def _resolve_run_info():
    """Resolve (name, subdir, checkpoint_monitor, output_dir) from -c configs and CLI overrides.

    name:               from ``trainer.logger.init_args.name`` in the last config that
                        sets it, overridden by ``--trainer.logger.init_args.name`` on
                        the CLI. Defaults to ``"default"``.
    subdir:             from the last ``-c`` config path, mirroring its directory layout
                        (see :func:`_config_to_subdir`). Falls back to ``name`` when no
                        config path is supplied.
    checkpoint_monitor: from ``checkpoint_monitor`` in any config (last value wins),
                        overridden by ``--checkpoint_monitor`` on the CLI.
                        Defaults to ``"metrics/val_iou_all"`` when not set.
                        Use ``"metrics/val_pq_all"`` for panoptic configs and
                        ``"metrics/val_ap_all"`` for instance configs.
    output_dir:         root for checkpoints and logs, from ``--output_dir`` on the CLI.
                        Defaults to ``"outputs"``.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-c", "--config", action="append", default=[])
    pre_parser.add_argument(
        "--trainer.logger.init_args.name", dest="logger_name", default=None
    )
    pre_parser.add_argument("--checkpoint_monitor", dest="checkpoint_monitor", default=None)
    pre_parser.add_argument("--output_dir", dest="output_dir", default="outputs")
    temp_args, _ = pre_parser.parse_known_args()

    name = "default"
    checkpoint_monitor = "metrics/val_iou_all"
    for config_path in temp_args.config:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            cfg_name = (
                cfg.get("trainer", {})
                .get("logger", {})
                .get("init_args", {})
                .get("name")
            )
            if cfg_name is not None:
                name = cfg_name
            cfg_monitor = cfg.get("checkpoint_monitor")
            if cfg_monitor is not None:
                checkpoint_monitor = cfg_monitor
        except (OSError, AttributeError):
            pass
    if temp_args.logger_name is not None:
        name = temp_args.logger_name
    if temp_args.checkpoint_monitor is not None:
        checkpoint_monitor = temp_args.checkpoint_monitor

    subdir = None
    for config_path in reversed(temp_args.config):
        subdir = _config_to_subdir(config_path)
        if subdir:
            break
    if subdir is None:
        subdir = name

    return name, subdir, checkpoint_monitor, temp_args.output_dir


if __name__ == "__main__":
    name, subdir, checkpoint_monitor, output_dir = _resolve_run_info()
    run_ts = _get_run_timestamp()
    print(f"Running experiment: {name}")
    print(f"Output directory:   {output_dir}/{subdir}/{run_ts}/")
    print(f"Checkpoint monitor: {checkpoint_monitor}")
    cli_main(name, subdir, run_ts, checkpoint_monitor, output_dir)
