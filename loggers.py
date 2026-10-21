import logging
import os
import socket
from typing import ClassVar, Optional, Union

from lightning.pytorch.loggers.wandb import WandbLogger


class OfflineFallbackWandbLogger(WandbLogger):
    """WandbLogger that automatically falls back to offline mode when wandb.ai
    is unreachable (e.g. on compute nodes without external internet access).

    Before the first ``wandb.init()`` call a fast TCP probe to
    ``api.wandb.ai:443`` is performed.  If the probe fails (e.g. the node has
    no external network — ``ENETUNREACH`` returns instantly), the run is
    initialised with ``mode="offline"`` immediately, bypassing the default
    90-second init timeout entirely.

    Use ``wandb sync <wandb-run-dir>`` after the job to upload the run.
    """

    _WANDB_HOST: ClassVar[str] = "api.wandb.ai"
    _WANDB_PORT: ClassVar[int] = 443
    _CONNECT_TIMEOUT: ClassVar[float] = 5.0  # seconds; ENETUNREACH nodes fail instantly

    def __init__(
        self,
        resume: Optional[Union[bool, str]] = None,
        **kwargs,
    ):
        """Expose ``resume`` explicitly so jsonargparse / LightningCLI accept it.

        ``WandbLogger.__init__`` does not declare ``resume`` as a parameter --
        it is forwarded into ``wandb.init()`` through ``**kwargs``. jsonargparse
        cannot introspect ``**kwargs``, so any YAML setting ``resume: allow``
        is rejected with "Key 'resume' is not expected". Declaring it here puts
        it on the subclass signature, and we hand it back to the parent.
        """
        if resume is not None:
            kwargs["resume"] = resume
        super().__init__(**kwargs)

    @classmethod
    def _has_wandb_connectivity(cls) -> bool:
        """Return True if api.wandb.ai:443 is reachable within _CONNECT_TIMEOUT."""
        try:
            with socket.create_connection(
                (cls._WANDB_HOST, cls._WANDB_PORT),
                timeout=cls._CONNECT_TIMEOUT,
            ):
                return True
        except OSError:
            return False

    def _force_offline(self, reason: str) -> None:
        """Switch this logger instance to offline mode and emit a warning."""
        logging.warning(
            "\n*** wandb: %s ***\n"
            "    Switching to OFFLINE mode.\n"
            "    Run `wandb sync <wandb-run-dir>` after the job to upload logs.\n",
            reason,
        )
        os.environ["WANDB_MODE"] = "offline"
        self._wandb_init["mode"] = "offline"
        self._experiment = None  # ensure super().experiment re-inits cleanly

    @property
    def experiment(self):
        if self._experiment is not None:
            return self._experiment
        # Fast pre-flight: avoids the 90-second wandb timeout on offline nodes.
        if not self._has_wandb_connectivity():
            self._force_offline(
                f"no connectivity to {self._WANDB_HOST}:{self._WANDB_PORT}"
            )
        return super().experiment
