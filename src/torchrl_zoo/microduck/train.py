"""Common Hydra entry point for the MicroDuck games."""

from __future__ import annotations

import signal

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchrl import torchrl_logger


class _PilotBudgetExpired(TimeoutError):
    pass


def _budget_expired(signum, frame):
    raise _PilotBudgetExpired("The game training wall-clock cap has expired.")


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:
    """Build the selected game recipe and run TorchRL's trainer."""
    torch.set_num_threads(config.runtime.num_threads)
    trainer = instantiate(config.trainer, recipe=config, _recursive_=False)
    seconds = trainer.game_hooks.pilot_seconds
    previous_handler = None
    try:
        if seconds is not None:
            remaining = seconds - trainer.game_hooks.elapsed_seconds
            if remaining <= 0:
                raise ValueError(
                    "This run has consumed its pilot budget; choose a further budget before resuming."
                )
            previous_handler = signal.signal(signal.SIGALRM, _budget_expired)
            signal.setitimer(signal.ITIMER_REAL, remaining)
        trainer.train()
    except _PilotBudgetExpired:
        # A hard cap can interrupt an update. Retain the last complete periodic
        # checkpoint instead of exporting partially updated optimizer state.
        torchrl_logger.info("Pilot cap reached; retain the last completed checkpoint.")
    finally:
        if previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        trainer.game_hooks.close()
        trainer.collector.shutdown()


if __name__ == "__main__":
    main()
