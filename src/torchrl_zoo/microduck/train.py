"""Common Hydra entry point for the MicroDuck games."""

from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(config: DictConfig) -> None:
    """Build the selected game recipe and run TorchRL's trainer."""
    trainer = instantiate(config.trainer, recipe=config, _recursive_=False)
    try:
        trainer.train()
    finally:
        trainer.game_hooks.close()
        trainer.collector.shutdown()


if __name__ == "__main__":
    main()
