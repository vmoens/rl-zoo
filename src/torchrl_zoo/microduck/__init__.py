"""MicroDuck games using TorchRL skills and trainers."""

from __future__ import annotations

from .games.ctf import MicroDuckCTFEnv
from .games.football import MicroDuckFootballEnv
from .games.hide_and_seek import MicroDuckHideAndSeekEnv
from .games.pushing import MicroDuckPushingEnv
from .games.relay import MicroDuckRelayEnv
from .games.tag import MicroDuckTagEnv

__all__ = [
    "MicroDuckFootballEnv",
    "MicroDuckTagEnv",
    "MicroDuckCTFEnv",
    "MicroDuckPushingEnv",
    "MicroDuckRelayEnv",
    "MicroDuckHideAndSeekEnv",
]
