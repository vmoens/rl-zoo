"""Team tag: upright proximity captures with alternating seeker roles."""

from __future__ import annotations

from typing import Any, Literal

import torch
from tensordict import TensorDictBase

from ._arena import _ArenaEnv


class MicroDuckTagEnv(_ArenaEnv):
    """Two teams play one round of tag using a frozen locomotion controller.

    A seeker captures a live, upright runner within ``tag_radius``. A runner
    can be captured once and then stays inactive until the round ends. All
    runners captured ends the round; the time limit awards survival to them.
    Random roles alternate each round; spawns mirror every two rounds.

    Args:
        scene: Optional already-built arena MJCF, supplied by the arena factory.
        role: Team assigned seeker duty, or alternating roles with a seeded start.
        tag_radius: Planar torso-to-torso capture distance in metres.
        kwargs: Native arena options including ``players_per_team`` (1 or 2),
            ``microduck_root``, ``download``, ``num_envs`` and ``max_episode_steps``.

    Examples:
        >>> env = MicroDuckTagEnv(download=True, players_per_team=2)  # doctest: +SKIP
        >>> rollout = env.rollout(10)  # doctest: +SKIP
    """

    GAME = "tag"
    GAME_FEATURE_DIM = 2

    def __init__(
        self,
        scene: str,
        *,
        role: Literal["random", "blue", "red"] = "random",
        tag_radius: float = 0.16,
        **kwargs: Any,
    ):
        if role not in ("random", "blue", "red") or not 0 < tag_radius < 1:
            raise ValueError(
                "role must be random/blue/red and tag_radius must be in (0, 1)."
            )
        self.role = role
        self.tag_radius = tag_radius
        super().__init__(scene, **kwargs)

    def _initial_game_state(self):
        state = super()._initial_game_state()
        state["captured"] = torch.zeros(
            1, self.num_agents, dtype=torch.bool, device=self.device
        )
        state["seeker_team"] = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        return state

    def _reset_game(self):
        if self.role == "random":
            self._game_state["seeker_team"] = (
                self._game_state["round"] + self.rng.initial_seed()
            ) % 2
        else:
            self._game_state["seeker_team"].fill_(0 if self.role == "blue" else 1)

    def _game_features(self, state):
        seekers = self._team[None] == self._game_state["seeker_team"]
        return torch.stack(
            (seekers.to(self.dtype), torch.zeros_like(seekers, dtype=self.dtype)), -1
        )

    def _captures(self, q: torch.Tensor) -> torch.Tensor:
        seekers = self._team[None] == self._game_state["seeker_team"]
        upright_live = self._game_state["active"] & ~self._fallen
        nearby = torch.cdist(q[..., :2], q[..., :2]) <= self.tag_radius
        eligible = (
            nearby
            & (seekers & upright_live)[..., None]
            & (~seekers & upright_live)[:, None]
        )
        return eligible.any(dim=1)

    def _step_game(self, state: TensorDictBase, next_state: TensorDictBase):
        old_q, _ = self._ducks(state)
        q, _ = self._ducks(next_state)
        seekers = self._team[None] == self._game_state["seeker_team"]
        runners = ~seekers & ~self._game_state["captured"]
        old_distance = torch.cdist(old_q[..., :2], old_q[..., :2]).masked_fill(
            ~runners[:, None], float("inf")
        )
        distance = torch.cdist(q[..., :2], q[..., :2]).masked_fill(
            ~runners[:, None], float("inf")
        )
        progress = old_distance.min(-1).values - distance.min(-1).values
        progress = torch.nan_to_num(progress)[seekers].mean() * 0.2
        captures = self._captures(q) & ~self._game_state["captured"]
        self._game_state["captured"] |= captures
        self._game_state["active"] &= ~captures
        count = captures.sum().to(self.dtype)
        self._game_state["event"].fill_(count)
        sign = torch.where(seekers, 1.0, -1.0)
        reward = sign[..., None] * (progress + 10 * count / self.players_per_team)
        all_captured = (self._game_state["captured"] | seekers).all(-1, keepdim=True)
        expired = (self._step_count >= self.max_episode_steps)[:, None] & ~all_captured
        survival = expired & (self._game_state["outcome"] == 0)
        reward -= sign[..., None] * survival[..., None] * 10
        blue_seeks = self._game_state["seeker_team"] == 0
        self._game_state["outcome"] = torch.where(
            all_captured | expired,
            torch.where(all_captured == blue_seeks, 1.0, -1.0),
            self._game_state["outcome"],
        )
        return reward, all_captured
