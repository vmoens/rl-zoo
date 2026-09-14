"""An ordered cooperative relay with one visible baton and a required handoff."""

from __future__ import annotations

from typing import Any

import mujoco
import torch

from ._arena import _ArenaEnv


class MicroDuckRelayEnv(_ArenaEnv):
    """Carry the baton through three checkpoints, handing it over at the middle.

    Duck zero starts with the baton. It visits checkpoint zero, then both
    upright ducks enter the middle handoff zone. Ownership transfers once to
    duck one, which must visit the final checkpoint. Falling preserves baton
    ownership while the duck respawns; it cannot advance a checkpoint.

    Args:
        scene: Optional prebuilt arena MJCF.
        checkpoint_radius: Radius of each checkpoint and the handoff zone.
        handoff_radius: Maximum inter-duck distance for the handoff.
        kwargs: Native arena options. Barriers default to zero height.

    Examples:
        >>> env = MicroDuckRelayEnv(download=True)  # doctest: +SKIP
    """

    GAME = "relay"
    GAME_FEATURE_DIM = 8

    def __init__(
        self,
        scene: str,
        *,
        checkpoint_radius: float = 0.16,
        handoff_radius: float = 0.18,
        **kwargs: Any,
    ):
        if min(checkpoint_radius, handoff_radius) <= 0:
            raise ValueError("Checkpoint and handoff radii must be positive.")
        self.checkpoint_radius = checkpoint_radius
        self.handoff_radius = handoff_radius
        super().__init__(scene, **kwargs)

    def _initial_game_state(self):
        state = super()._initial_game_state()
        state["carrier"] = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        state["checkpoint"] = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        return state

    def _game_features(self, state):
        q, _ = self._ducks(state)
        index = self._game_state["checkpoint"].clamp_max(2)
        target = q.new_tensor([[-0.6, -0.1], [0, 0.2], [0.8, 0.2]])[index.squeeze(-1)]
        carrier = self._game_state["carrier"].squeeze(-1)
        return torch.cat(
            (
                self._relative(target[:, None] - q[..., :2], self._yaw(q[..., 3:7])),
                torch.nn.functional.one_hot(carrier, 2)[:, None].expand(-1, 2, -1),
                torch.nn.functional.one_hot(
                    self._game_state["checkpoint"].squeeze(-1), 4
                )[:, None].expand(-1, 2, -1),
            ),
            -1,
        )

    def _sync_game_scene(self):
        q, _ = self._ducks(self._state_td())
        carrier = int(self._game_state["carrier"].item())
        marker = self._backend.mj_model.body("baton").mocapid[0]
        self._backend._d.mocap_pos[marker] = q[0, carrier, :3].cpu().numpy() + [
            0,
            0,
            0.1,
        ]
        mujoco.mj_forward(self._backend.mj_model, self._backend._d)

    def _step_game(self, state, next_state):
        q, _ = self._ducks(next_state)
        old_q, _ = self._ducks(state)
        stage = int(self._game_state["checkpoint"].item())
        carrier = int(self._game_state["carrier"].item())
        if stage == 3:
            self._game_state["event"].zero_()
            return q.new_zeros(1, 2, 1), torch.ones(
                1, 1, dtype=torch.bool, device=q.device
            )
        target = q.new_tensor([[-0.6, -0.1], [0, 0.2], [0.8, 0.2]])[stage]
        distance = (q[..., :2] - target).norm(dim=-1)
        progress = (old_q[0, carrier, :2] - target).norm() - distance[0, carrier]
        reached = bool(distance[0, carrier] < self.checkpoint_radius) and not bool(
            self._fallen[0, carrier]
        )
        if stage == 1:
            reached = (
                bool((distance < self.checkpoint_radius).all())
                and bool((q[0, 0, :2] - q[0, 1, :2]).norm() < self.handoff_radius)
                and not bool(self._fallen.any())
            )
        if reached:
            self._game_state["checkpoint"] += 1
            if stage == 1:
                self._game_state["carrier"].fill_(1)
        delivered = self._game_state["checkpoint"] == 3
        self._game_state["event"].fill_(float(reached))
        self._game_state["outcome"] = delivered.to(self.dtype)
        reward = 0.5 * progress + 2 * reached + 8 * delivered.to(self.dtype)
        return reward.reshape(1, 1, 1).expand(-1, 2, -1).clone(), delivered
