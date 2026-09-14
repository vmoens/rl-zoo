"""Two ducks jointly deliver a physical box and let it settle."""

from __future__ import annotations

from typing import Any

import torch

from ._arena import _ArenaEnv


class MicroDuckPushingEnv(_ArenaEnv):
    """Cooperative box delivery with shared progress and a settling requirement.

    Args:
        scene: Optional prebuilt arena MJCF.
        delivery_radius: Maximum planar box-centre distance from the target.
        settle_speed: Maximum box linear and angular speed during delivery.
        dwell_seconds: Continuous time in the target below the speed limits.
        kwargs: Native arena options. ``arena.box_mass`` sets box mass.

    Examples:
        >>> env = MicroDuckPushingEnv(download=True)  # doctest: +SKIP
    """

    GAME = "pushing"
    GAME_FEATURE_DIM = 11

    def __init__(
        self,
        scene: str,
        *,
        delivery_radius: float = 0.15,
        settle_speed: float = 0.03,
        dwell_seconds: float = 0.5,
        **kwargs: Any,
    ):
        if min(delivery_radius, settle_speed, dwell_seconds) <= 0:
            raise ValueError(
                "Delivery radius, speed threshold and dwell must be positive."
            )
        self.delivery_radius = delivery_radius
        self.settle_speed = settle_speed
        self.dwell_seconds = dwell_seconds
        super().__init__(scene, **kwargs)

    def _initial_game_state(self):
        state = super()._initial_game_state()
        state["dwell"] = torch.zeros(1, 1, device=self.device)
        return state

    def _game_features(self, state):
        q, _ = self._ducks(state)
        box = state["qpos"][..., -7:]
        box_v = state["qvel"][..., -6:]
        yaw = self._yaw(q[..., 3:7])
        return torch.cat(
            (
                self._relative(box[:, None, :2] - q[..., :2], yaw),
                self._relative(box_v[:, None, :2].expand(-1, 2, -1), yaw),
                self._relative(q.new_tensor([0.8, 0]) - q[..., :2], yaw),
                box[:, None, 3:7].expand(-1, 2, -1),
                (self._game_state["dwell"] / self.dwell_seconds)[:, None].expand(
                    -1, 2, -1
                ),
            ),
            -1,
        )

    def _step_game(self, state, next_state):
        box = next_state["qpos"][..., -7:]
        velocity = next_state["qvel"][..., -6:]
        target = box.new_tensor([0.8, 0])
        distance = (box[..., :2] - target).norm(dim=-1, keepdim=True)
        previous = (state["qpos"][..., -7:-5] - target).norm(dim=-1, keepdim=True)
        upright = 1 - 2 * box[..., 4:6].square().sum(-1, keepdim=True) > 0.9
        settled = (
            (distance < self.delivery_radius)
            & upright
            & (velocity[..., :3].norm(dim=-1, keepdim=True) < self.settle_speed)
            & (velocity[..., 3:].norm(dim=-1, keepdim=True) < self.settle_speed)
        )
        self._game_state["dwell"] = torch.where(
            settled,
            self._game_state["dwell"] + self.frame_skip * self._backend.timestep,
            0,
        )
        delivered = self._game_state["dwell"] + 1e-6 >= self.dwell_seconds
        event = delivered & (self._game_state["outcome"] == 0)
        self._game_state["outcome"] = torch.where(
            delivered, 1.0, self._game_state["outcome"]
        )
        self._game_state["event"] = event.to(self.dtype)
        reward = 5 * (previous - distance) + 10 * event
        return reward[:, None].expand(-1, 2, -1).clone(), delivered
