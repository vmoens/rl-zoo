"""Hide-and-seek with a movable cover and sustained camera-visible discovery."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np
import torch

from .tag import MicroDuckTagEnv


class MicroDuckHideAndSeekEnv(MicroDuckTagEnv):
    """Hiders prepare behind a movable cover before seekers can move and discover them.

    The arena carries a single 25 cm free box (80 g) the hider rolls between
    themselves and the seeker to block the head camera's line of sight. Both
    roles read the cover's position and yaw in their state observation.

    Discovery requires one upright seeker to see an upright hider continuously
    for ``discovery_seconds``. Visibility uses the mounted head camera's square
    frustum, a range limit, and a MuJoCo ray whose first hit must belong to the
    hider. Occlusion or leaving the frustum resets that seeker's timer.
    Captured hiders remain inactive through the round, as in tag.

    Args:
        scene: Optional prebuilt arena MJCF.
        preparation_seconds: Time allowed for hiders to move before seeking.
        discovery_seconds: Required continuous visibility from one seeker.
        discovery_range: Maximum camera-to-hider range in metres.
        kwargs: Tag role and native arena options.

    Examples:
        >>> env = MicroDuckHideAndSeekEnv(download=True)  # doctest: +SKIP
    """

    GAME = "hide_and_seek"
    GAME_FEATURE_DIM = 6

    def __init__(
        self,
        scene: str,
        *,
        preparation_seconds: float = 2.0,
        discovery_seconds: float = 0.5,
        discovery_range: float = 1.2,
        **kwargs: Any,
    ):
        if preparation_seconds < 0 or min(discovery_seconds, discovery_range) <= 0:
            raise ValueError(
                "Preparation must be nonnegative; discovery duration and range positive."
            )
        self.preparation_seconds = preparation_seconds
        self.discovery_seconds = discovery_seconds
        self.discovery_range = discovery_range
        super().__init__(scene, **kwargs)
        if (
            preparation_seconds
            >= self.max_episode_steps * self.frame_skip * self._backend.timestep
        ):
            raise ValueError("The round must leave time to seek after preparation.")

    def _initial_game_state(self):
        state = super()._initial_game_state()
        state["seen_steps"] = torch.zeros(
            1, self.num_agents, self.num_agents, dtype=torch.long, device=self.device
        )
        return state

    def _reset_game(self):
        super()._reset_game()
        if self.preparation_seconds > 0:
            self._game_state["active"] &= (
                self._team[None] != self._game_state["seeker_team"]
            )

    def _game_features(self, state):
        # Inherited TagEnv fills features[..., 0] with the seeker-team bit and
        # leaves features[..., 1] zero; extend to
        #   [seeker, prep_left, cover_x, cover_y, cover_yaw_cos, cover_yaw_sin]
        features = torch.zeros(
            1, self.num_agents, self.GAME_FEATURE_DIM, device=self.device
        )
        seekers = self._team[None] == self._game_state["seeker_team"]
        features[..., 0] = seekers.to(self.dtype)
        elapsed = self._step_count * self.frame_skip * self._backend.timestep
        features[..., 1] = (
            1 - elapsed / max(self.preparation_seconds, 1e-6)
        ).clamp_min(0).expand(self.num_agents)
        cover = state["qpos"][:, self.num_agents * self.DUCK_NQ :]
        # Normalize against the same half-extent used for duck positions so
        # the magnitude is comparable across the arena.
        features[..., 2:4] = cover[:, :2] / cover.new_tensor(
            [self.length / 2, self.width / 2]
        )
        yaw = self._yaw(cover[:, 3:7])
        features[..., 4] = yaw.cos()
        features[..., 5] = yaw.sin()
        return features

    def _captures(self, q):
        model, data = self._backend.mj_model, self._backend._d
        seekers = self._team == self._game_state["seeker_team"].item()
        live = self._game_state["active"][0] & ~self._fallen[0]
        visible = torch.zeros_like(self._game_state["seen_steps"], dtype=torch.bool)
        for seeker in torch.where(seekers & live)[0].tolist():
            team, member = divmod(seeker, self.players_per_team)
            camera = model.camera(
                f"{'blue' if team == 0 else 'red'}{member}/head_camera"
            ).id
            rotation = data.cam_xmat[camera].reshape(3, 3)
            origin = data.cam_xpos[camera]
            tangent = math.tan(math.radians(float(model.cam_fovy[camera])) / 2)
            for target in torch.where(~seekers & live)[0].tolist():
                vector = q[0, target, :3].cpu().numpy().astype(np.float64) - origin
                distance = np.linalg.norm(vector)
                local = rotation.T @ vector
                if not (
                    0 < distance <= self.discovery_range
                    and local[2] < 0
                    and max(abs(local[0]), abs(local[1])) <= -local[2] * tangent
                ):
                    continue
                hit = np.full(1, -1, dtype=np.int32)
                mujoco.mj_ray(
                    model,
                    data,
                    origin,
                    vector / distance,
                    None,
                    True,
                    int(model.cam_bodyid[camera]),
                    hit,
                )
                target_joint = np.flatnonzero(
                    model.jnt_qposadr == target * self.DUCK_NQ
                )[0]
                target_root = model.jnt_bodyid[target_joint]
                visible[0, seeker, target] = bool(
                    hit[0] >= 0
                    and model.body_rootid[model.geom_bodyid[hit[0]]] == target_root
                )
        self._game_state["seen_steps"] = torch.where(
            visible, self._game_state["seen_steps"] + 1, 0
        )
        required = math.ceil(
            self.discovery_seconds / (self.frame_skip * self._backend.timestep)
        )
        return (self._game_state["seen_steps"] >= required).any(1)

    def _step_game(self, state, next_state):
        elapsed = (
            float(self._step_count.item()) * self.frame_skip * self._backend.timestep
        )
        if elapsed < self.preparation_seconds:
            self._game_state["event"].zero_()
            return torch.zeros(1, self.num_agents, 1, device=self.device), torch.zeros(
                1, 1, dtype=torch.bool, device=self.device
            )
        self._game_state["active"] = ~self._game_state["captured"]
        return super()._step_game(state, next_state)
