"""Two-on-two capture the flag with explicit simultaneous-event ordering."""

from __future__ import annotations

from typing import Any

import mujoco
import torch

from ._arena import _ArenaEnv


class MicroDuckCTFEnv(_ArenaEnv):
    """Capture the enemy flag while your own flag is home.

    Upright ducks pick up the enemy flag or return their dropped flag by
    proximity. A carrier drops its flag when fallen or tagged by an upright
    enemy; tagged carriers cannot pick up for ``tag_cooldown_seconds``.
    Resolve drops, friendly returns, enemy pickups, then captures each step.
    A returned flag cannot be picked up in the same step. Nearest eligible
    duck wins simultaneous pickup attempts; agent index breaks exact ties.
    Flag-coloured markers follow ownership. The first capture ends the round.

    Args:
        scene: Optional prebuilt arena MJCF.
        interaction_radius: Pickup, return and capture radius in metres.
        tag_radius: Proximity at which an upright enemy tags a carrier.
        tag_cooldown_seconds: Pickup lockout after being tagged.
        kwargs: Native arena options. This game has two ducks per team and
            defaults to 3,000 physical steps (60 seconds) per round.

    Examples:
        >>> env = MicroDuckCTFEnv(download=True)  # doctest: +SKIP
    """

    GAME = "ctf"
    PLAYERS = 2
    GAME_FEATURE_DIM = 20

    def __init__(
        self,
        scene: str,
        *,
        interaction_radius: float = 0.16,
        tag_radius: float = 0.16,
        tag_cooldown_seconds: float = 1.0,
        **kwargs: Any,
    ):
        if min(interaction_radius, tag_radius, tag_cooldown_seconds) <= 0:
            raise ValueError("Interaction radii and the tag cooldown must be positive.")
        self.interaction_radius = interaction_radius
        self.tag_radius = tag_radius
        self.tag_cooldown_seconds = tag_cooldown_seconds
        kwargs.setdefault("max_episode_steps", 3000)
        super().__init__(scene, **kwargs)

    def _initial_game_state(self):
        state = super()._initial_game_state()
        home = torch.tensor(
            [[[-self.length / 2 + 0.2, 0.0], [self.length / 2 - 0.2, 0.0]]],
            device=self.device,
        )
        state["flag_position"] = home
        state["flag_status"] = torch.zeros(
            1, 2, dtype=torch.long, device=self.device
        )  # home, dropped, carried
        state["carrier"] = torch.full((1, 2), -1, dtype=torch.long, device=self.device)
        state["cooldown"] = torch.zeros(1, 4, device=self.device)
        return state

    def _game_features(self, state):
        q, _ = self._ducks(state)
        relative = self._relative(
            self._game_state["flag_position"][:, None] - q[:, :, None, :2],
            self._yaw(q[..., 3:7])[..., None],
        ).flatten(-2)
        status = torch.nn.functional.one_hot(
            self._game_state["flag_status"], 3
        ).flatten(-2)
        carrier = (
            self._game_state["carrier"][..., None] == torch.arange(4, device=q.device)
        ).flatten(-2)
        return torch.cat(
            (
                relative,
                status[:, None].expand(-1, 4, -1),
                carrier[:, None].expand(-1, 4, -1),
                self._team[None, :, None],
                self._game_state["cooldown"][..., None],
            ),
            -1,
        )

    def _sync_game_scene(self):
        q, _ = self._ducks(self._state_td())
        for flag in range(2):
            carrier = int(self._game_state["carrier"][0, flag])
            marker = self._backend.mj_model.body(f"flag_{flag}").mocapid[0]
            xy = self._game_state["flag_position"][0, flag].cpu().numpy()
            self._backend._d.mocap_pos[marker, :2] = xy
            self._backend._d.mocap_pos[marker, 2] = (
                float(q[0, carrier, 2]) + 0.12 if carrier >= 0 else 0.15
            )
        mujoco.mj_forward(self._backend.mj_model, self._backend._d)

    def _step_game(self, state, next_state):
        q, _ = self._ducks(next_state)
        xy = q[0, :, :2]
        gs = self._game_state
        gs["cooldown"] = (
            gs["cooldown"] - self.frame_skip * self._backend.timestep
        ).clamp_min(0)
        live = ~self._fallen[0]
        distance = torch.cdist(xy, xy)
        home = xy.new_tensor([[-self.length / 2 + 0.2, 0], [self.length / 2 - 0.2, 0]])
        # Compute tags from one snapshot, so two opposing carriers can both drop.
        drops = []
        for flag in range(2):
            carrier = int(gs["carrier"][0, flag])
            drop = False
            if carrier >= 0:
                gs["flag_position"][0, flag] = xy[carrier]
                tagged = (
                    (distance[carrier] <= self.tag_radius)
                    & live
                    & (self._team != self._team[carrier])
                ).any()
                drop = bool(tagged) or not bool(live[carrier])
            drops.append(drop)
        for flag, drop in enumerate(drops):
            if drop:
                carrier = int(gs["carrier"][0, flag])
                gs["cooldown"][0, carrier] = self.tag_cooldown_seconds
                gs["carrier"][0, flag] = -1
                gs["flag_status"][0, flag] = 1
        returned = [False, False]
        for flag in range(2):
            if int(gs["flag_status"][0, flag]) != 1:
                continue
            near = (xy - gs["flag_position"][0, flag]).norm(
                dim=-1
            ) <= self.interaction_radius
            if bool((near & live & (self._team == flag)).any()):
                gs["flag_position"][0, flag] = home[flag]
                gs["flag_status"][0, flag] = 0
                returned[flag] = True
        for flag in range(2):
            if int(gs["carrier"][0, flag]) >= 0 or returned[flag]:
                continue
            distances = (xy - gs["flag_position"][0, flag]).norm(dim=-1)
            eligible = live & (self._team != flag) & (gs["cooldown"][0] == 0)
            distances = distances.masked_fill(~eligible, float("inf"))
            nearest = int(distances.argmin())
            if float(distances[nearest]) <= self.interaction_radius:
                gs["carrier"][0, flag] = nearest
                gs["flag_status"][0, flag] = 2
                gs["flag_position"][0, flag] = xy[nearest]
        scores = q.new_zeros(2)
        for team in range(2):
            carrier = int(gs["carrier"][0, 1 - team])
            if (
                carrier >= 0
                and bool(live[carrier])
                and int(gs["flag_status"][0, team]) == 0
            ):
                scores[team] = (
                    xy[carrier] - home[team]
                ).norm() <= self.interaction_radius
        fresh = gs["outcome"] == 0
        scored = scores.sum() > 0
        gs["event"] = (scores.sum() * fresh).to(self.dtype)
        gs["outcome"] = torch.where(
            scored, (scores[0] - scores[1]).reshape(1, 1), gs["outcome"]
        )
        reward = (
            torch.where(self._team == 0, 1.0, -1.0)[None, :, None]
            * (scores[0] - scores[1])
            * 10
            * fresh[..., None]
        )
        return reward, scored.reshape(1, 1)
