"""Native MuJoCo arena and duck observations shared by the new games."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import ClassVar

import mujoco
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Bounded, Composite, Unbounded
from torchrl.envs import MicroDuckEnv
from torchrl.envs.custom.mujoco.base import MujocoEnv, _MujocoMeta
from torchrl.envs.custom.mujoco.microduck import (
    _body_frame_linear_velocity,
    _low_cost_collision_scene,
    _projected_gravity,
)

from ..sensors import _GameSensors
from ._scene import (
    _attach_duck,
    _camera_axes,
    _quaternion_product,
    _share_meshes,
    _yaw_quaternion,
)


def _arena_scene(
    robot_path: Path,
    *,
    game: str,
    players: int,
    length: float = 3.0,
    width: float = 2.0,
    box_mass: float = 0.05,
    box_size: tuple[float, float, float] = (0.14, 0.2, 0.1),
    barrier_height: float = 0.0,
) -> str:
    if (
        len(box_size) != 3
        or players < 1
        or min(length, width, box_mass, *box_size) <= 0
        or barrier_height < 0
    ):
        raise ValueError(
            "Arena counts/dimensions/mass must be positive; barrier height nonnegative."
        )
    if (game == "ctf" and players != 2) or (
        game in ("pushing", "relay") and players != 1
    ):
        raise ValueError(
            "CTF uses two ducks per team; pushing and relay use two ducks total."
        )
    if game in ("tag", "hide_and_seek") and players not in (1, 2):
        raise ValueError("Tag and hide-and-seek support one or two ducks per team.")
    with _low_cost_collision_scene(robot_path.resolve()) as robot_scene:
        robot = mujoco.MjSpec.from_file(str(robot_scene))
        stand = next(key for key in robot.keys if key.name == "STAND")
        if len(stand.qpos) != 21 or len(stand.ctrl) != 14:
            raise ValueError(
                "The arena needs a MicroDuck with a free root and 14 actuators."
            )
        spec = mujoco.MjSpec()
        spec.modelname = f"microduck_{game}"
        spec.compiler.degree = False
        spec.compiler.autolimits = True
        for flag in (
            "fitaabb",
            "inertiafromgeom",
            "balanceinertia",
            "boundmass",
            "boundinertia",
            "settotalmass",
        ):
            setattr(spec.compiler, flag, getattr(robot.compiler, flag))
        spec.option.timestep = robot.option.timestep
        meshdir = Path(robot.meshdir)
        spec.meshdir = str(
            meshdir if meshdir.is_absolute() else Path(robot.modelfiledir) / meshdir
        )
        world = spec.worldbody
        world.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[length, width, 0.05],
            rgba=[0.22, 0.38, 0.3, 1],
            conaffinity=3,
        )
        world.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
        for axis, half in enumerate((length / 2, width / 2)):
            for sign in (-1, 1):
                pos = [0.0, 0.0, 0.1]
                pos[axis] = sign * (half + 0.02)
                size = [length / 2 + 0.04, width / 2 + 0.04, 0.1]
                size[axis] = 0.02
                world.add_geom(
                    name=f"wall_{axis}_{sign}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=pos,
                    size=size,
                    conaffinity=3,
                    rgba=[0.6, 0.65, 0.7, 0.7],
                )
        camera_pos = (0.0, -width / 2 - length / 2, length / 2)
        world.add_camera(
            name="broadcast",
            pos=camera_pos,
            xyaxes=_camera_axes(camera_pos, (0, 0, 0.1)),
            fovy=50,
        )
        world.add_camera(
            name="topdown", pos=[0, 0, length], xyaxes=[1, 0, 0, 0, 1, 0], fovy=50
        )
        key_qpos, key_ctrl = [], []
        for team in range(2):
            sign = 1 - 2 * team
            color = [0.2, 0.45, 0.95, 1] if team == 0 else [0.95, 0.25, 0.2, 1]
            for index in range(players):
                x = -sign * length * 0.25
                y = (index - (players - 1) / 2) * 0.4
                yaw = 0 if team == 0 else math.pi
                if game == "pushing":
                    x, y, yaw = (
                        -box_size[0] / 2 - 0.21,
                        (team - 0.5) * box_size[1] * 2 / 3,
                        0,
                    )
                elif game == "relay":
                    x, y, yaw = (-1.0, -0.1, 0) if team == 0 else (0.0, 0.3, 0)
                quat = _yaw_quaternion(yaw)
                frame = world.add_frame(pos=[x, y, 0], quat=quat)
                prefix = f"{'blue' if team == 0 else 'red'}{index}/"
                _attach_duck(spec, robot_scene, prefix, frame, color)
                key_qpos += [x, y, float(stand.qpos[2])]
                key_qpos += _quaternion_product(quat, list(stand.qpos)[3:7])
                key_qpos += list(stand.qpos)[7:]
                key_ctrl += list(stand.ctrl)
        _share_meshes(spec)
        if game == "hide_and_seek":
            for index, (x, y) in enumerate(((0.0, 0.0), (-0.45, 0.5), (0.45, -0.5))):
                world.add_geom(
                    name=f"cover_{index}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[x, y, 0.2],
                    size=[0.1, 0.25, 0.2],
                    rgba=[0.65, 0.55, 0.35, 1],
                    conaffinity=3,
                )
        if game == "ctf":
            for team in range(2):
                marker = world.add_body(
                    name=f"flag_{team}",
                    mocap=True,
                    pos=[(2 * team - 1) * (length / 2 - 0.2), 0, 0.15],
                )
                marker.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=[0.025, 0.12, 0],
                    contype=0,
                    conaffinity=0,
                    rgba=[0.2, 0.45, 0.95, 1] if team == 0 else [0.95, 0.25, 0.2, 1],
                )
        if game == "pushing":
            box = world.add_body(name="box", pos=[0, 0, box_size[2] / 2 + 0.005])
            box.add_joint(name="box_free", type=mujoco.mjtJoint.mjJNT_FREE)
            box.add_geom(
                name="box_geom",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[dimension / 2 for dimension in box_size],
                mass=box_mass,
                conaffinity=3,
                contype=3,
                friction=[0.4, 0.005, 0.001],
                rgba=[0.9, 0.7, 0.15, 1],
            )
            key_qpos += [0, 0, box_size[2] / 2 + 0.005, 1, 0, 0, 0]
            world.add_geom(
                name="delivery",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[0.8, 0, 0.002],
                size=[0.18, 0.002, 0],
                contype=0,
                conaffinity=0,
                rgba=[0.4, 0.9, 0.4, 0.7],
            )
        if game == "relay":
            for index, (x, y) in enumerate(((-0.6, -0.1), (0, 0.2), (0.8, 0.2))):
                world.add_geom(
                    name=f"checkpoint_{index}",
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    pos=[x, y, 0.002],
                    size=[0.16, 0.002, 0],
                    contype=0,
                    conaffinity=0,
                    rgba=[0.3, 0.6 + 0.1 * index, 0.9, 0.6],
                )
            baton = world.add_body(name="baton", mocap=True, pos=[-1, -0.1, 0.3])
            baton.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.035, 0, 0],
                contype=0,
                conaffinity=0,
                rgba=[1, 0.85, 0.1, 1],
            )
            if barrier_height:
                world.add_geom(
                    name="barrier",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    pos=[0.45, 0.2, barrier_height / 2],
                    size=[0.03, 0.4, barrier_height / 2],
                    conaffinity=3,
                    rgba=[0.65, 0.4, 0.3, 1],
                )
        spec.add_key(name="STAND", qpos=key_qpos, ctrl=key_ctrl)
        spec.add_numeric(name="arena", data=[length, width, players])
        spec.compile()
        return spec.to_xml()


class _ArenaMeta(_MujocoMeta):
    def __call__(
        cls,
        scene=None,
        *,
        microduck_root=None,
        root=None,
        download=False,
        players_per_team=None,
        arena=None,
        **kwargs,
    ):
        if scene is None:
            robot = MicroDuckEnv.resolve_scene(
                microduck_root, root=root, download=download
            )
            xml = _arena_scene(
                robot,
                game=cls.GAME,
                players=cls.PLAYERS if players_per_team is None else players_per_team,
                **(arena or {}),
            )
            directory = Path(root or "~/.cache/torchrl/microduck").expanduser() / "zoo"
            directory.mkdir(parents=True, exist_ok=True)
            scene = (
                directory
                / f"{cls.GAME}-{hashlib.sha256(xml.encode()).hexdigest()[:20]}.xml"
            )
            if not scene.exists():
                scene.write_text(xml)
        return super().__call__(scene, **kwargs)


class _ArenaEnv(MujocoEnv, metaclass=_ArenaMeta):
    GAME: ClassVar[str]
    PLAYERS: ClassVar[int] = 1
    GAME_FEATURE_DIM: ClassVar[int] = 0
    DEFAULT_BACKEND = "mujoco"
    FRAME_SKIP = MicroDuckEnv.FRAME_SKIP
    DUCK_NQ = 7 + MicroDuckEnv.NUM_JOINTS
    DUCK_NV = 6 + MicroDuckEnv.NUM_JOINTS

    def __init__(
        self,
        scene,
        *,
        action_scale=1.0,
        spawn_noise=0.03,
        observations="state",
        sensor_kwargs=None,
        max_episode_steps=500,
        backend="mujoco",
        **kwargs,
    ):
        if backend != "mujoco":
            raise ValueError("The initial arena recipes support native MuJoCo.")
        if action_scale <= 0 or spawn_noise < 0:
            raise ValueError(
                "action_scale must be positive and spawn_noise nonnegative."
            )
        self.scene_path = Path(scene)
        self.action_scale = action_scale
        self.spawn_noise = spawn_noise
        if observations not in ("state", "proprioception", "proprioception_vision"):
            raise ValueError("Unknown observation mode.")
        self.observations, self.sensor_kwargs = observations, dict(sensor_kwargs or {})
        self._sensors = None
        super().__init__(
            xml_path=scene,
            patch_xml=False,
            backend=backend,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )

    def _make_specs(self):
        model = self._backend.mj_model
        values = model.numeric("arena").data
        self.length, self.width, players = map(float, values)
        self.players_per_team = int(players)
        self.num_agents = 2 * self.players_per_team
        self._team = (
            torch.arange(self.num_agents, device=self.device) // self.players_per_team
        )
        key = model.key("STAND")
        self._stand_qpos = torch.as_tensor(
            key.qpos.copy(), dtype=self.dtype, device=self.device
        )
        self._home_joints = self._stand_qpos[7 : self.DUCK_NQ]
        self._home_ctrl = torch.as_tensor(
            key.ctrl.copy(), dtype=self.dtype, device=self.device
        )
        self._standing_height = float(self._stand_qpos[2])
        self.observation_dim = (
            MicroDuckEnv.OBSERVATION_DIM
            + 5
            + 5 * self.num_agents
            + self.GAME_FEATURE_DIM
        )
        self._previous_action = torch.zeros(1, self.num_agents, 14, device=self.device)
        self._fallen = torch.zeros(
            1, self.num_agents, dtype=torch.bool, device=self.device
        )
        self._down_steps = torch.zeros(
            1, self.num_agents, dtype=torch.long, device=self.device
        )
        self._game_state = self._initial_game_state()
        super()._make_specs()
        shape = (1, self.num_agents)
        self.action_spec = Composite(
            agents=Composite(
                action=Bounded(-1, 1, shape=(*shape, 14), device=self.device),
                shape=shape,
            ),
            shape=(1,),
            device=self.device,
        )
        self.reward_spec = Composite(
            agents=Composite(
                reward=Unbounded((*shape, 1), device=self.device),
                shape=shape,
            ),
            shape=(1,),
            device=self.device,
        )

    def _make_obs_spec(self):
        shape = (1, self.num_agents)
        spec = Composite(
            agents=Composite(
                observation=Unbounded(
                    (*shape, self.observation_dim), device=self.device
                ),
                fallen=Binary(shape=(*shape, 1), dtype=torch.bool, device=self.device),
                active=Binary(shape=(*shape, 1), dtype=torch.bool, device=self.device),
                shape=shape,
            ),
            outcome=Unbounded((1, 1), device=self.device),
            event=Unbounded((1, 1), device=self.device),
            events_total=Unbounded((1, 1), device=self.device),
            physics_error=Binary(shape=(1, 1), dtype=torch.bool, device=self.device),
            shape=(1,),
            device=self.device,
        )

        if self.observations != "state":
            self._sensors = _GameSensors(
                self,
                vision=self.observations == "proprioception_vision",
                **self.sensor_kwargs,
            )
            self._sensors.add_specs(spec["agents"])
        return spec

    def _initial_game_state(self):
        return TensorDict(
            {
                "active": torch.ones(1, self.num_agents, dtype=torch.bool),
                "outcome": torch.zeros(1, 1),
                "event": torch.zeros(1, 1),
                "events_total": torch.zeros(1, 1),
                "physics_error": torch.zeros(1, 1, dtype=torch.bool),
                "round": torch.full((1, 1), -1, dtype=torch.long),
            },
            [1],
            device=self.device,
        )

    def _sample_initial_state(self, n, tensordict=None):
        q = self._stand_qpos.expand(n, -1).clone()
        ducks = q[:, : self.num_agents * self.DUCK_NQ].view(
            n, self.num_agents, self.DUCK_NQ
        )
        ducks[..., :2] += (
            torch.rand(n, self.num_agents, 2, generator=self.rng, device=self.device)
            * 2
            - 1
        ) * self.spawn_noise
        if self.GAME in ("tag", "hide_and_seek") and bool(
            ((self._game_state["round"] + 1) // 2) % 2
        ):
            ducks[..., :2].neg_()
            w, x, y, z = ducks[..., 3:7].clone().unbind(-1)
            ducks[..., 3:7] = torch.stack((-z, -y, x, w), -1)
        return q, torch.zeros(n, self._backend.nv, device=self.device)

    def _on_reset_all(self, tensordict=None):
        if self._sensors is not None:
            self._sensors.reset()
        round_index = self._game_state["round"] + 1
        self._previous_action.zero_()
        self._fallen.zero_()
        self._down_steps.zero_()
        self._game_state = self._initial_game_state()
        self._game_state["round"] = round_index
        self._reset_game()
        self._sync_game_scene()

    def _on_reset_mask(self, mask, tensordict=None):
        if bool(mask.any()):
            self._on_reset_all(tensordict)

    def _reset_game(self):
        pass

    def _sync_game_scene(self):
        pass

    def _ducks(self, state):
        return (
            state["qpos"][:, : self.num_agents * self.DUCK_NQ].reshape(
                1, self.num_agents, self.DUCK_NQ
            ),
            state["qvel"][:, : self.num_agents * self.DUCK_NV].reshape(
                1, self.num_agents, self.DUCK_NV
            ),
        )

    @staticmethod
    def _yaw(quaternion):
        w, x, y, z = quaternion.unbind(-1)
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))

    @staticmethod
    def _relative(vector, yaw):
        x, y = vector.unbind(-1)
        return torch.stack(
            (yaw.cos() * x + yaw.sin() * y, -yaw.sin() * x + yaw.cos() * y), -1
        )

    def _make_obs(self, state):
        q, v = self._ducks(state)
        elapsed = (
            self._step_count.to(self.dtype) * self.frame_skip * self._backend.timestep
        )
        phase = (
            MicroDuckEnv.GAIT_PHASE_OFFSET
            + 2 * math.pi * MicroDuckEnv.GAIT_FREQUENCY_HZ * elapsed
        )
        clock = torch.stack(
            (
                phase.sin(),
                phase.cos(),
                (elapsed / MicroDuckEnv.GAIT_RAMP_DURATION_S).clamp_max(1),
            ),
            -1,
        )
        proprio = torch.cat(
            (
                _projected_gravity(q[..., 3:7]),
                v[..., 3:6],
                _body_frame_linear_velocity(q[..., 3:7], v[..., :3]),
                torch.zeros(1, self.num_agents, 2, device=self.device),
                q[..., 7:] - self._home_joints,
                v[..., 6:],
                clock[:, None].expand(-1, self.num_agents, -1),
                self._previous_action,
            ),
            -1,
        )
        yaw = self._yaw(q[..., 3:7])
        relative_q = self._relative(
            q[:, None, :, :2] - q[:, :, None, :2], yaw[..., None]
        )
        relative_v = self._relative(
            v[:, None, :, :2] - v[:, :, None, :2], yaw[..., None]
        )
        active = self._game_state["active"][:, None].expand(-1, self.num_agents, -1)
        time_left = (
            (1 - self._step_count / self.max_episode_steps)
            .reshape(1, 1, 1)
            .expand(-1, self.num_agents, -1)
        )
        features = torch.cat(
            (
                q[..., :2] / q.new_tensor([self.length / 2, self.width / 2]),
                torch.stack((yaw.cos(), yaw.sin()), -1),
                relative_q.flatten(-2),
                relative_v.flatten(-2),
                active,
                time_left,
                self._game_features(state),
            ),
            -1,
        )
        return torch.cat((proprio, features), -1)

    def _game_features(self, state):
        return torch.empty(1, self.num_agents, 0, device=self.device)

    def _build_obs_dict(self, state):
        result = {
            "agents": TensorDict(
                {
                    "observation": self._make_obs(state),
                    "fallen": self._fallen[..., None].clone(),
                    "active": self._game_state["active"][..., None].clone(),
                },
                [1, self.num_agents],
            ),
            "outcome": self._game_state["outcome"].clone(),
            "event": self._game_state["event"].clone(),
            "events_total": self._game_state["events_total"].clone(),
            "physics_error": self._game_state["physics_error"].clone(),
        }
        if self._sensors is not None:
            self._sensors.update(result["agents"])
        if self.from_pixels:
            result["pixels"] = self._render_pixels()
        return result

    def _prepare_ctrl(self, action):
        return self._home_ctrl + self.action_scale * action.reshape(1, -1)

    def _compute_reward(self, state, action, next_state):
        raise NotImplementedError("Arena games compute events and reward in _step.")

    def _compute_done(self, state, next_state):
        raise NotImplementedError("Arena games compute termination in _step.")

    def _step(self, tensordict):
        if not bool(tensordict.get("_step", torch.ones(1, dtype=torch.bool)).all()):
            return self._skip_tensordict(tensordict).update(
                self.full_reward_spec.zero()
            )
        state = self._state_td()
        action = tensordict["agents", "action"].to(self.dtype).clamp(-1, 1)
        active = self._game_state["active"].clone()
        action = torch.where((active & ~self._fallen)[..., None], action, 0)
        warning_ids = [
            int(mujoco.mjtWarning.mjWARN_BADQPOS),
            int(mujoco.mjtWarning.mjWARN_BADQVEL),
            int(mujoco.mjtWarning.mjWARN_BADQACC),
        ]
        warnings_before = self._backend._d.warning.number[warning_ids].copy()
        self._backend.step(self._prepare_ctrl(action), self.frame_skip)
        mujoco.mj_forward(self._backend.mj_model, self._backend._d)
        self._step_count += 1
        self._render_counter += 1
        next_state = self._state_td()
        q, v = self._ducks(next_state)
        fallen = (q[..., 2] < MicroDuckEnv.MIN_HEIGHT_RATIO * self._standing_height) | (
            -_projected_gravity(q[..., 3:7])[..., 2] < MicroDuckEnv.MIN_UPRIGHT
        )
        newly_fallen = fallen & ~self._fallen
        self._fallen = fallen
        self._game_state["event"].zero_()
        unstable = bool(
            (self._backend._d.warning.number[warning_ids] > warnings_before).any()
        )
        unstable |= not bool(torch.isfinite(q).all() & torch.isfinite(v).all())
        self._game_state["physics_error"].fill_(unstable)
        if unstable:
            # MuJoCo may reset invalid physics to finite qpos automatically.
            # That teleport must never create a capture or a delivery reward.
            reward = torch.full((1, self.num_agents, 1), -1.0, device=self.device)
            terminated = torch.ones(1, 1, dtype=torch.bool, device=self.device)
            self._game_state["outcome"].zero_()
        else:
            reward, terminated = self._step_game(state, next_state)
        self._game_state["events_total"] += self._game_state["event"]
        if self.GAME in ("pushing", "relay"):
            reward -= newly_fallen.sum(-1, keepdim=True)[..., None]
        else:
            reward -= newly_fallen[..., None] * 1.0
        # Inactive ducks stay in place. Fallen live ducks restart upright after
        # half a second; the raw fall signal resets only their walker memory.
        self._down_steps = torch.where(
            fallen & self._game_state["active"], self._down_steps + 1, 0
        )
        ready = self._down_steps >= math.ceil(
            0.5 / (self.frame_skip * self._backend.timestep)
        )
        inactive = ~active
        if bool((ready | inactive).any()):
            old_q, _ = self._ducks(state)
            q = q.clone()
            v = v.clone()
            q[inactive] = old_q[inactive]
            v[inactive | ready] = 0
            if bool(ready.any()):
                yaw = self._yaw(q[..., 3:7]) / 2
                upright = torch.stack(
                    (
                        yaw.cos(),
                        torch.zeros_like(yaw),
                        torch.zeros_like(yaw),
                        yaw.sin(),
                    ),
                    -1,
                )
                q[..., 2] = torch.where(ready, self._standing_height, q[..., 2])
                q[..., 3:7] = torch.where(ready[..., None], upright, q[..., 3:7])
                q[..., 7:] = torch.where(
                    ready[..., None], self._home_joints, q[..., 7:]
                )
                self._down_steps[ready] = 0
            native = self._backend._d
            native.qpos[: self.num_agents * self.DUCK_NQ] = q.reshape(-1).cpu().numpy()
            native.qvel[: self.num_agents * self.DUCK_NV] = v.reshape(-1).cpu().numpy()
            mujoco.mj_forward(self._backend.mj_model, native)
            next_state = self._state_td()
        self._previous_action = torch.where((fallen | inactive)[..., None], 0, action)
        self._sync_game_scene()
        finite = torch.isfinite(q).all() & torch.isfinite(v).all()
        terminated |= ~finite
        truncated = (self._step_count >= self.max_episode_steps)[:, None]
        obs = self._build_obs_dict(next_state)
        obs["agents"]["reward"] = reward
        return TensorDict(
            {
                **obs,
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            [1],
            device=self.device,
        )

    def _step_game(self, state, next_state):
        raise NotImplementedError

    def _index_extra_state(self, index):
        return {
            "game": self._game_state[index].clone(),
            "previous_action": self._previous_action[index].clone(),
            "fallen": self._fallen[index].clone(),
            "down_steps": self._down_steps[index].clone(),
        }

    def _load_indexed_extra_state(self, state):
        self._game_state = state["game"].clone()
        if self._sensors is not None:
            self._sensors = self._sensors.clone_for(self)
        self._previous_action = state["previous_action"].clone()
        self._fallen = state["fallen"].clone()
        self._down_steps = state["down_steps"].clone()

    def _set_indexed_extra_state(self, index, source):
        self._game_state[index] = source._game_state
        if self._sensors is not None and self._sensors.vision:
            self._sensors = source._sensors.clone_for(self)
        self._previous_action[index] = source._previous_action
        self._fallen[index] = source._fallen
        self._down_steps[index] = source._down_steps
