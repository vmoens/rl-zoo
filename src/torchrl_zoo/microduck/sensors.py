"""Game-owned camera ordering, declared role commands and recurrent selectors."""

from __future__ import annotations

import importlib
import math
from collections import deque
from collections.abc import Sequence
from copy import copy
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl.data import Binary, Composite, Unbounded
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.modules import GRUModule, set_recurrent_mode


def _align_microduck_cameras(model: Any):
    """Convert head-camera sites (+X forward, +Z up) to MuJoCo optical axes."""
    mujoco = importlib.import_module("mujoco")
    for index in range(model.ncam):
        camera = model.camera(index)
        if camera.name.rsplit("/", 1)[-1] != "head_camera":
            continue
        try:
            site = model.site(camera.name)
        except KeyError:
            continue
        if model.site_bodyid[site.id] != model.cam_bodyid[index]:
            raise ValueError("MicroDuck camera and matching site must share a body.")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, site.quat)
        rotation = rotation.reshape(3, 3)
        # Camera right=-site Y, up=site Z, back=-site X.
        optical = np.column_stack((-rotation[:, 1], rotation[:, 2], -rotation[:, 0]))
        mujoco.mju_mat2Quat(camera.quat, optical.ravel())


class _MicroDuckSensorEncoder(torch.nn.Module):
    """Encode only declared proprioception and optional raw RGB inputs."""

    def __init__(self, hidden_size: int, *, vision: bool, device="cpu"):
        super().__init__()
        self.vision = vision
        self.proprioception = torch.nn.Linear(53, hidden_size, device=device)
        if vision:
            self.camera = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 5, stride=2, device=device),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, 3, stride=2, device=device),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, 3, stride=2, device=device),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((2, 2)),
                torch.nn.Flatten(),
                torch.nn.Linear(128, hidden_size, device=device),
            )
            self.availability = torch.nn.Linear(2, hidden_size, device=device)

    def forward(self, proprioception, pixels=None, age=None, valid=None):
        features = self.proprioception(proprioception)
        if self.vision:
            batch = pixels.shape[:-3]
            images = (
                pixels.reshape(-1, *pixels.shape[-3:]).movedim(-1, 1).to(features.dtype)
                / 255
            )
            image_features = self.camera(images).reshape(*batch, -1)
            available = valid.to(features.dtype)
            features = (
                features
                + image_features * available
                + self.availability(torch.cat((age, available), -1))
            )
        return torch.tanh(features)


class _MicroDuckSensors:
    """Assemble the simulator sensors used by the experimental game selectors."""

    DIM = 53
    SCHEMA = "microduck-proprioception-v1"

    def __init__(
        self,
        env: Any,
        *,
        vision: bool = False,
        camera_names: Sequence[str] = ("head_camera",),
        image_size: int = 64,
        camera_fps: float = 30.0,
        camera_hfov: float = 62.0,
        camera_delay_s: float = 0.0,
        camera_dropout: float = 0.0,
        gyro_noise_std: float = 0.0,
        joint_position_noise_std: float = 0.0,
        joint_velocity_noise_std: float = 0.0,
    ):
        if (
            image_size < 32
            or not math.isfinite(camera_fps)
            or camera_fps <= 0
            or not 0 < camera_hfov < 180
        ):
            raise ValueError("Camera size, rate and field of view must be valid.")
        if (
            not 0 <= camera_dropout <= 1
            or any(
                not math.isfinite(value)
                for value in (
                    camera_delay_s,
                    gyro_noise_std,
                    joint_position_noise_std,
                    joint_velocity_noise_std,
                )
            )
            or min(
                camera_delay_s,
                gyro_noise_std,
                joint_position_noise_std,
                joint_velocity_noise_std,
            )
            < 0
        ):
            raise ValueError(
                "Sensor noise/delay must be nonnegative and dropout in [0, 1]."
            )
        if vision and env.backend_name != "mujoco":
            raise ValueError(
                "MicroDuck sensor cameras currently require native MuJoCo."
            )
        self.env = env
        self.vision = vision
        self.image_size, self.camera_fps = image_size, camera_fps
        self.camera_delay_s, self.camera_dropout = camera_delay_s, camera_dropout
        self.noise = (
            gyro_noise_std,
            joint_position_noise_std,
            joint_velocity_noise_std,
        )
        self.camera_names = tuple(camera_names)
        self.camera_ids = []
        if vision:
            model = env._backend.mj_model
            _align_microduck_cameras(model)
            vfov = math.degrees(
                2 * math.atan(math.tan(math.radians(camera_hfov) / 2) * 9 / 16)
            )
            for name in self.camera_names:
                camera = model.camera(name)
                self.camera_ids.append(camera.id)
                model.cam_fovy[camera.id] = vfov
        self.reset()

    def reset(self):
        self._next_capture = 0.0
        self._pending = deque()
        self._pixels = self._stamp = self._valid = None

    def clone_for(self, env):
        sensor = copy(self)
        sensor.env = env
        sensor._pending = deque(
            (stamp, pixels.clone(), valid.clone())
            for stamp, pixels, valid in self._pending
        )
        for name in ("_pixels", "_stamp", "_valid"):
            value = getattr(self, name)
            setattr(sensor, name, None if value is None else value.clone())
        return sensor

    def add_specs(self, spec: Composite):
        shape = spec["observation"].shape[:-1]
        spec["proprioception"] = Unbounded(
            (*shape, self.DIM), dtype=spec["observation"].dtype, device=spec.device
        )
        if self.vision:
            spec["camera_pixels"] = Unbounded(
                (*shape, self.image_size, self.image_size, 3),
                dtype=torch.uint8,
                device=spec.device,
            )
            spec["camera_age"] = Unbounded((*shape, 1), device=spec.device)
            spec["camera_valid"] = Binary(
                n=1, shape=(*shape, 1), dtype=torch.bool, device=spec.device
            )

    def update(self, td: TensorDictBase):
        observation = td["observation"]
        proprioception = torch.cat((observation[..., :6], observation[..., 9:56]), -1)
        for section, std in zip((slice(3, 6), slice(8, 22), slice(22, 36)), self.noise):
            if std:
                values = proprioception[..., section]
                values.add_(
                    torch.randn(
                        values.shape,
                        generator=self.env.rng,
                        device=values.device,
                        dtype=values.dtype,
                    )
                    * std
                )
        td["proprioception"] = proprioception
        if not self.vision:
            return td
        now = (
            float(self.env._step_count[0])
            * self.env.frame_skip
            * self.env._backend.timestep
        )
        if self._pixels is None:
            shape = observation.shape[:-1]
            self._pixels = torch.zeros(
                (*shape, self.image_size, self.image_size, 3),
                dtype=torch.uint8,
                device=observation.device,
            )
            self._stamp = torch.zeros((*shape, 1), device=observation.device)
            self._valid = torch.zeros(
                (*shape, 1), dtype=torch.bool, device=observation.device
            )
        if now + 1e-8 >= self._next_capture:
            images = torch.stack(
                [
                    self.env._backend.render(
                        camera_id=camera,
                        width=self.image_size,
                        height=self.image_size,
                    )
                    for camera in self.camera_ids
                ],
                dim=1,
            )
            if observation.ndim == 2:
                images = images.squeeze(1)
            received = (
                torch.rand(
                    self._valid.shape,
                    generator=self.env.rng,
                    device=observation.device,
                )
                >= self.camera_dropout
            )
            self._pending.append((now, images, received))
            self._next_capture = (
                math.floor(now * self.camera_fps + 1e-8) + 1
            ) / self.camera_fps
        while self._pending and self._pending[0][0] + self.camera_delay_s <= now + 1e-8:
            stamp, images, received = self._pending.popleft()
            self._pixels = torch.where(received[..., None, None], images, self._pixels)
            self._stamp = torch.where(received, stamp, self._stamp)
            self._valid = self._valid | received
        td["camera_pixels"] = self._pixels.clone()
        td["camera_age"] = torch.full_like(self._stamp, now) - self._stamp
        td["camera_valid"] = self._valid.clone()
        return td


class _GameSensors(_MicroDuckSensors):
    """Use head cameras in action order and explicit referee role commands.

    ``game_command`` contains team assignment (blue/red) and the assigned
    seeker role in tag/hide-and-seek, communicated at round start. It contains
    no positions, visibility, possession, opponent activity or game clock.
    """

    def __init__(self, env, *, vision, **kwargs):
        names = [
            f"{team}{index}/head_camera"
            for team in ("blue", "red")
            for index in range(env.players_per_team)
        ]
        super().__init__(env, vision=vision, camera_names=names, **kwargs)

    def add_specs(self, spec):
        super().add_specs(spec)
        shape = spec["observation"].shape[:-1]
        spec["game_command"] = Unbounded((*shape, 2), device=spec.device)
        spec["is_init"] = Binary(
            n=1, shape=(*shape, 1), dtype=torch.bool, device=spec.device
        )

    def update(self, td):
        super().update(td)
        shape = td["proprioception"].shape[:-1]
        team = torch.arange(shape[-1], device=td.device) // self.env.players_per_team
        command = td["proprioception"].new_zeros((*shape, 2))
        command[..., 0] = team
        if getattr(self.env, "GAME", None) in ("tag", "hide_and_seek"):
            command[..., 1] = team == self.env._game_state["seeker_team"]
        td["game_command"] = command
        td["is_init"] = td["fallen"] | (self.env._step_count == 0)[:, None, None]
        return td


class _SensorSelectorFeatures(TensorDictModuleBase):
    """Share an encoder/GRU across ducks while keeping their memories separate.

    A named time dimension selects sequence recomputation. Training samples
    must contain complete episodes, so the proximal actor never starts from a
    hidden state produced by different learner weights. Inference receives
    only proprioception, optional camera data, declared commands and memory.
    """

    def __init__(self, num_agents, hidden_size, *, vision, device="cpu"):
        super().__init__()
        self.num_agents, self.hidden_size = num_agents, hidden_size
        self.vision = vision
        self.encoder = _MicroDuckSensorEncoder(
            hidden_size, vision=vision, device=device
        )
        self.command = torch.nn.Linear(2, hidden_size, device=device)
        self.gru = GRUModule(
            input_size=hidden_size,
            hidden_size=hidden_size,
            in_keys=["embed", "state", "is_init"],
            out_keys=["features", ("next", "state")],
            device=device,
        )
        self.sensor_keys = ["proprioception"]
        if vision:
            self.sensor_keys += ["camera_pixels", "camera_age", "camera_valid"]
        self.in_keys = [
            ("agents", key)
            for key in self.sensor_keys + ["game_command", "selector_state", "is_init"]
        ]
        self.out_keys = [("agents", "features"), ("next", "agents", "selector_state")]

    def make_tensordict_primer(self):
        return TensorDictPrimer(
            {
                ("agents", "selector_state"): Unbounded(
                    (self.num_agents, 1, self.hidden_size),
                    device=next(self.parameters()).device,
                )
            },
            expand_specs=True,
        )

    def forward(self, td):
        agents = td["agents"]
        features = self.encoder(*(agents[key] for key in self.sensor_keys))
        features = torch.tanh(features + self.command(agents["game_command"]))
        inputs = TensorDict(
            {"embed": features, "is_init": agents["is_init"]},
            batch_size=features.shape[:-1],
        )
        state = agents.get("selector_state")
        if state is not None:
            inputs["state"] = state
        temporal = "time" in td.names
        if temporal:
            time_dim = td.names.index("time")
            inputs = inputs.transpose(time_dim, -1)
        with set_recurrent_mode(temporal):
            outputs = self.gru(inputs)
        if temporal:
            outputs = outputs.transpose(time_dim, -1)
        td["agents", "features"] = outputs["features"]
        td["next", "agents", "selector_state"] = outputs["next", "state"]
        return td
