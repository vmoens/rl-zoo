"""Game-owned camera ordering, declared role commands and recurrent selectors."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from torchrl.data import Binary, Composite, Unbounded
from torchrl.envs.custom.mujoco._sensor_models import _MicroDuckSensorEncoder
from torchrl.envs.custom.mujoco._sensors import _MicroDuckSensors
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.modules import GRUModule, set_recurrent_mode


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
