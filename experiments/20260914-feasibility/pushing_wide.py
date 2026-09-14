from __future__ import annotations

import json
from pathlib import Path

import torch
from tensordict.nn import TensorDictModuleBase
from torchrl.envs import load_microduck_walker, microduck_skill_env
from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl_zoo.microduck.games.pushing import MicroDuckPushingEnv


class Forward(TensorDictModuleBase):
    in_keys = []
    out_keys = [("agents", "skill")]

    def __init__(self, skills):
        super().__init__()
        self.skills = torch.tensor([skills])

    def forward(self, td):
        td["agents", "skill"] = self.skills.expand(*td.shape, 2).clone()
        return td


torch.set_num_threads(1)
walker, tasks = load_microduck_walker(
    "https://huggingface.co/torchrl/microduck-skills/resolve/01ebcefca08850231edc0eb428a0151474559a85/priors/nine-skills-20260913/walker.ckpt",
    sha256="9fbf1e15b25dd1ce65fcceeb2d854240028be703b72f37b5af05b9b50f115876",
    action_scale=1.0,
)
results = []
for size, mass in (
    ((0.28, 0.36, 0.1), 0.025),
    ((0.28, 0.36, 0.1), 0.05),
    ((0.36, 0.4, 0.1), 0.025),
):
    for skills in ((0, 0), (1, 0), (1, 1)):
        for seed in range(1):
            base = MicroDuckPushingEnv(
                download=True,
                arena={"box_mass": mass, "box_size": size},
                spawn_noise=0.0,
                seed=seed,
                max_episode_steps=500,
            )
            q = base._stand_qpos[:42].view(2, 21)
            q[:, 0] = -0.35
            q[:, 1] = torch.tensor([-0.12, 0.12])
            q[:, 3:7] = torch.tensor([1.0, 0, 0, 0])
            env = microduck_skill_env(base, walker, tasks, steps=5)
            try:
                start = env.reset()
                initial = base.get_state()["qpos"][0, -7:-5].clone()
                states = []
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    td = start
                    for step in range(100):
                        transition = env.step(Forward(skills)(td))
                        state = base.get_state()
                        box = state["qpos"][0, -7:]
                        vel = state["qvel"][0, -6:]
                        nxt = transition["next"]
                        states.append(
                            dict(
                                x=float(box[0]),
                                y=float(box[1]),
                                z=float(box[2]),
                                upright=float(1 - 2 * box[4:6].square().sum()),
                                speed=float(vel[:3].norm()),
                                angular_speed=float(vel[3:].norm()),
                                falls=int(nxt["agents", "fallen"].sum()),
                                physics_error=bool(nxt["physics_error"].any()),
                                delivered=bool(nxt["outcome"].any()),
                            )
                        )
                        if nxt["done"].any():
                            break
                        td = env.step_mdp(transition)
                results.append(
                    dict(
                        size=size,
                        mass=mass,
                        skills=skills,
                        seed=seed,
                        decisions=len(states),
                        box_displacement_x=states[-1]["x"] - float(initial[0]),
                        max_box_height=max(x["z"] for x in states),
                        min_box_upright=min(x["upright"] for x in states),
                        max_box_speed=max(x["speed"] for x in states),
                        falls=sum(x["falls"] for x in states),
                        physics_error=any(x["physics_error"] for x in states),
                        delivered=states[-1]["delivered"],
                    )
                )
            finally:
                env.close()
Path("pushing-wide-feasibility.json").write_text(
    json.dumps(
        dict(
            protocol="Native MuJoCo, 100 skill decisions, box approach from x=-0.24 and y=+-0.08, fixed stand/forward policies; feasibility and task baselines only.",
            results=results,
        ),
        indent=2,
    )
    + "\n"
)
