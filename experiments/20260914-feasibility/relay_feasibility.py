from __future__ import annotations

import json
from pathlib import Path

import torch
from torchrl.envs import MicroDuckSkillEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules.tensordict_module.zoo import MicroDuckSkills

from torchrl_zoo.microduck.games.relay import MicroDuckRelayEnv

torch.set_num_threads(1)
skill_artifact = MicroDuckSkills.from_pretrained(
    revision="01ebcefca08850231edc0eb428a0151474559a85",
    filename="priors/nine-skills-20260913/walker.ckpt",
    sha256="9fbf1e15b25dd1ce65fcceeb2d854240028be703b72f37b5af05b9b50f115876",
)
results = []
for height in (0.0, 0.005, 0.01, 0.015):
    for skill in (1, 5):
        for seed in range(2):
            base = MicroDuckRelayEnv(
                download=True,
                arena={"barrier_height": height},
                spawn_noise=0.01,
                seed=seed,
                max_episode_steps=500,
            )
            q = base._stand_qpos.view(2, 21)
            q[0, 0] = -0.05
            q[0, 1] = 0.2
            q[0, 3:7] = torch.tensor([1.0, 0, 0, 0])
            q[1, 0] = -0.9
            q[1, 1] = -0.7
            env = MicroDuckSkillEnv.from_env(
                base, skill_artifact, control_steps_per_decision=5
            )
            try:
                td = env.reset()
                any_fall = False
                crossed = False
                max_x = -99.0
                error = False
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    for step in range(100):
                        td["agents", "skill"] = torch.tensor([[skill, 0]])
                        transition = env.step(td)
                        nxt = transition["next"]
                        xy = base.get_state()["qpos"][0, :2]
                        any_fall |= bool(nxt["agents", "fallen"][0, 0])
                        error |= bool(nxt["physics_error"].any())
                        max_x = max(max_x, float(xy[0]))
                        if xy[0] > 0.58 and abs(float(xy[1]) - 0.2) < 0.25:
                            crossed = True
                            break
                        if nxt["done"].any():
                            break
                        td = env.step_mdp(transition)
                results.append(
                    dict(
                        barrier_height=height,
                        skill=skill,
                        seed=seed,
                        decisions=step + 1,
                        crossed=crossed,
                        fall_before_crossing=any_fall,
                        upright_crossing=crossed and not any_fall,
                        physics_error=error,
                        max_x=max_x,
                    )
                )
            finally:
                env.close()
Path("relay-feasibility.json").write_text(
    json.dumps(
        dict(
            protocol="Actual flat/5/10/15mm barrier crossings, 1cm spawn perturbations, fixed forward or forward-hop skills; no reset may count as upright crossing.",
            results=results,
        ),
        indent=2,
    )
    + "\n"
)
