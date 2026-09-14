# Published artifact compatibility

Immutable files stay in `torchrl/microduck-skills`. Game code lives in this
repository; skill definitions, models and training live in TorchRL.

| Artifact | Revision | File | SHA-256 |
| --- | --- | --- | --- |
| Legacy seven-skill walker | `8c31e2696520c402980a723b37d025e594197d9d` | `walker.ckpt` | `cbb5023d70bac278b3d066914289698c4ca53c6c8502ee39f5285fe51cf62c31` |
| Nine-skill walker | `01ebcefca08850231edc0eb428a0151474559a85` | `priors/nine-skills-20260913/walker.ckpt` | `9fbf1e15b25dd1ce65fcceeb2d854240028be703b72f37b5af05b9b50f115876` |
| Nine-skill football selector | `0bb7f4bd4c2660bb71a24aa9b12ed9c3585ac48d` | `football/nine-skills-20260913/selector.ckpt` | `bfa581396f8dae6ac201ef79a9dc8c3758fa7b1cb79c71d3f9887f87f0b8dbc7` |

The repository's published history contains the seven-skill **walker**, not a
seven-skill football selector. The legacy seven-skill recipe and synthetic
seven-skill checkpoint round trips remain supported. Do not claim validation
of a published selector that is absent.

The nine-skill selector records its training machine's local walker path.
Override it explicitly with the matching immutable URL; preserve the recorded
hash, all nine indices, action scale 1.0 and decision period 5:

```python
import torch
from huggingface_hub import hf_hub_download
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.render import load_checkpoint
from torchrl_zoo.microduck.football import make_env, make_models, load_parameters

selector = hf_hub_download(
    "torchrl/microduck-skills", "football/nine-skills-20260913/selector.ckpt",
    revision="0bb7f4bd4c2660bb71a24aa9b12ed9c3585ac48d",
)
walker_url = (
    "https://huggingface.co/torchrl/microduck-skills/resolve/"
    "01ebcefca08850231edc0eb428a0151474559a85/"
    "priors/nine-skills-20260913/walker.ckpt"
)
payload = load_checkpoint(selector)
env = make_env(
    {"policy": {"walker_checkpoint": walker_url}}, checkpoint=payload,
    num_envs=1, parallel=False, download=True,
)
try:
    actor, critic = make_models(env, **payload["policy_kwargs"])
    load_parameters(selector, actor, critic)
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        rollout = env.rollout(300, actor)
finally:
    env.close()
```

Use the same `cfg.policy.walker_checkpoint` override inside rlrender's
`--env-kwargs`, with `torchrl_zoo.microduck.football:make_env` and
`torchrl_zoo.microduck.football:make_render_policy`. No checkpoint rewrite or
remote Python execution is needed. These remain inference/warm-start exports;
optimizer state is present only in separate trainer snapshots.

New game releases use `games/<game>/<release>/`. Their bundle must record the
source and dependency revisions, simulator asset revision, observation schema,
ordered skills, exact walker revision/hash, resolved configuration, seed,
evaluation protocol, metrics and videos. A simulation selector using the
legacy privileged walker is labeled accordingly, including when its own inputs
are camera images. Existing football paths are never repurposed.
