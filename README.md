# TorchRL games zoo

Game environments and recipes that consume installed TorchRL components.
The [six-game catalog](docs/catalog.md) tracks mechanics, pipeline validation
and evaluated behavior separately. All six environments have implemented
mechanics. New-game learned behavior remains under evaluation. Recurrent sensor recipes
have mechanics and training-pipeline checks; see [sensor selectors](docs/sensors.md).
Skill definitions, the reusable low-level policy architecture and locomotion
training remain in TorchRL. The zoo composes the published skill artifact with
game dynamics through TorchRL's explicit skill-environment API:

```python
from torchrl.envs import MicroDuckSkillEnv
from torchrl.modules.tensordict_module.zoo import MicroDuckSkills

skills = MicroDuckSkills.from_pretrained()
env = MicroDuckSkillEnv.from_env(base_game_env, skills)
```

Install in a virtual environment with Python 3.12. The initial configuration
uses native MuJoCo on CPU and pinned source installation; no PyPI publication
or extra Hugging Face repository is required.

```bash
python -m pip install -r https://raw.githubusercontent.com/vmoens/rl-zoo/b1a4dcbdfd232f7c84ccf5190ea7ac4493842f2e/requirements-runtime.txt
python -m pip install --no-build-isolation --no-deps 'torchrl @ https://github.com/pytorch/rl/archive/2342f186380cc0ed7abd68ea3a6f80659fa13193.zip'
python -m pip install 'torchrl-zoo[test,video] @ https://github.com/vmoens/rl-zoo/archive/b1a4dcbdfd232f7c84ccf5190ea7ac4493842f2e.zip'
python -m torchrl_zoo.microduck.train game=football observations=state algorithm=ppo runtime=macbook env.download=true smoke=true
```

TorchRL's source build needs a C++ compiler. Video export needs FFmpeg available
to TorchCodec. The smoke command collects 40 skill decisions and evaluates and
renders short matches. Remove `smoke=true` for the configured training run.
Machine-specific asset paths can be supplied with `env.microduck_root`.

Use `algorithm=ppo_ewma` to train with a distinct proximal actor updated after
each successful optimizer step (`target_net_updater.eps=0.99`). Evaluation,
critic-only warmup and frozen opponents keep their own roles; the reference-KL
actor remains separate from the proximal actor. The KL-adaptive scheduler uses
the PPO loss's proximal-policy KL and excludes frozen/inactive ducks.
Resume with `resume=/path/to/trainer.ckpt` and the same architecture/algorithm.
The trainer checkpoint includes learner/proximal/reference/opponent parameters,
optimizer, scheduler, updater, RNG and evaluation-hook state. Best/latest
inference exports are separate; resumed physics begins with fresh matches.

On Linux CPU machines, install matching CPU wheels before the requirements:
`python -m pip install torch==2.11.0 torchcodec==0.16.0 --index-url https://download.pytorch.org/whl/cpu`.
The default Linux TorchCodec wheel requires CUDA libraries; see its
[installation instructions](https://github.com/meta-pytorch/torchcodec#installing-torchcodec).

The zoo remains optional for TorchRL. Its runtime imports never import
`examples.microduck`. See [football rules and recipes](docs/football.md) and
[artifact compatibility](docs/artifacts.md) for existing selectors and their
exact matching skill artifacts, plus the [new arena games](docs/arena-games.md).
Full game tests live here; TorchRL runs a bounded
integration tutorial against a pinned zoo revision.

Run native mechanics, training and resume regressions with `python -m pytest`.
The default MacBook runtime starts with one worker. New-game pilot budgets
are at most two hours each after mechanics and smoke checks pass. Initial
football relocation is verified by a bit-for-bit 32-step rollout comparison;
that is separate from the question of whether a trained selector plays well.

Source is MIT licensed. [NOTICE](NOTICE) retains attribution to TorchRL and
describes the separately downloaded robot assets.
