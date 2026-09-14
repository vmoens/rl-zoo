# MicroDuck games

Skills and prior training live in [TorchRL](https://github.com/pytorch/rl).
This optional package owns game mechanics, scenes, selectors and recipes.

| Game | Mechanics | Pipeline | Learned behavior |
| --- | --- | --- | --- |
| [Football](football.md) | Implemented; relocated deterministic rollout matches the original | Native MuJoCo smoke and fixed-update resume checked | Existing published policies retain their exact walker pairings; no new learning claim |
| [Team tag](arena-games.md) | Implemented: 1v1 and 2v2, one capture per target, inactive until round end | State PPO smoke checked; recurrent visual path planned | Not evaluated |
| [Hide-and-seek](arena-games.md) | Implemented: cover, preparation, sustained camera-visible discovery | State mechanics smoke checked; visual training follows visual tag | Not evaluated |
| [Capture the flag](arena-games.md) | Implemented: 2v2, pickup/drop/return/capture, own flag home to score | State PPO smoke checked | Not evaluated |
| [Cooperative pushing](arena-games.md) | Implemented: two ducks, a physical box, settled delivery | State PPO smoke checked; physical feasibility trial pending | Not evaluated |
| [Obstacle relay](arena-games.md) | Implemented: ordered checkpoints, single visible baton and handoff zone | Flat state PPO smoke checked; barriers await hop-clearance measurements | Not evaluated |

An implemented environment has tested mechanics. An evaluated game has a
recorded experiment and its measured outcomes; completing a pilot does not
establish successful learning. Initial new-game pilots have a two-hour cap
per game after mechanics and pipeline checks. Further training and additional
seeds need a separately selected budget.

New artifacts use `games/<game>/<release>/` in
[torchrl/microduck-skills](https://huggingface.co/torchrl/microduck-skills).
Existing football files and immutable URLs remain unchanged.
