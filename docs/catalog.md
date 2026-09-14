# MicroDuck games

Skills and prior training live in [TorchRL](https://github.com/pytorch/rl).
This optional package owns game mechanics, scenes, selectors and recipes.

| Game | Mechanics | Pipeline | Learned behavior |
| --- | --- | --- | --- |
| [Football](football.md) | Implemented; relocated deterministic rollout matches the original | Native MuJoCo smoke and fixed-update resume checked | Existing published policies retain their exact walker pairings; no new learning claim |
| Team tag | Planned: 1v1 then 2v2, one capture per target, inactive until round end | Planned | Not evaluated |
| Hide-and-seek | Planned: cover, preparation, sustained visible discovery | Planned; follows visual tag | Not evaluated |
| Capture the flag | Planned: 2v2, pickup/drop/return/capture, own flag home to score | Planned | Not evaluated |
| Cooperative pushing | Planned: two ducks, a physical box, settled delivery | Planned | Not evaluated |
| Obstacle relay | Planned: ordered checkpoints, single visible baton and handoff zone | Planned; flat relay first | Not evaluated |

An implemented environment has tested mechanics. An evaluated game has a
recorded experiment and its measured outcomes; completing a pilot does not
establish successful learning. Initial new-game pilots have a two-hour cap
per game after mechanics and pipeline checks. Further training and additional
seeds need a separately selected budget.

New artifacts use `games/<game>/<release>/` in
[torchrl/microduck-skills](https://huggingface.co/torchrl/microduck-skills).
Existing football files and immutable URLs remain unchanged.
