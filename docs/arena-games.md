# Arena games

These environments have native MuJoCo mechanics and state-based PPOTrainer
smokes. The [catalog](catalog.md) records evaluated pilots and remaining learning gaps. Recurrent visual selectors have passed pipeline checks; the sensor-only
skill policy still needs training. See [sensor inputs](sensors.md).

All new recipes preserve the nine-skill prior and its order. The actor chooses
one skill per duck every five physical steps. Its state observation begins
with the legacy 56-value skill-policy vector, followed by arena position, heading,
relative positions and velocities, activity, time remaining, game state, and
the previous skill. These are privileged simulator observations. The critic
has its own model; frozen opponents and inactive ducks are masked from PPO.
Captures end an individual duck's GAE sequence without ending its teammates'
sequences. Skill-step rewards accumulate over physical steps. Cumulative event
counters retain events that occur between selector decisions.

| Recipe | Rules and initial scale | Primary outcome |
| --- | --- | --- |
| `game=tag` | 1v1; `env.players_per_team=2` enables 2v2. Upright seekers capture upright runners within 0.16 m. Each runner is captured once and held inactive until reset. Roles alternate each round from a seeded start; spawn sides mirror every two rounds. Capture all runners to win; runners win at the time limit. | Capture / survival in both roles |
| `game=hide_and_seek` | Fixed cover; hiders have 2 s to prepare while seekers stay inactive. Discovery requires 0.5 s continuous visibility by one upright seeker, within 1.2 m and the mounted camera's square frustum. A MuJoCo ray must first hit the hider; occlusion or leaving the frustum resets the timer. Captured hiders stay inactive. | Discovery / survival; current state recipe is a mechanics baseline |
| `game=ctf` | 2v2. Upright ducks pick up enemy flags within 0.16 m. Falls or enemy proximity tags drop a carried flag and lock out that carrier's pickup for 1 s. Resolve drops, friendly returns, enemy pickups, then captures. A returned flag cannot be stolen in that same step. Nearest eligible duck wins a pickup; index resolves exact ties. Own flag must be home to capture. First capture wins. | Captures against fixed opponents in both team assignments |
| `game=pushing` | Two cooperative ducks and one 28x36x10 cm, 50 g box. Shared reward is five times distance progress plus ten on delivery. Delivery requires the box centre within 0.15 m of the target, upright, with both linear and angular speed below 0.03, continuously for 0.5 s. Leaving or moving resets dwell. | Settled delivery against standing / single-duck baselines |
| `game=relay` | Two cooperative ducks. Duck zero starts with the visible baton and visits the first checkpoint. Both upright ducks must enter the middle checkpoint within 0.18 m of each other to hand off once. Duck one then visits the finish. Each checkpoint pays once. The default course is flat. | Ordered completion against no-handoff / single-duck baselines |

Nonfinite physics terminates a round. The configured time limit truncates it.
Fallen active ducks respawn upright in place after 0.5 s; fallen observations
reset that duck's skill-policy memory. Cooperative games share fall penalties too.
Tag captures are proximity events and do not depend on contacts. CTF tags only
affect flag carriers. Flag-coloured markers follow carriers; relay has one
baton marker following its sole owner.

```bash
python -m torchrl_zoo.microduck.train game=tag env.download=true smoke=true
python -m torchrl_zoo.microduck.train game=ctf env.download=true smoke=true
python -m torchrl_zoo.microduck.benchmark --game tag --output tag-workers.json
```

The same command accepts `pushing`, `relay`, or `hide_and_seek`. The hide-and-seek
smoke shortens preparation and discovery to exercise the pipeline. Training
uses the configured durations. A smoke checks collection, one optimizer loop,
evaluation, checkpointing and video; it is not a training result.

The MacBook runtime caps training at 7,200 seconds, including evaluations.
The trainer requests a clean stop with 60 seconds reserved for final evaluation
and checkpointing. A process timer enforces the cap if a batch or evaluation
takes too long; in that case retain the last complete periodic checkpoint.
Resuming carries elapsed training time forward. Further training requires an
explicitly chosen budget. Run each pilot in its own output directory.

Native measurements with one PyTorch thread and the same initialized tag
selector: 153 / 224 / 454 skill decisions per second for 1 / 2 / 4 workers,
respectively (300 steps per worker; macOS 26.6.2 arm64). These measure collection,
not end-to-end optimization or visual throughput. The [physical feasibility report](../experiments/20260914-feasibility/README.md)
records box geometry, fixed-skill failures and actual low-barrier crossings.
The first relay pilot remains flat.

Use the existing render entry points for new game checkpoints:

```bash
rlrender --ckpt tag_best.ckpt \
  --env torchrl_zoo.microduck.football:make_env \
  --policy torchrl_zoo.microduck.football:make_render_policy \
  --env-kwargs '{"download":true}' --format mp4 --out tag.mp4
```

The factory reads the ordinary game class and options saved in the checkpoint's
resolved Hydra configuration. No remote code, registry, or new checkpoint
format is involved. The football module path is retained for compatibility.

CTF rounds default to 3,000 physical steps (60 seconds). The initial pilot used
500 steps (10 seconds), which is too short for the roughly four-metre flag
round trip at the observed forward walking speed of about 0.22 m/s. Preserve
that recorded configuration when interpreting its results. A longer diagnostic
evaluation can assess the same weights without further training:

```bash
python -m torchrl_zoo.microduck.evaluate ctf_latest.ckpt \
  --output ctf-60s --max-episode-steps 3000 --video
```

The report records the horizon override separately from the training config.
Longer evaluation does not replace a new training experiment with an adequate
horizon and a separately chosen budget.
