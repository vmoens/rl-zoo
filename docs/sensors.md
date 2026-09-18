# MicroDuck sensor selectors

Select `observations=state`, `proprioception`, or `proprioception_vision` with the
same game-training entry point. The state configuration preserves published
football checkpoint layouts. Sensor modes build a shared encoder and GRU with
separate memory for every duck. A critic reads privileged state by default;
`observations.critic_mode=actor` uses sensor features instead (decentralized).

```bash
python -m torchrl_zoo.microduck.train game=tag observations=proprioception_vision \
    algorithm=ppo_ewma runtime=macbook env.download=true smoke=true
python -m torchrl_zoo.microduck.train game=hide_and_seek observations=proprioception_vision \
    algorithm=ppo runtime=macbook env.download=true smoke=true
```

The actor reads `agents.proprioception` (53 values), `agents.game_command`
(team assignment and seeker role), `agents.selector_state` and `agents.is_init`.
Vision additionally reads `agents.camera_pixels` (uint8 RGB HWC),
`agents.camera_age` (seconds) and `agents.camera_valid`. It does not receive
world poses, hidden opponents, visibility labels, object coordinates, contacts,
flag possession or simulator velocities. Game commands are communicated by
the referee at round start. Critic and reward inputs remain explicit simulator
state and never enter the actor's sensor encoder.

The zoo owns this experimental proprioceptive schema, camera sampling and
selector composition; they are game-recipe components rather than part of the
TorchRL environment API. Cameras follow action
order: blue ducks, then red ducks, with an independent GRU state per duck.
Default policy images are 64-square centred crops of the nominal 16:9/62-degree
horizontal-FOV head image, sampled at 30 Hz. Configure size, rate, FOV, delay,
dropout and proprioceptive noise under `observations.sensors`. Missing first
frames are black/invalid; held frames expose their age. Episode resets clear
history; per-duck falls reset recurrent memory. Spectator rendering is separate.
The pinned robot asset's camera quaternion points backward. TorchRL preserves
the lens position and converts the matching site's forward/up axes to MuJoCo's
optical frame. Arena visibility uses the same corrected mount, including state
recipes; this is a simulation frame correction, not physical camera calibration.

These configurations currently pair the sensor selector with the published
nine-skill **privileged skill policy**. They are an **intermediate simulation
configuration**, not onboard-compatible control. Replacing that policy requires
a separately trained sensor prior from TorchRL with its exact artifact,
schema, task order and calibration. A visual selector alone does not remove
the low-level policy's simulator-state dependency.

Training collects complete episodes before recurrent updates. Both the learner
and EWMA proximal actor recompute sequences from episode boundaries; frozen
opponent state is kept separate and excluded from losses. Resume restores the
optimizer, scheduler, proximal/reference/opponent weights and hook state;
physics and per-agent memory restart at fresh episodes. Raw images are kept
as uint8 until the encoder to limit replay memory. Estimate the image replay
budget before increasing episode horizon, worker count or image resolution.

Validation includes input isolation, action-order camera images, camera mounting,
held frames, independent memory and fixed-rollout PPO-EWMA update/resume. The
recurrent visual tag smoke is a pipeline check; it is not evidence of learned
visual play. Hide-and-seek discovery uses sustained unoccluded visibility from
the head camera, and the visual selector must learn from its pixels.

Measure native throughput before each pilot:

```bash
python -m torchrl_zoo.microduck.benchmark --game tag \
    --observations proprioception_vision --workers 1 2 --steps 100 \
    --output visual-tag-workers.json
```

The report includes policy inference and environment-step time, end-to-end
throughput and learner-process peak RSS. Worker memory is additional. Record
concurrent machine load and resolved sensor settings alongside the result.
