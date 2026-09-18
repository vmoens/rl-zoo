
## Football

`python -m torchrl_zoo.microduck.train game=football` trains two teams of ducks to play football against each
other on `torchrl_zoo.microduck.MicroDuckFootballEnv`. The scene is built in Python
with `mujoco.MjSpec` by `torchrl_zoo.microduck.games.football.build_football_scene`: one copy of the
walking robot per player, named `blue<i>/` or `red<i>/`, on a 3 x 2 m pitch
with walls, two goals, a 70 mm ball and two cameras (`broadcast` and
`topdown`). Nothing new is vendored: the meshes still come from the pinned
`microduck_rl` checkout, the MJCF is cached as text under
`~/.cache/torchrl/microduck/football` and `MicroDuckFootballEnv.write_scene`
exports it for inspection or sharing.

The env is multi-agent: `("agents", "action")` holds the 14 joint offsets of
every duck, `("agents", "observation")` its legacy 56-value skill-policy observation (which includes privileged simulator
state) followed by match features (position on the pitch, heading, ball,
goals, teammates and opponents, all in the duck's own frame and in its team's
frame), `("agents", "reward")` its reward. A goal ends the match with `+10`
for the scoring team and `-10` for the other; the ball's velocity toward the
goal and the duck's velocity toward the ball are dense shaping terms, a fall
costs `1` and respawns the duck in place by default (`env.respawn_mode` selects the reset position). Because every quantity
is team relative, one set of parameters plays both sides: the example is
plain self-play with a shared policy and a per-duck critic
(`MultiAgentMLP`; `policy.centralized_critic=true` opts into a centralized critic).

The ducks do not learn locomotion again. `policy.skills` identifies a
`MicroDuckSkills` artifact (by default the published low-level policy in the
[`torchrl/microduck-skills`](https://huggingface.co/torchrl/microduck-skills)
Hugging Face repository, a seven-skill policy trained with a football task
library: standing, straight forward and backward gaits, both sidesteps and
turning in place either way, every task keeping the head level so the camera
looks at the horizon, without the tutorial's jump skill). The artifact is
downloaded at its immutable `revision` and checked against `sha256`.
`torchrl.envs.MicroDuckSkillEnv.from_env` builds the high-level training env
using generic `ClosedLoopMultiAction` deployment: the football
policy picks one task-conditioned skill per duck every
`policy.control_steps_per_decision`
control steps (stand, walk forward or backward, sidestep left or right, turn
left or right, the library indices in `policy.skill_ids`), and the frozen policy
drives the joints at 50 Hz in between, fed the task's command and gait clock
in its observation. `policy.skills=null` trains joint-level actions end
to end at 50 Hz instead, the comparison the skill-based run should beat.

```bash
WANDB_BASE_URL=https://api.wandb.ai \
python -m torchrl_zoo.microduck.train game=football \
  env.microduck_root="$MICRODUCK_RL_ROOT" \
  logger.entity=YOUR_ENTITY evaluation.video.interval=2
```

[`conf/config.yaml`](../src/torchrl_zoo/microduck/conf/config.yaml) holds every setting. `env.players_per_team`
sets the team size (start with `1` or `2`: matches are shorter and the
credit assignment easier), `env.pitch` passes geometry arguments to
`build_football_scene` (for instance `'env.pitch={pitch_length:2.0}'`),
`env.reward_weights` retunes the reward (`'env.reward_weights={approach_ball:0.0}'`
once the ducks find the ball on their own). Evaluation plays
`evaluation.num_matches` deterministic matches and logs goals per team, the
fraction of decided matches, match length, falls per duck and the ball's
progress; `evaluation.video.interval=2` films one match from the broadcast
camera at every second evaluation (10 frames per second with a decision
period of 5, 50 with joint-level control). `smoke=true` runs a pipeline check with a
short clip written by a CSV logger.

For a warm-started policy, `ppo.reference_kl_coeff` anchors the actor to its
initial distribution. `ppo.reference_kl_final_coeff` optionally changes this
weight linearly after `ppo.critic_warmup_iterations`; leaving it `null` keeps
the anchor fixed. Annealing is an experiment option, not an established
improvement over a fixed anchor. With `ppo.train_team=blue`, checkpoint
selection uses blue's wins minus losses against the configured opponent,
including knockouts, so a policy is not promoted for conceding more goals.

The native MuJoCo recipe starts with one worker. Measure worker throughput on
the target machine before increasing `env.num_envs`; rollout steps count match
skill decisions, each spanning `policy.control_steps_per_decision` control steps. Other
physics backends remain available in the environment but are outside the
initial zoo CI support matrix.

Checkpoints are unified TorchRL checkpoints. To render one:

```bash
rlrender \
  --ckpt microduck_football_best.ckpt \
  --policy torchrl_zoo.microduck.football:make_render_policy \
  --env torchrl_zoo.microduck.football:make_env \
  --deterministic \
  --env-kwargs "{\"microduck_root\":\"$MICRODUCK_RL_ROOT\",\"num_envs\":1,\"parallel\":false}" \
  --render-backend env --max-steps 300 --fps 10 \
  --format mp4 --out football.mp4 --overwrite
```

A trainer snapshot (`ppo.save_trainer_file`) includes the optimizer, scheduler,
learner, reference policy, opponent and game-hook counters. Resume with
`resume=/path/to/trainer.ckpt`. Native MuJoCo starts a fresh episode; snapshots
do not claim bitwise continuation of live physics. Best/latest actor exports
use the existing TorchRL render checkpoint format and remain separate from
resumable training state.
