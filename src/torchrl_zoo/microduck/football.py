# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Multi-agent PPO for MicroDuck football, five against five by default.

Both teams share one policy (self-play) and every quantity a duck observes is
expressed in its own frame and in its team's frame, so the same parameters
play both sides. The critic is centralized: it reads every duck's observation
and predicts one value per duck. Data flows through
:class:`~torchrl_zoo.microduck.MicroDuckFootballEnv`, a
:class:`~torchrl.collectors.Collector`, a
:class:`~torchrl.objectives.ClipPPOLoss` with GAE and a
:class:`~torchrl.data.ReplayBuffer` for the minibatches.

The ducks come with a walker. ``policy.walker_checkpoint`` names a MicroDuck
locomotion policy trained with ``ppo_mujoco.py`` (a local path or a URL,
verified against ``policy.walker_sha256``);
:func:`~torchrl.envs.microduck_skill_env` builds an env in which the football policy picks
one of the walker's tasks per duck (stand, walk forward or backward, sidestep
left or right) every ``policy.decision_period`` control steps, and the frozen
walker drives the joints in between. ``policy.walker_checkpoint=null`` trains
joint-level actions end to end instead, at 50 Hz.

Evaluation runs deterministic matches with a
:class:`~torchrl.collectors.Evaluator` and, on request, films one from the
broadcast camera into the logger. Checkpoints are unified TorchRL checkpoints
written with :func:`~torchrl.render.save_render_checkpoint`; ``rlrender``
rebuilds the match with :func:`make_env` and :func:`make_render_policy`.

The script is configured with Hydra from ``conf/config.yaml``. Run a short CPU
job from an installed zoo::

    python -m torchrl_zoo.microduck.train game=football env.download=true smoke=true

and a 5-a-side training run with::

    python -m torchrl_zoo.microduck.train game=football env.download=true \\
        logger.entity=YOUR_ENTITY evaluation.video.interval=2

``env.download=true`` fetches the pinned ``microduck_rl`` assets into
``~/.cache/torchrl/microduck``; set ``env.microduck_root`` or
``MICRODUCK_RL_ROOT`` to use an existing checkout instead.
"""

from __future__ import annotations

import functools as ft
import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import NestedKey, TensorDictBase
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictModuleBase
from torch import nn
from torch.distributions import Categorical

from torchrl import torchrl_logger
from torchrl.checkpoint import Checkpoint, GlobalRNGState
from torchrl.collectors import Collector, Evaluator
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.data.tensor_specs import Categorical as CategoricalSpec
from torchrl.envs import (
    EnvBase,
    TransformedEnv,
    microduck_skill_env,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, KLAdaptiveLR, ValueEstimators
from torchrl.record import VideoRecorder
from torchrl.record.loggers import Logger, generate_exp_name, get_logger
from torchrl.render import load_checkpoint, save_render_checkpoint
from torchrl.trainers import ReplayBufferTrainer
from torchrl.trainers.algorithms import PPOTrainer
from torchrl_zoo.microduck.training import GameTrainingHooks

PACKAGE_DIR = Path(__file__).resolve().parent

from torchrl.envs import load_microduck_walker
from torchrl_zoo.microduck.games.football import MicroDuckFootballEnv

# The asset location is machine specific and is never taken from a checkpoint.
ASSET_KEYS = ("microduck_root", "root", "download")
CONTROL_PERIOD_S = MicroDuckFootballEnv.FRAME_SKIP * 0.002
OBSERVATION_KEY = ("agents", "observation")
REWARD_KEY = ("agents", "reward")
VALUE_KEY = ("agents", "state_value")


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------


def make_env(
    cfg: DictConfig | Mapping[str, Any] | None = None,
    *,
    checkpoint: Mapping[str, Any] | None = None,
    microduck_root: str | Path | None = None,
    root: str | Path | None = None,
    download: bool | str | None = None,
    num_envs: int | None = None,
    parallel: bool | None = None,
    device: torch.device | str | None = None,
    from_pixels: bool = False,
    render_width: int | None = None,
    render_height: int | None = None,
) -> TransformedEnv:
    """Build the match env from the ``env`` and ``policy`` sections of ``conf/config.yaml``.

    ``cfg`` is the whole configuration, as the Hydra ``DictConfig`` or a plain
    mapping; missing entries take the defaults of ``conf/config.yaml``.
    ``rlrender`` passes the training checkpoint, whose recorded config (minus
    the asset location, which is machine specific) sits between those defaults
    and ``cfg``. The keyword arguments override single entries so a checkpoint
    renders with one match from a local asset path.

    With ``policy.walker_checkpoint`` set, the joint-level
    :class:`~torchrl_zoo.microduck.MicroDuckFootballEnv` is wrapped by
    :func:`~torchrl.envs.microduck_skill_env`, driven by the walker, and the
    env's actions are skill indices. ``from_pixels`` adds a rendered
    ``pixels`` observation for the video.
    """
    recorded = checkpoint if isinstance(checkpoint, Mapping) else {}
    recorded_config = dict(recorded.get("config") or {})
    recorded_config["env"] = {
        key: value
        for key, value in (recorded_config.get("env") or {}).items()
        if key not in ASSET_KEYS
    }
    recorded_config = {
        key: recorded_config[key] for key in ("env", "policy") if key in recorded_config
    }
    overrides = {
        "microduck_root": None if microduck_root is None else str(microduck_root),
        "root": None if root is None else str(root),
        "download": download,
        "num_envs": num_envs,
        "parallel": parallel,
        "device": None if device is None else str(device),
        "render_width": render_width,
        "render_height": render_height,
    }
    if cfg is not None and not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(dict(cfg))
    merged = OmegaConf.to_container(
        OmegaConf.merge(
            OmegaConf.load(PACKAGE_DIR / "conf" / "config.yaml"),
            recorded_config,
            cfg or {},
            {
                "env": {
                    key: value for key, value in overrides.items() if value is not None
                }
            },
        ),
        resolve=True,
    )
    env_cfg = merged["env"]
    policy_cfg = merged["policy"]
    kwargs: dict[str, Any] = {
        "root": env_cfg["root"],
        "download": env_cfg["download"],
        "players_per_team": env_cfg["players_per_team"],
        "pitch": dict(env_cfg["pitch"] or {}),
        "backend": env_cfg["backend"],
        "num_envs": env_cfg["num_envs"],
        "device": torch.device(env_cfg["device"]),
        "seed": env_cfg["seed"],
        "max_episode_steps": env_cfg["max_episode_steps"],
        "action_scale": env_cfg["action_scale"],
        "reward_weights": dict(env_cfg["reward_weights"] or {}),
        "spawn_noise": env_cfg["spawn_noise"],
        "yaw_noise": env_cfg["yaw_noise"],
        "joint_reset_noise_scale": env_cfg["joint_reset_noise_scale"],
        "ball_noise": env_cfg["ball_noise"],
        "respawn": env_cfg["respawn"],
        "knockout": env_cfg["knockout"],
        "respawn_mode": env_cfg["respawn_mode"],
        "respawn_delay_s": env_cfg["respawn_delay_s"],
        "approach_players": env_cfg["approach_players"],
        "progress_players": env_cfg["progress_players"],
        "camera_id": env_cfg["camera_id"],
        "render_width": env_cfg["render_width"],
        "render_height": env_cfg["render_height"],
        "from_pixels": from_pixels,
    }
    if env_cfg["backend"] == "mujoco":
        kwargs["parallel"] = env_cfg["parallel"]
    env: EnvBase = MicroDuckFootballEnv(
        microduck_root=env_cfg["microduck_root"], **kwargs
    )
    if policy_cfg["walker_checkpoint"] is not None:
        walker, tasks = load_microduck_walker(
            policy_cfg["walker_checkpoint"],
            device=env.device,
            root=env_cfg["root"],
            sha256=policy_cfg["walker_sha256"],
            action_scale=env_cfg["action_scale"],
        )
        env = microduck_skill_env(
            env,
            walker,
            tasks,
            skills=policy_cfg["skills"],
            steps=policy_cfg["decision_period"],
            control_period_s=CONTROL_PERIOD_S,
        )
    return TransformedEnv(env)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------


def make_models(
    env: EnvBase,
    *,
    device: torch.device | str = "cpu",
    hidden_size: int = 256,
    depth: int = 2,
    initial_policy_scale: float = 1.0,
    centralized_critic: bool = False,
) -> tuple[ProbabilisticActor, TensorDictModule]:
    """Create the shared-parameter actor and the critic.

    Both are :class:`~torchrl.modules.MultiAgentMLP` networks built on
    ``device`` and shared by every duck. The actor is decentralized (each duck
    acts on its own observation): a categorical head over the skills when the
    env's action is a skill index, a tanh-squashed Gaussian over the 14 joint
    offsets otherwise, with a state-independent exploration scale starting at
    ``initial_policy_scale``. The critic returns one value per duck from that
    duck's observation, which already describes the whole match in the duck's
    team frame; with ``centralized_critic`` it reads every duck's observation
    instead and returns the same value for all of them, which cannot tell the
    two teams of a zero-sum match apart.
    """
    if not math.isfinite(initial_policy_scale) or initial_policy_scale <= 0:
        raise ValueError("initial_policy_scale must be finite and positive.")
    device = torch.device(device)
    observation = env.observation_spec[OBSERVATION_KEY]
    num_agents, observation_dim = observation.shape[-2:]
    action_spec = env.full_action_spec_unbatched[env.action_key]
    network_kwargs = {
        "n_agents": num_agents,
        "share_params": True,
        "device": device,
        "depth": depth,
        "num_cells": hidden_size,
        "activation_class": nn.Tanh,
    }
    if isinstance(action_spec, CategoricalSpec):
        head = TensorDictModule(
            MultiAgentMLP(
                observation_dim,
                action_spec.space.n,
                centralized=False,
                **network_kwargs,
            ),
            in_keys=[OBSERVATION_KEY],
            out_keys=[("agents", "logits")],
        )
        actor = ProbabilisticActor(
            module=head,
            spec=action_spec,
            in_keys={"logits": ("agents", "logits")},
            out_keys=[env.action_key],
            distribution_class=Categorical,
            return_log_prob=True,
        )
    else:
        head = TensorDictModule(
            nn.Sequential(
                MultiAgentMLP(
                    observation_dim,
                    2 * action_spec.shape[-1],
                    centralized=False,
                    **network_kwargs,
                ),
                NormalParamExtractor(
                    scale_mapping=f"biased_softplus_{initial_policy_scale}"
                ),
            ),
            in_keys=[OBSERVATION_KEY],
            out_keys=[("agents", "loc"), ("agents", "scale")],
        )
        actor = ProbabilisticActor(
            module=head,
            spec=action_spec,
            in_keys={"loc": ("agents", "loc"), "scale": ("agents", "scale")},
            out_keys=[env.action_key],
            distribution_class=TanhNormal,
            distribution_kwargs={"low": -1.0, "high": 1.0},
            return_log_prob=True,
        )
    critic = TensorDictModule(
        MultiAgentMLP(
            observation_dim, 1, centralized=centralized_critic, **network_kwargs
        ),
        in_keys=[OBSERVATION_KEY],
        out_keys=[VALUE_KEY],
    )
    return actor, critic


class OpponentPolicy(TensorDictModuleBase):
    """Run the actor for every duck, then hand the red team to its opponent.

    The opponent curriculum: blue learns against a team of statues (``skill``
    0, standing), of ducks running one fixed skill, or of ducks driven by a
    frozen copy of the actor (``opponent``, refreshed with :meth:`refresh`
    for fictitious self-play) while red's transitions carry no learning
    signal (``ppo.train_team=blue``). The actor's own parameters are shared,
    so the loss keeps training it. The executed actions' log-probabilities
    are recomputed under the actor, so every duck's importance ratio starts
    at one.
    """

    def __init__(
        self,
        actor: ProbabilisticActor,
        players_per_team: int,
        action_key: NestedKey,
        *,
        skill: int | None = None,
        opponent: ProbabilisticActor | None = None,
    ):
        super().__init__()
        if (skill is None) == (opponent is None):
            raise ValueError("Pass exactly one of skill and opponent.")
        self.actor = actor
        self.opponent = opponent
        self.players_per_team = int(players_per_team)
        self.skill = None if skill is None else int(skill)
        self.action_key = action_key
        self.in_keys = list(actor.in_keys)
        self.out_keys = list(actor.out_keys)

    def refresh(self) -> None:
        """Copy the actor's current parameters into the frozen opponent."""
        if self.opponent is not None:
            self.opponent.load_state_dict(self.actor.state_dict())

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = self.actor(tensordict)
        action = tensordict.get(self.action_key).clone()
        if self.skill is not None:
            action[..., self.players_per_team :] = self.skill
        else:
            with torch.no_grad():
                red = self.opponent(tensordict.select(*self.opponent.in_keys))
            action[..., self.players_per_team :] = red.get(self.action_key)[
                ..., self.players_per_team :
            ]
        tensordict.set(self.action_key, action)
        with torch.no_grad():
            log_prob = self.actor.get_dist(tensordict).log_prob(action)
        return tensordict.set(self.actor.log_prob_keys[0], log_prob)


def make_render_policy(
    env: EnvBase,
    *,
    device: torch.device | str = "cpu",
    checkpoint: Mapping[str, Any] | None = None,
    hidden_size: int | None = None,
    depth: int | None = None,
    initial_policy_scale: float | None = None,
) -> ProbabilisticActor:
    """Build the actor whose weights an ``rlrender`` checkpoint provides.

    Architecture arguments default to the ``policy_kwargs`` recorded in the
    training checkpoint, which ``rlrender`` passes as ``checkpoint``; explicit
    ``--policy-kwargs`` override them.
    """
    recorded = (
        dict(checkpoint.get("policy_kwargs") or {})
        if isinstance(checkpoint, Mapping)
        else {}
    )
    overrides = {
        "hidden_size": hidden_size,
        "depth": depth,
        "initial_policy_scale": initial_policy_scale,
    }
    recorded.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    actor, _ = make_models(env, device=device, **recorded)
    return actor


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    actor: ProbabilisticActor,
    critic: TensorDictModule,
    *,
    frames: int,
    policy_kwargs: Mapping[str, Any],
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Path:
    """Write a unified TorchRL checkpoint that ``rlrender`` and ``init_from`` read."""
    return save_render_checkpoint(
        path,
        actor,
        env_metadata={"policy_kwargs": dict(policy_kwargs)},
        frames=frames,
        metrics=dict(metrics),
        config=dict(config),
        extra={"critic_state_dict": critic.state_dict()},
        format="archive",
    )


def load_parameters(
    path: str | Path, actor: ProbabilisticActor, critic: TensorDictModule
) -> int:
    """Load actor and critic parameters from a checkpoint written by :func:`save_checkpoint`.

    Returns:
        The number of env steps the checkpoint was trained on.
    """
    payload = load_checkpoint(path)
    try:
        actor.load_state_dict(payload["model_state_dict"])
        critic.load_state_dict(payload["critic_state_dict"])
    except RuntimeError as err:
        raise RuntimeError(
            f"The checkpoint {path} was trained with policy kwargs "
            f"{payload.get('policy_kwargs')} and config "
            f"{payload.get('config')}; the current models must match them."
        ) from err
    return int(payload.get("frames", 0))


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def _fall_events(fallen: torch.Tensor, dim: int) -> torch.Tensor:
    """Steps on which a duck goes down, from the flag that stays up while it lies there.

    ``dim`` is the time dimension of ``fallen``.
    """
    first = fallen.narrow(dim, 0, 1)
    later = fallen.narrow(dim, 1, fallen.shape[dim] - 1)
    earlier = fallen.narrow(dim, 0, fallen.shape[dim] - 1)
    return torch.cat((first, later & ~earlier), dim=dim)


def football_metrics(trajectories: TensorDictBase) -> dict[str, float]:
    """Match statistics of the padded trajectory batch an :class:`Evaluator` collects.

    Goals are counted per match for each team (both teams play the same
    policy, so a lasting difference between the two is a sign of an asymmetry
    in the env), together with the knockouts per team, the fraction of matches
    decided by a goal or a knockout, the match length, the falls per duck and
    per match (a duck that lies down for a while counts once), and how far the
    ball traveled along blue's attacking direction.
    """
    mask = trajectories["collector", "mask"]
    lengths = mask.sum(-1)
    last = (lengths - 1).unsqueeze(-1)
    goal = trajectories["next", "goal"].squeeze(-1)
    goal = torch.where(mask, goal, torch.zeros_like(goal))
    knockout = trajectories["next", "knockout"].squeeze(-1)
    knockout = torch.where(mask, knockout, torch.zeros_like(knockout))
    fallen = trajectories["next", "agents", "fallen"].squeeze(-1) & mask.unsqueeze(-1)
    fallen = _fall_events(fallen, dim=trajectories.ndim - 1)
    ball_x = trajectories["next", "ball_position"][..., 0]
    start_x = trajectories["ball_position"][..., 0, 0]
    end_x = ball_x.gather(-1, last).squeeze(-1)
    return {
        "goals_blue": float((goal == 1).sum(-1).float().mean()),
        "goals_red": float((goal == -1).sum(-1).float().mean()),
        "knockouts_blue": float((knockout == 1).sum(-1).float().mean()),
        "knockouts_red": float((knockout == -1).sum(-1).float().mean()),
        "decided_rate": float(((goal != 0) | (knockout != 0)).any(-1).float().mean()),
        "match_length": float(lengths.float().mean()),
        "falls_per_duck": float(
            fallen.sum(dim=(1, 2)).float().mean() / fallen.shape[-1]
        ),
        "ball_progress_blue": float((end_x - start_x).mean()),
    }


def _collection_metrics(data: TensorDictBase) -> dict[str, float]:
    reward = data["next", REWARD_KEY].squeeze(-1)
    done = data["next", "done"].squeeze(-1)
    goal = data["next", "goal"].squeeze(-1)
    knockout = data["next", "knockout"].squeeze(-1)
    fallen = _fall_events(
        data["next", "agents", "fallen"].squeeze(-1), dim=data.ndim - 1
    )
    players = reward.shape[-1] // 2
    traj_ids = data["collector", "traj_ids"]
    unique_ids, inverse = torch.unique(traj_ids, return_inverse=True)
    lengths = torch.zeros(unique_ids.numel()).index_add_(
        0, inverse.reshape(-1), torch.ones(inverse.numel())
    )
    returns = torch.zeros(unique_ids.numel(), reward.shape[-1]).index_add_(
        0, inverse.reshape(-1), reward.reshape(-1, reward.shape[-1])
    )
    finished = float(done.sum())
    return {
        "collection/reward_mean": float(reward.mean()),
        "collection/reward_blue": float(reward[..., :players].mean()),
        "collection/reward_red": float(reward[..., players:].mean()),
        "collection/falls_per_duck_step": float(fallen.float().mean()),
        "collection/goals_blue": float((goal == 1).sum()),
        "collection/goals_red": float((goal == -1).sum()),
        "collection/knockouts_blue": float((knockout == 1).sum()),
        "collection/knockouts_red": float((knockout == -1).sum()),
        "episode/finished": finished,
        "episode/decided_rate": float(((goal != 0) | (knockout != 0)).sum())
        / max(finished, 1.0),
        "episode/length_mean": float(lengths.mean()),
        "episode/return_mean": float(returns.mean()),
    }


def evaluation_score(
    metrics: Mapping[str, float], *, train_team: Literal["both", "blue"] = "both"
) -> tuple[float, ...]:
    """Rank checkpoints by match outcomes, then ball progress, then fewer falls.

    Symmetric self-play rewards goals by either team. Against a separate
    opponent, rank blue's wins minus losses instead: conceding goals or
    losing by knockout must not promote a checkpoint.
    """
    blue = metrics["evaluation/goals_blue"]
    red = metrics["evaluation/goals_red"]
    outcomes = blue + red
    if train_team == "blue":
        outcomes = (
            blue
            - red
            + metrics.get("evaluation/knockouts_blue", 0.0)
            - metrics.get("evaluation/knockouts_red", 0.0)
        )
    return (
        outcomes,
        metrics["evaluation/ball_progress_blue"],
        -metrics["evaluation/falls_per_duck"],
    )


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def make_trainer(
    env: EnvBase,
    actor: ProbabilisticActor,
    critic: TensorDictModule,
    *,
    total_frames: int = 5_000_000,
    frames_per_batch: int = 4800,
    epochs: int = 4,
    minibatch_size: int = 1200,
    learning_rate: float = 3e-4,
    target_kl: float | None = 0.02,
    max_learning_rate: float = 1e-2,
    clip_epsilon: float = 0.2,
    entropy_coeff: float = 0.01,
    critic_coeff: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    max_grad_norm: float = 1.0,
    evaluator: Evaluator | None = None,
    evaluation_interval: int | None = None,
    video_recorder: Callable[[int], None] | None = None,
    video_interval: int | None = None,
    best_checkpoint_path: str | Path | None = None,
    latest_checkpoint_path: str | Path | None = None,
    policy_kwargs: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    logger: Logger | None = None,
    train_team: Literal["both", "blue"] = "both",
    critic_warmup_iterations: int = 0,
    reference_kl_coeff: float = 0.0,
    reference_kl_final_coeff: float | None = None,
    collection_policy: TensorDictModuleBase | None = None,
    iteration_callback: Callable[[int], None] | None = None,
    save_trainer_file: str | Path | None = None,
) -> PPOTrainer:
    """Build the football PPOTrainer, including evaluation and curriculum hooks.

    The returned trainer owns collection, minibatch optimization, weight
    synchronization and resumable state. ``trainer.game_hooks.history`` holds
    one metrics dictionary per collection. Checkpoints export best/latest
    policies separately from the trainer's resumable optimizer state.
    """
    if min(total_frames, frames_per_batch, epochs, minibatch_size) < 1:
        raise ValueError("PPO frame, epoch and minibatch sizes must be positive.")
    if reference_kl_final_coeff is None:
        reference_kl_final_coeff = reference_kl_coeff
    if any(
        not math.isfinite(coeff) or coeff < 0
        for coeff in (reference_kl_coeff, reference_kl_final_coeff)
    ):
        raise ValueError("Reference KL coefficients must be finite and non-negative.")
    num_envs = env.batch_size.numel()
    if frames_per_batch % num_envs:
        raise ValueError(
            f"frames_per_batch ({frames_per_batch}) must be a multiple of the "
            f"{num_envs} matches simulated in parallel."
        )
    if minibatch_size > frames_per_batch:
        raise ValueError("minibatch_size cannot exceed frames_per_batch.")
    if evaluation_interval is not None and (
        evaluation_interval < 1 or evaluator is None
    ):
        raise ValueError(
            "evaluation_interval requires a positive value and an evaluator."
        )
    if video_recorder is not None and (video_interval is None or video_interval < 1):
        raise ValueError("video_recorder requires a positive video_interval.")
    checkpointing = (
        best_checkpoint_path is not None or latest_checkpoint_path is not None
    )
    if checkpointing and evaluation_interval is None:
        raise ValueError("Checkpoint paths require periodic evaluation.")
    if checkpointing and (config is None or policy_kwargs is None):
        raise ValueError("Checkpoint paths require config and policy_kwargs.")

    device = next(actor.parameters()).device
    collection_policy = actor if collection_policy is None else collection_policy
    collector = Collector(
        env,
        collection_policy,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        storing_device="cpu",
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_bonus=True,
        entropy_coeff=entropy_coeff,
        critic_coeff=critic_coeff,
        loss_critic_type="smooth_l1",
        normalize_advantage=train_team == "both",
    )
    loss_module.set_keys(
        reward=REWARD_KEY,
        action=env.action_key,
        value=VALUE_KEY,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.loss_mask_key = ("agents", "train_mask")
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=gae_lambda)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=learning_rate)
    scheduler = (
        KLAdaptiveLR(optimizer, target_kl=target_kl, max_lr=max_learning_rate)
        if target_kl is not None
        else None
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch),
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,
    )
    trainer = PPOTrainer(
        collector=collector,
        total_frames=total_frames,
        frame_skip=1,
        optim_steps_per_batch=frames_per_batch // minibatch_size,
        num_epochs=epochs,
        loss_module=loss_module,
        optimizer=optimizer,
        clip_norm=max_grad_norm,
        add_gae=False,
        enable_logging=False,
        progress_bar=False,
        auto_log_optim_steps=False,
        reward_key=REWARD_KEY,
        action_key=env.action_key,
        done_key=("agents", "done"),
        terminated_key=("agents", "terminated"),
        weight_update_map={"policy": "collection_policy"},
        checkpoint=Checkpoint(rng=GlobalRNGState()),
        save_trainer_file=save_trainer_file,
        save_trainer_interval=frames_per_batch,
    )
    trainer.collection_policy = collection_policy
    hooks = GameTrainingHooks(
        trainer,
        actor,
        critic,
        replay_buffer,
        scheduler=scheduler,
        train_team=train_team,
        critic_warmup_iterations=critic_warmup_iterations,
        actor_iterations=math.ceil(total_frames / frames_per_batch)
        - critic_warmup_iterations,
        reference_kl_coeff=reference_kl_coeff,
        reference_kl_final_coeff=reference_kl_final_coeff,
        evaluator=evaluator,
        evaluation_interval=evaluation_interval,
        video_recorder=video_recorder,
        video_interval=video_interval,
        best_checkpoint_path=best_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        policy_kwargs=policy_kwargs,
        config=config,
        logger=logger,
        iteration_callback=iteration_callback,
        collection_metrics=_collection_metrics,
        evaluation_score=evaluation_score,
        save_checkpoint=save_checkpoint,
    )
    trainer.game_hooks = hooks
    trainer.register_module("game_hooks", hooks)
    trainer.register_op("setup", hooks.setup)
    trainer.register_op("batch_process", hooks.prepare)
    rb_hooks = ReplayBufferTrainer(replay_buffer, device=device)
    trainer.register_op("process_optim_batch", rb_hooks.sample)
    trainer.register_op("process_loss", hooks.process_loss)
    trainer.register_op("post_steps", hooks.finish_batch)
    return trainer


def train_mappo(
    env: EnvBase, actor: ProbabilisticActor, critic: TensorDictModule, **kwargs: Any
) -> list[dict[str, float]]:
    """Run :func:`make_trainer` and return its batch metrics.

    Evaluation exports best and latest actors. The live trainer keeps the latest
    parameters paired with its optimizer, so saving and resuming is consistent.
    """
    trainer = make_trainer(env, actor, critic, **kwargs)
    try:
        trainer.train()
    finally:
        trainer.collector.shutdown(close_env=False)
    return trainer.game_hooks.history


# ----------------------------------------------------------------------
# Video
# ----------------------------------------------------------------------


def record_match(
    env: TransformedEnv,
    recorder: VideoRecorder,
    actor: ProbabilisticActor,
    step: int,
    *,
    steps: int,
) -> None:
    """Play one deterministic match on the filmed env and log the clip at ``step``."""
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        env.rollout(steps, actor, break_when_any_done=True)
    recorder.dump(step=step)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def make_training(recipe: DictConfig) -> PPOTrainer:
    """Construct the configured game trainer and its evaluation resources."""
    cfg = recipe
    if cfg.smoke:
        # One match in one process, a few decisions, and a short clip written
        # by a CSV logger: a pipeline check, not a speed test.
        cfg.env.backend = "mujoco"
        cfg.env.parallel = False
        cfg.env.device = "cpu"
        cfg.env.num_envs = 1
        cfg.env.players_per_team = 1
        cfg.env.max_episode_steps = 30
        cfg.policy.hidden_size = 32
        cfg.ppo.total_frames = 40
        cfg.ppo.frames_per_batch = 20
        cfg.ppo.minibatch_size = 10
        cfg.ppo.epochs = 1
        cfg.evaluation.interval = 1
        cfg.evaluation.num_matches = 1
        cfg.evaluation.steps = 6
        cfg.evaluation.video.interval = 1
        cfg.evaluation.video.steps = 6
        cfg.evaluation.video.width = 160
        cfg.evaluation.video.height = 90
        cfg.evaluation.best_checkpoint_path = None
        cfg.evaluation.latest_checkpoint_path = None
        cfg.logger.backend = "csv"
    if cfg.logger.backend == "wandb" and not cfg.logger.entity:
        raise ValueError(
            "W&B logging requires logger.entity so runs do not land in an "
            "unintended default workspace."
        )
    torch.manual_seed(cfg.env.seed)
    config = OmegaConf.to_container(cfg, resolve=True)
    policy_kwargs = {
        "hidden_size": cfg.policy.hidden_size,
        "depth": cfg.policy.depth,
        "initial_policy_scale": cfg.policy.initial_policy_scale,
        "centralized_critic": cfg.policy.centralized_critic,
    }
    env = make_env(cfg)
    skills = cfg.policy.walker_checkpoint is not None
    evaluator = None
    video_env = None
    logger = None
    actor, critic = make_models(env, device=env.device, **policy_kwargs)
    if cfg.policy.init_from:
        trained = load_parameters(cfg.policy.init_from, actor, critic)
        torchrl_logger.info(
            "Initialized actor and critic from %s (%d frames).",
            cfg.policy.init_from,
            trained,
        )
    policy = actor
    iteration_callback = None
    if cfg.policy.opponent_skill is not None and cfg.policy.opponent_checkpoint:
        raise ValueError(
            "policy.opponent_skill and policy.opponent_checkpoint are exclusive."
        )
    if cfg.policy.opponent_skill is not None:
        policy = OpponentPolicy(
            actor,
            cfg.env.players_per_team,
            env.action_key,
            skill=cfg.policy.opponent_skill,
        )
    elif cfg.policy.opponent_checkpoint:
        # Fictitious self-play: red runs a frozen copy of the actor, taken
        # from the checkpoint (``self``: the actor's initial parameters)
        # and refreshed every ``ppo.opponent_refresh_interval`` iterations.
        opponent = deepcopy(actor).requires_grad_(False)
        if cfg.policy.opponent_checkpoint != "self":
            opponent_critic = deepcopy(critic)
            load_parameters(cfg.policy.opponent_checkpoint, opponent, opponent_critic)
        policy = OpponentPolicy(
            actor, cfg.env.players_per_team, env.action_key, opponent=opponent
        )
        refresh_interval = cfg.ppo.opponent_refresh_interval
        if refresh_interval is not None:

            def refresh_opponent(iteration: int) -> None:
                if iteration % refresh_interval == 0:
                    policy.refresh()
                    torchrl_logger.info(
                        "Opponent refreshed at iteration %d.", iteration
                    )

            iteration_callback = refresh_opponent
    if cfg.evaluation.interval is not None:
        evaluator = Evaluator(
            make_env(cfg, num_envs=1, parallel=False),
            policy,
            num_trajectories=cfg.evaluation.num_matches,
            max_steps=cfg.evaluation.steps,
            metrics_fn=football_metrics,
            reward_keys=("next", *REWARD_KEY),
            log_prefix="evaluation",
        )
    mode = "skills" if skills else "joints"
    logger = get_logger(
        cfg.logger.backend,
        logger_name="microduck_football",
        experiment_name=cfg.logger.exp_name
        or generate_exp_name("football", f"{mode}-{cfg.env.backend}"),
        wandb_kwargs={
            "project": cfg.logger.project,
            "entity": cfg.logger.entity,
            "mode": cfg.logger.mode,
            "config": config,
        },
    )
    video_callback = None
    if cfg.evaluation.video.interval is not None and logger is not None:
        # One match filmed by the broadcast camera; with skills the
        # recorder sees one frame per decision.
        fps = 1.0 / CONTROL_PERIOD_S
        if skills:
            fps /= cfg.policy.decision_period
        recorder = VideoRecorder(
            logger, tag="evaluation/match", skip=1, fps=int(round(fps))
        )
        video_env = make_env(
            cfg,
            num_envs=1,
            parallel=False,
            from_pixels=True,
            render_width=cfg.evaluation.video.width,
            render_height=cfg.evaluation.video.height,
        )
        video_env.append_transform(recorder)

        video_callback = ft.partial(
            record_match, video_env, recorder, policy, steps=cfg.evaluation.video.steps
        )

    ppo_kwargs = dict(config["ppo"])
    ppo_kwargs.pop("opponent_refresh_interval")
    trainer = make_trainer(
        env,
        actor,
        critic,
        **ppo_kwargs,
        evaluator=evaluator,
        evaluation_interval=cfg.evaluation.interval,
        video_recorder=video_callback,
        video_interval=cfg.evaluation.video.interval,
        best_checkpoint_path=cfg.evaluation.best_checkpoint_path,
        latest_checkpoint_path=cfg.evaluation.latest_checkpoint_path,
        policy_kwargs=policy_kwargs,
        config=config,
        logger=logger,
        collection_policy=policy,
        iteration_callback=iteration_callback,
    )
    trainer.game_hooks.video_env = video_env
    trainer.register_op("shutdown", trainer.game_hooks.close)
    if cfg.get("resume"):
        trainer.load_from_file(cfg.resume)
    return trainer
