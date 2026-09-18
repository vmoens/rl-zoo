"""Evaluate a game export against fixed opponents and declared task baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

import torch
from torchcodec.encoders import VideoEncoder
from torchrl.envs import TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.render import load_checkpoint

from .football import OpponentPolicy, make_env, make_render_policy


def main() -> None:
    """Record individual held-out episodes, conditions, outcomes and videos."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1100, 1101, 1102, 1103]
    )
    parser.add_argument("--opponent-skills", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--sensor-delay", type=float, default=0)
    parser.add_argument("--sensor-dropout", type=float, default=0)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        help="Override the recorded physical-step horizon for a diagnostic evaluation.",
    )
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.max_episode_steps is not None and args.max_episode_steps <= 0:
        parser.error("--max-episode-steps must be positive")
    torch.set_num_threads(1)
    payload = load_checkpoint(args.checkpoint)
    config = deepcopy(payload["config"])
    game = config["game"]["name"]
    competitive = game in ("football", "tag", "hide_and_seek", "ctf")
    roles = ("blue", "red") if game in ("tag", "hide_and_seek") else (None,)
    conditions = (
        [
            (team, skill, role, "learned")
            for team in ("blue", "red")
            for skill in args.opponent_skills
            for role in roles
        ]
        if competitive
        else [
            ("both", None, None, baseline)
            for baseline in ("learned", "stand", "forward", "single_forward")
        ]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = []
    for team, opponent_skill, role, baseline in conditions:
        for seed in args.seeds:
            cfg = deepcopy(config)
            cfg["env"].update(seed=seed, num_envs=1, parallel=False, download=True)
            if args.max_episode_steps is not None:
                cfg["env"]["max_episode_steps"] = args.max_episode_steps
            if role is not None:
                cfg["game"]["options"]["role"] = role
            if cfg["observations"]["mode"] == "proprioception_vision":
                cfg["observations"]["sensors"].update(
                    camera_delay_s=args.sensor_delay, camera_dropout=args.sensor_dropout
                )
            env = make_env(cfg)
            try:
                actor = make_render_policy(env, checkpoint=payload)
                actor.load_state_dict(payload["model_state_dict"])
                actor.eval()
                policy = (
                    OpponentPolicy(
                        actor,
                        cfg["env"]["players_per_team"],
                        env.action_key,
                        skill=opponent_skill,
                        opponent_team="red" if team == "blue" else "blue",
                    )
                    if competitive
                    else actor
                )
                base = env
                while isinstance(base, TransformedEnv):
                    base = base.base_env
                td = env.reset()
                period = cfg["policy"]["control_steps_per_decision"]
                horizon = math.ceil(cfg["env"]["max_episode_steps"] / period)
                return_per_duck = torch.zeros(base.num_agents)
                falls = torch.zeros(base.num_agents, dtype=torch.long)
                previous_fallen = torch.zeros(base.num_agents, dtype=torch.bool)
                skills = torch.zeros(
                    base.num_agents,
                    len(cfg["policy"]["skill_ids"]),
                    dtype=torch.long,
                )
                spectator, egocentric = [], []
                trajectory = []
                first_event_seconds = None
                boundary_samples = torch.zeros(base.num_agents, dtype=torch.long)
                film = args.video and seed == args.seeds[0] and baseline == "learned"
                positions = base.get_state()["qpos"]
                box_initial = positions[0, -7:-5].clone() if game == "pushing" else None
                with (
                    torch.no_grad(),
                    set_exploration_type(ExplorationType.DETERMINISTIC),
                ):
                    for step in range(horizon):
                        if film:
                            spectator.append(
                                base.render(camera_id=0, width=640, height=360)[0].cpu()
                            )
                            if ("agents", "camera_pixels") in td.keys(True, True):
                                egocentric.append(
                                    torch.cat(
                                        list(
                                            td["agents", "camera_pixels"][0].unbind(0)
                                        ),
                                        dim=1,
                                    ).cpu()
                                )
                        td = policy(td)
                        if baseline != "learned":
                            td[env.action_key].fill_(0 if baseline == "stand" else 1)
                            if baseline == "single_forward":
                                td[env.action_key][..., 1:] = 0
                        chosen = td[env.action_key][0].cpu()
                        skills.scatter_add_(
                            1,
                            chosen[:, None],
                            torch.ones(base.num_agents, 1, dtype=torch.long),
                        )
                        transition = env.step(td)
                        nxt = transition["next"]
                        return_per_duck += nxt["agents", "reward"][0, :, 0].cpu()
                        fallen = nxt["agents", "fallen"][0, :, 0].cpu()
                        falls += fallen & ~previous_fallen
                        previous_fallen = fallen
                        if game != "football":
                            physical = base.get_state()
                            ducks, _ = base._ducks(physical)
                            xy = ducks[0, :, :2].cpu()
                            boundary_samples += (
                                xy.abs() > xy.new_tensor([base.length, base.width]) / 2
                            ).any(-1)
                            seconds = (
                                float(base._step_count[0])
                                * base.frame_skip
                                * base._backend.timestep
                            )
                            events = float(nxt["events_total"].item())
                            if first_event_seconds is None and events > 0:
                                first_event_seconds = seconds
                            sample = {
                                "seconds": seconds,
                                "duck_xy": xy.tolist(),
                                "duck_yaw": base._yaw(ducks[0, :, 3:7]).tolist(),
                                "skills": chosen.tolist(),
                                "events_total": events,
                            }
                            for key in (
                                "captured",
                                "carrier",
                                "flag_status",
                                "flag_position",
                                "checkpoint",
                            ):
                                if key in base._game_state:
                                    sample[key] = base._game_state[key][0].tolist()
                            if game == "pushing":
                                sample["box_xy"] = physical["qpos"][0, -7:-5].tolist()
                            trajectory.append(sample)
                        if nxt["done"].any():
                            break
                        td = env.step_mdp(transition)
                outcome = float(
                    nxt.get("outcome", nxt.get("goal", torch.zeros(1))).item()
                )
                row = {
                    "seed": seed,
                    "learner_team": team,
                    "seeker_team": role,
                    "baseline": baseline,
                    "opponent_skill": opponent_skill,
                    "decisions": step + 1,
                    "outcome": outcome,
                    "learner_win": outcome == (1 if team != "red" else -1),
                    "undecided": outcome == 0,
                    "returns": return_per_duck.tolist(),
                    "fall_events": falls.tolist(),
                    "skill_counts": skills.tolist(),
                    "physics_error": bool(
                        nxt.get("physics_error", torch.zeros(1)).any()
                    ),
                }
                if game != "football":
                    row["events"] = float(nxt["events_total"].item())
                    row["first_event_seconds"] = first_event_seconds
                    row["boundary_violation_samples"] = boundary_samples.tolist()
                    row["trajectory"] = trajectory
                if game == "pushing":
                    box = base.get_state()["qpos"][0, -7:-5]
                    row.update(
                        box_displacement=(box - box_initial).tolist(),
                        final_target_distance=float(
                            (box - torch.tensor([0.8, 0])).norm()
                        ),
                    )
                if game == "relay":
                    row["checkpoints"] = int(base._game_state["checkpoint"].item())
                if film:
                    prefix = f"{team}-opponent{opponent_skill}-role{role}-seed{seed}"
                    for name, frames in (
                        ("spectator", spectator),
                        ("egocentric", egocentric),
                    ):
                        if frames:
                            path = args.output / f"{prefix}-{name}.mp4"
                            VideoEncoder(
                                frames=torch.stack(frames).movedim(-1, 1),
                                frame_rate=50 / period,
                            ).to_file(str(path))
                            row[f"{name}_video"] = path.name
                episodes.append(row)
                # Persist after each complete episode, so a lengthy visual suite
                # can be audited without waiting for all scenarios to finish.
                report = {
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": hashlib.sha256(
                        args.checkpoint.read_bytes()
                    ).hexdigest(),
                    "training_frames": payload.get("frames"),
                    "game": game,
                    "evaluation_dependencies": {
                        name: version(name)
                        for name in (
                            "torchrl",
                            "torchrl-zoo",
                            "torch",
                            "tensordict",
                            "mujoco",
                            "torchcodec",
                        )
                    },
                    "evaluation": {
                        "seeds": args.seeds,
                        "opponent_skills": args.opponent_skills,
                        "sensor_delay_s": args.sensor_delay,
                        "sensor_dropout": args.sensor_dropout,
                        "max_episode_steps": args.max_episode_steps,
                        "deterministic": True,
                    },
                    "config": config,
                    "episodes": episodes,
                }
                (args.output / "evaluation.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
            finally:
                env.close()


if __name__ == "__main__":
    main()
