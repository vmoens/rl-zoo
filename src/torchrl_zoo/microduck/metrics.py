"""Outcome metrics for arena games; football retains its original scorecard."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
from tensordict import TensorDictBase


def arena_metrics(trajectories: TensorDictBase) -> dict[str, float]:
    """Measure completed outcomes and events, excluding evaluator padding."""
    mask = trajectories["collector", "mask"]
    lengths = mask.sum(-1)
    last = (lengths - 1).unsqueeze(-1)
    outcome = trajectories["next", "outcome"].squeeze(-1).gather(-1, last)
    events = (
        (trajectories["next", "events_total"] - trajectories["events_total"])
        .squeeze(-1)
        .masked_fill(~mask, 0)
    )
    fallen = trajectories["next", "agents", "fallen"].squeeze(-1) & mask[..., None]
    prior = torch.cat((torch.zeros_like(fallen[..., :1, :]), fallen[..., :-1, :]), -2)
    return {
        "wins_blue": float((outcome > 0).float().mean()),
        "wins_red": float((outcome < 0).float().mean()),
        "undecided": float((outcome == 0).float().mean()),
        "events": float(events.sum(-1).mean()),
        "match_length": float(lengths.float().mean()),
        "falls_per_duck": float((fallen & ~prior).sum(-2).float().mean()),
        "physics_error_rate": float(
            (trajectories["next", "physics_error"].squeeze(-1) & mask)
            .any(-1)
            .float()
            .mean()
        ),
    }


def collection_metrics(data: TensorDictBase) -> dict[str, float]:
    """Summarize rewards, events and completed rounds from a collection batch."""
    reward = data["next", "agents", "reward"]
    active = data["agents", "active"]
    done = data["next", "done"]
    outcome = data["next", "outcome"]
    return {
        "collection/reward_mean": float(reward[active].mean()),
        "collection/events": float(
            (data["next", "events_total"] - data["events_total"]).sum()
        ),
        "episode/finished": float(done.sum()),
        "episode/wins_blue": float(((outcome > 0) & done).sum()),
        "episode/wins_red": float(((outcome < 0) & done).sum()),
        "collection/physics_errors": float(data["next", "physics_error"].sum()),
    }


def evaluation_score(
    metrics: Mapping[str, float], *, train_team: Literal["both", "blue"] = "both"
) -> tuple[float, ...]:
    """Rank blue against a fixed opponent, or completion in symmetric self-play."""
    blue, red = metrics["evaluation/wins_blue"], metrics["evaluation/wins_red"]
    return (
        blue - red if train_team == "blue" else blue + red,
        metrics["evaluation/events"],
        -metrics["evaluation/falls_per_duck"],
    )
