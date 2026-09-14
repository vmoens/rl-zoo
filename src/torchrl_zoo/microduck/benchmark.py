"""Measure native skill-decision throughput before selecting a pilot worker count."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torchrl import timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type

from .football import make_env, make_models


def main() -> None:
    """Benchmark installed game factories with one, two and four native workers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game",
        choices=["football", "tag", "hide_and_seek", "ctf", "pushing", "relay"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    for workers in args.workers:
        with initialize_config_dir(
            config_dir=str(Path(__file__).parent / "conf"), version_base="1.3"
        ):
            config = compose(
                config_name="config",
                overrides=[
                    f"game={args.game}",
                    "env.download=true",
                    f"env.num_envs={workers}",
                    "env.parallel=true",
                ],
            )
        env = make_env(config)
        torch.manual_seed(config.env.seed)
        actor, _ = make_models(env, hidden_size=128, depth=2)
        try:
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                env.rollout(10, actor, break_when_any_done=False)
                with timeit(f"{args.game}/{workers}") as timer:
                    data = env.rollout(args.steps, actor, break_when_any_done=False)
                seconds = timer.elapsed()
            results.append(
                {
                    "workers": workers,
                    "decisions": data.numel(),
                    "seconds": seconds,
                    "decisions_per_second": data.numel() / seconds,
                    "fallen_fraction": float(
                        data["next", "agents", "fallen"].float().mean()
                    ),
                }
            )
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "game": args.game,
                "machine": platform.platform(),
                "threads": 1,
                "seed": config.env.seed,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
