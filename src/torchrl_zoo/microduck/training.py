"""Game-specific hooks for TorchRL's shared PPO training loop."""

from __future__ import annotations

from copy import deepcopy

import torch
from tensordict import TensorDictBase

from torchrl import torchrl_logger


class GameTrainingHooks:
    """Preserve curriculum, evaluation ordering and resume state between batches."""

    def __init__(
        self,
        trainer,
        actor,
        critic,
        replay_buffer,
        *,
        scheduler,
        train_team,
        critic_warmup_iterations,
        actor_iterations,
        reference_kl_coeff,
        reference_kl_final_coeff,
        evaluator,
        evaluation_interval,
        video_recorder,
        video_interval,
        best_checkpoint_path,
        latest_checkpoint_path,
        policy_kwargs,
        config,
        logger,
        iteration_callback,
        collection_metrics,
        evaluation_score,
        save_checkpoint,
    ):
        self.trainer = trainer
        self.actor = actor
        self.critic = critic
        self.replay_buffer = replay_buffer
        self.scheduler = scheduler
        self.train_team = train_team
        self.critic_warmup_iterations = critic_warmup_iterations
        self.actor_iterations = actor_iterations
        self.reference_kl_coeff = reference_kl_coeff
        self.reference_kl_final_coeff = reference_kl_final_coeff
        self.reference = (
            deepcopy(actor).requires_grad_(False).eval()
            if max(reference_kl_coeff, reference_kl_final_coeff) > 0
            else None
        )
        self.evaluator = evaluator
        self.evaluation_interval = evaluation_interval
        self.video_recorder = video_recorder
        self.video_interval = video_interval
        self.best_checkpoint_path = best_checkpoint_path
        self.latest_checkpoint_path = latest_checkpoint_path
        self.policy_kwargs = policy_kwargs
        self.config = config
        self.logger = logger
        self.iteration_callback = iteration_callback
        self.collection_metrics = collection_metrics
        self.evaluation_score = evaluation_score
        self.save_checkpoint = save_checkpoint
        self.iteration = 0
        self.evaluations = 0
        self.best_score = None
        self.history = []
        self.metrics = {}
        self.updates = []
        self.video_env = None

    @property
    def reference_weight(self) -> float:
        fraction = min(
            1.0,
            max(
                0.0,
                (self.iteration - self.critic_warmup_iterations - 1)
                / max(self.actor_iterations - 1, 1),
            ),
        )
        return self.reference_kl_coeff + fraction * (
            self.reference_kl_final_coeff - self.reference_kl_coeff
        )

    def setup(self) -> None:
        if self.evaluation_interval is not None and self.evaluations == 0:
            self.log(self.evaluate())

    @torch.no_grad()
    def prepare(self, batch: TensorDictBase) -> TensorDictBase:
        self.iteration += 1
        self.metrics = self.collection_metrics(batch)
        self.metrics["collection/frames"] = float(batch.numel())
        self.updates = []
        device = next(self.actor.parameters()).device
        batch = batch.to(device)
        reward = batch["next", "agents", "reward"]
        for key in ("done", "terminated"):
            batch["next", "agents", key] = (
                batch["next", key].unsqueeze(-1).expand_as(reward)
            )
        batch = self.trainer.loss_module.value_estimator(batch)
        advantage = batch["advantage"]
        mask = torch.ones_like(advantage, dtype=torch.bool)
        if self.train_team == "blue":
            mask[..., advantage.shape[-2] // 2 :, :] = False
        batch["agents", "train_mask"] = mask
        if self.train_team == "blue":
            selected = advantage[mask]
            scale = (
                selected.std(unbiased=selected.numel() > 1)
                + torch.finfo(advantage.dtype).eps
            )
            batch["advantage"] = torch.where(
                mask, (advantage - selected.mean()) / scale, 0.0
            )
        target = batch["value_target"][mask]
        error = target - batch["agents", "state_value"][mask]
        self.metrics["value/explained_variance"] = float(
            1 - error.var() / target.var().clamp_min(torch.finfo(target.dtype).eps)
        )
        self.replay_buffer.empty()
        self.replay_buffer.extend(batch.reshape(-1).cpu())
        return batch

    def process_loss(
        self, batch: TensorDictBase, losses: TensorDictBase
    ) -> TensorDictBase:
        if self.iteration <= self.critic_warmup_iterations:
            # Detaching removes these gradients, including optimizer momentum
            # updates that a zero-valued actor loss would still permit.
            losses["loss_objective"] = losses["loss_objective"].detach()
            losses["loss_entropy"] = losses["loss_entropy"].detach()
        elif self.reference is not None:
            with torch.no_grad():
                prior = self.reference.get_dist(batch)
            kl = torch.distributions.kl_divergence(prior, self.actor.get_dist(batch))
            mask = batch["agents", "train_mask"].squeeze(-1)
            reference_kl = kl[mask].mean()
            losses["loss_reference"] = self.reference_weight * reference_kl
            losses["reference_kl"] = reference_kl.detach()
        self.updates.append(
            {key: float(value.detach().mean()) for key, value in losses.items()}
        )
        return losses

    def evaluate(self) -> dict[str, float]:
        step = self.trainer.collected_frames
        result = self.evaluator.evaluate(
            weights=self.trainer.collection_policy, step=step
        )
        metrics = {
            key.replace("/custom/", "/"): float(value)
            for key, value in result.items()
            if isinstance(value, (int, float))
        }
        if (
            self.video_recorder is not None
            and self.evaluations % self.video_interval == 0
        ):
            self.video_recorder(step)
        self.evaluations += 1
        score = self.evaluation_score(metrics, train_team=self.train_team)
        is_best = self.best_score is None or score > self.best_score
        if is_best:
            self.best_score = score
        paths = [self.latest_checkpoint_path]
        if is_best:
            paths.append(self.best_checkpoint_path)
        for path in paths:
            if path is not None:
                self.save_checkpoint(
                    path,
                    self.actor,
                    self.critic,
                    frames=step,
                    policy_kwargs=self.policy_kwargs,
                    config=self.config,
                    metrics={**metrics, "evaluation_score": list(score)},
                )
        metrics["evaluation/is_best"] = float(is_best)
        return metrics

    def finish_batch(self) -> None:
        metrics = self.metrics
        if self.updates:
            for key in self.updates[0]:
                metrics[f"ppo/{key}"] = sum(row[key] for row in self.updates) / len(
                    self.updates
                )
        if (
            self.scheduler is not None
            and self.iteration > self.critic_warmup_iterations
        ):
            self.scheduler.step(metrics["ppo/kl_approx"])
        metrics["ppo/learning_rate"] = self.trainer.optimizer.param_groups[0]["lr"]
        metrics["ppo/reference_kl_coeff"] = self.reference_weight
        metrics["progress/frames"] = float(self.trainer.collected_frames)
        if self.evaluation_interval is not None and (
            self.iteration % self.evaluation_interval == 0
            or self.trainer.collected_frames >= self.trainer.total_frames
        ):
            metrics.update(self.evaluate())
        self.history.append(dict(metrics))
        self.log(metrics)
        # Keep evaluation against the previous opponent, then refresh it.
        if self.iteration_callback is not None:
            self.iteration_callback(self.iteration)
        self.replay_buffer.empty()
        torchrl_logger.info(
            "Game PPO frames=%d/%d reward=%+.4f lr=%.2e",
            self.trainer.collected_frames,
            self.trainer.total_frames,
            metrics["collection/reward_mean"],
            metrics["ppo/learning_rate"],
        )

    def log(self, metrics: dict[str, float]) -> None:
        if self.logger is not None:
            for key, value in metrics.items():
                self.logger.log_scalar(key, value, step=self.trainer.collected_frames)

    def close(self) -> None:
        if self.evaluator is not None:
            self.evaluator.shutdown()
            self.evaluator = None
        if self.video_env is not None and not self.video_env.is_closed:
            self.video_env.close()
        if self.logger is not None and hasattr(self.logger.experiment, "finish"):
            self.logger.experiment.finish()
            self.logger = None

    def state_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "evaluations": self.evaluations,
            "best_score": self.best_score,
            "scheduler": None
            if self.scheduler is None
            else self.scheduler.state_dict(),
            "reference": None
            if self.reference is None
            else self.reference.state_dict(),
            "collection_policy": self.trainer.collection_policy.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.iteration = state["iteration"]
        self.evaluations = state["evaluations"]
        self.best_score = state["best_score"]
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.reference is not None:
            self.reference.load_state_dict(state["reference"])
        self.trainer.collection_policy.load_state_dict(state["collection_policy"])
