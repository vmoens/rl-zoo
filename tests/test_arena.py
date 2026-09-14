"""Arena state transitions and inactive-agent learning boundaries."""

from __future__ import annotations

import mujoco
import pytest
import torch
from _fixtures import write_microduck_fixture
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import MicroDuckEnv, microduck_skill_env
from torchrl.envs.utils import check_env_specs

from torchrl_zoo.microduck.football import make_models, make_trainer
from torchrl_zoo.microduck.games.ctf import MicroDuckCTFEnv
from torchrl_zoo.microduck.games.hide_and_seek import MicroDuckHideAndSeekEnv
from torchrl_zoo.microduck.games.pushing import MicroDuckPushingEnv
from torchrl_zoo.microduck.games.relay import MicroDuckRelayEnv
from torchrl_zoo.microduck.games.tag import MicroDuckTagEnv
from torchrl_zoo.microduck.metrics import collection_metrics, evaluation_score


@pytest.fixture
def tag(tmp_path):
    env = MicroDuckTagEnv(
        microduck_root=write_microduck_fixture(tmp_path),
        root=tmp_path / "cache",
        players_per_team=2,
        role="blue",
        spawn_noise=0,
        seed=0,
    )
    env.reset()
    yield env
    env.close()


def test_tag_capture_once_inactive_until_round_end(tag):
    state = tag._state_td()
    q, _ = tag._ducks(state)
    q[0, :, :2] = torch.tensor([[0.0, 0.0], [-0.1, 0.0], [0.1, 0.0], [1.0, 0.0]])
    tag._backend.reset(state["qpos"], state["qvel"])
    tag._fallen.zero_()
    reward, done = tag._step_game(state, state)
    assert tag._game_state["captured"].tolist() == [[False, False, True, False]]
    assert tag._game_state["active"].tolist() == [[True, True, False, True]]
    assert tag._game_state["event"].item() == 1
    assert not done.any()
    torch.testing.assert_close(reward[0, :, 0], torch.tensor([5.0, 5.0, -5.0, -5.0]))
    reward, _ = tag._step_game(state, state)
    assert tag._game_state["event"].item() == 0
    assert reward.count_nonzero() == 0
    # The native step also freezes the captured duck's state.
    td = tag._build_obs_dict(state)
    action = tag.full_action_spec.rand()
    action.update(td)
    before = tag._backend.qpos.clone()
    tag.step(action)
    torch.testing.assert_close(tag._backend.qpos[:, 42:63], before[:, 42:63])
    q[0, 3, :2] = torch.tensor([0.0, 0.1])
    _, done = tag._step_game(state, state)
    assert done.all()
    assert tag._game_state["outcome"].item() == 1
    tag.reset()
    assert tag._game_state["active"].all()
    assert not tag._game_state["captured"].any()


def test_tag_requires_upright_seeker_and_target(tag):
    state = tag._state_td()
    q, _ = tag._ducks(state)
    q[0, :, :2] = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.1, 0.0], [-1.0, -1.0]])
    for fallen in (0, 2):
        tag._fallen.zero_()
        tag._fallen[0, fallen] = True
        _, done = tag._step_game(state, state)
        assert not tag._game_state["captured"].any()
        assert not done.any()
    tag._fallen.zero_()
    tag._step_count.fill_(tag.max_episode_steps)
    q[0, 2, 0] = 0.5
    reward, done = tag._step_game(state, state)
    assert not done.any()  # Time limits truncate; captures terminate.
    assert tag._game_state["outcome"].item() == -1
    assert reward[0, 0, 0] == -10


def test_capture_cuts_gae_and_excludes_inactive_and_opponent_losses(tag):
    walker = TensorDictModule(
        nn.Linear(56, 14), in_keys=["observation"], out_keys=["action"]
    ).requires_grad_(False)
    for parameter in walker.parameters():
        nn.init.zeros_(parameter)
    env = microduck_skill_env(
        tag,
        walker,
        [MicroDuckEnv.standing_task(), MicroDuckEnv.tracking_task(0.2)],
        steps=1,
    )
    actor, critic = make_models(env, hidden_size=8, depth=1)
    trainer = make_trainer(
        env,
        actor,
        critic,
        total_frames=8,
        frames_per_batch=8,
        minibatch_size=8,
        epochs=1,
        train_team="blue",
        collection_metrics_fn=collection_metrics,
        evaluation_score_fn=evaluation_score,
    )
    try:
        batch = next(iter(trainer.collector))
        batch["agents", "active"][:, 3:, 0] = False
        batch["next", "agents", "active"][:, 2:, 0] = False
        prepared = trainer.game_hooks.prepare(batch.clone())
        assert prepared["next", "agents", "terminated"][0, 2, 0]
        torch.testing.assert_close(
            prepared["value_target"][0, 2, 0],
            batch["next", "agents", "reward"][0, 2, 0],
        )
        mask = prepared["agents", "train_mask"]
        assert not mask[:, 3:, 0].any()
        assert not mask[..., 2:, :].any()
        expected = trainer.loss_module(prepared)
        poisoned = prepared.clone()
        poisoned["advantage"][~mask] = 1e6
        poisoned["value_target"][~mask] = -1e6
        actual = trainer.loss_module(poisoned)
        for key in ("loss_objective", "loss_entropy", "loss_critic", "kl_approx"):
            torch.testing.assert_close(actual[key], expected[key])
    finally:
        trainer.collector.shutdown()


def test_unstable_physics_cannot_score_after_mujoco_reset(tag, monkeypatch):
    def unstable_step(ctrl, steps):
        tag._backend._d.warning.number[int(mujoco.mjtWarning.mjWARN_BADQACC)] += 1

    monkeypatch.setattr(tag._backend, "step", unstable_step)
    td = tag.reset()
    td.update(tag.full_action_spec.zero())
    result = tag.step(td)["next"]
    assert result["terminated"].all()
    assert result["physics_error"].all()
    assert not result["event"].any()
    assert not result["outcome"].any()
    assert (result["agents", "reward"] < 0).all()


@pytest.mark.parametrize("players", [1, 2])
def test_tag_balanced_roles_spawns_and_specs(tmp_path, players):
    env = MicroDuckTagEnv(
        microduck_root=write_microduck_fixture(tmp_path),
        root=tmp_path / "cache",
        players_per_team=players,
        spawn_noise=0,
        seed=0,
    )
    try:
        rounds = []
        for _ in range(4):
            env.reset()
            rounds.append(
                (env._game_state["seeker_team"].item(), env._backend.qpos[0, 0].item())
            )
        assert rounds[0][0] != rounds[1][0]
        assert rounds[2][0] != rounds[3][0]
        assert rounds[0][1] == -rounds[2][1]
        check_env_specs(env)
    finally:
        env.close()


@pytest.fixture(
    params=[
        MicroDuckCTFEnv,
        MicroDuckPushingEnv,
        MicroDuckRelayEnv,
        MicroDuckHideAndSeekEnv,
    ]
)
def arena(request, tmp_path):
    env = request.param(
        microduck_root=write_microduck_fixture(tmp_path),
        root=tmp_path / "cache",
        spawn_noise=0,
        seed=0,
    )
    env.reset()
    yield env
    env.close()


def test_native_game_specs_and_reset(arena):
    check_env_specs(arena)
    arena.rollout(5)
    reset = arena.reset()
    assert reset["event"].item() == 0
    assert reset["outcome"].item() == 0


def test_ctf_drop_return_and_simultaneous_pickups(tmp_path):
    env = MicroDuckCTFEnv(
        microduck_root=write_microduck_fixture(tmp_path),
        root=tmp_path / "cache",
        spawn_noise=0,
    )
    try:
        env.reset()
        state = env._state_td()
        q, _ = env._ducks(state)
        # Both blue ducks reach red's flag. The nearest gets sole ownership.
        q[0, :, :2] = torch.tensor(
            [[1.25, 0.0], [1.28, 0.0], [-0.3, -0.5], [-0.3, 0.5]]
        )
        env._step_game(state, state)
        assert env._game_state["carrier"].tolist() == [[-1, 1]]
        # An upright red duck tags the carrier, then returns its dropped flag.
        q[0, 2, :2] = q[0, 1, :2] + torch.tensor([-0.1, 0.0])
        env._step_game(state, state)
        assert env._game_state["carrier"].tolist() == [[-1, -1]]
        assert env._game_state["flag_status"].tolist() == [[0, 0]]
        assert env._game_state["cooldown"][0, 1] > 0
        # Returns have priority; nearby blue cannot repick the returned flag.
        torch.testing.assert_close(
            env._game_state["flag_position"][0, 1], torch.tensor([1.3, 0.0])
        )
        # Both flags stolen in the same step: neither team may score at home.
        env.reset()
        q[0, :, :2] = torch.tensor([[1.3, 0.0], [0.0, -0.5], [-1.3, 0.0], [0.0, 0.5]])
        env._step_game(state, state)
        assert env._game_state["carrier"].tolist() == [[2, 0]]
        q[0, 0, :2] = torch.tensor([-1.3, 0.0])
        q[0, 2, :2] = torch.tensor([1.3, 0.0])
        reward, done = env._step_game(state, state)
        assert not done.any()
        assert not reward.any()
        # Dropping the blue flag allows a friendly return, then blue's capture.
        env._fallen[0, 2] = True
        q[0, 1, :2] = q[0, 2, :2]
        reward, done = env._step_game(state, state)
        assert done.all()
        assert env._game_state["outcome"].item() == 1
        torch.testing.assert_close(
            reward[0, :, 0], torch.tensor([10.0, 10.0, -10.0, -10.0])
        )
    finally:
        env.close()


def test_pushing_requires_continuous_settled_delivery(tmp_path):
    env = MicroDuckPushingEnv(
        microduck_root=write_microduck_fixture(tmp_path),
        root=tmp_path / "cache",
        dwell_seconds=0.06,
    )
    try:
        env.reset()
        state = env._state_td()
        state["qpos"][..., -7:-5] = torch.tensor([0.8, 0.0])
        state["qvel"].zero_()
        for _ in range(2):
            reward, done = env._step_game(state, state)
            assert not done.any()
            assert not reward.any()
        state["qvel"][..., -6] = 1  # Sliding through the target is not a delivery.
        env._step_game(state, state)
        assert env._game_state["dwell"].item() == 0
        state["qvel"].zero_()
        for _ in range(3):
            reward, done = env._step_game(state, state)
        assert done.all()
        torch.testing.assert_close(reward, torch.full((1, 2, 1), 10.0))
        reward, _ = env._step_game(state, state)
        assert not reward.any()  # Pay the event once.
    finally:
        env.close()


def test_relay_order_and_single_handoff(tmp_path):
    env = MicroDuckRelayEnv(
        microduck_root=write_microduck_fixture(tmp_path), root=tmp_path / "cache"
    )
    try:
        env.reset()
        state = env._state_td()
        q, _ = env._ducks(state)
        q[0, 0, :2] = torch.tensor([0.8, 0.2])
        env._step_game(state, state)
        assert env._game_state["checkpoint"].item() == 0
        q[0, 0, :2] = torch.tensor([-0.6, -0.1])
        env._step_game(state, state)
        assert env._game_state["checkpoint"].item() == 1
        q[0, 0, :2] = torch.tensor([0.0, 0.2])
        q[0, 1, :2] = torch.tensor([0.5, 0.2])
        env._step_game(state, state)
        assert env._game_state["carrier"].item() == 0
        q[0, 1, :2] = torch.tensor([0.1, 0.2])
        env._fallen[0, 1] = True
        env._step_game(state, state)
        assert env._game_state["checkpoint"].item() == 1
        env._fallen.zero_()
        env._step_game(state, state)
        assert env._game_state["checkpoint"].item() == 2
        assert env._game_state["carrier"].item() == 1
        env._step_game(state, state)
        assert env._game_state["event"].item() == 0
        assert env._game_state["carrier"].item() == 1
        q[0, 0, :2] = torch.tensor([0.8, 0.2])
        _, done = env._step_game(state, state)
        assert not done.any()  # The non-carrier cannot finish.
        q[0, 1, :2] = torch.tensor([0.8, 0.2])
        _, done = env._step_game(state, state)
        assert done.all()
    finally:
        env.close()


def test_discovery_requires_unoccluded_sustained_visibility(tmp_path):
    env = MicroDuckHideAndSeekEnv(
        microduck_root=write_microduck_fixture(tmp_path),
        root=tmp_path / "cache",
        role="blue",
        preparation_seconds=0.04,
        discovery_seconds=0.04,
        spawn_noise=0,
    )
    try:
        env.reset()
        assert env._game_state["active"].tolist() == [[False, True]]
        state = env._state_td()
        q, _ = env._ducks(state)
        q[0, :, :2] = torch.tensor([[-0.3, 0.0], [0.3, 0.0]])
        q[0, :, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        env._backend.reset(state["qpos"], state["qvel"])
        env._step_count.fill_(2)
        for _ in range(3):
            env._step_game(state, state)
        assert not env._game_state["captured"].any()  # Centre cover occludes.
        q[0, :, 1] = -0.6
        env._backend.reset(state["qpos"], state["qvel"])
        env._step_game(state, state)
        assert not env._game_state["captured"].any()
        assert env._game_state["seen_steps"].sum() == 1
        q[0, 1, :2] = torch.tensor([-0.3, -0.9])  # Outside camera FOV resets the timer.
        env._backend.reset(state["qpos"], state["qvel"])
        env._step_game(state, state)
        assert env._game_state["seen_steps"].sum() == 0
        q[0, 1, :2] = torch.tensor([0.3, -0.6])
        env._backend.reset(state["qpos"], state["qvel"])
        for _ in range(2):
            _, done = env._step_game(state, state)
        assert done.all()
    finally:
        env.close()


if __name__ == "__main__":
    pytest.main([__file__])
