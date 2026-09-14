"""Arena state transitions and inactive-agent learning boundaries."""

from __future__ import annotations

import functools as ft
from copy import deepcopy

import mujoco
import pytest
import torch
from _fixtures import write_microduck_fixture
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import MicroDuckEnv, microduck_skill_env
from torchrl.envs.utils import ExplorationType, check_env_specs, set_exploration_type
from torchrl.objectives import SoftUpdate

from torchrl_zoo.microduck.football import OpponentPolicy, make_models, make_trainer
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


@pytest.mark.parametrize("observations", ["proprioception", "proprioception_vision"])
@pytest.mark.parametrize("critic_observation", ["state", "actor"])
def test_sensor_selector_sequences_resume_and_opponent_memory(
    tmp_path, observations, critic_observation
):
    torch.manual_seed(4)
    base = MicroDuckTagEnv(
        microduck_root=write_microduck_fixture(tmp_path),
        root=tmp_path / "cache",
        observations=observations,
        players_per_team=1,
        max_episode_steps=12,
        spawn_noise=0,
        seed=0,
    )
    walker = TensorDictModule(
        nn.Linear(56, 14), in_keys=["observation"], out_keys=["action"]
    )
    with torch.no_grad():
        walker.module.weight.zero_()
        walker.module.bias.zero_()
    env = microduck_skill_env(
        base,
        walker,
        [MicroDuckEnv.standing_task(), MicroDuckEnv.tracking_task(0.2)],
        steps=2,
    )
    actor, critic = make_models(
        env,
        hidden_size=8,
        depth=1,
        observations=observations,
        critic_observation=critic_observation,
    )
    opponent = deepcopy(actor).requires_grad_(False)
    with torch.no_grad():
        next(opponent.parameters()).add_(0.5)
    policy = OpponentPolicy(actor, 1, env.action_key, opponent=opponent)
    trainer = make_trainer(
        env,
        actor,
        critic,
        total_frames=48,
        frames_per_batch=12,
        minibatch_size=6,
        epochs=1,
        recurrent_episode_steps=6,
        train_team="blue",
        collection_policy=policy,
        reference_kl_coeff=0.1,
        loss_kwargs={"delay_actor": True},
        target_net_updater=ft.partial(SoftUpdate, eps=0.9),
        collection_metrics_fn=collection_metrics,
        evaluation_score_fn=evaluation_score,
    )
    try:
        reset = env.reset()
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            blue = actor(reset.clone())
            red = opponent(reset.clone())
            combined = policy(reset.clone())
        state_key = ("next", "agents", "selector_state")
        torch.testing.assert_close(
            combined[state_key][..., :1, :, :], blue[state_key][..., :1, :, :]
        )
        torch.testing.assert_close(
            combined[state_key][..., 1:, :, :], red[state_key][..., 1:, :, :]
        )
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            reversed_policy = OpponentPolicy(
                actor, 1, env.action_key, opponent=opponent, opponent_team="blue"
            )
            reversed_output = reversed_policy(reset.clone())
        torch.testing.assert_close(
            reversed_output[state_key][..., :1, :, :], red[state_key][..., :1, :, :]
        )
        torch.testing.assert_close(
            reversed_output[state_key][..., 1:, :, :], blue[state_key][..., 1:, :, :]
        )
        torch.testing.assert_close(
            reversed_output[env.action_key][..., :1], red[env.action_key][..., :1]
        )
        torch.testing.assert_close(
            reversed_output[env.action_key][..., 1:], blue[env.action_key][..., 1:]
        )
        # With identical allowed inputs, changing all privileged state has no
        # effect, and the exported actor needs none of those fields.
        allowed = reset.select(*actor.in_keys, strict=False)
        poisoned = reset.clone()
        poisoned["agents", "observation"].fill_(float("nan"))
        with torch.no_grad():
            expected = actor.get_dist(allowed.clone()).probs
            actual = actor.get_dist(poisoned).probs
        torch.testing.assert_close(actual, expected)
        # One duck's reset must not reset the other duck's memory.
        memory = reset.clone()
        memory["agents", "selector_state"].fill_(1)
        memory["agents", "is_init"].zero_()
        with torch.no_grad():
            normal = actor(memory.clone())
        memory["agents", "is_init"][..., 0, :] = True
        with torch.no_grad():
            partial = actor(memory)
        torch.testing.assert_close(
            partial[state_key][..., 1, :, :], normal[state_key][..., 1, :, :]
        )
        assert not torch.equal(
            partial[state_key][..., 0, :, :], normal[state_key][..., 0, :, :]
        )
        batch = next(iter(trainer.collector)).clone()
        prepared = trainer.game_hooks.prepare(batch.clone())
        with torch.no_grad():
            values = critic(prepared.clone())["agents", "state_value"]
            following = critic(prepared["next"].clone())["agents", "state_value"]
            reward = prepared["next", "agents", "reward"]
            done = prepared["next", "agents", "done"]
            terminated = prepared["next", "agents", "terminated"]
            expected = torch.zeros_like(values)
            future = torch.zeros_like(values[0])
            for t in reversed(range(prepared.shape[0])):
                delta = reward[t] + 0.99 * following[t] * (~terminated[t]) - values[t]
                future = delta + 0.99 * 0.95 * (~done[t]) * future
                expected[t] = future + values[t]
            torch.testing.assert_close(
                prepared["value_target"], expected, atol=1e-5, rtol=1e-5
            )
        sample = trainer.game_hooks.sample(prepared)
        assert sample.names == ["time"]
        assert sample["agents", "is_init"][0].all()
        assert sample["next", "done"][-1].all()
        # Recurrent evaluation must match explicit, isolated, sequential steps.
        with torch.no_grad():
            sequence = actor(sample.clone())
            state = None
            for t in range(sample.shape[0]):
                step = sample[t].clone()
                if state is not None:
                    step["agents", "selector_state"] = state
                output = actor(step)
                torch.testing.assert_close(
                    sequence["agents", "logits"][t],
                    output["agents", "logits"],
                    atol=1e-5,
                    rtol=1e-5,
                )
                state = output[state_key]
        original = trainer.loss_module(sample.clone())
        poisoned = sample.clone()
        mask = poisoned["agents", "train_mask"]
        poisoned["advantage"][~mask] = 1e6
        poisoned["value_target"][~mask] = -1e6
        actual = trainer.loss_module(poisoned)
        for key in ("loss_objective", "loss_entropy", "loss_critic", "kl_approx"):
            torch.testing.assert_close(actual[key], original[key])
        trainer.collected_frames = batch.numel()
        trainer.optim_steps(prepared)
        trainer._post_steps_hook()
        checkpoint = tmp_path / "visual.trainer.ckpt"
        trainer.checkpoint.save(checkpoint)
        results = []
        for _ in range(2):
            trainer.load_from_file(checkpoint)
            prepared = trainer.game_hooks.prepare(batch.clone())
            trainer.collected_frames += batch.numel()
            trainer.optim_steps(prepared)
            trainer._post_steps_hook()
            results.append(deepcopy(trainer.state_dict()))
        for key, value in results[0]["loss_module"].items():
            torch.testing.assert_close(
                value, results[1]["loss_module"][key], rtol=0, atol=0
            )
    finally:
        trainer.collector.shutdown()


def test_camera_agent_order_mount_and_held_frames(tmp_path):
    env = MicroDuckTagEnv(
        microduck_root=write_microduck_fixture(tmp_path),
        root=tmp_path / "cache",
        observations="proprioception_vision",
        sensor_kwargs={"camera_fps": 10},
        spawn_noise=0,
        players_per_team=2,
        seed=0,
    )
    try:
        td = env.reset()
        check_env_specs(env)
        td = env.reset()
        model, data = env._backend.mj_model, env._backend._d
        for index, name in enumerate(
            (
                "blue0/head_camera",
                "blue1/head_camera",
                "red0/head_camera",
                "red1/head_camera",
            )
        ):
            camera = model.camera(name).id
            forward = -torch.as_tensor(data.cam_xmat[camera].copy()).reshape(3, 3)[:, 2]
            q, _ = env._ducks(env._state_td())
            yaw = env._yaw(q[0, index, 3:7])
            expected_forward = torch.stack((yaw.cos(), yaw.sin(), yaw.new_zeros(())))
            assert float(forward @ expected_forward.to(forward)) > 0.9
            expected = env._backend.render(camera_id=camera, width=64, height=64)
            torch.testing.assert_close(
                td["agents", "camera_pixels"][:, index], expected
            )
        stepped = env.rand_step(td)["next"]
        torch.testing.assert_close(
            stepped["agents", "camera_pixels"], td["agents", "camera_pixels"]
        )
        assert (stepped["agents", "camera_age"] > 0).all()
        assert env.reset()["agents", "camera_age"].count_nonzero() == 0
    finally:
        env.close()


if __name__ == "__main__":
    pytest.main([__file__])
