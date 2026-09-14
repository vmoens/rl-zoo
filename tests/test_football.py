"""Football physics, game events and controller integration regressions."""

from __future__ import annotations

import functools as ft
import math
import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from _fixtures import write_microduck_fixture
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torch import nn
from torchrl.envs import (
    MicroDuckEnv,
    microduck_skill_env,
)
from torchrl.envs.custom.mujoco._backends import (
    _has_jax,
    _has_mjx,
    _has_mujoco,
    _has_mujoco_torch,
)
from torchrl.envs.utils import check_env_specs
from torchrl.modules import GRUModule
from torchrl.objectives import SoftUpdate

from torchrl_zoo.microduck import football as football_mappo
from torchrl_zoo.microduck.games.football import (
    FOOTBALL_NUMERIC,
    MicroDuckFootballEnv,
    build_football_scene,
    kickoff_positions,
)

_has_hydra = True
_test_accelerated = os.environ.get("TORCHRL_ZOO_TEST_ACCELERATED") == "1"
if _has_mujoco:
    import mujoco
_AVAILABLE_BACKENDS = [
    name
    for name, installed in (
        ("mujoco", _has_mujoco),
        ("mujoco-torch", _has_mujoco_torch),
        ("mjx", _has_mjx and _has_jax),
    )
    if installed and (name == "mujoco" or _test_accelerated)
]


class TestFootball:
    @pytest.mark.parametrize("ewma", [False, True])
    def test_trainer_resume_matches_next_update_with_frozen_opponent(
        self, tmp_path, ewma
    ):
        torch.manual_seed(7)
        base = self._football_env(tmp_path, max_episode_steps=12)
        low = TensorDictModule(
            nn.Linear(MicroDuckEnv.OBSERVATION_DIM, MicroDuckEnv.NUM_JOINTS),
            in_keys=["observation"],
            out_keys=["action"],
        ).requires_grad_(False)
        env = microduck_skill_env(
            base,
            low,
            [MicroDuckEnv.standing_task(), MicroDuckEnv.tracking_task(0.2)],
            steps=2,
        )
        actor, critic = football_mappo.make_models(env, hidden_size=8, depth=1)
        opponent = deepcopy(actor).requires_grad_(False)
        policy = football_mappo.OpponentPolicy(
            actor, 1, env.action_key, opponent=opponent
        )
        trainer = football_mappo.make_trainer(
            env,
            actor,
            critic,
            total_frames=24,
            frames_per_batch=8,
            minibatch_size=8,
            epochs=1,
            train_team="blue",
            collection_policy=policy,
            reference_kl_coeff=0.5,
            reference_kl_final_coeff=0.1,
            loss_kwargs={"delay_actor": ewma},
            target_net_updater=ft.partial(SoftUpdate, eps=0.9) if ewma else None,
        )
        try:
            batch = next(iter(trainer.collector)).clone()
            prepared = trainer.game_hooks.prepare(batch.clone())
            with torch.no_grad():
                values = critic(batch.clone())["agents", "state_value"]
                next_values = critic(batch["next"].clone())["agents", "state_value"]
                rewards = batch["next", "agents", "reward"]
                expected_advantage = torch.zeros_like(rewards)
                carry = torch.zeros_like(rewards[:, 0])
                for t in reversed(range(batch.shape[-1])):
                    done = batch["next", "done"][:, t].unsqueeze(-1)
                    terminated = batch["next", "terminated"][:, t].unsqueeze(-1)
                    delta = (
                        rewards[:, t]
                        + 0.99 * ~terminated * next_values[:, t]
                        - values[:, t]
                    )
                    carry = delta + 0.99 * 0.95 * ~done * carry
                    expected_advantage[:, t] = carry
                torch.testing.assert_close(
                    prepared["value_target"], expected_advantage + values
                )
                blue = expected_advantage[..., :1, :]
                normalized = (blue - blue.mean()) / (
                    blue.std() + torch.finfo(blue.dtype).eps
                )
                torch.testing.assert_close(
                    prepared["advantage"][..., :1, :], normalized
                )
            reference_loss = deepcopy(trainer.loss_module)
            reference_updater = SoftUpdate(reference_loss, eps=0.9) if ewma else None
            reference_optimizer = torch.optim.Adam(reference_loss.parameters(), lr=3e-4)
            losses = reference_loss(prepared.reshape(-1))
            sum(
                value for key, value in losses.items() if key.startswith("loss_")
            ).backward()
            nn.utils.clip_grad_norm_(reference_loss.parameters(), 1.0)
            reference_optimizer.step()
            if reference_updater is not None:
                reference_updater.step()
            trainer.collected_frames = 8
            trainer.optim_steps(prepared)
            for key, value in reference_loss.state_dict().items():
                torch.testing.assert_close(
                    value, trainer.loss_module.state_dict()[key], atol=1e-5, rtol=1e-5
                )
            trainer._post_steps_hook()
            checkpoint = tmp_path / "trainer.ckpt"
            trainer.checkpoint.save(checkpoint)
            before_opponent = deepcopy(opponent.state_dict())
            # Replay the same rollout after loading, with identical RNG and
            # optimizer/scheduler/curriculum state. Physics resumes at a new
            # episode; the update itself must match exactly.
            results = []
            for _ in range(2):
                trainer.load_from_file(checkpoint)
                prepared = trainer.game_hooks.prepare(batch.clone())
                trainer.collected_frames += 8
                trainer.optim_steps(prepared)
                trainer._post_steps_hook()
                results.append(deepcopy(trainer.state_dict()))
            for key, value in results[0]["loss_module"].items():
                torch.testing.assert_close(
                    value, results[1]["loss_module"][key], rtol=0, atol=0
                )
            assert (
                results[0]["game_hooks"]["scheduler"]
                == results[1]["game_hooks"]["scheduler"]
            )
            for key, value in before_opponent.items():
                torch.testing.assert_close(
                    opponent.state_dict()[key], value, rtol=0, atol=0
                )
        finally:
            trainer.collector.shutdown()

    def _football_env(self, tmp_path: Path, **kwargs):
        """A football env over the MicroDuck fixture with every reset noise off."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        scene = write_microduck_fixture(tmp_path)
        settings = {
            "players_per_team": 1,
            "backend": "mujoco",
            "seed": 0,
            "spawn_noise": 0.0,
            "yaw_noise": 0.0,
            "joint_reset_noise_scale": 0.0,
            "ball_noise": 0.0,
        }
        settings.update(kwargs)
        return MicroDuckFootballEnv(
            microduck_root=scene, root=tmp_path / "cache", **settings
        )

    @staticmethod
    def _football_action(env, value: torch.Tensor | float = 0.0) -> TensorDict:
        action = torch.as_tensor(value, dtype=torch.float32).expand(
            1, env.num_agents, MicroDuckEnv.NUM_JOINTS
        )
        return TensorDict({("agents", "action"): action.clone()}, batch_size=[1])

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    def test_football_scene_builder(self, tmp_path):
        scene = write_microduck_fixture(tmp_path)
        geometry = {"pitch_length": 2.0, "pitch_width": 1.4, "goal_width": 0.5}
        xml = build_football_scene(scene, players_per_team=2, **geometry)
        model = mujoco.MjModel.from_xml_string(xml)
        assert (model.nq, model.nv, model.nu) == (4 * 21 + 7, 4 * 20 + 6, 4 * 14)
        cameras = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
            for index in range(2)
        ]
        assert cameras == list(MicroDuckFootballEnv.CAMERAS)
        actuator = mujoco.mjtObj.mjOBJ_ACTUATOR
        assert mujoco.mj_id2name(model, actuator, 0) == "blue0/actuator0"
        assert mujoco.mj_id2name(model, actuator, model.nu - 1) == "red1/actuator13"
        # The ball's joint comes last, so its state trails the ducks' blocks.
        assert (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt - 1)
            == "ball_free"
        )
        for name in ("blue0/left_foot_collision", "red1/right_foot_collision"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
        for name in ("blue1/left_foot", "red0/right_foot", "blue_goal", "red_goal"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
        numeric = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_NUMERIC, FOOTBALL_NUMERIC
        )
        start = model.numeric_adr[numeric]
        assert model.numeric_data[start : start + 7].tolist() == pytest.approx(
            [2.0, 1.4, 0.5, 0.25, 0.25, 0.035, 2.0]
        )
        # The STAND keyframe puts every duck on its kickoff slot, red mirrored
        # through the center and facing -x, and the ball on the spot.
        assert model.nkey == 1
        key = model.key_qpos[0]
        slots = kickoff_positions(2, 2.0, 1.4)
        assert key[:2].tolist() == pytest.approx(list(slots[0]))
        red = 2 * 21
        assert key[red : red + 2].tolist() == pytest.approx(
            [-slots[0][0], -slots[0][1]]
        )
        assert key[red + 3 : red + 7].tolist() == pytest.approx(
            [0.0, 0.0, 0.0, 1.0], abs=1e-6
        )
        assert key[-7:].tolist() == pytest.approx([0.0, 0.0, 0.035, 1.0, 0.0, 0.0, 0.0])
        # The cache is content addressed.
        cache = tmp_path / "cache"
        first = MicroDuckFootballEnv.write_scene(
            scene, root=cache, players_per_team=2, **geometry
        )
        second = MicroDuckFootballEnv.write_scene(
            scene, root=cache, players_per_team=2, **geometry
        )
        assert first == second
        assert first.read_text() == xml
        assert first != MicroDuckFootballEnv.write_scene(
            scene, root=cache, players_per_team=1
        )
        with pytest.raises(ValueError, match="players_per_team"):
            build_football_scene(scene, players_per_team=0)
        with pytest.raises(ValueError, match="goal_width"):
            build_football_scene(scene, goal_width=3.0, pitch_width=2.0)
        no_key = tmp_path / "no_key.xml"
        no_key.write_text(scene.read_text().replace('name="STAND"', 'name="OTHER"'))
        with pytest.raises(ValueError, match="STAND"):
            build_football_scene(no_key)

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    def test_football_scene_keeps_the_robot_compiler_flags(self, tmp_path):
        # The collision proxies of the walking scene are fitted by the compiler
        # (fitaabb); the attaching spec's compiler must carry the same flags.
        # MjSpec.to_xml bakes the fitted boxes into the geoms and drops the flag
        # itself, so the round trip is checked on a flag the writer keeps.
        import mujoco

        scene = write_microduck_fixture(tmp_path)
        flagged = scene.with_name("robot_flags.xml")
        flagged.write_text(
            scene.read_text().replace(
                '<mujoco model="microduck-test">',
                '<mujoco model="microduck-test">'
                '<compiler fitaabb="true" boundmass="0.001"/>',
            )
        )
        spec = mujoco.MjSpec.from_string(
            build_football_scene(flagged, players_per_team=1)
        )
        assert spec.compiler.boundmass == pytest.approx(0.001)

    @pytest.mark.parametrize("backend", _AVAILABLE_BACKENDS)
    def test_football_env_specs_and_rollout(self, tmp_path, backend):
        num_envs = 1 if backend == "mujoco" else 2
        env = self._football_env(
            tmp_path,
            players_per_team=2,
            backend=backend,
            num_envs=num_envs,
            pitch={"pitch_length": 2.0, "pitch_width": 1.4},
        )
        assert env.num_agents == 4
        assert env.observation_dim == MicroDuckEnv.OBSERVATION_DIM + 26
        assert (env.pitch_length, env.pitch_width) == (2.0, 1.4)
        assert env.action_key == ("agents", "action")
        assert env.reward_key == ("agents", "reward")
        check_env_specs(env)
        observation = env.reset()["agents", "observation"]
        features = observation[..., MicroDuckEnv.OBSERVATION_DIM :]
        # At kickoff every duck faces the goal it attacks from its own half.
        torch.testing.assert_close(
            features[..., 2:4],
            torch.tensor([1.0, 0.0]).expand_as(features[..., 2:4]),
            atol=1e-5,
            rtol=0,
        )
        assert (features[..., 0] < 0).all()
        rollout = env.rollout(4, break_when_any_done=False)
        assert rollout["agents", "observation"].shape == (
            num_envs,
            4,
            4,
            env.observation_dim,
        )
        assert rollout["next", "agents", "reward"].shape == (num_envs, 4, 4, 1)
        assert rollout["next", "goal"].shape == (num_envs, 4, 1)
        assert torch.isfinite(rollout["next", "agents", "reward"]).all()
        assert (rollout["next", "goal"] == 0).all()
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    @pytest.mark.parametrize("scorer", [1, -1])
    def test_football_goal_terminates_and_pays_both_teams(self, tmp_path, scorer):
        env = self._football_env(tmp_path)
        env.reset()
        state = env.get_state()
        qpos, qvel = state["qpos"].clone(), state["qvel"].clone()
        # Roll the ball toward the goal at +x (blue scores) or at -x (red does).
        qpos[0, -7] = scorer * (env.pitch_length / 2 - 0.05)
        qpos[0, -6] = 0.0
        qvel[0, -6] = scorer * 1.0
        env.reset(TensorDict(qpos=qpos, qvel=qvel, batch_size=[1]), set_state=True)
        first = env.step(self._football_action(env))["next"]
        assert first["goal"].item() == 0
        blue, red = first["agents", "reward"][0, :, 0]
        # The ball's progress pays one team and charges the other.
        assert torch.sign(blue).item() == scorer
        torch.testing.assert_close(blue, -red)
        for _ in range(20):
            step = env.step(self._football_action(env))["next"]
            if step["goal"].item() != 0:
                break
        else:
            pytest.fail("the ball never crossed the goal line")
        assert step["goal"].item() == scorer
        assert step["terminated"].item() and step["done"].item()
        blue, red = step["agents", "reward"][0, :, 0]
        weight = MicroDuckFootballEnv.REWARD_WEIGHTS["goal"]
        assert abs(blue.item() - scorer * weight) < 0.5
        assert abs(red.item() + scorer * weight) < 0.5
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    @pytest.mark.parametrize(
        "respawn, mode, delay_s",
        [
            (False, "in_place", 0.0),
            (True, "kickoff", 0.0),
            (True, "in_place", 0.0),
            (True, "in_place", 0.1),
        ],
    )
    def test_football_fallen_duck_is_charged_once_and_respawns(
        self, tmp_path, respawn, mode, delay_s
    ):
        env = self._football_env(
            tmp_path, respawn=respawn, respawn_mode=mode, respawn_delay_s=delay_s
        )
        env.reset()
        state = env.get_state()
        qpos = state["qpos"].clone()
        # Blue's duck lies on its side on the floor, away from its slot.
        qpos[0, 0] += 0.5
        qpos[0, 1] += 0.2
        qpos[0, 2] = 0.02
        qpos[0, 3:7] = torch.tensor(
            [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0]
        )
        env.reset(
            TensorDict(qpos=qpos, qvel=state["qvel"], batch_size=[1]), set_state=True
        )
        first = env.step(self._football_action(env))["next"]
        assert first["agents", "fallen"][0, :, 0].tolist() == [True, False]
        fall = MicroDuckFootballEnv.REWARD_WEIGHTS["fall"]
        assert abs(first["agents", "reward"][0, 0, 0].item() - fall) < 0.05
        assert not first["done"].item()
        duck = env.get_state()["qpos"][0, : MicroDuckFootballEnv.DUCK_NQ]
        if not respawn:
            assert duck[2].item() < 0.05
            second = env.step(self._football_action(env))["next"]
            assert second["agents", "fallen"][0, 0, 0]
            # Down for a second step: no second fall penalty.
            assert second["agents", "reward"][0, 0, 0].item() > fall / 2
            env.close()
            return
        delay_steps = round(delay_s / (env.frame_skip * env._backend.timestep))
        for index in range(delay_steps):
            # Still on the floor, flagged, not charged again.
            assert duck[2].item() < 0.05
            step = env.step(self._football_action(env))["next"]
            assert step["agents", "fallen"][0, 0, 0].item() == (index < delay_steps - 1)
            assert step["agents", "reward"][0, 0, 0].item() > fall / 2
            duck = env.get_state()["qpos"][0, : MicroDuckFootballEnv.DUCK_NQ]
        # Standing again, still, facing the goal blue attacks (+x).
        assert duck[2].item() == pytest.approx(0.12, abs=1e-5)
        assert duck[3:7].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
        slot = kickoff_positions(1, env.pitch_length, env.pitch_width)[0]
        if mode == "kickoff":
            assert duck[:2].tolist() == pytest.approx(list(slot), abs=1e-5)
        else:
            # Where it lay when it got up: it slid a few millimeters on the floor.
            assert duck[:2].tolist() == pytest.approx(
                [slot[0] + 0.5, slot[1] + 0.2], abs=0.02
            )
        after = env.step(self._football_action(env))["next"]
        assert not after["agents", "fallen"].any()
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    def test_football_crowding_charges_each_close_neighbor(self, tmp_path):
        weights = {name: 0.0 for name in MicroDuckFootballEnv.REWARD_WEIGHTS}
        weights["crowd"] = -1.0
        env = self._football_env(tmp_path, players_per_team=2, reward_weights=weights)
        env.reset()
        state = env.get_state()
        qpos = state["qpos"].clone()
        nq = MicroDuckFootballEnv.DUCK_NQ
        # blue0 and blue1 stand 10 cm apart; red0 is 15 cm from both, but only
        # teammates count; red1 is far away.
        for index, (x, y) in enumerate(
            [(0.0, 0.0), (0.1, 0.0), (0.05, 0.14), (1.0, 0.8)]
        ):
            qpos[0, index * nq] = x
            qpos[0, index * nq + 1] = y
        env.reset(
            TensorDict(qpos=qpos, qvel=state["qvel"], batch_size=[1]), set_state=True
        )
        reward = env.step(self._football_action(env))["next"]["agents", "reward"][
            0, :, 0
        ]
        dt = env.frame_skip * env._backend.timestep
        assert reward.tolist() == pytest.approx([-dt, -dt, 0.0, 0.0], abs=1e-6)
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    def test_football_approach_pays_the_closest_ducks_only(self, tmp_path):
        weights = {name: 0.0 for name in MicroDuckFootballEnv.REWARD_WEIGHTS}
        weights["approach_ball"] = 1.0
        env = self._football_env(
            tmp_path, players_per_team=2, reward_weights=weights, approach_players=1
        )
        env.reset()
        state = env.get_state()
        qpos, qvel = state["qpos"].clone(), state["qvel"].clone()
        nq, nv = MicroDuckFootballEnv.DUCK_NQ, MicroDuckFootballEnv.DUCK_NV
        # The ball sits at the origin; blue1 and red0 are its closest ducks
        # and every duck moves toward it at the same speed.
        for index, (x, y) in enumerate(
            [(-0.9, 0.0), (-0.4, 0.0), (0.4, 0.0), (0.9, 0.0)]
        ):
            qpos[0, index * nq] = x
            qpos[0, index * nq + 1] = y
            qvel[0, index * nv] = 0.3 if x < 0 else -0.3
        qpos[0, -MicroDuckFootballEnv.BALL_NQ : -MicroDuckFootballEnv.BALL_NQ + 2] = 0.0
        env.reset(TensorDict(qpos=qpos, qvel=qvel, batch_size=[1]), set_state=True)
        reward = env.step(self._football_action(env))["next"]["agents", "reward"][
            0, :, 0
        ]
        assert reward[0] == 0.0 and reward[3] == 0.0
        assert reward[1] > 0.0 and reward[2] > 0.0
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    def test_football_approach_skips_ducks_that_are_down(self, tmp_path):
        weights = {name: 0.0 for name in MicroDuckFootballEnv.REWARD_WEIGHTS}
        weights["approach_ball"] = 1.0
        env = self._football_env(
            tmp_path,
            players_per_team=2,
            reward_weights=weights,
            approach_players=1,
            respawn=False,
        )
        env.reset()
        state = env.get_state()
        qpos, qvel = state["qpos"].clone(), state["qvel"].clone()
        nq, nv = MicroDuckFootballEnv.DUCK_NQ, MicroDuckFootballEnv.DUCK_NV
        # blue0 lies on its side next to the ball; blue1 stands farther away
        # and walks toward it, so blue1 is the team's closest standing duck.
        for index, x in enumerate([-0.2, -0.6, 0.6, 0.9]):
            qpos[0, index * nq] = x
            qpos[0, index * nq + 1] = 0.0
            qvel[0, index * nv] = 0.3 if x < 0 else -0.3
        qpos[0, 2] = 0.05
        qpos[0, 3:7] = torch.tensor(
            [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0], dtype=qpos.dtype
        )
        qpos[0, -MicroDuckFootballEnv.BALL_NQ : -MicroDuckFootballEnv.BALL_NQ + 2] = 0.0
        env.reset(TensorDict(qpos=qpos, qvel=qvel, batch_size=[1]), set_state=True)
        step = env.step(self._football_action(env))["next"]
        assert bool(step["agents", "fallen"][0, 0, 0])
        reward = step["agents", "reward"][0, :, 0]
        assert reward[1] > 0.0
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    def test_football_progress_pays_the_closest_ducks_only(self, tmp_path):
        weights = {name: 0.0 for name in MicroDuckFootballEnv.REWARD_WEIGHTS}
        weights["ball_progress"] = 1.0
        env = self._football_env(
            tmp_path, players_per_team=2, reward_weights=weights, progress_players=1
        )
        env.reset()
        state = env.get_state()
        qpos, qvel = state["qpos"].clone(), state["qvel"].clone()
        nq = MicroDuckFootballEnv.DUCK_NQ
        # The ball rolls along +x from the origin; blue1 and red0 are the
        # closest ducks of their teams.
        for index, (x, y) in enumerate(
            [(-0.9, 0.0), (-0.4, 0.0), (0.4, 0.0), (0.9, 0.0)]
        ):
            qpos[0, index * nq] = x
            qpos[0, index * nq + 1] = y
        qpos[0, -MicroDuckFootballEnv.BALL_NQ : -MicroDuckFootballEnv.BALL_NQ + 2] = 0.0
        qvel[0, -MicroDuckFootballEnv.BALL_NV] = 0.5
        env.reset(TensorDict(qpos=qpos, qvel=qvel, batch_size=[1]), set_state=True)
        reward = env.step(self._football_action(env))["next"]["agents", "reward"][
            0, :, 0
        ]
        assert reward[0] == 0.0 and reward[3] == 0.0
        assert reward[1] > 0.0 and reward[2] < 0.0
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    def test_football_knockout_ends_the_match_for_the_standing_team(self, tmp_path):
        weights = {name: 0.0 for name in MicroDuckFootballEnv.REWARD_WEIGHTS}
        weights["knockout"] = 1.0
        env = self._football_env(
            tmp_path,
            players_per_team=1,
            reward_weights=weights,
            respawn=False,
            knockout=True,
        )
        env.reset()
        state = env.get_state()
        qpos = state["qpos"].clone()
        nq = MicroDuckFootballEnv.DUCK_NQ
        # The red duck lies on its side; blue stands.
        qpos[0, nq + 2] = 0.05
        qpos[0, nq + 3 : nq + 7] = torch.tensor(
            [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0], dtype=qpos.dtype
        )
        env.reset(
            TensorDict(qpos=qpos, qvel=state["qvel"], batch_size=[1]), set_state=True
        )
        step = env.step(self._football_action(env))["next"]
        assert bool(step["terminated"][0, 0])
        assert step["knockout"].tolist() == [[1]]
        assert step["goal"].tolist() == [[0]]
        assert step["agents", "reward"][0, :, 0].tolist() == pytest.approx([1.0, -1.0])
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    def test_football_ball_rolls_without_speeding_up(self, tmp_path):
        # The ball's mass sits at its center: a rolling ball keeps (slowly
        # loses) its speed instead of wobbling like an eccentric wheel.
        env = self._football_env(tmp_path, players_per_team=1)
        env.reset()
        assert env._backend._m.body("ball").ipos.tolist() == pytest.approx(
            [0.0, 0.0, 0.0]
        )
        state = env.get_state()
        qvel = state["qvel"].clone()
        radius = env.ball_radius
        qvel[0, -6] = 0.5  # rolling along +x without slipping
        qvel[0, -2] = 0.5 / radius
        env.reset(
            TensorDict(qpos=state["qpos"], qvel=qvel, batch_size=[1]), set_state=True
        )
        action = self._football_action(env)
        dt = env.frame_skip * env._backend.timestep
        start = env.step(action)["next", "ball_position"][0, 0].item()
        for _ in range(24):
            step = env.step(action)
        speed = (step["next", "ball_position"][0, 0].item() - start) / (24 * dt)
        assert 0.35 < speed <= 0.5
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    def test_football_pitch_collides_with_the_robot_body(self, tmp_path):
        # The robot's body boxes are collision class 2 (its feet class 1):
        # every pitch geom must accept both, or a fallen duck sinks through
        # the grass and the walls.
        env = self._football_env(tmp_path, players_per_team=1)
        model = env._backend._m
        pitch = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            for i in range(model.ngeom)
            if model.geom_bodyid[i] == 0 and model.geom_contype[i]
        ]
        assert "floor" in pitch and any("wall" in name for name in pitch)
        for name in pitch:
            assert model.geom(name).conaffinity[0] & 3 == 3, name
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    def test_football_observation_is_team_symmetric(self, tmp_path):
        env = self._football_env(tmp_path, players_per_team=2)
        env.reset()
        players = env.players_per_team

        def duck(x: float, y: float, yaw: float, vx: float, vy: float):
            qpos = [x, y, 0.12, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
            qpos += [0.0] * MicroDuckEnv.NUM_JOINTS
            qvel = [vx, vy, 0.0, 0.0, 0.0, 0.0] + [0.0] * MicroDuckEnv.NUM_JOINTS
            return qpos, qvel

        # Red mirrors blue through the pitch center, the ball sits on the spot.
        qpos, qvel = [], []
        for index in range(players):
            blue_q, blue_v = duck(-0.5 - 0.2 * index, 0.2 + 0.1 * index, 0.3, 0.1, 0.05)
            qpos += blue_q
            qvel += blue_v
        for index in range(players):
            red_q, red_v = duck(
                0.5 + 0.2 * index, -0.2 - 0.1 * index, 0.3 + math.pi, -0.1, -0.05
            )
            qpos += red_q
            qvel += red_v
        qpos += [0.0, 0.0, env.ball_radius, 1.0, 0.0, 0.0, 0.0]
        qvel += [0.0] * 6
        td = env.reset(
            TensorDict(
                qpos=torch.tensor([qpos]), qvel=torch.tensor([qvel]), batch_size=[1]
            ),
            set_state=True,
        )
        observation = td["agents", "observation"][0]
        torch.testing.assert_close(
            observation[:players], observation[players:], atol=1e-5, rtol=0
        )
        # Blue's first duck sees the ball ahead and to its right.
        features = observation[0, MicroDuckEnv.OBSERVATION_DIM :]
        assert features[4] > 0 and features[5] < 0
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    @pytest.mark.parametrize(
        "train_team, final_coeff", [("both", None), ("blue", 0.05)]
    )
    def test_microduck_football_skill_training(self, tmp_path, train_team, final_coeff):
        low = TensorDictModule(
            nn.Linear(MicroDuckEnv.OBSERVATION_DIM, MicroDuckEnv.NUM_JOINTS),
            in_keys=["observation"],
            out_keys=["action"],
        )
        with torch.no_grad():
            low.module.weight.zero_()
            low.module.bias.zero_()
        env = microduck_skill_env(
            self._football_env(tmp_path, players_per_team=5, max_episode_steps=12),
            low,
            [MicroDuckEnv.standing_task(), MicroDuckEnv.tracking_task(0.2)],
            steps=3,
        )
        actor, critic = football_mappo.make_models(env, hidden_size=8, depth=1)
        before = [p.detach().clone() for p in actor.parameters()]
        critic_before = [p.detach().clone() for p in critic.parameters()]
        low_before = [p.detach().clone() for p in low.parameters()]

        def check_warmup(iteration):
            if iteration == 1:
                for old, new in zip(before, actor.parameters()):
                    torch.testing.assert_close(old, new, rtol=0, atol=0)
                assert any(
                    not torch.equal(old, new)
                    for old, new in zip(critic_before, critic.parameters())
                )

        metrics = football_mappo.train_mappo(
            env,
            actor,
            critic,
            total_frames=24,
            frames_per_batch=8,
            epochs=1,
            minibatch_size=4,
            target_kl=None,
            train_team=train_team,
            critic_warmup_iterations=1,
            reference_kl_coeff=0.5,
            reference_kl_final_coeff=final_coeff,
            iteration_callback=check_warmup,
        )
        assert [row["ppo/reference_kl_coeff"] for row in metrics] == pytest.approx(
            [0.5, 0.5, 0.5 if final_coeff is None else final_coeff]
        )
        assert metrics[-1]["ppo/reference_kl"] > 0
        assert any(
            not torch.equal(old, new) for old, new in zip(before, actor.parameters())
        )
        for old, new in zip(low_before, low.parameters()):
            torch.testing.assert_close(old, new)
            assert new.grad is None
        env.close(raise_if_closed=False)

    @pytest.mark.skipif(not _has_hydra, reason="Hydra is not installed")
    def test_football_checkpoint_ranking_against_opponent(self):
        draw = {
            "evaluation/goals_blue": 0.0,
            "evaluation/goals_red": 0.0,
            "evaluation/ball_progress_blue": 0.0,
            "evaluation/falls_per_duck": 0.0,
        }
        loss = {**draw, "evaluation/goals_red": 1.0}
        win = {**draw, "evaluation/goals_blue": 1.0}
        knockout = {**draw, "evaluation/knockouts_red": 1.0}
        score = ft.partial(football_mappo.evaluation_score, train_team="blue")
        assert score(win) > score(draw) > score(loss)
        assert score(knockout) == score(loss)
        # Symmetric self-play still rewards scoring on either side.
        assert football_mappo.evaluation_score(win) == football_mappo.evaluation_score(
            loss
        )

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    def test_microduck_football_skip_keeps_physics_state(self, tmp_path):
        env = self._football_env(tmp_path)
        td = env.reset().update(self._football_action(env))
        state = env.get_state().clone()
        td["_step"] = torch.zeros(1, dtype=torch.bool)
        result = env.step(td)["next"]
        torch.testing.assert_close(env.get_state()["qpos"], state["qpos"])
        torch.testing.assert_close(env.get_state()["qvel"], state["qvel"])
        assert env._step_count.item() == 0
        assert not result["agents", "reward"].any()
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    def test_microduck_skill_env_executes_skills_with_the_walker(self, tmp_path):
        tasks = [
            MicroDuckEnv.standing_task(),
            MicroDuckEnv.speed_range_task(0.2, 0.2),
            MicroDuckEnv.sidestep_task(0.15),
        ]
        joint_action = torch.linspace(-0.5, 0.5, MicroDuckEnv.NUM_JOINTS)

        class Walker(TensorDictModuleBase):
            in_keys = ["observation", "task_id", "is_init"]
            out_keys = ["action"]

            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, tensordict):
                self.calls.append(tensordict.clone())
                tensordict["action"] = joint_action.expand(
                    tensordict.shape[0], -1
                ).clone()
                return tensordict

        walker = Walker()
        base = self._football_env(tmp_path, max_episode_steps=7)
        env = microduck_skill_env(base, walker, tasks, skills=[1, 2], steps=3)
        assert env.action_spec["agents", "skill"].space.n == 2
        check_env_specs(env)
        walker.calls.clear()
        td = env.reset()
        assert td["agents", "observation"].shape == (1, 2, base.observation_dim + 2)
        assert (td["agents", "observation"][..., -2:] == torch.tensor([1.0, 0.0])).all()
        skills = TensorDict(
            {("agents", "skill"): torch.tensor([[0, 1]])}, batch_size=[1]
        )
        out = env.step(td.update(skills))["next"]
        assert base._step_count.item() == 3
        assert len(walker.calls) == 3
        first, second = walker.calls[:2]
        # Each duck's walker gets its skill's library index, command and clock.
        assert first["task_id"].squeeze(-1).tolist() == [1, 2]
        assert first["is_init"].all() and not second["is_init"].any()
        command = MicroDuckEnv.COMMAND_START
        torch.testing.assert_close(
            first["observation"][:, command : command + 2],
            torch.tensor([[0.2, 0.0], [0.0, 0.15]]),
        )
        clock = MicroDuckEnv.GAIT_PHASE_START
        frequency = torch.tensor([2.0, MicroDuckEnv.GAIT_FREQUENCY_HZ])
        phase = MicroDuckEnv.GAIT_PHASE_OFFSET + 2 * math.pi * frequency * 0.02
        torch.testing.assert_close(
            second["observation"][:, clock], phase.sin(), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            second["observation"][:, clock + 1], phase.cos(), atol=1e-5, rtol=0
        )
        assert second["observation"][:, clock + 2].tolist() == pytest.approx(
            [0.02 / MicroDuckEnv.GAIT_RAMP_DURATION_S] * 2
        )
        assert (
            out["agents", "observation"][..., -2:]
            == torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        ).all()
        # Rewards are summed over the window; the observation is the last one.
        twin = self._football_env(tmp_path / "twin", max_episode_steps=7)
        twin.reset()
        total = None
        for _ in range(3):
            step = twin.step(self._football_action(twin, joint_action))["next"]
            reward = step["agents", "reward"]
            total = reward if total is None else total + reward
        torch.testing.assert_close(out["agents", "reward"], total)
        torch.testing.assert_close(
            out["agents", "observation"][..., : base.observation_dim],
            step["agents", "observation"],
        )
        assert not out["done"].any()
        # A truncation inside the window ends it early and is latched.
        td = env.step_mdp(td).update(skills)
        env.step(td)
        assert base._step_count.item() == 6
        td = env.step_mdp(td).update(skills)
        out = env.step(td)["next"]
        assert out["truncated"].item() and out["done"].item()
        assert base._step_count.item() == 7
        env.close()
        twin.close()

    @pytest.mark.skipif(
        not (_has_mujoco_torch and _test_accelerated),
        reason="accelerated backend tests require TORCHRL_ZOO_TEST_ACCELERATED=1",
    )
    def test_microduck_skills_stop_finished_matches(self, tmp_path):
        base = self._football_env(
            tmp_path, backend="mujoco-torch", num_envs=2, max_episode_steps=20
        )
        policy = TensorDictModule(
            nn.Linear(MicroDuckEnv.OBSERVATION_DIM, MicroDuckEnv.NUM_JOINTS),
            in_keys=["observation"],
            out_keys=["action"],
        )
        with torch.no_grad():
            policy.module.weight.zero_()
            policy.module.bias.zero_()
        env = microduck_skill_env(base, policy, [MicroDuckEnv.standing_task()], steps=3)
        td = env.reset()
        base._step_count[0] = 19
        transition = env.rand_step(td)
        torch.testing.assert_close(base._step_count, torch.tensor([20, 3]))
        assert transition["next", "done"][0].all()
        assert not transition["next", "done"][1].any()
        torch.testing.assert_close(
            transition["next", "agents", "_controller", "gait_elapsed"],
            torch.tensor([[0.02, 0.02], [0.06, 0.06]]),
        )
        env.close()

    @pytest.mark.skipif(not _has_mujoco, reason="MuJoCo is not installed")
    def test_microduck_skill_env_carries_the_walker_state(self, tmp_path):
        walker = TensorDictSequential(
            TensorDictModule(
                nn.Linear(MicroDuckEnv.OBSERVATION_DIM, 8),
                in_keys=["observation"],
                out_keys=["embed"],
            ),
            GRUModule(
                input_size=8,
                hidden_size=8,
                in_keys=["embed", "recurrent_state", "is_init"],
                out_keys=["features", ("next", "recurrent_state")],
            ),
            TensorDictModule(
                nn.Linear(8, MicroDuckEnv.NUM_JOINTS),
                in_keys=["features"],
                out_keys=["action"],
            ),
        )
        base = self._football_env(tmp_path)
        env = microduck_skill_env(base, walker, [MicroDuckEnv.standing_task()], steps=2)
        td = env.reset()
        state_key = ("agents", "_controller", "recurrent_state")
        assert (td[state_key] == 0).all()
        td["agents", "skill"] = torch.zeros(1, 2, dtype=torch.long)
        transition = env.step(td)
        state = transition["next"].get(state_key)
        assert state.shape == (1, 2, 1, 8)
        assert (state != 0).any()
        assert (env.reset().get(state_key) == 0).all()
        env.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
