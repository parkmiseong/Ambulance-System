"""V4 score ordering, exchangeable hospitals, version safety and validation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from dqn_agent import DQNAmbulanceAgent, HospitalQNetwork
from experiment_config import SimulationConfig, curriculum_multiplier
from reward import calculate_reward, failure_penalty, HOSPITAL_TYPES
import train_agent
from test_simulation_system import FakeAgent, hospital
import test_simulation_system as fixtures


class RewardVersionTests(unittest.TestCase):
    def test_v4_all_failures_rank_below_success_even_when_late(self):
        failures = []
        for ktas in range(1, 6):
            for kind in HOSPITAL_TYPES:
                h = dict(type=kind)
                for occupancy in (0., .85, 1.):
                    for elapsed in (0., 120., 5000.):
                        success = calculate_reward(elapsed, ktas, h, occupancy=occupancy)
                        failed = calculate_reward(elapsed, ktas, h, occupancy=occupancy, route_failed=True)
                        self.assertLess(failed, success)
                        failures.append(failed)
        self.assertEqual(set(failures), {-failure_penalty(4)})
        self.assertEqual(failure_penalty(4), 85.)

    def test_legacy_reward_is_preserved_explicitly(self):
        h = dict(type=HOSPITAL_TYPES[0])
        self.assertEqual(failure_penalty(3), 20.)
        self.assertLess(calculate_reward(5000, 1, h, occupancy=.8, reward_version=3), -20.)
        self.assertEqual(calculate_reward(120, 1, h, reward_version=3),
                         calculate_reward(120, 1, h, reward_version=4))

    def test_v4_waiting_failure_is_in_total_learning_reward(self):
        config = SimulationConfig(arrival_steps=2, max_wait=2, mission_timeout=10,
                                  treatment_seconds=10, bed_fraction=.1)
        result, rows, agent = fixtures.SharedEngineTests().run_case(config)
        self.assertEqual(rows[1]["reward"], -85.)
        self.assertAlmostEqual(sum(r[0][2] for r in agent.records), result["reward"])


class HospitalNetworkTests(unittest.TestCase):
    def test_hospital_permutation_permutates_scores(self):
        torch.manual_seed(42)
        net = HospitalQNetwork(42, 3)
        state = torch.rand(4, 42)
        order = [2, 0, 1]
        permuted = torch.cat((state[:, :15],
                              state[:, 15:].reshape(4, 3, 9)[:, order].reshape(4, 27)), dim=1)
        with torch.no_grad():
            self.assertTrue(torch.allclose(net(permuted), net(state)[:, order], atol=1e-6))

    def test_shared_checkpoint_restores_architecture_and_scaled_replay(self):
        agent = DQNAmbulanceAgent(33, 2, architecture="hospital_shared_v4", reward_scale=100.)
        state = np.arange(33, dtype=np.float32) / 33
        agent.store_transition(state, 0, -85., state, True)
        self.assertAlmostEqual(agent.memory[0][2], -.85)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "checkpoint.pth")
            agent.save_model(path, episode=3, full_state=True)
            inference = DQNAmbulanceAgent(33, 2)
            inference.load_model(path)
            self.assertEqual(inference.architecture, "hospital_shared_v4")
            np.testing.assert_allclose(inference.q_values(state), agent.q_values(state))
            resumed = DQNAmbulanceAgent(33, 2, architecture="hospital_shared_v4", reward_scale=100.)
            resumed.load_model(path, resume=True)
            self.assertAlmostEqual(resumed.memory[0][2], -.85)
            self.assertEqual(resumed.loaded_episode, 3)
            before = [parameter.detach().clone() for parameter in resumed.model.parameters()]
            self.assertTrue(np.isfinite(resumed.train_step(1)))
            self.assertTrue(any(not torch.equal(old, new) for old, new in
                                zip(before, resumed.model.parameters())))
            with self.assertRaisesRegex(ValueError, "network/reward scaling"):
                DQNAmbulanceAgent(33, 2).load_model(path, resume=True)

    def test_legacy_checkpoint_without_architecture_still_loads(self):
        agent = DQNAmbulanceAgent(33, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "legacy.pth")
            agent.save_model(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint.pop("architecture")
            checkpoint.pop("reward_scale")
            with patch("dqn_agent.torch.load", return_value=checkpoint):
                restored = DQNAmbulanceAgent(33, 2, architecture="hospital_shared_v4")
                restored.load_model(path)
            self.assertEqual(restored.architecture, "flat_v3")
            np.testing.assert_allclose(restored.q_values(np.zeros(33)), agent.q_values(np.zeros(33)))


class TrainingBoundaryTests(unittest.TestCase):
    def test_cyclic_curriculum_returns_to_low_load(self):
        self.assertEqual([curriculum_multiplier(ep, (.1, .3), cycle=True)
                          for ep in (1, 9, 10, 18, 19, 28)], [.1, .1, .3, .3, .1, .3])

    def test_validation_covers_all_requested_demands_without_changing_model_mode(self):
        agent = FakeAgent()
        response = dict(generated_patients=2, reward=10., arrived=2, golden_success=1)
        with patch.object(train_agent, "run_episode", return_value=response) as episode:
            result = train_agent.validate_panel(None, [], agent, SimulationConfig(), (1,),
                                                (1, 2), 10, "severity", 300, "unused",
                                                quiet=True, demand_levels=(.1, .3))
        self.assertEqual([call.args[5].demand_scale for call in episode.call_args_list], [.1, .3, .1, .3])
        self.assertEqual(len({str(c.kwargs["output_dir"]) for c in episode.call_args_list}), 4)
        self.assertEqual(result["avg_reward"], 5.)
        self.assertEqual(agent.epsilon, .5)
        self.assertTrue(agent.model.training)

    def test_final_episode_is_validated_and_existing_experiment_is_not_overwritten(self):
        agent = FakeAgent()
        result = dict(generated_patients=2, missions=2, arrived=1, failed=1, censored=0,
                      golden_rate=50., avg_wait=1., reward=-40., avg_reward=-20., avg_loss=.1)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(train_agent, "prepare_environment", return_value=(None, [hospital()], ["e"], [["h"]])), \
             patch.object(train_agent, "make_manifest", return_value={"version":4}), \
             patch.object(train_agent, "DQNAmbulanceAgent", return_value=agent), \
             patch.object(train_agent, "run_episode", return_value=copy.deepcopy(result)), \
             patch.object(train_agent, "validate_panel", return_value=dict(avg_reward=-5., arrival_rate=50.)) as panel, \
             patch.object(train_agent, "append_record"), patch.object(train_agent, "plot_training_metrics"):
            kwargs = dict(num_episodes=1, eval_every=20, quiet=True, output_root=directory)
            train_agent.train(**kwargs)
            panel.assert_called_once()
            manifest = next((Path(directory) / "training").glob("experiment_*.json"))
            original = manifest.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                train_agent.train(**kwargs)
            self.assertEqual(manifest.read_bytes(), original)
            self.assertTrue(any("v4_" in item[0] for item in agent.saves))


if __name__ == "__main__":
    unittest.main()
