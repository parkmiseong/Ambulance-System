"""V5 exposes dispatch-time estimates without leaking future outcomes."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from ambulance_sim import Mission, MissionManager
from baseline import build_state_vector, state_dimension
from dqn_agent import DQNAmbulanceAgent, HospitalHistoryQNetwork
from experiment_config import SimulationConfig
import simulation_setup
from test_simulation_system import hospital
import test_simulation_system as fixtures


class PendingEstimateTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()
        self.manager = MissionManager()
        self.manager.active["A"] = Mission(
            dict(patient_id="old", ktas=1, occurrence_time=0), "A", 2., None,
            estimated_reward=-12.)

    def state(self, config=None, now=10):
        return build_state_vector((0., 0.), 2, [hospital()], manager=self.manager,
                                  now=now, config=config or self.config)

    def test_new_feature_does_not_replace_old_features(self):
        legacy = self.state(replace(self.config, observation_version=3))
        current = self.state()
        self.assertEqual(len(legacy), 24)
        self.assertEqual(len(current), 25)
        self.assertAlmostEqual(float(current[15]), -.12)
        np.testing.assert_array_equal(np.delete(current, 15), legacy)

    def test_estimate_is_latched_and_only_active_missions_contribute(self):
        self.manager.active["B"] = Mission(
            dict(patient_id="other", ktas=4, occurrence_time=1), "B", 3., None,
            estimated_reward=5.)
        self.assertAlmostEqual(float(self.state()[15]), -.07)
        # Hypothetical future outcome fields and elapsed time cannot replace estimates.
        self.manager.active["A"].patient.update(actual_reward=999., finish_time=999.)
        self.assertAlmostEqual(float(self.state(now=100)[15]), -.07)
        del self.manager.active["A"]
        self.assertAlmostEqual(float(self.state()[15]), .05)
        self.manager.active.clear()
        self.assertEqual(self.state()[15], 0.)

    def test_zero_control_has_identical_shape_and_other_features(self):
        estimate = self.state()
        zero = self.state(replace(self.config, pending_reward_feature="zero"))
        self.assertEqual(zero[15], 0.)
        self.assertEqual(zero.shape, estimate.shape)
        np.testing.assert_array_equal(np.delete(estimate, 15), np.delete(zero, 15))

    def test_shared_engine_exposes_previous_mission_prediction_before_its_result(self):
        config = replace(self.config, arrival_steps=2, max_wait=20, mission_timeout=10,
                         treatment_seconds=3, bed_fraction=.02)
        decisions = []
        _, outcomes, _ = fixtures.SharedEngineTests().run_case(
            config, duration=3, ambulance_count=2, on_decision=decisions.append)
        self.assertEqual(decisions[0]["state"][15], 0.)
        self.assertLess(decisions[1]["time"], outcomes[0]["finish_time"])
        self.assertAlmostEqual(decisions[1]["state"][15],
                               outcomes[0]["estimated_reward_at_dispatch"] / 100., places=6)
        zero_decisions = []
        _, zero_outcomes, _ = fixtures.SharedEngineTests().run_case(
            replace(config, pending_reward_feature="zero"), duration=3, ambulance_count=2,
            on_decision=zero_decisions.append)
        self.assertEqual(outcomes, zero_outcomes)
        self.assertEqual(zero_decisions[1]["state"][15], 0.)
        np.testing.assert_array_equal(np.delete(decisions[1]["state"], 15),
                                      np.delete(zero_decisions[1]["state"], 15))


class VersionCompatibilityTests(unittest.TestCase):
    def test_v5_hospital_permutation_preserves_pending_reward_context(self):
        net = HospitalHistoryQNetwork(43, 3)
        state = torch.rand(4, 43)
        permutation = [2, 0, 1]
        swapped = torch.cat((state[:, :16],
                             state[:, 16:].reshape(4, 3, 9)[:, permutation].reshape(4, 27)), dim=1)
        with torch.no_grad():
            self.assertTrue(torch.allclose(net(swapped), net(state)[:, permutation], atol=1e-6))

    def test_manifest_loader_preserves_v3_v4_and_v5_state_sizes(self):
        for version, observation, architecture, scale in (
                (3, 3, "flat_v3", 1.), (4, 3, "hospital_shared_v4", 100.),
                (5, 4, "hospital_shared_v5", 100.)):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                config = SimulationConfig(reward_version=3 if version == 3 else 4,
                                          observation_version=observation)
                with patch.object(simulation_setup, "network_digest", return_value="abc"):
                    manifest = simulation_setup.make_manifest(None, [hospital()], ["e"], [["h"]],
                                                               config, 10, "severity", 300, ())
                manifest["version"] = version
                if version < 5:
                    manifest["config"].pop("observation_version")
                    manifest["config"].pop("pending_reward_feature")
                if version == 3:
                    manifest["config"].pop("reward_version")
                    manifest.pop("agent_architecture")
                    manifest.pop("learning_reward_scale")
                manifest["signature"] = simulation_setup.manifest_signature(manifest)
                manifest_path = Path(directory) / "experiment.json"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                agent = DQNAmbulanceAgent(state_dimension(1, observation), 1,
                                          architecture=architecture, reward_scale=scale)
                agent.metadata = dict(manifest_path=str(manifest_path),
                                      experiment_signature=manifest["signature"])
                path = str(Path(directory) / "model.pth")
                agent.save_model(path, episode=18)
                with patch.object(simulation_setup, "SumoMedicalEnvironment", return_value=SimpleNamespace()), \
                     patch.object(simulation_setup, "network_digest", return_value="abc"):
                    _, _, restored, settings, _ = simulation_setup.load_experiment(path)
                self.assertEqual(settings.observation_version, observation)
                self.assertEqual(restored.state_dim, 24 if version < 5 else 25)
                self.assertEqual(restored.architecture, architecture)
                state = np.zeros(restored.state_dim, dtype=np.float32)
                np.testing.assert_allclose(restored.q_values(state), agent.q_values(state))

    def test_v5_resume_restores_scaled_replay_and_rejects_v4_network(self):
        state = np.zeros(34, dtype=np.float32)
        agent = DQNAmbulanceAgent(34, 2, architecture="hospital_shared_v5", reward_scale=100.)
        agent.store_transition(state, 1, -85., state, True)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "checkpoint.pth")
            agent.save_model(path, episode=2, full_state=True)
            resumed = DQNAmbulanceAgent(34, 2, architecture="hospital_shared_v5", reward_scale=100.)
            resumed.load_model(path, resume=True)
            self.assertEqual(resumed.loaded_episode, 2)
            self.assertEqual(resumed.memory[0][2], -.85)
            self.assertTrue(np.isfinite(resumed.train_step(1)))
            with self.assertRaisesRegex(ValueError, "network/reward scaling"):
                DQNAmbulanceAgent(34, 2, reward_scale=100.).load_model(path, resume=True)


if __name__ == "__main__":
    unittest.main()
