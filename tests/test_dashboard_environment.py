"""Shared DQN hospital-set transfer must preserve every learned weight."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from baseline import state_dimension
from dashboard_environment import resize_inference_agent
from dqn_agent import DQNAmbulanceAgent
from experiment_config import SimulationConfig


class HospitalSetTransferTests(unittest.TestCase):
    def test_shared_models_score_larger_and_smaller_sets_without_changing_weights(self):
        for architecture, version in (("hospital_shared_v4", 3), ("hospital_shared_v5", 4), ("hospital_shared_v6", 5)):
            config = SimulationConfig(observation_version=version)
            source = DQNAmbulanceAgent(state_dimension(9, version), 9, architecture=architecture)
            source.training_updates, source.loaded_episode = 123, 10
            source.metadata = {"experiment_signature": "original"}
            for count in (7, 11):
                with self.subTest(architecture=architecture, count=count):
                    adapted = resize_inference_agent(source, count, config)
                    self.assertEqual(adapted.action_dim, count)
                    self.assertEqual(adapted.state_dim, state_dimension(count, version))
                    self.assertEqual(adapted.training_updates, 123)
                    self.assertEqual(adapted.loaded_episode, 10)
                    self.assertEqual(adapted.epsilon, 0.)
                    self.assertFalse(adapted.model.training)
                    for name, value in source.model.state_dict().items():
                        self.assertTrue(torch.equal(value, adapted.model.state_dict()[name]))
                        self.assertNotEqual(value.data_ptr(), adapted.model.state_dict()[name].data_ptr())
                    for name, value in source.target_model.state_dict().items():
                        self.assertTrue(torch.equal(value, adapted.target_model.state_dict()[name]))
                    state = np.zeros(adapted.state_dim, dtype=np.float32)
                    self.assertEqual(adapted.q_values(state).shape, (count,))
                    self.assertTrue(np.isfinite(adapted.q_values(state)).all())
                    self.assertEqual(adapted.select_action(state, [count-1]), count-1)
            self.assertEqual(source.action_dim, 9)
            self.assertEqual(source.metadata, {"experiment_signature": "original"})

    def test_fixed_dimension_legacy_models_only_accept_their_original_count(self):
        config = SimulationConfig(observation_version=3, reward_version=3)
        source = DQNAmbulanceAgent(state_dimension(9, 3), 9)
        self.assertIs(resize_inference_agent(source, 9, config), source)
        with self.assertRaisesRegex(ValueError, "v3"):
            resize_inference_agent(source, 7, config)
        with self.assertRaises(ValueError):
            resize_inference_agent(source, 0, config)
