"""Regression checks for sequential replay; no SUMO process is started."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from episode_transitions import EpisodeTransitions
from dqn_agent import DQNAmbulanceAgent, REPLAY_SCHEMA_VERSION


class RecordingAgent:
    def __init__(self):
        self.records = []

    def store_transition(self, *args, **kwargs):
        self.records.append((args, kwargs))

    def train_step(self, batch_size):
        return None


class EpisodeTransitionsTests(unittest.TestCase):
    def test_rewards_follow_decision_intervals_not_completion_order(self):
        agent = RecordingAgent()
        episode = EpisodeTransitions(agent)
        # Missions A, B and C overlap. B completes before A.
        episode.begin([1, 0], 0, [0, 1])
        episode.begin([2, 0], 1, [1])
        episode.add_reward(7)  # B finishes.
        episode.add_reward(-3)  # A finishes after B.
        episode.begin([3, 1], 0, [0])
        episode.add_reward(5)  # C finishes during the final drain.
        episode.add_reward(-20)  # A waiting patient is censored.
        episode.close()
        episode.close()
        records = agent.records
        self.assertEqual(len(records), 3)
        self.assertEqual([r[0][2] for r in records], [0, 4, -15])
        self.assertEqual([r[0][4] for r in records], [False, False, True])
        np.testing.assert_array_equal(records[0][0][3], [2, 0])
        np.testing.assert_array_equal(records[1][0][3], [3, 1])
        self.assertEqual(records[0][1]["next_valid_actions"], (1,))
        self.assertEqual(records[1][1]["next_valid_actions"], (0,))
        self.assertEqual(records[2][1]["next_valid_actions"], ())
        self.assertEqual(sum(r[0][2] for r in records), episode.total_reward)

    def test_no_action_does_not_fabricate_replay_and_snapshots_are_owned(self):
        agent = RecordingAgent()
        episode = EpisodeTransitions(agent)
        episode.add_reward(-20)
        episode.close()
        self.assertEqual(agent.records, [])
        state = np.array([1, 2], dtype=np.float32)
        episode.begin(state, 0, [0])
        state[:] = 9
        episode.add_reward(3)
        episode.close()
        np.testing.assert_array_equal(agent.records[0][0][0], [1, 2])
        self.assertEqual(episode.unattributed_reward, -20)
        self.assertEqual(episode.total_reward, -17)
        next_episode = EpisodeTransitions(agent)
        next_episode.begin([4, 5], 0, [0])
        next_episode.close()
        self.assertEqual(agent.records[-1][0][2], 0)
        self.assertTrue(agent.records[-1][0][4])

    def test_invalid_action_does_not_commit_an_interval(self):
        agent = RecordingAgent()
        episode = EpisodeTransitions(agent)
        episode.begin([1, 0], 0, [0])
        with self.assertRaises(ValueError):
            episode.begin([2, 0], 1, [0])
        self.assertEqual(agent.records, [])
        episode.add_reward(6)
        episode.close()
        self.assertEqual(agent.records[0][0][2], 6)


class MaskedDoubleDQNTests(unittest.TestCase):
    def setUp(self):
        self.agent = DQNAmbulanceAgent(2, 2)
        with torch.no_grad():
            for model in (self.agent.model, self.agent.target_model):
                for parameter in model.parameters():
                    parameter.zero_()
            self.agent.model.fc[-1].bias.copy_(torch.tensor([1., 100.]))
            self.agent.target_model.fc[-1].bias.copy_(torch.tensor([5., 200.]))
        # Keep Q values fixed while testing targets.
        for group in self.agent.optimizer.param_groups:
            group["lr"] = 0.

    def test_nonterminal_target_uses_only_valid_next_actions(self):
        self.agent.store_transition([0, 0], 0, 2, [1, 0], False,
                                    next_valid_actions=[0])
        # Target = 2 + .95 * 5 = 6.75; Huber(1 - 6.75) = 5.25.
        self.assertAlmostEqual(self.agent.train_step(1), 5.25, places=5)

    def test_terminal_target_does_not_bootstrap_with_empty_mask(self):
        self.agent.store_transition([0, 0], 0, 2, [0, 0], True,
                                    next_valid_actions=[])
        self.assertAlmostEqual(self.agent.train_step(1), .5, places=5)

    def test_mixed_terminal_and_nonterminal_batch(self):
        self.agent.store_transition([0, 0], 0, 2, [1, 0], False,
                                    next_valid_actions=[0])
        self.agent.store_transition([0, 0], 0, 2, [0, 0], True)
        self.assertAlmostEqual(self.agent.train_step(2), 2.875, places=5)

    def test_missing_empty_and_out_of_range_next_actions_are_rejected(self):
        for actions in (None, [], [-1], [2]):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                self.agent.store_transition([0, 0], 0, 0, [0, 0], False,
                                            next_valid_actions=actions)
        self.assertEqual(len(self.agent.memory), 0)

    def test_replay_owns_state_arrays_and_next_mask(self):
        state = np.array([1, 2], dtype=np.float32)
        following = np.array([3, 4], dtype=np.float32)
        actions = [0]
        self.agent.store_transition(state, 0, 2, following, False,
                                    next_valid_actions=actions)
        state[:] = 9
        following[:] = 9
        actions.append(1)
        transition = self.agent.memory[0]
        np.testing.assert_array_equal(transition[0], [1, 2])
        np.testing.assert_array_equal(transition[3], [3, 4])
        np.testing.assert_array_equal(transition[5], [True, False])

    def test_legacy_resume_is_rejected_before_loading_parameters(self):
        checkpoint = {"state_dim": 2, "action_dim": 2}
        with patch("dqn_agent.os.path.exists", return_value=True), \
             patch("dqn_agent.torch.load", return_value=checkpoint), \
             self.assertRaisesRegex(ValueError, "replay schema"):
            self.agent.load_model("legacy.pth", resume=True)

    def test_checkpoint_round_trip_preserves_masked_replay(self):
        self.agent.store_transition([1, 2], 0, 3, [3, 4], False,
                                    next_valid_actions=[1])
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "checkpoint.pth")
            self.agent.save_model(path, episode=2, full_state=True)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["replay_schema_version"], REPLAY_SCHEMA_VERSION)
            restored = DQNAmbulanceAgent(2, 2)
            self.assertTrue(restored.load_model(path, resume=True))
            self.assertEqual(len(restored.memory), 1)
            np.testing.assert_array_equal(restored.memory[0][5], [False, True])


if __name__ == "__main__":
    unittest.main()
