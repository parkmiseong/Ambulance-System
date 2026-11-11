"""Guard logical delayed-reward controls against completion-order mistakes."""
from pathlib import Path
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from diagnose_learning import single_patient_case
from diagnose_overlap import discounted_return, oracle_return, train_episode


class CaptureAgent:
    def __init__(self):
        self.records = []

    def store_transition(self, *args, **kwargs):
        self.records.append((args, kwargs))

    def train_step(self, batch_size):
        return None


class OverlapControlTests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(42)
        self.cases = [single_patient_case(rng) for _ in range(3)]
        self.actions = [0, 1, 0]

    def test_delay_moves_rewards_to_final_interval_without_changing_total_or_states(self):
        immediate, delayed = CaptureAgent(), CaptureAgent()
        a = train_episode(immediate, self.cases, False, forced_actions=self.actions)
        b = train_episode(delayed, self.cases, True, forced_actions=self.actions)
        self.assertAlmostEqual(a["total_reward"], b["total_reward"])
        self.assertEqual([r[0][2] for r in immediate.records], a["patient_rewards"])
        np.testing.assert_allclose([r[0][2] for r in delayed.records], [0, 0, sum(b["patient_rewards"])])
        for first, second in zip(immediate.records, delayed.records):
            np.testing.assert_array_equal(first[0][0], second[0][0])
            np.testing.assert_array_equal(first[0][3], second[0][3])
        self.assertEqual([r[0][4] for r in delayed.records], [False, False, True])

    def test_reversing_completion_order_does_not_reorder_decision_transitions(self):
        forward, reverse = CaptureAgent(), CaptureAgent()
        a = train_episode(forward, self.cases, True, forced_actions=self.actions)
        b = train_episode(reverse, self.cases, True, reverse=True, forced_actions=self.actions)
        self.assertEqual(a["finish_order"], [0, 1, 2])
        self.assertEqual(b["finish_order"], [2, 1, 0])
        self.assertEqual([r[0][1] for r in reverse.records], self.actions)
        np.testing.assert_allclose([r[0][2] for r in forward.records], [r[0][2] for r in reverse.records])

    def test_exhaustive_oracle_agrees_with_independent_optima_in_this_restricted_task(self):
        best_rewards = [max(case[2]) for case in self.cases]
        for delayed in (False, True):
            self.assertAlmostEqual(oracle_return(self.cases, delayed),
                                   discounted_return(best_rewards, delayed))

    def test_history_control_exposes_only_prior_selected_rewards(self):
        agent = CaptureAgent()
        result = train_episode(agent, self.cases, True, forced_actions=self.actions, history=True)
        states = [r[0][0] for r in agent.records]
        self.assertEqual(states[0][14], 0.)
        self.assertAlmostEqual(float(states[1][14]), result["patient_rewards"][0] / 100., places=6)
        self.assertAlmostEqual(float(states[2][14]), sum(result["patient_rewards"][:2]) / 100., places=6)


if __name__ == "__main__":
    unittest.main()
