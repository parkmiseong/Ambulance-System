"""Check diagnostic attribution without altering the production reward collector."""
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from diagnose_learning import audit_episode, single_patient_case


class AttributionTests(unittest.TestCase):
    def test_overlapping_patient_rewards_follow_log_order(self):
        events = [
            ("mission", dict(patient_id="unassigned", reward=-85)),
            ("decision", dict(patient_id="A")),
            ("decision", dict(patient_id="B")),
            ("mission", dict(patient_id="B", reward=7)),
            ("mission", dict(patient_id="A", reward=-3)),
            ("decision", dict(patient_id="C")),
            ("mission", dict(patient_id="C", reward=5)),
            ("mission", dict(patient_id="waiting", reward=-85)),
        ]
        result = audit_episode(events)
        self.assertEqual(result["interval_rewards"], [0., 4., -80.])
        self.assertEqual(result["unattributed_reward"], -85.)
        outcomes = {row["patient_id"]: row for row in result["outcomes"]}
        self.assertTrue(outcomes["B"]["same_patient_interval"])
        self.assertFalse(outcomes["A"]["same_patient_interval"])
        self.assertEqual(outcomes["A"]["intervening_decisions"], 1)
        self.assertIsNone(outcomes["waiting"]["own_decision_index"])

    def test_scripted_case_has_a_unique_reward_optimum_without_index_shortcut(self):
        rng = random.Random(42)
        best_indices = set()
        for _ in range(100):
            state, actions, rewards, travel = single_patient_case(rng)
            self.assertEqual(state.shape, (33,))
            self.assertEqual(actions, [0, 1])
            best = max(actions, key=lambda a: rewards[a])
            self.assertEqual(best, min(actions, key=lambda a: travel[a]))
            self.assertGreater(rewards[best], rewards[1 - best])
            best_indices.add(best)
        self.assertEqual(best_indices, {0, 1})


if __name__ == "__main__":
    unittest.main()
