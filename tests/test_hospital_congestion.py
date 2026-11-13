"""Per-hospital congestion bonuses, saved profiles and actual reporting."""
import csv
from dataclasses import replace
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from baseline import HospitalRouters, build_state_vector
from dashboard_results import patient_record, patients_csv
from experiment_config import SimulationConfig
from reward import (HOSPITAL_TYPES, calculate_reward, failure_penalty,
                    hospital_congestion_components, reward_components)
from web_dashboard import hospital_rows


class HospitalCongestionTests(unittest.TestCase):
    def test_balanced_bonus_zero_and_penalty_follow_each_ratio_continuously(self):
        examples = ((0., 8.), (.2, 4.8), (.5, 0.), (.6, -1.6),
                    (.7, -3.2), (.8, -6.466666666666667), (1., -13.))
        values = []
        for ratio, expected in examples:
            with self.subTest(ratio=ratio):
                actual = sum(hospital_congestion_components(ratio, mode="balanced").values())
                self.assertAlmostEqual(actual, expected)
                values.append(actual)
        self.assertEqual(values, sorted(values, reverse=True))

    def test_legacy_profile_reproduces_original_formula(self):
        for ratio in (0., .2, .5, .7, .75, 1.):
            parts = hospital_congestion_components(ratio)
            self.assertEqual(parts["occupancy"], -8. * ratio)
            self.assertEqual(parts["overcrowding"], -5. * max(0., ratio - .7) / .3)

    def test_final_reward_uses_selected_snapshot_instead_of_later_occupancy(self):
        hospital = dict(type=HOSPITAL_TYPES[0], occupancy=.9)
        parts = reward_components(60., 1, hospital, occupancy_at_selection=.2,
                                  congestion_reward_mode="balanced")
        self.assertAlmostEqual(parts["occupancy"], 4.8)
        self.assertEqual(parts["overcrowding"], 0.)
        self.assertAlmostEqual(sum(parts.values()), calculate_reward(
            60., 1, hospital, occupancy_at_selection=.2, congestion_reward_mode="balanced"))

    def test_failure_still_costs_more_than_any_bounded_success(self):
        for ktas in range(1, 6):
            for kind in HOSPITAL_TYPES:
                for occupancy in (0., .5, 1.):
                    hospital = dict(type=kind)
                    parts = reward_components(100000., ktas, hospital, occupancy=occupancy,
                                              congestion_reward_mode="balanced")
                    failed = reward_components(100000., ktas, hospital, occupancy=occupancy,
                                               route_failed=True, congestion_reward_mode="balanced")
                    self.assertGreater(sum(parts.values()), sum(failed.values()))
                    self.assertEqual(failed["occupancy"], 0.)
                    self.assertEqual(sum(failed.values()), -failure_penalty())

    def test_old_config_signatures_are_preserved_and_new_profile_roundtrips(self):
        old = SimulationConfig()
        self.assertNotIn("congestion_reward_mode", old.to_dict())
        current = replace(old, congestion_reward_mode="balanced")
        self.assertEqual(SimulationConfig(**current.to_dict()), current)
        with self.assertRaises(ValueError):
            replace(old, congestion_reward_mode="unknown")

    def test_web_preview_keeps_each_hospital_ratio_and_unknown_values_distinct(self):
        rows = hospital_rows([dict(id="low", occupancy=.2), dict(id="high", occupancy=.8),
                              dict(id="idle", occupancy=None)])
        self.assertAlmostEqual(rows[0]["congestion_reward_preview"], 4.8)
        self.assertAlmostEqual(rows[1]["congestion_reward_preview"], -6.466666666666667)
        self.assertIsNone(rows[2]["congestion_reward_preview"])

    def test_patient_csv_uses_recorded_components_and_leaves_missing_history_blank(self):
        row = dict(patient_id="p", hospital_id="H", arrived=True, ktas=1,
                   reward_components=dict(occupancy=4.8, overcrowding=0.), occupancy_at_selection=.2,
                   congestion_reward_mode="balanced")
        saved = patient_record(row, {}, 1, "dqn")
        self.assertEqual(saved["hospital_congestion_reward"], 4.8)
        exported = next(csv.DictReader(io.StringIO(patients_csv([saved]).decode("utf-8-sig"))))
        self.assertEqual(exported["실제 적용 병원 혼잡도 보상"], "4.8")
        self.assertEqual(exported["병원 혼잡도 보상 방식"], "balanced")
        old = patient_record(dict(patient_id="old", ktas=1), {}, 1, "dqn")
        self.assertIsNone(old["hospital_congestion_reward"])

    def test_reward_greedy_and_candidate_observation_use_the_active_profile(self):
        hospitals = [dict(type=HOSPITAL_TYPES[0], occupancy=ratio, reachable=True,
                          estimated_travel_time=60.) for ratio in (.8, .2)]
        self.assertEqual(HospitalRouters(hospitals).expected_reward_strategy(
            1, [0, 1], congestion_reward_mode="balanced"), 1)
        config = SimulationConfig(candidate_reward_feature=True, congestion_reward_mode="balanced")
        state = build_state_vector((0., 0.), 1, hospitals, config=config)
        for i, hospital in enumerate(hospitals):
            expected = calculate_reward(60., 1, hospital, occupancy_at_selection=hospital["occupancy"],
                                        congestion_reward_mode="balanced") / 100.
            self.assertAlmostEqual(state[16 + 11 * i + 10], expected, places=6)
