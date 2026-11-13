"""Exact virtual hospital composition, deterministic rounding, and type-dependent behavior."""
import copy
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from baseline import build_state_vector
from reward import get_hospital_suitability
from virtual_hospital_mix import (DEFAULT_WEIGHTS, TYPE_KEYS, TYPE_LABELS, apply_hospital_mix,
                                   hospital_plan, mix_controls)


class HospitalMixTests(unittest.TestCase):
    def test_two_tertiary_centers_are_fixed_and_remaining_counts_sum_exactly(self):
        for count in (0, *range(2, 11)):
            with self.subTest(count=count):
                plan = hospital_plan(count)
                self.assertEqual(sum(plan["counts"].values()), count)
                self.assertEqual(plan["counts"]["tertiary"], 2 if count else 0)
        self.assertEqual(hospital_plan(10)["counts"], dict(tertiary=2, local_center=4, local_institution=3, reported_er=1))

    def test_ratio_extremes_ties_and_zero_extra_are_unambiguous(self):
        for key in TYPE_KEYS[1:]:
            weights = {category: int(category == key) for category in TYPE_KEYS[1:]}
            plan = hospital_plan(9, weights)
            self.assertEqual(plan["counts"][key], 7)
            self.assertEqual(plan["counts"]["tertiary"], 2)
        self.assertEqual(hospital_plan(3, dict(local_center=1, local_institution=1, reported_er=1))["counts"]["local_center"], 1)
        self.assertEqual(sum(hospital_plan(2, dict(local_center=0, local_institution=0, reported_er=0))["counts"].values()), 2)

    def test_assignment_preserves_geometry_order_and_real_hospitals_and_does_not_consume_global_rng(self):
        hospitals = [dict(id="real", type="지역응급의료센터", name="Real", capability=.8)]
        hospitals += [dict(id=f"v{i}", type="권역응급의료센터", name="Old", virtual=True,
                           sumo_x=i, sumo_y=i*2, verified_access_edge=f"e{i}", capability=1.) for i in range(10)]
        original = copy.deepcopy(hospitals)
        random.seed(5); before_rng = random.getstate()
        plan = apply_hospital_mix(hospitals)
        self.assertEqual(random.getstate(), before_rng)
        self.assertEqual(hospitals[0], original[0])
        self.assertEqual([h["id"] for h in hospitals], [h["id"] for h in original])
        for old, updated in zip(original[1:], hospitals[1:]):
            for key in ("id", "sumo_x", "sumo_y", "verified_access_edge"):
                self.assertEqual(updated[key], old[key])
        for key, label in TYPE_LABELS.items():
            self.assertEqual(sum(h.get("virtual") and h["type"] == label for h in hospitals if h.get("virtual")), plan["counts"][key])
        again = copy.deepcopy(original)
        apply_hospital_mix(again)
        self.assertEqual(hospitals, again)

    def test_actual_type_changes_the_existing_state_features_and_suitability(self):
        hospitals = [dict(id=f"v{i}", virtual=True, type="권역응급의료센터", sumo_x=i, sumo_y=0.) for i in range(3)]
        apply_hospital_mix(hospitals, dict(local_center=0, local_institution=0, reported_er=1))
        state = build_state_vector((0., 0.), 1, hospitals)
        self.assertEqual(hospitals[2]["type"], "응급실운영신고기관")
        self.assertLess(get_hospital_suitability(1, hospitals[2]), get_hospital_suitability(1, hospitals[0]))
        self.assertAlmostEqual(float(state[16+11*2+2]), .3)

    def test_controls_metadata_does_not_mutate_default_ratios(self):
        config = mix_controls()
        config["default_weights"]["local_center"] = 0
        self.assertEqual(DEFAULT_WEIGHTS["local_center"], 5)
