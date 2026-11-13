"""Hourly real-world demand calibration and compatibility with the shared episode engine."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from dashboard_demand import (DEFAULT_DURATION, PATIENTS_PER_HOUR, REFERENCE_PER_HOUR,
                              dashboard_scenario, demand_metadata)
from experiment_config import SCENARIOS, SimulationConfig, generate_schedule


class HourlyDemandTests(unittest.TestCase):
    def scenario(self, level=1, traffic=1):
        return dashboard_scenario(dict(demand=level, traffic=traffic), SCENARIOS)

    def test_official_daily_activity_proxy_and_ten_percent_default_are_explicit(self):
        self.assertEqual(DEFAULT_DURATION, 3600)
        self.assertAlmostEqual(REFERENCE_PER_HOUR, 46.6 / 24)
        self.assertAlmostEqual(PATIENTS_PER_HOUR[1], 2.1358333333333337)
        self.assertAlmostEqual(PATIENTS_PER_HOUR[1] / REFERENCE_PER_HOUR, 1.1)
        profile = demand_metadata()
        self.assertEqual(profile["reference_year"], 2024)
        self.assertEqual(profile["reference_metric"], "119 구급활동")
        self.assertEqual(profile["reference_unit"], "건/일")
        self.assertEqual(profile["source_page"], 6)

    def test_hourly_rate_is_converted_to_probability_per_simulated_second(self):
        for level, rate in enumerate(PATIENTS_PER_HOUR):
            with self.subTest(level=level):
                scenario = self.scenario(level)
                self.assertAlmostEqual(scenario["prob"] * 3600, rate)
                self.assertLess(scenario["prob"], .001)
                self.assertEqual(scenario["patients_per_hour"], rate)

    def test_short_runs_can_have_no_patients_without_forced_first_arrivals(self):
        rows = generate_schedule(self.scenario(), ["e"], 42, SimulationConfig(arrival_steps=30))
        self.assertEqual(rows, {})

    def test_one_hour_is_sparse_and_same_seed_levels_remain_nested(self):
        config = SimulationConfig(arrival_steps=3600)
        schedules = [generate_schedule(self.scenario(level), ["a", "b"], 42, config) for level in range(3)]
        self.assertTrue(schedules[1])
        self.assertLess(len(schedules[1]), 10)
        self.assertNotIn(1, schedules[1])
        self.assertTrue(set(schedules[0]) <= set(schedules[1]) <= set(schedules[2]))
        for tick in schedules[0]:
            self.assertEqual(schedules[0][tick], schedules[2][tick])
        self.assertTrue(all(1 <= tick <= 3600 for schedule in schedules for tick in schedule))

    def test_traffic_is_independent_and_legacy_cli_scenarios_are_not_rewritten(self):
        original = copy.deepcopy(SCENARIOS)
        scenarios = [self.scenario(1, traffic) for traffic in range(3)]
        self.assertEqual([s["scale"] for s in scenarios], [.3, .5, .8])
        self.assertEqual(len({s["prob"] for s in scenarios}), 1)
        self.assertEqual(SCENARIOS, original)
        self.assertEqual({s["prob"] for s in SCENARIOS}, {.2, .3, .4})

    def test_api_metadata_cannot_mutate_the_calibration(self):
        metadata = demand_metadata()
        metadata["patients_per_hour"][1] = 10000
        metadata["scenario_multipliers"][1] = 10000
        self.assertAlmostEqual(demand_metadata()["patients_per_hour"][1], 2.1358333333333337)
        self.assertEqual(demand_metadata()["scenario_multipliers"][1], 1.1)

    def test_custom_hourly_rate_is_not_multiplied_by_the_official_preset(self):
        for rate in (0., 2.14, 12.5, 3600.):
            options = dict(demand=2, traffic=1, demand_mode="rate", custom_patients_per_hour=rate)
            scenario = dashboard_scenario(options, SCENARIOS)
            self.assertAlmostEqual(scenario["prob"] * 3600, rate)
            self.assertNotIn("fixed_patient_count", scenario)
            self.assertEqual(scenario["demand_profile"]["profile_id"], "user_hourly_v1")

    def test_fixed_count_is_exact_reproducible_and_uses_unique_in_range_times(self):
        config = SimulationConfig(arrival_steps=60)
        options = dict(demand=1, traffic=2, duration=60, demand_mode="count", patient_count=17)
        scenario = dashboard_scenario(options, SCENARIOS)
        first = generate_schedule(scenario, ["a", "b"], 42, config)
        self.assertEqual(len(first), 17)
        self.assertEqual(first, generate_schedule(scenario, ["a", "b"], 42, config))
        self.assertTrue(all(1 <= step <= 60 for step in first))
        self.assertNotEqual(set(first), set(generate_schedule(scenario, ["a", "b"], 43, config)))
        self.assertEqual(len({row["patient_id"] for row in first.values()}), 17)
        self.assertAlmostEqual(scenario["patients_per_hour"], 1020.)
        other_traffic = dashboard_scenario({**options, "traffic": 0}, SCENARIOS)
        self.assertEqual(first, generate_schedule(other_traffic, ["a", "b"], 42, config))

    def test_fixed_zero_and_one_patient_per_second_boundaries(self):
        config = SimulationConfig(arrival_steps=30)
        for count in (0, 30):
            scenario = dashboard_scenario(dict(demand=1, traffic=1, duration=30, demand_mode="count", patient_count=count), SCENARIOS)
            rows = generate_schedule(scenario, ["a"], 42, config)
            self.assertEqual(len(rows), count)
            if count:
                self.assertEqual(set(rows), set(range(1, 31)))

    def test_fixed_count_rejects_invalid_counts_and_an_extra_demand_multiplier(self):
        config = SimulationConfig(arrival_steps=30)
        for count in (-1, 31, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                generate_schedule(dict(prob=.2, fixed_patient_count=count), ["a"], 42, config)
        with self.assertRaises(ValueError):
            generate_schedule(dict(prob=.2, fixed_patient_count=2), ["a"], 42, replace(config, demand_scale=.5))
