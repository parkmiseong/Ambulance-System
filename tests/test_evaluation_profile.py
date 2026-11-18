"""Saved web demand, explicit overrides, and portable frozen model loading."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from dqn_agent import DQNAmbulanceAgent
from evaluation_audit import file_sha256
from evaluation_profile import (experiment_root, freeze_evaluation_model,
                                resolve_evaluation_profile, validate_evaluation_demands)
from experiment_config import LearningConfig, SCENARIOS, SimulationConfig, generate_schedule
import simulation_setup
import run_scenarios
import compare_queue_policies
import test_best_model_selection as best_fixtures
import test_simulation_system as fixtures


class EvaluationProfileTests(unittest.TestCase):
    def saved(self, mode="preset"):
        scenario = dict(id=5, prob=15. / 3600., scale=.5, name="saved",
                        patients_per_hour=15., demand_mode=mode,
                        initial_occupancy_range=[.35, .5],
                        demand_profile=dict(patients_per_hour=[10., 15., 20.]),
                        _route_files=["unportable.xml"], background_traffic={"source_trips": 2})
        if mode == "count":
            scenario["fixed_patient_count"] = 3
        return dict(web_scenario=scenario)

    def test_default_restores_exact_web_patient_stream_and_occupancy(self):
        manifest = self.saved()
        config = SimulationConfig(arrival_steps=3600)
        scenarios, occupancy, source = resolve_evaluation_profile(manifest, config)
        self.assertEqual(source, "saved_web")
        self.assertEqual(occupancy, (.35, .5))
        self.assertEqual(scenarios[0]["id"], 5)
        self.assertEqual(scenarios[0]["prob"], 15. / 3600.)
        self.assertNotIn("_route_files", scenarios[0])
        self.assertNotIn("background_traffic", scenarios[0])
        self.assertEqual(generate_schedule(scenarios[0], ["a", "b"], 32001, config),
                         generate_schedule(manifest["web_scenario"], ["a", "b"], 32001, config))
        self.assertIn("_route_files", manifest["web_scenario"])

    def test_all_scenarios_use_saved_preset_rates_without_dense_fallback(self):
        scenarios, occupancy, _ = resolve_evaluation_profile(self.saved(), SimulationConfig(), 0)
        self.assertEqual([s["prob"] * 3600 for s in scenarios], [10.] * 3 + [15.] * 3 + [20.] * 3)
        self.assertEqual([s["scale"] for s in scenarios], [.3, .5, .8] * 3)
        self.assertEqual(occupancy, (.35, .5))

    def test_custom_rate_and_fixed_count_are_preserved_for_scenario_overrides(self):
        for mode in ("rate", "count"):
            with self.subTest(mode=mode):
                scenarios, _, _ = resolve_evaluation_profile(self.saved(mode), SimulationConfig(), 0)
                self.assertTrue(all(s["prob"] == 15. / 3600. for s in scenarios))
                if mode == "count":
                    self.assertTrue(all(s["fixed_patient_count"] == 3 for s in scenarios))
                    validate_evaluation_demands(scenarios, SimulationConfig(), [1.])
                    with self.assertRaisesRegex(ValueError, "fixed-count"):
                        validate_evaluation_demands(scenarios, SimulationConfig(), [.3])

    def test_standard_profile_requires_explicit_selection_for_web_dense_demand(self):
        saved, _, _ = resolve_evaluation_profile(self.saved(), SimulationConfig())
        standard, occupancy, origin = resolve_evaluation_profile(self.saved(), SimulationConfig(), 5, "standard")
        self.assertLess(saved[0]["prob"], .01)
        self.assertEqual(standard[0]["prob"], .3)
        self.assertEqual(origin, "standard_override")
        self.assertEqual(occupancy, (.35, .5))

    def test_cli_saved_hourly_profile_and_training_scenario_remain_unchanged(self):
        config = SimulationConfig(demand_rates_per_hour=(10., 15., 20.))
        scenarios, occupancy, source = resolve_evaluation_profile(dict(training_scenarios=[6]), config)
        self.assertEqual((source, occupancy), ("saved_cli", (.2, .6)))
        self.assertEqual((scenarios[0]["id"], scenarios[0]["prob"] * 3600), (6, 15.))

    def test_invalid_saved_conditions_fail_before_simulation(self):
        for key, value in (("prob", float("nan")), ("scale", -1.),
                           ("fixed_patient_count", 501), ("initial_occupancy_range", [.6, .2])):
            with self.subTest(key=key):
                manifest = self.saved()
                manifest["web_scenario"][key] = value
                with self.assertRaises(ValueError):
                    resolve_evaluation_profile(manifest, SimulationConfig())

    def test_both_cli_entrypoints_pass_restored_web_demand_and_occupancy(self):
        helper = best_fixtures.BestModelSelectionTests()
        for module in (run_scenarios, compare_queue_policies):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                helper.model(directory, "dqn_v6_web_best.pth")
                argv = [module.__name__ + ".py", "--output-root", directory, "--seeds", "32001"]
                with helper.cli(module, directory, argv) as (load, record, _):
                    load.return_value[4].update(self.saved())
                    module.main()
                    self.assertTrue(module.run_single.call_args_list)
                    for call in module.run_single.call_args_list:
                        self.assertEqual(call.args[2]["id"], 5)
                        self.assertEqual(call.args[2]["prob"], 15. / 3600.)
                        self.assertEqual(call.kwargs["initial_occupancy_range"], (.35, .5))
                    start = next(c for c in record.call_args_list if c.args[2] == "run_start")
                    self.assertEqual(start.kwargs["metrics"]["profile_source"], "saved_web")

    def test_cli_repeat_override_and_occupancy_reach_shared_wrapper(self):
        with patch.object(run_scenarios, "run_episode", return_value={}) as episode:
            run_scenarios.run_single(None, [], SCENARIOS[0], 32001, "shortest_time",
                                     config=SimulationConfig(), initial_occupancy_range=(.35, .5))
            self.assertEqual(episode.call_args.kwargs["initial_occupancy_range"], (.35, .5))
        helper = best_fixtures.BestModelSelectionTests()
        for module in (run_scenarios, compare_queue_policies):
            for flag, expected in (("--repeat-background-traffic", True),
                                   ("--no-repeat-background-traffic", False)):
                with self.subTest(module=module.__name__, flag=flag), tempfile.TemporaryDirectory() as directory:
                    helper.model(directory, "dqn_v6_web_best.pth")
                    argv = [module.__name__ + ".py", "--output-root", directory, "--seeds", "32001", flag]
                    with helper.cli(module, directory, argv):
                        module.main()
                        self.assertTrue(all(c.kwargs["config"].repeat_background_traffic == expected
                                            for c in module.run_single.call_args_list))

    def test_frozen_checkpoint_loads_after_original_model_and_manifest_change(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            with patch.object(simulation_setup, "network_digest", return_value="abc"):
                manifest = simulation_setup.make_manifest(None, [fixtures.hospital()], ["e"], [["h"]],
                                                          SimulationConfig(), 10, "severity", 300, ())
            tag = "v6_" + manifest["signature"]
            mp = root / "training" / f"experiment_{tag}.json"
            mp.parent.mkdir()
            mp.write_text(json.dumps(manifest), encoding="utf-8")
            agent = DQNAmbulanceAgent(27, 1, architecture="hospital_shared_v6", learning_config=LearningConfig())
            agent.metadata = dict(manifest_path=str(mp), experiment_signature=manifest["signature"])
            model = root / "models" / f"dqn_{tag}_best.pth"
            agent.save_model(str(model), episode=7)
            frozen, recorded_root, snapshot = freeze_evaluation_model(model, "evaluation_proof")
            self.assertEqual(recorded_root, root.resolve())
            self.assertEqual(file_sha256(frozen), snapshot["model_sha256"])
            self.assertNotEqual(frozen, model)
            model.write_bytes(b"replaced weights")
            mp.write_text("{}", encoding="utf-8")
            with patch.object(simulation_setup, "SumoMedicalEnvironment", return_value=SimpleNamespace()), \
                 patch.object(simulation_setup, "network_digest", return_value="abc"):
                _, _, restored, _, _ = simulation_setup.load_experiment(frozen)
            self.assertEqual(restored.loaded_episode, 7)
            self.assertEqual(Path(restored.metadata["manifest_path"]), Path(snapshot["manifest_path"]))
            self.assertEqual(experiment_root(root / "experiment.json"), root.resolve())


if __name__ == "__main__":
    unittest.main()
