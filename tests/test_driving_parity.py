"""One saved emergency-driving configuration across web, train and evaluation."""
from contextlib import contextmanager, redirect_stdout
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
from emergency_yield import yield_settings
from experiment_config import SimulationConfig, LearningConfig
import compare_queue_policies
import run_scenarios
import simulation_setup
import train_agent
from web_dashboard import validate_options
import test_best_model_selection as best_fixtures
import test_simulation_system as fixtures


class DrivingParityTests(unittest.TestCase):
    def manifest(self, enabled=False):
        with patch.object(simulation_setup, "network_digest", return_value="abc"):
            return simulation_setup.make_manifest(None, [fixtures.hospital()], ["e"], [["h"]],
                SimulationConfig(emergency_yield=enabled), 10, "severity", 300, ())

    @contextmanager
    def training_runner(self):
        result = dict(generated_patients=2, missions=2, arrived=2, failed=0,
                      censored=0, golden_success=1, golden_rate=50., avg_wait=0.,
                      reward=4., avg_reward=2., avg_loss=0.)
        with patch.object(train_agent, "prepare_environment",
                          return_value=(None, [fixtures.hospital()], ["e"], [["h"]])) as setup, \
             patch.object(simulation_setup, "network_digest", return_value="abc"), \
             patch.object(train_agent, "run_episode", side_effect=lambda *a, **k: dict(result)) as episode, \
             patch.object(train_agent, "append_record"), \
             patch.object(train_agent, "plot_training_metrics"), redirect_stdout(io.StringIO()):
            yield setup, episode

    def test_cli_matches_web_default_and_supports_legacy_off(self):
        for args, enabled in (([], True), (["--emergency-yield"], True),
                              (["--no-emergency-yield"], False)):
            with self.subTest(args=args), patch.object(sys, "argv", ["train_agent.py", *args]), \
                 patch.object(train_agent, "train") as train:
                train_agent.main()
                self.assertEqual(train.call_args.args[10].emergency_yield, enabled)
        self.assertTrue(validate_options({})["emergency_yield"])
        for value in (None, 0, 1, "false"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SimulationConfig(emergency_yield=value)

    def test_driving_settings_change_signature_and_keep_legacy_config(self):
        legacy, current = self.manifest(False), self.manifest(True)
        self.assertNotIn("emergency_yield", legacy["config"])
        self.assertNotIn("emergency_yield", legacy)
        self.assertFalse(SimulationConfig(**legacy["config"]).emergency_yield)
        self.assertTrue(current["config"]["emergency_yield"])
        self.assertEqual(current["emergency_yield"], yield_settings(True))
        self.assertNotEqual(current["signature"], legacy["signature"])

    def test_preparation_inherits_saved_setting(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), \
                 patch.object(simulation_setup, "SumoMedicalEnvironment", return_value=SimpleNamespace()), \
                 patch.object(simulation_setup, "start_sumo", side_effect=RuntimeError("stop setup")) as start, \
                 patch.object(simulation_setup.traci, "isLoaded", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "stop setup"):
                    simulation_setup.prepare_environment(SimulationConfig(emergency_yield=enabled))
                self.assertEqual(start.call_args.kwargs["emergency_yield"], enabled)

    def test_shared_engine_passes_setting_to_native_launcher_in_train_and_eval(self):
        original = fixtures.FakeTraCI.start
        helper = fixtures.SharedEngineTests()
        for enabled in (False, True):
            for training in (False, True):
                config = SimulationConfig(arrival_steps=2, max_wait=20, mission_timeout=10,
                    treatment_seconds=3, bed_fraction=.1, emergency_yield=enabled)
                with self.subTest(enabled=enabled, training=training), \
                     patch.object(fixtures.FakeTraCI, "start", autospec=True, side_effect=original) as start:
                    result, _, _ = helper.run_case(config=config, training=training)
                    self.assertEqual(start.call_args.kwargs["emergency_yield"], enabled)
                    self.assertEqual(result["emergency_yield"], yield_settings(enabled))

    def test_training_validation_and_same_setting_resume_share_config(self):
        with tempfile.TemporaryDirectory() as directory, self.training_runner() as (setup, episode):
            options = dict(num_episodes=3, eval_every=3, quiet=True, output_root=directory,
                           training_scenarios=(1, 5, 9), validation_scenarios=(1, 5, 9),
                           validation_seeds=(30001,))
            train_agent.train(**options)
            self.assertTrue(setup.call_args.args[0].emergency_yield)
            self.assertEqual(len(episode.call_args_list), 6)
            self.assertTrue(all(c.args[5].emergency_yield for c in episode.call_args_list))
            episode.reset_mock()
            train_agent.train(**{**options, "num_episodes": 1}, resume=True)
            self.assertTrue(all(c.args[5].emergency_yield for c in episode.call_args_list))
            self.assertEqual(episode.call_args_list[0].args[3], 10046)

    def test_changing_rules_cannot_resume_or_overwrite_legacy_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, self.training_runner() as (_, episode):
            options = dict(num_episodes=1, eval_every=0, quiet=True, output_root=directory)
            train_agent.train(config=SimulationConfig(), **options)
            checkpoint = next((Path(directory) / "training").glob("checkpoint_*.pth"))
            original = checkpoint.read_bytes()
            episode.reset_mock()
            with self.assertRaisesRegex(FileNotFoundError, "emergency driving"):
                train_agent.train(config=SimulationConfig(emergency_yield=True), resume=True, **options)
            episode.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), original)
            train_agent.train(config=SimulationConfig(), resume=True, **options)
            self.assertFalse(episode.call_args.args[5].emergency_yield)

    def load_manifest(self, manifest, directory):
        manifest["signature"] = simulation_setup.manifest_signature(manifest)
        mp = Path(directory) / "manifest.json"
        mp.write_text(json.dumps(manifest), encoding="utf-8")
        agent = DQNAmbulanceAgent(27, 1, architecture="hospital_shared_v6", learning_config=LearningConfig())
        agent.metadata = dict(manifest_path=str(mp), experiment_signature=manifest["signature"])
        model = Path(directory) / "model_best.pth"
        with redirect_stdout(io.StringIO()):
            agent.save_model(model, episode=3)
            with patch.object(simulation_setup, "SumoMedicalEnvironment", return_value=SimpleNamespace()), \
                 patch.object(simulation_setup, "network_digest", return_value="abc"):
                return simulation_setup.load_experiment(model)[3]

    def test_loader_restores_new_cli_legacy_cli_and_legacy_web_driving(self):
        legacy_web = self.manifest(False)
        legacy_web["web_emergency_yield"] = yield_settings(True)
        for manifest, enabled in ((self.manifest(False), False), (self.manifest(True), True),
                                  (legacy_web, True)):
            with self.subTest(enabled=enabled, keys=list(manifest)), tempfile.TemporaryDirectory() as directory:
                self.assertEqual(self.load_manifest(manifest, directory).emergency_yield, enabled)

    def test_loader_rejects_changed_native_parameters(self):
        manifest = self.manifest(True)
        manifest["emergency_yield"]["reaction_distance_m"] = 99.
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "driving settings"):
            self.load_manifest(manifest, directory)

    def test_evaluation_clis_use_saved_setting_or_explicit_override_and_log_both(self):
        helper = best_fixtures.BestModelSelectionTests()
        for module in (run_scenarios, compare_queue_policies):
            for saved, flags, expected in ((True, [], True), (False, [], False),
                                          (False, ["--emergency-yield"], True),
                                          (True, ["--no-emergency-yield"], False)):
                with self.subTest(module=module.__name__, flags=flags, saved=saved), \
                     tempfile.TemporaryDirectory() as directory:
                    helper.model(directory, "dqn_v6_current_best.pth")
                    argv = [module.__name__ + ".py", "--output-root", directory, "--seeds", "32001", *flags]
                    with helper.cli(module, directory, argv) as (load, record, _):
                        env, hospitals, agent, config, manifest = load.return_value
                        load.return_value = env, hospitals, agent, replace(config, emergency_yield=saved), manifest
                        module.main()
                        self.assertTrue(all(c.kwargs["config"].emergency_yield == expected
                                            for c in module.run_single.call_args_list))
                        start = next(c for c in record.call_args_list if c.args[2] == "run_start")
                        self.assertEqual(start.kwargs["metrics"]["emergency_yield"], yield_settings(expected))
                        self.assertEqual(start.kwargs["metrics"]["trained_emergency_yield"], yield_settings(saved))

    def test_mixed_driving_cannot_be_aggregated(self):
        with self.assertRaisesRegex(ValueError, "driving settings"):
            run_scenarios.aggregate_results([dict(emergency_yield=yield_settings(True)), {}])


if __name__ == "__main__":
    unittest.main()
