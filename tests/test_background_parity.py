"""Periodic traffic parity, legacy restoration and learned-best selection."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from dashboard_traffic import PERIOD, background_settings, prepare_background
from dqn_agent import DQNAmbulanceAgent
from experiment_config import SimulationConfig, LearningConfig
import simulation
import simulation_setup
import train_agent
import test_simulation_system as fixtures


class BackgroundParityTests(unittest.TestCase):
    def manifest(self, enabled=False):
        with patch.object(simulation_setup, "network_digest", return_value="abc"):
            return simulation_setup.make_manifest(None, [fixtures.hospital()], ["e"], [["h"]],
                SimulationConfig(repeat_background_traffic=enabled), 10, "severity", 300, ())

    def test_new_cli_defaults_match_web_and_legacy_switch_is_explicit(self):
        for flags, enabled in (([], True), (["--no-repeat-background-traffic"], False),
                               (["--repeat-background-traffic"], True)):
            with self.subTest(flags=flags), patch.object(sys, "argv", ["train_agent.py", *flags]), \
                 patch.object(train_agent, "train") as train:
                train_agent.main()
                self.assertEqual(train.call_args.args[10].repeat_background_traffic, enabled)
        for value in (None, "false", 0, 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SimulationConfig(repeat_background_traffic=value)

    def test_new_rule_changes_signature_and_false_keeps_legacy_serialization(self):
        legacy, periodic = self.manifest(False), self.manifest(True)
        self.assertNotIn("repeat_background_traffic", legacy["config"])
        self.assertNotIn("background_traffic", legacy)
        self.assertFalse(SimulationConfig(**legacy["config"]).repeat_background_traffic)
        self.assertEqual(periodic["background_traffic"], background_settings(True))
        self.assertTrue(periodic["config"]["repeat_background_traffic"])
        self.assertNotEqual(legacy["signature"], periodic["signature"])

    def test_repeat_generation_covers_drain_and_keeps_original_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            original = '<routes><vType id="car"/><trip id="first" type="car" depart="0" from="a" to="b"/><trip id="later" type="car" depart="10" from="a" to="b"/></routes>'
            (directory / "trips.xml").write_text(original)
            cfg = directory / "run.sumocfg"
            cfg.write_text('<configuration><input><route-files value="trips.xml"/></input></configuration>')
            paths, metadata = prepare_background(cfg, directory / "output", PERIOD + 5)
            self.assertEqual((directory / "trips.xml").read_text(), original)
            self.assertEqual(metadata["period_seconds"], PERIOD)
            self.assertEqual(metadata["repeated_trips"], 1)
            repeated = ET.parse(paths[-1]).getroot().findall("trip")
            self.assertEqual([float(t.get("depart")) for t in repeated], [float(PERIOD)])
            self.assertTrue(repeated[0].get("id").endswith("_first"))

    def test_shared_engine_prepares_periodic_traffic_for_cli(self):
        native_start, original_ns = fixtures.FakeTraCI.start, fixtures.SimpleNamespace
        def namespace(*args, **kwargs):
            if "ambulance_edges" in kwargs:
                kwargs["cfg_file"] = "test.sumocfg"
            return original_ns(*args, **kwargs)
        config = SimulationConfig(arrival_steps=2, max_wait=20, mission_timeout=10,
            treatment_seconds=3, bed_fraction=.1, repeat_background_traffic=True)
        with patch.object(fixtures, "SimpleNamespace", side_effect=namespace), \
             patch.object(fixtures.FakeTraCI, "start", autospec=True, side_effect=native_start) as start, \
             patch("dashboard_traffic.prepare_background", return_value=(["base.xml", "repeat.xml"],
                   {"repeated_trips": 2, "sources": []})) as prepare:
            result, _, _ = fixtures.SharedEngineTests().run_case(config=config)
        self.assertEqual(prepare.call_args.args[0], "test.sumocfg")
        self.assertEqual(prepare.call_args.args[2], config.drain_deadline)
        self.assertEqual(start.call_args.args[2]["_route_files"], ["base.xml", "repeat.xml"])
        self.assertEqual(result["background_traffic"]["repeated_trips"], 2)
        self.assertTrue(result["background_traffic"]["enabled"])

    def test_web_prepared_routes_are_not_repeated_twice(self):
        prepared = dict(id=1, prob=.2, scale=.3, _route_files=["web-repeat.xml"],
                        background_traffic={"repeated_trips": 2})
        config = SimulationConfig(arrival_steps=2, max_wait=20, mission_timeout=10,
            treatment_seconds=3, bed_fraction=.1, repeat_background_traffic=True)
        with patch.object(simulation, "scenario_for_config", return_value=prepared), \
             patch("dashboard_traffic.prepare_background") as prepare:
            result, _, _ = fixtures.SharedEngineTests().run_case(config=config)
        prepare.assert_not_called()
        self.assertEqual(result["background_traffic"]["repeated_trips"], 2)

    def load(self, manifest, directory):
        manifest["signature"] = simulation_setup.manifest_signature(manifest)
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        agent = DQNAmbulanceAgent(27, 1, architecture="hospital_shared_v6", learning_config=LearningConfig())
        agent.metadata = dict(manifest_path=str(path), experiment_signature=manifest["signature"])
        model = Path(directory) / "model_best.pth"
        with redirect_stdout(io.StringIO()):
            agent.save_model(model, episode=2)
            with patch.object(simulation_setup, "SumoMedicalEnvironment", return_value=SimpleNamespace()), \
                 patch.object(simulation_setup, "network_digest", return_value="abc"):
                return simulation_setup.load_experiment(model)[3]

    def test_loader_preserves_legacy_cli_web_and_new_rule(self):
        old_web = self.manifest(False)
        old_web["web_scenario"] = {"background_traffic": {"period_seconds": PERIOD, "repeated_trips": 2}}
        for manifest, expected in ((self.manifest(False), False), (self.manifest(True), True),
                                   (old_web, True)):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                self.assertEqual(self.load(manifest, directory).repeat_background_traffic, expected)

    def test_loader_rejects_changed_repeat_period(self):
        manifest = self.manifest(True)
        manifest["background_traffic"]["period_seconds"] = 1800
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "background traffic"):
            self.load(manifest, directory)

    def test_zero_update_best_is_rejected_for_legacy_and_hourly_demand(self):
        result = dict(generated_patients=2, missions=2, arrived=2, failed=0,
                      censored=0, golden_success=1, golden_rate=50., avg_wait=0.,
                      reward=4., avg_reward=2., avg_loss=0.)
        validation = dict(generated_patients=2, avg_reward=2., arrival_rate=100., golden_rate=50.)
        for rates in ((), (1., 2., 3.)):
            for updates in (0, 1):
                agent = fixtures.FakeAgent(27)
                agent.training_updates = updates
                with self.subTest(rates=rates, updates=updates), tempfile.TemporaryDirectory() as directory, \
                     patch.object(train_agent, "prepare_environment", return_value=(None, [fixtures.hospital()], ["e"], [["h"]])), \
                     patch.object(train_agent, "make_manifest", return_value={"version": 6}), \
                     patch.object(train_agent, "DQNAmbulanceAgent", return_value=agent), \
                     patch.object(train_agent, "run_episode", side_effect=lambda *a, **k: dict(result)), \
                     patch.object(train_agent, "validate_panel", return_value=validation), \
                     patch.object(train_agent, "append_record"), patch.object(train_agent, "plot_training_metrics"), \
                     redirect_stdout(io.StringIO()):
                    train_agent.train(num_episodes=1, eval_every=1, quiet=True, output_root=directory,
                                      config=SimulationConfig(demand_rates_per_hour=rates))
                    best = [save for save in agent.saves if str(save[0]).endswith("_best.pth")]
                    self.assertEqual(bool(best), bool(updates))


if __name__ == "__main__":
    unittest.main()
