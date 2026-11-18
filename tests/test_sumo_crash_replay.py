"""Diagnostic replay must restore training state and preserve source artifacts."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from dqn_agent import DQNAmbulanceAgent
from experiment_config import LearningConfig, SimulationConfig
import diagnose_sumo_crash as diagnostic
import simulation_setup
import test_simulation_system as fixtures
from sumo_lane_snapshot import LaneSnapshotProbe
import traci
from test_diagnostic_route_input import fixture as route_fixture


class CrashReplayTests(unittest.TestCase):
    def checkpoint(self, root):
        config = SimulationConfig(demand_scale=1. / 72., emergency_yield=True,
                                  repeat_background_traffic=True, admission_mode="arrival",
                                  congestion_reward_mode="balanced")
        with patch.object(simulation_setup, "network_digest", return_value="abc"):
            manifest = simulation_setup.make_manifest(None, [fixtures.hospital()], ["e"], [["h"]],
                config, 10, "severity", 300, (), learning_config=LearningConfig())
        tag = "v6_" + manifest["signature"]
        mp = root / "training" / f"experiment_{tag}.json"
        mp.parent.mkdir(parents=True)
        mp.write_text(json.dumps(manifest), encoding="utf-8")
        a = DQNAmbulanceAgent(27, 1, architecture="hospital_shared_v6", learning_config=LearningConfig())
        a.metadata = dict(manifest_path=str(mp), experiment_signature=manifest["signature"])
        a.training_updates, a.epsilon = 17, .8
        a.store_transition(np.zeros(27), 0, 3., np.zeros(27), True)
        checkpoint = root / "training" / f"checkpoint_{tag}.pth"
        a.save_model(str(checkpoint), episode=24, full_state=True)
        return checkpoint, mp, config

    def test_next_episode_keeps_scenario_seed_and_curriculum_position(self):
        episode, scenario, seed, config = diagnostic.resolve_episode(
            dict(curriculum=[.1, .3], cycle_curriculum=True), SimulationConfig(), 24)
        self.assertEqual((episode, scenario["id"], seed), (25, 7, 10067))
        self.assertAlmostEqual(config.demand_scale, .1)
        episode, scenario, seed, config = diagnostic.resolve_episode(
            dict(training_scenarios=[6, 2], curriculum=[.1, .3]), SimulationConfig(), 24)
        self.assertEqual((episode, scenario["id"], seed), (25, 6, 10067))
        self.assertAlmostEqual(config.demand_scale, .3)

    def test_full_training_state_is_replayed_without_saving_or_touching_the_source(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            checkpoint, mp, config = self.checkpoint(root)
            model_bytes, manifest_bytes = checkpoint.read_bytes(), mp.read_bytes()
            env = SimpleNamespace()
            def fail(*args, **kwargs):
                agent = kwargs["agent"]
                self.assertTrue(kwargs["training"])
                self.assertTrue(agent.model.training)
                self.assertEqual(agent.loaded_episode, 24)
                self.assertEqual(agent.training_updates, 17)
                self.assertEqual(agent.epsilon, .8)
                self.assertEqual(len(agent.memory), 1)
                self.assertEqual(args[5], config)
                self.assertEqual((args[2]["id"], args[3]), (7, 10067))
                self.assertTrue(env.sumo_diagnostics)
                raise RuntimeError("native connection closed")
            with patch.object(simulation_setup, "SumoMedicalEnvironment", return_value=env), \
                 patch.object(simulation_setup, "network_digest", return_value="abc"), \
                 patch.object(diagnostic, "run_episode", side_effect=fail), \
                 patch.object(diagnostic, "prepare_diagnostic_route_input") as route_override, \
                 patch.object(DQNAmbulanceAgent, "save_model") as save:
                summary = diagnostic.run_diagnostic(checkpoint, root / "diagnostics")
            save.assert_not_called()
            route_override.assert_not_called()
            self.assertEqual(summary["status"], "diagnostic_failed")
            self.assertFalse(summary["expected_disconnect_reproduced"])
            self.assertTrue(summary["source_checkpoint_unchanged"])
            self.assertTrue(summary["source_manifest_unchanged"])
            self.assertEqual(checkpoint.read_bytes(), model_bytes)
            self.assertEqual(mp.read_bytes(), manifest_bytes)
            folder = Path(summary["output"])
            self.assertTrue((folder / "python_traceback.txt").exists())
            saved = json.loads((folder / "diagnostic_run.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["epsilon"], .8)
            self.assertEqual(saved["snapshot"]["checkpoint_sha256"], diagnostic.file_sha256(checkpoint))
            self.assertNotEqual(saved["snapshot"]["checkpoint"], str(checkpoint))

    def test_snapshot_rejects_weights_only_files_and_existing_destination(self):
        import torch
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            model = root / "model_best.pth"
            torch.save(dict(model_state_dict={}), model)
            with self.assertRaisesRegex(ValueError, "full training checkpoint"):
                diagnostic.snapshot_checkpoint(model, root / "bad")
            checkpoint, _, _ = self.checkpoint(root / "source")
            diagnostic.snapshot_checkpoint(checkpoint, root / "snapshot")
            with self.assertRaises(FileExistsError):
                diagnostic.snapshot_checkpoint(checkpoint, root / "snapshot")

    def test_locked_state_log_does_not_abort_simulation_or_claim_a_native_crash(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            checkpoint, _, _ = self.checkpoint(root)
            original_write = diagnostic.write_json
            state_write_attempts = []
            def locked_state(path, value):
                if Path(path).name == "last_successful_state.json":
                    state_write_attempts.append(value)
                    raise PermissionError("OneDrive temporarily locked the state file")
                original_write(path, value)
            def finish(*args, **kwargs):
                for tick in (1., 2., 3.):
                    kwargs["on_step"](dict(time=tick, manager=SimpleNamespace(waiting=[], active={}),
                                           outcomes=[], hospitals=[]))
                self.assertEqual(state_write_attempts, [])
                return dict(generated_patients=0)
            with patch.object(simulation_setup, "SumoMedicalEnvironment", return_value=SimpleNamespace()), \
                 patch.object(simulation_setup, "network_digest", return_value="abc"), \
                 patch.object(diagnostic, "run_episode", side_effect=finish), \
                 patch.object(diagnostic, "write_json", side_effect=locked_state):
                summary = diagnostic.run_diagnostic(checkpoint, root / "diagnostics")
            self.assertEqual(summary["status"], "completed_without_failure")
            self.assertIsNone(summary["failure"])
            self.assertFalse(summary["expected_disconnect_reproduced"])
            self.assertEqual(len(state_write_attempts), 1)
            self.assertEqual(summary["last_successful_state"]["time"], 3.)
            self.assertTrue(any("temporarily locked" in text for text in summary["logging_errors"]))

    def test_probe_json_failure_preserves_native_failure_and_reports_opt_in_settings(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            checkpoint, _, _ = self.checkpoint(root)
            native = SimpleNamespace(vehicle=SimpleNamespace(
                getIDList=lambda: ["amb_e_5"], getLaneID=lambda vid: "a_0",
                getTypeID=lambda vid: "ambulance_custom", getBestLanes=lambda vid: ()),
                lane=SimpleNamespace())
            def probe_factory(directory, start, end, **kwargs):
                return LaneSnapshotProbe(directory, start, end, native=native, **kwargs)
            def fail_after_observation(*args, **kwargs):
                state = dict(time=1100., manager=SimpleNamespace(waiting=[], active={}),
                             outcomes=[], hospitals=[])
                with patch("sumo_lane_snapshot.json.dumps", side_effect=TypeError("bad lane JSON")):
                    kwargs["on_step"](state)
                raise traci.exceptions.FatalTraCIError("Connection closed by SUMO.")
            with patch.object(simulation_setup, "SumoMedicalEnvironment", return_value=SimpleNamespace()), \
                 patch.object(simulation_setup, "network_digest", return_value="abc"), \
                 patch.object(diagnostic, "LaneSnapshotProbe", side_effect=probe_factory), \
                 patch.object(diagnostic, "run_episode", side_effect=fail_after_observation):
                summary = diagnostic.run_diagnostic(checkpoint, root / "diagnostics",
                    lane_probe_start=1100., lane_probe_end=1140.)
            self.assertEqual(summary["status"], "reproduced_sumo_disconnect")
            self.assertEqual(summary["failure"]["type"], "FatalTraCIError")
            self.assertEqual(summary["lane_probe"]["observed_ticks"], 1)
            self.assertEqual(summary["lane_probe"]["rows_written"], 0)
            self.assertEqual(summary["lane_probe"]["logging_error_count"], 1)
            self.assertTrue(any("bad lane JSON" in line for line in summary["logging_errors"]))
            self.assertTrue(summary["source_checkpoint_unchanged"])
            descriptor = json.loads((Path(summary["output"]) / "diagnostic_run.json").read_text(encoding="utf-8"))
            self.assertEqual(descriptor["lane_probe"]["start"], 1100.)
            self.assertEqual(descriptor["lane_probe"]["end"], 1140.)

    def test_route_override_uses_only_a_frozen_cfg_copy_and_preserves_learning_and_scenario(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            checkpoint, manifest, config = self.checkpoint(root / "experiment")
            cfg, route, candidate = route_fixture(root / "routes")
            before = {path: path.read_bytes() for path in (checkpoint, manifest, cfg, route, candidate)}
            original_env = SimpleNamespace(cfg_file=cfg)
            def finish(env, hospitals, scenario, seed, mode, actual_config, **kwargs):
                self.assertIsNot(env, original_env)
                self.assertEqual(original_env.cfg_file, cfg)
                self.assertNotEqual(env.cfg_file, cfg)
                self.assertEqual(env.cfg_file.parent.name, "input")
                self.assertEqual(actual_config, config)
                self.assertEqual((scenario["id"], seed), (7, 10067))
                self.assertNotIn("_route_files", scenario)
                self.assertEqual(kwargs["agent"].training_updates, 17)
                return dict(generated_patients=0)
            with patch.object(simulation_setup, "SumoMedicalEnvironment", return_value=original_env), \
                 patch.object(simulation_setup, "network_digest", return_value="abc"), \
                 patch.object(diagnostic, "run_episode", side_effect=finish):
                summary = diagnostic.run_diagnostic(checkpoint, root / "diagnostics",
                    background_route_override=candidate)
            self.assertEqual(summary["status"], "completed_without_failure")
            self.assertEqual(summary["background_route_override"]["kept_vehicle_ids"], ["a", "c"])
            self.assertTrue(all(summary["background_route_input_integrity"].values()))
            self.assertEqual({path: path.read_bytes() for path in before}, before)
            descriptor = json.loads((Path(summary["output"]) / "diagnostic_run.json").read_text(encoding="utf-8"))
            self.assertEqual(descriptor["background_route_override"], summary["background_route_override"])


if __name__ == "__main__":
    unittest.main()
