"""Training/validation bookkeeping independent of native SUMO and model weights."""
import json
import copy
from pathlib import Path
import queue
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from web_dashboard import run_worker, validate_options


class WorkerValidationTests(unittest.TestCase):
    def test_same_count_dqn_applies_new_types_without_rebuilding_its_roads_or_weights(self):
        from experiment_config import SimulationConfig
        original = [dict(id="real", name="Real", type="지역응급의료센터")]
        original += [dict(id=f"v{i}", virtual=True, type="권역응급의료센터", sumo_x=i, sumo_y=0.) for i in range(3)]
        env = SimpleNamespace(cfg_file="test", hospital_access_edges=[["saved"]])
        agent = SimpleNamespace(architecture="hospital_shared_v6")
        events = queue.Queue()
        options = validate_options(dict(mode="simulation", policy="dqn", virtual_count=3, duration=30,
            virtual_mix=dict(local_center=0, local_institution=0, reported_er=1)))
        result = dict(reward=0., avg_reward=0., generated_patients=0)
        with tempfile.TemporaryDirectory() as temp, patch("web_dashboard.ROOT", Path(temp)), \
             patch("web_dashboard.OUTPUT", Path(temp) / "runs"), patch("torch.set_num_threads"), \
             patch("simulation_setup.load_experiment", return_value=(env, copy.deepcopy(original), agent, SimulationConfig(), {"signature":"source"})), \
             patch("simulation_setup.prepare_environment") as prepare, \
             patch("dashboard_environment.resize_inference_agent") as resize, \
             patch("dashboard_traffic.prepare_background", return_value=([], {})), \
             patch("simulation.run_episode", return_value=result) as episode:
            run_worker(options, "source.pth", events, threading.Event(), threading.Event(), SimpleNamespace(value=300))
            prepare.assert_not_called(); resize.assert_not_called()
            virtual = [h for h in episode.call_args.args[1] if h.get("virtual")]
            self.assertEqual(sum(h["type"] == "권역응급의료센터" for h in virtual), 2)
            self.assertEqual(sum(h["type"] == "응급실운영신고기관" for h in virtual), 1)
            saved = json.loads(next((Path(temp)/"runs").glob("*/run.json")).read_text(encoding="utf-8"))
            self.assertTrue(saved["hospital_transfer"]["hospital_types_changed"])
            self.assertEqual(saved["virtual_hospital_mix"]["counts"]["reported_er"], 1)
            self.assertEqual(env.hospital_access_edges, [["saved"]])
            self.assertTrue(all(h["type"] == "권역응급의료센터" for h in original if h.get("virtual")))

    def run_training(self, validation_count, updates=0, demand_options=None):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            agent = SimpleNamespace(epsilon=1., training_updates=updates, model=Mock(), save_model=Mock())
            hospitals = [dict(id="h", name="Hospital")]
            events = queue.Queue()
            result = dict(reward=-2., avg_reward=-1., generated_patients=2, episode_updates=updates, training_updates=updates)
            options = validate_options(dict(mode="training", episodes=1, duration=30, emergency_yield=False, **(demand_options or {})))
            with patch("web_dashboard.ROOT", directory), patch("web_dashboard.OUTPUT", directory / "runs"), \
                 patch("torch.set_num_threads"), \
                 patch("simulation_setup.prepare_environment", return_value=(SimpleNamespace(cfg_file="test"), hospitals, [], {})), \
                 patch("dashboard_traffic.prepare_background", return_value=(["input.xml"], {"period_seconds": 3600})), \
                 patch("simulation_setup.make_manifest", return_value=dict(version=6, hospitals=hospitals)) as manifest, \
                 patch("dqn_agent.DQNAmbulanceAgent", return_value=agent), \
                 patch("simulation.run_episode", side_effect=[result, {**result, "generated_patients": validation_count}]) as episode:
                run_worker(options, None, events, threading.Event(), threading.Event(), SimpleNamespace(value=300))
            messages = list(events.queue)
            self.assertNotIn("error", [m.get("status") for m in messages])
            if not updates:
                self.assertIn("가중치 업데이트는 0회", messages[-1]["message"])
            self.assertEqual(manifest.call_args.args[8], ())
            saved_manifest = json.loads(next((directory / "runs").glob("*/experiment.json")).read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["validation_seeds"], [50042])
            if options["demand_mode"] == "count":
                self.assertEqual(saved_manifest["web_scenario"]["fixed_patient_count"], options["patient_count"])
                self.assertTrue(all(call.args[2]["fixed_patient_count"] == options["patient_count"] for call in episode.call_args_list))
            self.assertEqual([call.kwargs["emergency_yield"] for call in episode.call_args_list], [False, False])
            self.assertEqual([call.args[3] for call in episode.call_args_list], [10043, 50042])
            self.assertEqual([call.args[5].admission_mode for call in episode.call_args_list], ["arrival", "arrival"])
            self.assertEqual([call.args[5].congestion_reward_mode for call in episode.call_args_list], ["balanced", "balanced"])
            self.assertEqual(episode.call_args_list[0].args[2]["_route_files"], ["input.xml"])
            persisted = json.loads(next((directory / "runs").glob("*/run.json")).read_text(encoding="utf-8"))
            self.assertNotIn("_route_files", persisted["scenario"])
            self.assertEqual(persisted["hospital_congestion"]["mode"], "balanced")
            return [Path(call.args[0]).name for call in agent.save_model.call_args_list], messages

    def test_empty_validation_does_not_promote_an_unscored_best_model(self):
        paths, messages = self.run_training(0)
        self.assertFalse(any(path.endswith("_best.pth") for path in paths))
        result = next(m["result"] for m in messages if m["kind"] == "episode_end")
        self.assertIsNone(result["validation_avg_reward"])
        self.assertEqual(result["validation_generated_patients"], 0)

    def test_nonempty_validation_can_select_best_using_the_recorded_panel(self):
        paths, messages = self.run_training(2, updates=1)
        self.assertTrue(any(path.endswith("_best.pth") for path in paths))
        result = next(m["result"] for m in messages if m["kind"] == "episode_end")
        self.assertEqual(result["validation_avg_reward"], -1.)

    def test_initial_weights_are_not_promoted_as_a_new_trained_best(self):
        paths, _ = self.run_training(2, updates=0)
        self.assertFalse(any(path.endswith("_best.pth") for path in paths))

    def test_fixed_count_reaches_training_validation_and_the_saved_manifest(self):
        self.run_training(2, updates=1, demand_options=dict(demand_mode="count", patient_count=2))
