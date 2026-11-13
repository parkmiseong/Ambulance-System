"""Dashboard input and local HTTP boundary tests; SUMO is exercised separately."""
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from web_dashboard import (Dashboard, LEVELS, RunStopped, ThreadingHTTPServer, handler_for,
                           model_hospital_counts, run_worker, validate_options)


class DashboardOptionsTests(unittest.TestCase):
    def test_real_hospital_admission_defaults_on_and_requires_a_boolean(self):
        self.assertTrue(validate_options({})["hospital_rejection"])
        self.assertFalse(validate_options({"hospital_rejection":False})["hospital_rejection"])
        for value in (0, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_options({"hospital_rejection":value})

    def test_all_27_scenarios_keep_the_three_independent_levels(self):
        combinations = set()
        for road in range(3):
            for demand in range(3):
                for occupancy in range(3):
                    options = validate_options(dict(traffic=road, demand=demand, occupancy=occupancy))
                    combinations.add((LEVELS["traffic"][options["traffic"]],
                                      LEVELS["demand"][options["demand"]],
                                      LEVELS["occupancy"][options["occupancy"]]))
        self.assertEqual(len(combinations), 27)

    def test_invalid_inputs_cannot_launch_work(self):
        invalid = [None, [], dict(traffic=3), dict(demand=-1), dict(occupancy=True),
                   dict(ambulances=11), dict(duration=0), dict(episodes=1.5),
                   dict(seed=2**31), dict(mode="shell"), dict(policy="unknown"),
                   dict(virtual_count=-1), dict(virtual_count=11), dict(virtual_count=1.5),
                   dict(virtual_count=True), dict(virtual_count="2"), dict(virtual_count=1)]
        for body in invalid:
            with self.subTest(body=body), self.assertRaises(ValueError):
                validate_options(body)

    def test_virtual_hospital_count_allows_zero_and_defaults_to_two(self):
        self.assertEqual(validate_options({})["virtual_count"], 2)
        for count in (0, 4, 10):
            self.assertEqual(validate_options({"virtual_count": count})["virtual_count"], count)

    def test_realistic_demand_defaults_to_one_hour_and_allows_a_day(self):
        self.assertEqual(validate_options({})["duration"], 3600)
        self.assertEqual(validate_options({"duration": 86400})["duration"], 86400)
        with self.assertRaises(ValueError):
            validate_options({"duration": 86401})

    def test_custom_demand_accepts_explicit_zero_and_fractional_hourly_rates(self):
        self.assertEqual(validate_options({})["demand_mode"], "preset")
        for rate in (0, 2.14, 50, 3600):
            self.assertEqual(validate_options(dict(demand_mode="rate", custom_patients_per_hour=rate))["custom_patients_per_hour"], rate)
        for count in (0, 15, 30):
            self.assertEqual(validate_options(dict(demand_mode="count", duration=30, patient_count=count))["patient_count"], count)

    def test_custom_demand_rejects_missing_nan_boolean_and_out_of_range_inputs(self):
        for value in (None, -1, 3601, True, "20", float("nan"), float("inf")):
            with self.subTest(rate=value), self.assertRaises(ValueError):
                validate_options(dict(demand_mode="rate", custom_patients_per_hour=value))
        for value in (None, -1, 31, 1.5, True, "2"):
            with self.subTest(count=value), self.assertRaises(ValueError):
                validate_options(dict(demand_mode="count", duration=30, patient_count=value))
        with self.assertRaises(ValueError):
            validate_options(dict(demand_mode="unknown"))


class VirtualHospitalConfigurationTests(unittest.TestCase):
    def test_model_count_comes_from_the_matching_saved_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "models" / "dqn_v6_abc_best.pth"
            manifest = dict(version=6, signature="abc", hospitals=[{}, {"virtual": True}, {"virtual": True}])
            (root / "experiment.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(model_hospital_counts(model), dict(hospital_count=3, virtual_count=2))
            manifest["signature"] = "another-experiment"
            (root / "experiment.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIsNone(model_hospital_counts(model)["virtual_count"])

    def test_legacy_model_directory_manifest_supports_zero_virtual_hospitals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "training").mkdir()
            manifest = dict(version=6, signature="abc", hospitals=[{}, {}])
            (root / "training" / "experiment_v6_abc.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(model_hospital_counts(root / "models" / "dqn_v6_abc_best.pth"),
                             dict(hospital_count=2, virtual_count=0))

    def test_requested_count_reaches_the_existing_environment_builder(self):
        for mode, policy in (("training", "dqn"), ("simulation", "shortest_time")):
            for count in (0, 4):
                with self.subTest(mode=mode, count=count), tempfile.TemporaryDirectory() as temp, \
                     patch("web_dashboard.OUTPUT", Path(temp)), patch("web_dashboard.ROOT", Path(temp)), \
                     patch("torch.set_num_threads"), \
                     patch("simulation_setup.prepare_environment", side_effect=RunStopped) as prepare:
                    options = validate_options(dict(mode=mode, policy=policy, virtual_count=count))
                    run_worker(options, None, queue.Queue(), threading.Event(), threading.Event(),
                               SimpleNamespace(value=300))
                    self.assertEqual(prepare.call_args.kwargs["virtual_count"], count)
                    self.assertTrue(prepare.call_args.kwargs["emergency_yield"])
                    self.assertEqual(prepare.call_args.kwargs["virtual_mix"], dict(local_center=5, local_institution=4, reported_er=1))

    def test_saved_dqn_reuses_its_original_environment_when_the_count_matches(self):
        from experiment_config import SimulationConfig
        stopped = threading.Event()
        hospitals = [{"id": "real"}, {"id": "virtual1", "virtual": True}, {"id": "virtual2", "virtual": True}]
        def restore(_):
            stopped.set()
            return None, hospitals, None, SimulationConfig(), {}
        with tempfile.TemporaryDirectory() as temp, patch("web_dashboard.OUTPUT", Path(temp)), \
             patch("web_dashboard.ROOT", Path(temp)), \
             patch("torch.set_num_threads"), patch("simulation_setup.load_experiment", side_effect=restore), \
             patch("simulation_setup.prepare_environment") as prepare:
            options = validate_options(dict(mode="simulation", policy="dqn", virtual_count=2))
            run_worker(options, "saved-best.pth", queue.Queue(), stopped, threading.Event(),
                       SimpleNamespace(value=300))
            prepare.assert_not_called()
            self.assertEqual(options["virtual_count"], 2)

    def test_saved_shared_dqn_rebuilds_the_requested_hospital_configuration(self):
        from experiment_config import SimulationConfig
        hospitals = [{"id": "real"}, {"id": "virtual", "virtual": True}]
        for count in (0, 4):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temp, \
                 patch("web_dashboard.OUTPUT", Path(temp)), patch("web_dashboard.ROOT", Path(temp)), \
                 patch("torch.set_num_threads"), \
                 patch("simulation_setup.load_experiment", return_value=(None, hospitals, SimpleNamespace(architecture="hospital_shared_v6"), SimulationConfig(), {})), \
                 patch("simulation_setup.prepare_environment", side_effect=RunStopped) as prepare:
                options = validate_options(dict(mode="simulation", policy="dqn", virtual_count=count))
                run_worker(options, "saved-best.pth", queue.Queue(), threading.Event(), threading.Event(), SimpleNamespace(value=300))
                self.assertEqual(prepare.call_args.kwargs["virtual_count"], count)

    def test_flat_legacy_dqn_does_not_fabricate_weights_for_new_hospitals(self):
        from experiment_config import SimulationConfig
        events = queue.Queue()
        with tempfile.TemporaryDirectory() as temp, patch("web_dashboard.OUTPUT", Path(temp)), \
             patch("web_dashboard.ROOT", Path(temp)), patch("torch.set_num_threads"), patch("traceback.print_exc"), \
             patch("simulation_setup.load_experiment", return_value=(None, [{"id": "real"}], SimpleNamespace(architecture="flat_v3"), SimulationConfig(), {})), \
             patch("simulation_setup.prepare_environment") as prepare:
            options = validate_options(dict(mode="simulation", policy="dqn", virtual_count=2))
            run_worker(options, "saved-best.pth", events, threading.Event(), threading.Event(), SimpleNamespace(value=300))
            prepare.assert_not_called()
            self.assertEqual(list(events.queue)[-1]["status"], "error")
            self.assertIn("v3", list(events.queue)[-1]["message"])

    def test_virtual_mix_validation_rejects_invalid_weights_and_zero_additional_mix(self):
        for weights in ({"local_center": 1}, {"local_center": True, "local_institution": 4, "reported_er": 1},
                        {"local_center": -1, "local_institution": 4, "reported_er": 1},
                        {"local_center": 1.5, "local_institution": 4, "reported_er": 1},
                        {"local_center": 0, "local_institution": 0, "reported_er": 0}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                validate_options(dict(virtual_count=3, virtual_mix=weights))
        options = validate_options(dict(virtual_count=2, virtual_mix=dict(local_center=0, local_institution=0, reported_er=0)))
        self.assertEqual(sum(options["virtual_mix"].values()), 0)


class LocalHTTPTests(unittest.TestCase):
    def setUp(self):
        class FakeDashboard:
            token = "test-local-token"
            starts = 0
            def start(self, body):
                validate_options(body)
                self.starts += 1
            def snapshot(self):
                return {"status": "idle"}
        self.dashboard = FakeDashboard()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.dashboard))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_reading_state_does_not_start_a_run(self):
        with urlopen(self.url + "/api/state") as response:
            self.assertEqual(json.load(response), {"status": "idle"})
        self.assertEqual(self.dashboard.starts, 0)

    def test_launch_requires_the_local_dashboard_token(self):
        request = Request(self.url + "/api/start", data=b"{}", method="POST")
        with self.assertRaises(HTTPError) as rejected:
            urlopen(request)
        self.assertEqual(rejected.exception.code, 403)
        self.assertEqual(self.dashboard.starts, 0)
        request.add_header("X-Dashboard-Token", self.dashboard.token)
        with urlopen(request) as response:
            self.assertEqual(response.status, 200)
        self.assertEqual(self.dashboard.starts, 1)


class RunLifecycleTests(unittest.TestCase):
    def test_actual_refusal_updates_live_penalty_before_the_next_sumo_frame(self):
        state = self.collect([
            dict(kind="episode_start",episode=1),
            dict(kind="frame",time=1,reward=5.,completed_reward=5.,reward_components={"golden":5.}),
            dict(kind="hospital_rejection",row=dict(record_id="1:p1:1",penalty=-12.,hospital_id="h")),
            dict(kind="status",status="stopped",message="stopped")])
        self.assertEqual(state["live_rejection_count"],1)
        self.assertEqual(state["frame"]["reward"],-7.)
        self.assertEqual(state["frame"]["reward_components"]["rejection"],-12.)
        self.assertEqual(state["frame"]["completed_reward"],5.)

    def test_terminal_state_exposes_worker_cleanup_until_it_really_exits(self):
        dashboard = Dashboard.__new__(Dashboard)
        dashboard.lock = threading.RLock()
        dashboard.state = {"status": "completed"}
        dashboard.process = SimpleNamespace(is_alive=lambda: True)
        self.assertTrue(dashboard.snapshot()["worker_active"])
        dashboard.process = SimpleNamespace(is_alive=lambda: False)
        self.assertFalse(dashboard.snapshot()["worker_active"])

    def collect(self, messages, superseded=False):
        dashboard = Dashboard.__new__(Dashboard)
        dashboard.lock = threading.RLock()
        dashboard.pause_event = threading.Event()
        dashboard.state = {"status": "running", "history": []}
        dashboard.outcomes, dashboard.outcome_ids = [], set()
        class Events(queue.Queue):
            def close(self):
                pass
        events = Events()
        for message in messages:
            events.put(message)
        process = SimpleNamespace(join=lambda timeout: None, is_alive=lambda: False, exitcode=7)
        dashboard.process = SimpleNamespace(is_alive=lambda: True) if superseded else process
        dashboard.collect(process, events)
        return dashboard.snapshot()

    def test_stopping_clears_vehicles_after_sumo_closes(self):
        state = self.collect([
            dict(kind="frame", active=2, waiting=3, vehicles=[{"id": "a"}], patients=[{}], reward=5),
            dict(kind="status", status="stopped", message="stopped")])
        self.assertEqual(state["status"], "stopped")
        self.assertEqual(state["frame"]["active"], 0)
        self.assertEqual(state["frame"]["vehicles"], [])
        self.assertEqual(state["frame"]["reward"], 5)
        self.assertGreater(state["frame_revision"], 1)

    def test_unexpected_worker_exit_also_clears_map_traffic(self):
        state = self.collect([dict(kind="frame", active=1, vehicles=[{}], traffic=[{}], yielding_count=1)])
        self.assertEqual(state["status"], "error")
        self.assertEqual(state["frame"]["vehicles"], [])
        self.assertEqual(state["frame"]["yielding_count"], 0)

    def test_previous_workers_cannot_change_the_next_run(self):
        state = self.collect([dict(kind="status", status="error", message="old worker")], superseded=True)
        self.assertEqual(state["status"], "running")
        self.assertNotIn("message", state)

    def test_patient_events_keep_all_episodes_and_ignore_duplicate_delivery(self):
        messages = [dict(kind="outcome", row=dict(record_id=f"1:p{i}")) for i in range(20)]
        messages += [dict(kind="outcome", row=dict(record_id="1:p0")),
                     dict(kind="outcome", row=dict(record_id="2:p0")),
                     dict(kind="status", status="completed", message="completed")]
        state = self.collect(messages)
        self.assertEqual(state["patient_count"], 21)

    def test_final_reward_is_available_during_validation_and_history_is_not_duplicated(self):
        result = dict(reward=10, generated_patients=2)
        state = self.collect([
            dict(kind="frame", active=1, waiting=0, vehicles=[{}]),
            dict(kind="episode_result", episode=1, result=result),
            dict(kind="episode_end", episode=1, result={**result, "validation_avg_reward": 4}),
            dict(kind="status", status="completed", message="completed")])
        self.assertEqual(len(state["history"]), 1)
        self.assertEqual(state["result"]["reward"], 10)
        self.assertEqual(state["result"]["validation_avg_reward"], 4)
        self.assertEqual(state["frame"]["active"], 0)
