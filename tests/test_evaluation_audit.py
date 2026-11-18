"""Runtime comparability and evaluation artifact retention without native SUMO."""
from copy import deepcopy
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "CODE"))
from compare_evaluations import compare_runs
from evaluation_audit import (build_evaluation_environment, file_sha256,
                              snapshot_evaluation_model, validate_evaluation_environment)
from experiment_config import SimulationConfig


class EvaluationAuditTests(unittest.TestCase):
    def environment(self, **overrides):
        args = dict(env=None, hospitals=[dict(id="H", type="regional", sumo_x=1., sumo_y=2.)],
                    config=SimulationConfig(), scenarios=[dict(id=1, prob=.2, scale=.3)],
                    demand_levels=[1.], ambulance_count=10, queue_policy="severity", aging_seconds=300.,
                    initial_occupancy_range=(.2, .6))
        args.update(overrides)
        return build_evaluation_environment(**args)

    def inputs(self, directory, suffix):
        directory.mkdir()
        (directory / "net.xml").write_text("<net/>")
        (directory / "trips.xml").write_text('<routes><trip id="p" depart="0" from="a" to="b"/></routes>')
        (directory / "extra.xml").write_text(f'<additional><edgeData id="e" file="{suffix}.xml"/></additional>')
        config = directory / "run.sumocfg"
        config.write_text(f'<configuration><input><net-file value="net.xml"/><route-files value="trips.xml"/>'
                          f'<additional-files value="extra.xml"/></input><processing><ignore-route-errors value="true"/>'
                          f'</processing><output><tripinfo-output value="{suffix}.xml"/></output></configuration>')
        return SimpleNamespace(cfg_file=config, net_file=directory / "net.xml",
                               ambulance_edges=["a", "b"], hospital_access_edges=[["b"]])

    def test_same_runtime_is_independent_of_training_seed_and_observation(self):
        baseline = self.environment()
        changed = self.environment(config=SimulationConfig(seed=43, observation_version=4),
                                   aging_seconds=300, demand_levels=[1])
        self.assertEqual(baseline["fingerprint"], changed["fingerprint"])
        self.assertEqual(validate_evaluation_environment(baseline), baseline["fingerprint"])

    def test_vehicle_count_capacity_wait_queue_occupancy_and_demand_affect_fingerprint(self):
        baseline = self.environment()["fingerprint"]
        changes = [dict(ambulance_count=1), dict(queue_policy="fifo"), dict(aging_seconds=200),
                   dict(initial_occupancy_range=(.35, .5)), dict(demand_levels=[.3]),
                   dict(config=replace(SimulationConfig(), max_wait=100)),
                   dict(config=replace(SimulationConfig(), bed_fraction=.1)),
                   dict(scenarios=[dict(id=1, prob=.001, scale=.3)]),
                   dict(hospitals=[dict(id="different", type="regional", sumo_x=1., sumo_y=2.)])]
        for options in changes:
            with self.subTest(options=options):
                self.assertNotEqual(self.environment(**options)["fingerprint"], baseline)

    def test_background_flag_and_yield_change_runtime_identity(self):
        config = SimpleNamespace(**vars(SimulationConfig()))
        config.repeat_background_traffic = True
        self.assertNotEqual(self.environment(config=config)["fingerprint"], self.environment()["fingerprint"])
        self.assertNotEqual(self.environment(config=SimulationConfig(emergency_yield=True))["fingerprint"],
                            self.environment()["fingerprint"])

    def test_current_reward_and_admission_tables_are_part_of_the_recorded_environment(self):
        baseline = self.environment()["fingerprint"]
        with patch("evaluation_audit.DEFAULT_UNCERTAINTY", .4):
            self.assertNotEqual(self.environment()["fingerprint"], baseline)
        with patch("evaluation_audit.REQUIRED_LEVEL", {1: 1}):
            self.assertNotEqual(self.environment()["fingerprint"], baseline)

    def test_original_input_content_and_engine_version_are_recorded_without_output_paths(self):
        with tempfile.TemporaryDirectory() as directory, patch("evaluation_audit._sumo_version", return_value="SUMO test"):
            root = Path(directory)
            a, b = self.inputs(root / "a", "/output_a"), self.inputs(root / "b", "/output_b")
            first, moved = self.environment(env=a), self.environment(env=b)
            self.assertEqual(first["fingerprint"], moved["fingerprint"])
            self.assertEqual(first["settings"]["inputs"]["sumo_version"], "SUMO test")
            (b.cfg_file.parent / "trips.xml").write_text('<routes><trip id="different" depart="0" from="a" to="b"/></routes>')
            self.assertNotEqual(self.environment(env=b)["fingerprint"], first["fingerprint"])
            with patch("evaluation_audit._sumo_version", return_value="SUMO other"):
                self.assertNotEqual(self.environment(env=a)["fingerprint"], first["fingerprint"])

    def test_missing_environment_is_explicit_and_does_not_start_sumo(self):
        with patch("evaluation_audit._sumo_version") as version:
            environment = self.environment()
        version.assert_not_called()
        self.assertEqual(environment["settings"]["inputs"]["status"], "unavailable")

    def test_cfg_network_is_checked_even_when_python_environment_uses_another_network(self):
        with tempfile.TemporaryDirectory() as directory, patch("evaluation_audit._sumo_version", return_value="SUMO test"):
            root = Path(directory)
            env = self.inputs(root / "input", "/output")
            separate = root / "python_net.xml"; separate.write_text("<net/>")
            env.net_file = separate
            baseline = self.environment(env=env)
            (env.cfg_file.parent / "net.xml").write_text('<net changed="true"/>')
            changed = self.environment(env=env)
            self.assertEqual(baseline["settings"]["inputs"]["network"], changed["settings"]["inputs"]["network"])
            self.assertNotEqual(baseline["fingerprint"], changed["fingerprint"])

    def test_arrival_mode_records_the_filtered_origins_and_hospital_access_used_by_the_engine(self):
        env = SimpleNamespace(net=SimpleNamespace(getEdges=lambda: []), ambulance_edges=["a", "outside"],
                              hospital_access_edges=[["b", "outside"]])
        with patch("route_safety.prepare_access", return_value=([["b"]], (["a"], dict(profile="test")))) as prepare:
            result = self.environment(env=env, config=SimulationConfig(admission_mode="arrival"))
        prepare.assert_called_once()
        self.assertEqual(result["settings"]["ambulance_edges"], ["a", "outside"])
        self.assertEqual(result["settings"]["effective_ambulance_edges"], ["a"])
        self.assertEqual(result["settings"]["effective_hospital_access_edges"], [["b"]])

    def test_environment_payload_tampering_is_detected(self):
        environment = self.environment()
        environment["settings"]["ambulance_count"] = 1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_evaluation_environment(environment)

    def test_snapshot_retains_exact_model_and_adjacent_manifest_after_source_best_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "dqn_v6_test_best.pth"; model.write_bytes(b"trained-model-v1")
            manifest = root / "manifest.json"; manifest.write_bytes(b'{"version":6}')
            metadata = snapshot_evaluation_model(model, manifest, root / "evaluation")
            frozen = Path(metadata["model_path"])
            self.assertEqual(frozen.parent.parent / "experiment.json", Path(metadata["manifest_path"]))
            self.assertEqual(metadata["model_sha256"], hashlib.sha256(b"trained-model-v1").hexdigest())
            self.assertEqual(metadata["manifest_sha256"], file_sha256(manifest))
            self.assertEqual(json.loads((root / "evaluation" / "snapshot.json").read_text()), metadata)
            model.write_bytes(b"trained-model-v2")
            self.assertEqual(frozen.read_bytes(), b"trained-model-v1")
            self.assertNotEqual(file_sha256(model), metadata["model_sha256"])
            with self.assertRaises(FileExistsError):
                snapshot_evaluation_model(model, manifest, root / "evaluation")
            self.assertEqual(frozen.read_bytes(), b"trained-model-v1")

    def test_snapshot_rejects_last_models_and_in_place_copy_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = root / "model_last.pth"; last.write_bytes(b"model")
            manifest = root / "manifest.json"; manifest.write_text("{}")
            with self.assertRaisesRegex(ValueError, "best"):
                snapshot_evaluation_model(last, manifest, root / "last")
            best = root / "model_best.pth"; best.write_bytes(b"model")
            with patch("evaluation_audit.os.fstat", side_effect=[
                    SimpleNamespace(st_size=5, st_mtime_ns=1), SimpleNamespace(st_size=5, st_mtime_ns=2)]):
                with self.assertRaisesRegex(ValueError, "changed in place"):
                    snapshot_evaluation_model(best, manifest, root / "changed")
            self.assertFalse((root / "changed/models/model_best.pth").exists())
            self.assertFalse((root / "changed/models/model_best.pth.tmp").exists())

    def log(self, path, environments):
        fields = ["run_id", "record_type", "model_path", "metrics_json", "status", "strategy", "scenario_id"]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for run, environment in zip(("a", "b"), environments):
                start = dict(model_kind="best", model_episode=3, evaluation_split="test",
                             model_snapshot=dict(model_path=f"snapshot/{run}_best.pth", model_sha256="1" * 64))
                if environment is not None:
                    start["evaluation_environment"] = environment
                writer.writerow(dict(run_id=run, record_type="run_start", model_path=f"{run}_best.pth",
                                     metrics_json=json.dumps(start)))
                patient = dict(patient_id="p1", seed=32001, demand_scale=1., reward=1.,
                               ktas=1, occurrence_time=1., edge_id="e", arrived=True,
                               golden_success=True, wait_seconds=0.)
                writer.writerow(dict(run_id=run, record_type="mission", model_path=f"{run}_best.pth",
                                     strategy="dqn", scenario_id=1, metrics_json=json.dumps(patient)))
                writer.writerow(dict(run_id=run, record_type="run_end", model_path=f"{run}_best.pth",
                                     status="completed", metrics_json="{}"))

    def test_comparison_rejects_different_vehicle_counts_even_with_matching_patient_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.csv"
            self.log(path, (self.environment(), self.environment(ambulance_count=1)))
            for allow_legacy in (False, True):
                with self.subTest(allow_legacy=allow_legacy), self.assertRaisesRegex(ValueError, "operating conditions"):
                    compare_runs(path, ["a", "b"], allow_legacy_environment=allow_legacy)

    def test_matching_comparison_retains_frozen_model_hash_and_environment_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.csv"; environment = self.environment()
            self.log(path, (environment, deepcopy(environment)))
            report = compare_runs(path, ["a", "b"])
        self.assertTrue(report["environment_verified"])
        self.assertEqual(report["environment_fingerprint"], environment["fingerprint"])
        self.assertEqual(report["runs"]["b"]["model_sha256"], "1" * 64)
        self.assertEqual(report["runs"]["b"]["model_snapshot"]["model_path"], "snapshot/b_best.pth")

    def test_comparison_requires_explicit_legacy_override_and_marks_it_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.csv"; self.log(path, (None, None))
            with self.assertRaisesRegex(ValueError, "allow-legacy-environment"):
                compare_runs(path, ["a", "b"])
            report = compare_runs(path, ["a", "b"], allow_legacy_environment=True)
        self.assertFalse(report["environment_verified"])
        self.assertIsNone(report["environment_fingerprint"])

    def test_comparison_rejects_forged_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.csv"; environment = self.environment()
            altered = deepcopy(environment); altered["settings"]["queue_policy"] = "fifo"
            self.log(path, (environment, altered))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                compare_runs(path, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
