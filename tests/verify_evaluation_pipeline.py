"""Optional native SUMO verification of frozen CLI evaluation artifacts.

Run: .venv/Scripts/python.exe -B tests/verify_evaluation_pipeline.py
Creates an isolated two-edge network and an untrained DQN test fixture. This
checks execution and reproducibility, not model quality or learning progress.
"""
import argparse
from contextlib import redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "CODE"))

import sumolib
import torch

from baseline import state_dimension
from dqn_agent import DQNAmbulanceAgent
from evaluation_audit import file_sha256, validate_evaluation_environment
from experiment_config import LearningConfig, SimulationConfig, seed_everything
from experiment_log import new_run_id
from reward import HOSPITAL_TYPES
import compare_queue_policies
import run_scenarios
import simulation_setup


def read_records(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{**row, "metrics": json.loads(row["metrics_json"])} for row in csv.DictReader(stream)]


def verify(output_root):
    output_root = Path(output_root).resolve()
    fixture = output_root / new_run_id("fixture")
    inputs = fixture / "inputs"
    inputs.mkdir(parents=True, exist_ok=False)
    (inputs / "nodes.xml").write_text(
        '<nodes><node id="a" x="0" y="0"/><node id="b" x="120" y="0"/>'
        '<node id="c" x="240" y="0"/></nodes>', encoding="utf-8")
    (inputs / "edges.xml").write_text(
        '<edges><edge id="road" from="a" to="b" numLanes="2" speed="20"/>'
        '<edge id="exit" from="b" to="c" numLanes="2" speed="20"/></edges>', encoding="utf-8")
    subprocess.run([sumolib.checkBinary("netconvert"), "-n", str(inputs / "nodes.xml"),
                    "-e", str(inputs / "edges.xml"), "-o", str(inputs / "net.xml")],
                   check=True, capture_output=True)
    (inputs / "background.trips.xml").write_text(
        '<routes><vType id="car" vClass="passenger"/>'
        '<trip id="background" type="car" depart="0" from="road" to="exit"/></routes>', encoding="utf-8")
    (inputs / "run.sumocfg").write_text(
        '<configuration><input><net-file value="net.xml"/>'
        '<route-files value="background.trips.xml"/></input></configuration>', encoding="utf-8")
    hospitals = [dict(id="native_hospital", name="Native fixture hospital", type=HOSPITAL_TYPES[0],
                      occupancy=.4, sumo_x=230., sumo_y=0., accepting=True, reachable=True)]

    class NativeEnvironment:
        def __init__(self):
            self.cfg_file = inputs / "run.sumocfg"
            self.net_file = inputs / "net.xml"
            self.net = sumolib.net.readNet(str(self.net_file))
            self.hospitals = copy.deepcopy(hospitals)
            self.ambulance_edges = ["road"]
            self.hospital_access_edges = [["exit"]]

        def update_hospital_coordinates(self):
            # Tiny fixture coordinates already use this network's SUMO frame.
            return None

        def get_hospitals(self):
            return self.hospitals

    torch.set_num_threads(2)
    seed_everything(42)
    config = SimulationConfig(arrival_steps=5, max_wait=10., mission_timeout=60.,
                              treatment_seconds=10., emergency_yield=True,
                              repeat_background_traffic=True)
    learning = LearningConfig()
    environment = NativeEnvironment()
    manifest = simulation_setup.make_manifest(environment, hospitals, ["road"], [["exit"]],
        config, 1, "severity", 300., (), learning_config=learning)
    manifest.update(
        validation_seeds=[30001], validation_demands=[1.],
        web_scenario=dict(id=5, prob=.2, scale=.5, fixed_patient_count=1,
                          demand_mode="count", patients_per_hour=720.,
                          initial_occupancy_range=[.35, .5]),
        test_fixture_only=True)
    manifest["signature"] = simulation_setup.manifest_signature(manifest)
    manifest_path = fixture / "experiment.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    model = fixture / "models" / f"dqn_v6_{manifest['signature']}_best.pth"
    agent = DQNAmbulanceAgent(state_dimension(1, config.observation_version), 1,
                              architecture="hospital_shared_v6", learning_config=learning)
    agent.metadata = dict(manifest_path=str(manifest_path), experiment_signature=manifest["signature"],
                          selection_metric="native_untrained_test_fixture")
    capture = io.StringIO()
    with redirect_stdout(capture):
        agent.save_model(model, episode=0)
    source_hash = file_sha256(model)
    source_manifest_hash = file_sha256(manifest_path)
    with patch.object(simulation_setup, "SumoMedicalEnvironment", NativeEnvironment), redirect_stdout(capture):
        with patch.object(sys, "argv", ["run_scenarios.py", "--model", str(model),
                "--strategies", "shortest_time,dqn", "--seeds", "32001",
                "--bootstrap-samples", "30", "--quiet"]):
            run_scenarios.main()
        with patch.object(sys, "argv", ["compare_queue_policies.py", "--model", str(model),
                                        "--seeds", "32001", "--quiet"]):
            compare_queue_policies.main()

        records = read_records(fixture / "experiment_history.csv")
        starts = [row for row in records if row["record_type"] == "run_start"]
        assert len(starts) == 2
        evidence = []
        for start in starts:
            metadata = start["metrics"]
            assert metadata["profile_source"] == "saved_web"
            snapshot = metadata["model_snapshot"]
            assert file_sha256(snapshot["model_path"]) == snapshot["model_sha256"] == source_hash
            assert file_sha256(snapshot["manifest_path"]) == snapshot["manifest_sha256"] == source_manifest_hash
            descriptor = Path(snapshot["manifest_path"]).parent / "evaluation.json"
            saved_descriptor = json.loads(descriptor.read_text(encoding="utf-8"))
            assert saved_descriptor["profile_source"] == "saved_web"
            # Resolve the copied adjacent manifest rather than its embedded source path.
            _, _, frozen_agent, _, frozen_manifest = simulation_setup.load_experiment(snapshot["model_path"])
            assert Path(frozen_agent.metadata["manifest_path"]) == Path(snapshot["manifest_path"])
            assert frozen_manifest["signature"] == manifest["signature"]
            environments = (metadata.get("evaluation_environments")
                            or {"policies": metadata["evaluation_environment"]})
            for applied in environments.values():
                validate_evaluation_environment(applied)
                settings = applied["settings"]
                assert settings["scenarios"] == [dict(id=5, prob=.2, scale=.5, fixed_patient_count=1)]
                assert settings["initial_occupancy_range"] == [.35, .5]
                assert settings["ambulance_count"] == 1
                assert settings["background_traffic"]["repeat"] is True
                assert settings["emergency_yield"]["enabled"] is True
                assert settings["inputs"]["status"] == "recorded"
            results = [row for row in records if row["run_id"] == start["run_id"]
                       and row["record_type"] in ("strategy_result", "queue_policy_result")]
            expected = {"shortest_time", "dqn"} if start["run_type"] == "evaluation" else {"fifo", "severity", "hybrid"}
            assert {row["strategy"] for row in results} == expected
            assert all(row["metrics"]["generated_patients"] == 1 for row in results)
            assert all(row["metrics"]["arrived"] == 1 for row in results)
            assert all(row["scenario_id"] == "5" for row in results)
            assert any(row["run_id"] == start["run_id"] and row["record_type"] == "run_end"
                       and row["status"] == "completed" for row in records)
            evidence.append(dict(run_id=start["run_id"], run_type=start["run_type"],
                profile_source="saved_web", policies=sorted(expected), patients_per_policy=1,
                arrived_per_policy=1, snapshot_sha256=snapshot["model_sha256"],
                model_path=snapshot["model_path"], evaluation_json=str(descriptor),
                environment_fingerprints={key: value["fingerprint"] for key, value in environments.items()}))
    assert file_sha256(model) == source_hash
    assert file_sha256(manifest_path) == source_manifest_hash
    (fixture / "cli.log").write_text(capture.getvalue(), encoding="utf-8")
    proof = dict(passed=True, test_fixture_only=True, performance_comparison=False,
                 source_best_unchanged=True, source_manifest_unchanged=True,
                 source_best=str(model), source_sha256=source_hash, evaluations=evidence)
    (fixture / "proof.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "proof.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT /
                        "results/verification/evaluation_reproducibility_20261116/native_cli")
    print(json.dumps(verify(parser.parse_args().output_root), ensure_ascii=False, indent=2))
