"""Native SUMO proof of occupancy bonuses/penalties and replay accounting."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "CODE"))
import sumolib
import torch
from baseline import state_dimension
from dqn_agent import DQNAmbulanceAgent
from experiment_config import SimulationConfig, LearningConfig
from reward import congestion_reward_profile
from simulation import run_episode


def verify():
    torch.set_num_threads(2)
    directory = ROOT / "results/verification/hospital_congestion"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "nodes.xml").write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="600" y="0"/><node id="c" x="1500" y="0"/></nodes>')
    (directory / "edges.xml").write_text('<edges><edge id="road" from="a" to="b" numLanes="2" speed="20"/><edge id="exit" from="b" to="c" numLanes="2" speed="20"/></edges>')
    subprocess.run([sumolib.checkBinary("netconvert"), "-n", str(directory / "nodes.xml"),
                    "-e", str(directory / "edges.xml"), "-o", str(directory / "net.xml")],
                   check=True, capture_output=True)
    (directory / "empty.rou.xml").write_text('<routes/>')
    config_file = directory / "run.sumocfg"
    config_file.write_text('<configuration><input><net-file value="net.xml"/><route-files value="empty.rou.xml"/></input></configuration>')
    hospitals = [dict(id=f"h{i}", name=f"Hospital {i}", type="권역응급의료센터",
                      occupancy=.2, sumo_x=1400., sumo_y=0.) for i in range(2)]
    env = SimpleNamespace(cfg_file=config_file, net_file=directory / "net.xml",
                          net=sumolib.net.readNet(str(directory / "net.xml")),
                          ambulance_edges=["road"], hospital_access_edges=[["exit"], ["exit"]],
                          update_hospital_coordinates=lambda: None, get_hospitals=lambda: hospitals)
    results, records = {}, {}
    for name, mode, occupancy in (("legacy", "penalty", .2), ("low", "balanced", .2),
                                  ("high", "balanced", .75)):
        config = SimulationConfig(arrival_steps=3, max_wait=300, mission_timeout=400,
                                  treatment_seconds=3, bed_fraction=.1, candidate_limit=1,
                                  congestion_reward_mode=mode)
        rows = []
        result = run_episode(env, hospitals, dict(id=1, prob=1., scale=1., fixed_patient_count=3),
                             42, "shortest_time", config, ambulance_count=3,
                             initial_occupancy_range=(occupancy, occupancy), output_dir=directory / name,
                             quiet=True, on_outcome=rows.append)
        assert result["arrived"] == 3 and result["remaining_reservations"] == 0
        assert result["congestion_reward_mode"] == mode
        assert abs(result["reward"] - sum(result["reward_components"].values())) < 1e-8
        results[name], records[name] = result, rows
        (directory / f"{name}.json").write_text(json.dumps(dict(result=result, patients=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    for old, new in zip(records["legacy"], records["low"]):
        assert (old["patient_id"], old["hospital_id"], old["finish_time"]) == (new["patient_id"], new["hospital_id"], new["finish_time"])
    low_by_occurrence = sorted(records["low"], key=lambda row: row["occurrence_time"])
    assert [round(row["reward_components"]["occupancy"], 6) for row in low_by_occurrence] == [4.8, 3.2, 1.6]
    assert abs(results["low"]["reward"] - results["legacy"]["reward"] - 16.8) < 1e-8
    assert all(row["reward_components"]["occupancy"] + row["reward_components"]["overcrowding"] < 0 for row in records["high"])
    config = SimulationConfig(arrival_steps=3, max_wait=300, mission_timeout=400,
                              treatment_seconds=3, bed_fraction=.1, candidate_limit=1,
                              congestion_reward_mode="balanced")
    agent = DQNAmbulanceAgent(state_dimension(2), 2, architecture="hospital_shared_v6",
                              learning_config=LearningConfig(batch_size=2, target_sync_interval=2))
    before = {key: value.detach().cpu().clone() for key, value in agent.model.state_dict().items()}
    learned = run_episode(env, hospitals, dict(id=1, prob=1., scale=1., fixed_patient_count=3),
                          42, "dqn", config, agent=agent, training=True, ambulance_count=3,
                          initial_occupancy_range=(.2, .2), output_dir=directory / "training", quiet=True)
    assert learned["episode_updates"] > 0
    assert abs(learned["learning_reward"] - learned["reward"]) < 1e-8
    assert learned["unattributed_learning_reward"] == 0
    assert any(not torch.equal(value, agent.model.state_dict()[key].detach().cpu()) for key, value in before.items())
    proof = dict(passed=True, profile=congestion_reward_profile("balanced"), patients_per_run=3,
                 low_congestion_bonuses=[row["reward_components"]["occupancy"] for row in low_by_occurrence],
                 high_congestion_scores=[sum(row["reward_components"][key] for key in ("occupancy", "overcrowding")) for row in records["high"]],
                 legacy_reward=results["legacy"]["reward"], balanced_reward=results["low"]["reward"],
                 identical_transports=True, exact_reward_difference=16.8,
                 training_updates=learned["episode_updates"], learning_reward_matches_patient_totals=True)
    (directory / "proof.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify()
