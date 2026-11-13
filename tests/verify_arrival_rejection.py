"""Native SUMO: admission after arrival and physical same-vehicle transfers."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "CODE"))
import sumolib
import torch
import traci
from baseline import state_dimension
from dqn_agent import DQNAmbulanceAgent
from experiment_config import SimulationConfig, LearningConfig
from simulation import run_episode


def verify():
    torch.set_num_threads(2)
    directory = ROOT / "results/verification/arrival_rejection"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "nodes.xml").write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="600" y="0"/><node id="c" x="1100" y="0"/><node id="d" x="1700" y="0"/><node id="e" x="1700" y="400"/><node id="f" x="0" y="400"/></nodes>')
    (directory / "edges.xml").write_text('<edges>' + ''.join(
        f'<edge id="{name}" from="{a}" to="{b}" numLanes="2" speed="20"/>'
        for name, a, b in [('road','a','b'),('hospital_a','b','c'),('hospital_b','c','d'),
                           ('return_d','d','e'),('return_e','e','f'),('return_f','f','a')]) + '</edges>')
    subprocess.run([sumolib.checkBinary("netconvert"), "-n", str(directory / "nodes.xml"),
                    "-e", str(directory / "edges.xml"), "-o", str(directory / "net.xml")], check=True, capture_output=True)
    (directory / "empty.rou.xml").write_text('<routes/>')
    cfg = directory / "run.sumocfg"
    cfg.write_text('<configuration><input><net-file value="net.xml"/><route-files value="empty.rou.xml"/></input></configuration>')
    hospitals = [dict(id=name, name=name, type="권역응급의료센터", occupancy=.7, sumo_x=x, sumo_y=0.)
                 for name, x in [('A', 1000.), ('B', 1600.)]]
    env = SimpleNamespace(cfg_file=cfg, net_file=directory / "net.xml",
                          net=sumolib.net.readNet(str(directory / "net.xml")), ambulance_edges=["road"],
                          hospital_access_edges=[["hospital_a"], ["hospital_b"]],
                          update_hospital_coordinates=lambda: None, get_hospitals=lambda: hospitals)
    config = SimulationConfig(arrival_steps=3, max_wait=1000, mission_timeout=1200,
                              treatment_seconds=240, bed_fraction=.1, candidate_limit=1,
                              admission_mode="arrival", congestion_reward_mode="balanced")
    rows, events, positions, reservation_counts = [], [], {}, []
    def rejection(event):
        assert traci.vehicle.isStopped(event["vehicle_id"])
        assert traci.vehicle.getRoadID(event["vehicle_id"]) == event["edge_id"]
        assert event["time"] > event["leg_dispatch_time"] >= event["dispatch_time"]
        events.append({**event, "physical_stop_confirmed": True,
                       "position": list(traci.vehicle.getPosition(event["vehicle_id"]))})
    def frame(snap):
        reservation_counts.append(sum(h["reserved_count"] for h in snap["hospitals"]))
        visible = set(traci.vehicle.getIDList())
        for vid in snap["manager"].active:
            if vid in visible:
                positions.setdefault(vid, []).append((snap["time"], *traci.vehicle.getPosition(vid)))
    result = run_episode(env, hospitals, dict(id=1, prob=1., scale=1., fixed_patient_count=3),
                         42, "shortest_time", config, ambulance_count=3,
                         initial_occupancy_range=(.7, .7), output_dir=directory / "simulation",
                         quiet=True, on_outcome=rows.append, on_rejection=rejection, on_step=frame)
    assert result["arrived"] == 3 and len(events) > 0 and result["hospital_redirects"] > 0
    assert not any(reservation_counts) and result["remaining_reservations"] == 0
    assert result["reward_components"]["rejection"] == -12 * len(events)
    assert result["rejection_count"] == len(events)
    assert len(positions) == 3  # No replacement vehicles were spawned for rejected patients.
    for row in rows:
        assert {visit["vehicle_id"] for visit in row["hospital_visits"]} == {row["vehicle_id"]}
        assert row["hospital_visits"][-1]["accepted"]
        assert row["wait_seconds"] == row["dispatch_time"] - row["occurrence_time"]
        assert abs(sum(row["reward_components"].values()) - row["reward"]) < 1e-8
    transferred = next(row for row in rows if len(row["hospital_visits"]) > 1)
    first, second = transferred["hospital_visits"][:2]
    assert first["edge_id"] != second["edge_id"] and second["arrival_time"] > first["arrival_time"]
    travel_positions = [(x, y) for time, x, y in positions[transferred["vehicle_id"]]
                        if first["arrival_time"] <= time < second["arrival_time"]]
    assert len(set(travel_positions)) > 2
    (directory / "simulation.json").write_text(json.dumps(dict(result=result, patients=rows, events=events), ensure_ascii=False, indent=2), encoding="utf-8")
    agent = DQNAmbulanceAgent(state_dimension(2), 2, architecture="hospital_shared_v6",
                              learning_config=LearningConfig(batch_size=2, target_sync_interval=2))
    trained = run_episode(env, hospitals, dict(id=1, prob=1., scale=1., fixed_patient_count=3),
                          42, "dqn", config, agent=agent, training=True, ambulance_count=3,
                          initial_occupancy_range=(.7, .7), output_dir=directory / "training", quiet=True)
    assert trained["arrived"] == 3 and trained["rejection_count"] > 0 and trained["episode_updates"] > 0
    assert abs(trained["learning_reward"] - trained["reward"]) < 1e-8
    assert trained["unattributed_learning_reward"] == 0
    proof = dict(passed=True, mode="arrival", pre_departure_check=True, advance_reservations=0,
                 patients=3, arrived=result["arrived"], rejections=len(events), redirects=result["hospital_redirects"],
                 penalty=result["reward_components"]["rejection"], same_vehicle=True,
                 physical_movement_after_refusal=True, events=events,
                 transferred_patient_visits=transferred["hospital_visits"],
                 training_refusals=trained["rejection_count"], training_updates=trained["episode_updates"],
                 learning_reward_matches_patient_totals=True)
    (directory / "proof.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify()
