"""Native SUMO proof of admission refusal, reselection, and reward accounting."""
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
from simulation import run_episode


def verify():
    torch.set_num_threads(2)
    directory = ROOT / "results/verification/hospital_rejection"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "nodes.xml").write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="600" y="0"/><node id="c" x="1500" y="0"/></nodes>')
    (directory / "edges.xml").write_text('<edges><edge id="road" from="a" to="b" numLanes="2" speed="20"/><edge id="exit" from="b" to="c" numLanes="2" speed="20"/></edges>')
    subprocess.run([sumolib.checkBinary("netconvert"),"-n",str(directory/"nodes.xml"),"-e",str(directory/"edges.xml"),"-o",str(directory/"net.xml")],check=True,capture_output=True)
    (directory / "empty.rou.xml").write_text('<routes/>')
    config_file = directory / "run.sumocfg"
    config_file.write_text('<configuration><input><net-file value="net.xml"/><route-files value="empty.rou.xml"/></input></configuration>')
    hospitals = [dict(id=f"h{i}",name=f"Hospital {i}",type="권역응급의료센터",occupancy=.75,sumo_x=1400.,sumo_y=0.) for i in range(2)]
    env = SimpleNamespace(cfg_file=config_file,net_file=directory/"net.xml",net=sumolib.net.readNet(str(directory/"net.xml")),
                          ambulance_edges=["road"],hospital_access_edges=[["exit"],["exit"]],
                          update_hospital_coordinates=lambda:None,get_hospitals=lambda:hospitals)
    results, records, events = {}, {}, []
    for mode in ("precheck","request"):
        rows = []
        config = SimulationConfig(arrival_steps=5,max_wait=300,mission_timeout=400,treatment_seconds=3,
                                  bed_fraction=.1,candidate_limit=1,admission_mode=mode)
        result = run_episode(env,hospitals,dict(id=1,prob=1.,scale=1.,fixed_patient_count=5),42,
            "shortest_time",config,ambulance_count=5,initial_occupancy_range=(.75,.75),
            output_dir=directory/mode,quiet=True,on_outcome=rows.append,
            on_rejection=events.append if mode=="request" else None)
        results[mode], records[mode] = result, rows
        (directory/f"{mode}.json").write_text(json.dumps(dict(result=result,patients=rows),ensure_ascii=False,indent=2),encoding="utf-8")
    old, current = results["precheck"], results["request"]
    assert old["rejection_count"] == 0
    assert current["rejection_count"] > 0 and current["rejection_count"] == len(events)
    assert current["arrived"] == old["arrived"] == 5
    assert current["remaining_reservations"] == 0
    assert current["reward_components"]["rejection"] == -12 * len(events)
    assert abs(current["reward"] - old["reward"] + 12*len(events)) < 1e-8
    assert sum(row["rejection_count"] for row in records["request"]) == len(events)
    assert any(row["hospital_id"] != row["rejection_history"][0]["hospital_id"] for row in records["request"] if row["rejection_history"])
    old_by_id = {row["patient_id"]:row for row in records["precheck"]}
    for row in records["request"]:
        assert row["dispatch_time"] == old_by_id[row["patient_id"]]["dispatch_time"]
        assert row["finish_time"] == old_by_id[row["patient_id"]]["finish_time"]
        assert abs(sum(row["reward_components"].values())-row["reward"]) < 1e-8
    agent = DQNAmbulanceAgent(state_dimension(2),2,architecture="hospital_shared_v6",
                              learning_config=LearningConfig(batch_size=2,target_sync_interval=2))
    before = {key:value.detach().cpu().clone() for key,value in agent.model.state_dict().items()}
    learned = run_episode(env,hospitals,dict(id=1,prob=1.,scale=1.,fixed_patient_count=5),42,
        "dqn",config,agent=agent,training=True,ambulance_count=5,initial_occupancy_range=(.75,.75),
        output_dir=directory/"training",quiet=True)
    assert learned["rejection_count"] > 0 and agent.training_updates > 0
    assert abs(learned["learning_reward"]-learned["reward"]) < 1e-8
    assert learned["unattributed_learning_reward"] == 0
    assert any(not torch.equal(value,agent.model.state_dict()[key].detach().cpu()) for key,value in before.items())
    (directory/"training.json").write_text(json.dumps(learned,ensure_ascii=False,indent=2),encoding="utf-8")
    proof = dict(passed=True,patients=5,arrived=5,rejections=len(events),penalty_per_refusal=-12,
                 legacy_reward=old["reward"],request_reward=current["reward"],
                 actual_reselection=True,transport_timing_unchanged=True,remaining_reservations=0,events=events)
    proof.update(training_updates=agent.training_updates,training_refusals=learned["rejection_count"],
                 training_reward_matches_patient_totals=True)
    (directory/"proof.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({key:value for key,value in proof.items() if key != "events"},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    verify()
