"""Optional native SUMO check: two lanes, ordinary cars and one ambulance.

Run with .venv/Scripts/python.exe -B tests/verify_emergency_yield.py.
Uses the production launcher and writes evidence under results/verification.
"""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "CODE"))
import sumolib
import traci
from simulation_setup import start_sumo
from emergency_yield import nearby_traffic
from dashboard_traffic import prepare_background


def verify(enabled):
    directory = ROOT / "results" / "verification" / "emergency_yield" / ("enabled" if enabled else "disabled")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "nodes.xml").write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="600" y="0"/><node id="c" x="1500" y="0"/></nodes>')
    (directory / "edges.xml").write_text('<edges><edge id="road" from="a" to="b" numLanes="2" speed="20" width="3.2"/><edge id="exit" from="b" to="c" numLanes="2" speed="20" width="3.2"/></edges>')
    subprocess.run([sumolib.checkBinary("netconvert"), "-n", str(directory / "nodes.xml"),
                    "-e", str(directory / "edges.xml"), "-o", str(directory / "net.xml")],
                   check=True, capture_output=True)
    (directory / "routes.xml").write_text('<routes><vType id="car" maxSpeed="5" speedFactor="1" sigma="0" width="1.8"/><route id="r" edges="road exit"/></routes>')
    config = directory / "run.sumocfg"
    config.write_text('<configuration><input><net-file value="net.xml"/><route-files value="routes.xml"/></input></configuration>')
    env = SimpleNamespace(cfg_file=config, update_hospital_coordinates=lambda: None,
                          get_hospitals=lambda: [], net=sumolib.net.readNet(str(directory / "net.xml")))
    rows, subscribed = [], set()
    try:
        start_sumo(env, {"scale": 1.}, 42, output_dir=directory, quiet=True, emergency_yield=enabled)
        traci.vehicletype.setMaxSpeed("ambulance_custom", 14.)
        traci.vehicletype.setSpeedFactor("ambulance_custom", 1.)
        traci.vehicle.add("right", "r", "car", departLane="0", departPos="170", departSpeed="5")
        traci.vehicle.add("left", "r", "car", departLane="1", departPos="170", departSpeed="5")
        traci.vehicle.add("ambulance", "r", "ambulance_custom", departLane="0", departPos="115", departSpeed="14")
        for tick in range(80):
            traci.simulationStep()
            traffic, _ = nearby_traffic(env, ["ambulance"], subscribed, enabled)
            rows.append({"tick": tick + 1, "ambulance": {"position": traci.vehicle.getPosition("ambulance"),
                "speed": traci.vehicle.getSpeed("ambulance"), "lateral": traci.vehicle.getLateralLanePosition("ambulance")},
                "cars": [{"id": vid, "position": traci.vehicle.getPosition(vid),
                "lateral": traci.vehicle.getLateralLanePosition(vid),
                "rescueLane": traci.vehicle.getParameter(vid, "rescueLane"),
                "lane": traci.vehicle.getLaneID(vid)} for vid in ("right", "left")],
                "telemetry_yielding": [v["id"] for v in traffic if v["yielding"]]})
    finally:
        if traci.isLoaded():
            traci.close()
    (directory / "observations.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    influenced = [car for row in rows for car in row["cars"] if "ambulance" in car["rescueLane"].split()]
    detected = sorted({vid for row in rows for vid in row["telemetry_yielding"]})
    if enabled:
        assert {car["id"] for car in influenced} == {"left", "right"}, "Both lanes must react"
        assert any(car["lateral"] < -.1 for car in influenced if car["id"] == "right"), "Right lane must pull right"
        assert any(car["lateral"] > .1 for car in influenced if car["id"] == "left"), "Left lane must pull left"
        assert detected == ["left", "right"], "Dashboard must detect native yielding"
        assert rows[-1]["ambulance"]["position"][0] > max(car["position"][0] for car in rows[-1]["cars"]), "Ambulance must pass both cars"
        assert all(not car["rescueLane"] for car in rows[-1]["cars"]), "Cars must stop yielding after passage"
    else:
        assert not influenced and not detected, "Disabled mode must not trigger emergency yielding"
    return {"enabled": enabled, "responding_cars": detected,
            "right_min_lateral_m": min(car["lateral"] for row in rows for car in row["cars"] if car["id"] == "right"),
            "left_max_lateral_m": max(car["lateral"] for row in rows for car in row["cars"] if car["id"] == "left"),
            "restored_after_passage": all(not car["rescueLane"] for car in rows[-1]["cars"])}


def verify_background():
    directory = ROOT / "results/verification/emergency_yield/background"
    directory.mkdir(parents=True, exist_ok=True)
    network = ROOT / "results/verification/emergency_yield/enabled/net.xml"
    (directory / "trips.xml").write_text('<routes><vType id="car" vClass="passenger"/><trip id="first" type="car" depart="0" from="road" to="exit"/><trip id="last" type="car" depart="3599" from="road" to="exit"/></routes>')
    config = directory / "run.sumocfg"
    config.write_text(f'<configuration><input><net-file value="{network.as_posix()}"/><route-files value="trips.xml"/></input></configuration>', encoding="utf-8")
    paths, metadata = prepare_background(config, directory, 3605)
    env = SimpleNamespace(cfg_file=config, update_hospital_coordinates=lambda: None, get_hospitals=lambda: [])
    try:
        start_sumo(env, {"scale": 1., "_route_files": paths}, 42, output_dir=directory, quiet=True)
        traci.simulationStep(3600.)
        traci.simulationStep()
        departed = list(traci.simulation.getDepartedIDList())
        assert "dashboard_repeat_1_0_first" in departed, "Repeated traffic must still depart after one hour"
    finally:
        if traci.isLoaded():
            traci.close()
    proof = {"passed": True, "departed_after_one_hour": departed, **metadata}
    (directory / "proof.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    return proof


if __name__ == "__main__":
    proof = [verify(False), verify(True)]
    (ROOT / "results/verification/emergency_yield/proof.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps(proof, indent=2))
    print(json.dumps(verify_background(), ensure_ascii=False, indent=2))
