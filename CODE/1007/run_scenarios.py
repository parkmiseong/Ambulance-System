"""KTAS DQN 및 5개 baseline 시나리오 평가."""

import contextlib
import copy
import os
import random
import sys

import numpy as np
import sumolib
import traci

from baselines import HospitalRouters, build_state_vector
from dqn_agent import DQNAmbulanceAgent
from environment import SumoMedicalEnvironment
from reward import GOLDEN_TIME_LIMITS, calculate_reward, normalize_ktas

MAX_STEPS = 600
MODEL_PATH = "dqn_ambulance_model_ktas_v1.pth"
MAX_ESTIMATED_TRAVEL_TIME = 600.0
HOSPITAL_FULL_THRESHOLD = 0.85

SCENARIOS = [
    {"id": 1, "prob": 0.20, "scale": 0.3, "name": "원활-원활"},
    {"id": 2, "prob": 0.20, "scale": 0.5, "name": "원활-보통"},
    {"id": 3, "prob": 0.20, "scale": 0.8, "name": "원활-혼잡"},
    {"id": 4, "prob": 0.30, "scale": 0.3, "name": "보통-원활"},
    {"id": 5, "prob": 0.30, "scale": 0.5, "name": "보통-보통"},
    {"id": 6, "prob": 0.30, "scale": 0.8, "name": "보통-혼잡"},
    {"id": 7, "prob": 0.40, "scale": 0.3, "name": "혼잡-원활"},
    {"id": 8, "prob": 0.40, "scale": 0.5, "name": "혼잡-보통"},
    {"id": 9, "prob": 0.40, "scale": 0.8, "name": "혼잡-혼잡"},
]

STRATEGIES = [
    ("shortest_distance", "최단거리"),
    ("shortest_time", "최단시간"),
    ("bed", "병상기반"),
    ("rule", "치료역량기반"),
    ("heuristic", "종합휴리스틱"),
    ("dqn", "DQN"),
]


@contextlib.contextmanager
def suppress_sumo_stdout():
    old_out, old_err = sys.stdout, sys.stderr
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stdout, sys.stderr = devnull, devnull
            yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def start_sumo(env, scenario, seed):
    binary = sumolib.checkBinary("sumo")
    cmd = [binary, "-c", str(env.cfg_file), "--scale", str(scenario["scale"]),
           "--no-warnings", "true", "--no-step-log", "true",
           "--duration-log.disable", "true", "--seed", str(seed), "--start"]
    with suppress_sumo_stdout():
        traci.start(cmd)
        try:
            traci.vehicletype.copy("DEFAULT_VEHTYPE", "ambulance_custom")
            traci.vehicletype.setVehicleClass("ambulance_custom", "emergency")
            traci.vehicletype.setShapeClass("ambulance_custom", "emergency")
        except Exception:
            pass
    env.update_hospital_coordinates()


def hospital_routes(env, patient_edge, hospitals):
    result = copy.deepcopy(hospitals)
    for h in result:
        h["reachable"] = False
        h["hospital_edge"] = None
        h["route_length"] = None
        h["estimated_travel_time"] = MAX_ESTIMATED_TRAVEL_TIME
        x, y = h.get("sumo_x"), h.get("sumo_y")
        if x is None or y is None:
            continue
        try:
            nearby = env.net.getNeighboringEdges(x, y, 5000.0)
            candidates = [e for e, _ in sorted(nearby, key=lambda z: z[1]) if not e.getID().startswith(":")]
            for edge in candidates:
                try:
                    with suppress_sumo_stdout():
                        route = traci.simulation.findRoute(patient_edge, edge.getID(), vType="ambulance_custom", depart=-1)
                    if route is None or len(route.edges) == 0:
                        continue
                    h["reachable"] = True
                    h["hospital_edge"] = edge.getID()
                    h["route_length"] = float(getattr(route, "length", 0.0))
                    travel = float(getattr(route, "travelTime", 0.0))
                    if travel <= 0:
                        travel = max(h["route_length"] / 13.9, 1.0)
                    h["estimated_travel_time"] = min(travel, MAX_ESTIMATED_TRAVEL_TIME)
                    break
                except Exception:
                    continue
        except Exception:
            continue
    return result


def schedule(scenario, valid_edges, seed):
    rng = random.Random(seed)
    edges = [e for e in valid_edges if e and not str(e).startswith(":")]
    out = {}
    for step in range(1, MAX_STEPS + 1):
        if rng.random() < scenario["prob"]:
            out[step] = {"edge_id": rng.choice(edges), "ktas": rng.randint(1, 5)}
    return out


def valid_actions(hospitals):
    actions = [i for i, h in enumerate(hospitals)
               if h.get("reachable", False) and float(h.get("occupancy", 1.0)) < HOSPITAL_FULL_THRESHOLD]
    return actions or [i for i, h in enumerate(hospitals) if h.get("reachable", False)]


def choose(mode, router, patient_pos, ktas, hospitals, state, agent, actions):
    if mode == "dqn":
        return int(agent.select_action(state, actions))
    if mode == "shortest_time":
        return router.shortest_time_strategy(actions)
    if mode == "shortest_distance":
        return router.shortest_distance_strategy(actions)
    if mode == "bed":
        return router.bed_strategy(ktas, actions)
    if mode == "rule":
        return router.rule_based_strategy(ktas, actions)
    if mode == "heuristic":
        return router.heuristic_strategy(ktas, actions)
    raise ValueError(mode)


def run_single(env, base_hospitals, scenario, seed, mode, agent=None):
    random.seed(seed)
    np.random.seed(seed)
    hospitals = copy.deepcopy(base_hospitals)
    for h in hospitals:
        h["occupancy"] = random.uniform(0.20, 0.60)

    router = HospitalRouters()
    patients = schedule(scenario, env.get_valid_edges(), seed)
    total_reward = 0.0
    success = 0
    total_time = 0.0
    rejections = 0
    assignments = {h.get("id", str(i)): 0 for i, h in enumerate(hospitals)}

    try:
        start_sumo(env, scenario, seed)
        for step, patient in patients.items():
            try:
                edge_obj = env.net.getEdge(patient["edge_id"])
                pos = edge_obj.getFromNode().getCoord()[:2]
            except Exception:
                continue
            ktas = normalize_ktas(patient["ktas"])
            hospitals = hospital_routes(env, patient["edge_id"], hospitals)
            actions = valid_actions(hospitals)
            if not actions:
                continue
            state = build_state_vector(pos, ktas, hospitals)
            action = choose(mode, router, pos, ktas, hospitals, state, agent, actions)
            action = int(action)
            if action < 0 or action >= len(hospitals):
                continue
            selected = hospitals[action]
            rejected = action not in actions
            if rejected:
                rejections += 1
            occupancy = float(selected.get("occupancy", 0.0))
            travel = float(selected.get("estimated_travel_time", MAX_ESTIMATED_TRAVEL_TIME))
            failed = not selected.get("reachable", False)
            reward = calculate_reward(
                travel_time=travel, ktas=ktas, hospital=selected,
                golden_limit=GOLDEN_TIME_LIMITS[ktas],
                rejection_count=1 if rejected else 0,
                occupancy_at_selection=occupancy,
                route_failed=failed,
            )
            total_reward += reward
            total_time += travel
            if travel <= GOLDEN_TIME_LIMITS[ktas] and not failed:
                success += 1
            selected["occupancy"] = min(1.0, occupancy + 0.08)
            assignments[selected.get("id", str(action))] = assignments.get(selected.get("id", str(action)), 0) + 1
            try:
                traci.simulationStep()
            except Exception:
                pass
    finally:
        try:
            traci.close()
        except Exception:
            pass

    n = sum(assignments.values())
    return {
        "reward": total_reward,
        "avg_reward": total_reward / n if n else 0.0,
        "golden_rate": 100.0 * success / n if n else 0.0,
        "rejections": rejections,
        "avg_time": total_time / n if n else 0.0,
        "load_std": float(np.std(list(assignments.values()))) if assignments else 0.0,
        "missions": n,
    }


def print_result(label, r):
    print(f"  {label:<12} | 보상 {r['reward']:9.2f} | 평균보상 {r['avg_reward']:7.2f} | "
          f"골든타임 {r['golden_rate']:6.1f}% | 거부 {r['rejections']:3d} | "
          f"평균시간 {r['avg_time']:7.1f}s | 완료 {r['missions']:3d} | 부하편차 {r['load_std']:.2f}")


def main():
    env = SumoMedicalEnvironment()
    base_hospitals = copy.deepcopy(env.get_hospitals())
    state_dim = 3 + len(base_hospitals) * 4
    agent = DQNAmbulanceAgent(state_dim, len(base_hospitals))
    if not agent.load_model(MODEL_PATH):
        print(f"[오류] KTAS 모델이 없습니다: {MODEL_PATH}")
        print("먼저 train_agent.py를 실행하세요.")
        return
    agent.epsilon = 0.0
    agent.model.eval()

    print("\n" + "=" * 100)
    print("KTAS + 4종 응급의료기관 기반 시나리오 평가")
    print("=" * 100)
    for sid, _ in [(s["id"], s["name"]) for s in SCENARIOS]:
        print(f"[{sid}] {next(s['name'] for s in SCENARIOS if s['id'] == sid)}")
    selected = input("실행할 시나리오 번호(0=전체): ").strip()
    scenarios = SCENARIOS if selected == "0" else [s for s in SCENARIOS if s["id"] == int(selected)]
    if not scenarios:
        print("[오류] 시나리오 번호가 올바르지 않습니다.")
        return

    overall = {m: [] for m, _ in STRATEGIES}
    for sc in scenarios:
        seed = 10000 + sc["id"] * 100
        print(f"\n{'=' * 100}\n시나리오 {sc['id']} 평가 시작: {sc['name']}\n{'=' * 100}")
        for mode, label in STRATEGIES:
            print(f"\n[{label}]")
            result = run_single(env, base_hospitals, sc, seed, mode, agent if mode == "dqn" else None)
            overall[mode].append(result)
            print_result(label, result)

    print("\n" + "=" * 100)
    print("최종 평균 결과")
    print("=" * 100)
    for mode, label in STRATEGIES:
        rs = overall[mode]
        if not rs:
            continue
        print(f"{label:<16} | 총보상 {sum(x['reward'] for x in rs):10.2f} | "
              f"골든타임 {np.mean([x['golden_rate'] for x in rs]):6.1f}% | "
              f"거부 {sum(x['rejections'] for x in rs):4d} | "
              f"평균시간 {np.mean([x['avg_time'] for x in rs]):7.1f}s | "
              f"부하편차 {np.mean([x['load_std'] for x in rs]):.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
