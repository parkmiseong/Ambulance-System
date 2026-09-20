#!/usr/bin/env python3

import contextlib
import copy
import os
import random
import sys
import time

import numpy as np
import sumolib
import torch
import traci

from baselines import HospitalRouters
from dqn_agent import DQNAmbulanceAgent
from environment import SumoMedicalEnvironment
from reward import calculate_reward


traci.init_log = lambda *args, **kwargs: None

MAX_STEPS = 600
MAX_AMBULANCES = 10
MISSION_TIMEOUT_SECONDS = 350.0
ARRIVAL_DISTANCE = 120.0
MODEL_PATH = "dqn_ambulance_model_v2.pth"

GOLDEN_TIME_LIMITS = {
    4: 120.0,
    3: 180.0,
    2: 240.0,
    1: 300.0,
}

SCENARIOS = [
    {"id": 1, "patient_freq": "원활", "traffic": "원활", "complexity": 2, "prob": 0.20, "scale": 0.3},
    {"id": 2, "patient_freq": "원활", "traffic": "보통", "complexity": 3, "prob": 0.20, "scale": 0.5},
    {"id": 3, "patient_freq": "원활", "traffic": "혼잡", "complexity": 4, "prob": 0.20, "scale": 0.8},
    {"id": 4, "patient_freq": "보통", "traffic": "원활", "complexity": 3, "prob": 0.30, "scale": 0.3},
    {"id": 5, "patient_freq": "보통", "traffic": "보통", "complexity": 4, "prob": 0.30, "scale": 0.5},
    {"id": 6, "patient_freq": "보통", "traffic": "혼잡", "complexity": 5, "prob": 0.30, "scale": 0.8},
    {"id": 7, "patient_freq": "혼잡", "traffic": "원활", "complexity": 4, "prob": 0.40, "scale": 0.3},
    {"id": 8, "patient_freq": "혼잡", "traffic": "보통", "complexity": 5, "prob": 0.40, "scale": 0.5},
    {"id": 9, "patient_freq": "혼잡", "traffic": "혼잡", "complexity": 6, "prob": 0.40, "scale": 0.8},
]


@contextlib.contextmanager
def suppress_sumo_stdout():
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), original_stdout_fd)
        os.dup2(devnull.fileno(), original_stderr_fd)
        try:
            yield
        finally:
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def get_reachable_edge_id(net, x, y, from_edge=None, vtype="ambulance_custom"):
    """병원 좌표 주변에서 실제 경로가 존재하는 edge를 찾습니다."""
    if x is None or y is None:
        return None

    radius = 100
    visited_edges = set()

    while radius <= 5000:
        nearby = net.getNeighboringEdges(x, y, radius)
        for edge, _distance in sorted(nearby, key=lambda item: item[1]):
            edge_id = edge.getID()
            if edge_id.startswith(":") or edge_id in visited_edges:
                continue
            visited_edges.add(edge_id)

            if from_edge is None or from_edge == edge_id:
                return edge_id

            try:
                route = traci.simulation.findRoute(from_edge, edge_id, vType=vtype)
                if route and len(route.edges) > 0:
                    return edge_id
            except Exception:
                continue

        radius += 200

    return None


def generate_fixed_patient_schedule(scenario, valid_edges, seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    schedule = {}

    for step in range(1, MAX_STEPS + 1):
        if random.random() < scenario["prob"]:
            schedule[step] = {
                "edge_id": random.choice(valid_edges),
                "severity": random.randint(1, 4),
            }

    return schedule


def build_state_vector(patient_pos, severity, hospitals):
    occupancies = [float(h["occupancy"]) for h in hospitals]
    mean_occupancy = float(np.mean(occupancies)) if occupancies else 0.0
    hospital_types = [
        1.0 if ("권역" in str(h.get("type", "")) or "지역" in str(h.get("type", ""))) else 0.0
        for h in hospitals
    ]

    state = np.asarray(
        [
            float(patient_pos[0]) / 10000.0,
            float(patient_pos[1]) / 10000.0,
            float(severity) / 4.0,
            0.0,
            1.0,
            1.0,
            mean_occupancy,
        ]
        + occupancies
        + hospital_types,
        dtype=np.float32,
    )

    expected_dim = 7 + len(hospitals) * 2
    if state.size != expected_dim:
        raise ValueError(f"상태 벡터 차원 오류: {state.size} != {expected_dim}")
    return state


def _start_sumo(env, scenario, seed_val):
    sumo_binary = sumolib.checkBinary("sumo")
    config = [
        sumo_binary,
        "-c",
        str(env.cfg_file),
        "--scale",
        str(scenario["scale"]),
        "--no-warnings",
        "true",
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
        "--seed",
        str(seed_val),
        "--start",
    ]

    with suppress_sumo_stdout():
        traci.start(config)
        traci.vehicletype.copy("DEFAULT_VEHTYPE", "ambulance_custom")
        traci.vehicletype.setVehicleClass("ambulance_custom", "emergency")
        traci.vehicletype.setShapeClass("ambulance_custom", "emergency")

    env.update_hospital_coordinates()
    delta_t = float(traci.simulation.getDeltaT())
    return delta_t if delta_t > 0 else 1.0


def _find_route(from_edge, to_edge):
    try:
        route = traci.simulation.findRoute(from_edge, to_edge, vType="ambulance_custom")
        if route and len(route.edges) > 0:
            return route
    except Exception:
        pass

    try:
        route = traci.simulation.findRoute(from_edge, to_edge)
        if route and len(route.edges) > 0:
            return route
    except Exception:
        pass

    return None


def _select_action(mode, router, patient_pos, severity, hospitals, agent, state, valid_actions=None):
    if mode == "dqn":
        return int(agent.select_action(state, valid_actions))
    if mode == "nearest":
        return int(router.nearest_strategy(patient_pos))
    if mode == "rule":
        return int(router.rule_based_strategy(severity))
    if mode == "heuristic":
        return int(router.heuristic_strategy(patient_pos, severity))
    raise ValueError(f"알 수 없는 전략: {mode}")


def _prepare_hospitals(initial_hospitals, env):
    """초기 점유율은 시나리오 복사본에서 유지하고 좌표만 최신 SUMO 좌표로 반영합니다."""
    coordinate_map = {
        h["id"]: (h.get("sumo_x"), h.get("sumo_y"))
        for h in env.get_hospitals()
    }
    hospitals = copy.deepcopy(initial_hospitals)

    for hospital in hospitals:
        if hospital["id"] in coordinate_map:
            x, y = coordinate_map[hospital["id"]]
            hospital["sumo_x"] = x
            hospital["sumo_y"] = y

    return hospitals


def run_single_scenario(scenario, initial_hospitals, seed_val, patient_schedule, mode="dqn", agent=None):
    env = SumoMedicalEnvironment()
    random.seed(seed_val)
    np.random.seed(seed_val)

    total_reward = 0.0
    completed_missions = 0
    golden_success_count = 0
    total_rejections = 0
    total_transfer_time = 0.0
    hospital_assignment_counts = {}

    try:
        step_length = _start_sumo(env, scenario, seed_val)
        hospitals = _prepare_hospitals(initial_hospitals, env)
        router = HospitalRouters(hospitals)
        hospital_assignment_counts = {h["id"]: 0 for h in hospitals}
        active_missions = {}
        spawned_count = 0

        for step in range(1, MAX_STEPS + 1):
            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError:
                break

            for hospital in hospitals:
                hospital["occupancy"] = max(0.1, hospital["occupancy"] - 0.0005)

            ambulance_ids = set(traci.vehicle.getIDList())

            # -------------------------------------------------------------
            # 1. 활성 미션 처리
            # -------------------------------------------------------------
            for amb_id in list(active_missions.keys()):
                mission = active_missions[amb_id]

                if amb_id not in ambulance_ids:
                    # 차량이 사라진 경우 실패 처리
                    travel_time = mission["elapsed_time"]
                    arrived = False
                    total_reward += calculate_reward(
                        travel_time,
                        mission["severity"],
                        mission["hospital"],
                        GOLDEN_TIME_LIMITS[mission["severity"]],
                        mission["rejections"],
                        mission["occupancy_at_selection"],
                        route_failed=True,
                    )
                    completed_missions += 1
                    total_transfer_time += travel_time
                    del active_missions[amb_id]
                    continue

                mission["elapsed_time"] += step_length

                try:
                    amb_pos = traci.vehicle.getPosition(amb_id)[:2]
                    target_pos = mission["target_pos"]
                    distance = float(np.hypot(amb_pos[0] - target_pos[0], amb_pos[1] - target_pos[1]))
                except Exception:
                    distance = float("inf")

                arrived = distance <= ARRIVAL_DISTANCE
                timeout = mission["elapsed_time"] >= MISSION_TIMEOUT_SECONDS

                if not arrived and not timeout:
                    continue

                travel_time = mission["elapsed_time"]
                severity = mission["severity"]
                golden_limit = GOLDEN_TIME_LIMITS[severity]
                success = arrived and travel_time <= golden_limit

                if success:
                    golden_success_count += 1

                reward = calculate_reward(
                    travel_time=travel_time,
                    severity=severity,
                    hospital=mission["hospital"],
                    golden_limit=golden_limit,
                    rejection_count=mission["rejections"],
                    occupancy_at_selection=mission["occupancy_at_selection"],
                    route_failed=not arrived,
                )

                total_reward += reward
                completed_missions += 1
                total_transfer_time += travel_time

                try:
                    traci.vehicle.remove(amb_id)
                except Exception:
                    pass
                del active_missions[amb_id]

            # -------------------------------------------------------------
            # 2. 신규 환자 발생
            # -------------------------------------------------------------
            if step not in patient_schedule:
                continue
            if len(active_missions) >= MAX_AMBULANCES:
                continue

            p_info = patient_schedule[step]
            patient_edge = p_info["edge_id"]
            severity = p_info["severity"]
            edge_obj = env.net.getEdge(patient_edge)
            if edge_obj is None:
                continue

            patient_pos = edge_obj.getFromNode().getCoord()[:2]
            state = build_state_vector(patient_pos, severity, hospitals)

            # [핵심] 규칙 기반 필터링을 통해 유효한 병원 후보(valid_actions) 추출
            valid_actions = []
            for i, hospital in enumerate(hospitals):
                if hospital["occupancy"] >= 0.85:
                    continue
                
                hospital_type = str(hospital.get("type", "일반"))
                if severity >= 3 and ("권역" in hospital_type or "지역" in hospital_type):
                    valid_actions.append(i)
                elif severity < 3 and "일반" in hospital_type:
                    valid_actions.append(i)

            # 적합한 병원이 모두 가득 찼을 경우: 점유율이 가장 낮은 병원들로 Fallback
            if not valid_actions:
                min_occ = min([h["occupancy"] for h in hospitals])
                valid_actions = [i for i, h in enumerate(hospitals) if h["occupancy"] <= min_occ + 0.1]

            try:
                # 필터링된 valid_actions 내에서만 행동 선택
                action = _select_action(
                    mode,
                    router,
                    patient_pos,
                    severity,
                    hospitals,
                    agent,
                    state,
                    valid_actions=valid_actions
                )
            except Exception:
                continue

            action = max(0, min(int(action), len(hospitals) - 1))
            selected_hosp = hospitals[action]
            rejection_count = 0

            # 거부가 발생하면 필터링된 후보 안에서 다시 선택 (평가 모드이므로 패널티 저장은 없음)
            while selected_hosp["occupancy"] >= 0.85 and rejection_count < 4:
                rejection_count += 1
                total_rejections += 1
                
                action = _select_action(
                    mode,
                    router,
                    patient_pos,
                    severity,
                    hospitals,
                    agent,
                    state,
                    valid_actions=valid_actions
                )
                action = int(max(0, min(action, len(hospitals) - 1)))
                selected_hosp = hospitals[action]

            occupancy_at_selection = float(selected_hosp["occupancy"])
            selected_hosp["occupancy"] = min(1.0, selected_hosp["occupancy"] + 0.08)

            hospital_assignment_counts[selected_hosp["id"]] += 1

            hospital_edge = get_reachable_edge_id(
                env.net,
                selected_hosp.get("sumo_x"),
                selected_hosp.get("sumo_y"),
                from_edge=patient_edge,
            )

            if hospital_edge is None:
                total_reward += calculate_reward(
                    0.0,
                    severity,
                    selected_hosp,
                    GOLDEN_TIME_LIMITS[severity],
                    rejection_count,
                    occupancy_at_selection,
                    route_failed=True,
                )
                completed_missions += 1
                continue

            route = _find_route(patient_edge, hospital_edge)
            if route is None:
                total_reward += calculate_reward(
                    0.0,
                    severity,
                    selected_hosp,
                    GOLDEN_TIME_LIMITS[severity],
                    rejection_count,
                    occupancy_at_selection,
                    route_failed=True,
                )
                completed_missions += 1
                continue

            spawned_count += 1
            ambulance_id = f"amb_eval_{step}_{spawned_count}"
            route_id = f"route_{ambulance_id}"

            try:
                traci.route.add(route_id, list(route.edges))
                traci.vehicle.add(
                    vehID=ambulance_id,
                    routeID=route_id,
                    typeID="ambulance_custom",
                    depart="now",
                )
            except Exception:
                total_reward += calculate_reward(
                    0.0,
                    severity,
                    selected_hosp,
                    GOLDEN_TIME_LIMITS[severity],
                    rejection_count,
                    occupancy_at_selection,
                    route_failed=True,
                )
                completed_missions += 1
                continue

            active_missions[ambulance_id] = {
                "severity": severity,
                "hospital": selected_hosp,
                "elapsed_time": 0.0,
                "rejections": rejection_count,
                "occupancy_at_selection": occupancy_at_selection,
                "state_vector": state,
                "action": action,
                "target_pos": (selected_hosp["sumo_x"], selected_hosp["sumo_y"]),
            }

        # -------------------------------------------------------------
        # 3. 시뮬레이션 종료 시 미완료 미션을 실패로 정리
        # -------------------------------------------------------------
        for amb_id, mission in list(active_missions.items()):
            travel_time = mission["elapsed_time"]
            total_reward += calculate_reward(
                travel_time,
                mission["severity"],
                mission["hospital"],
                GOLDEN_TIME_LIMITS[mission["severity"]],
                mission["rejections"],
                mission["occupancy_at_selection"],
                route_failed=True,
            )
            completed_missions += 1
            total_transfer_time += travel_time
            try:
                traci.vehicle.remove(amb_id)
            except Exception:
                pass

    except Exception as exc:
        return {
            "reward": 0.0,
            "avg_reward": 0.0,
            "golden_success_rate": 0.0,
            "rejections": total_rejections,
            "avg_time": 0.0,
            "load_std": 0.0,
            "completed_missions": completed_missions,
            "zero_reason": str(exc),
        }
    finally:
        try:
            traci.close()
        except Exception:
            pass

    success_rate = (
        golden_success_count / completed_missions * 100.0
        if completed_missions > 0
        else 0.0
    )
    avg_time = total_transfer_time / completed_missions if completed_missions > 0 else 0.0
    avg_reward = total_reward / completed_missions if completed_missions > 0 else 0.0
    load_std = float(np.std(list(hospital_assignment_counts.values()))) if hospital_assignment_counts else 0.0

    return {
        "reward": total_reward,
        "avg_reward": avg_reward,
        "golden_success_rate": success_rate,
        "rejections": total_rejections,
        "avg_time": avg_time,
        "load_std": load_std,
        "completed_missions": completed_missions,
    }


def print_result(label, result):
    print(
        f"  · {label:<10} | "
        f"보상: {result['reward']:8.1f} | "
        f"평균보상: {result['avg_reward']:6.2f} | "
        f"골든타임: {result['golden_success_rate']:5.1f}% | "
        f"거부: {result['rejections']:3d} | "
        f"평균시간: {result['avg_time']:6.1f}s | "
        f"완료: {result['completed_missions']:3d} | "
        f"부하편차: {result['load_std']:.2f}"
    )


if __name__ == "__main__":
    print("=== 시나리오 기반 응급실 라우팅 평가 시스템 ===")

    temp_env = SumoMedicalEnvironment()
    valid_edges = temp_env.get_valid_edges()
    action_dim = len(temp_env.get_hospitals())
    state_dim = 7 + action_dim * 2

    dqn_agent = DQNAmbulanceAgent(state_dim, action_dim)
    dqn_agent.load_model(MODEL_PATH)
    dqn_agent.epsilon = 0.0
    print(f"[시스템] DQN 평가 모드 | state_dim={state_dim}, action_dim={action_dim}")

    print("\n[시나리오 목록]")
    for scenario in SCENARIOS:
        print(
            f"  [{scenario['id']}] 시나리오 {scenario['id']} "
            f"(환자: {scenario['patient_freq']} | "
            f"교통: {scenario['traffic']} | "
            f"복잡도: {scenario['complexity']})"
        )
    print("  [0] 전체 시나리오 (1~9)")

    selected_input = input("\n실행할 시나리오 번호를 입력하세요 (0 ~ 9): ").strip()

    if selected_input == "0":
        selected_scenarios = SCENARIOS
    else:
        try:
            selected_id = int(selected_input)
            selected_scenarios = [s for s in SCENARIOS if s["id"] == selected_id]
            if not selected_scenarios:
                selected_scenarios = [SCENARIOS[0]]
        except ValueError:
            selected_scenarios = [SCENARIOS[0]]

    overall = {
        "nearest": {"reward": 0.0, "golden": 0.0, "rejections": 0, "time": 0.0, "load": 0.0, "completed": 0},
        "rule": {"reward": 0.0, "golden": 0.0, "rejections": 0, "time": 0.0, "load": 0.0, "completed": 0},
        "heuristic": {"reward": 0.0, "golden": 0.0, "rejections": 0, "time": 0.0, "load": 0.0, "completed": 0},
        "dqn": {"reward": 0.0, "golden": 0.0, "rejections": 0, "time": 0.0, "load": 0.0, "completed": 0},
    }

    session_offset = int(time.time() * 1000) % 100000

    for scenario in selected_scenarios:
        print(f"\n[진행 중] 시나리오 {scenario['id']} 평가...")
        seed = scenario["id"] * 1000 + session_offset
        fixed_schedule = generate_fixed_patient_schedule(scenario, valid_edges, seed)
        scenario_hospitals = copy.deepcopy(temp_env.get_hospitals())

        results = {
            "nearest": run_single_scenario(scenario, scenario_hospitals, seed, fixed_schedule, mode="nearest"),
            "rule": run_single_scenario(scenario, scenario_hospitals, seed, fixed_schedule, mode="rule"),
            "heuristic": run_single_scenario(scenario, scenario_hospitals, seed, fixed_schedule, mode="heuristic"),
            "dqn": run_single_scenario(scenario, scenario_hospitals, seed, fixed_schedule, mode="dqn", agent=dqn_agent),
        }

        print(f"\n[시나리오 {scenario['id']} 상세 결과]")
        print_result("최단거리", results["nearest"])
        print_result("규칙기반", results["rule"])
        print_result("휴리스틱", results["heuristic"])
        print_result("DQN", results["dqn"])
        print("-" * 110)

        for key, result in results.items():
            overall[key]["reward"] += result["reward"]
            overall[key]["golden"] += result["golden_success_rate"]
            overall[key]["rejections"] += result["rejections"]
            overall[key]["time"] += result["avg_time"]
            overall[key]["load"] += result["load_std"]
            overall[key]["completed"] += result["completed_missions"]

    count = len(selected_scenarios)
    print("\n=== 최종 시나리오 평가 종합 결과 ===")
    for key, label in [
        ("nearest", "최단거리"),
        ("rule", "규칙기반"),
        ("heuristic", "휴리스틱"),
        ("dqn", "DQN 에이전트"),
    ]:
        item = overall[key]
        print(
            f"  · {label:<10} | "
            f"총 보상: {item['reward']:8.1f} | "
            f"평균 골든타임: {item['golden'] / count:5.1f}% | "
            f"총 거부: {item['rejections']:4d} | "
            f"평균 소요시간: {item['time'] / count:6.1f}s | "
            f"완료미션: {item['completed']:4d} | "
            f"평균 부하편차: {item['load'] / count:.2f}"
        )

    print("=" * 110)
