#!/usr/bin/env python3

import contextlib
import copy
import os
import random
import sys
import time

import numpy as np
import sumolib
import traci

from baselines import HospitalRouters
from dqn_agent import DQNAmbulanceAgent
from environment import SumoMedicalEnvironment
from reward import calculate_reward


MODEL_PATH = "dqn_ambulance_model_v2.pth"
MAX_STEPS = 500
MAX_AMBULANCES = 30
MISSION_TIMEOUT_SECONDS = 350.0
ARRIVAL_DISTANCE = 120.0

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

GOLDEN_TIME_LIMITS = {4: 120.0, 3: 180.0, 2: 240.0, 1: 300.0}


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
    """좌표 주변에서 실제로 route가 생성되는 도로 edge를 찾습니다."""
    radius = 100
    visited_edges = set()

    while radius <= 5000:
        nearby = net.getNeighboringEdges(x, y, radius)
        for edge, _distance in sorted(nearby, key=lambda item: item[1]):
            edge_id = edge.getID()
            if edge_id.startswith(":") or edge_id in visited_edges:
                continue
            visited_edges.add(edge_id)

            if from_edge is None:
                return edge_id

            if edge_id == from_edge:
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
    hospital_count = len(hospitals)
    occupancies = [float(h["occupancy"]) for h in hospitals]
    mean_occupancy = float(np.mean(occupancies)) if occupancies else 0.0
    hospital_types = [
        1.0 if ("권역" in str(h.get("type", "")) or "지역" in str(h.get("type", ""))) else 0.0
        for h in hospitals
    ]

    base_info = [
        float(patient_pos[0]) / 10000.0,
        float(patient_pos[1]) / 10000.0,
        float(severity) / 4.0,
        0.0,
        1.0,
        1.0,
        mean_occupancy,
    ]

    state = np.asarray(base_info + occupancies + hospital_types, dtype=np.float32)
    expected_dim = 7 + hospital_count * 2
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
    step_length = float(traci.simulation.getDeltaT())
    if step_length <= 0:
        step_length = 1.0
    return step_length


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


def _valid_actions(hospitals):
    return list(range(len(hospitals)))


def train_dqn_fast(num_episodes=50):
    env = SumoMedicalEnvironment()
    valid_edges = env.get_valid_edges()
    temp_hospitals = env.get_hospitals()

    action_dim = len(temp_hospitals)
    state_dim = 7 + action_dim * 2

    agent = DQNAmbulanceAgent(state_dim, action_dim)
    agent.epsilon = 1.0

    print(f"=== DQN 강화 학습 시작 ({num_episodes} 에피소드) ===")
    print(f"[설정] state_dim={state_dim}, action_dim={action_dim}, device={agent.device}")

    for ep in range(1, num_episodes + 1):
        ep_start = time.time()
        scenario = random.choice(SCENARIOS)
        seed_val = random.randint(1, 999999)
        patient_schedule = generate_fixed_patient_schedule(scenario, valid_edges, seed_val)
        hospitals = copy.deepcopy(temp_hospitals)
        router = HospitalRouters(hospitals)

        active_missions = {}
        spawned_count = 0
        step_length = 1.0
        completed = 0
        total_episode_reward = 0.0

        try:
            step_length = _start_sumo(env, scenario, seed_val)

            for h in hospitals:
                for env_h in env.get_hospitals():
                    if h["id"] == env_h["id"]:
                        h["sumo_x"] = env_h["sumo_x"]
                        h["sumo_y"] = env_h["sumo_y"]

            for step in range(1, MAX_STEPS + 1):
                try:
                    traci.simulationStep()
                except traci.exceptions.FatalTraCIError:
                    break

                for hospital in hospitals:
                    hospital["occupancy"] = max(0.1, hospital["occupancy"] - 0.0005)

                ambulance_ids = set(traci.vehicle.getIDList())

                # ---------------------------------------------------------
                # 1. 기존 미션 업데이트
                # ---------------------------------------------------------
                for amb_id in list(active_missions.keys()):
                    mission = active_missions[amb_id]

                    if amb_id not in ambulance_ids:
                        # 차량이 사라진 경우에는 실패 transition으로 처리합니다.
                        reward = calculate_reward(
                            travel_time=mission["elapsed_time"],
                            severity=mission["severity"],
                            hospital=mission["hospital"],
                            golden_limit=GOLDEN_TIME_LIMITS[mission["severity"]],
                            rejection_count=mission["rejections"],
                            occupancy_at_selection=mission["occupancy_at_selection"],
                            route_failed=True,
                        )
                        agent.store_transition(
                            mission["state_vector"],
                            mission["action"],
                            reward,
                            np.zeros_like(mission["state_vector"]),
                            True,
                        )
                        agent.train_step()
                        total_episode_reward += reward
                        completed += 1
                        del active_missions[amb_id]
                        continue

                    mission["elapsed_time"] += step_length

                    try:
                        ambulance_pos = traci.vehicle.getPosition(amb_id)[:2]
                        target = mission["target_pos"]
                        distance = float(np.hypot(ambulance_pos[0] - target[0], ambulance_pos[1] - target[1]))
                    except Exception:
                        distance = float("inf")

                    arrived = distance <= ARRIVAL_DISTANCE
                    timeout = mission["elapsed_time"] >= MISSION_TIMEOUT_SECONDS

                    if not arrived and not timeout:
                        continue

                    travel_time = mission["elapsed_time"]
                    golden_limit = GOLDEN_TIME_LIMITS[mission["severity"]]
                    reward = calculate_reward(
                        travel_time=travel_time,
                        severity=mission["severity"],
                        hospital=mission["hospital"],
                        golden_limit=golden_limit,
                        rejection_count=mission["rejections"],
                        occupancy_at_selection=mission["occupancy_at_selection"],
                        route_failed=not arrived,
                    )

                    # 미션 종료는 반드시 terminal transition입니다.
                    agent.store_transition(
                        mission["state_vector"],
                        mission["action"],
                        reward,
                        np.zeros_like(mission["state_vector"]),
                        True,
                    )
                    agent.train_step()

                    total_episode_reward += reward
                    completed += 1

                    try:
                        traci.vehicle.remove(amb_id)
                    except Exception:
                        pass
                    del active_missions[amb_id]

                # ---------------------------------------------------------
                # 2. 신규 환자 발생
                # ---------------------------------------------------------
                if step not in patient_schedule or len(active_missions) >= MAX_AMBULANCES:
                    continue

                patient_info = patient_schedule[step]
                patient_edge = patient_info["edge_id"]
                severity = patient_info["severity"]
                edge_obj = env.net.getEdge(patient_edge)
                if edge_obj is None:
                    continue

                patient_pos = edge_obj.getFromNode().getCoord()[:2]
                state = build_state_vector(patient_pos, severity, hospitals)

                # [수정 후] 규칙 기반 알고리즘의 유효성 필터링 적용 (Action Masking)
                valid_actions = []
                for i, hospital in enumerate(hospitals):
                    # 점유율이 85% 이상(포화)인 병원은 우선 제외
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

                # 유효한 병원 목록(valid_actions) 중에서만 DQN 행동 선택
                if ep <= 15 and random.random() < 0.25:
                    action = router.rule_based_strategy(severity)
                else:
                    action = agent.select_action(state, valid_actions)

                action = int(max(0, min(action, len(hospitals) - 1)))
                selected_hosp = hospitals[action]
                rejection_count = 0

                # 병원 거부가 발생하면 새로운 상태에서 다시 선택합니다.
                # 평가 코드와 동일한 방식으로 한 번씩 재선택합니다.
                while selected_hosp["occupancy"] >= 0.85 and rejection_count < 4:
                    penalty_reward = -4.0  # reward.py의 REJECTION_PENALTY 수준
                    agent.store_transition(state, action, penalty_reward, state, True)
                    agent.train_step()

                    rejection_count += 1
                    # 새로운 병원 재선택
                    if ep <= 15 and random.random() < 0.25:
                        action = router.rule_based_strategy(severity)
                    else:
                        action = agent.select_action(state, _valid_actions(hospitals))
                    action = int(max(0, min(action, len(hospitals) - 1)))
                    selected_hosp = hospitals[action]

                occupancy_at_selection = float(selected_hosp["occupancy"])
                # 병원 선택 후 점유율을 증가시킵니다.
                selected_hosp["occupancy"] = min(1.0, selected_hosp["occupancy"] + 0.08)

                # 환자 edge -> 병원 주변의 실제 접근 가능한 edge를 탐색합니다.
                hospital_edge = get_reachable_edge_id(
                    env.net,
                    selected_hosp["sumo_x"],
                    selected_hosp["sumo_y"],
                    from_edge=patient_edge,
                )

                if hospital_edge is None:
                    reward = calculate_reward(
                        travel_time=0.0,
                        severity=severity,
                        hospital=selected_hosp,
                        golden_limit=GOLDEN_TIME_LIMITS[severity],
                        rejection_count=rejection_count,
                        occupancy_at_selection=occupancy_at_selection,
                        route_failed=True,
                    )
                    agent.store_transition(
                        state,
                        action,
                        reward,
                        np.zeros_like(state),
                        True,
                    )
                    agent.train_step()
                    continue

                route = _find_route(patient_edge, hospital_edge)
                if route is None:
                    reward = calculate_reward(
                        travel_time=0.0,
                        severity=severity,
                        hospital=selected_hosp,
                        golden_limit=GOLDEN_TIME_LIMITS[severity],
                        rejection_count=rejection_count,
                        occupancy_at_selection=occupancy_at_selection,
                        route_failed=True,
                    )
                    agent.store_transition(
                        state,
                        action,
                        reward,
                        np.zeros_like(state),
                        True,
                    )
                    agent.train_step()
                    continue

                spawned_count += 1
                ambulance_id = f"amb_train_{step}_{spawned_count}"
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
                    reward = calculate_reward(
                        travel_time=0.0,
                        severity=severity,
                        hospital=selected_hosp,
                        golden_limit=GOLDEN_TIME_LIMITS[severity],
                        rejection_count=rejection_count,
                        occupancy_at_selection=occupancy_at_selection,
                        route_failed=True,
                    )
                    agent.store_transition(state, action, reward, np.zeros_like(state), True)
                    agent.train_step()
                    continue

                active_missions[ambulance_id] = {
                    "hospital": selected_hosp,
                    "severity": severity,
                    "elapsed_time": 0.0,
                    "rejections": rejection_count,
                    "occupancy_at_selection": occupancy_at_selection,
                    "state_vector": state,
                    "action": action,
                    "target_pos": (selected_hosp["sumo_x"], selected_hosp["sumo_y"]),
                }

                # 미션마다 즉시 여러 번 학습하지 않고, terminal transition이 생길 때 학습합니다.

        except Exception as exc:
            print(f"[Episode {ep}] 실행 중 오류: {exc}")
        finally:
            # 에피소드 종료 시 아직 살아 있는 미션도 terminal 실패로 정리합니다.
            for amb_id, mission in list(active_missions.items()):
                reward = calculate_reward(
                    travel_time=mission["elapsed_time"],
                    severity=mission["severity"],
                    hospital=mission["hospital"],
                    golden_limit=GOLDEN_TIME_LIMITS[mission["severity"]],
                    rejection_count=mission["rejections"],
                    occupancy_at_selection=mission["occupancy_at_selection"],
                    route_failed=True,
                )
                agent.store_transition(
                    mission["state_vector"],
                    mission["action"],
                    reward,
                    np.zeros_like(mission["state_vector"]),
                    True,
                )
                
                agent.train_step()
                try:
                    traci.vehicle.remove(amb_id)
                except Exception:
                    pass

            active_missions.clear()
            try:
                traci.close()
            except Exception:
                pass

        if ep % 5 == 0:
            agent.update_target_network()
            agent.save_model(MODEL_PATH)

        elapsed = time.time() - ep_start
        print(
            f"Episode {ep}/{num_episodes} 완료 | "
            f"시나리오: {scenario['id']} | "
            f"소요시간: {elapsed:.1f}s | "
            f"완료미션: {completed} | "
            f"에피소드 보상: {total_episode_reward:.2f} | "
            f"메모리: {len(agent.memory)} | "
            f"Epsilon: {agent.epsilon:.4f}"
        )

    agent.update_target_network()
    agent.save_model(MODEL_PATH)
    print("\n=== 학습 완료 ===")
    print(f"모델: {MODEL_PATH}")


if __name__ == "__main__":
    train_dqn_fast(num_episodes=50)
