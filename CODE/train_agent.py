#!/usr/bin/env python3

import os
import sys
import time
import random
import copy
import numpy as np
import traci
import sumolib
import contextlib

from environment import SumoMedicalEnvironment
from dqn_agent import DQNAmbulanceAgent
from baselines import HospitalRouters

MODEL_PATH = "dqn_ambulance_model.pth"
MAX_STEPS = 500
MAX_AMBULANCES = 30
REJECTION_PENALTY_WEIGHT = 30.0

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

def get_closest_edge_id(net, x, y):
    radius = 100
    while radius < 5000:
        edges = net.getNeighboringEdges(x, y, radius)
        if edges:
            sorted_edges = sorted(edges, key=lambda e: e[1])
            for edge, dist in sorted_edges:
                edge_id = edge.getID()
                if not edge_id.startswith(':'):
                    return edge_id
        radius += 200
    return None

def generate_fixed_patient_schedule(scenario, valid_edges, seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    patient_schedule = {}
    patient_prob = scenario["prob"]
    for step in range(1, MAX_STEPS + 1):
        if random.random() < patient_prob:
            patient_edge_id = random.choice(valid_edges)
            severity = random.randint(1, 4)
            patient_schedule[step] = {"edge_id": patient_edge_id, "severity": severity}
    return patient_schedule

def build_state_vector(patient_pos, severity, hospitals):
    base_info = [
        patient_pos[0] / 10000.0,
        patient_pos[1] / 10000.0,
        severity / 4.0,
        0.0, 1.0, 1.0,
        np.mean([h['occupancy'] for h in hospitals])
    ]
    hosp_occupancies = [h['occupancy'] for h in hospitals]
    hosp_types = [1.0 if ('권역' in h['type'] or '지역' in h['type']) else 0.0 for h in hospitals]
    return np.array(base_info + hosp_occupancies + hosp_types, dtype=np.float32)

def train_dqn_fast(num_episodes=50):
    env = SumoMedicalEnvironment()
    valid_edges = env.get_valid_edges()
    
    state_dim = 27
    temp_hospitals = env.get_hospitals()
    action_dim = len(temp_hospitals)
    
    agent = DQNAmbulanceAgent(state_dim, action_dim)
    agent.epsilon = 0.5 
    
    sumo_binary = sumolib.checkBinary("sumo")
    print(f"=== DQN 강화 학습 시작 (총 {num_episodes} 에피소드) ===")
    start_total_time = time.time()

    for ep in range(1, num_episodes + 1):
        ep_start_t = time.time()
        sc = random.choice(SCENARIOS)
        seed_val = random.randint(1, 999999)
        patient_schedule = generate_fixed_patient_schedule(sc, valid_edges, seed_val)
        hospitals = copy.deepcopy(temp_hospitals)
        router = HospitalRouters(hospitals)

        sumo_config = [
            sumo_binary, "-c", str(env.cfg_file), 
            "--scale", str(sc["scale"]), 
            "--no-warnings", "true", 
            "--no-step-log", "true",
            "--duration-log.disable", "true",
            "--seed", str(seed_val), "--start"
        ]

        try:
            with suppress_sumo_stdout():
                traci.start(sumo_config)
                traci.vehicletype.copy("DEFAULT_VEHTYPE", "ambulance_custom")
                traci.vehicletype.setVehicleClass("ambulance_custom", "emergency")
                traci.vehicletype.setShapeClass("ambulance_custom", "emergency")
        except Exception:
            continue

        active_missions = {}
        spawned_amb_counter = 0

        try:
            for step in range(1, MAX_STEPS + 1):
                try:
                    traci.simulationStep()
                except traci.exceptions.FatalTraCIError:
                    break

                for h in hospitals:
                    h['occupancy'] = max(0.1, h['occupancy'] - 0.0005)

                ambulances = set(traci.vehicle.getIDList())

                # 1. 활성 미션 업데이트
                for amb_id in list(active_missions.keys()):
                    if amb_id not in ambulances:
                        del active_missions[amb_id]
                        continue

                    active_missions[amb_id]['elapsed_time'] += 1
                    mission = active_missions[amb_id]
                    
                    if step % 2 == 0 or mission['elapsed_time'] >= 200:
                        try:
                            amb_pos = traci.vehicle.getPosition(amb_id)
                            hosp_pos = mission['target_pos']
                            dist_to_hosp = np.sqrt((amb_pos[0] - hosp_pos[0])**2 + (amb_pos[1] - hosp_pos[1])**2)
                        except:
                            dist_to_hosp = 999.0
                    else:
                        dist_to_hosp = 999.0

                    # 이송 도착 기준 완화 (120m)
                    if dist_to_hosp <= 120.0 or mission['elapsed_time'] >= 350:
                        travel_time = mission['elapsed_time']
                        severity = mission['severity']
                        chosen_hosp = mission['hospital']
                        mission_rejections = mission.get('rejections', 0)
                        
                        mismatch_penalty = 50.0 if (severity >= 3 and '일반' in chosen_hosp['type']) else 0.0
                        rejection_penalty = mission_rejections * REJECTION_PENALTY_WEIGHT
                        
                        reward = - (travel_time * 0.1) - (chosen_hosp['occupancy'] * 20.0) - mismatch_penalty - rejection_penalty
                        next_state = build_state_vector(mission['target_pos'], severity, hospitals)

                        agent.store_transition(mission['state_vector'], mission['action'], reward, next_state, False)
                        for _ in range(5):
                            agent.train_step()

                        try:
                            traci.vehicle.remove(amb_id)
                        except:
                            pass
                        del active_missions[amb_id]

                # 2. 환자 발생 및 스폰 (Fallback 적용)
                if step in patient_schedule and len(active_missions) < MAX_AMBULANCES:
                    try:
                        p_info = patient_schedule[step]
                        patient_edge_id = p_info["edge_id"]
                        severity = p_info["severity"]

                        edge_obj = env.net.getEdge(patient_edge_id)
                        patient_pos = edge_obj.getFromNode().getCoord()
                        state = build_state_vector(patient_pos, severity, hospitals)
                        
                        selected_hosp = None
                        action = None
                        mission_rejection_count = 0

                        # 초반 모방 가이드 및 Fallback 탐색
                        for attempt in range(5):
                            if ep <= 15 and random.random() < 0.4:
                                action = router.rule_based_strategy(severity)
                            else:
                                action = agent.select_action(state)

                            action = min(action, len(hospitals) - 1)
                            candidate_hosp = hospitals[action]

                            if candidate_hosp['occupancy'] >= 0.85 and random.random() < 0.7:
                                mission_rejection_count += 1
                                continue
                            
                            selected_hosp = candidate_hosp
                            break

                        if selected_hosp is None:
                            selected_hosp = min(hospitals, key=lambda h: h['occupancy'])
                            action = selected_hosp['id']

                        selected_hosp['occupancy'] = min(1.0, selected_hosp['occupancy'] + 0.08)
                        h_edge = get_closest_edge_id(env.net, selected_hosp['sumo_x'], selected_hosp['sumo_y'])
                        
                        if h_edge:
                            try:
                                route = traci.simulation.findRoute(patient_edge_id, h_edge, vType="ambulance_custom")
                            except:
                                route = traci.simulation.findRoute(patient_edge_id, h_edge)

                            if route and len(route.edges) > 0:
                                spawned_amb_counter += 1
                                amb_id = f"amb_train_{step}_{spawned_amb_counter}"
                                route_id = f"r_{amb_id}"
                                traci.route.add(route_id, list(route.edges))
                                traci.vehicle.add(vehID=amb_id, routeID=route_id, typeID="ambulance_custom", depart="now")

                                active_missions[amb_id] = {
                                    'pat_pos': patient_pos,
                                    'hospital': selected_hosp,
                                    'severity': severity,
                                    'elapsed_time': 0,
                                    'rejections': mission_rejection_count,
                                    'state_vector': state,
                                    'action': action,
                                    'target_pos': (selected_hosp['sumo_x'], selected_hosp['sumo_y'])
                                }
                    except Exception:
                        continue
        finally:
            try:
                traci.close()
            except:
                pass

        ep_elapsed = time.time() - ep_start_t
        print(f"Episode {ep}/{num_episodes} 완료 | 소요시간: {ep_elapsed:.1f}s | 메모리 누적: {len(agent.memory)}개 | Epsilon: {agent.epsilon:.4f}")

        if ep % 5 == 0:
            agent.save_model(MODEL_PATH)

    agent.update_target_network()
    agent.save_model(MODEL_PATH)
    print("\n=== 학습 완료 ===")

if __name__ == "__main__":
    train_dqn_fast(num_episodes=50)