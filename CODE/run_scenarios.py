#!/usr/bin/env python3

import os
import sys
import time
import random
import copy
import numpy as np
import traci
import sumolib
import torch
import contextlib

from environment import SumoMedicalEnvironment
from dqn_agent import DQNAmbulanceAgent
from baselines import HospitalRouters

traci.init_log = lambda *args, **kwargs: None

MAX_STEPS = 600
MAX_AMBULANCES = 30
REJECTION_PENALTY_WEIGHT = 30.0
MODEL_PATH = "dqn_ambulance_model.pth"

GOLDEN_TIME_LIMITS = {
    4: 120,
    3: 180,
    2: 240,
    1: 300
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

def run_single_scenario(scenario, initial_hospitals, seed_val, patient_schedule, mode='dqn', agent=None):
    env = SumoMedicalEnvironment()

    random.seed(seed_val)
    np.random.seed(seed_val)

    sumo_binary = sumolib.checkBinary("sumo")
    sumo_config = [
        sumo_binary, "-c", str(env.cfg_file), 
        "--scale", str(scenario["scale"]), 
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
    except Exception as e:
        return {
            "reward": 0, "avg_reward": 0, "golden_success_rate": 0.0,
            "rejections": 0, "avg_time": 0.0, "load_std": 0.0,
            "zero_reason": f"SUMO 프로세스 실행 실패: {e}"
        }

    hospitals = copy.deepcopy(initial_hospitals)
    router = HospitalRouters(hospitals)
    
    total_reward = 0
    active_missions = {}
    completed_missions_count = 0
    golden_success_count = 0
    total_rejections = 0
    total_transfer_time = 0
    hospital_assignment_counts = {h['id']: 0 for h in hospitals}
    
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
                
                try:
                    amb_pos = traci.vehicle.getPosition(amb_id)
                    dist = np.sqrt((amb_pos[0] - mission['target_pos'][0])**2 + (amb_pos[1] - mission['target_pos'][1])**2)
                except:
                    dist = 999.0
                
                # 이송 도착 기준 완화 (120m)
                if dist <= 120.0 or mission['elapsed_time'] >= 350:
                    travel_time = mission['elapsed_time']
                    severity = mission['severity']
                    chosen_hosp = mission['hospital']
                    mission_rejections = mission.get('rejections', 0)
                    
                    completed_missions_count += 1
                    total_transfer_time += travel_time
                    
                    golden_limit = GOLDEN_TIME_LIMITS.get(severity, 240)
                    if travel_time <= golden_limit:
                        golden_success_count += 1
                    
                    mismatch_penalty = 50.0 if (severity >= 3 and '일반' in chosen_hosp['type']) else 0.0
                    rejection_penalty = mission_rejections * REJECTION_PENALTY_WEIGHT
                    
                    reward = - (travel_time * 0.1) - (chosen_hosp['occupancy'] * 20.0) - mismatch_penalty - rejection_penalty
                    total_reward += reward

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

                    for attempt in range(5):
                        if mode == 'dqn':
                            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                            with torch.no_grad():
                                q_values = agent.model(state_tensor).squeeze(0)
                            sorted_actions = torch.argsort(q_values, descending=True).cpu().numpy()
                            action = int(sorted_actions[min(attempt, len(hospitals) - 1)])
                        elif mode == 'nearest':
                            action = router.nearest_strategy(patient_pos)
                        elif mode == 'rule':
                            action = router.rule_based_strategy(severity)
                        elif mode == 'heuristic':
                            action = router.heuristic_strategy(patient_pos, severity)

                        action = min(action, len(hospitals) - 1)
                        candidate_hosp = hospitals[action]

                        if candidate_hosp['occupancy'] >= 0.85 and random.random() < 0.7:
                            total_rejections += 1
                            mission_rejection_count += 1
                            continue
                        
                        selected_hosp = candidate_hosp
                        break

                    if selected_hosp is None:
                        selected_hosp = min(hospitals, key=lambda h: h['occupancy'])
                        action = selected_hosp['id']

                    hospital_assignment_counts[selected_hosp['id']] += 1
                    selected_hosp['occupancy'] = min(1.0, selected_hosp['occupancy'] + 0.08)

                    h_edge = get_closest_edge_id(env.net, selected_hosp['sumo_x'], selected_hosp['sumo_y'])
                    if h_edge:
                        try:
                            route = traci.simulation.findRoute(patient_edge_id, h_edge, vType="ambulance_custom")
                        except:
                            route = traci.simulation.findRoute(patient_edge_id, h_edge)

                        if route and len(route.edges) > 0:
                            spawned_amb_counter += 1
                            amb_id = f"amb_eval_{step}_{spawned_amb_counter}"
                            route_id = f"r_{amb_id}"
                            
                            traci.route.add(route_id, list(route.edges))
                            traci.vehicle.add(vehID=amb_id, routeID=route_id, typeID="ambulance_custom", depart="now")

                            active_missions[amb_id] = {
                                'severity': severity,
                                'hospital': selected_hosp,
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

    success_rate = (golden_success_count / completed_missions_count * 100.0) if completed_missions_count > 0 else 0.0
    avg_time = (total_transfer_time / completed_missions_count) if completed_missions_count > 0 else 0.0
    avg_reward = (total_reward / completed_missions_count) if completed_missions_count > 0 else 0.0
    load_std = float(np.std(list(hospital_assignment_counts.values())))

    return {
        "reward": total_reward,
        "avg_reward": avg_reward,
        "golden_success_rate": success_rate,
        "rejections": total_rejections,
        "avg_time": avg_time,
        "load_std": load_std
    }

if __name__ == "__main__":
    print("=== 시나리오 기반 응급실 라우팅 평가 시스템 ===")
    
    temp_env = SumoMedicalEnvironment()
    valid_edges = temp_env.get_valid_edges()

    state_dim = 27
    action_dim = len(temp_env.get_hospitals())
    
    dqn_agent = DQNAmbulanceAgent(state_dim, action_dim)
    dqn_agent.load_model(MODEL_PATH)
    dqn_agent.epsilon = 0.0
    print("[시스템] DQN 평가 모드 전환 완료 (Epsilon = 0.0)")

    print("\n[시나리오 목록]")
    for sc in SCENARIOS:
        print(f"  [{sc['id']}] 시나리오 {sc['id']} (환자: {sc['patient_freq']} | 교통: {sc['traffic']} | 복잡도: {sc['complexity']})")
    print("  [0] 전체 시나리오 (1~9) 순차 평가 및 실행")

    selected_input = input("\n실행할 시나리오 번호를 입력하세요 (0 ~ 9): ").strip()
    
    selected_scenarios = []
    if selected_input == '0':
        selected_scenarios = SCENARIOS
    else:
        try:
            sc_id = int(selected_input)
            matched = [s for s in SCENARIOS if s['id'] == sc_id]
            selected_scenarios = matched if matched else [SCENARIOS[0]]
        except ValueError:
            selected_scenarios = [SCENARIOS[0]]

    total_scenarios = len(selected_scenarios)
    overall_reward = [0.0, 0.0, 0.0, 0.0]
    overall_golden_rate = [0.0, 0.0, 0.0, 0.0]
    overall_rejections = [0, 0, 0, 0]
    overall_avg_time = [0.0, 0.0, 0.0, 0.0]
    overall_load_std = [0.0, 0.0, 0.0, 0.0]

    session_random_offset = int(time.time() * 1000) % 100000

    for sc in selected_scenarios:
        print(f"\n[진행 중] 시나리오 {sc['id']} 평가 진행...")

        base_seed = sc['id'] * 1000 + session_random_offset
        scenario_hospitals = copy.deepcopy(temp_env.get_hospitals())
        fixed_patient_schedule = generate_fixed_patient_schedule(sc, valid_edges, base_seed)

        res_nearest = run_single_scenario(sc, scenario_hospitals, base_seed, fixed_patient_schedule, mode='nearest')
        res_rule = run_single_scenario(sc, scenario_hospitals, base_seed, fixed_patient_schedule, mode='rule')
        res_heur = run_single_scenario(sc, scenario_hospitals, base_seed, fixed_patient_schedule, mode='heuristic')
        res_dqn = run_single_scenario(sc, scenario_hospitals, base_seed, fixed_patient_schedule, mode='dqn', agent=dqn_agent)

        results = [res_nearest, res_rule, res_heur, res_dqn]

        print(f"\n[시나리오 {sc['id']} 상세 지표 요약]")
        print(f"  · 최단거리    | 보상 : {res_nearest['reward']:6.1f} | 골든타임 성공률: {res_nearest['golden_success_rate']:5.1f}% | 거부: {res_nearest['rejections']:2d} | 평균소요시간: {res_nearest['avg_time']:5.1f}s | 부하편차: {res_nearest['load_std']:.2f}")
        print(f"  · 규칙기반    | 보상 : {res_rule['reward']:6.1f} | 골든타임 성공률: {res_rule['golden_success_rate']:5.1f}% | 거부: {res_rule['rejections']:2d} | 평균소요시간: {res_rule['avg_time']:5.1f}s | 부하편차: {res_rule['load_std']:.2f}")
        print(f"  · 휴리스틱    | 보상 : {res_heur['reward']:6.1f} | 골든타임 성공률: {res_heur['golden_success_rate']:5.1f}% | 거부: {res_heur['rejections']:2d} | 평균소요시간: {res_heur['avg_time']:5.1f}s | 부하편차: {res_heur['load_std']:.2f}")
        print(f"  · DQN 에이전트| 보상 : {res_dqn['reward']:6.1f} | 골든타임 성공률: {res_dqn['golden_success_rate']:5.1f}% | 거부: {res_dqn['rejections']:2d} | 평균소요시간: {res_dqn['avg_time']:5.1f}s | 부하편차: {res_dqn['load_std']:.2f}")
        print("-" * 80)

        for i in range(4):
            overall_reward[i] += results[i]['reward']
            overall_golden_rate[i] += results[i]['golden_success_rate']
            overall_rejections[i] += results[i]['rejections']
            overall_avg_time[i] += results[i]['avg_time']
            overall_load_std[i] += results[i]['load_std']

    print("\n=== 최종 시나리오 평가 종합 결과 ===")
    print(f"  · 최단거리    | 총 보상 : {overall_reward[0]:6.1f} | 평균 골든타임 성공률: {overall_golden_rate[0]/total_scenarios:.1f}% | 총 거부: {overall_rejections[0]:2d} | 평균 소요시간: {overall_avg_time[0]/total_scenarios:.1f}s | 평균 부하편차: {overall_load_std[0]/total_scenarios:.2f}")
    print(f"  · 규칙기반    | 총 보상 : {overall_reward[1]:6.1f} | 평균 골든타임 성공률: {overall_golden_rate[1]/total_scenarios:.1f}% | 총 거부: {overall_rejections[1]:2d} | 평균 소요시간: {overall_avg_time[1]/total_scenarios:.1f}s | 평균 부하편차: {overall_load_std[1]/total_scenarios:.2f}")
    print(f"  · 휴리스틱    | 총 보상 : {overall_reward[2]:6.1f} | 평균 골든타임 성공률: {overall_golden_rate[2]/total_scenarios:.1f}% | 총 거부: {overall_rejections[2]:2d} | 평균 소요시간: {overall_avg_time[2]/total_scenarios:.1f}s | 평균 부하편차: {overall_load_std[2]/total_scenarios:.2f}")
    print(f"  · DQN 에이전트| 총 보상 : {overall_reward[3]:6.1f} | 평균 골든타임 성공률: {overall_golden_rate[3]/total_scenarios:.1f}% | 총 거부: {overall_rejections[3]:2d} | 평균 소요시간: {overall_avg_time[3]/total_scenarios:.1f}s | 평균 부하편차: {overall_load_std[3]/total_scenarios:.2f}")
    print("=" * 80)