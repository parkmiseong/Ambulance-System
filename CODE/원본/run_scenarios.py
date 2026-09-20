#!/usr/bin/env python3
## 라이브러리 ##
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
# 사용자 정의 모듈
from environment import SumoMedicalEnvironment
from dqn_agent import DQNAmbulanceAgent
from baselines import HospitalRouters

# SUMO 로그 출력 억제
traci.init_log = lambda *args, **kwargs: None

## 상수 선언 ##
MAX_STEPS = 600                                 # 시뮬레이션 최대 스텝 수
MAX_AMBULANCES = 10                             # 최대 동시 운행 가능한 구급차 수
REJECTION_PENALTY_WEIGHT = 30.0                 # 거부 횟수에 따른 보상 패널티 가중치
MODEL_PATH = "CODE/dqn_ambulance_model.pth"     # DQN 모델 파일 경로

# 골든타임 기준 시간 (단위: 초)
GOLDEN_TIME_LIMITS = {
    4: 120,
    3: 180,
    2: 240,
    1: 300
}

# 시나리오 정의
SCENARIOS = [
    {"id": 1, "patient_freq": "원활", "traffic": "원활", "complexity": 2, "prob": 0.20, "scale": 0.3},  # 낮은 복잡도, 낮은 환자 발생률, 원활한 교통
    {"id": 2, "patient_freq": "원활", "traffic": "보통", "complexity": 3, "prob": 0.20, "scale": 0.5},  # 중간 복잡도, 낮은 환자 발생률, 보통 교통
    {"id": 3, "patient_freq": "원활", "traffic": "혼잡", "complexity": 4, "prob": 0.20, "scale": 0.8},  # 높은 복잡도, 낮은 환자 발생률, 혼잡한 교통
    {"id": 4, "patient_freq": "보통", "traffic": "원활", "complexity": 3, "prob": 0.30, "scale": 0.3},  # 중간 복잡도, 보통 환자 발생률, 원활한 교통
    {"id": 5, "patient_freq": "보통", "traffic": "보통", "complexity": 4, "prob": 0.30, "scale": 0.5},  # 중간 복잡도, 보통 환자 발생률, 보통 교통
    {"id": 6, "patient_freq": "보통", "traffic": "혼잡", "complexity": 5, "prob": 0.30, "scale": 0.8},  # 높은 복잡도, 보통 환자 발생률, 혼잡한 교통
    {"id": 7, "patient_freq": "혼잡", "traffic": "원활", "complexity": 4, "prob": 0.40, "scale": 0.3},  # 높은 복잡도, 높은 환자 발생률, 원활한 교통
    {"id": 8, "patient_freq": "혼잡", "traffic": "보통", "complexity": 5, "prob": 0.40, "scale": 0.5},  # 높은 복잡도, 높은 환자 발생률, 보통 교통
    {"id": 9, "patient_freq": "혼잡", "traffic": "혼잡", "complexity": 6, "prob": 0.40, "scale": 0.8},  # 매우 높은 복잡도, 높은 환자 발생률, 혼잡한 교통
]

@contextlib.contextmanager
## SUMO 표준 출력 억제 컨텍스트 매니저 ##
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

## 실제 접근 가능한 edge ID 탐색하는 함수 ##
def get_reachable_edge_id(net, x, y, from_edge=None, vtype="ambulance_custom"):
    radius = 100
    visited_edges = set() # 중복 검사 방지용

    while radius < 5000:
        edges = net.getNeighboringEdges(x, y, radius)   # radius만큼 근처의 좌표 가져오기
        if edges:
            sorted_edges = sorted(edges, key=lambda e: e[1])    # 가까운 순서로 정렬
            for edge, dist in sorted_edges:
                edge_id = edge.getID()

                # SUMO 내부 edge 제외 : 일반 도로만 선별
                if edge_id.startswith(':') or edge_id in visited_edges:
                    continue

                visited_edges.add(edge_id)  # 확인한 엣지 추가
                
                # 출발 edge가 있으면 실제 경로 가능 여부 확인
                if from_edge is not None:
                    if from_edge == edge_id:
                        return edge_id
                    try:
                        # 실제 이동 가능한 경로 존재 여부 확인
                        route = traci.simulation.findRoute(
                            from_edge,
                            edge_id,
                            vType=vtype
                        )
                        if route and len(route.edges) > 0:
                            return edge_id
                    except traci.exceptions.TraCIException:
                        continue
                else:
                    # 출발 edge가 없을 때 차량 타입 통행 가능 여부 확인
                    if hasattr(edge, 'allows') and not edge.allows(vtype):
                        continue
                    return edge_id
        radius += 200   # radius 범위 넓히기
    return None

## 환자 발생 스케줄을 생성하는 함수 ##
def generate_fixed_patient_schedule(scenario, valid_edges, seed_val):
    random.seed(seed_val)           # 난수 기준값 설정
    np.random.seed(seed_val)
    patient_schedule = {}           # 환자 발생 일정 저장 딕셔너리
    patient_prob = scenario["prob"] # 환자 발생 확률

    # 각 step마다 환자 발생 여부 결정
    for step in range(1, MAX_STEPS + 1):
        if random.random() < patient_prob:
            patient_edge_id = random.choice(valid_edges)
            # 중증도를 무작위 설정
            severity = random.randint(1, 4)
            # 현재 step에 환자 발생 정보 저장
            patient_schedule[step] = {
                "edge_id": patient_edge_id,     # 환자 발생 도로
                "severity": severity            # 환자 중증도
            }
    return patient_schedule

## 상태 벡터를 구성하는 함수 ##
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

## 단일 시나리오를 실행하는 함수 ##
'''
1. 환경 설정
2. SUMO 실행
3. 기존 미션 처리
4. 환자 발생
5. DQN/baseline 병원 선택
6. 경로 생성
7. 미션 등록
8. 최종 평가
'''
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
                    amb_pos = traci.vehicle.getPosition(amb_id)[:2]  # (x, y) 2D 좌표 슬라이싱
                    target_pos = mission['target_pos']
                    dist = np.sqrt((amb_pos[0] - target_pos[0])**2 + (amb_pos[1] - target_pos[1])**2)
                except Exception:
                    dist = 999.0
                
                # 이송 도착 기준 (120m 또는 최대 시간 350스텝 초과 시)
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
                    
                    reward = - (travel_time * 0.1) - mismatch_penalty - rejection_penalty
                    total_reward += reward

                    try:
                        traci.vehicle.remove(amb_id)
                    except Exception:
                        pass

                    del active_missions[amb_id]

            # 2. 환자 발생 및 스폰 (Fallback 적용)
            if step in patient_schedule and len(active_missions) < MAX_AMBULANCES:
                try:
                    p_info = patient_schedule[step]
                    patient_edge_id = p_info["edge_id"]
                    severity = p_info["severity"]

                    edge_obj = env.net.getEdge(patient_edge_id)
                    if edge_obj is None:
                        continue
                        
                    patient_pos = edge_obj.getFromNode().getCoord()[:2]
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

                    h_edge = get_reachable_edge_id(env.net, selected_hosp['sumo_x'], selected_hosp['sumo_y'], from_edge=patient_edge_id)
                    if h_edge:
                        try:
                            route = traci.simulation.findRoute(patient_edge_id, h_edge, vType="ambulance_custom")
                        except Exception:
                            try:
                                route = traci.simulation.findRoute(patient_edge_id, h_edge)
                            except Exception:
                                route = None

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
        except Exception:
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
## 메인 실행부 ##
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

    # 시나리오 평가 결과 초기화
    total_scenarios = len(selected_scenarios)       # 선택된 시나리오의 총 개수
    overall_reward = [0.0, 0.0, 0.0, 0.0]           # 각 전략별 총 보상 합계
    overall_golden_rate = [0.0, 0.0, 0.0, 0.0]      # 각 전략별 골든타임 성공률 합계
    overall_rejections = [0, 0, 0, 0]               # 각 전략별 총 거부 횟수 합계
    overall_avg_time = [0.0, 0.0, 0.0, 0.0]         # 각 전략별 평균 소요시간 합계
    overall_load_std = [0.0, 0.0, 0.0, 0.0]         # 각 전략별 부하 편차 합계

    # 세션별 랜덤 오프셋 생성 (시나리오 ID와 결합하여 고유 시드 생성)
    session_random_offset = int(time.time() * 1000) % 100000

    # 선택된 시나리오 순차 평가
    for sc in selected_scenarios:
        print(f"\n[진행 중] 시나리오 {sc['id']} 평가 진행...")

        # 시나리오별 고유 시드 생성
        base_seed = sc['id'] * 1000 + session_random_offset                                     # 시나리오 ID와 세션 오프셋을 결합하여 고유 시드 생성
        scenario_hospitals = copy.deepcopy(temp_env.get_hospitals())                            # 시나리오별 병원 정보 복사
        fixed_patient_schedule = generate_fixed_patient_schedule(sc, valid_edges, base_seed)    # 고정 환자 발생 스케줄 생성

        # 각 전략별 시나리오 실행
        res_nearest = run_single_scenario(sc, scenario_hospitals, base_seed, fixed_patient_schedule, mode='nearest')
        res_rule = run_single_scenario(sc, scenario_hospitals, base_seed, fixed_patient_schedule, mode='rule')
        res_heur = run_single_scenario(sc, scenario_hospitals, base_seed, fixed_patient_schedule, mode='heuristic')
        res_dqn = run_single_scenario(sc, scenario_hospitals, base_seed, fixed_patient_schedule, mode='dqn', agent=dqn_agent)

        # 결과 요약
        results = [res_nearest, res_rule, res_heur, res_dqn]

        # 시나리오별 상세 지표 출력
        print(f"\n[시나리오 {sc['id']} 상세 지표 요약]")
        print(f"  · 최단거리    | 보상 : {res_nearest['reward']:6.1f} | 골든타임 성공률: {res_nearest['golden_success_rate']:5.1f}% | 거부: {res_nearest['rejections']:2d} | 평균소요시간: {res_nearest['avg_time']:5.1f}s | 부하편차: {res_nearest['load_std']:.2f}")
        print(f"  · 규칙기반    | 보상 : {res_rule['reward']:6.1f} | 골든타임 성공률: {res_rule['golden_success_rate']:5.1f}% | 거부: {res_rule['rejections']:2d} | 평균소요시간: {res_rule['avg_time']:5.1f}s | 부하편차: {res_rule['load_std']:.2f}")
        print(f"  · 휴리스틱    | 보상 : {res_heur['reward']:6.1f} | 골든타임 성공률: {res_heur['golden_success_rate']:5.1f}% | 거부: {res_heur['rejections']:2d} | 평균소요시간: {res_heur['avg_time']:5.1f}s | 부하편차: {res_heur['load_std']:.2f}")
        print(f"  · DQN 에이전트| 보상 : {res_dqn['reward']:6.1f} | 골든타임 성공률: {res_dqn['golden_success_rate']:5.1f}% | 거부: {res_dqn['rejections']:2d} | 평균소요시간: {res_dqn['avg_time']:5.1f}s | 부하편차: {res_dqn['load_std']:.2f}")
        print("-" * 80)

        # 결과 저장
        for i in range(4):
            overall_reward[i] += results[i]['reward']
            overall_golden_rate[i] += results[i]['golden_success_rate']
            overall_rejections[i] += results[i]['rejections']
            overall_avg_time[i] += results[i]['avg_time']
            overall_load_std[i] += results[i]['load_std']

    # 최종 출력
    print("\n=== 최종 시나리오 평가 종합 결과 ===")
    print(f"  · 최단거리    | 총 보상 : {overall_reward[0]:6.1f} | 평균 골든타임 성공률: {overall_golden_rate[0]/total_scenarios:.1f}% | 총 거부: {overall_rejections[0]:2d} | 평균 소요시간: {overall_avg_time[0]/total_scenarios:.1f}s | 평균 부하편차: {overall_load_std[0]/total_scenarios:.2f}")
    print(f"  · 규칙기반    | 총 보상 : {overall_reward[1]:6.1f} | 평균 골든타임 성공률: {overall_golden_rate[1]/total_scenarios:.1f}% | 총 거부: {overall_rejections[1]:2d} | 평균 소요시간: {overall_avg_time[1]/total_scenarios:.1f}s | 평균 부하편차: {overall_load_std[1]/total_scenarios:.2f}")
    print(f"  · 휴리스틱    | 총 보상 : {overall_reward[2]:6.1f} | 평균 골든타임 성공률: {overall_golden_rate[2]/total_scenarios:.1f}% | 총 거부: {overall_rejections[2]:2d} | 평균 소요시간: {overall_avg_time[2]/total_scenarios:.1f}s | 평균 부하편차: {overall_load_std[2]/total_scenarios:.2f}")
    print(f"  · DQN 에이전트| 총 보상 : {overall_reward[3]:6.1f} | 평균 골든타임 성공률: {overall_golden_rate[3]/total_scenarios:.1f}% | 총 거부: {overall_rejections[3]:2d} | 평균 소요시간: {overall_avg_time[3]/total_scenarios:.1f}s | 평균 부하편차: {overall_load_std[3]/total_scenarios:.2f}")
    print("=" * 80)