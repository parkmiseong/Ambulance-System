#run_scenarios.py
#!/usr/bin/env python3
## 라이브러리 ##
import os
import sys
import time
import random
import copy
import numpy as np
import traci
from pathlib import Path
import sumolib
import contextlib

from environment import SumoMedicalEnvironment
from dqn_agent import DQNAmbulanceAgent
from baselines import HospitalRouters

## SUMO 로그 출력을 억제하기 위해 traci.init_log를 무시하도록 설정 ##
traci.init_log = lambda *args, **kwargs: None

## 시나리오 설정 ##
MAX_STEPS = 400            # 최대 시뮬레이션 스텝 수
GOLDEN_TIME_LIMIT = 240     # 골든타임 기준 (초 단위)

## 시나리오 정의 ##
'''
    id : 시나리오 ID
    patient_freq : 환자 발생 빈도 (원활, 보통, 혼잡)
    traffic : 교통 상황 (원활, 보통, 혼잡)
    complexity : 시나리오 복잡도 (1~6)
    prob : 환자 발생 확률 (0~1)
    scale : SUMO 시뮬레이션 스케일 (0~1)
'''
SCENARIOS = [
    {"id": 1, "patient_freq": "원활", "traffic": "원활", "complexity": 2, "prob": 0.05, "scale": 0.3},
    {"id": 2, "patient_freq": "원활", "traffic": "보통", "complexity": 3, "prob": 0.05, "scale": 0.5},
    {"id": 3, "patient_freq": "원활", "traffic": "혼잡", "complexity": 4, "prob": 0.05, "scale": 0.8},
    {"id": 4, "patient_freq": "보통", "traffic": "원활", "complexity": 3, "prob": 0.10, "scale": 0.3},
    {"id": 5, "patient_freq": "보통", "traffic": "보통", "complexity": 4, "prob": 0.10, "scale": 0.5},
    {"id": 6, "patient_freq": "보통", "traffic": "혼잡", "complexity": 5, "prob": 0.10, "scale": 0.8},
    {"id": 7, "patient_freq": "혼잡", "traffic": "원활", "complexity": 4, "prob": 0.15, "scale": 0.3},
    {"id": 8, "patient_freq": "혼잡", "traffic": "보통", "complexity": 5, "prob": 0.15, "scale": 0.5},
    {"id": 9, "patient_freq": "혼잡", "traffic": "혼잡", "complexity": 6, "prob": 0.15, "scale": 0.8},
]

# SUMO 출력 억제 컨텍스트 매니저 #
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


## 병원과 환자 위치를 기반으로 가장 가까운 엣지 ID를 반환하는 함수 ##
def get_closest_edge_id(net, x, y):
    radius = 50             # 초기 탐색 반경 (미터 단위)

    # 엣지 탐색을 위한 반경을 점진적으로 증가시키며 가장 가까운 엣지를 찾음
    while radius < 3000:
        # 주변 엣지 검색
        edges = net.getNeighboringEdges(x, y, radius)
        # 엣지가 존재하면 거리 기준으로 정렬 후 가장 가까운 엣지 ID 반환
        if edges:
            sorted_edges = sorted(edges, key=lambda e: e[1])
            for edge, dist in sorted_edges:
                edge_id = edge.getID()
                if not edge_id.startswith(':'):
                    return edge_id
        radius += 100   # 탐색 반경 증가
    return None

## 환자 위치를 기반으로 접근 가능한 엣지 ID를 반환하는 함수 ##
def get_reachable_patient_edge(net, amb_x, amb_y, valid_edges):
    radius = 200            # 초기 탐색 반경 (미터 단위)
    # 주변 엣지 탐색을 위한 반경을 점진적으로 증가시키며 접근 가능한 엣지를 찾음
    while radius < 4000:
        edges = net.getNeighboringEdges(amb_x, amb_y, radius)
        # 접근 가능한 엣지 중 유효한 엣지 필터링
        if edges:
            candidates = [e[0].getID() for e in edges if not e[0].getID().startswith(':')]
            if candidates:
                return random.choice(candidates)    # 접근 가능한 엣지 중 무작위 선택
        radius += 200   # 탐색 반경 증가
    return random.choice(valid_edges)

## 단일 시나리오 실행 함수 ##
def run_single_scenario(scenario, initial_hospitals, seed_val, mode='dqn', agent=None):
    env = SumoMedicalEnvironment()

    # 시드 값 설정 (재현성을 위해)
    random.seed(seed_val)
    np.random.seed(seed_val)

    # SUMO 실행 파일 경로 확인 및 시뮬레이션 구성 설정
    sumo_binary = sumolib.checkBinary("sumo")
    # 불필요한 에러 및 경고 출력을 막기 위한 플래그 추가
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
    except Exception as e:
        return {
            "reward": 0, "avg_reward": 0, "golden_success_rate": 0.0,
            "rejections": 0, "avg_time": 0.0, "load_std": 0.0,
            "zero_reason": f"SUMO 프로세스 실행 실패: {e}"
        }
    
    hospitals = copy.deepcopy(initial_hospitals)
    router = HospitalRouters(hospitals)
    valid_edges = env.get_valid_edges()
    
    total_reward = 0
    active_missions = {}
    completed_missions_count = 0
    golden_success_count = 0
    total_rejections = 0
    total_transfer_time = 0
    hospital_assignment_counts = {h['id']: 0 for h in hospitals}
    
    patient_spawn_attempts = 0
    patient_spawn_success = 0
    ambulances_detected_count = 0

    patient_prob = scenario["prob"]

    try:
        for step in range(1, MAX_STEPS + 1):
            try:
                traci.simulationStep()
            except traci.exceptions.FatalTraCIError:
                break

            for h in hospitals:
                h['occupancy'] = max(0.1, h['occupancy'] - 0.0005)

            ambulances = [v for v in traci.vehicle.getIDList() if 'amb' in v.lower() or 'ambulance' in v.lower()]
            if ambulances:
                ambulances_detected_count += 1

            for amb_id in list(active_missions.keys()):
                if amb_id not in ambulances:
                    del active_missions[amb_id]
                    continue

                active_missions[amb_id]['elapsed_time'] += 1
                amb_pos = traci.vehicle.getPosition(amb_id)
                mission = active_missions[amb_id]
                
                if mission['state_phase'] == 'to_patient':
                    dist_to_pat = np.sqrt((amb_pos[0] - mission['pat_pos'][0])**2 + (amb_pos[1] - mission['pat_pos'][1])**2)
                    if dist_to_pat <= 60.0 or mission['elapsed_time'] > 150:
                        mission['state_phase'] = 'to_hospital'
                        mission['elapsed_time'] = 0
                        
                        chosen_hosp = mission['hospital']
                        print(f"[이송 픽업] Step {step} | 구급차({amb_id})가 환자 위치 도착! -> 병원({chosen_hosp['name']})으로 이송 시작")

                        h_edge = get_closest_edge_id(env.net, chosen_hosp['sumo_x'], chosen_hosp['sumo_y'])
                        curr_edge = traci.vehicle.getRoadID(amb_id)
                        if curr_edge.startswith(':'):
                            curr_edge = traci.vehicle.getLaneID(amb_id).rsplit('_', 1)[0]
                        
                        if h_edge and curr_edge:
                            try:
                                route = traci.simulation.findRoute(curr_edge, h_edge, vType="ambulance")
                                if route and len(route.edges) > 0:
                                    traci.vehicle.setRoute(amb_id, list(route.edges))
                                else:
                                    traci.vehicle.changeTarget(amb_id, h_edge)
                            except:
                                pass
                
                elif mission['state_phase'] == 'to_hospital':
                    hosp_pos = mission['target_pos']
                    dist_to_hosp = np.sqrt((amb_pos[0] - hosp_pos[0])**2 + (amb_pos[1] - hosp_pos[1])**2)
                    
                    if dist_to_hosp <= 80.0 or mission['elapsed_time'] >= 350:
                        travel_time = mission['elapsed_time']
                        severity = mission['severity']
                        chosen_hosp = mission['hospital']
                        
                        completed_missions_count += 1
                        total_transfer_time += travel_time
                        is_golden = travel_time <= GOLDEN_TIME_LIMIT
                        if is_golden:
                            golden_success_count += 1
                        
                        mismatch_penalty = 50.0 if (severity >= 3 and '일반' in chosen_hosp['type']) else 0.0
                        reward = - (travel_time * 0.1) - (chosen_hosp['occupancy'] * 20.0) - mismatch_penalty
                        total_reward += reward

                        print(f"[이송 완료] Step {step} | 구급차({amb_id}) -> 병원({chosen_hosp['name']}) 도착 | 소요시간: {travel_time}s (골든타임: {'성공' if is_golden else '실패'}), 보상: {reward:.1f}")

                        if mode == 'dqn' and agent:
                            next_state = np.array([
                                hosp_pos[0] / 10000.0,
                                hosp_pos[1] / 10000.0,
                                severity / 4.0,
                                travel_time / 100.0,
                                1.0, 1.0,
                                np.mean([h['occupancy'] for h in hospitals])
                            ], dtype=np.float32)
                            
                            agent.store_transition(
                                mission['state_vector'],
                                mission['action'],
                                reward,
                                next_state,
                                False
                            )
                            agent.train_step()

                        del active_missions[amb_id]

            if random.random() < patient_prob:
                patient_spawn_attempts += 1
                if ambulances:
                    try:
                        idle_ambs = [a for a in ambulances if a not in active_missions]
                        if not idle_ambs:
                            continue

                        best_amb = random.choice(idle_ambs)
                        amb_pos = traci.vehicle.getPosition(best_amb)

                        patient_edge_id = get_reachable_patient_edge(env.net, amb_pos[0], amb_pos[1], valid_edges)
                        edge_obj = env.net.getEdge(patient_edge_id)
                        patient_pos = edge_obj.getFromNode().getCoord()
                        
                        curr_edge = traci.vehicle.getRoadID(best_amb)
                        if curr_edge.startswith(':'):
                            curr_edge = traci.vehicle.getLaneID(best_amb).rsplit('_', 1)[0]

                        mean_speed = traci.edge.getLastStepMeanSpeed(curr_edge) if curr_edge else 10.0
                        travel_time_val = traci.edge.getAdaptedTraveltime(curr_edge, 0) if curr_edge else 1.0
                        severity = random.randint(1, 4)
                        mean_occupancy = np.mean([h['occupancy'] for h in hospitals])

                        state = np.array([
                            patient_pos[0] / 10000.0,
                            patient_pos[1] / 10000.0,
                            severity / 4.0,
                            0.0,
                            mean_speed / 20.0,
                            travel_time_val / 100.0,
                            mean_occupancy
                        ], dtype=np.float32)

                        for _ in range(3):
                            if mode == 'dqn':
                                action = agent.select_action(state)
                            elif mode == 'nearest':
                                action = router.nearest_strategy(patient_pos)
                            elif mode == 'rule':
                                action = router.rule_based_strategy(severity)
                            elif mode == 'heuristic':
                                action = router.heuristic_strategy(patient_pos, severity)

                            action = min(action, len(hospitals) - 1)
                            selected_hosp = hospitals[action]

                            if selected_hosp['occupancy'] >= 0.85 and random.random() < 0.7:
                                total_rejections += 1
                                continue
                            break

                        hospital_assignment_counts[selected_hosp['id']] += 1
                        selected_hosp['occupancy'] = min(1.0, selected_hosp['occupancy'] + 0.08)

                        print(f"\n[환자 발생] Step {step} | 중증도: Lv.{severity} | 배정 병원: {selected_hosp['name']}")
                        print(f"  └ 구급차({best_amb}) 위치: (X: {amb_pos[0]:.1f}, Y: {amb_pos[1]:.1f}) -> 환자 위치: (X: {patient_pos[0]:.1f}, Y: {patient_pos[1]:.1f}) 출동 시작")

                        try:
                            route = traci.simulation.findRoute(curr_edge, patient_edge_id, vType="ambulance")
                            if route and len(route.edges) > 0:
                                traci.vehicle.setRoute(best_amb, list(route.edges))
                            else:
                                traci.vehicle.changeTarget(best_amb, patient_edge_id)
                        except:
                            continue

                        patient_spawn_success += 1
                        active_missions[best_amb] = {
                            'state_phase': 'to_patient',
                            'pat_pos': patient_pos,
                            'hospital': selected_hosp,
                            'severity': severity,
                            'elapsed_time': 0,
                            'state_vector': state,
                            'action': action,
                            'target_pos': (selected_hosp['sumo_x'], selected_hosp['sumo_y'])
                        }
                    except:
                        continue
    finally:
        try:
            traci.close()
        except:
            pass

    zero_reason = ""
    if completed_missions_count == 0:
        reasons = []
        if ambulances_detected_count == 0:
            reasons.append("시뮬레이션 내에 구급차가 전혀 스폰되지 않음")
        if patient_spawn_attempts == 0:
            reasons.append("환자 발생 확률 조건이 충족되지 않음")
        elif patient_spawn_success == 0:
            reasons.append("환자는 발생했으나 경로 탐색(findRoute) 또는 유휴 구급차 부재로 배정 실패")
        else:
            reasons.append("환자 배정은 되었으나 시뮬레이션 종료(MAX_STEPS) 전 병원 이송 미완료")
        zero_reason = " | ".join(reasons)

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
        "load_std": load_std,
        "zero_reason": zero_reason
    }

if __name__ == "__main__":
    print("=== 9가지 시나리오별 4대 지표 통합 평가 시작 (에러 로그 완벽 차단) ===")
    
    temp_env = SumoMedicalEnvironment()
    valid_edges = temp_env.get_valid_edges()
    print(f"네트워크 로딩 완료. (총 유효 엣지 수: {len(valid_edges)}개)")

    state_dim = 7
    action_dim = len(temp_env.get_hospitals())
    dqn_agent = DQNAmbulanceAgent(state_dim, action_dim)

    for sc in SCENARIOS:
        print(f"\n[진행 중] 시나리오 {sc['id']} / 9 (복잡도: {sc['complexity']} | 환자: {sc['patient_freq']} | 교통: {sc['traffic']}) 실행 중...")

        # 시나리오 지표 총합 저장용 리스트 초기화 #
        reward = [0, 0, 0, 0]
        golden_success_rate = [0.0, 0.0, 0.0, 0.0]
        rejections = [0, 0, 0, 0]
        avg_time = [0.0, 0.0, 0.0, 0.0]
        load_std = [0.0, 0.0, 0.0, 0.0]


        base_seed = sc['id'] * 100
        scenario_hospitals = copy.deepcopy(temp_env.get_hospitals())

        sys.stdout.write("\n  -> 최단 거리 알고리즘 평가 중... ")
        sys.stdout.flush()
        start_t = time.time()
        res_nearest = run_single_scenario(sc, scenario_hospitals, base_seed, mode='nearest')
        elapsed_n = time.time() - start_t
        print(f"완료(reward : {res_nearest['reward']:.1f}, time : {elapsed_n:.1f}s)")
        if res_nearest['reward'] == 0:
            print(f"    [원인 분석] {res_nearest['zero_reason']}")

        sys.stdout.write("\n  -> 규칙 기반 알고리즘 평가 중... ")
        sys.stdout.flush()
        start_t = time.time()
        res_rule = run_single_scenario(sc, scenario_hospitals, base_seed, mode='rule')
        elapsed_r = time.time() - start_t
        print(f"완료(reward : {res_rule['reward']:.1f}, time : {elapsed_r:.1f}s)")
        if res_rule['reward'] == 0:
            print(f"    [원인 분석] {res_rule['zero_reason']}")

        sys.stdout.write("\n  -> 휴리스틱 알고리즘 평가 중... ")
        sys.stdout.flush()
        start_t = time.time()
        res_heur = run_single_scenario(sc, scenario_hospitals, base_seed, mode='heuristic')
        elapsed_h = time.time() - start_t
        print(f"완료(reward : {res_heur['reward']:.1f}, time : {elapsed_h:.1f}s)")
        if res_heur['reward'] == 0:
            print(f"    [원인 분석] {res_heur['zero_reason']}")

        sys.stdout.write("\n  -> DQN 에이전트 학습 및 평가 중... ")
        sys.stdout.flush()
        start_t = time.time()
        res_dqn = run_single_scenario(sc, scenario_hospitals, base_seed, mode='dqn', agent=dqn_agent)
        elapsed_d = time.time() - start_t
        print(f"완료(reward : {res_dqn['reward']:.1f}, time : {elapsed_d:.1f}s)")
        if res_dqn['reward'] == 0:
            print(f"    [원인 분석] {res_dqn['zero_reason']}")

        print(f"\n[시나리오 {sc['id']} 상세 지표 요약]")
        print(f"  · 최단거리    | 보상 : {res_nearest['reward']:6.1f} | 평균보상: {res_nearest['avg_reward']:6.1f} | 골든타임 성공률: {res_nearest['golden_success_rate']:5.1f}% | 거부: {res_nearest['rejections']:2d} | 평균소요시간: {res_nearest['avg_time']:5.1f}s | 부하편차: {res_nearest['load_std']:.2f}")
        print(f"  · 규칙기반    | 보상 : {res_rule['reward']:6.1f} | 평균보상: {res_rule['avg_reward']:6.1f} | 골든타임 성공률: {res_rule['golden_success_rate']:5.1f}% | 거부: {res_rule['rejections']:2d} | 평균소요시간: {res_rule['avg_time']:5.1f}s | 부하편차: {res_rule['load_std']:.2f}")
        print(f"  · 휴리스틱    | 보상 : {res_heur['reward']:6.1f} | 평균보상: {res_heur['avg_reward']:6.1f} | 골든타임 성공률: {res_heur['golden_success_rate']:5.1f}% | 거부: {res_heur['rejections']:2d} | 평균소요시간: {res_heur['avg_time']:5.1f}s | 부하편차: {res_heur['load_std']:.2f}")
        print(f"  · DQN 에이전트| 보상 : {res_dqn['reward']:6.1f} | 평균보상: {res_dqn['avg_reward']:6.1f} | 골든타임 성공률: {res_dqn['golden_success_rate']:5.1f}% | 거부: {res_dqn['rejections']:2d} | 평균소요시간: {res_dqn['avg_time']:5.1f}s | 부하편차: {res_dqn['load_std']:.2f}")
        print("-" * 80)

        # 전체 시나리오 평가용 데이터 수집 #
        for i in range(4):
            reward[i] += [res_nearest, res_rule, res_heur, res_dqn][i]['reward']
            golden_success_rate[i] += [res_nearest, res_rule, res_heur, res_dqn][i]['golden_success_rate']
            rejections[i] += [res_nearest, res_rule, res_heur, res_dqn][i]['rejections']
            avg_time[i] += [res_nearest, res_rule, res_heur, res_dqn][i]['avg_time']
            load_std[i] += [res_nearest, res_rule, res_heur, res_dqn][i]['load_std']
        

    # 전체 시나리오 평가 상세 지표 #
    print("\n=== 전체 시나리오 평가 상세 지표 ===")
    print(f"  · 최단거리    | 총 보상 : {reward[0]:6.1f} | 평균 골든타임 성공률: {golden_success_rate[0]/9:.1f}% | 총 거부: {rejections[0]:2d} | 평균 소요시간: {avg_time[0]/9:.1f}s | 평균 부하편차: {load_std[0]/9:.2f}")
    print(f"  · 규칙기반    | 총 보상 : {reward[1]:6.1f} | 평균 골든타임 성공률: {golden_success_rate[1]/9:.1f}% | 총 거부: {rejections[1]:2d} | 평균 소요시간: {avg_time[1]/9:.1f}s | 평균 부하편차: {load_std[1]/9:.2f}")
    print(f"  · 휴리스틱    | 총 보상 : {reward[2]:6.1f} | 평균 골든타임 성공률: {golden_success_rate[2]/9:.1f}% | 총 거부: {rejections[2]:2d} | 평균 소요시간: {avg_time[2]/9:.1f}s | 평균 부하편차: {load_std[2]/9:.2f}")
    print(f"  · DQN 에이전트| 총 보상 : {reward[3]:6.1f} | 평균 골든타임 성공률: {golden_success_rate[3]/9:.1f}% | 총 거부: {rejections[3]:2d} | 평균 소요시간: {avg_time[3]/9:.1f}s | 평균 부하편차: {load_std[3]/9:.2f}")
    print(f"  · 평가 완료 시점 | 총 시나리오 수: 9 | 총 평가 시간: {sum([elapsed_n, elapsed_r, elapsed_h, elapsed_d]):.1f}s")
    print("=" * 80)


    print("\n모든 시나리오 평가가 성공적으로 완료되었습니다.")