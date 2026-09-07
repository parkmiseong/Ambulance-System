#!/usr/bin/env python3
import random
import numpy as np
import traci
from environment import SumoEnvironment
from dqn_agent import DQNAmbulanceAgent
from baselines import BaselineRouters

MAX_STEPS = 500
EPISODES = 3

def run_simulation_with_strategy(mode='dqn', agent=None):
    env = SumoEnvironment()
    env.start_simulation(gui=False)
    
    net = env.get_net()
    valid_edges = env.get_valid_edges()
    baseline = BaselineRouters()
    
    total_episode_reward = 0
    patients = []
    active_missions = {}

    try:
        for step in range(1, MAX_STEPS + 1):
            traci.simulationStep()

            # 1. 환자 랜덤 발생 (최대 5명)
            if random.random() < 0.04 and len(patients) < 5:
                p_edge = random.choice(valid_edges)
                p_coord = net.getEdge(p_edge).getFromNode().getCoord()
                urgency = random.uniform(0.1, 1.0)
                patients.append({'edge': p_edge, 'pos': p_coord, 'urgency': urgency})

            # 2. 기존 구급차 목록 확인
            all_vehicles = traci.vehicle.getIDList()
            ambulances = [v for v in all_vehicles if 'amb' in v.lower() or 'ambulance' in v.lower()]

            # 3. 알고리즘별 배차 로직 실행
            if ambulances and patients:
                for amb_id in ambulances:
                    if amb_id in active_missions:
                        continue

                    try:
                        amb_pos = traci.vehicle.getPosition(amb_id)
                        patient_coords = [p['pos'] for p in patients]
                        urgencies = [p['urgency'] for p in patients]

                        dists = [np.sqrt((amb_pos[0] - pc[0])**2 + (amb_pos[1] - pc[1])**2) for pc in patient_coords]
                        min_dist_val = min(dists)
                        mean_dist_val = np.mean(dists)
                        num_patients = len(patients)
                        avg_urgency = np.mean(urgencies)

                        # 모드별 액션(환자 선택) 결정
                        if mode == 'dqn':
                            state = np.array([
                                min_dist_val / 1000.0,
                                mean_dist_val / 1000.0,
                                num_patients / 5.0,
                                avg_urgency
                            ], dtype=np.float32)
                            action = agent.select_action(state, min(len(patients), agent.action_dim))
                        elif mode == 'nearest':
                            action = baseline.nearest_strategy(amb_pos, patients)
                        elif mode == 'rule':
                            action = baseline.rule_based_strategy(amb_pos, patients, urgencies)
                        elif mode == 'heuristic':
                            action = baseline.heuristic_strategy(amb_pos, patients, urgencies)

                        # 인덱스 범위 예외 처리
                        action = min(action, len(patients) - 1)
                        target_patient = patients[action]

                        # [추가된 검증 로직] 구급차 위치 파악 및 경로 유효성 확인
                        curr_edge = traci.vehicle.getRoadID(amb_id)
                        if curr_edge.startswith(":"):
                            curr_edge = traci.vehicle.getLaneID(amb_id).rsplit("_", 1)[0]
                        
                        target_edge = target_patient['edge']

                        route = traci.simulation.findRoute(curr_edge, target_edge)
                        if route and len(route.edges) > 0:
                            target_patient = patients.pop(action)
                            active_missions[amb_id] = target_edge
                            traci.vehicle.changeTarget(amb_id, target_edge)
                        else:
                            continue

                    except Exception as e:
                        continue

            # 4. 현장 도착 여부 체크 및 보상 계산
            for amb_id, p_edge in list(active_missions.items()):
                if amb_id in traci.vehicle.getIDList():
                    curr_edge = traci.vehicle.getRoadID(amb_id)
                    if curr_edge.startswith(":"):
                        curr_edge = traci.vehicle.getLaneID(amb_id).rsplit("_", 1)[0]

                    if curr_edge == p_edge:
                        reward = 100.0
                        total_episode_reward += reward
                        
                        if mode == 'dqn' and agent:
                            # DQN 학습용 피드백 저장
                            next_state = np.array([0.0, 0.0, len(patients)/5.0, 0.5], dtype=np.float32)
                            agent.store_transition(state, action, reward, next_state, False)
                            agent.train_step()
                            
                        del active_missions[amb_id]
                else:
                    del active_missions[amb_id]

    finally:
        env.close_simulation()

    return total_episode_reward

if __name__ == "__main__":
    print("=== 응급차 배차 알고리즘 비교 실험 시작 ===")
    
    state_dim = 4
    action_dim = 5
    dqn_agent = DQNAmbulanceAgent(state_dim, action_dim)

    # 1. 최단 거리 알고리즘 평가
    print("\n[1] 최단 거리(Nearest) 알고리즘 실행 중...")
    nearest_rewards = [run_simulation_with_strategy(mode='nearest') for _ in range(EPISODES)]
    
    # 2. 규칙 기반 알고리즘 평가
    print("\n[2] 규칙 기반(Rule-based) 알고리즘 실행 중...")
    rule_rewards = [run_simulation_with_strategy(mode='rule') for _ in range(EPISODES)]
    
    # 3. 휴리스틱 알고리즘 평가
    print("\n[3] 휴리스틱(Heuristic) 알고리즘 실행 중...")
    heuristic_rewards = [run_simulation_with_strategy(mode='heuristic') for _ in range(EPISODES)]
    
    # 4. DQN 에이전트 학습 및 평가
    print("\n[4] DQN 에이전트 학습 및 실행 중...")
    dqn_rewards = [run_simulation_with_strategy(mode='dqn', agent=dqn_agent) for _ in range(EPISODES)]

    # 최종 결과 요약 출력
    print("\n========================================")
    print("           알고리즘별 성능 비교 결과        ")
    print("========================================")
    print(f"최단 거리 (Nearest) 평균 보상:     {np.mean(nearest_rewards):.2f}")
    print(f"규칙 기반 (Rule-based) 평균 보상:   {np.mean(rule_rewards):.2f}")
    print(f"휴리스틱 (Heuristic) 평균 보상:   {np.mean(heuristic_rewards):.2f}")
    print(f"DQN 에이전트 평균 보상:           {np.mean(dqn_rewards):.2f}")
    print("========================================")