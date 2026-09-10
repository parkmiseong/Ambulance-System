#!/usr/init/env python3
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

MAX_STEPS = 400
GOLDEN_TIME_LIMIT = 180

SCENARIOS = [
    {"id": 1, "patient_freq": "원활", "traffic": "원활", "complexity": 2, "prob": 0.02, "scale": 0.3},
    {"id": 2, "patient_freq": "원활", "traffic": "보통", "complexity": 3, "prob": 0.02, "scale": 0.5},
    {"id": 3, "patient_freq": "원활", "traffic": "혼잡", "complexity": 4, "prob": 0.02, "scale": 0.8},
    {"id": 4, "patient_freq": "보통", "traffic": "원활", "complexity": 3, "prob": 0.05, "scale": 0.3},
    {"id": 5, "patient_freq": "보통", "traffic": "보통", "complexity": 4, "prob": 0.05, "scale": 0.5},
    {"id": 6, "patient_freq": "보통", "traffic": "혼잡", "complexity": 5, "prob": 0.05, "scale": 0.8},
    {"id": 7, "patient_freq": "혼잡", "traffic": "원활", "complexity": 4, "prob": 0.10, "scale": 0.3},
    {"id": 8, "patient_freq": "혼잡", "traffic": "보통", "complexity": 5, "prob": 0.10, "scale": 0.5},
    {"id": 9, "patient_freq": "혼잡", "traffic": "혼잡", "complexity": 6, "prob": 0.10, "scale": 0.8},
]

@contextlib.contextmanager
def suppress_sumo_stdout():
    original_stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), original_stdout_fd)
        try:
            yield
        finally:
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.close(saved_stdout_fd)

def get_closest_edge_id(net, x, y):
    radius = 50
    while radius < 2000:
        edges = net.getNeighboringEdges(x, y, radius)
        if edges:
            sorted_edges = sorted(edges, key=lambda e: e[1])
            for edge, dist in sorted_edges:
                edge_id = edge.getID()
                if not edge_id.startswith(':'):
                    return edge_id
        radius += 100
    return None

def run_single_scenario(scenario, initial_hospitals, seed_val, mode='dqn', agent=None):
    env = SumoMedicalEnvironment()
    
    random.seed(seed_val)
    np.random.seed(seed_val)

    sumo_binary = sumolib.checkBinary("sumo-gui")
    sumo_config = [sumo_binary, "-c", str(env.cfg_file), "--scale", str(scenario["scale"]), "--no-warnings", "--seed", str(seed_val), "--start"]
    
    try:
        with suppress_sumo_stdout():
            traci.start(sumo_config)
    except Exception as e:
        print(f"\n[에러] SUMO 실행 실패 (시나리오 {scenario['id']}): {e}")
        return {
            "reward": 0, "avg_reward": 0, "golden_success_rate": 0.0,
            "rejections": 0, "avg_time": 0.0, "load_std": 0.0
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

            for amb_id in list(active_missions.keys()):
                if amb_id not in ambulances:
                    poi_id = active_missions[amb_id].get('poi_id')
                    if poi_id:
                        try: traci.poi.remove(poi_id)
                        except: pass
                    del active_missions[amb_id]
                    continue

                active_missions[amb_id]['elapsed_time'] += 1
                amb_pos = traci.vehicle.getPosition(amb_id)
                mission = active_missions[amb_id]
                
                if mission['state_phase'] == 'to_patient':
                    dist_to_pat = np.sqrt((amb_pos[0] - mission['pat_pos'][0])**2 + (amb_pos[1] - mission['pat_pos'][1])**2)
                    if dist_to_pat <= 40.0:
                        mission['state_phase'] = 'to_hospital'
                        h_edge = get_closest_edge_id(env.net, mission['hospital']['sumo_x'], mission['hospital']['sumo_y'])
                        curr_edge = traci.vehicle.getRoadID(amb_id)
                        if curr_edge.startswith(':'):
                            curr_edge = traci.vehicle.getLaneID(amb_id).rsplit('_', 1)[0]
                        
                        if h_edge and curr_edge:
                            try:
                                route = traci.simulation.findRoute(curr_edge, h_edge)
                                if route and len(route.edges) > 0:
                                    traci.vehicle.changeTarget(amb_id, h_edge)
                            except:
                                pass
                
                elif mission['state_phase'] == 'to_hospital':
                    hosp_pos = mission['target_pos']
                    dist_to_hosp = np.sqrt((amb_pos[0] - hosp_pos[0])**2 + (amb_pos[1] - hosp_pos[1])**2)
                    
                    if dist_to_hosp <= 60.0 or mission['elapsed_time'] >= 300:
                        travel_time = mission['elapsed_time']
                        severity = mission['severity']
                        chosen_hosp = mission['hospital']
                        
                        completed_missions_count += 1
                        total_transfer_time += travel_time
                        if travel_time <= GOLDEN_TIME_LIMIT:
                            golden_success_count += 1
                        
                        mismatch_penalty = 50.0 if (severity >= 3 and '일반' in chosen_hosp['type']) else 0.0
                        reward = - (travel_time * 0.1) - (chosen_hosp['occupancy'] * 20.0) - mismatch_penalty
                        total_reward += reward

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

                        poi_id = mission.get('poi_id')
                        if poi_id:
                            try: traci.poi.remove(poi_id)
                            except: pass

                        del active_missions[amb_id]

            if random.random() < patient_prob and ambulances:
                try:
                    idle_ambs = [a for a in ambulances if a not in active_missions]
                    if not idle_ambs:
                        continue

                    patient_edge_id = random.choice(valid_edges)
                    edge_obj = env.net.getEdge(patient_edge_id)
                    patient_pos = edge_obj.getFromNode().getCoord()

                    best_amb = min(idle_ambs, key=lambda a: np.sum((np.array(traci.vehicle.getPosition(a)) - np.array(patient_pos))**2))
                    
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

                    try:
                        route = traci.simulation.findRoute(curr_edge, patient_edge_id)
                        if route and len(route.edges) > 0:
                            traci.vehicle.changeTarget(best_amb, patient_edge_id)
                        else:
                            continue
                    except:
                        continue

                    poi_id = f"patient_{step}_{best_amb}"
                    try:
                        traci.poi.add(
                            poiID=poi_id,
                            x=patient_pos[0],
                            y=patient_pos[1],
                            color=(255, 0, 0, 255),
                            layer=100,
                            type="patient_marker"
                        )
                    except:
                        pass

                    try:
                        view_ids = traci.gui.getIDList()
                        if view_ids:
                            traci.gui.trackVehicle(view_ids[0], best_amb)
                            traci.gui.setZoom(view_ids[0], 1500)
                    except:
                        pass

                    active_missions[best_amb] = {
                        'state_phase': 'to_patient',
                        'pat_pos': patient_pos,
                        'hospital': selected_hosp,
                        'severity': severity,
                        'elapsed_time': 0,
                        'state_vector': state,
                        'action': action,
                        'target_pos': (selected_hosp['sumo_x'], selected_hosp['sumo_y']),
                        'poi_id': poi_id
                    }
                except:
                    continue
    finally:
        try:
            view_ids = traci.gui.getIDList()
            if view_ids:
                traci.gui.close(view_ids[0])
        except:
            pass
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
    print("=== 9가지 시나리오별 4대 지표 통합 평가 시작 ===")
    
    temp_env = SumoMedicalEnvironment()
    valid_edges = temp_env.get_valid_edges()
    print(f"네트워크 로딩 완료. (총 유효 엣지 수: {len(valid_edges)}개)")

    state_dim = 7
    action_dim = len(temp_env.get_hospitals())
    dqn_agent = DQNAmbulanceAgent(state_dim, action_dim)

    for sc in SCENARIOS:
        print(f"\n[진행 중] 시나리오 {sc['id']} / 9 (복잡도: {sc['complexity']} | 환자: {sc['patient_freq']} | 교통: {sc['traffic']}) 실행 중...")
        
        base_seed = sc['id'] * 100
        scenario_hospitals = copy.deepcopy(temp_env.get_hospitals())

        sys.stdout.write("  -> 최단 거리 알고리즘 평가 중... ")
        sys.stdout.flush()
        start_t = time.time()
        res_nearest = run_single_scenario(sc, scenario_hospitals, base_seed, mode='nearest')
        elapsed_n = time.time() - start_t
        print(f"완료(reward : {res_nearest['reward']:.1f}, time : {elapsed_n:.1f}s)")

        sys.stdout.write("  -> 규칙 기반 알고리즘 평가 중... ")
        sys.stdout.flush()
        start_t = time.time()
        res_rule = run_single_scenario(sc, scenario_hospitals, base_seed, mode='rule')
        elapsed_r = time.time() - start_t
        print(f"완료(reward : {res_rule['reward']:.1f}, time : {elapsed_r:.1f}s)")

        sys.stdout.write("  -> 휴리스틱 알고리즘 평가 중... ")
        sys.stdout.flush()
        start_t = time.time()
        res_heur = run_single_scenario(sc, scenario_hospitals, base_seed, mode='heuristic')
        elapsed_h = time.time() - start_t
        print(f"완료(reward : {res_heur['reward']:.1f}, time : {elapsed_h:.1f}s)")

        sys.stdout.write("  -> DQN 에이전트 학습 및 평가 중... ")
        sys.stdout.flush()
        start_t = time.time()
        res_dqn = run_single_scenario(sc, scenario_hospitals, base_seed, mode='dqn', agent=dqn_agent)
        elapsed_d = time.time() - start_t
        print(f"완료(reward : {res_dqn['reward']:.1f}, time : {elapsed_d:.1f}s)")

        print(f"\n[시나리오 {sc['id']} 상세 지표 요약]")
        print(f"  · 최단거리    | 평균보상: {res_nearest['avg_reward']:6.1f} | 골든타임 성공률: {res_nearest['golden_success_rate']:5.1f}% | 거부: {res_nearest['rejections']:2d} | 평균소요시간: {res_nearest['avg_time']:5.1f}s | 부하편차: {res_nearest['load_std']:.2f}")
        print(f"  · 규칙기반    | 평균보상: {res_rule['avg_reward']:6.1f} | 골든타임 성공률: {res_rule['golden_success_rate']:5.1f}% | 거부: {res_rule['rejections']:2d} | 평균소요시간: {res_rule['avg_time']:5.1f}s | 부하편차: {res_rule['load_std']:.2f}")
        print(f"  · 휴리스틱    | 평균보상: {res_heur['avg_reward']:6.1f} | 골든타임 성공률: {res_heur['golden_success_rate']:5.1f}% | 거부: {res_heur['rejections']:2d} | 평균소요시간: {res_heur['avg_time']:5.1f}s | 부하편차: {res_heur['load_std']:.2f}")
        print(f"  · DQN 에이전트| 평균보상: {res_dqn['avg_reward']:6.1f} | 골든타임 성공률: {res_dqn['golden_success_rate']:5.1f}% | 거부: {res_dqn['rejections']:2d} | 평균소요시간: {res_dqn['avg_time']:5.1f}s | 부하편차: {res_dqn['load_std']:.2f}")
        print("-" * 80)

    print("\n모든 시나리오 평가가 성공적으로 완료되었습니다.")