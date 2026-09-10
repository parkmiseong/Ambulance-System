# run_scenarios.py  ── 개선 버전
#!/usr/bin/env python3
"""
[수정 사항]
  1. State 벡터: dim 7 → 1+4*H (병원별 이동시간·점유율·역량·기기 포함)
  2. next_state: hosp_pos 오용 → pending_transition 방식으로 교체
  3. reward: 단순 시간-점유율 → 중증도 가중 MDP 보상
  4. target network: 미업데이트 → train_step() 내 주기적 업데이트
  5. 학습-평가 분리: 동일 에피소드 → 시나리오별 학습 후 greedy 평가
  6. 점유율 인위 감소 제거 → 실제 이송 완료 시만 변경
"""
import os, sys, time, random, copy, contextlib
import numpy as np
import traci
from pathlib import Path
import sumolib

from environment import SumoMedicalEnvironment
from dqn_agent import DQNAmbulanceAgent
from baselines import HospitalRouters, req_level, eq_score, SEV_EQ

traci.init_log = lambda *args, **kwargs: None

MAX_STEPS         = 600
GOLDEN_TIME_LIMIT = 240

SCENARIOS = [
    {"id":1,"patient_freq":"원활","traffic":"원활","complexity":2,"prob":0.05,"scale":0.3},
    {"id":2,"patient_freq":"원활","traffic":"보통","complexity":3,"prob":0.05,"scale":0.5},
    {"id":3,"patient_freq":"원활","traffic":"혼잡","complexity":4,"prob":0.05,"scale":0.8},
    {"id":4,"patient_freq":"보통","traffic":"원활","complexity":3,"prob":0.10,"scale":0.3},
    {"id":5,"patient_freq":"보통","traffic":"보통","complexity":4,"prob":0.10,"scale":0.5},
    {"id":6,"patient_freq":"보통","traffic":"혼잡","complexity":5,"prob":0.10,"scale":0.8},
    {"id":7,"patient_freq":"혼잡","traffic":"원활","complexity":4,"prob":0.15,"scale":0.3},
    {"id":8,"patient_freq":"혼잡","traffic":"보통","complexity":5,"prob":0.15,"scale":0.5},
    {"id":9,"patient_freq":"혼잡","traffic":"혼잡","complexity":6,"prob":0.15,"scale":0.8},
]


# ══════════════════════════════════════════════════════════════
# [수정 ①] State 벡터 빌더 — 병원별 특성 포함
# ══════════════════════════════════════════════════════════════
def build_state(severity: int, travel_times: list, hospitals: list) -> np.ndarray:
    """
    State = [severity/4, (tt_j/300, occ_j, cap_j/3, eq_j) × H]
    차원  = 1 + 4 × len(hospitals)

    원본 state (dim=7):
        [pat_x/10000, pat_y/10000, severity/4, 0, speed/20, tt/100, mean_occ]
        → 병원별 정보 없음 → DQN 병원 구별 불가

    개선 state:
        각 병원 j 별로: 이동시간, 점유율, 치료역량, 기기점수 포함
        → DQN이 각 병원의 현재 상태를 직접 비교 가능
    """
    obs = [severity / 4.0]
    max_tt = max(travel_times) + 1e-8
    for j, h in enumerate(hospitals):
        obs.extend([
            travel_times[j] / max_tt,            # 정규화된 이동시간
            min(h['occupancy'], 1.0),             # 점유율
            h['capability'] / 3.0,               # 치료역량
            eq_score(severity, h),               # 기기 보유율
        ])
    return np.array(obs, dtype=np.float32)


# ── MDP 보상 함수 ──────────────────────────────────────────────
def compute_reward(severity: int, hospital: dict, travel_time: float) -> float:
    """
    [수정 ③] Reward = -(severity_w × delay + cap_gap_pen + eq_pen)

    원본: -travel_time*0.1 - occupancy*20 - mismatch(50)
      → 중증도 가중치 없음, capability 세분화 없음

    개선:
      severity_w = severity^1.5  (중증일수록 지연의 비용 증가)
      cap_gap    = 치료역량 부족 등급 × 25점
      eq_pen     = 필요 기기 미보유 × 20점
    """
    OVERFLOW_U  = 8.0
    MISMATCH_U  = 25.0
    EQ_U        = 20.0

    ovf      = max(0, int(hospital['occupancy']*hospital['capacity'])
                   + 1 - hospital['capacity'])
    cap_gap  = max(0, req_level(severity) - hospital['capability'])
    req_eq   = SEV_EQ.get(severity, [])
    eq_ok    = all(e in hospital.get('equipment',[]) for e in req_eq)
    eq_pen   = 0 if eq_ok else EQ_U * len(req_eq)

    sev_w  = severity ** 1.5
    delay  = travel_time + ovf * OVERFLOW_U
    return -(sev_w * delay + cap_gap * MISMATCH_U + eq_pen)


# ── 유틸 ──────────────────────────────────────────────────────
@contextlib.contextmanager
def suppress_sumo_stdout():
    """SUMO 표준 출력 억제 (원본과 동일)"""
    ofd, efd = sys.stdout.fileno(), sys.stderr.fileno()
    sfd, sefd = os.dup(ofd), os.dup(efd)
    with open(os.devnull, "w") as dn:
        os.dup2(dn.fileno(), ofd); os.dup2(dn.fileno(), efd)
        try:    yield
        finally:
            os.dup2(sfd, ofd); os.dup2(sefd, efd)
            os.close(sfd); os.close(sefd)

def get_closest_edge_id(net, x, y):
    """원본과 동일"""
    radius = 50
    while radius < 3000:
        edges = net.getNeighboringEdges(x, y, radius)
        if edges:
            for edge, _ in sorted(edges, key=lambda e: e[1]):
                if not edge.getID().startswith(':'):
                    return edge.getID()
        radius += 100
    return None

def get_travel_times(net, from_edge: str, hospitals: list) -> list:
    """
    각 병원까지 TraCI findRoute 이동시간(초).
    실패 시 유클리드 거리로 fallback.
    """
    try:
        from_edge_obj = net.getEdge(from_edge)
        fx, fy = from_edge_obj.getFromNode().getCoord()
    except Exception:
        fx, fy = 0, 0

    tts = []
    for h in hospitals:
        try:
            h_edge = get_closest_edge_id(net, h['sumo_x'], h['sumo_y'])
            if h_edge and not from_edge.startswith(':'):
                route = traci.simulation.findRoute(from_edge, h_edge,
                                                   vType="ambulance")
                tts.append(route.travelTime if route else 999.0)
            else:
                raise Exception("edge 없음")
        except Exception:
            tts.append(np.hypot(fx - h['sumo_x'], fy - h['sumo_y']) / 10.0)
    return tts


# ══════════════════════════════════════════════════════════════
# 에피소드 실행
# ══════════════════════════════════════════════════════════════
def run_single_scenario(scenario, initial_hospitals, seed_val,
                         mode='dqn', agent=None, evaluate=False):
    """
    evaluate=True  → agent.greedy_action() 사용 (epsilon=0, 학습 없음)
    evaluate=False → agent.select_action()  사용 (epsilon-greedy, 학습)
    """
    env = SumoMedicalEnvironment()
    random.seed(seed_val); np.random.seed(seed_val)

    sumo_binary = sumolib.checkBinary("sumo")
    sumo_config = [sumo_binary, "-c", str(env.cfg_file),
                   "--scale", str(scenario["scale"]),
                   "--no-warnings", "true", "--no-step-log", "true",
                   "--duration-log.disable", "true",
                   "--seed", str(seed_val), "--start"]
    try:
        with suppress_sumo_stdout():
            traci.start(sumo_config)
    except Exception as e:
        return {"reward":0,"avg_reward":0,"golden_success_rate":0.0,
                "rejections":0,"avg_time":0.0,"load_std":0.0,
                "zero_reason":f"SUMO 실행 실패: {e}"}

    hospitals = copy.deepcopy(initial_hospitals)
    router    = HospitalRouters(hospitals)
    valid_edges = env.get_valid_edges()

    total_reward, completed, golden, rejections = 0.0, 0, 0, 0
    total_time, assign_counts = 0.0, {h['id']:0 for h in hospitals}
    active_missions = {}
    pending_trans   = None       # [수정 ②] (obs, action, reward) 대기 전이

    try:
        for step in range(1, MAX_STEPS + 1):
            try: traci.simulationStep()
            except traci.exceptions.FatalTraCIError: break

            # [수정 ⑥] 인위 점유율 감소 제거 → 이송 완료 시에만 변경

            ambulances = [v for v in traci.vehicle.getIDList()
                          if 'amb' in v.lower() or 'ambulance' in v.lower()]

            # ── 미션 진행 상태 업데이트 ──────────────────────────
            for amb_id in list(active_missions.keys()):
                if amb_id not in ambulances:
                    del active_missions[amb_id]; continue

                active_missions[amb_id]['elapsed_time'] += 1
                amb_pos  = traci.vehicle.getPosition(amb_id)
                mission  = active_missions[amb_id]

                if mission['state_phase'] == 'to_patient':
                    dist = np.hypot(amb_pos[0]-mission['pat_pos'][0],
                                    amb_pos[1]-mission['pat_pos'][1])
                    if dist <= 60.0 or mission['elapsed_time'] > 150:
                        mission['state_phase']   = 'to_hospital'
                        mission['elapsed_time']  = 0
                        h = mission['hospital']
                        print(f"[픽업 Step{step}] {amb_id} → {h['name']} 이송 시작")
                        h_edge = get_closest_edge_id(env.net, h['sumo_x'], h['sumo_y'])
                        curr_e = traci.vehicle.getRoadID(amb_id)
                        if curr_e.startswith(':'):
                            curr_e = traci.vehicle.getLaneID(amb_id).rsplit('_',1)[0]
                        if h_edge and curr_e:
                            try:
                                rt = traci.simulation.findRoute(curr_e, h_edge,
                                                                vType="ambulance")
                                if rt and rt.edges:
                                    traci.vehicle.setRoute(amb_id, list(rt.edges))
                                else:
                                    traci.vehicle.changeTarget(amb_id, h_edge)
                            except: pass

                elif mission['state_phase'] == 'to_hospital':
                    dist = np.hypot(amb_pos[0]-mission['target_pos'][0],
                                    amb_pos[1]-mission['target_pos'][1])
                    if dist <= 80.0 or mission['elapsed_time'] >= 350:
                        tt      = mission['elapsed_time']
                        sev     = mission['severity']
                        chosen  = mission['hospital']
                        completed += 1
                        total_time += tt
                        if tt <= GOLDEN_TIME_LIMIT: golden += 1

                        # [수정 ③] 개선된 보상 함수
                        reward = compute_reward(sev, chosen, tt)
                        total_reward += reward

                        print(f"[완료 Step{step}] {amb_id} → {chosen['name']} "
                              f"| {tt}s | 골든: {'O' if tt<=GOLDEN_TIME_LIMIT else 'X'} "
                              f"| r={reward:.1f}")

                        # [수정 ②] next_state 기반 전이 완성
                        if mode == 'dqn' and agent and not evaluate:
                            # 도착 후 현재 상태를 next_obs로 사용
                            curr_e = traci.vehicle.getRoadID(amb_id)
                            if curr_e.startswith(':'):
                                curr_e = traci.vehicle.getLaneID(amb_id).rsplit('_',1)[0]
                            try:
                                next_tts  = get_travel_times(env.net, curr_e, hospitals)
                                next_obs  = build_state(sev, next_tts, hospitals)
                            except Exception:
                                next_obs  = mission['state_vector']

                            if pending_trans is not None:
                                p_obs, p_act, p_rwd = pending_trans
                                agent.store_transition(p_obs, p_act, p_rwd,
                                                       next_obs, False)
                                # [수정 ④] train_step 내에서 target 업데이트 처리
                                agent.train_step()

                            pending_trans = (mission['state_vector'],
                                            mission['action'], reward)

                        # [수정 ⑥] 실제 점유율 반영
                        chosen['occupied'] = max(0, chosen.get('occupied',0)-1)
                        chosen['occupancy'] = chosen['occupied']/max(chosen['capacity'],1)
                        del active_missions[amb_id]

            # ── 환자 발생 ─────────────────────────────────────────
            if random.random() < scenario["prob"]:
                idle_ambs = [a for a in ambulances if a not in active_missions]
                if not idle_ambs: continue
                best_amb = random.choice(idle_ambs)

                try:
                    amb_pos = traci.vehicle.getPosition(best_amb)
                    curr_e  = traci.vehicle.getRoadID(best_amb)
                    if curr_e.startswith(':'):
                        curr_e = traci.vehicle.getLaneID(best_amb).rsplit('_',1)[0]

                    # 환자 위치
                    radius = 200
                    p_edge = None
                    while radius < 4000:
                        cands = [e[0].getID() for e in
                                 env.net.getNeighboringEdges(amb_pos[0],amb_pos[1],radius)
                                 if not e[0].getID().startswith(':')]
                        if cands: p_edge = random.choice(cands); break
                        radius += 200
                    if not p_edge: p_edge = random.choice(valid_edges)

                    p_xy  = env.net.getEdge(p_edge).getFromNode().getCoord()
                    sev   = random.randint(1, 4)

                    # [수정 ①] 병원별 이동시간 계산 → state 빌드
                    tts   = get_travel_times(env.net, curr_e, hospitals)
                    state = build_state(sev, tts, hospitals)

                    # ── 정책별 행동 선택 ──
                    for _ in range(3):
                        if mode == 'dqn' and agent:
                            action = (agent.greedy_action(state) if evaluate
                                      else agent.select_action(state))
                        elif mode == 'nearest':
                            action = router.nearest_strategy(p_xy, tts)
                        elif mode == 'rule':
                            action = router.rule_based_strategy(sev, tts)
                        elif mode == 'heuristic':
                            action = router.heuristic_strategy(p_xy, sev, tts)
                        else:
                            action = 0

                        action = min(action, len(hospitals)-1)
                        sel_h  = hospitals[action]

                        if sel_h['occupancy'] >= 0.85 and random.random() < 0.7:
                            rejections += 1; continue
                        break

                    assign_counts[sel_h['id']] += 1
                    # [수정 ⑥] 이송 시작 시 점유 처리
                    sel_h['occupied'] = sel_h.get('occupied', 0) + 1
                    sel_h['occupancy'] = min(1.0, sel_h['occupied']/max(sel_h['capacity'],1))

                    print(f"\n[환자 Step{step}] Sev={sev} → {sel_h['name']} "
                          f"(ε={agent.epsilon:.3f} buf={len(agent.memory)})"
                          if mode == 'dqn' and agent else
                          f"\n[환자 Step{step}] Sev={sev} → {sel_h['name']} [{mode}]")

                    rt = traci.simulation.findRoute(curr_e, p_edge, vType="ambulance")
                    if rt and rt.edges:
                        traci.vehicle.setRoute(best_amb, list(rt.edges))
                    else:
                        traci.vehicle.changeTarget(best_amb, p_edge)

                    active_missions[best_amb] = {
                        'state_phase' : 'to_patient',
                        'pat_pos'     : p_xy,
                        'hospital'    : sel_h,
                        'severity'    : sev,
                        'elapsed_time': 0,
                        'state_vector': state,
                        'action'      : action,
                        'target_pos'  : (sel_h['sumo_x'], sel_h['sumo_y']),
                    }
                except Exception:
                    continue

    finally:
        try: traci.close()
        except: pass

    success_rate = golden/completed*100.0 if completed else 0.0
    avg_time     = total_time/completed if completed else 0.0
    avg_reward   = total_reward/completed if completed else 0.0
    load_std     = float(np.std(list(assign_counts.values())))

    return {"reward":total_reward,"avg_reward":avg_reward,
            "golden_success_rate":success_rate,"rejections":rejections,
            "avg_time":avg_time,"load_std":load_std,"zero_reason":""}


# ══════════════════════════════════════════════════════════════
# 메인: [수정 ⑤] 사전 학습 → greedy 평가 분리
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== 9가지 시나리오 통합 평가 ===")

    temp_env    = SumoMedicalEnvironment()
    state_dim   = 1 + 4 * len(temp_env.get_hospitals())   # [수정 ①]
    action_dim  = len(temp_env.get_hospitals())

    # [수정 ⑤] DQN 사전 학습: 시나리오 실행 전에 충분히 학습
    dqn_agent = DQNAmbulanceAgent(
        state_dim, action_dim,
        eps_decay_steps=3000,      # 3000 학습 스텝에 걸쳐 탐색→활용
        target_update_freq=200,    # 200 스텝마다 target 동기화
    )

    print(f"\n[DQN 설정]")
    print(f"  state_dim  = {state_dim} (1 + 4×{action_dim}병원)")
    print(f"  action_dim = {action_dim}")
    print(f"  eps_decay  = {dqn_agent.eps_decay_steps} 스텝")
    print(f"  target_upd = 매 {dqn_agent.target_update_freq} 스텝")

    for sc in SCENARIOS:
        print(f"\n{'='*70}")
        print(f"시나리오 {sc['id']}/9 | 환자:{sc['patient_freq']} | 교통:{sc['traffic']}")
        print(f"{'='*70}")

        base_seed = sc['id'] * 100
        sc_hosps  = copy.deepcopy(temp_env.get_hospitals())

        results = {}
        for mode in ['nearest','rule','heuristic']:
            sys.stdout.write(f"  [{mode:10s}] 평가 중... ")
            sys.stdout.flush()
            t0 = time.time()
            results[mode] = run_single_scenario(
                sc, sc_hosps, base_seed, mode=mode)
            print(f"완료 ({time.time()-t0:.1f}s) "
                  f"reward={results[mode]['reward']:.1f}")

        # DQN: 학습 단계 (epsilon-greedy)
        sys.stdout.write(f"  [DQN(학습)] 실행 중... ")
        sys.stdout.flush()
        t0 = time.time()
        run_single_scenario(sc, sc_hosps, base_seed,
                            mode='dqn', agent=dqn_agent, evaluate=False)
        print(f"완료 ({time.time()-t0:.1f}s) | "
              f"buf={len(dqn_agent.memory)} "
              f"steps={dqn_agent.train_count} "
              f"ε={dqn_agent.epsilon:.3f}")

        # [수정 ⑤] DQN: 평가 단계 (greedy, epsilon=0)
        sys.stdout.write(f"  [DQN(평가)] 실행 중... ")
        sys.stdout.flush()
        t0 = time.time()
        results['dqn'] = run_single_scenario(
            sc, sc_hosps, base_seed+1,
            mode='dqn', agent=dqn_agent, evaluate=True)
        print(f"완료 ({time.time()-t0:.1f}s) "
              f"reward={results['dqn']['reward']:.1f}")

        print(f"\n  {'정책':<12} {'평균보상':>8} {'골든%':>7} "
              f"{'거부':>5} {'평균시간':>8} {'부하편차':>8}")
        print(f"  {'-'*55}")
        for name, r in results.items():
            print(f"  {name:<12} {r['avg_reward']:>8.1f} "
                  f"{r['golden_success_rate']:>7.1f} "
                  f"{r['rejections']:>5d} "
                  f"{r['avg_time']:>8.1f}s "
                  f"{r['load_std']:>8.2f}")

    print("\n=== 모든 시나리오 평가 완료 ===")
