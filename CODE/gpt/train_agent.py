# -*- coding: utf-8 -*-
"""KTAS DQN 학습.

핵심 수정:
- 환자 Edge를 무작위 전체 edge에서 뽑지 않고 ambulance 후보에서 생성.
- 실제 TraCI findRoute로 환자 출발 가능 여부를 검증.
- 환자 생성/유효화/route 실패/학습 transition을 각각 카운트.
- numpy import 누락 수정.
- DQN의 valid action mask 사용.
- 모델과 실험 로그 자동 저장.
"""

import contextlib
import copy
import csv
import datetime
import os
import random
import sys
import argparse
import time
from experiment_log import append_record, new_run_id, LOG_PATH

import numpy as np
import sumolib
import traci

from baseline import HospitalRouters, build_state_vector
from dqn_agent import DQNAmbulanceAgent
from environment import SumoMedicalEnvironment
from reward import GOLDEN_TIME_LIMITS, calculate_reward, normalize_ktas
from ambulance_sim import MissionManager


MODEL_PATH = "dqn_ambulance_model_ktas_v2.pth"
NUM_EPISODES = 100
MAX_AMBULANCES = 10  # 구급차 최대 대수
DEFAULT_AMBULANCES = 10
MAX_STEPS = 500
HOSPITAL_FULL_THRESHOLD = 0.85
MAX_ESTIMATED_TRAVEL_TIME = 600.0
BATCH_SIZE = 32
TARGET_UPDATE_EVERY = 10

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


@contextlib.contextmanager
def suppress_sumo_stdout():
    old_out, old_err = sys.stdout, sys.stderr
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def start_sumo(env, scenario, seed, gui=False):
    binary = sumolib.checkBinary("sumo-gui" if gui else "sumo")
    cmd = [
        binary, "-c", str(env.cfg_file),
        "--scale", str(scenario["scale"]),
        "--no-warnings", "true",
        "--no-step-log", "true",
        "--duration-log.disable", "true",
        "--seed", str(seed),
        "--start",
    ]

    with suppress_sumo_stdout():
        traci.start(cmd)

        try:
            traci.vehicletype.copy("DEFAULT_VEHTYPE", "ambulance_custom")
            traci.vehicletype.setVehicleClass("ambulance_custom", "emergency")
            traci.vehicletype.setShapeClass("ambulance_custom", "emergency")
        except Exception as exc:
            raise RuntimeError(f"ambulance_custom 초기화 실패: {exc}") from exc

    # 반드시 SUMO 시작 후 좌표 변환
    env.update_hospital_coordinates()


def edge_allows_ambulance(env, edge_id):
    """TraCI에서 실제 ambulance_custom 출발 가능 여부를 1회 확인."""
    try:
        route = traci.simulation.findRoute(
            edge_id, edge_id,
            vType="ambulance_custom",
            depart=-1,
        )
        return route is not None and len(route.edges) > 0
    except Exception:
        return False


def build_ambulance_edges(env, max_check=1200):
    """후보 edge 중 실제 구급차 출발 가능 edge를 확보.

    모든 26,000개 edge에 findRoute를 수행하면 너무 느리므로
    deterministic하게 후보를 순회하며 충분한 수를 확보한다.
    """
    candidates = env.get_ambulance_candidate_edges()
    valid = []

    # 네트워크가 크므로 처음부터 전부 검사하지 않고 순서를 섞어 다양한 지역 확보
    rng = random.Random(20261007)
    candidates = list(candidates)
    rng.shuffle(candidates)

    for edge_id in candidates:
        if edge_allows_ambulance(env, edge_id):
            valid.append(edge_id)
            if len(valid) >= max_check:
                break

    if not valid:
        raise RuntimeError(
            "ambulance_custom이 출발 가능한 Edge를 하나도 찾지 못했습니다. "
            "SUMO 차량 타입/도로 permission을 확인하세요."
        )

    return valid


def generate_schedule(scenario, ambulance_edges, seed):
    rng = random.Random(seed)
    schedule = {}

    for step in range(1, MAX_STEPS + 1):
        if rng.random() < scenario["prob"]:
            pre_ktas = rng.randint(1, 5)
            schedule[step] = {
                "edge_id": rng.choice(ambulance_edges),
                "pre_ktas": pre_ktas,
                "ktas": pre_ktas,  # 기존 코드 호환용 별칭
            }

    return schedule


def valid_actions(hospitals):
    actions = [
        i for i, h in enumerate(hospitals)
        if h.get("reachable", False)
        and float(h.get("occupancy", 1.0)) < HOSPITAL_FULL_THRESHOLD
    ]

    if actions:
        return actions

    return [
        i for i, h in enumerate(hospitals)
        if h.get("reachable", False)
    ]


def get_reachable_hospital_routes(env, patient_edge, hospitals):
    result = copy.deepcopy(hospitals)
    failures = 0

    for h in result:
        h["reachable"] = False
        h["hospital_edge"] = None
        h["route_length"] = None
        h["estimated_travel_time"] = MAX_ESTIMATED_TRAVEL_TIME

        x, y = h.get("sumo_x"), h.get("sumo_y")
        if x is None or y is None:
            failures += 1
            continue

        try:
            nearby = env.net.getNeighboringEdges(x, y, 5000.0)
            nearby = sorted(nearby, key=lambda z: z[1])

            for edge, _ in nearby:
                edge_id = edge.getID()

                if not edge_id or edge_id.startswith(":"):
                    continue

                try:
                    with suppress_sumo_stdout():
                        route = traci.simulation.findRoute(
                            patient_edge,
                            edge_id,
                            vType="ambulance_custom",
                            depart=-1,
                        )

                    if route is None or len(route.edges) == 0:
                        continue

                    length = float(getattr(route, "length", 0.0))
                    travel = float(getattr(route, "travelTime", 0.0))

                    if travel <= 0:
                        travel = max(length / 13.9, 1.0)

                    h["reachable"] = True
                    h["hospital_edge"] = edge_id
                    h["route_length"] = length
                    h["estimated_travel_time"] = min(
                        travel, MAX_ESTIMATED_TRAVEL_TIME
                    )
                    break

                except Exception:
                    continue

        except Exception:
            pass

        if not h["reachable"]:
            failures += 1

    return result, failures


def train(num_episodes=NUM_EPISODES, gui=False, ambulance_count=DEFAULT_AMBULANCES):
    ambulance_count = max(1, min(MAX_AMBULANCES, int(ambulance_count)))
    print(f"[설정] SUMO 모드={'GUI' if gui else '백그라운드'}, 구급차={ambulance_count}/{MAX_AMBULANCES}")
    env = SumoMedicalEnvironment()

    action_dim = len(env.get_hospitals())
    state_dim = 3 + action_dim * 4

    agent = DQNAmbulanceAgent(state_dim, action_dim)
    router = HospitalRouters(env.get_hospitals())

    print(f"[KTAS DQN] state_dim={state_dim}, action_dim={action_dim}")
    print(f"[KTAS DQN] 모델={MODEL_PATH}")

    # episode 1에서 한 번만 실제 출발 가능 edge를 확인
    print("[검증] ambulance_custom 출발 가능 Edge 확인 중...")
    temp_seed = 20261007
    start_sumo(env, SCENARIOS[0], temp_seed, gui=False)

    try:
        ambulance_edges = build_ambulance_edges(env, max_check=1200)
    finally:
        try:
            traci.close()
        except Exception:
            pass

    print(f"[검증 완료] 구급차 출발 가능 Edge 후보: {len(ambulance_edges)}개")
    base_hospitals = copy.deepcopy(env.get_hospitals())

    results_dir = os.path.join("results", "training")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = new_run_id("train")
    run_started = time.monotonic()
    append_record(run_id, "training", "run_start", status="started",
                  ambulance_limit=ambulance_count, model_path=MODEL_PATH,
                  metrics={"requested_episodes": num_episodes, "gui": gui,
                           "ambulance_edges": len(ambulance_edges)})
    print(f"[통합 실험 로그] {LOG_PATH} | 실행 ID: {run_id}")

    best_reward = -float("inf")
    history = {"episode": [], "reward": [], "avg_reward": [], "avg_loss": [], "missions": []}

    for episode in range(1, num_episodes + 1):
        scenario = SCENARIOS[(episode - 1) % len(SCENARIOS)]
        seed = 10000 + episode

        hospitals = copy.deepcopy(base_hospitals)
        for h in hospitals:
            h["occupancy"] = random.uniform(0.20, 0.60)

        schedule = generate_schedule(scenario, ambulance_edges, seed)

        generated = len(schedule)
        valid_patients = 0
        route_failures = 0
        missions = 0
        episode_reward = 0.0
        losses = []

        manager = MissionManager(max_ambulances=ambulance_count, timeout=1800,
                                 max_wait=1800, verbose=False)
        outcome_rows = []

        def dispatch(patient, vid):
            nonlocal valid_patients, route_failures
            edge_id = patient["edge_id"]
            ktas = normalize_ktas(patient["ktas"])
            try:
                patient_pos = env.net.getEdge(edge_id).getFromNode().getCoord()[:2]
            except Exception:
                route_failures += 1
                return None
            routed, failures = get_reachable_hospital_routes(env, edge_id, hospitals)
            route_failures += failures
            actions = valid_actions(routed)
            if not actions:
                return None
            valid_patients += 1
            state = build_state_vector(patient_pos, ktas, routed)
            action = int(agent.select_action(state, actions))
            if action not in actions:
                return None
            selected = copy.deepcopy(routed[action])
            occupancy = float(selected.get("occupancy", 0.))
            return selected["hospital_edge"], (state, action, selected, occupancy)

        def finish(patient, payload, arrived, elapsed, wait, reason):
            nonlocal episode_reward, missions, route_failures
            ktas = normalize_ktas(patient["ktas"])
            if payload is None:
                # No hospital action was taken. Log separately; never fabricate a DQN action.
                route_failures += 1
                outcome_rows.append((patient['occurrence_time'], wait, elapsed,
                                     False, reason, -1))
                return
            state, action, selected, occupancy = payload
            failed = not arrived
            if failed:
                route_failures += 1
            reward = calculate_reward(
                travel_time=elapsed, ktas=ktas, hospital=selected,
                golden_limit=GOLDEN_TIME_LIMITS[ktas],
                rejection_count=0, occupancy_at_selection=occupancy,
                route_failed=failed, pre_ktas=ktas,
            )
            agent.store_transition(state, action, reward,
                                   np.zeros(state_dim, dtype=np.float32), True)
            loss = agent.train_step(BATCH_SIZE)
            if loss is not None:
                losses.append(loss)
            episode_reward += reward
            missions += 1
            if arrived:
                hospitals[action]['occupancy'] = min(
                    1., float(hospitals[action].get('occupancy', 0.)) + 0.08)
            outcome_rows.append((patient['occurrence_time'], wait, elapsed,
                                 arrived, reason, action))

        try:
            start_sumo(env, scenario, seed, gui=gui)
            # One global SUMO clock. Events are queued even when 10 ambulances are active.
            # Allow additional simulation time for the last patients to finish.
            horizon = MAX_STEPS + 1800
            traci.simulationStep()  # time=1: 이벤트 시각과 실제 출동 시각 정렬
            for step in range(1, horizon + 1):
                if step <= MAX_STEPS and step in schedule:
                    patient = dict(schedule[step])
                    patient['occurrence_time'] = float(step)
                    manager.enqueue(patient)
                manager.step(dispatch, finish, prefix=f'amb_e{episode}_')
                if step >= MAX_STEPS and not manager.active and not manager.waiting:
                    break
            manager.close_unfinished(finish)
            if episode % TARGET_UPDATE_EVERY == 0:
                agent.update_target_network()
        finally:
            try:
                traci.close()
            except Exception:
                pass

        print(f"[대기열] 최대 동시운행={manager.peak_active}/{ambulance_count}, "
              f"최대 대기={manager.peak_waiting}, 도착={manager.completed}, "
              f"실패={manager.failed}, 평균 대기="
              f"{manager.wait_total/max(1, manager.wait_count):.1f}초")
        for occurrence, wait, total, arrived, reason, action in outcome_rows:
            append_record(run_id, "training", "mission", episode=episode,
                          scenario_id=scenario["id"], scenario_name=scenario["name"],
                          status="arrived" if arrived else "failed",
                          ambulance_limit=ambulance_count, model_path=MODEL_PATH,
                          metrics={"occurrence_time": occurrence, "wait_seconds": wait,
                                   "total_seconds": total, "arrived": arrived,
                                   "reason": reason, "hospital_action": action})

        avg_reward = episode_reward / missions if missions else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        print(
            f"Episode {episode:3d}/{num_episodes} | "
            f"Scenario {scenario['id']} | "
            f"생성 {generated:3d} | 유효 {valid_patients:3d} | "
            f"경로실패 {route_failures:3d} | "
            f"환자 {missions:3d} | "
            f"Reward {episode_reward:9.2f} | "
            f"Avg {avg_reward:7.2f} | "
            f"Loss {avg_loss:.5f} | "
            f"Epsilon {agent.epsilon:.4f} | "
            f"Memory {len(agent.memory):5d}"
        )

        append_record(
            run_id, "training", "episode", episode=episode,
            scenario_id=scenario["id"], scenario_name=scenario["name"],
            status="completed", ambulance_limit=ambulance_count,
            model_path=MODEL_PATH,
            duration_seconds=round(time.monotonic() - run_started, 2),
            metrics={"generated_patients": generated, "valid_patients": valid_patients,
                     "route_failures": route_failures, "missions": missions,
                     "reward": episode_reward, "avg_reward": avg_reward,
                     "avg_loss": avg_loss, "epsilon": agent.epsilon,
                     "replay_memory": len(agent.memory),
                     "peak_active": manager.peak_active,
                     "peak_waiting": manager.peak_waiting,
                     "arrived": manager.completed, "failed": manager.failed,
                     "avg_wait_seconds": manager.wait_total/max(1, manager.wait_count)}
        )

        history["episode"].append(episode)
        history["reward"].append(episode_reward)
        history["avg_reward"].append(avg_reward)
        history["avg_loss"].append(avg_loss)
        history["missions"].append(missions)

        if missions > 0 and episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(MODEL_PATH)

        # 환자 0이 반복되면 100회 학습을 낭비하지 않도록 즉시 중단
        if episode >= 1 and generated > 0 and valid_patients == 0:
            raise RuntimeError(
                "환자는 생성되었지만 유효한 환자가 0명입니다. "
                "ambulance_custom의 route/edge permission 또는 SUMO network를 확인하세요."
            )

    agent.save_model(MODEL_PATH)
    save_training_graphs(history, results_dir, timestamp)
    append_record(run_id, "training", "run_end", status="completed",
                  ambulance_limit=ambulance_count, model_path=MODEL_PATH,
                  duration_seconds=round(time.monotonic() - run_started, 2),
                  metrics={"completed_episodes": len(history["episode"])})

    print("\n" + "=" * 70)
    print("DQN 학습 완료")
    print(f"모델: {MODEL_PATH}")
    print(f"통합 실험 로그 CSV: {LOG_PATH}")
    print(f"실행 ID: {run_id}")
    print("=" * 70)


def save_training_graphs(history, results_dir, timestamp):
    """학습 그래프를 PNG로 저장한다. matplotlib 미설치 시 로그는 그대로 보존한다."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for key, title, ylabel in [
            ("avg_reward", "DQN Average Reward", "Average reward"),
            ("avg_loss", "DQN Training Loss", "Loss"),
            ("missions", "Valid Ambulance Missions", "Missions"),
        ]:
            plt.figure(figsize=(10, 5))
            plt.plot(history["episode"], history[key])
            plt.title(title); plt.xlabel("Episode"); plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3); plt.tight_layout()
            out = os.path.join(results_dir, f"{key}_{timestamp}.png")
            plt.savefig(out, dpi=150); plt.close()
            print(f"그래프 저장: {out}")
    except ImportError:
        print("[안내] 그래프 저장을 위해 'pip install matplotlib'을 실행하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="구급차 DQN 학습")
    parser.add_argument("--gui", action="store_true", help="SUMO-GUI로 실행")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--ambulances", type=int, default=DEFAULT_AMBULANCES,
                        help="구급차 대수(최대 10)")
    args = parser.parse_args()
    train(args.episodes, gui=args.gui, ambulance_count=args.ambulances)
