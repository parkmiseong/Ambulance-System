# -*- coding: utf-8 -*-
"""KTAS DQN + 5개 baseline 평가."""

import contextlib
import copy
import csv
import datetime
import os
import random
import sys
import argparse
import time
from collections import Counter
from experiment_log import append_record, new_run_id, LOG_PATH

import numpy as np
import sumolib
import traci

from baseline import HospitalRouters, build_state_vector
from dqn_agent import DQNAmbulanceAgent
from environment import SumoMedicalEnvironment
from reward import GOLDEN_TIME_LIMITS, calculate_reward, normalize_ktas
from ambulance_sim import MissionManager


MAX_STEPS = 600
MAX_AMBULANCES = 10
DEFAULT_AMBULANCES = 10
MODEL_PATH = "dqn_ambulance_model_ktas_v2.pth"
MAX_ESTIMATED_TRAVEL_TIME = 600.0
HOSPITAL_FULL_THRESHOLD = 0.85

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

STRATEGIES = [
    ("shortest_distance", "최단거리"),
    ("shortest_time", "최단시간"),
    ("bed", "병상기반"),
    ("rule", "치료역량기반"),
    ("heuristic", "종합휴리스틱"),
    ("dqn", "DQN"),
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
        sys.stdout = old_out
        sys.stderr = old_err


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
        except Exception:
            pass

    env.update_hospital_coordinates()


def schedule(scenario, valid_edges, seed):
    rng = random.Random(seed)
    out = {}

    for step in range(1, MAX_STEPS + 1):
        if rng.random() < scenario["prob"]:
            pre_ktas = rng.randint(1, 5)
            out[step] = {
                "edge_id": rng.choice(valid_edges),
                "pre_ktas": pre_ktas,
                "ktas": pre_ktas,
            }

    return out


def hospital_routes(env, patient_edge, hospitals):
    result = copy.deepcopy(hospitals)

    for h in result:
        h["reachable"] = False
        h["hospital_edge"] = None
        h["route_length"] = None
        h["estimated_travel_time"] = MAX_ESTIMATED_TRAVEL_TIME

        x, y = h.get("sumo_x"), h.get("sumo_y")
        if x is None or y is None:
            continue

        try:
            nearby = sorted(
                env.net.getNeighboringEdges(x, y, 5000.0),
                key=lambda z: z[1],
            )

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
            continue

    return result


def valid_actions(hospitals):
    a = [
        i for i, h in enumerate(hospitals)
        if h.get("reachable", False)
        and float(h.get("occupancy", 1.0)) < HOSPITAL_FULL_THRESHOLD
    ]

    if a:
        return a

    return [
        i for i, h in enumerate(hospitals)
        if h.get("reachable", False)
    ]


def choose(mode, router, patient_pos, ktas, hospitals, state, agent, actions):
    router.hospitals = hospitals

    if mode == "dqn":
        return int(agent.select_action(state, actions))
    if mode == "shortest_time":
        return router.shortest_time_strategy(actions)
    if mode == "shortest_distance":
        router._patient_x, router._patient_y = patient_pos[:2]
        return router.shortest_distance_strategy(actions)
    if mode == "bed":
        return router.bed_strategy(ktas, actions)
    if mode == "rule":
        return router.rule_based_strategy(ktas, actions)
    if mode == "heuristic":
        return router.heuristic_strategy(ktas, actions)

    raise ValueError(mode)


def run_single(env, base_hospitals, scenario, seed, mode, agent=None, gui=False, ambulance_count=DEFAULT_AMBULANCES):
    ambulance_count = max(1, min(MAX_AMBULANCES, int(ambulance_count)))
    random.seed(seed)
    np.random.seed(seed)

    hospitals = copy.deepcopy(base_hospitals)
    for h in hospitals:
        h["occupancy"] = random.uniform(0.20, 0.60)

    patients = schedule(
        scenario,
        [e for e in env.get_valid_edges() if any(l.allows("emergency") for l in env.net.getEdge(e).getLanes())],
        seed,
    )

    router = HospitalRouters(hospitals)

    total_reward = 0.0
    total_time = 0.0
    golden_success = 0
    missions = 0
    route_failures = 0
    route_diagnostics = Counter()
    assignments = {h.get("id", str(i)): 0 for i, h in enumerate(hospitals)}

    manager = MissionManager(max_ambulances=ambulance_count, timeout=1800,
                             max_wait=1800, verbose=False)
    outcome_rows = []
    def dispatch(patient, vid):
        nonlocal route_failures
        try:
            patient_pos = env.net.getEdge(patient['edge_id']).getFromNode().getCoord()[:2]
        except Exception:
            route_failures += 1
            route_diagnostics["invalid_patient_edge"] += 1
            return None
        ktas = normalize_ktas(patient['ktas'])
        routed = hospital_routes(env, patient['edge_id'], hospitals)
        actions = valid_actions(routed)
        if not actions:
            route_diagnostics["no_reachable_hospital"] += 1
            return None
        state = build_state_vector(patient_pos, ktas, routed)
        action = choose(mode, router, patient_pos, ktas,
                        routed, state, agent, actions)
        if action not in actions:
            route_diagnostics["invalid_action"] += 1
            return None
        selected = copy.deepcopy(routed[action])
        return selected['hospital_edge'], (action, selected, float(selected.get('occupancy', 0.)))

    def finish(patient, payload, arrived, elapsed, wait, reason):
        nonlocal total_reward, total_time, golden_success, missions, route_failures
        if payload is None:
            route_failures += 1
            outcome_rows.append((patient['occurrence_time'], wait, elapsed, False, reason, -1))
            return
        action, selected, occupancy = payload
        ktas = normalize_ktas(patient['ktas'])
        failed = not arrived
        if failed:
            route_failures += 1
            route_diagnostics["mission_failure"] += 1
        reward = calculate_reward(
            travel_time=elapsed, ktas=ktas,
            pre_ktas=patient.get('pre_ktas', ktas),
            hospital=selected, golden_limit=GOLDEN_TIME_LIMITS[ktas],
            occupancy_at_selection=occupancy, route_failed=failed)
        total_reward += reward
        total_time += elapsed
        missions += 1
        if arrived and elapsed <= GOLDEN_TIME_LIMITS[ktas]:
            golden_success += 1
        hid = selected.get('id', str(action))
        assignments[hid] = assignments.get(hid, 0) + 1
        if arrived:
            hospitals[action]['occupancy'] = min(
                1., float(hospitals[action].get('occupancy', 0.)) + 0.08)
        outcome_rows.append((patient['occurrence_time'], wait, elapsed, arrived, reason, action))

    try:
        start_sumo(env, scenario, seed, gui=gui)
        # 병원 좌표는 SUMO 시작 후 변환된다. 이전 복사본의 None 좌표를 갱신.
        fresh = env.get_hospitals()
        for idx, hospital in enumerate(hospitals):
            hospital['sumo_x'] = fresh[idx].get('sumo_x')
            hospital['sumo_y'] = fresh[idx].get('sumo_y')
        traci.simulationStep()  # time=1: 이벤트 시각과 실제 출동 시각 정렬
        for step in range(1, MAX_STEPS + 1800 + 1):
            if step <= MAX_STEPS and step in patients:
                patient = dict(patients[step])
                patient['occurrence_time'] = float(step)
                manager.enqueue(patient)
            manager.step(dispatch, finish, prefix=f'amb_eval_{mode}_{scenario["id"]}_')
            if step >= MAX_STEPS and not manager.active and not manager.waiting:
                break
        manager.close_unfinished(finish)
    finally:
        try:
            traci.close()
        except Exception:
            pass
    print(f"[대기열] {mode} 최대 운행={manager.peak_active}/{ambulance_count}, "
          f"최대 대기={manager.peak_waiting}, 도착={manager.completed}, "
          f"실패={manager.failed}, 평균 대기={manager.wait_total/max(1,manager.wait_count):.1f}s")

    return {
        "reward": total_reward,
        "avg_reward": total_reward / missions if missions else 0.0,
        "golden_rate": 100.0 * golden_success / missions if missions else 0.0,
        "rejections": 0,
        "avg_time": total_time / missions if missions else 0.0,
        "missions": missions,
        "route_failures": route_failures,
        "load_std": float(np.std(list(assignments.values()))),
        "peak_active": manager.peak_active,
        "peak_waiting": manager.peak_waiting,
        "avg_wait": manager.wait_total / max(1, manager.wait_count),
        "arrived": manager.completed,
        "failed": manager.failed,
        "failure_reasons": dict(manager.failure_reasons),
        "route_diagnostics": dict(route_diagnostics),
        "ktas_stats": manager.ktas_stats,
    }


def print_result(label, result):
    print(
        f"  {label:<10} | "
        f"보상 {result['reward']:9.2f} | "
        f"평균 {result['avg_reward']:7.2f} | "
        f"골든타임 {result['golden_rate']:6.1f}% | "
        f"완료 {result['missions']:3d} | "
        f"경로실패 {result['route_failures']:3d} | "
        f"평균시간 {result['avg_time']:7.1f}s | "
        f"부하편차 {result['load_std']:.2f}"
    )


def main(gui=False, ambulance_count=DEFAULT_AMBULANCES):
    env = SumoMedicalEnvironment()
    base_hospitals = copy.deepcopy(env.get_hospitals())

    state_dim = 3 + len(base_hospitals) * 4
    agent = DQNAmbulanceAgent(state_dim, len(base_hospitals))

    if not agent.load_model(MODEL_PATH):
        print(f"[오류] 모델을 찾을 수 없습니다: {MODEL_PATH}")
        print("먼저 train_agent.py를 실행하세요.")
        return

    agent.epsilon = 0.0
    agent.model.eval()

    selected = input("실행할 시나리오 번호(0=전체): ").strip()

    if selected == "0":
        scenarios = SCENARIOS
    else:
        try:
            sid = int(selected)
            scenarios = [s for s in SCENARIOS if s["id"] == sid]
        except ValueError:
            scenarios = []

    if not scenarios:
        print("[오류] 시나리오 번호가 올바르지 않습니다.")
        return

    results_dir = os.path.join("results", "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = new_run_id("eval")
    run_started = time.monotonic()
    append_record(run_id, "evaluation", "run_start", status="started",
                  ambulance_limit=ambulance_count, model_path=MODEL_PATH,
                  metrics={"scenario_ids": [s["id"] for s in scenarios],
                           "gui": gui, "strategies": [label for _, label in STRATEGIES]})
    print(f"[통합 실험 로그] {LOG_PATH} | 실행 ID: {run_id}")

    overall = {mode: [] for mode, _ in STRATEGIES}

    for scenario in scenarios:
        seed = 30000 + scenario["id"]

        print(
            f"\n{'=' * 90}\n"
            f"시나리오 {scenario['id']} 평가: {scenario['name']}\n"
            f"{'=' * 90}"
        )

        for mode, label in STRATEGIES:
            result = run_single(
                env,
                base_hospitals,
                scenario,
                seed,
                mode,
                agent if mode == "dqn" else None,
                gui=gui, ambulance_count=ambulance_count,
            )

            overall[mode].append(result)
            print_result(label, result)

            append_record(
                run_id, "evaluation", "strategy_result",
                scenario_id=scenario["id"], scenario_name=scenario["name"],
                strategy=label, status="completed",
                ambulance_limit=ambulance_count, model_path=MODEL_PATH,
                duration_seconds=round(time.monotonic() - run_started, 2),
                metrics=result
            )

    print("\n" + "=" * 90)
    print("최종 평균 결과")
    print("=" * 90)

    for mode, label in STRATEGIES:
        rs = overall[mode]
        if not rs:
            continue

        line = (
            f"{label:<14} | "
            f"보상 {sum(r['reward'] for r in rs):10.2f} | "
            f"골든타임 {np.mean([r['golden_rate'] for r in rs]):6.1f}% | "
            f"완료 {sum(r['missions'] for r in rs):4d} | "
            f"평균시간 {np.mean([r['avg_time'] for r in rs]):7.1f}s"
        )
        print(line)

        append_record(run_id, "evaluation", "strategy_average", strategy=label,
                      status="completed", ambulance_limit=ambulance_count,
                      model_path=MODEL_PATH,
                      metrics={"reward_sum": sum(r["reward"] for r in rs),
                               "golden_rate_mean": float(np.mean([r["golden_rate"] for r in rs])),
                               "missions_sum": sum(r["missions"] for r in rs),
                               "avg_time_mean": float(np.mean([r["avg_time"] for r in rs]))})

    save_evaluation_graphs(overall, results_dir, timestamp)
    append_record(run_id, "evaluation", "run_end", status="completed",
                  ambulance_limit=ambulance_count, model_path=MODEL_PATH,
                  duration_seconds=round(time.monotonic() - run_started, 2),
                  metrics={"evaluated_scenarios": len(scenarios)})
    print(f"\n통합 실험 로그 CSV: {LOG_PATH}")
    print(f"실행 ID: {run_id}")


def save_evaluation_graphs(overall, results_dir, timestamp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [label for _, label in STRATEGIES]
        rewards = [np.mean([r["avg_reward"] for r in overall[mode]]) if overall[mode] else 0
                   for mode, _ in STRATEGIES]
        golden = [np.mean([r["golden_rate"] for r in overall[mode]]) if overall[mode] else 0
                  for mode, _ in STRATEGIES]
        for values, title, ylabel, suffix in [
            (rewards, "Average Reward by Strategy", "Average reward", "avg_reward"),
            (golden, "Golden-Time Success Rate by Strategy", "Success rate (%)", "golden_rate"),
        ]:
            plt.figure(figsize=(10, 5)); plt.bar(labels, values)
            plt.title(title); plt.ylabel(ylabel); plt.xticks(rotation=20, ha="right")
            plt.grid(axis="y", alpha=0.3); plt.tight_layout()
            out = os.path.join(results_dir, f"{suffix}_{timestamp}.png")
            plt.savefig(out, dpi=150); plt.close(); print(f"그래프 저장: {out}")
    except ImportError:
        print("[안내] 그래프 저장을 위해 'pip install matplotlib'을 실행하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="구급차 경로 전략 평가")
    parser.add_argument("--gui", action="store_true", help="SUMO-GUI로 실행")
    parser.add_argument("--ambulances", type=int, default=DEFAULT_AMBULANCES,
                        help="구급차 대수(최대 10)")
    args = parser.parse_args()
    main(gui=args.gui, ambulance_count=args.ambulances)
