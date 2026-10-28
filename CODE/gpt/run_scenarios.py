# -*- coding: utf-8 -*-
# run_scenarios.py
"""KTAS DQN + 5개 baseline 평가."""

import contextlib
import copy
import csv
import datetime
import math
import os
import random
import sys
import argparse
import time
import json
from collections import Counter
from experiment_log import append_record, new_run_id, LOG_PATH

import numpy as np
import sumolib
import traci

from baseline import HospitalRouters, build_state_vector
from dqn_agent import DQNAmbulanceAgent
from environment import SumoMedicalEnvironment
from reward import GOLDEN_TIME_LIMITS, calculate_reward, normalize_ktas, normalize_hospital_type, reward_components, get_hospital_suitability, REWARD_PROFILE
from ambulance_sim import MissionManager


MAX_STEPS = 600
MAX_AMBULANCES = 10
DEFAULT_AMBULANCES = 10
MODEL_PATH = os.path.join("results", "models", REWARD_PROFILE, "dqn_ambulance_model_ktas_v2.pth")
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
                    if not math.isfinite(length) or length < 0:
                        continue
                    if not math.isfinite(travel):
                        continue
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


def run_single(env, base_hospitals, scenario, seed, mode, agent=None, gui=False, ambulance_count=DEFAULT_AMBULANCES, queue_policy="severity", aging_seconds=300):
    ambulance_count = max(1, min(MAX_AMBULANCES, int(ambulance_count)))
    random.seed(seed)
    np.random.seed(seed)

    hospitals = copy.deepcopy(base_hospitals)
    for h in hospitals:
        normalize_hospital_type(h.get("type"))
    for h in hospitals:
        h["occupancy"] = random.uniform(0.20, 0.60)

    patients = None  # SUMO 시작 후 실제 ambulance_custom 출발 가능 Edge로 생성

    router = HospitalRouters(hospitals)

    total_reward = 0.0
    reward_breakdown = Counter()
    hospital_type_assignments = Counter()
    total_time = 0.0
    golden_success = 0
    missions = 0
    route_failures = 0
    route_diagnostics = Counter()
    assignments = {h.get("id", str(i)): 0 for i, h in enumerate(hospitals)}

    manager = MissionManager(max_ambulances=ambulance_count, timeout=1800,
                             max_wait=1800, verbose=False, queue_policy=queue_policy, aging_seconds=aging_seconds)
    outcome_rows = []
    patient_records = []
    candidate_reachable_counts = []
    hospital_diagnostics = [dict(hospital_index=i, hospital_id=str(h.get('id', i)),
                                 hospital_type=str(h.get('type', '')),
                                 checked=0, reachable=0, valid_action=0, selected=0)
                            for i, h in enumerate(hospitals)]
    # Each observation is taken BEFORE one MissionManager.step (one SUMO second).
    active_vehicle_seconds = 0
    full_capacity_seconds = 0
    queued_patient_seconds = 0
    observed_seconds = 0
    dispatched_service_seconds = 0.0
    completed_dispatches = 0
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
        candidate_reachable_counts.append(sum(bool(h.get("reachable")) for h in routed))
        actions = valid_actions(routed)
        for i, h in enumerate(routed):
            stat = hospital_diagnostics[i]
            stat['checked'] += 1
            stat['reachable'] += int(bool(h.get('reachable', False)))
            stat['valid_action'] += int(i in actions)
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
        hospital_diagnostics[action]['selected'] += 1
        # 선택 가능한 병원 집합에서의 반사실적 최단시간 대안.
        # 대기열에서 dispatch가 호출되는 시점의 병원 상태를 기록한다.
        alternatives = [(i, float(routed[i].get("estimated_travel_time", MAX_ESTIMATED_TRAVEL_TIME)))
                        for i in actions]
        fastest_idx, fastest_time = min(alternatives, key=lambda pair: pair[1])
        patient["_dispatch_time"] = float(traci.simulation.getTime())
        patient["_candidate_count"] = len(actions)
        patient["_fastest_hospital_id"] = str(routed[fastest_idx].get("id", fastest_idx))
        patient["_fastest_hospital_type"] = str(routed[fastest_idx].get("type", "unknown"))
        patient["_fastest_estimated_time"] = fastest_time
        patient["_selected_time_rank"] = 1 + sum(t < float(selected["estimated_travel_time"]) - 1e-6
                                                  for _, t in alternatives)
        patient["_extra_estimated_time"] = max(0.0, float(selected["estimated_travel_time"]) - fastest_time)
        patient["_selected_hospital_type"] = str(selected.get("type", "unknown"))
        patient["_selected_hospital_id"] = str(selected.get("id", action))
        patient["_selected_estimated_time"] = selected.get("estimated_travel_time")
        patient["_selected_occupancy"] = float(selected.get("occupancy", 0.0))
        # 선택 가능한 후보에 한정한 연구용 병원 적합도 진단.
        fits = [(i, get_hospital_suitability(ktas, routed[i])) for i in actions]
        best_fit_idx, best_fit = max(fits, key=lambda pair: pair[1])
        patient["_selected_suitability"] = get_hospital_suitability(ktas, selected)
        patient["_best_candidate_suitability"] = best_fit
        patient["_best_fit_hospital_id"] = str(routed[best_fit_idx].get("id", best_fit_idx))
        patient["_suitability_gap"] = max(0.0, best_fit - patient["_selected_suitability"])
        return selected['hospital_edge'], (action, selected, float(selected.get('occupancy', 0.)))

    def finish(patient, payload, arrived, elapsed, wait, reason):
        nonlocal dispatched_service_seconds, completed_dispatches
        if payload is not None:
            dispatched_service_seconds += max(0.0, float(elapsed) - float(wait))
            completed_dispatches += 1
        # finish 호출 시점은 임무 종료 시각이며 구급차 재가용 시각과 같다고 보장할 수 없다.
        patient_records.append({
            "occurrence_time": patient.get("occurrence_time", ""),
            "pre_ktas": patient.get("pre_ktas", ""),
            "ktas": patient.get("ktas", ""),
            "patient_edge": patient.get("edge_id", ""),
            "hospital_id": (str(payload[1].get("id", payload[0])) if payload is not None else ""),
            "hospital_type": (str(payload[1].get("type", "unknown")) if payload is not None else ""),
            "estimated_travel_time_sec": (payload[1].get("estimated_travel_time", "") if payload is not None else ""),
            "occupancy_at_selection": (payload[2] if payload is not None else ""),
            "arrived": int(bool(arrived)),
            "wait_sec": wait,
            "elapsed_sec": elapsed,
            "post_dispatch_sec": (max(0.0, float(elapsed) - float(wait)) if payload is not None else ""),
            "failure_reason": reason or "",
            "was_dispatched": int(payload is not None),
            "selected_hospital_id_at_dispatch": patient.get("_selected_hospital_id", ""),
            "dispatch_time": patient.get("_dispatch_time", ""),
            "finish_callback_time": (float(traci.simulation.getTime()) if traci.isLoaded() else ""),
            "candidate_count": patient.get("_candidate_count", ""),
            "fastest_hospital_id": patient.get("_fastest_hospital_id", ""),
            "fastest_hospital_type": patient.get("_fastest_hospital_type", ""),
            "fastest_estimated_time_sec": patient.get("_fastest_estimated_time", ""),
            "selected_time_rank": patient.get("_selected_time_rank", ""),
            "extra_estimated_time_sec": patient.get("_extra_estimated_time", ""),
            "selected_suitability": patient.get("_selected_suitability", ""),
            "best_candidate_suitability": patient.get("_best_candidate_suitability", ""),
            "best_fit_hospital_id": patient.get("_best_fit_hospital_id", ""),
            "suitability_gap": patient.get("_suitability_gap", ""),
        })
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
        if not math.isfinite(float(elapsed)) or elapsed < 0:
            raise ValueError(f"평가 이송 경과시간 오류: {elapsed!r}, reason={reason!r}")
        reward = calculate_reward(
            travel_time=max(0.0, elapsed - wait), ktas=ktas,
            pre_ktas=patient.get('pre_ktas', ktas),
            hospital=selected, golden_limit=GOLDEN_TIME_LIMITS[ktas],
            occupancy_at_selection=occupancy, route_failed=failed)
        parts = reward_components(
            travel_time=max(0.0, elapsed - wait), ktas=ktas,
            pre_ktas=patient.get('pre_ktas', ktas),
            hospital=selected, golden_limit=GOLDEN_TIME_LIMITS[ktas],
            occupancy_at_selection=occupancy, route_failed=failed)
        reward_breakdown.update(parts)
        hospital_type_assignments[str(selected.get("type", "unknown"))] += 1
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
        # 학습과 동일한 검증된 출발 Edge 후보군 사용 (실제 TraCI 차량 타입 검사)
        from train_agent import build_ambulance_edges
        ambulance_edges = build_ambulance_edges(env, max_check=1200)
        patients = schedule(scenario, ambulance_edges, seed)
        traci.simulationStep()  # time=1: 이벤트 시각과 실제 출동 시각 정렬
        for step in range(1, MAX_STEPS + 1800 + 1):
            if step <= MAX_STEPS and step in patients:
                patient = dict(patients[step])
                patient['occurrence_time'] = float(step)
                manager.enqueue(patient)
            # Sample capacity and queue BEFORE the step; does not advance SUMO.
            active_vehicle_seconds += len(manager.active)
            full_capacity_seconds += int(len(manager.active) >= manager.limit)
            queued_patient_seconds += len(manager.waiting)
            observed_seconds += 1
            manager.step(dispatch, finish, prefix=f'amb_eval_{mode}_{scenario["id"]}_')
            if step >= MAX_STEPS and not manager.active and not manager.waiting:
                break
        manager.close_unfinished(finish)
    finally:
        try:
            traci.close()
        except Exception:
            pass
    if candidate_reachable_counts:
        print(f"[병원 후보 진단] {mode} 검사 환자={len(candidate_reachable_counts)}, "
              f"도달 가능 병원 평균={np.mean(candidate_reachable_counts):.2f}/10, "
              f"0개={sum(v == 0 for v in candidate_reachable_counts)}, "
              f"1~3개={sum(1 <= v <= 3 for v in candidate_reachable_counts)}, "
              f"4~7개={sum(4 <= v <= 7 for v in candidate_reachable_counts)}, "
              f"8~10개={sum(8 <= v <= 10 for v in candidate_reachable_counts)}")
    utilization = active_vehicle_seconds / (ambulance_count * observed_seconds) if observed_seconds else 0.0
    print(f"[가용성 진단] {mode} 차량점유율={100*utilization:.1f}% "
          f"전대운행시간비율={100*full_capacity_seconds/max(1,observed_seconds):.1f}% "
          f"평균대기열={queued_patient_seconds/max(1,observed_seconds):.1f}명 "
          f"출동임무평균점유시간={dispatched_service_seconds/max(1,completed_dispatches):.1f}s")
    print(f"[병원별 도달성] {mode} " + ', '.join(
        f"{h['hospital_index']}:{h['reachable']}/{h['checked']}"
        for h in hospital_diagnostics))
    print(f"[대기열] {mode} 최대 운행={manager.peak_active}/{ambulance_count}, "
          f"최대 대기={manager.peak_waiting}, 도착={manager.completed}, "
          f"실패={manager.failed}, 평균 대기={manager.wait_total/max(1,manager.wait_count):.1f}s")

    return {
        "patient_records": patient_records,
        "hospital_diagnostics": hospital_diagnostics,
        "vehicle_utilization_pct": 100.0 * utilization,
        "full_capacity_pct": 100.0 * full_capacity_seconds / max(1, observed_seconds),
        "mean_queue_length": queued_patient_seconds / max(1, observed_seconds),
        "mean_dispatch_service_sec": dispatched_service_seconds / max(1, completed_dispatches),
        "reward": total_reward,
        "reward_profile": REWARD_PROFILE,
        "reward_components_sum": dict(reward_breakdown),
        "hospital_type_assignments": dict(hospital_type_assignments),
        "avg_reward": total_reward / missions if missions else 0.0,
        "golden_rate": 100.0 * golden_success / missions if missions else 0.0,
        "rejections": 0,
        "avg_time": total_time / missions if missions else 0.0,
        "missions": missions,
        "route_failures": route_failures,
        "load_std": float(np.std(list(assignments.values()))),
        "queue_policy": queue_policy,
        "peak_active": manager.peak_active,
        "peak_waiting": manager.peak_waiting,
        "avg_wait": manager.wait_total / max(1, manager.wait_count),
        "arrived": manager.completed,
        "failed": manager.failed,
        "failure_reasons": dict(manager.failure_reasons),
        "route_diagnostics": dict(route_diagnostics),
        "candidate_reachable_mean": float(np.mean(candidate_reachable_counts)) if candidate_reachable_counts else None,
        "candidate_reachable_zero": sum(v == 0 for v in candidate_reachable_counts),
        "ktas_stats": manager.ktas_stats,
        "golden_success": golden_success,
        "patients_total": manager.completed + manager.failed,
        "arrival_rate": 100.0 * manager.completed / max(1, manager.completed + manager.failed),
        "failure_rate": 100.0 * manager.failed / max(1, manager.completed + manager.failed),
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


PROFILES = ("baseline", "strong_mismatch", "balanced", "travel_guard", "fit_guard")
MODEL_FILENAME = "dqn_ambulance_model_ktas_v2.pth"


def load_agents(hospitals, profiles):
    state_dim = 3 + len(hospitals) * 4
    agents = {}
    for profile in profiles:
        path = os.path.join("results", "models", profile, MODEL_FILENAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[{profile}] 모델이 없습니다: {path}")
        agent = DQNAmbulanceAgent(state_dim, len(hospitals))
        if not agent.load_model(path):
            raise RuntimeError(f"[{profile}] 모델 로딩 실패: {path}")
        agent.epsilon = 0.0
        agent.model.eval()
        agents[profile] = agent
        print(f"[모델 로딩] {profile}: {path}")
    return agents


def flatten_result(scenario, repeat, seed, strategy, result):
    total = result["arrived"] + result["failed"]
    row = {
        "scenario_id": scenario["id"], "scenario_name": scenario["name"],
        "repeat": repeat + 1, "seed": seed, "strategy": strategy,
        "arrived": result["arrived"], "failed": result["failed"],
        "patients_total": total,
        "missions": result["missions"],
        "arrival_rate_pct": 100.0 * result["arrived"] / total if total else 0.0,
        "failure_rate_pct": 100.0 * result["failed"] / total if total else 0.0,
        "golden_success": result.get("golden_success", 0),
        "golden_rate_pct": result["golden_rate"],
        "avg_wait_sec": result["avg_wait"],
        "avg_mission_sec": result["avg_time"],
        "peak_waiting": result["peak_waiting"],
        "route_failures": result["route_failures"],
        "vehicle_utilization_pct": result.get("vehicle_utilization_pct"),
        "full_capacity_pct": result.get("full_capacity_pct"),
        "mean_queue_length": result.get("mean_queue_length"),
        "mean_dispatch_service_sec": result.get("mean_dispatch_service_sec"),
        "hospital_load_std": result["load_std"],
        "evaluation_reward_profile": REWARD_PROFILE,
        "reward_sum": result["reward"],
        "failure_reasons_json": json.dumps(result["failure_reasons"], ensure_ascii=False),
        "route_diagnostics_json": json.dumps(result["route_diagnostics"], ensure_ascii=False),
        "candidate_reachable_mean": result.get("candidate_reachable_mean"),
        "candidate_reachable_zero": result.get("candidate_reachable_zero"),
        "hospital_types_json": json.dumps(result["hospital_type_assignments"], ensure_ascii=False),
    }
    for ktas in range(1, 6):
        stats = result["ktas_stats"].get(ktas, result["ktas_stats"].get(str(ktas), {}))
        arrived = int(stats.get("arrived", 0))
        failed = int(stats.get("failed", 0))
        row[f"ktas{ktas}_arrived"] = arrived
        row[f"ktas{ktas}_failed"] = failed
        row[f"ktas{ktas}_arrival_rate_pct"] = 100.0 * arrived / (arrived + failed) if arrived + failed else ""
    return row


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV 저장] {path}")


def make_summary(rows):
    by_strategy = {}
    for row in rows:
        by_strategy.setdefault(row["strategy"], []).append(row)
    summary = []
    for strategy, group in by_strategy.items():
        total = sum(x["patients_total"] for x in group)
        arrived = sum(x["arrived"] for x in group)
        failed = sum(x["failed"] for x in group)
        # 골든타임은 실제 평가된 유효 임무(missions) 기준으로 집계한다.
        golden_success = sum(x["golden_success"] for x in group)
        missions = sum(x["missions"] for x in group) if "missions" in group[0] else 0
        record = {
            "strategy": strategy, "runs": len(group), "patients_total": total,
            "arrived": arrived, "failed": failed,
            "arrival_rate_pct": 100.0 * arrived / total if total else 0.0,
            "failure_rate_pct": 100.0 * failed / total if total else 0.0,
            "golden_rate_pct": 100.0 * golden_success / missions if missions else 0.0,
            "avg_wait_sec": float(np.mean([x["avg_wait_sec"] for x in group])),
            "avg_mission_sec": float(np.mean([x["avg_mission_sec"] for x in group])),
            "peak_waiting_mean": float(np.mean([x["peak_waiting"] for x in group])),
        }
        for ktas in range(1, 6):
            a = sum(x[f"ktas{ktas}_arrived"] for x in group)
            f = sum(x[f"ktas{ktas}_failed"] for x in group)
            record[f"ktas{ktas}_arrival_rate_pct"] = 100.0 * a / (a + f) if a + f else ""
        summary.append(record)
    return summary


def save_comparison_graphs(summary, output_dir, timestamp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("[안내] 그래프를 저장하려면 pip install matplotlib")
        return
    labels = [r["strategy"] for r in summary]
    for key, title, ylabel in [
        ("arrival_rate_pct", "Patient Arrival Rate", "Arrival rate (%)"),
        ("golden_rate_pct", "Golden-Time Success Rate", "Success rate (%)"),
        ("avg_wait_sec", "Average Queue Waiting Time", "Seconds"),
    ]:
        plt.figure(figsize=(12, 5))
        plt.bar(labels, [r[key] for r in summary])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        path = os.path.join(output_dir, f"comparison_{key}_{timestamp}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[그래프 저장] {path}")


def main(gui=False, ambulance_count=DEFAULT_AMBULANCES, scenario_ids=None,
         repeats=1, profiles=None, seed_base=30000, queue_policy="severity",
         aging_seconds=300):
    if repeats < 1:
        raise ValueError("--repeats는 1 이상이어야 합니다")
    profiles = list(PROFILES if profiles is None else profiles)
    env = SumoMedicalEnvironment()
    base_hospitals = copy.deepcopy(env.get_hospitals())
    for hospital in base_hospitals:
        normalize_hospital_type(hospital.get("type"))
    agents = load_agents(base_hospitals, profiles)
    scenarios = [s for s in SCENARIOS if scenario_ids is None or s["id"] in scenario_ids]
    if not scenarios:
        raise ValueError("선택된 시나리오가 없습니다")
    if gui and (len(scenarios) * repeats * (len(agents) + 5) > 1):
        print("[주의] GUI 모드에서는 SUMO 창이 실험마다 실행됩니다")
    results_dir = os.path.join("results", "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = new_run_id("compare")
    started = time.monotonic()
    all_rows = []
    all_patients = []
    all_hospitals = []
    modes = [(mode, label, None) for mode, label in STRATEGIES if mode != "dqn"]
    modes += [("dqn", f"dqn_{profile}", agents[profile]) for profile in profiles]
    print(f"[비교 실험] {len(scenarios)}개 시나리오 × {repeats}회 × {len(modes)}개 전략")
    print(f"[공통 평가 보상 프로필] {REWARD_PROFILE} (모델 학습 프로필과 별개)")
    append_record(run_id, "evaluation", "run_start", status="started",
                  ambulance_limit=ambulance_count,
                  metrics={"scenarios": [s["id"] for s in scenarios],
                           "repeats": repeats, "profiles": profiles,
                           "seed_base": seed_base, "evaluation_reward_profile": REWARD_PROFILE})
    for scenario in scenarios:
        for repeat in range(repeats):
            seed = seed_base + scenario["id"] + repeat * 1000
            print(f"\n{'=' * 76}\n시나리오 {scenario['id']} ({scenario['name']}) "
                  f"반복 {repeat+1}/{repeats} | seed={seed}\n{'=' * 76}")
            for mode, label, agent in modes:
                result = run_single(env, base_hospitals, scenario, seed, mode,
                                    agent=agent, gui=gui, ambulance_count=ambulance_count,
                                    queue_policy=queue_policy, aging_seconds=aging_seconds)
                print_result(label, result)
                for h in result.pop('hospital_diagnostics', []):
                    all_hospitals.append({'scenario_id': scenario['id'],
                                          'repeat': repeat + 1, 'seed': seed,
                                          'strategy': label, **h})
                write_csv(os.path.join(results_dir, f"hospital_reachability_{timestamp}.csv"), all_hospitals)
                for patient_row in result.pop("patient_records", []):
                    all_patients.append({"scenario_id": scenario["id"],
                                         "repeat": repeat + 1, "seed": seed,
                                         "strategy": label, **patient_row})
                write_csv(os.path.join(results_dir, f"comparison_patients_{timestamp}.csv"), all_patients)
                row = flatten_result(scenario, repeat, seed, label, result)
                all_rows.append(row)
                append_record(run_id, "evaluation", "strategy_result",
                              scenario_id=scenario["id"], scenario_name=scenario["name"],
                              strategy=label, status="completed",
                              ambulance_limit=ambulance_count,
                              model_path=os.path.join("results", "models", label[4:], MODEL_FILENAME)
                              if label.startswith("dqn_") else "",
                              duration_seconds=round(time.monotonic() - started, 2),
                              metrics=result)
                # 중간에 실행이 중단되어도 완료된 평가를 보존한다.
                write_csv(os.path.join(results_dir, f"comparison_detail_{timestamp}.csv"), all_rows)
    summary = make_summary(all_rows)
    # 환자 단위 선택 기록으로 별도 진단 지표 집계 (미출동 환자 제외).
    for record in summary:
        selected = [p for p in all_patients if p["strategy"] == record["strategy"]
                    and p.get("selected_suitability", "") != ""]
        severe = [p for p in selected if int(p["pre_ktas"]) in (1, 2)]
        record["dispatched_count"] = len(selected)
        record["mean_selected_suitability"] = (float(np.mean([float(p["selected_suitability"]) for p in selected])) if selected else "")
        record["mean_suitability_gap"] = (float(np.mean([float(p["suitability_gap"]) for p in selected])) if selected else "")
        record["mean_extra_estimated_time_sec"] = (float(np.mean([float(p["extra_estimated_time_sec"]) for p in selected])) if selected else "")
        record["severe_dispatched_count"] = len(severe)
        record["severe_low_fit_count"] = sum(float(p["selected_suitability"]) < 0.5 for p in severe)
        record["severe_low_fit_pct"] = (100.0 * record["severe_low_fit_count"] / len(severe) if severe else "")
    write_csv(os.path.join(results_dir, f"comparison_summary_{timestamp}.csv"), summary)
    print("\n" + "=" * 76 + "\n최종 비교 (도착률 / 골든타임 / 평균 대기)\n" + "=" * 76)
    for row in summary:
        print(f"{row['strategy']:<23} | {row['arrival_rate_pct']:5.1f}% | "
              f"{row['golden_rate_pct']:5.1f}% | {row['avg_wait_sec']:7.1f}s")
    save_comparison_graphs(summary, results_dir, timestamp)
    append_record(run_id, "evaluation", "run_end", status="completed",
                  ambulance_limit=ambulance_count,
                  duration_seconds=round(time.monotonic() - started, 2),
                  metrics={"rows": len(all_rows), "summary": summary})
    print(f"\n[완료] 실행 ID: {run_id} | 결과 폴더: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KTAS 구급차 병원배정 전략 및 DQN 보상 프로필 비교")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--ambulances", type=int, default=DEFAULT_AMBULANCES)
    parser.add_argument("--scenarios", nargs="+", type=int, choices=range(1, 10),
                        metavar="ID", help="시나리오 번호 1~9 (기본: 전체)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--seed-base", type=int, default=30000)
    parser.add_argument("--queue-policy", choices=["fifo", "severity", "hybrid"], default="severity")
    parser.add_argument("--aging-seconds", type=int, default=300)
    args = parser.parse_args()
    main(gui=args.gui, ambulance_count=args.ambulances,
         scenario_ids=args.scenarios, repeats=args.repeats,
         profiles=args.profiles, seed_base=args.seed_base,
         queue_policy=args.queue_policy, aging_seconds=args.aging_seconds)
