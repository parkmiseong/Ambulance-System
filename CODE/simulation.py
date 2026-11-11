"""One shared episode engine for training, validation, and baseline evaluation."""
from collections import Counter
import copy
from pathlib import Path
import random
import time

import numpy as np
import traci

from ambulance_sim import DEFERRED, MissionManager
from baseline import HospitalRouters, build_state_vector
from episode_transitions import EpisodeTransitions
from experiment_config import generate_schedule, seed_everything
from hospital_capacity import HospitalCapacity
from hospital_policy import valid_actions, explain_hospital_eligibility
from reward import GOLDEN_TIME_LIMITS, calculate_reward, normalize_ktas, failure_penalty
from simulation_setup import start_sumo, get_reachable_hospital_routes, _hospital_access_edges
from traffic_router import TrafficRouter


def choose(mode, router, patient_pos, ktas, hospitals, state, agent, actions):
    router.hospitals = hospitals
    if mode == "dqn":
        return int(agent.select_action(state, actions))
    if mode == "shortest_distance":
        return router.shortest_distance_strategy(actions)
    if mode == "shortest_time":
        return router.shortest_time_strategy(actions)
    if mode == "bed":
        return router.bed_strategy(ktas, actions)
    if mode == "rule":
        return router.rule_based_strategy(ktas, actions)
    if mode == "heuristic":
        return router.heuristic_strategy(ktas, actions)
    raise ValueError(f"Unknown strategy: {mode}")


def run_episode(env, base_hospitals, scenario, seed, mode, config, agent=None,
                training=False, gui=False, ambulance_count=10, queue_policy="severity",
                aging_seconds=300, output_dir=None, on_outcome=None, quiet=False,
                on_decision=None):
    if training and (mode != "dqn" or agent is None):
        raise ValueError("Training requires a DQN agent.")
    if not 1 <= ambulance_count <= 10:
        raise ValueError("ambulance_count must be in 1..10.")
    seed_everything(seed)
    hospitals = copy.deepcopy(base_hospitals)
    occupancy_rng = random.Random(seed + 200000)
    for hospital in hospitals:
        hospital["occupancy"] = occupancy_rng.uniform(.20, .60)
    beds = HospitalCapacity(hospitals, config)
    schedule = generate_schedule(scenario, env.ambulance_edges, seed, config)
    access = getattr(env, "hospital_access_edges", None)
    if access is None:
        access = _hospital_access_edges(env, hospitals)
    manager = MissionManager(ambulance_count, timeout=config.mission_timeout,
                             max_wait=config.max_wait, queue_policy=queue_policy,
                             aging_seconds=aging_seconds)
    traffic = TrafficRouter(refresh_interval=30, net=env.net)
    router = HospitalRouters(hospitals)
    transitions = EpisodeTransitions(agent) if training else None
    outcomes = []
    diagnostics = Counter()
    rejection_pairs = Counter()
    reachability = {}
    progress_at = time.monotonic()
    selected_times, extra_times, selected_actions = [], [], Counter()

    def dispatch(patient, vid):
        now = float(traci.simulation.getTime())
        beds.refresh(now)
        edge_id = patient["edge_id"]
        ktas = normalize_ktas(patient["ktas"])
        patient_pos = env.net.getEdge(edge_id).getFromNode().getCoord()[:2]
        # If every hospital is currently masked even assuming a route exists,
        # no route costs can affect a decision. Reuse only static reachability
        # for rejection accounting; recompute travel times before any assignment.
        optimistic = [dict(h, reachable=True) for h in hospitals]
        if edge_id in reachability and not valid_actions(optimistic, ktas, config.occupancy_limit):
            flags, failures = reachability[edge_id]
            routed = [dict(h, reachable=flag) for h, flag in zip(hospitals, flags)]
            diagnostics["masked_route_reuses"] += 1
        else:
            routed, failures = get_reachable_hospital_routes(env, edge_id, hospitals, access)
            reachability[edge_id] = ([h.get("reachable", False) for h in routed], failures)
        diagnostics["hospital_candidate_route_failure"] += failures
        actions = valid_actions(routed, ktas, config.occupancy_limit)
        if not actions:
            reasons = [explain_hospital_eligibility(h, ktas, config.occupancy_limit)
                       for h in routed]
            for rejected in reasons:
                rejection_pairs.update(rejected)
            # Retry only temporary bed shortages. A missing route/capability is permanent
            # for this dispatch and must not consume retries until the waiting timeout.
            if any(set(items) <= {"hospital_full_or_invalid_occupancy"} for items in reasons):
                diagnostics["capacity_deferrals"] += 1
                patient["retry_after"] = now + min(30., config.treatment_seconds)
                patient["last_dispatch_reason"] = "hospital_capacity"
                return DEFERRED
            diagnostics["no_valid_hospital"] += 1
            if not any(h.get("reachable", False) for h in routed):
                diagnostics["patient_no_route"] += 1
            return None
        state = build_state_vector(patient_pos, ktas, routed, patient=patient,
                                   manager=manager, now=now, config=config)
        action = choose(mode, router, patient_pos, ktas, routed, state, agent, actions)
        if action not in actions:
            raise ValueError("Strategy selected a masked hospital.")
        fastest = min(actions, key=lambda i: routed[i]["estimated_travel_time"])
        selected_time = float(routed[action]["estimated_travel_time"])
        extra = selected_time - float(routed[fastest]["estimated_travel_time"])
        selected_times.append(selected_time)
        extra_times.append(extra)
        selected_actions[action] += 1
        if on_decision is not None:
            q_values = agent.q_values(state) if mode == "dqn" else None
            on_decision({
                "patient_id": patient["patient_id"], "ktas": ktas, "time": now,
                "wait_seconds": now - patient["occurrence_time"],
                "selected_action": action, "fastest_action": fastest,
                "extra_estimated_seconds": extra,
                "exploratory": bool(getattr(agent, "last_action_exploratory", False)) if mode == "dqn" else False,
                "q_reward_scale": getattr(agent, "reward_scale", 1.) if mode == "dqn" else None,
                "reward_version": config.reward_version,
                "observation_version": config.observation_version,
                "pending_reward_feature": config.pending_reward_feature,
                "state": state.tolist(),
                "candidates": [dict(
                    action=i, hospital_id=h["id"], valid=i in actions,
                    rejections=explain_hospital_eligibility(h, ktas, config.occupancy_limit),
                    estimated_travel_time=h["estimated_travel_time"],
                    occupancy=h["occupancy"],
                    q_value=float(q_values[i]) if q_values is not None else None,
                    # A diagnostic estimate, not a simulated counterfactual return.
                    estimated_patient_reward=calculate_reward(
                        h["estimated_travel_time"] + now - patient["occurrence_time"],
                        ktas=ktas, hospital=h, occupancy_at_selection=h["occupancy"],
                        reward_version=config.reward_version) if i in actions else None,
                ) for i, h in enumerate(routed)],
            })
        selected = copy.deepcopy(routed[action])
        occupancy = float(selected["occupancy"])
        selected["estimated_total_seconds_at_dispatch"] = selected_time + now - patient["occurrence_time"]
        selected["estimated_reward_at_dispatch"] = calculate_reward(
            selected["estimated_total_seconds_at_dispatch"], ktas=ktas,
            pre_ktas=patient.get("pre_ktas", ktas), hospital=selected,
            occupancy_at_selection=occupancy, reward_version=config.reward_version)
        beds.reserve(action, vid)
        try:
            if transitions is not None:
                transitions.begin(state, action, actions)
        except Exception:
            beds.complete(vid, False, now)
            raise
        return selected["hospital_edge"], (action, selected, occupancy, vid)

    def finish(patient, payload, arrived, elapsed, wait, reason):
        now = float(traci.simulation.getTime())
        censored = reason.startswith("horizon")
        ktas = normalize_ktas(patient["ktas"])
        action = -1
        predicted_reward = predicted_total = None
        if payload is None:
            reward = 0. if censored else -failure_penalty(config.reward_version)
        else:
            action, selected, occupancy, reservation = payload
            predicted_reward = selected.get("estimated_reward_at_dispatch")
            predicted_total = selected.get("estimated_total_seconds_at_dispatch")
            beds.complete(reservation, arrived, now)
            reward = 0. if censored else calculate_reward(
                travel_time=elapsed, ktas=ktas, pre_ktas=patient.get("pre_ktas", ktas),
                hospital=selected, occupancy_at_selection=occupancy,
                route_failed=not arrived, golden_limit=GOLDEN_TIME_LIMITS[ktas],
                reward_version=config.reward_version)
        if transitions is not None:
            transitions.add_reward(reward)
        row = {
            "patient_id": patient["patient_id"], "ktas": ktas,
            "pre_ktas": patient.get("pre_ktas", ktas),
            "occurrence_time": patient["occurrence_time"], "wait_seconds": wait,
            "dispatch_time": patient["occurrence_time"] + wait if payload is not None else None,
            "finish_time": now, "total_seconds": elapsed,
            "travel_seconds": elapsed - wait if payload is not None else None,
            "arrived": bool(arrived), "censored": censored, "reason": reason,
            "hospital_action": action,
            "hospital_id": hospitals[action]["id"] if action >= 0 else None,
            "reward": reward,
            "golden_success": bool(arrived and elapsed <= GOLDEN_TIME_LIMITS[ktas]),
            "last_dispatch_reason": patient.get("last_dispatch_reason"),
            "reward_version": config.reward_version,
            "estimated_reward_at_dispatch": predicted_reward,
            "estimated_total_seconds_at_dispatch": predicted_total,
        }
        outcomes.append(row)
        if on_outcome:
            on_outcome(row)

    try:
        start_sumo(env, scenario, seed, gui=gui, output_dir=output_dir, quiet=quiet)
        traci.simulationStep()  # The first scheduled occurrence is SUMO time 1.
        traffic.reset()
        while traci.simulation.getTime() <= config.drain_deadline:
            now = float(traci.simulation.getTime())
            beds.refresh(now)
            traffic.refresh()
            if int(now) in schedule:
                manager.enqueue(dict(schedule[int(now)]))
            manager.step(dispatch, finish, prefix="amb_e_" if training else "amb_eval_")
            if time.monotonic() - progress_at >= 60:
                print(f"[Progress] {Path(output_dir).name if output_dir else mode} | "
                      f"sim={now:.0f}s | waiting={len(manager.waiting)} "
                      f"active={len(manager.active)} completed={len(outcomes)}/{len(schedule)}",
                      flush=True)
                progress_at = time.monotonic()
            if now >= config.arrival_steps and not manager.active and not manager.waiting:
                break
        manager.close_unfinished(finish)
        if transitions is not None:
            transitions.close()
        end_time = float(traci.simulation.getTime())
    finally:
        if traci.isLoaded():
            traci.close()
    generated = len(schedule)
    if len(outcomes) != generated:
        raise RuntimeError(f"Outcome accounting mismatch: {len(outcomes)} != {generated}")
    reward = sum(row["reward"] for row in outcomes)
    arrived = sum(row["arrived"] for row in outcomes)
    censored = sum(row["censored"] for row in outcomes)
    golden = sum(row["golden_success"] for row in outcomes)
    arrivals = [row for row in outcomes if row["arrived"]]
    assignments = Counter(row["hospital_action"] for row in outcomes if row["hospital_action"] >= 0)
    elapsed = [row["total_seconds"] for row in outcomes]
    waits = [row["wait_seconds"] for row in outcomes]
    ktas_stats = {}
    for k in range(1, 6):
        group = [row for row in outcomes if row["ktas"] == k]
        ktas_stats[k] = {
            "count": len(group), "arrived": sum(row["arrived"] for row in group),
            "failed": sum(not row["arrived"] and not row["censored"] for row in group),
            "censored": sum(row["censored"] for row in group),
            "golden_success": sum(row["golden_success"] for row in group),
            "wait_total": sum(row["wait_seconds"] for row in group),
        }
    return {
        "generated_patients": generated, "missions": sum(assignments.values()),
        "arrived": arrived, "failed": generated - arrived - censored, "censored": censored,
        "arrival_rate": 100. * arrived / generated if generated else 0.,
        "golden_rate": 100. * golden / generated if generated else 0.,
        "golden_success": golden, "reward": reward,
        "avg_reward": reward / generated if generated else 0.,
        "avg_time": float(np.mean(elapsed)) if elapsed else 0.,
        "avg_wait": float(np.mean(waits)) if waits else 0.,
        "p90_wait": float(np.percentile(waits, 90)) if waits else 0.,
        "avg_arrival_time": float(np.mean([r["total_seconds"] for r in arrivals])) if arrivals else None,
        "avg_travel_time": float(np.mean([r["travel_seconds"] for r in arrivals])) if arrivals else None,
        "peak_active": manager.peak_active, "peak_waiting": manager.peak_waiting,
        "queue_policy": queue_policy, "failure_reasons": dict(manager.failure_reasons),
        "censor_reasons": dict(manager.censor_reasons),
        "route_failures": diagnostics["patient_no_route"],
        "route_diagnostics": dict(diagnostics),
        "hospital_rejection_reasons": dict(rejection_pairs),
        "ktas_stats": ktas_stats, "simulation_end_time": end_time,
        "load_std": float(np.std([assignments[i] for i in range(len(hospitals))])),
        "peak_hospital_occupancy": beds.peak_occupancy,
        "remaining_reservations": len(beds.reservations),
        "learning_reward": transitions.total_reward if transitions else reward,
        "unattributed_learning_reward": transitions.unattributed_reward if transitions else 0.,
        "learning_transitions": transitions.count if transitions else 0,
        "avg_loss": float(np.mean(transitions.losses)) if transitions and transitions.losses else 0.,
        "seed": seed, "demand_scale": config.demand_scale,
        "reward_version": config.reward_version,
        "observation_version": config.observation_version,
        "pending_reward_feature": config.pending_reward_feature,
        "selected_hospitals": dict(selected_actions),
        "avg_selected_estimated_time": float(np.mean(selected_times)) if selected_times else None,
        "avg_extra_estimated_time": float(np.mean(extra_times)) if extra_times else None,
    }
