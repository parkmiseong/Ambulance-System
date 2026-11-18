"""One shared episode engine for training, validation, and baseline evaluation."""
from collections import Counter
import copy
from pathlib import Path
import random
import time

import numpy as np
import traci

from ambulance_sim import DEFERRED, MissionManager, hold_at_hospital
from baseline import HospitalRouters, build_state_vector
from episode_transitions import EpisodeTransitions
from experiment_config import generate_schedule, seed_everything, scenario_for_config
from hospital_capacity import HospitalCapacity
from hospital_policy import valid_actions, explain_hospital_eligibility, request_actions, ADMISSION_REASONS
from reward import (GOLDEN_TIME_LIMITS, calculate_reward, normalize_ktas, failure_penalty,
                    reward_components, get_hospital_suitability, REWARD_COMPONENT_NAMES, WEIGHTS)
from simulation_setup import start_sumo, get_reachable_hospital_routes, _hospital_access_edges
from traffic_router import TrafficRouter
from emergency_yield import yield_settings
from dashboard_traffic import background_settings
from sumo_diagnostics import diagnostic_step, record_sumo_failure, close_sumo_safely


def choose(mode, router, patient_pos, ktas, hospitals, state, agent, actions, *,
           wait_seconds=0., reward_version=4, pre_ktas=None, congestion_reward_mode="penalty"):
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
    if mode == "greedy_reward":
        return router.expected_reward_strategy(ktas, actions, wait_seconds, reward_version, pre_ktas,
                                                congestion_reward_mode)
    raise ValueError(f"Unknown strategy: {mode}")


def run_episode(env, base_hospitals, scenario, seed, mode, config, agent=None,
                training=False, gui=False, ambulance_count=10, queue_policy="severity",
                aging_seconds=300, output_dir=None, on_outcome=None, quiet=False,
                on_decision=None, on_step=None, initial_occupancy_range=None, emergency_yield=None,
                on_rejection=None):
    if emergency_yield is None:
        emergency_yield = config.emergency_yield
    if training and (mode != "dqn" or agent is None):
        raise ValueError("Training requires a DQN agent.")
    if not 1 <= ambulance_count <= 10:
        raise ValueError("ambulance_count must be in 1..10.")
    occupancy_range = (.20, .60) if initial_occupancy_range is None else initial_occupancy_range
    if (len(occupancy_range) != 2
            or not 0 <= occupancy_range[0] <= occupancy_range[1] <= 1):
        raise ValueError("initial_occupancy_range must satisfy 0 <= low <= high <= 1.")
    seed_everything(seed)
    scenario = dict(scenario_for_config(scenario, config))
    hospitals = copy.deepcopy(base_hospitals)
    occupancy_rng = random.Random(seed + 200000)
    for hospital in hospitals:
        hospital["occupancy"] = occupancy_rng.uniform(*occupancy_range)
    beds = HospitalCapacity(hospitals, config)
    access = getattr(env, "hospital_access_edges", None)
    if access is None:
        access = _hospital_access_edges(env, hospitals)
    route_safety_report = None
    source_edges = env.ambulance_edges
    if config.admission_mode == "arrival":
        from route_safety import prepare_access
        access, prepared = prepare_access(env, hospitals, access)
        if prepared is not None:
            source_edges, route_safety_report = prepared
    schedule = generate_schedule(scenario, source_edges, seed, config)
    manager = MissionManager(ambulance_count, timeout=config.mission_timeout,
                             max_wait=config.max_wait, queue_policy=queue_policy,
                             aging_seconds=aging_seconds)
    traffic = TrafficRouter(refresh_interval=30, net=env.net)
    router = HospitalRouters(hospitals)
    transitions = EpisodeTransitions(agent) if training else None
    updates_before = getattr(agent, "training_updates", 0)
    outcomes = []
    diagnostics = Counter()
    rejection_pairs = Counter()
    admission_reasons = Counter()
    reachability = {}
    progress_at = time.monotonic()
    selected_times, extra_times, selected_actions = [], [], Counter()
    arrival_admission = config.admission_mode == "arrival"

    def selection_actions(patient, routed):
        if config.admission_mode == "request":
            return request_actions(routed, normalize_ktas(patient["ktas"]), config.occupancy_limit,
                                   patient.get("_rejected_hospitals", ()))
        return valid_actions(routed, normalize_ktas(patient["ktas"]), config.occupancy_limit)

    def dispatch(patient, vid, origin_edge=None, current_hospital_id=None):
        now = float(traci.simulation.getTime())
        beds.refresh(now)
        edge_id = origin_edge or patient["edge_id"]
        ktas = normalize_ktas(patient["ktas"])
        patient_pos = (traci.vehicle.getPosition(vid) if origin_edge is not None
                       else env.net.getEdge(edge_id).getFromNode().getCoord()[:2])
        optimistic = [dict(h, reachable=True) for h in hospitals]
        if not arrival_admission and edge_id in reachability and not selection_actions(patient, optimistic):
            flags, failures = reachability[edge_id]
            routed = [dict(h, reachable=flag) for h, flag in zip(hospitals, flags)]
            diagnostics["masked_route_reuses"] += 1
        else:
            route_access = access
            if arrival_admission and origin_edge is not None:
                # A different hospital needs a route beyond the stop already reached.
                route_access = [[edge for edge in group if edge != origin_edge or h["id"] == current_hospital_id]
                                for h, group in zip(hospitals, access)]
            route_options = ({"arrival_safe": True, "source_lane": traci.vehicle.getLaneID(vid) if origin_edge else None}
                             if arrival_admission and hasattr(env.net, "getEdges") else {})
            routed, failures = get_reachable_hospital_routes(env, edge_id, hospitals, route_access, **route_options)
            reachability[edge_id] = ([h.get("reachable", False) for h in routed], failures)
        diagnostics["hospital_candidate_route_failure"] += failures
        while True:
            actions = selection_actions(patient, routed)
            if not actions:
                reasons = [explain_hospital_eligibility(h, ktas, config.occupancy_limit) for h in routed]
                for excluded in reasons:
                    rejection_pairs.update(excluded)
                retryable = ADMISSION_REASONS if config.admission_mode in ("request", "arrival") else {"hospital_full_or_invalid_occupancy"}
                if any(set(items) <= retryable for items in reasons):
                    diagnostics["capacity_deferrals"] += 1
                    patient["retry_after"] = now + min(30., config.treatment_seconds)
                    patient["last_dispatch_reason"] = "hospital_admission_pending" if config.admission_mode == "request" else "hospital_capacity"
                    return DEFERRED
                diagnostics["no_valid_hospital"] += 1
                if not any(h.get("reachable", False) for h in routed):
                    diagnostics["patient_no_route"] += 1
                return None
            state = build_state_vector(patient_pos, ktas, routed, patient=patient,
                                       manager=manager, now=now, config=config)
            action = choose(mode, router, patient_pos, ktas, routed, state, agent, actions,
                            wait_seconds=now - patient["occurrence_time"], reward_version=config.reward_version,
                            pre_ktas=patient.get("pre_ktas", ktas), congestion_reward_mode=config.congestion_reward_mode)
            if action not in actions:
                raise ValueError("Strategy selected a masked hospital.")
            if mode == "dqn":
                diagnostics["exploratory_decisions"] += int(getattr(agent, "last_action_exploratory", False))
                if training and hasattr(agent, "record_decision"):
                    agent.record_decision()
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
                    "hospital_admission_mode": config.admission_mode,
                    "congestion_reward_mode": config.congestion_reward_mode,
                    "observation_version": config.observation_version,
                    "pending_reward_feature": config.pending_reward_feature,
                    "state": state.tolist(),
                    "candidates": [dict(
                        action=i, hospital_id=h["id"], valid=i in actions,
                        rejections=explain_hospital_eligibility(h, ktas, config.occupancy_limit),
                        admission_available=not explain_hospital_eligibility(h, ktas, config.occupancy_limit),
                        estimated_travel_time=h["estimated_travel_time"],
                        occupancy=h["occupancy"],
                        q_value=float(q_values[i]) if q_values is not None else None,
                        estimated_patient_reward=calculate_reward(
                            h["estimated_travel_time"] + now - patient["occurrence_time"],
                            ktas=ktas, hospital=h, occupancy_at_selection=h["occupancy"],
                            rejection_count=len(patient.get("rejection_history", ())),
                            reward_version=config.reward_version,
                            congestion_reward_mode=config.congestion_reward_mode) if i in actions else None,
                    ) for i, h in enumerate(routed)],
                })
            selected = copy.deepcopy(routed[action])
            occupancy = float(selected["occupancy"])
            if arrival_admission:
                # The pre-departure check is information, not an exclusive bed reservation.
                if transitions is not None:
                    transitions.begin(state, action, actions)
                edge = env.net.getEdge(selected["hospital_edge"])
                selected["arrival_stop_pos"] = max(0., float(edge.getLength()) - .5)
                selected.setdefault("arrival_stop_lane", next((i for i, lane in enumerate(edge.getLanes())
                                                      if lane.allows("emergency")), 0))
                selected["leg_dispatch_time"] = now
            elif config.admission_mode == "request":
                if transitions is not None:
                    transitions.begin(state, action, actions)
                diagnostics["admission_requests"] += 1
                patient["_admission_request_count"] = patient.get("_admission_request_count", 0) + 1
                refusal = beds.request_admission(action, vid)
                if refusal:
                    history = patient.setdefault("rejection_history", [])
                    event = dict(patient_id=patient["patient_id"], hospital_id=selected["id"],
                                 hospital_name=selected.get("name", selected["id"]), time=now,
                                 reason=refusal, penalty=-float(WEIGHTS.rejection),
                                 request_number=patient["_admission_request_count"], rejection_number=len(history)+1)
                    history.append(event)
                    patient.setdefault("_rejected_hospitals", set()).add(selected["id"])
                    diagnostics["admission_rejections"] += 1
                    admission_reasons[refusal] += 1
                    if transitions is not None:
                        transitions.add_reward(event["penalty"])
                    if on_rejection is not None:
                        on_rejection(dict(event))
                    continue
            else:
                beds.reserve(action, vid)
                try:
                    if transitions is not None:
                        transitions.begin(state, action, actions)
                except Exception:
                    beds.complete(vid, False, now)
                    raise
            selected["estimated_total_seconds_at_dispatch"] = selected_time + now - patient["occurrence_time"]
            selected["estimated_reward_at_dispatch"] = calculate_reward(
                selected["estimated_total_seconds_at_dispatch"], ktas=ktas,
                pre_ktas=patient.get("pre_ktas", ktas), hospital=selected,
                rejection_count=len(patient.get("rejection_history", ())),
                occupancy_at_selection=occupancy, reward_version=config.reward_version,
                congestion_reward_mode=config.congestion_reward_mode)
            return selected["hospital_edge"], (action, selected, occupancy, vid)

    def on_hospital_arrival(mission, now):
        """Accept at the physical hospital stop, or redirect this same vehicle."""
        beds.refresh(now)
        patient, vid = mission.patient, mission.vehicle_id
        action, selected, _, _ = mission.payload
        if not selected.get("admission_checked"):
            selected["admission_checked"] = True
            diagnostics["admission_requests"] += 1
            patient["_admission_request_count"] = patient.get("_admission_request_count", 0) + 1
            at_arrival = float(hospitals[action]["occupancy"])
            refusal = beds.admit_on_arrival(action, vid, now)
            visit = dict(hospital_id=selected["id"], hospital_name=selected.get("name", selected["id"]),
                         vehicle_id=vid, leg_dispatch_time=selected["leg_dispatch_time"], arrival_time=now,
                         occupancy_at_selection=mission.payload[2], occupancy_at_arrival=at_arrival,
                         edge_id=traci.vehicle.getRoadID(vid), accepted=refusal is None, reason=refusal or "accepted")
            patient.setdefault("hospital_visits", []).append(visit)
            selected["visit"] = visit
            if refusal is None:
                return True
            history = patient.setdefault("rejection_history", [])
            event = dict(patient_id=patient["patient_id"], hospital_id=selected["id"],
                         hospital_name=selected.get("name", selected["id"]), time=now,
                         reason=refusal, penalty=-float(WEIGHTS.rejection), stage="after_arrival",
                         vehicle_id=vid, dispatch_time=mission.dispatch_time,
                         leg_dispatch_time=selected["leg_dispatch_time"], edge_id=visit["edge_id"],
                         occupancy_at_arrival=at_arrival,
                         request_number=patient["_admission_request_count"], rejection_number=len(history) + 1)
            history.append(event)
            diagnostics["admission_rejections"] += 1
            admission_reasons[refusal] += 1
            if transitions is not None:
                transitions.add_reward(event["penalty"])
            if on_rejection is not None:
                on_rejection(dict(event))
        elif hospitals[action].get("can_admit") and hospitals[action].get("accepting", True) is True:
            # Waiting at a refused hospital never creates repeated refusal penalties.
            diagnostics["admission_requests"] += 1
            if beds.admit_on_arrival(action, vid, now) is None:
                selected["visit"].update(accepted=True, accepted_time=now, reason="accepted_after_wait")
                return True
        if patient.get("retry_after", 0.) > now:
            return None
        assignment = dispatch(patient, vid, origin_edge=traci.vehicle.getRoadID(vid),
                              current_hospital_id=selected["id"])
        if assignment is DEFERRED:
            return None
        if assignment is None:
            return "no_route_after_rejection"
        hospital_edge, payload = assignment
        mission.payload = payload
        mission.estimated_reward = float(payload[1].get("estimated_reward_at_dispatch", 0.))
        try:
            traci.vehicle.resume(vid)
            traci.vehicle.setRoute(vid, payload[1]["route_edges"])
            hold_at_hospital(vid, hospital_edge, payload[1]["arrival_stop_pos"], payload[1]["arrival_stop_lane"])
        except traci.TraCIException:
            return "redirect_failed"
        diagnostics["hospital_redirects"] += 1
        return None

    def finish(patient, payload, arrived, elapsed, wait, reason):
        now = float(traci.simulation.getTime())
        censored = reason.startswith("horizon")
        ktas = normalize_ktas(patient["ktas"])
        action = -1
        predicted_reward = predicted_total = None
        parts = dict.fromkeys(REWARD_COMPONENT_NAMES, 0.)
        fit = None
        history = copy.deepcopy(patient.get("rejection_history", []))
        rejection_count = len(history)
        if payload is None:
            parts["route_failure"] = 0. if censored else -failure_penalty(config.reward_version)
            parts["rejection"] = -WEIGHTS.rejection * rejection_count
        else:
            action, selected, occupancy, reservation = payload
            predicted_reward = selected.get("estimated_reward_at_dispatch")
            predicted_total = selected.get("estimated_total_seconds_at_dispatch")
            if not arrival_admission:
                beds.complete(reservation, arrived, now)
            fit = get_hospital_suitability(patient.get("pre_ktas", ktas), selected)
            if not censored:
                parts = reward_components(
                    travel_time=elapsed, ktas=ktas, pre_ktas=patient.get("pre_ktas", ktas),
                    hospital=selected, occupancy_at_selection=occupancy,
                    route_failed=not arrived, golden_limit=GOLDEN_TIME_LIMITS[ktas],
                    rejection_count=rejection_count,
                    reward_version=config.reward_version, congestion_reward_mode=config.congestion_reward_mode)
            else:
                parts["rejection"] = -WEIGHTS.rejection * rejection_count
        reward = float(sum(parts.values()))
        if transitions is not None:
            # Each refusal was charged at its own decision; never charge it twice.
            transitions.add_reward(reward - parts["rejection"])
        row = {
            "patient_id": patient["patient_id"], "edge_id": patient.get("edge_id"), "ktas": ktas,
            "reward_components": parts, "hospital_suitability": fit,
            "occupancy_at_selection": occupancy if payload is not None else None,
            "pre_ktas": patient.get("pre_ktas", ktas),
            "occurrence_time": patient["occurrence_time"], "wait_seconds": wait,
            "dispatch_time": patient["occurrence_time"] + wait if payload is not None else None,
            "finish_time": now, "total_seconds": elapsed,
            "travel_seconds": elapsed - wait if payload is not None else None,
            "arrived": bool(arrived), "censored": censored, "reason": reason,
            "hospital_action": action,
            "hospital_id": hospitals[action]["id"] if action >= 0 else None,
            "vehicle_id": payload[3] if payload is not None else None,
            "reward": reward,
            "golden_success": bool(arrived and elapsed <= GOLDEN_TIME_LIMITS[ktas]),
            "last_dispatch_reason": patient.get("last_dispatch_reason"),
            "rejection_count": rejection_count, "rejection_history": history,
            "hospital_visits": copy.deepcopy(patient.get("hospital_visits", [])),
            "hospital_admission_mode": config.admission_mode,
            "congestion_reward_mode": config.congestion_reward_mode,
            "reward_version": config.reward_version,
            "estimated_reward_at_dispatch": predicted_reward,
            "estimated_total_seconds_at_dispatch": predicted_total,
        }
        outcomes.append(row)
        if on_outcome:
            on_outcome(row)

    primary_exception = None
    try:
        if config.repeat_background_traffic and "_route_files" not in scenario:
            from dashboard_traffic import prepare_background
            background_dir = Path(output_dir or Path(__file__).resolve().parents[1] / "results" / "sumo")
            routes, background = prepare_background(env.cfg_file, background_dir, config.drain_deadline)
            scenario.update(_route_files=routes, background_traffic=background)
        start_sumo(env, scenario, seed, gui=gui, output_dir=output_dir, quiet=quiet,
                   emergency_yield=emergency_yield)
        traci.simulationStep()  # The first scheduled occurrence is SUMO time 1.
        diagnostic_step(env, traci)
        traffic.reset()
        while traci.simulation.getTime() <= config.drain_deadline:
            now = float(traci.simulation.getTime())
            beds.refresh(now)
            if arrival_admission:
                traffic.refresh(excluded_vehicle_ids=manager.removed_vehicle_ids)
            else:
                traffic.refresh()
            if int(now) in schedule:
                manager.enqueue(dict(schedule[int(now)]))
            manager.step(dispatch, finish, prefix="amb_e_" if training else "amb_eval_",
                         on_hospital_arrival=on_hospital_arrival if arrival_admission else None)
            diagnostic_step(env, traci)
            if on_step is not None:
                on_step({"time": float(traci.simulation.getTime()),
                         "hospitals": hospitals, "manager": manager, "outcomes": outcomes,
                         "rejection_count": diagnostics["admission_rejections"],
                         "rejection_penalty": -WEIGHTS.rejection * diagnostics["admission_rejections"]})
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
    except BaseException as exc:
        primary_exception = exc
        record_sumo_failure(env, traci, exc, dict(
            phase="run_episode", seed=seed, mode=mode, scenario_id=scenario.get("id"),
            last_loop_time=locals().get("now"), completed=len(outcomes),
            generated=len(schedule), waiting=len(manager.waiting),
            active_vehicle_ids=list(manager.active), config=config.to_dict()))
        raise
    finally:
        close_sumo_safely(env, traci, primary_exception)
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
        "emergency_yield": yield_settings(emergency_yield),
        "background_traffic": {**(scenario.get("background_traffic") or {}),
                               **background_settings(config.repeat_background_traffic)},
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
        "hospital_exclusion_reasons": dict(rejection_pairs),
        "hospital_admission_mode": config.admission_mode,
        "congestion_reward_mode": config.congestion_reward_mode,
        "admission_requests": diagnostics["admission_requests"],
        "hospital_redirects": diagnostics["hospital_redirects"],
        "rejection_count": diagnostics["admission_rejections"],
        "rejected_patients": sum(row["rejection_count"] > 0 for row in outcomes),
        "admission_rejection_reasons": dict(admission_reasons),
        "ktas_stats": ktas_stats, "simulation_end_time": end_time,
        "load_std": float(np.std([assignments[i] for i in range(len(hospitals))])),
        "peak_hospital_occupancy": beds.peak_occupancy,
        "remaining_reservations": len(beds.reservations),
        "route_safety": route_safety_report,
        "learning_reward": transitions.total_reward if transitions else reward,
        "unattributed_learning_reward": transitions.unattributed_reward if transitions else 0.,
        "learning_transitions": transitions.count if transitions else 0,
        "avg_loss": float(np.mean(transitions.losses)) if transitions and transitions.losses else 0.,
        "seed": seed, "demand_scale": config.demand_scale,
        "demand_rates_per_hour": list(config.demand_rates_per_hour),
        "patients_per_hour": scenario.get("patients_per_hour"),
        "reward_version": config.reward_version,
        "observation_version": config.observation_version,
        "pending_reward_feature": config.pending_reward_feature,
        "golden_margin_feature": config.golden_margin_feature,
        "candidate_reward_feature": config.candidate_reward_feature,
        "reward_components": {k: sum(r["reward_components"][k] for r in outcomes) for k in REWARD_COMPONENT_NAMES},
        "avg_reward_components": {k: sum(r["reward_components"][k] for r in outcomes) / generated
                                   if generated else 0. for k in REWARD_COMPONENT_NAMES},
        "training_updates": getattr(agent, "training_updates", 0),
        "episode_updates": getattr(agent, "training_updates", 0) - updates_before,
        "target_updates": getattr(agent, "target_updates", 0),
        "total_decisions": getattr(agent, "total_decisions", 0),
        "episode_decisions": sum(selected_actions.values()),
        "episode_exploratory_decisions": diagnostics["exploratory_decisions"],
        "exploration_rate": diagnostics["exploratory_decisions"] / sum(selected_actions.values())
                             if selected_actions else 0.,
        "selected_hospitals": dict(selected_actions),
        "avg_selected_estimated_time": float(np.mean(selected_times)) if selected_times else None,
        "avg_extra_estimated_time": float(np.mean(extra_times)) if extra_times else None,
    }
