# -*- coding: utf-8 -*-
"""Single SUMO clock, FIFO patient queue, at most ten research-managed missions.

Only MissionManager.step() advances SUMO. Never call await_arrival() from training.
"""
from collections import deque, Counter
from dataclasses import dataclass
import traci

DEFERRED = object()

MAX_AMBULANCES = 10
MANAGED_PREFIXES = ("amb_e", "amb_eval_")


def managed_ambulances():
    return [v for v in traci.vehicle.getIDList() if v.startswith(MANAGED_PREFIXES)]


def spawn_at_patient(patient_edge, hospital_edge, vehicle_id, max_ambulances=10, route_edges=None):
    """Legacy-compatible creation; capacity also guarded by MissionManager.active."""
    limit = max(1, min(MAX_AMBULANCES, int(max_ambulances)))
    if len(managed_ambulances()) >= limit:
        return False, "ambulance_capacity"
    if not patient_edge or not hospital_edge:
        return False, "missing_edge"
    try:
        if "ambulance_custom" not in traci.vehicletype.getIDList():
            return False, "missing_vehicle_type"
        if route_edges is not None:
            edges = list(route_edges)
        else:
            route = traci.simulation.findRoute(patient_edge, hospital_edge,
                                              vType="ambulance_custom", routingMode=0)
            edges = list(route.edges)
        if not edges or edges[0] != patient_edge or edges[-1] != hospital_edge:
            return False, "unreachable"
        route_id = "route_" + vehicle_id
        traci.route.add(route_id, edges)
        traci.vehicle.add(vehicle_id, route_id, typeID="ambulance_custom",
                          depart="now", departLane="best", departPos="base",
                          departSpeed="0")
        return True, "created_pending_departure"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


@dataclass
class Mission:
    patient: dict
    vehicle_id: str
    dispatch_time: float
    payload: object
    appeared: bool = False
    estimated_reward: float = 0.


class MissionManager:
    """One shared clock; priority queue with retryable temporary bed shortages."""
    def __init__(self, max_ambulances=10, timeout=1800, max_wait=1800,
                 verbose=False, queue_policy="severity", aging_seconds=300):
        if queue_policy not in ("fifo", "severity", "hybrid"):
            raise ValueError(f"Unknown queue policy: {queue_policy}")
        if timeout <= 0 or max_wait <= 0 or aging_seconds <= 0:
            raise ValueError("Timeouts and aging_seconds must be positive.")
        self.limit = max(1, min(MAX_AMBULANCES, int(max_ambulances)))
        self.timeout, self.max_wait = float(timeout), float(max_wait)
        self.queue_policy, self.aging_seconds = queue_policy, float(aging_seconds)
        self.verbose = verbose
        self.waiting, self.active = deque(), {}
        self.peak_active = self.peak_waiting = 0
        self.completed = self.failed = self.censored = 0
        self.wait_total, self.wait_count = 0., 0
        self._serial = 0
        self.failure_reasons, self.censor_reasons = Counter(), Counter()
        self.ktas_stats = {k: dict(arrived=0, failed=0, censored=0, wait_total=0., count=0)
                           for k in range(1, 6)}

    def enqueue(self, patient):
        self.waiting.append(patient)
        self.peak_waiting = max(self.peak_waiting, len(self.waiting))

    def _priority_key(self, patient, now):
        ktas = max(1, min(5, int(patient.get("ktas", patient.get("pre_ktas", 5)))))
        occurrence = float(patient["occurrence_time"])
        if self.queue_policy == "fifo":
            return (occurrence,)
        if self.queue_policy == "hybrid":
            return (ktas - max(0., now - occurrence) / self.aging_seconds, occurrence)
        return (ktas, occurrence)

    def _record_outcome(self, patient, payload, arrived, now, dispatch_time, reason, finish):
        wait = max(0., dispatch_time - float(patient["occurrence_time"]))
        elapsed = max(0., now - float(patient["occurrence_time"]))
        censored = reason.startswith("horizon")
        if arrived:
            self.completed += 1
        elif censored:
            self.censored += 1
            self.censor_reasons[reason] += 1
        else:
            self.failed += 1
            self.failure_reasons[reason.split(":", 1)[0]] += 1
        self.wait_total += wait
        self.wait_count += 1
        ktas = max(1, min(5, int(patient.get("ktas", patient.get("pre_ktas", 5)))))
        stats = self.ktas_stats[ktas]
        stats["count"] += 1
        stats["wait_total"] += wait
        stats["arrived" if arrived else ("censored" if censored else "failed")] += 1
        finish(patient, payload, arrived, elapsed, wait, reason)

    def _expire_waiting(self, now, finish):
        survivors = deque()
        while self.waiting:
            patient = self.waiting.popleft()
            if now - float(patient["occurrence_time"]) >= self.max_wait:
                self._record_outcome(patient, None, False, now, now, "waiting_timeout", finish)
            else:
                survivors.append(patient)
        self.waiting = survivors

    def _dispatch(self, now, dispatch, finish, prefix):
        attempted = set()
        while len(self.active) < self.limit:
            ready = [p for p in self.waiting
                     if id(p) not in attempted and p.get("retry_after", 0.) <= now]
            if not ready:
                break
            patient = min(ready, key=lambda p: self._priority_key(p, now))
            self.waiting.remove(patient)
            attempted.add(id(patient))
            self._serial += 1
            vid = f"{prefix}{self._serial}"
            # Unexpected callback/optimizer errors must abort, not silently drop a patient.
            assignment = dispatch(patient, vid)
            if assignment is DEFERRED:
                self.waiting.append(patient)
                continue
            if assignment is None:
                self._record_outcome(patient, None, False, now, now, "no_valid_hospital", finish)
                continue
            hospital_edge, payload = assignment
            selected = next((value for value in payload
                             if isinstance(value, dict) and "route_edges" in value), None)
            ok, reason = spawn_at_patient(
                patient["edge_id"], hospital_edge, vid, self.limit,
                route_edges=selected.get("route_edges") if selected else None)
            if not ok:
                self._record_outcome(patient, payload, False, now, now, reason, finish)
                continue
            self.active[vid] = Mission(
                patient, vid, now, payload,
                estimated_reward=float(selected.get("estimated_reward_at_dispatch", 0.)) if selected else 0.)
            self.peak_active = max(self.peak_active, len(self.active))

    @staticmethod
    def _remove_vehicle(vid):
        try:
            # Also remove vehicles still waiting for insertion.
            traci.vehicle.remove(vid)
        except traci.TraCIException:
            pass

    def step(self, dispatch, finish, prefix="amb_e"):
        now = float(traci.simulation.getTime())
        self._expire_waiting(now, finish)
        self._dispatch(now, dispatch, finish, prefix)
        traci.simulationStep()
        now = float(traci.simulation.getTime())
        arrived = set(traci.simulation.getArrivedIDList())
        running = set(traci.vehicle.getIDList())
        for vid, mission in list(self.active.items()):
            if vid in running:
                mission.appeared = True
            reason = None
            if vid in arrived:
                reason = "arrived"
            elif now - mission.dispatch_time >= self.timeout:
                reason = "mission_timeout"
                self._remove_vehicle(vid)
            elif mission.appeared and vid not in running:
                reason = "vehicle_disappeared"
            if reason is not None:
                del self.active[vid]
                self._record_outcome(mission.patient, mission.payload, reason == "arrived",
                                     now, mission.dispatch_time, reason, finish)
        return now

    def close_unfinished(self, finish):
        """Record safety-boundary truncation separately from operational failures."""
        now = float(traci.simulation.getTime())
        for vid, mission in list(self.active.items()):
            self._remove_vehicle(vid)
            del self.active[vid]
            self._record_outcome(mission.patient, mission.payload, False, now,
                                 mission.dispatch_time, "horizon_timeout", finish)
        while self.waiting:
            patient = self.waiting.popleft()
            self._record_outcome(patient, None, False, now, now, "horizon_waiting", finish)


def await_arrival(vehicle_id, max_steps=1800):
    """Legacy helper. Do not use with MissionManager (it advances SUMO clock)."""
    appeared = False
    start = float(traci.simulation.getTime())
    for _ in range(max_steps):
        traci.simulationStep()
        now = float(traci.simulation.getTime())
        if vehicle_id in traci.simulation.getArrivedIDList():
            return True, now-start
        running = vehicle_id in traci.vehicle.getIDList()
        appeared |= running
        if appeared and not running:
            return False, now-start
    if vehicle_id in traci.vehicle.getIDList():
        traci.vehicle.remove(vehicle_id)
    return False, float(traci.simulation.getTime())-start
