# -*- coding: utf-8 -*-
"""Single SUMO clock, FIFO patient queue, at most ten research-managed missions.

Only MissionManager.step() advances SUMO. Never call await_arrival() from training.
"""
from collections import deque
from dataclasses import dataclass
import traci

MAX_AMBULANCES = 10
MANAGED_PREFIXES = ("amb_e", "amb_eval_")


def managed_ambulances():
    return [v for v in traci.vehicle.getIDList() if v.startswith(MANAGED_PREFIXES)]


def spawn_at_patient(patient_edge, hospital_edge, vehicle_id, max_ambulances=10):
    """Legacy-compatible creation; capacity also guarded by MissionManager.active."""
    limit = max(1, min(MAX_AMBULANCES, int(max_ambulances)))
    if len(managed_ambulances()) >= limit:
        return False, "ambulance_capacity"
    if not patient_edge or not hospital_edge:
        return False, "missing_edge"
    try:
        if "ambulance_custom" not in traci.vehicletype.getIDList():
            return False, "missing_vehicle_type"
        route = traci.simulation.findRoute(patient_edge, hospital_edge,
                                          vType="ambulance_custom")
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


class MissionManager:
    """FIFO queue. Callbacks: dispatch(patient, vehicle_id)->(edge, payload) or None;
    finish(patient, payload, arrived, elapsed, wait, reason)->None.
    Patient dict must contain occurrence_time and edge_id.
    """
    def __init__(self, max_ambulances=10, timeout=1800, max_wait=1800, verbose=False):
        self.limit = max(1, min(MAX_AMBULANCES, int(max_ambulances)))
        self.timeout = float(timeout)
        self.max_wait = float(max_wait)
        self.waiting = deque()
        self.active = {}
        self.verbose = verbose
        self.peak_active = 0
        self.peak_waiting = 0
        self.completed = 0
        self.failed = 0
        self.wait_total = 0.0
        self.wait_count = 0
        self._serial = 0

    def enqueue(self, patient):
        self.waiting.append(patient)
        self.peak_waiting = max(self.peak_waiting, len(self.waiting))

    def _complete(self, mission, arrived, now, reason, finish):
        patient = mission.patient
        wait = max(0., mission.dispatch_time - float(patient['occurrence_time']))
        elapsed = max(0., now - float(patient['occurrence_time']))
        self.wait_total += wait
        self.wait_count += 1
        if arrived:
            self.completed += 1
        else:
            self.failed += 1
        finish(patient, mission.payload, arrived, elapsed, wait, reason)

    def _expire_waiting(self, now, finish):
        while self.waiting and now - float(self.waiting[0]['occurrence_time']) >= self.max_wait:
            patient = self.waiting.popleft()
            wait = max(0., now - float(patient['occurrence_time']))
            self.failed += 1
            self.wait_total += wait
            self.wait_count += 1
            finish(patient, None, False, wait, wait, 'waiting_timeout')

    def _dispatch(self, now, dispatch, finish, prefix):
        # The manager's registry includes pending departures, unlike getIDList().
        while self.waiting and len(self.active) < self.limit:
            patient = self.waiting.popleft()
            self._serial += 1
            vid = f"{prefix}{self._serial}"
            try:
                assignment = dispatch(patient, vid)
                if assignment is None:
                    self.failed += 1
                    wait = max(0., now - float(patient['occurrence_time']))
                    self.wait_total += wait
                    self.wait_count += 1
                    finish(patient, None, False, wait, wait, 'no_valid_hospital')
                    continue
                hospital_edge, payload = assignment
                ok, reason = spawn_at_patient(patient['edge_id'], hospital_edge, vid,
                                               self.limit)
                if not ok:
                    self.failed += 1
                    wait = max(0., now - float(patient['occurrence_time']))
                    self.wait_total += wait
                    self.wait_count += 1
                    finish(patient, payload, False, wait, wait, reason)
                    continue
                self.active[vid] = Mission(patient, vid, now, payload)
                self.peak_active = max(self.peak_active, len(self.active))
                if self.verbose:
                    print(f"[출동] {vid} 발생={patient['occurrence_time']:.0f}s "
                          f"출동={now:.0f}s 운행={len(self.active)}/{self.limit} "
                          f"대기={len(self.waiting)}")
            except Exception as exc:
                self.failed += 1
                wait = max(0., now - float(patient['occurrence_time']))
                self.wait_total += wait
                self.wait_count += 1
                finish(patient, None, False, wait, wait, f'dispatch_exception: {exc}')

    def step(self, dispatch, finish, prefix='amb_e'):
        """Process arrivals, dispatch waiting patients, advance exactly ONE SUMO step."""
        now = float(traci.simulation.getTime())
        self._expire_waiting(now, finish)
        self._dispatch(now, dispatch, finish, prefix)
        traci.simulationStep()
        now = float(traci.simulation.getTime())
        arrived_ids = set(traci.simulation.getArrivedIDList())
        running = set(traci.vehicle.getIDList())
        for vid, mission in list(self.active.items()):
            if vid in running:
                mission.appeared = True
            reason = None
            if vid in arrived_ids:
                reason = 'arrived'
            elif now - mission.dispatch_time >= self.timeout:
                reason = 'mission_timeout'
                if vid in running:
                    traci.vehicle.remove(vid)
            elif mission.appeared and vid not in running:
                reason = 'vehicle_disappeared'
            # Vehicle may still be queued for insertion: do not treat as failure.
            if reason is not None:
                del self.active[vid]
                self._complete(mission, reason == 'arrived', now, reason, finish)
        return now

    def close_unfinished(self, finish):
        """End-of-horizon censoring: record unresolved patients as failures."""
        now = float(traci.simulation.getTime())
        for vid, mission in list(self.active.items()):
            if vid in traci.vehicle.getIDList():
                traci.vehicle.remove(vid)
            del self.active[vid]
            self._complete(mission, False, now, 'horizon_timeout', finish)
        while self.waiting:
            patient = self.waiting.popleft()
            wait = max(0., now - float(patient['occurrence_time']))
            self.failed += 1
            self.wait_total += wait
            self.wait_count += 1
            finish(patient, None, False, wait, wait, 'horizon_waiting')


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
