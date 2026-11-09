# -*- coding: utf-8 -*-
"""Traffic-aware SUMO routing weights based on observed per-edge mean speeds.

Call reset() after each new traci.start() and refresh() before dispatch.
Uses TraCI's global adapted travel times consumed by simulation.findRoute().
"""
import math
import traci
from traci.constants import VAR_ROAD_ID


class TrafficRouter:
    def __init__(self, refresh_interval=30, min_speed=1.0, horizon=60, net=None):
        self.refresh_interval = float(refresh_interval)
        self.min_speed = float(min_speed)
        self.horizon = float(horizon)
        self.last_refresh = float('-inf')
        self.last_updated = set()
        self.net = net
        self.geometry = {}
        self.subscribed = set()

    def reset(self):
        self.last_refresh = float('-inf')
        self.last_updated.clear()
        self.geometry.clear()
        self.subscribed.clear()

    def _geometry(self, edge_id):
        if edge_id not in self.geometry:
            if self.net is not None:
                lane = self.net.getEdge(edge_id).getLanes()[0]
                length, speed = float(lane.getLength()), float(lane.getSpeed())
            else:
                lane_id = f'{edge_id}_0'
                length = float(traci.lane.getLength(lane_id))
                speed = float(traci.lane.getMaxSpeed(lane_id))
            self.geometry[edge_id] = (length, speed)
        return self.geometry[edge_id]

    @staticmethod
    def _free_flow_time(edge_id):
        lane_count = traci.edge.getLaneNumber(edge_id)
        if lane_count <= 0:
            return None
        lane_id = traci.edge.getLaneID(edge_id, 0) if hasattr(traci.edge, 'getLaneID') else None
        # Edge lane IDs are generally edgeID_0; lane.getIDList is not needed.
        if lane_id is None:
            lane_id = f'{edge_id}_0'
        length = float(traci.lane.getLength(lane_id))
        speed = float(traci.lane.getMaxSpeed(lane_id))
        if length <= 0 or speed <= 0:
            return None
        return length / speed

    def refresh(self, force=False):
        now = float(traci.simulation.getTime())
        if not force and now - self.last_refresh < self.refresh_interval:
            return 0
        self.last_refresh = now

        # Observe occupied edges; no full-network TraCI scan every step.
        active_edges = set()
        vehicles = set(traci.vehicle.getIDList())
        # Subscriptions arrive with each simulation step; avoid one socket request
        # per background vehicle on every refresh.
        self.subscribed.intersection_update(vehicles)
        for vehicle_id in vehicles - self.subscribed:
            traci.vehicle.subscribe(vehicle_id, (VAR_ROAD_ID,))
            self.subscribed.add(vehicle_id)
        observations = traci.vehicle.getAllSubscriptionResults()
        for vehicle_id in vehicles:
            edge_id = observations.get(vehicle_id, {}).get(VAR_ROAD_ID)
            if edge_id is None:
                edge_id = traci.vehicle.getRoadID(vehicle_id)
            if edge_id and not edge_id.startswith(':'):
                active_edges.add(edge_id)

        # Restore free-flow costs for edges that were previously congested.
        for edge_id in self.last_updated - active_edges:
            try:
                length, speed = self._geometry(edge_id)
                base = length / speed if length > 0 and speed > 0 else None
                if base is not None:
                    traci.edge.adaptTraveltime(edge_id, base)
            except traci.TraCIException:
                pass

        updated = set()
        for edge_id in active_edges:
            try:
                length, speed = self._geometry(edge_id)
                if length <= 0 or speed <= 0:
                    continue
                base = length / speed
                mean_speed = float(traci.edge.getLastStepMeanSpeed(edge_id))
                if not math.isfinite(mean_speed) or mean_speed < 0:
                    continue
                observed = length / max(mean_speed, self.min_speed)
                # Never assume observed travel time is faster than free-flow.
                traci.edge.adaptTraveltime(edge_id, max(base, observed))
                updated.add(edge_id)
            except traci.TraCIException:
                continue
        self.last_updated = updated
        return len(updated)
