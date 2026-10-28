# -*- coding: utf-8 -*-
"""SUMO-GUI ambulance follow camera. Does not advance simulation time."""
import traci


class AmbulanceFollowCamera:
    def __init__(self, zoom=900.0, view_id="View #0", verbose=True):
        self.zoom = float(zoom)
        self.view_id = view_id
        self.verbose = verbose
        self.following = None
        self._unavailable = False

    def update(self, active_missions):
        """Call after MissionManager.step(). Follow oldest visible active mission.

        When the tracked mission is completed and removed from active_missions,
        select another visible ambulance. If none are visible, wait until insertion.
        """
        if self._unavailable:
            return
        try:
            running = set(traci.vehicle.getIDList())
            active = [vid for vid in active_missions if vid in running]
            if self.following in active:
                return
            if not active:
                self.following = None
                return
            next_vehicle = active[0]
            traci.gui.trackVehicle(self.view_id, next_vehicle)
            traci.gui.setZoom(self.view_id, self.zoom)
            self.following = next_vehicle
            if self.verbose:
                print(f"[GUI 카메라] 추적 구급차: {next_vehicle}")
        except (traci.exceptions.TraCIException, AttributeError) as exc:
            # A vehicle may disappear between checking and selecting it.
            self.following = None
            if self.verbose:
                print(f"[GUI 카메라] 이번 프레임 추적 실패: {exc}")
