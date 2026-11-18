"""Opt-in, bounded lane observations for native-crash diagnostic replays only."""
import json
import math
from pathlib import Path


class LaneSnapshotProbe:
    """Read lane state inside one window, without changing vehicle commands.

    Normal vehicles outside internal lanes receive no detailed queries. Vehicle
    type IDs are cached until departure; lane IDs must be refreshed every tick.
    Internal-lane best-lane mismatches are observations, never proof of a bug.
    """

    MAX_DETAILS = 500

    def __init__(self, directory, start=None, end=None, *, save_native_state=False,
                 native=None, logging_errors=None):
        if (start is None) != (end is None):
            raise ValueError("Lane probe start and end must be specified together.")
        if start is not None and (not math.isfinite(start) or not math.isfinite(end)
                                  or start < 0 or end < start):
            raise ValueError("Require finite lane probe bounds with 0 <= start <= end.")
        if save_native_state and start is None:
            raise ValueError("Saving native state requires a lane probe window.")
        self.directory = Path(directory)
        self.start, self.end = start, end
        self.save_native_state = save_native_state
        self.native = native
        self.logging_errors = logging_errors if logging_errors is not None else []
        self._type_ids = {}
        self._last_time = None
        self._state_attempted = False
        self._error_count = 0
        self._error_examples = 0
        self.ticks = 0
        self.rows_written = 0
        self.state_path = None

    @property
    def enabled(self):
        return self.start is not None

    def settings(self):
        return dict(enabled=self.enabled, start=self.start, end=self.end,
                    save_native_state=self.save_native_state,
                    detail_limit_per_tick=self.MAX_DETAILS,
                    path=str(self.directory / "lane_snapshots.jsonl") if self.enabled else None)

    def statistics(self):
        return dict(**self.settings(), observed_ticks=self.ticks,
                    rows_written=self.rows_written, logging_error_count=self._error_count,
                    native_state_path=str(self.state_path) if self.state_path else None)

    def _record_error(self, operation, exc):
        self._error_count += 1
        # A bad getter must not produce hundreds of thousands of summary lines.
        if self._error_examples < 100:
            self.logging_errors.append(f"lane probe {operation}: {type(exc).__name__}: {exc}")
            self._error_examples += 1
        elif self._error_examples == 100:
            self.logging_errors.append("lane probe: further error examples omitted; see logging_error_count.")
            self._error_examples += 1

    def _read(self, operation, function, *args):
        try:
            return function(*args)
        except Exception as exc:
            self._record_error(operation, exc)
            # A dead SUMO connection is still a failed diagnostic, including
            # when it is first detected by an observation getter.
            import traci
            if isinstance(exc, traci.exceptions.FatalTraCIError):
                raise
            return None

    def _optional(self, domain, method, *args):
        function = getattr(domain, method, None)
        if function is None:
            return None
        return self._read(f"{method} {args[0] if args else ''}", function, *args)

    def observe(self, time, managed_ids=()):
        try:
            self._observe(time, managed_ids)
        except Exception as exc:
            import traci
            if isinstance(exc, traci.exceptions.FatalTraCIError):
                raise
            self._record_error("snapshot collection", exc)

    def _observe(self, time, managed_ids):
        if not self.enabled or not self.start <= time <= self.end or time == self._last_time:
            return
        self._last_time = time
        self.ticks += 1
        if self.native is None:
            import traci
            self.native = traci
        native = self.native
        if self.save_native_state and not self._state_attempted:
            self._state_attempted = True
            target = self.directory / "native_state_at_probe_start.xml.gz"
            # saveState is an explicit opt-in export; never call loadState or
            # change routing, speeds, stops, lights, vehicle types or settings.
            result = self._read("saveState", native.simulation.saveState, str(target))
            if result is not None or target.is_file():
                self.state_path = target
        ids = self._read("getIDList", native.vehicle.getIDList)
        if ids is None:
            return
        ids = set(ids)
        self._type_ids = {vid: value for vid, value in self._type_ids.items() if vid in ids}
        managed = set(managed_ids)
        selected = []
        for vid in sorted(ids):
            lane = self._optional(native.vehicle, "getLaneID", vid)
            type_id = self._type_ids.get(vid)
            if type_id is None:
                type_id = self._optional(native.vehicle, "getTypeID", vid)
                if type_id is not None:
                    self._type_ids[vid] = type_id
            is_managed = vid in managed or vid.startswith(("amb_e_", "amb_eval_"))
            is_background = type_id == "ambulance" or bool(type_id and type_id.startswith("ambulance@"))
            internal = bool(lane and lane.startswith(":"))
            if is_managed or is_background or internal:
                category = "managed_ambulance" if is_managed else "background_ambulance" if is_background else "internal_lane"
                priority = 0 if is_managed else 1 if is_background else 2
                selected.append((priority, vid, lane, type_id, category, internal))
        selected.sort()
        details = []
        lane_links = {}
        for _, vid, lane, type_id, category, internal in selected[:self.MAX_DETAILS]:
            best = self._optional(native.vehicle, "getBestLanes", vid)
            flags = []
            if best is not None:
                if not best:
                    flags.append("empty_best_lanes")
                elif lane and not internal and lane not in [entry[0] for entry in best if entry]:
                    flags.append("current_normal_lane_absent_from_best_lanes")
            if lane and lane not in lane_links:
                lane_links[lane] = self._optional(native.lane, "getLinks", lane)
            row = dict(vehicle_id=vid, category=category, type_id=type_id,
                       lane_id=lane, internal_lane=internal, best_lanes=best,
                       route_index=self._optional(native.vehicle, "getRouteIndex", vid),
                       route_id=self._optional(native.vehicle, "getRouteID", vid),
                       allowed_speed=self._optional(native.vehicle, "getAllowedSpeed", vid),
                       speed=self._optional(native.vehicle, "getSpeed", vid),
                       lane_position=self._optional(native.vehicle, "getLanePosition", vid),
                       next_links=self._optional(native.vehicle, "getNextLinks", vid),
                       rescue_lane=self._optional(native.vehicle, "getParameter", vid, "rescueLane"),
                       observation_flags=flags)
            details.append(row)
        snapshot = dict(time=time, running_vehicle_count=len(ids),
                        candidate_count=len(selected), detailed_vehicle_count=len(details),
                        truncated_vehicle_count=max(0, len(selected) - self.MAX_DETAILS),
                        flags_are_observations_only=True, vehicles=details, lane_links=lane_links)
        try:
            text = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
            with (self.directory / "lane_snapshots.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(text + "\n")
            self.rows_written += 1
        except (OSError, TypeError, ValueError) as exc:
            self._record_error("JSONL write", exc)
