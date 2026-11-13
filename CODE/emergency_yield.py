"""SUMO native emergency corridor settings and bounded nearby-traffic telemetry."""
import traci
import traci.constants as tc

REACTION_DISTANCE = 60.
LATERAL_RESOLUTION = .4


def yield_settings(enabled):
    return dict(enabled=enabled, reaction_distance_m=REACTION_DISTANCE if enabled else None,
                lateral_resolution_m=LATERAL_RESOLUTION if enabled else None,
                signal_priority_included=enabled, implementation="sumo_bluelight_sublane")


def nearby_traffic(env, active_ids, subscribed, enabled, limit=150):
    """Subscribe around managed ambulances; never poll every background vehicle."""
    active = set(active_ids)
    subscribed.intersection_update(active)
    neighbors = {}
    for vid in active_ids:
        if vid not in subscribed:
            traci.vehicle.subscribeContext(vid, tc.CMD_GET_VEHICLE_VARIABLE, 100.,
                                           [tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_SPEED, tc.VAR_LENGTH, tc.VAR_WIDTH])
            subscribed.add(vid)
        neighbors.update(traci.vehicle.getContextSubscriptionResults(vid) or {})
    traffic = []
    for vid, values in neighbors.items():
        if vid.startswith(("amb_e", "amb_eval_")) and vid not in active:
            continue  # A manually removed ambulance can remain in this tick's context results.
        if vid in active or len(traffic) >= limit or tc.VAR_POSITION not in values:
            continue
        try:
            influenced = traci.vehicle.getParameter(vid, "rescueLane").split() if enabled else []
        except traci.TraCIException:
            continue
        yielding_to = [owner for owner in influenced if owner in active]
        x, y = values[tc.VAR_POSITION]
        traffic.append(dict(id=vid, x=x, y=y, angle=values.get(tc.VAR_ANGLE, 0.),
                            speed=values.get(tc.VAR_SPEED, 0.), length=values.get(tc.VAR_LENGTH, 5.),
                            width=values.get(tc.VAR_WIDTH, 1.8), yielding=bool(yielding_to), yielding_to=yielding_to))
    lanes, roads = [], set()
    for vid in active_ids:
        road = traci.vehicle.getRoadID(vid)
        if road and not road.startswith(":"):
            roads.add(road)
    for road in roads:
        for lane in env.net.getEdge(road).getLanes():
            lanes.append(dict(id=lane.getID(), shape=[list(point[:2]) for point in lane.getShape()],
                              width=lane.getWidth()))
    return traffic, lanes
