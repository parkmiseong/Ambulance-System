"""Read-only emergency-road connectivity and lane-aware hospital access checks."""
from collections import Counter


def emergency_component(net):
    """Largest strongly connected component, using permitted lane connections."""
    edges = {e.getID(): e for e in net.getEdges() if not e.getID().startswith(":") and e.allows("emergency")}
    outgoing = {key: [e.getID() for e in edge.getAllowedOutgoing("emergency") if e.getID() in edges]
                for key, edge in edges.items()}
    reverse = {key: [] for key in edges}
    for key, targets in outgoing.items():
        for target in targets:
            reverse[target].append(key)
    seen, order = set(), []
    for root in edges:
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(outgoing[root]))]
        while stack:
            key, targets = stack[-1]
            target = next(targets, None)
            if target is None:
                order.append(key)
                stack.pop()
            elif target not in seen:
                seen.add(target)
                stack.append((target, iter(outgoing[target])))
    seen, components = set(), []
    for root in reversed(order):
        if root in seen:
            continue
        component, stack = set(), [root]
        seen.add(root)
        while stack:
            key = stack.pop()
            component.add(key)
            for target in reverse[key]:
                if target not in seen:
                    seen.add(target)
                    stack.append(target)
        components.append(component)
    main = max(components, key=len, default=set())
    return main, dict(emergency_edges=len(edges), components=len(components),
                     largest_component_edges=len(main), excluded_edges=len(edges)-len(main),
                     component_sizes=sorted(Counter(map(len, components)).items(), reverse=True))


def prepare_access(env, hospitals, access):
    if not hasattr(env.net, "getEdges"):
        return access, None
    if not hasattr(env, "_emergency_component"):
        env._emergency_component = emergency_component(env.net)
    core, audit = env._emergency_component
    safe = [[edge for edge in group if edge in core] for group in access]
    if not any(group for h, group in zip(hospitals, safe) if h.get("type") == "권역응급의료센터"):
        raise RuntimeError("구급차 연결 도로망 안에 중증 환자를 수용할 병원 접근 도로가 없습니다.")
    sources = [edge for edge in env.ambulance_edges if edge in core]
    if not sources:
        raise RuntimeError("왕래 가능한 구급차 환자 출발 도로가 없습니다.")
    report = dict(audit, profile="emergency_scc_lane_v1", original_sources=len(env.ambulance_edges),
                  usable_sources=len(sources), hospital_access_counts={h["id"]: len(g) for h, g in zip(hospitals, safe)})
    # The source model/manifest and network file stay intact.
    return safe, (sources, report)


def arrival_lane(net, route, source_lane=None):
    """Choose a stop lane reachable from the route's incoming connection.

    Never force lane zero at the destination. No lateral move is assumed there.
    At a reroute immediately after a stop, constrain the first turn to that lane.
    """
    if not route:
        return None
    edges = [net.getEdge(key) for key in route]
    lanes = {lane for lane in edges[0].getLanes() if lane.allows("emergency")
             and (source_lane is None or lane.getID() == source_lane)}
    for index, (before, after) in enumerate(zip(edges, edges[1:])):
        if index > 0 and before.getLength() >= 40.:
            lanes = {lane for lane in before.getLanes() if lane.allows("emergency")}
        connections = before.getAllowedOutgoing("emergency").get(after, ())
        lanes = {c.getToLane() for c in connections if c.getFromLane() in lanes}
        if not lanes:
            return None
    # Blue-light ambulances favour the leftmost lane; do not make them cross to
    # lane zero just before their stop (which can trigger wrong-lane teleports).
    return max((i for i, lane in enumerate(edges[-1].getLanes()) if lane in lanes), default=None)


def lane_constrained_route(net, source, target, source_lane):
    """Try permitted first turns when edge routing picked a turn from another lane."""
    import traci
    from types import SimpleNamespace
    origin = net.getEdge(source)
    best = None
    for successor, connections in origin.getAllowedOutgoing("emergency").items():
        if not any(c.getFromLane().getID() == source_lane for c in connections):
            continue
        try:
            route = traci.simulation.findRoute(successor.getID(), target, vType="ambulance_custom", depart=-1, routingMode=0)
        except traci.TraCIException:
            continue
        edges = [source, *route.edges]
        if not route.edges or arrival_lane(net, edges, source_lane) is None:
            continue
        candidate = SimpleNamespace(edges=edges, length=origin.getLength() + route.length,
                                    travelTime=origin.getLength() / max(origin.getSpeed(), 1.) + route.travelTime)
        if best is None or candidate.travelTime < best.travelTime:
            best = candidate
    return best
