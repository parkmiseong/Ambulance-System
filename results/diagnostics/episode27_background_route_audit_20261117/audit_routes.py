"""Read-only audit of the original background ambulance routes; no simulation."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import sumolib

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NET = ROOT / "SUMO" / "\uac15\ub0a8" / "osm.net.xml.gz"
ROUTES = NET.parent / "ambulance_routes.rou.xml"
VCLASS = "emergency"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


original_hashes = {str(path): sha(path) for path in (NET, ROUTES)}
net = sumolib.net.readNet(str(NET), withInternal=True)
lane_change = {}
with gzip.open(NET, "rb") as stream:
    for event, element in ET.iterparse(stream, events=("end",)):
        if element.tag == "lane":
            lane_change[element.get("id")] = (element.get("changeLeft"), element.get("changeRight"))
        element.clear()

tree = ET.parse(ROUTES).getroot()
types = {node.get("id"): dict(node.attrib, params={p.get("key"): p.get("value") for p in node.findall("param")})
         for node in tree.findall("vType")}
named_routes = {node.get("id"): node.get("edges", "").split() for node in tree.findall("route")}
vehicles = tree.findall("vehicle")
groups = defaultdict(list)
for vehicle in vehicles:
    route = vehicle.find("route")
    edges = route.get("edges", "").split() if route is not None else named_routes.get(vehicle.get("route"), [])
    groups[tuple(edges)].append(vehicle)


def lane_closure(edge, initial):
    """Topological feasibility only; does not assert enough distance to change."""
    reached = set(initial)
    if edge.getFunction() == "internal":
        return reached
    lanes = edge.getLanes()
    changed = True
    while changed:
        changed = False
        for lane in list(reached):
            for direction, key in ((1, 0), (-1, 1)):
                index = lane.getIndex() + direction
                permission = lane_change.get(lane.getID(), (None, None))[key]
                if 0 <= index < len(lanes) and (permission is None or VCLASS in permission.split()):
                    candidate = lanes[index]
                    if candidate.allows(VCLASS) and candidate not in reached:
                        reached.add(candidate)
                        changed = True
    return reached


def via_allowed(connection):
    via = connection.getViaLaneID()
    visited = set()
    target = connection.getToLane()
    while via:
        if via in visited:
            return False
        visited.add(via)
        try:
            lane = net.getLane(via)
        except KeyError:
            return False
        if not lane.allows(VCLASS):
            return False
        continuation = [c for c in lane.getOutgoing()
                        if c.getToLane() == target and c.allows(VCLASS)]
        if not continuation:
            return False
        via = continuation[0].getViaLaneID()
    return True


audits = []
counts = Counter()
for index, (edge_ids, members) in enumerate(groups.items(), 1):
    missing = [key for key in edge_ids if not net.hasEdge(key)]
    edges = [net.getEdge(key) for key in edge_ids if net.hasEdge(key)]
    reasons = []
    if not edge_ids:
        reasons.append("empty_route")
    if missing:
        reasons.append("missing_edge")
    forbidden = [edge.getID() for edge in edges if not edge.allows(VCLASS)]
    if forbidden:
        reasons.append("edge_without_emergency_lane")
    pair_checks = []
    allowed_connections = []
    if not missing:
        for before, after in zip(edges, edges[1:]):
            connections = before.getConnections(after)
            permitted = [c for c in connections if c.getFromLane().allows(VCLASS)
                         and c.getToLane().allows(VCLASS) and c.allows(VCLASS) and via_allowed(c)]
            pair_checks.append(dict(before=before.getID(), after=after.getID(),
                adjacent_nodes=before.getToNode() == after.getFromNode(),
                connection_count=len(connections), permitted_connection_count=len(permitted),
                from_lanes=sorted({c.getFromLane().getID() for c in permitted}),
                to_lanes=sorted({c.getToLane().getID() for c in permitted})))
            allowed_connections.append(permitted)
        if any(not p["adjacent_nodes"] for p in pair_checks):
            reasons.append("nonadjacent_edge_nodes")
        if any(p["connection_count"] == 0 for p in pair_checks):
            reasons.append("missing_lane_connection")
        if any(p["permitted_connection_count"] == 0 for p in pair_checks):
            reasons.append("missing_emergency_lane_connection")
    vehicle_checks = []
    for vehicle in members:
        first_lanes = [l for l in edges[0].getLanes() if l.allows(VCLASS)] if edges and not missing else []
        depart_lane = vehicle.get("departLane", "first")
        if depart_lane.lstrip("-").isdigit():
            first_lanes = [lane for lane in first_lanes if lane.getIndex() == int(depart_lane)]
        elif depart_lane == "first" and first_lanes:
            first_lanes = [min(first_lanes, key=lambda lane: lane.getIndex())]
        # 'best/free/random' choices allow any permitted lane in this static audit.
        reachable = set(first_lanes)
        straight = set(first_lanes)
        for step, connections in enumerate(allowed_connections):
            reachable = lane_closure(edges[step], reachable)
            reachable = {c.getToLane() for c in connections if c.getFromLane() in reachable}
            straight = {c.getToLane() for c in connections if c.getFromLane() in straight}
        lane_feasible = bool(reachable) and not missing and bool(edge_ids)
        errors = list(reasons)
        if not first_lanes:
            errors.append("no_permitted_departure_lane")
        if not lane_feasible:
            errors.append("no_topological_lane_chain")
        valid = not errors
        counts["vehicles"] += 1
        counts["valid_vehicles"] += valid
        counts["invalid_vehicles"] += not valid
        for error in set(errors):
            counts[error] += 1
        vehicle_checks.append(dict(id=vehicle.get("id"), type=vehicle.get("type"),
            depart=vehicle.get("depart"), depart_lane=depart_lane,
            permitted_departure_lanes=[l.getID() for l in first_lanes],
            valid=valid, errors=sorted(set(errors)),
            topological_lane_chain=lane_feasible,
            continuous_lane_chain_without_lateral_changes=bool(straight)))
    audits.append(dict(route_group=index, edges=list(edge_ids), vehicle_count=len(members),
        missing_edges=missing, forbidden_emergency_edges=forbidden,
        pair_checks=pair_checks, route_errors=sorted(set(reasons)), vehicles=vehicle_checks))

unchanged = all(sha(path) == digest for path, digest in original_hashes.items())
assert unchanged
report = dict(schema_version=1, source_net=str(NET), source_routes=str(ROUTES),
    source_hashes=original_hashes, source_inputs_unchanged=unchanged,
    vehicle_types=types, summary=dict(counts, unique_route_count=len(groups),
        duplicate_route_group_count=sum(len(group) > 1 for group in groups.values()),
        redundant_route_copies=sum(len(group)-1 for group in groups.values()),
        valid_unique_routes=sum(not a["route_errors"] and all(v["valid"] for v in a["vehicles"]) for a in audits),
        invalid_unique_routes=sum(bool(a["route_errors"]) or any(not v["valid"] for v in a["vehicles"]) for a in audits)),
    limitations=["Static topological feasibility permits allowed intra-edge lane changes; it does not assert sufficient maneuver distance.",
        "No driving simulation, route repair, traffic filtering or training was performed.",
        "Invalid input findings do not by themselves prove the native NULL-pointer crash cause."],
    route_groups=audits)
target = OUT / "static_route_audit.json"
with target.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2)
print(json.dumps(report["summary"], ensure_ascii=False))
