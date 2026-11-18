"""Prepare review-only background route repairs; never change source assets."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "SUMO" / "\uac15\ub0a8" / "ambulance_routes.rou.xml"
NET = SOURCE.parent / "osm.net.xml.gz"
BINARY = ROOT / ".runtime/sumo-1.28.0/bin/duarouter.exe"
with (OUT / "preparation_started.txt").open("x", encoding="utf-8") as stream:
    stream.write("Review-only candidates. Do not connect these files to training without a new experiment decision.\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


source_hashes = {str(path): digest(path) for path in (SOURCE, NET)}
original = ET.parse(SOURCE).getroot()
original_vehicles = {v.get("id"): v for v in original.findall("vehicle")}
original_edges = {key: value.find("route").get("edges").split() for key, value in original_vehicles.items()}
audit = json.loads((OUT.parent / "static_route_audit.json").read_text(encoding="utf-8"))
previously_valid = {v["id"] for group in audit["route_groups"] for v in group["vehicles"] if v["valid"]}


def routes_from(path):
    root = ET.parse(path).getroot()
    named = {r.get("id"): r.get("edges", "").split() for r in root.findall("route")}
    result = {}
    for vehicle in root.findall("vehicle"):
        route = vehicle.find("route")
        result[vehicle.get("id")] = route.get("edges", "").split() if route is not None else named[vehicle.get("route")]
    return result


def write_source_preserving(path, selected_edges):
    root = ET.Element("routes", original.attrib)
    for child in original:
        if child.tag != "vehicle":
            root.append(copy.deepcopy(child))
        elif child.get("id") in selected_edges:
            vehicle = copy.deepcopy(child)
            vehicle.find("route").set("edges", " ".join(selected_edges[child.get("id")]))
            root.append(vehicle)
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    loaded = ET.parse(path).getroot()
    assert [ET.tostring(x) for x in loaded.findall("vType")] == [ET.tostring(x) for x in root.findall("vType")]
    for vehicle in loaded.findall("vehicle"):
        assert vehicle.attrib == original_vehicles[vehicle.get("id")].attrib


def native(label, source, repair=False, trim_endpoints=False, tolerate_errors=False):
    target = OUT / (label + ".rou.xml")
    command = [str(BINARY), "--net-file", str(NET), "--route-files", str(source),
        "--output-file", str(target), "--repair", str(repair).lower(),
        "--repair.from", str(trim_endpoints).lower(), "--repair.to", str(trim_endpoints).lower(),
        "--ignore-errors", str(tolerate_errors).lower(), "--skip-new-routes", "false" if repair else "true",
        "--log", str(OUT / (label + ".log")),
        "--error-log", str(OUT / (label + "_errors.log"))]
    result = subprocess.run(command, capture_output=True, timeout=90)
    (OUT / (label + "_stdout.txt")).write_bytes(result.stdout)
    (OUT / (label + "_stderr.txt")).write_bytes(result.stderr)
    info = dict(command=command, exit_code=result.returncode,
                repair=repair, repair_from=trim_endpoints, repair_to=trim_endpoints,
                ignore_errors=tolerate_errors, output=str(target))
    (OUT / (label + ".json")).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(phase=label, exit_code=result.returncode), ensure_ascii=False), flush=True)
    return target, info


def mapping(candidate):
    rows = []
    for key, before in original_edges.items():
        after = candidate.get(key)
        item = dict(id=key, previously_valid=key in previously_valid,
                    status="candidate" if after is not None else "unresolved",
                    original_first=before[0], original_last=before[-1], original_edge_count=len(before))
        if after is not None:
            item.update(candidate_first=after[0], candidate_last=after[-1],
                candidate_edge_count=len(after), first_changed=before[0] != after[0],
                last_changed=before[-1] != after[-1], route_changed=before != after,
                added_edge_occurrences=sum((Counter(after)-Counter(before)).values()),
                removed_edge_occurrences=sum((Counter(before)-Counter(after)).values()))
        rows.append(item)
    return rows


def summarize(candidate, strict, phase):
    rows = mapping(candidate)
    missing = sorted(set(original_edges)-set(candidate))
    accepted = routes_from(strict["output"])
    assert strict["exit_code"] == 0
    assert set(accepted) == set(candidate)
    assert all(accepted[k] == candidate[k] for k in candidate)
    return dict(phase=phase, strict_exit_code=strict["exit_code"],
        passed_vehicle_count=len(candidate), original_vehicle_count=len(original_edges),
        all_180_vehicle_ids_present=not missing, missing_vehicle_ids=missing,
        previously_valid_73_present=previously_valid <= set(candidate),
        previously_valid_73_routes_unchanged=all(candidate.get(key) == original_edges[key] for key in previously_valid),
        recovered_from_original_invalid=len(set(candidate)-previously_valid),
        changed_route_vehicles=sum(r.get("route_changed", False) for r in rows),
        changed_departure_edge_vehicles=sum(r.get("first_changed", False) for r in rows),
        changed_arrival_edge_vehicles=sum(r.get("last_changed", False) for r in rows),
        original_edge_occurrences_of_present_vehicles=sum(len(original_edges[key]) for key in candidate),
        candidate_edge_occurrences=sum(len(edges) for edges in candidate.values()),
        original_distinct_edges_of_present_vehicles=len({edge for key in candidate for edge in original_edges[key]}),
        candidate_distinct_edges=len({edge for edges in candidate.values() for edge in edges}),
        vehicle_mapping=rows)


# Collect partial repair results explicitly. Tolerated errors are not called success;
# every resulting candidate undergoes a separate no-repair strict validation.
phase1_raw, info1 = native("phase1_repair_same_endpoints", SOURCE, repair=True, tolerate_errors=True)
phase1 = routes_from(phase1_raw)
# Preserve all known-valid original routes even if the router proposed alternatives.
phase1.update({key: original_edges[key] for key in previously_valid})
assert all(edges[0] == original_edges[key][0] and edges[-1] == original_edges[key][-1]
           for key, edges in phase1.items())
phase1_candidate = OUT / "phase1_candidate_preserved_endpoints.rou.xml"
write_source_preserving(phase1_candidate, phase1)
strict1_path, strict1 = native("phase1_strict", phase1_candidate)
summary1 = summarize(phase1, strict1, "same_endpoints")
(OUT / "phase1_summary.json").write_text(json.dumps(summary1, ensure_ascii=False, indent=2), encoding="utf-8")

unresolved = {key: edges for key, edges in original_edges.items() if key not in phase1}
phase2_input = OUT / "phase2_unresolved_input.rou.xml"
write_source_preserving(phase2_input, unresolved)
phase2_raw, info2 = native("phase2_repair_allow_endpoint_trim", phase2_input,
                        repair=True, trim_endpoints=True, tolerate_errors=True)
phase2_additions = routes_from(phase2_raw)
assert not set(phase1) & set(phase2_additions)
phase2 = dict(phase1, **phase2_additions)
phase2_candidate = OUT / "phase2_candidate_endpoint_changes_review_only.rou.xml"
write_source_preserving(phase2_candidate, phase2)
strict2_path, strict2 = native("phase2_strict", phase2_candidate)
summary2 = summarize(phase2, strict2, "allow_endpoint_trim_for_remaining")
summary2["phase2_additional_repaired_vehicles"] = len(phase2_additions)
(OUT / "phase2_summary.json").write_text(json.dumps(summary2, ensure_ascii=False, indent=2), encoding="utf-8")

# Determine why the remaining original routes cannot be repaired by endpoint trimming.
route_audit = {v["id"]: group for group in audit["route_groups"] for v in group["vehicles"]}
missing2 = set(original_edges)-set(phase2)
no_usable_edges = sorted(key for key in missing2
    if set(original_edges[key]) <= set(route_audit[key]["forbidden_emergency_edges"]))
unchanged = all(digest(path) == expected for path, expected in source_hashes.items())
assert unchanged
result = dict(review_only=True, actual_simulation_or_training_started=False,
    source_inputs_unchanged=unchanged, source_hashes=source_hashes,
    original_vtype_and_vehicle_attributes_preserved=True,
    original_180_vehicle_ids_retained_in_mapping=True,
    phase1={k:v for k,v in summary1.items() if k != "vehicle_mapping"},
    phase2={k:v for k,v in summary2.items() if k != "vehicle_mapping"},
    unresolved_routes_without_any_emergency_usable_edge=no_usable_edges,
    unresolved_without_any_usable_edge_count=len(no_usable_edges),
    other_unresolved_vehicle_ids=sorted(missing2-set(no_usable_edges)),
    limitations=["Candidate files contain successfully repaired vehicles only. Missing original IDs are reported; no partial candidate was adopted for training.",
        "Phase 2 changes some departure and/or arrival edges. These changes have not been approved or applied to source assets.",
        "Strict route validation does not prove that a full SUMO episode will avoid the native NULL-pointer crash."])
(OUT / "repair_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"phase1_passed":len(phase1), "phase2_passed":len(phase2),
                  "phase2_additional":len(phase2_additions), "remaining":len(missing2),
                  "remaining_no_usable_edge":len(no_usable_edges),
                  "phase2_start_changes":summary2["changed_departure_edge_vehicles"],
                  "phase2_end_changes":summary2["changed_arrival_edge_vehicles"]},ensure_ascii=False), flush=True)
