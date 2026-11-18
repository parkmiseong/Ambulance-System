"""Freeze a validated background-ambulance subset for one diagnostic replay."""
import copy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _node(element):
    """Ignore formatting whitespace while retaining all operative XML content."""
    return (element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(),
            tuple((_node(child), (child.tail or "").strip()) for child in element))


def _vehicles(root):
    result = {}
    for element in root.findall("vehicle"):
        identifier = element.get("id")
        if not identifier or identifier in result:
            raise ValueError("Every background vehicle requires a unique nonempty ID.")
        result[identifier] = element
    return result


def validate_vehicle_subset(original_bytes, candidate_bytes):
    original, candidate = ET.fromstring(original_bytes), ET.fromstring(candidate_bytes)
    if original.tag != "routes" or candidate.tag != original.tag or candidate.attrib != original.attrib:
        raise ValueError("The candidate must preserve the original routes root and attributes.")
    # This preserves vType, its params, shared route definitions, and any other
    # operative non-vehicle input. Only whole vehicle elements may be omitted.
    original_definitions = [_node(child) for child in original if child.tag != "vehicle"]
    candidate_definitions = [_node(child) for child in candidate if child.tag != "vehicle"]
    if candidate_definitions != original_definitions:
        raise ValueError("Background vType, params and shared definitions must remain unchanged.")
    source_vehicles, kept_vehicles = _vehicles(original), _vehicles(candidate)
    kept_ids = list(kept_vehicles)
    if not kept_ids or not set(kept_ids) < set(source_vehicles):
        raise ValueError("The candidate vehicle IDs must be a nonempty strict subset of the original IDs.")
    expected_order = [identifier for identifier in source_vehicles if identifier in kept_vehicles]
    if kept_ids != expected_order:
        raise ValueError("Retained background vehicles must preserve their original order.")
    for identifier, vehicle in kept_vehicles.items():
        if _node(vehicle) != _node(source_vehicles[identifier]):
            raise ValueError(f"Retained background vehicle attributes or route changed: {identifier}")
    # Preserve definition/vehicle interleaving as well as the two individual
    # orders; moving a definition can change when SUMO resolves that reference.
    expected = copy.deepcopy(original)
    for child in list(expected):
        if child.tag == "vehicle" and child.get("id") not in kept_vehicles:
            expected.remove(child)
    if _node(candidate) != _node(expected):
        raise ValueError("The candidate may only remove complete vehicle elements.")
    dropped_ids = [identifier for identifier in source_vehicles if identifier not in kept_vehicles]
    return dict(original_vehicle_count=len(source_vehicles), kept_vehicle_count=len(kept_ids),
                dropped_vehicle_count=len(dropped_ids), kept_vehicle_ids=kept_ids,
                dropped_vehicle_ids=dropped_ids)


def _absolute_list(value, base):
    return [str((base / entry.strip()).resolve()) for entry in value.split(",") if entry.strip()]


def prepare_diagnostic_route_input(cfg_file, override_file, diagnostic_directory):
    """Copy inputs exclusively; source cfg/routes and manifest stay untouched.

    The copied cfg remains compatible with prepare_background(), so passenger
    traffic continues to repeat through the original shared implementation.
    """
    original_cfg, candidate_route = Path(cfg_file).resolve(), Path(override_file).resolve()
    cfg_bytes, candidate_bytes = original_cfg.read_bytes(), candidate_route.read_bytes()
    tree = ET.ElementTree(ET.fromstring(cfg_bytes))
    cfg_root = tree.getroot()
    route_nodes = cfg_root.findall("input/route-files")
    if len(route_nodes) != 1:
        raise ValueError("Exactly one route-files entry is required in the source SUMO config.")
    route_files = _absolute_list(route_nodes[0].get("value", ""), original_cfg.parent)
    matches = [index for index, route in enumerate(route_files)
               if Path(route).name == "ambulance_routes.rou.xml"]
    if len(matches) != 1:
        raise ValueError("Exactly one original ambulance_routes.rou.xml input is required.")
    original_route = Path(route_files[matches[0]])
    original_bytes = original_route.read_bytes()
    subset = validate_vehicle_subset(original_bytes, candidate_bytes)
    input_directory = Path(diagnostic_directory).resolve() / "input"
    applied_cfg = input_directory / original_cfg.name
    applied_route = input_directory / "ambulance_routes.filtered.rou.xml"
    if applied_cfg.exists() or applied_route.exists():
        raise FileExistsError("Diagnostic input files already exist; no files were overwritten.")
    if applied_cfg in (original_cfg, original_route, candidate_route) or applied_route in (original_cfg, original_route, candidate_route):
        raise ValueError("Diagnostic copies must be separate from all source inputs.")
    # Resolve all path-valued input options plus GUI settings against the
    # original cfg directory. Output paths are managed by start_sumo().
    input_node = cfg_root.find("input")
    for child in input_node if input_node is not None else ():
        if (child.tag.endswith(("-file", "-files")) or child.tag == "load-state") and child.get("value"):
            child.set("value", ",".join(_absolute_list(child.get("value"), original_cfg.parent)))
    for node in cfg_root.findall(".//gui-settings-file"):
        if node.get("value"):
            node.set("value", ",".join(_absolute_list(node.get("value"), original_cfg.parent)))
    route_files[matches[0]] = str(applied_route)
    route_nodes[0].set("value", ",".join(route_files))
    input_directory.mkdir(parents=True, exist_ok=True)
    with applied_route.open("xb") as stream:
        stream.write(candidate_bytes)
    with applied_cfg.open("xb") as stream:
        tree.write(stream, encoding="utf-8", xml_declaration=True)
    sources = dict(original_cfg=(original_cfg, cfg_bytes), original_route=(original_route, original_bytes),
                   candidate_route=(candidate_route, candidate_bytes))
    for name, (path, before) in sources.items():
        if path.read_bytes() != before:
            raise ValueError(f"Source input changed during diagnostic preparation: {name}")
    if applied_route.read_bytes() != candidate_bytes:
        raise ValueError("The frozen diagnostic route does not match the validated candidate.")
    metadata = dict(enabled=True, diagnostic_only=True, operation="background_ambulance_vehicle_subset",
                    applied_cfg=str(applied_cfg), applied_cfg_sha256=_sha(applied_cfg.read_bytes()),
                    applied_route=str(applied_route), applied_route_sha256=_sha(candidate_bytes),
                    original_inputs_unchanged_after_preparation=True, **subset)
    for name, (path, data) in sources.items():
        metadata[name] = str(path)
        metadata[name + "_sha256"] = _sha(data)
    return metadata
