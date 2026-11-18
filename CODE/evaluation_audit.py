"""Frozen evaluation artifacts and path-independent runtime-condition evidence."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from emergency_yield import yield_settings
from hospital_policy import HOSPITAL_LEVEL, REQUIRED_LEVEL
from reward import DEFAULT_UNCERTAINTY, GOLDEN_TIME_LIMITS, SUITABILITY, WEIGHTS


def _normalize(value):
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _json_bytes(value):
    return json.dumps(_normalize(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_stable(source, target):
    """A held source descriptor survives atomic replacement of a live best file."""
    temporary = target.with_name(target.name + ".tmp")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            before = os.fstat(reader.fileno())
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)
                digest.update(block)
            after = os.fstat(reader.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("Evaluation source changed in place while being copied; retry evaluation.")
            writer.flush()
            os.fsync(writer.fileno())
        if target.exists():
            raise FileExistsError(f"Evaluation snapshot already exists: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def snapshot_evaluation_model(model_path, manifest_path, directory):
    """Copy to a fresh run directory; load this model copy for the evaluation.

    The adjacent experiment.json keeps a copied checkpoint portable despite its
    embedded source-manifest path. Existing snapshots are never overwritten.
    """
    model_path, manifest_path = Path(model_path).resolve(), Path(manifest_path).resolve()
    if not model_path.name.endswith("_best.pth"):
        raise ValueError("Evaluation snapshots require a *_best.pth model.")
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("The evaluation model or its matching manifest is missing.")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    models = directory / "models"
    models.mkdir()
    frozen_model = models / model_path.name
    frozen_manifest = directory / "experiment.json"
    model_digest = _copy_stable(model_path, frozen_model)
    manifest_digest = _copy_stable(manifest_path, frozen_manifest)
    metadata = dict(model_path=str(frozen_model), model_sha256=model_digest,
                    manifest_path=str(frozen_manifest), manifest_sha256=manifest_digest,
                    source_model_path=str(model_path), source_manifest_path=str(manifest_path))
    descriptor = directory / "snapshot.json"
    temporary = directory / "snapshot.json.tmp"
    temporary.write_bytes(_json_bytes(metadata))
    os.replace(temporary, descriptor)
    return metadata


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _xml_value(element, omit_output_paths=False):
    attributes = dict(element.attrib)
    if omit_output_paths and _tag(element) in ("edgeData", "laneData", "timedEvent"):
        attributes.pop("file", None)
        attributes.pop("dest", None)
    return dict(tag=_tag(element), attributes=attributes,
                text=(element.text or "").strip(),
                children=[_xml_value(child, omit_output_paths) for child in element])


def _input_file(path, *, additional=False):
    path = Path(path)
    if not path.is_file():
        return dict(status="missing", sha256=None)
    if additional:
        # SUMO output destinations do not affect vehicle or patient dynamics.
        try:
            value = _xml_value(ET.parse(path).getroot(), omit_output_paths=True)
            return dict(status="recorded", sha256=hashlib.sha256(_json_bytes(value)).hexdigest())
        except ET.ParseError:
            pass
    return dict(status="recorded", sha256=file_sha256(path))


def _sumo_version():
    try:
        import sumolib
        result = subprocess.run([sumolib.checkBinary("sumo"), "--version"], capture_output=True,
                                text=True, timeout=10, check=True)
        return result.stdout.splitlines()[0].strip() or "unavailable"
    except (OSError, RuntimeError, subprocess.SubprocessError, IndexError):
        return "unavailable"


def _input_evidence(env):
    cfg_file = getattr(env, "cfg_file", None)
    net_file = getattr(env, "net_file", None)
    evidence = dict(status="unavailable", sumo_version="unavailable",
                    network=_input_file(net_file) if net_file else None,
                    configuration_network=None, configuration=None, route_files=[], additional_files=[])
    if not cfg_file or not Path(cfg_file).is_file():
        return evidence
    cfg_file = Path(cfg_file)
    root = ET.parse(cfg_file).getroot()
    runtime = [_xml_value(child) for child in root
               if _tag(child) not in ("input", "output", "report", "gui_only")]
    evidence.update(status="recorded", sumo_version=_sumo_version(),
                    configuration=hashlib.sha256(_json_bytes(runtime)).hexdigest())
    input_node = next((child for child in root if _tag(child) == "input"), None)
    if input_node is not None:
        for element in input_node:
            name = _tag(element)
            paths = [(cfg_file.parent / value.strip()).resolve()
                     for value in element.get("value", "").split(",") if value.strip()]
            if name == "net-file" and paths:
                evidence["configuration_network"] = _input_file(paths[0])
                if net_file is None:
                    evidence["network"] = evidence["configuration_network"]
            elif name in ("route-files", "additional-files"):
                evidence[name.replace("-", "_")] = [
                    _input_file(path, additional=name == "additional-files") for path in paths]
            elif name != "net-file":
                # Keep other input controls visible without leaking directory paths.
                if name.endswith(("-file", "-files")):
                    evidence[name.replace("-", "_")] = [_input_file(path) for path in paths]
                else:
                    evidence[name] = dict(element.attrib)
    return evidence


def _environment_digest(schema_version, settings):
    return hashlib.sha256(_json_bytes(dict(schema_version=schema_version, settings=settings))).hexdigest()


def validate_evaluation_environment(environment):
    if (not isinstance(environment, dict) or environment.get("schema_version") != 1
            or not isinstance(environment.get("settings"), dict)):
        raise ValueError("Invalid recorded evaluation environment.")
    expected = _environment_digest(environment["schema_version"], environment["settings"])
    if environment.get("fingerprint") != expected:
        raise ValueError("Evaluation environment fingerprint does not match its recorded settings.")
    return expected


def build_evaluation_environment(env, hospitals, config, scenarios, demand_levels,
                                 ambulance_count, queue_policy, aging_seconds,
                                 initial_occupancy_range=None):
    """Record applied dynamics, excluding the model's training choices and seed.

    Policy architecture/observation/learning settings intentionally remain free
    to differ in model comparisons. Input paths and telemetry outputs do not
    enter the fingerprint; original input contents and runtime controls do.
    """
    physical_fields = ("arrival_steps", "max_wait", "mission_timeout", "treatment_seconds",
                       "bed_fraction", "occupancy_limit", "reward_version", "admission_mode",
                       "congestion_reward_mode", "candidate_limit")
    hospital_fields = ("id", "type", "virtual", "sumo_x", "sumo_y", "accepting",
                       "arrival_stop_lane", "arrival_stop_pos", "hospital_edge")
    scenario_fields = ("id", "prob", "scale", "fixed_patient_count")
    origins = list(getattr(env, "ambulance_edges", ()))
    access = getattr(env, "hospital_access_edges", None)
    effective_origins, effective_access, safety = origins, access, None
    if (config.admission_mode == "arrival" and access is not None
            and callable(getattr(getattr(env, "net", None), "getEdges", None))):
        from route_safety import prepare_access
        effective_access, prepared = prepare_access(env, hospitals, access)
        if prepared is not None:
            effective_origins, safety = prepared
    background_enabled = getattr(config, "repeat_background_traffic", False)
    if background_enabled:
        from dashboard_traffic import PERIOD
        background = dict(repeat=True, period_seconds=PERIOD,
                          implementation="periodic_original_passenger_v1")
    else:
        background = dict(repeat=False, period_seconds=None, implementation="original_inputs")
    settings = dict(
        simulation={name: getattr(config, name) for name in physical_fields},
        scenarios=[{key: scenario[key] for key in scenario_fields if key in scenario}
                   for scenario in scenarios],
        demand_levels=list(demand_levels), ambulance_count=ambulance_count,
        queue_policy=queue_policy, aging_seconds=aging_seconds,
        initial_occupancy_range=list((.20, .60) if initial_occupancy_range is None
                                     else initial_occupancy_range),
        hospitals=[{key: hospital[key] for key in hospital_fields if key in hospital}
                   for hospital in hospitals],
        ambulance_edges=origins, hospital_access_edges=access,
        effective_ambulance_edges=effective_origins, effective_hospital_access_edges=effective_access,
        route_safety=safety,
        emergency_yield=yield_settings(config.emergency_yield), background_traffic=background,
        reward_weights=asdict(WEIGHTS),
        golden_time_limits={str(key): value for key, value in GOLDEN_TIME_LIMITS.items()},
        reward_suitability={str(key): value for key, value in SUITABILITY.items()},
        severity_uncertainty=DEFAULT_UNCERTAINTY,
        required_hospital_levels={str(key): value for key, value in REQUIRED_LEVEL.items()},
        hospital_levels=HOSPITAL_LEVEL,
        inputs=_input_evidence(env),
    )
    environment = dict(schema_version=1, settings=settings)
    environment["fingerprint"] = _environment_digest(1, settings)
    return environment
