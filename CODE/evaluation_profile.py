"""Restore saved patient/traffic conditions and freeze evaluation checkpoints."""
import copy
import math
from pathlib import Path

from experiment_config import SCENARIOS, scenario_for_config


def experiment_root(manifest_path):
    path = Path(manifest_path).resolve()
    return path.parent.parent if path.parent.name == "training" else path.parent


def freeze_evaluation_model(model_path, run_id, directory=None):
    """Copy first, then let the caller load only the immutable evaluation copy."""
    import torch
    from evaluation_audit import snapshot_evaluation_model
    from simulation_setup import resolve_manifest_path

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    manifest_path = resolve_manifest_path(model_path, checkpoint.get("metadata", {}))
    root = experiment_root(manifest_path)
    target = Path(directory) if directory is not None else root / "evaluations" / run_id
    snapshot = snapshot_evaluation_model(model_path, manifest_path, target)
    return Path(snapshot["model_path"]), root, snapshot


def resolve_evaluation_profile(manifest, config, scenario_id=None, source="saved"):
    """Use saved web demand by default; explicit standard mode uses CLI demand.

    A scenario override in saved mode selects traffic and, for calibrated web
    presets, its saved demand tier. Custom hourly/fixed-count demand is retained.
    Runtime route file paths are rebuilt by the common episode engine.
    """
    if source not in ("saved", "standard"):
        raise ValueError("Scenario profile must be saved or standard.")
    if scenario_id is not None and (type(scenario_id) is not int or not 0 <= scenario_id <= 9):
        raise ValueError("Scenario must be in 0..9.")
    web_scenario = manifest.get("web_scenario")
    saved = web_scenario if source == "saved" else None
    occupancy = (.20, .60)
    if isinstance(web_scenario, dict):
        occupancy = tuple(web_scenario.get("initial_occupancy_range", occupancy))
        if (len(occupancy) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                     or not math.isfinite(x) for x in occupancy)
                or not 0 <= occupancy[0] <= occupancy[1] <= 1):
            raise ValueError("Saved initial occupancy range is invalid.")
    if saved is not None:
        if not isinstance(saved, dict):
            raise ValueError("Saved web scenario must be an object.")
        template = {k: copy.deepcopy(v) for k, v in saved.items()
                    if not k.startswith("_") and k not in ("initial_occupancy_range", "background_traffic")}
        if type(template.get("id")) is not int or not 1 <= template["id"] <= 9:
            raise ValueError("Saved web scenario ID must be in 1..9.")
        if config.demand_rates_per_hour:
            raise ValueError("Saved web demand conflicts with CLI hourly demand; choose --scenario-profile standard.")
        ids = ([template["id"]] if scenario_id is None
               else list(range(1, 10)) if scenario_id == 0 else [scenario_id])
        profile = template.get("demand_profile", {})
        rates = profile.get("patients_per_hour")
        calibrated_preset = (template.get("demand_mode", "preset") == "preset"
                             and "fixed_patient_count" not in template and rates is not None)
        if calibrated_preset and (not isinstance(rates, (list, tuple)) or len(rates) != 3 or any(isinstance(rate, bool)
                or not isinstance(rate, (int, float)) or not math.isfinite(rate)
                or not 0 <= rate <= 3600 for rate in rates)):
            raise ValueError("Saved hourly preset must contain three rates in 0..3600.")
        scenarios = []
        for identity in ids:
            scenario = copy.deepcopy(template)
            if scenario_id is not None:
                scenario.update(id=identity, scale=SCENARIOS[identity - 1]["scale"])
                if calibrated_preset:
                    rate = rates[(identity - 1) // 3]
                    scenario.update(prob=rate / 3600., patients_per_hour=rate)
                scenario["name"] = f"saved_web_scenario{identity}_traffic{scenario['scale']}"
            scenarios.append(scenario)
        origin = "saved_web"
    else:
        selected = manifest.get("training_scenarios", [1])
        identity = scenario_id if scenario_id is not None else (selected[0] if len(selected) == 1 else 1)
        scenarios = [scenario_for_config(copy.deepcopy(s), config)
                     for s in (SCENARIOS if identity == 0 else [SCENARIOS[identity - 1]])]
        origin = "standard_override" if source == "standard" else "saved_cli"
    for scenario in scenarios:
        probability = scenario.get("prob")
        scale = scenario.get("scale")
        if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not math.isfinite(probability) or not 0 <= probability <= 1
                or isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not math.isfinite(scale) or scale < 0):
            raise ValueError("Saved scenario has invalid demand or traffic scale.")
        if "fixed_patient_count" in scenario:
            count = scenario["fixed_patient_count"]
            if type(count) is not int or not 0 <= count <= config.arrival_steps:
                raise ValueError("Saved fixed patient count must be in 0..arrival_steps.")
    return scenarios, occupancy, origin


def validate_evaluation_demands(scenarios, config, levels):
    from dataclasses import replace
    if not levels:
        raise ValueError("Provide at least one evaluation demand level.")
    for level in levels:
        replace(config, demand_scale=level)
        for scenario in scenarios:
            if "fixed_patient_count" in scenario and level != 1.:
                raise ValueError("Saved fixed-count demand requires --demand-levels 1.")
            if not 0 <= scenario["prob"] * level <= 1:
                raise ValueError("Scaled arrival probability must be in 0..1.")
