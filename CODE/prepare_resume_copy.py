"""Preflight a changed SUMO engine and prepare an isolated resume experiment.

No training is performed. Source checkpoints, manifests and histories stay intact.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import sumolib
import torch

from baseline import state_dimension
from dashboard_demand import demand_metadata
from dqn_agent import DQNAmbulanceAgent
from evaluation_audit import file_sha256
from experiment_config import EXPERIMENT_VERSION, LearningConfig, SimulationConfig
from experiment_log import new_run_id
from simulation_setup import (make_manifest, manifest_signature, prepare_environment,
                              resolve_manifest_path)

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def current_engine():
    binary = sumolib.checkBinary("sumo")
    path = Path(binary) if Path(binary).is_file() else Path(shutil.which(binary) or binary)
    if not path.is_file():
        raise FileNotFoundError(f"SUMO binary not found: {binary}")
    path = path.resolve()
    result = subprocess.run([str(path), "--version"], capture_output=True, text=True,
                            check=True, timeout=10)
    line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    match = re.search(r"(?:\bVersion\s+|\bsumo(?:-gui)?\s+)(\d+\.\d+(?:\.\d+)?)", line, re.IGNORECASE)
    return dict(path=str(path), sha256=file_sha256(path), version_line=line,
                version=match.group(1) if match else None)


def validate_recorded_engine(output_root):
    """A copied experiment must not accidentally resume on its old engine."""
    transition = Path(output_root) / "engine_transition.json"
    if not transition.is_file():
        return None
    recorded = json.loads(transition.read_text(encoding="utf-8"))
    expected = recorded.get("resumed_engine", {})
    if not expected.get("sha256") or not expected.get("path"):
        raise ValueError("The copied experiment lacks its recorded SUMO engine identity.")
    actual = current_engine()
    if actual["sha256"] != expected["sha256"]:
        raise ValueError(
            f"This resume copy requires SUMO {expected.get('version')} at {expected['path']}. "
            f"The selected engine is {actual.get('version')} at {actual['path']}. "
            "Set SUMO_BINARY to the verified portable engine before resuming.")
    return actual


def _source_engine(report):
    if report is None:
        default = ROOT / "results/diagnostics/sumo_crash_research_20261117.json"
        report = default if default.is_file() else None
    if report is None:
        return dict(status="not_recorded"), None
    path = Path(report).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    engine = copy.deepcopy(data.get("installed_binary", data))
    engine["evidence"] = "prior_diagnostic_report"
    return engine, str(path)


def _copy_exact(source, destination):
    source, destination = Path(source), Path(destination)
    expected = file_sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())
    if file_sha256(source) != expected or file_sha256(destination) != expected:
        raise ValueError(f"Source changed while copying: {source}")
    return expected


def prepare_resume_copy(checkpoint, output_root, preflight_root=None, source_engine_report=None):
    source = Path(checkpoint).resolve()
    target = Path(output_root).resolve()
    if target.exists():
        raise FileExistsError(f"Resume destination already exists; no files were overwritten: {target}")
    source_hash = file_sha256(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    required = ("replay_memory", "optimizer_state_dict", "python_random_state",
                "numpy_random_state", "torch_random_state")
    if any(key not in payload for key in required):
        raise ValueError("A full training checkpoint with replay, optimizer and RNG states is required.")
    if payload.get("metadata", {}).get("best_validation_reward") is not None:
        raise ValueError("This cross-engine copy requires a checkpoint before first best selection; "
                         "an inherited best score needs an explicit new-engine selection policy.")
    manifest_source = resolve_manifest_path(source, payload.get("metadata", {}))
    original = json.loads(manifest_source.read_text(encoding="utf-8"))
    manifest_hash = file_sha256(manifest_source)
    if original.get("version") != EXPERIMENT_VERSION:
        raise ValueError("Resume-copy preflight requires the current experiment version.")
    source_root = (manifest_source.parent.parent if manifest_source.parent.name == "training"
                   else manifest_source.parent)
    if target.is_relative_to(source_root):
        raise ValueError("Use a separate experiment root outside the source experiment.")
    config = SimulationConfig(**original["config"])
    learning = LearningConfig(**original["learning_config"])
    hospital_count = len(original["hospitals"])
    agent = DQNAmbulanceAgent(state_dimension(hospital_count, config.observation_version), hospital_count,
                             architecture=original["agent_architecture"], learning_config=learning)
    agent.metadata = dict(experiment_signature=original["signature"])
    agent.load_model(str(source), resume=True)
    completed = agent.loaded_episode
    learning_stats = agent.training_statistics()
    del agent
    work_root = Path(preflight_root or ROOT / "results/diagnostics").resolve()
    if work_root.is_relative_to(source_root):
        raise ValueError("Keep diagnostic output outside the source experiment.")
    work = work_root / new_run_id("resume_preflight")
    work.mkdir(parents=True, exist_ok=False)
    engine = current_engine()
    old_engine, old_engine_report = _source_engine(source_engine_report)
    virtual_count = sum(bool(h.get("virtual")) for h in original["hospitals"])
    if virtual_count not in (0, 2, 5):
        raise ValueError("This CLI resume-copy preflight requires a CLI-compatible virtual hospital count.")
    # Reproduce train() preparation and manifest construction without training.
    env, hospitals, edges, access = prepare_environment(config, virtual_count=virtual_count,
        virtual_strategy="far", placement_seed=42, output_dir=work / "setup", quiet=True)
    rebuilt = make_manifest(env, hospitals, edges, access, config,
        original["ambulance_count"], original["queue_policy"], original["aging_seconds"],
        original.get("curriculum", ()), learning_config=learning)
    for key in ("validation_seeds", "validation_scenarios", "validation_demands", "cycle_curriculum"):
        if key in original:
            rebuilt[key] = copy.deepcopy(original[key])
    if config.demand_rates_per_hour:
        rebuilt["demand_profile"] = demand_metadata()
    if "training_scenarios" in original:
        rebuilt["training_scenarios"] = copy.deepcopy(original["training_scenarios"])
    rebuilt["signature"] = manifest_signature(rebuilt)
    _write_json(work / "rebuilt_manifest.json", rebuilt)
    differences = [key for key in sorted(set(original) | set(rebuilt)) if original.get(key) != rebuilt.get(key)]
    report = dict(source_checkpoint=str(source), source_manifest=str(manifest_source),
        source_signature=original["signature"], rebuilt_signature=rebuilt["signature"],
        manifests_identical=original == rebuilt, differing_fields=differences,
        target=str(target), source_completed_episode=completed, engine=engine)
    _write_json(work / "preflight.json", report)
    if differences:
        raise ValueError(f"Prepared manifest differs from the saved experiment: {differences}. "
                         f"No resume copy was created. Inspect {work / 'preflight.json'}")
    if file_sha256(source) != source_hash or file_sha256(manifest_source) != manifest_hash:
        raise ValueError("The source experiment changed during preflight; retry from a stable checkpoint.")
    if file_sha256(engine["path"]) != engine["sha256"]:
        raise ValueError("The selected SUMO binary changed during preflight.")
    if target.exists():
        raise FileExistsError(f"Resume destination appeared during preflight: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / new_run_id("resume_staging")
    staging.mkdir(exist_ok=False)
    tag = f"v{original['version']}_{original['signature']}"
    copies = {}
    checkpoint_relative = Path("training") / f"checkpoint_{tag}.pth"
    manifest_relative = Path("training") / f"experiment_{tag}.json"
    copies[str(checkpoint_relative)] = _copy_exact(source, staging / checkpoint_relative)
    copies[str(manifest_relative)] = _copy_exact(manifest_source, staging / manifest_relative)
    source_last = source_root / "models" / f"dqn_{tag}_last.pth"
    last_note = "not_present"
    if source_last.is_file():
        last_payload = torch.load(source_last, map_location="cpu", weights_only=False)
        if (last_payload.get("episode") == completed
                and last_payload.get("metadata", {}).get("experiment_signature") == original["signature"]):
            relative = Path("models") / source_last.name
            copies[str(relative)] = _copy_exact(source_last, staging / relative)
            last_note = "copied_matching_episode"
        else:
            last_note = "not_copied_episode_or_signature_mismatch"
    transition = dict(schema_version=1, source_experiment=str(source_root),
        resumed_experiment=str(target), source_checkpoint=str(source),
        source_checkpoint_sha256=source_hash, source_manifest_sha256=manifest_hash,
        experiment_signature=original["signature"], preflight=str(work / "preflight.json"),
        source_completed_episode=completed, resume_start_episode=completed + 1,
        learning_statistics=learning_stats, source_engine=old_engine,
        source_engine_report=old_engine_report, resumed_engine=engine,
        training_provenance="mixed_engine_continuation",
        explanation="The copied checkpoint retains training from the source engine. Future episodes use the recorded resumed engine; this is not a single-engine training run from episode 1.",
        source_history=str(source_root / "experiment_history.csv"), source_history_copied=False,
        source_best_copied=False, source_last=last_note, copied_files_sha256=copies)
    _write_json(staging / "engine_transition.json", transition)
    if file_sha256(source) != source_hash or file_sha256(manifest_source) != manifest_hash:
        raise ValueError(f"Source changed before publishing the resume copy. Staging retained at {staging}")
    staging.rename(target)
    result = dict(output_root=str(target), checkpoint=str(target / checkpoint_relative),
        manifest=str(target / manifest_relative), preflight=str(work / "preflight.json"),
        engine_transition=str(target / "engine_transition.json"), source_completed_episode=completed,
        resume_start_episode=completed + 1, manifests_identical=True,
        source_checkpoint_unchanged=True, source_manifest_unchanged=True, copied_files_sha256=copies)
    _write_json(work / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path)
    parser.add_argument("--source-engine-report", type=Path)
    args = parser.parse_args()
    try:
        result = prepare_resume_copy(args.checkpoint, args.output_root, args.preflight_root,
                                     args.source_engine_report)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
