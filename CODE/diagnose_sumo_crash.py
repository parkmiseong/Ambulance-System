"""Replay the next training episode from an isolated full-checkpoint copy.

Never saves model weights or appends to the source experiment's history.
The same learning state, seed, hospital order and driving settings are used.
"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import traceback

from evaluation_audit import file_sha256
from experiment_config import SCENARIOS, curriculum_multiplier, scenario_for_config
from experiment_log import new_run_id
from simulation import run_episode
from simulation_setup import load_experiment, resolve_manifest_path
from sumo_lane_snapshot import LaneSnapshotProbe
from diagnostic_route_input import prepare_diagnostic_route_input

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(target)


def copy_unchanged(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    before = file_sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Diagnostic snapshot already exists: {target}")
    shutil.copyfile(source, target)
    if file_sha256(source) != before or file_sha256(target) != before:
        raise ValueError(f"Source changed while creating its diagnostic copy: {source}")
    return before


def snapshot_checkpoint(checkpoint, directory):
    import torch
    source = Path(checkpoint).resolve()
    directory = Path(directory).resolve()
    # Read a manifest reference, then load the copied payload for all execution.
    initial = torch.load(source, map_location="cpu", weights_only=False)
    if "replay_memory" not in initial or "python_random_state" not in initial:
        raise ValueError("A full training checkpoint is required; best/last model weights are insufficient.")
    manifest_source = resolve_manifest_path(source, initial.get("metadata", {}))
    directory.mkdir(parents=True, exist_ok=False)
    model_copy = directory / "models" / source.name
    manifest_copy = directory / "experiment.json"
    source_hash = copy_unchanged(source, model_copy)
    manifest_hash = copy_unchanged(manifest_source, manifest_copy)
    evidence = dict(checkpoint=str(model_copy), manifest=str(manifest_copy),
                    source_checkpoint=str(source), source_manifest=str(manifest_source),
                    checkpoint_sha256=source_hash, manifest_sha256=manifest_hash)
    write_json(directory / "snapshot.json", evidence)
    return evidence


def resolve_episode(manifest, config, completed_episode):
    episode = completed_episode + 1
    selected = manifest.get("training_scenarios", list(range(1, 10)))
    if not selected:
        raise ValueError("The recorded training scenario list is empty.")
    scenario_id = selected[(episode - 1) % len(selected)]
    scenario = scenario_for_config(SCENARIOS[scenario_id - 1], config)
    multiplier = curriculum_multiplier(episode, manifest.get("curriculum", ()),
                                       cycle=manifest.get("cycle_curriculum", False))
    episode_config = replace(config, demand_scale=config.demand_scale * multiplier)
    return episode, scenario, config.seed + 10000 + episode, episode_config


def run_diagnostic(checkpoint, output_root=None, *, lane_probe_start=None,
                   lane_probe_end=None, save_native_state=False,
                   background_route_override=None):
    root = Path(output_root or PROJECT_ROOT / "results" / "diagnostics").resolve()
    directory = root / new_run_id("sumo_crash")
    logging_errors = []
    lane_probe = LaneSnapshotProbe(directory, lane_probe_start, lane_probe_end,
                                   save_native_state=save_native_state,
                                   logging_errors=logging_errors)
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = snapshot_checkpoint(checkpoint, directory / "snapshot")
    frozen = Path(snapshot["checkpoint"])
    env, hospitals, agent, config, manifest = load_experiment(frozen)
    agent.load_model(str(frozen), resume=True)
    agent.metadata["manifest_path"] = snapshot["manifest"]
    agent.model.train()
    env.sumo_diagnostics = True
    route_input = None
    if background_route_override is not None:
        route_input = prepare_diagnostic_route_input(env.cfg_file, background_route_override, directory)
        env = copy.copy(env)
        env.cfg_file = Path(route_input["applied_cfg"])
    episode, scenario, seed, episode_config = resolve_episode(manifest, config, agent.loaded_episode)
    sumo_dir = directory / "sumo" / f"episode{episode}"
    before_statistics = copy.deepcopy(agent.training_statistics())
    descriptor = dict(
        diagnostic_only=True, saves_model_weights=False, snapshot=snapshot,
        previous_episode=agent.loaded_episode, episode=episode, seed=seed, scenario=scenario,
        config=episode_config.to_dict(), learning_config=agent.learning_config.to_dict(),
        ambulance_count=manifest["ambulance_count"], queue_policy=manifest["queue_policy"],
        aging_seconds=manifest["aging_seconds"], epsilon=agent.epsilon,
        replay_size=len(agent.memory), learning_before=before_statistics,
        sumo_output=str(sumo_dir),
        lane_probe=lane_probe.settings(),
        background_route_override=route_input,
    )
    write_json(directory / "diagnostic_run.json", descriptor)
    print(f"[Diagnostic] checkpoint episode={agent.loaded_episode} updates={agent.training_updates} "
          f"epsilon={agent.epsilon:.6f}", flush=True)
    print(f"[Diagnostic] replaying episode={episode} scenario={scenario['id']} seed={seed}", flush=True)
    print(f"[Diagnostic] output={directory}", flush=True)
    events = directory / "events.jsonl"
    latest_state = None

    def event(kind, row):
        try:
            with events.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(kind=kind, **row), ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            logging_errors.append(f"event {kind}: {exc}")

    def observe(state):
        nonlocal latest_state
        manager = state["manager"]
        missions = []
        for vehicle, mission in manager.active.items():
            selected = mission.payload[1] if mission.payload is not None else {}
            missions.append(dict(vehicle=vehicle, patient_id=mission.patient.get("patient_id"),
                ktas=mission.patient.get("ktas"), hospital_id=selected.get("id"),
                dispatch_time=mission.dispatch_time, route=selected.get("route_edges"),
                arrival_stop_pos=selected.get("arrival_stop_pos"),
                arrival_stop_lane=selected.get("arrival_stop_lane")))
        # Keep the tick in memory: Windows/OneDrive may temporarily lock a file
        # during replacement. Diagnostic I/O must not stop the simulation.
        latest_state = dict(
            time=state["time"], waiting=len(manager.waiting), completed=len(state["outcomes"]),
            active=missions, learning_updates=agent.training_updates,
            hospitals=[dict(id=h["id"], occupancy=h.get("occupancy"),
                            admitted_count=h.get("admitted_count")) for h in state["hospitals"]])
        lane_probe.observe(state["time"], manager.active)

    result, failure = None, None
    try:
        result = run_episode(env, hospitals, scenario, seed, "dqn", episode_config, agent=agent,
            training=True, ambulance_count=manifest["ambulance_count"],
            queue_policy=manifest["queue_policy"], aging_seconds=manifest["aging_seconds"],
            output_dir=sumo_dir, quiet=True, on_outcome=lambda row: event("mission", row),
            on_decision=lambda row: event("decision", row), on_step=observe)
    except (Exception, KeyboardInterrupt) as exc:
        failure = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        (directory / "python_traceback.txt").write_text(failure["traceback"], encoding="utf-8")
    finally:
        def unchanged(path, expected):
            try:
                return file_sha256(path) == expected
            except OSError as exc:
                logging_errors.append(f"source integrity: {exc}")
                return False
        unchanged_checkpoint = unchanged(snapshot["source_checkpoint"], snapshot["checkpoint_sha256"])
        unchanged_manifest = unchanged(snapshot["source_manifest"], snapshot["manifest_sha256"])
        route_integrity = None
        if route_input is not None:
            route_integrity = {name: unchanged(route_input[name], route_input[name + "_sha256"])
                               for name in ("original_cfg", "original_route", "candidate_route",
                                            "applied_cfg", "applied_route")}
        if latest_state is not None:
            try:
                write_json(directory / "last_successful_state.json", latest_state)
            except OSError as exc:
                logging_errors.append(f"last successful state: {exc}")
        disconnected = bool(failure and failure["type"] == "FatalTraCIError"
                            and "Connection closed by SUMO" in failure["message"])
        status = ("reproduced_sumo_disconnect" if disconnected else "diagnostic_failed") if failure else "completed_without_failure"
        if failure and failure["type"] == "KeyboardInterrupt":
            status = "interrupted"
        summary = dict(status=status, expected_disconnect_reproduced=disconnected,
            episode=episode, seed=seed, scenario_id=scenario["id"], failure=failure,
            result=result, source_checkpoint_unchanged=unchanged_checkpoint,
            source_manifest_unchanged=unchanged_manifest,
            learning_before=before_statistics, learning_after=agent.training_statistics(),
            model_weights_saved=False, last_successful_state=latest_state,
            lane_probe=lane_probe.statistics(),
            background_route_override=route_input,
            background_route_input_integrity=route_integrity,
            logging_errors=logging_errors, output=str(directory))
        write_json(directory / "diagnostic_summary.json", summary)
    print(f"[Diagnostic] status={summary['status']} source_checkpoint_unchanged={unchanged_checkpoint}", flush=True)
    if failure:
        print(f"[Diagnostic] {failure['type']}: {failure['message']}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Full checkpoint_*.pth from training/, including replay and RNG.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "diagnostics")
    parser.add_argument("--lane-probe-start", type=float,
                        help="First simulation second for opt-in detailed lane snapshots.")
    parser.add_argument("--lane-probe-end", type=float,
                        help="Last simulation second for lane snapshots (inclusive).")
    parser.add_argument("--save-native-state", action="store_true",
                        help="Export SUMO state once at the first observed tick in the probe window.")
    parser.add_argument("--background-route-override", type=Path,
                        help="Diagnostic-only strict subset of unchanged background ambulance vehicles.")
    args = parser.parse_args()
    try:
        summary = run_diagnostic(args.checkpoint, args.output_root,
            lane_probe_start=args.lane_probe_start, lane_probe_end=args.lane_probe_end,
            save_native_state=args.save_native_state,
            background_route_override=args.background_route_override)
    except ValueError as exc:
        parser.error(str(exc))
    if summary["failure"] is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
