"""Resume the verified SUMO 1.28 copy up to a cumulative episode target."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "all_scenarios_normal15_yield_repeat_sumo128_v6"


def resume_to_target(output_root=DEFAULT_OUTPUT, target_episodes=45, eval_every=9):
    import torch
    from experiment_config import LearningConfig, SimulationConfig
    from simulation_setup import manifest_signature

    root = Path(output_root).resolve()
    transition_path = root / "engine_transition.json"
    if not transition_path.is_file():
        raise FileNotFoundError("The verified SUMO 1.28 resume copy is not prepared: " + str(root))
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    signature = transition["experiment_signature"]
    manifests = list((root / "training").glob(f"experiment_v*_{signature}.json"))
    if len(manifests) != 1:
        raise ValueError("A unique matching experiment manifest is required.")
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_signature(manifest) != signature or manifest.get("signature") != signature:
        raise ValueError("The resume manifest no longer matches the verified experiment.")
    checkpoint = root / "training" / f"checkpoint_v{manifest['version']}_{signature}.pth"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("metadata", {}).get("experiment_signature") != signature:
        raise ValueError("The resume checkpoint and manifest do not match.")
    previous = int(payload.get("episode") or 0)
    if type(target_episodes) is not int or target_episodes < 1 or type(eval_every) is not int or eval_every < 1:
        raise ValueError("Target episodes and validation interval must be positive integers.")
    if previous >= target_episodes:
        print(f"[Resume] already completed {previous} episodes; target={target_episodes}.", flush=True)
        return checkpoint
    binary = Path(transition["resumed_engine"]["path"])
    if not binary.is_file():
        raise FileNotFoundError("The verified portable SUMO binary is missing: " + str(binary))
    previous_binary = os.environ.get("SUMO_BINARY")
    os.environ["SUMO_BINARY"] = str(binary)
    try:
        from prepare_resume_copy import validate_recorded_engine
        engine = validate_recorded_engine(root)
        from train_agent import train
        config = SimulationConfig(**manifest["config"])
        additional = target_episodes - previous
        print(f"[Resume] previous={previous} additional={additional} target_total={target_episodes}", flush=True)
        print(f"[Resume] SUMO={engine['version']} output={root}", flush=True)
        return train(num_episodes=additional, ambulance_count=manifest["ambulance_count"],
            eval_every=eval_every, resume=True, queue_policy=manifest["queue_policy"],
            aging_seconds=manifest["aging_seconds"],
            virtual_count=sum(bool(h.get("virtual")) for h in manifest["hospitals"]),
            config=config, curriculum=tuple(manifest.get("curriculum", ())),
            validation_seeds=tuple(manifest["validation_seeds"]),
            validation_scenarios=tuple(manifest["validation_scenarios"]),
            validation_demands=tuple(manifest["validation_demands"]),
            cycle_curriculum=manifest.get("cycle_curriculum", False),
            training_scenarios=manifest.get("training_scenarios"),
            learning_config=LearningConfig(**manifest["learning_config"]),
            output_root=root, quiet=True)
    finally:
        if previous_binary is None:
            os.environ.pop("SUMO_BINARY", None)
        else:
            os.environ["SUMO_BINARY"] = previous_binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-episodes", type=int, default=45,
                        help="Cumulative target; remaining episodes are calculated from the saved checkpoint.")
    parser.add_argument("--eval-every", type=int, default=45)
    args = parser.parse_args()
    try:
        resume_to_target(args.output_root, args.target_episodes, args.eval_every)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
