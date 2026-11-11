"""Compare queue policies using the model's recorded training experiment."""
import argparse
from functools import partial
from pathlib import Path

from experiment_config import SCENARIOS, parse_numbers
from experiment_log import append_record, new_run_id
from run_scenarios import run_single, aggregate_results, manifest_path_root
from simulation_setup import resolve_evaluation_model, load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        help="Best model path; a *_last.pth path selects its paired *_best.pth.")
    parser.add_argument("--output-root", type=Path,
                        help="Training output root containing models/; used when --model is omitted.")
    parser.add_argument("--scenario", type=int, choices=range(1, 10), default=1)
    parser.add_argument("--ambulances", type=int)
    parser.add_argument("--aging-seconds", type=float, default=300.)
    parser.add_argument("--seeds", default="30001,30002,30003")
    parser.add_argument("--quiet", action="store_true",
                        help="Hide SUMO output; keep one summary per queue policy.")
    args = parser.parse_args()
    try:
        model = resolve_evaluation_model(args.model, args.output_root)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if args.model is not None and args.model.name.endswith("_last.pth"):
        print(f"[Evaluation] requested={args.model}; using paired best={model}", flush=True)
    env, hospitals, agent, config, manifest = load_experiment(model)
    print(f"[Evaluation] model={model} episode={agent.loaded_episode} selection=best "
          f"reward_version={config.reward_version}", flush=True)
    root = manifest_path_root(agent)
    run_id = new_run_id("queue_compare")
    record = partial(append_record, log_path=root / "experiment_history.csv")
    record(run_id, "queue_comparison", "run_start", status="started", model_path=str(model),
           metrics={"model_kind": "best", "model_episode": agent.loaded_episode,
                    "requested_model": str(args.model) if args.model is not None else None,
                    "experiment_signature": manifest["signature"]})
    for policy in ("fifo", "severity", "hybrid"):
        results = []
        for seed in parse_numbers(args.seeds, int):
            result = run_single(
                env, hospitals, SCENARIOS[args.scenario - 1], seed, "dqn", agent,
                ambulance_count=args.ambulances or manifest["ambulance_count"],
                queue_policy=policy, aging_seconds=args.aging_seconds, config=config,
                output_dir=root / "sumo" / run_id / f"{policy}_{seed}", quiet=args.quiet)
            results.append(result)
            record(run_id, "queue_comparison", "queue_policy_result", strategy=policy,
                   scenario_id=args.scenario, status="completed", model_path=str(model),
                   metrics={**result, "aging_seconds": args.aging_seconds})
        print(policy, aggregate_results(results))


if __name__ == "__main__":
    main()
