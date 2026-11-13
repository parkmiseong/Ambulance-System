"""Compare queue policies using best models and held-out seed episodes."""
import argparse
from dataclasses import replace
from functools import partial
from pathlib import Path

from evaluation_metrics import TEST_SEEDS, validate_test_seeds
from experiment_config import SCENARIOS, parse_numbers
from experiment_log import append_record, new_run_id
from run_scenarios import run_single, aggregate_results, manifest_path_root
from simulation_setup import resolve_evaluation_model, load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        help="Best model; a last path selects the paired best.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--scenario", type=int, choices=range(1, 10))
    parser.add_argument("--ambulances", type=int)
    parser.add_argument("--aging-seconds", type=float, default=300.)
    parser.add_argument("--seeds", default=",".join(map(str, TEST_SEEDS)))
    parser.add_argument("--demand-levels")
    parser.add_argument("--diagnostic-seeds", action="store_true")
    parser.add_argument("--quiet", action="store_true")
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
    seeds = parse_numbers(args.seeds, int)
    try:
        validate_test_seeds(seeds, manifest, agent.loaded_episode, diagnostic=args.diagnostic_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    selected = manifest.get("training_scenarios", [1])
    scenario_id = args.scenario if args.scenario is not None else (selected[0] if len(selected) == 1 else 1)
    demands = (parse_numbers(args.demand_levels) if args.demand_levels
               else manifest.get("validation_demands", [config.demand_scale]))
    root = Path(manifest_path_root(agent))
    run_id = new_run_id("queue_compare")
    record = partial(append_record, log_path=root / "experiment_history.csv")
    record(run_id, "queue_comparison", "run_start", status="started", model_path=str(model),
           metrics={"model_kind": "best", "model_episode": agent.loaded_episode,
                    "evaluation_split": "diagnostic" if args.diagnostic_seeds else "test",
                    "seeds": seeds, "demand_levels": demands,
                    "requested_model": str(args.model) if args.model is not None else None,
                    "experiment_signature": manifest["signature"]})
    for demand in demands:
        for policy in ("fifo", "severity", "hybrid"):
            results = []
            for seed in seeds:
                result = run_single(
                    env, hospitals, SCENARIOS[scenario_id - 1], seed, "dqn", agent,
                    ambulance_count=args.ambulances or manifest["ambulance_count"],
                    queue_policy=policy, aging_seconds=args.aging_seconds,
                    config=replace(config, demand_scale=demand),
                    output_dir=root / "sumo" / run_id / f"d{demand}_{policy}_{seed}", quiet=args.quiet)
                results.append(result)
                record(run_id, "queue_comparison", "queue_policy_result", strategy=policy,
                       scenario_id=scenario_id, status="completed", model_path=str(model),
                       metrics={**result, "aging_seconds": args.aging_seconds})
            summary = aggregate_results(results)
            print(f"demand={demand:g} {policy}: {summary}")
            record(run_id, "queue_comparison", "queue_policy_average", strategy=policy,
                   scenario_id=scenario_id, status="completed", model_path=str(model),
                   metrics=dict(demand_scale=demand, **summary))
    record(run_id, "queue_comparison", "run_end", status="completed", model_path=str(model))


if __name__ == "__main__":
    main()
