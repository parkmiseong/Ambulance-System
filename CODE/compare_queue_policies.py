"""Compare queue policies using the exact v3 training experiment."""
import argparse
from functools import partial
from pathlib import Path

from experiment_config import SCENARIOS, parse_numbers
from experiment_log import append_record, new_run_id
from run_scenarios import run_single, aggregate_results, manifest_path_root
from simulation_setup import latest_model, load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--scenario", type=int, choices=range(1, 10), default=1)
    parser.add_argument("--ambulances", type=int)
    parser.add_argument("--aging-seconds", type=float, default=300.)
    parser.add_argument("--seeds", default="30001,30002,30003")
    parser.add_argument("--quiet", action="store_true",
                        help="Hide SUMO output; keep one summary per queue policy.")
    args = parser.parse_args()
    model = args.model or latest_model()
    env, hospitals, agent, config, manifest = load_experiment(model)
    root = manifest_path_root(agent)
    run_id = new_run_id("queue_compare")
    record = partial(append_record, log_path=root / "experiment_history.csv")
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
