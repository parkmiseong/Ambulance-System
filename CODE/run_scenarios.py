"""Compare policies on identical manifests, origin streams, seeds, and demand levels."""
import argparse
from collections import defaultdict
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from experiment_config import SCENARIOS, isolated_random_state, parse_numbers
from experiment_log import append_record, new_run_id
from simulation import run_episode
from simulation_setup import latest_model, load_experiment

STRATEGIES = [
    ("shortest_distance", "Shortest road distance"),
    ("shortest_time", "Shortest estimated time"),
    ("bed", "Available beds"), ("rule", "Suitability"),
    ("heuristic", "Heuristic"), ("dqn", "DQN"),
]


def run_single(env, base_hospitals, scenario, seed, mode, agent=None, gui=False,
               ambulance_count=10, queue_policy="severity", aging_seconds=300,
               config=None, output_dir=None, on_outcome=None, quiet=False):
    from experiment_config import SimulationConfig
    config = config or SimulationConfig()
    epsilon = agent.epsilon if agent is not None else None
    training = agent.model.training if agent is not None else None
    try:
        with isolated_random_state():
            if agent is not None:
                agent.epsilon = 0.
                agent.model.eval()
            return run_episode(
                env, base_hospitals, scenario, seed, mode, config, agent=agent, gui=gui,
                ambulance_count=ambulance_count, queue_policy=queue_policy,
                aging_seconds=aging_seconds, output_dir=output_dir, on_outcome=on_outcome,
                quiet=quiet)
    finally:
        if agent is not None:
            agent.epsilon = epsilon
            agent.model.train(training)


def aggregate_results(results):
    patients = sum(r["generated_patients"] for r in results)
    rewards = [r["avg_reward"] for r in results]
    return {
        "runs": len(results), "generated_patients": patients,
        "arrived": sum(r["arrived"] for r in results),
        "failed": sum(r["failed"] for r in results),
        "censored": sum(r["censored"] for r in results),
        "reward_per_patient": sum(r["reward"] for r in results) / patients if patients else None,
        "arrival_rate": 100 * sum(r["arrived"] for r in results) / patients if patients else 0.,
        "golden_rate": 100 * sum(r["golden_success"] for r in results) / patients if patients else 0.,
        "reward_seed_std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else None,
        "avg_wait": sum(r["avg_wait"] * r["generated_patients"] for r in results) / patients if patients else 0.,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--scenario", type=int, choices=range(10), default=1, help="0 = all")
    parser.add_argument("--seeds", default="30001,30002,30003")
    parser.add_argument("--demand-levels", help="e.g. 0.1,0.3,1; default = training demand")
    parser.add_argument("--strategies", default=",".join(mode for mode, _ in STRATEGIES))
    parser.add_argument("--ambulances", type=int, help="default = training configuration")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--quiet", action="store_true",
                        help="Hide SUMO output; keep one summary per strategy.")
    args = parser.parse_args()
    model = args.model or latest_model()
    env, hospitals, agent, config, manifest = load_experiment(model)
    ambulance_count = args.ambulances or manifest["ambulance_count"]
    modes = args.strategies.split(",")
    if any(mode not in dict(STRATEGIES) for mode in modes):
        parser.error("Unknown strategy.")
    seeds = parse_numbers(args.seeds, int)
    levels = parse_numbers(args.demand_levels) if args.demand_levels else [config.demand_scale]
    scenarios = SCENARIOS if args.scenario == 0 else [SCENARIOS[args.scenario - 1]]
    root = Path(manifest_path_root(agent))
    run_id = new_run_id("eval")
    record = partial(append_record, log_path=root / "experiment_history.csv")
    record(run_id, "evaluation", "run_start", status="started", model_path=str(model),
           metrics={"seeds": seeds, "demand_levels": levels, "modes": modes,
                    "experiment_signature": manifest["signature"]})
    grouped = defaultdict(list)
    for scenario in scenarios:
        for demand in levels:
            evaluation_config = replace(config, demand_scale=demand)
            for seed in seeds:
                for mode in modes:
                    def outcome(row):
                        record(run_id, "evaluation", "mission", scenario_id=scenario["id"],
                               strategy=mode, status="censored" if row["censored"] else (
                                   "arrived" if row["arrived"] else "failed"),
                               model_path=str(model), metrics={**row, "seed": seed, "demand_scale": demand})
                    result = run_single(
                        env, hospitals, scenario, seed, mode, agent if mode == "dqn" else None,
                        gui=args.gui, ambulance_count=ambulance_count,
                        queue_policy=manifest["queue_policy"], aging_seconds=manifest["aging_seconds"],
                        config=evaluation_config,
                        output_dir=root / "sumo" / run_id / f"s{scenario['id']}_d{demand}_seed{seed}_{mode}",
                        on_outcome=outcome, quiet=args.quiet)
                    grouped[(scenario["id"], demand, mode)].append(result)
                    record(run_id, "evaluation", "strategy_result", scenario_id=scenario["id"],
                           strategy=mode, status="completed", model_path=str(model), metrics=result)
    for (scenario_id, demand, mode), results in grouped.items():
        summary = aggregate_results(results)
        print(f"scenario={scenario_id} demand={demand:g} {mode}: {summary}")
        record(run_id, "evaluation", "strategy_average", scenario_id=scenario_id,
               strategy=mode, status="completed", model_path=str(model),
               metrics={**summary, "demand_scale": demand, "seeds": seeds})
    record(run_id, "evaluation", "run_end", status="completed", model_path=str(model),
           metrics={"runs": sum(len(r) for r in grouped.values())})


def manifest_path_root(agent):
    # Manifests live at <output_root>/training/experiment_v3_*.json.
    return Path(agent.metadata["manifest_path"]).resolve().parent.parent


if __name__ == "__main__":
    main()
