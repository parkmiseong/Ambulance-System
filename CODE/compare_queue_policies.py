"""Compare queue policies using best models and held-out seed episodes."""
import argparse
import json
from dataclasses import replace
from functools import partial
from pathlib import Path

from evaluation_metrics import TEST_SEEDS, validate_test_seeds
from experiment_config import parse_numbers
from experiment_log import append_record, new_run_id
from emergency_yield import yield_settings
from evaluation_audit import build_evaluation_environment
from evaluation_profile import (freeze_evaluation_model, resolve_evaluation_profile,
                                validate_evaluation_demands)
from run_scenarios import run_single, aggregate_results
from simulation_setup import resolve_evaluation_model, load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        help="Best model; a last path selects the paired best.")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--scenario", type=int, choices=range(10), help="0 = all; default = saved scenario.")
    parser.add_argument("--scenario-profile", choices=("saved", "standard"), default="saved")
    parser.add_argument("--ambulances", type=int, choices=range(1, 11))
    parser.add_argument("--aging-seconds", type=float, help="default = saved training configuration")
    parser.add_argument("--seeds", default=",".join(map(str, TEST_SEEDS)))
    parser.add_argument("--demand-levels")
    parser.add_argument("--diagnostic-seeds", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--emergency-yield", action=argparse.BooleanOptionalAction, default=None,
                        help="Override emergency driving ON/OFF; default = saved training setting.")
    parser.add_argument("--repeat-background-traffic", action=argparse.BooleanOptionalAction, default=None,
                        help="Override hourly passenger traffic repetition; default = saved training setting.")
    args = parser.parse_args()
    try:
        source_model = resolve_evaluation_model(args.model, args.output_root)
        run_id = new_run_id("queue_compare")
        model, root, snapshot = freeze_evaluation_model(source_model, run_id)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if args.model is not None and args.model.name.endswith("_last.pth"):
        print(f"[Evaluation] requested={args.model}; using paired best={source_model}", flush=True)
    env, hospitals, agent, config, manifest = load_experiment(model)
    trained_emergency_yield = config.emergency_yield
    if args.emergency_yield is not None:
        config = replace(config, emergency_yield=args.emergency_yield)
    if args.repeat_background_traffic is not None:
        config = replace(config, repeat_background_traffic=args.repeat_background_traffic)
    print(f"[Evaluation] model={model} episode={agent.loaded_episode} selection=best "
          f"reward_version={config.reward_version} "
          f"emergency_yield={'ON' if config.emergency_yield else 'OFF'} "
          f"trained_emergency_yield={'ON' if trained_emergency_yield else 'OFF'}", flush=True)
    seeds = parse_numbers(args.seeds, int)
    try:
        validate_test_seeds(seeds, manifest, agent.loaded_episode, diagnostic=args.diagnostic_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        demands = (parse_numbers(args.demand_levels) if args.demand_levels
                   else manifest.get("validation_demands", [config.demand_scale]))
        scenarios, occupancy, profile_source = resolve_evaluation_profile(
            manifest, config, args.scenario, args.scenario_profile)
        validate_evaluation_demands(scenarios, config, demands)
        aging_seconds = args.aging_seconds if args.aging_seconds is not None else manifest["aging_seconds"]
        if not 0 < aging_seconds < float("inf"):
            raise ValueError("aging-seconds must be finite and positive.")
    except ValueError as exc:
        parser.error(str(exc))
    ambulance_count = args.ambulances or manifest["ambulance_count"]
    environments = {policy: build_evaluation_environment(env, hospitals, config, scenarios, demands,
                    ambulance_count, policy, aging_seconds, occupancy)
                    for policy in ("fifo", "severity", "hybrid")}
    print(f"[Evaluation profile] source={profile_source} occupancy={occupancy} "
          f"repeat_background_traffic={config.repeat_background_traffic} "
          f"scenarios={[(s['id'], s['prob'], s['scale']) for s in scenarios]}", flush=True)
    record = partial(append_record, log_path=root / "experiment_history.csv")
    run_metadata = {"model_kind": "best", "model_episode": agent.loaded_episode,
                    "evaluation_split": "diagnostic" if args.diagnostic_seeds else "test",
                    "seeds": seeds, "demand_levels": demands,
                    "requested_model": str(args.model) if args.model is not None else None,
                    "experiment_signature": manifest["signature"],
                    "emergency_yield": yield_settings(config.emergency_yield),
                    "trained_emergency_yield": yield_settings(trained_emergency_yield),
                    "profile_source": profile_source, "evaluation_environments": environments,
                    "model_snapshot": snapshot}
    (Path(snapshot["manifest_path"]).parent / "evaluation.json").write_text(
        json.dumps(dict(run_id=run_id, **run_metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    record(run_id, "queue_comparison", "run_start", status="started", model_path=str(model),
           ambulance_limit=ambulance_count, metrics=run_metadata)
    for scenario in scenarios:
        for demand in demands:
            for policy in ("fifo", "severity", "hybrid"):
                results = []
                for seed in seeds:
                    result = run_single(
                        env, hospitals, scenario, seed, "dqn", agent,
                        ambulance_count=ambulance_count,
                        queue_policy=policy, aging_seconds=aging_seconds,
                        config=replace(config, demand_scale=demand), initial_occupancy_range=occupancy,
                        output_dir=root / "sumo" / run_id / f"s{scenario['id']}_d{demand}_{policy}_{seed}",
                        quiet=args.quiet)
                    results.append(result)
                    record(run_id, "queue_comparison", "queue_policy_result", strategy=policy,
                           scenario_id=scenario["id"], status="completed", model_path=str(model),
                           metrics={**result, "aging_seconds": aging_seconds,
                                    "evaluation_environment": environments[policy]})
                summary = aggregate_results(results)
                print(f"scenario={scenario['id']} demand={demand:g} {policy}: {summary}")
                record(run_id, "queue_comparison", "queue_policy_average", strategy=policy,
                       scenario_id=scenario["id"], status="completed", model_path=str(model),
                       metrics=dict(demand_scale=demand, evaluation_environment=environments[policy], **summary))
    record(run_id, "queue_comparison", "run_end", status="completed", model_path=str(model))


if __name__ == "__main__":
    main()
