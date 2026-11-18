"""Compare policies on identical manifests, origin streams, seeds, and demand levels."""
import argparse
import json
from collections import defaultdict
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from experiment_config import isolated_random_state, parse_numbers
from experiment_log import append_record, new_run_id
from emergency_yield import yield_settings
from evaluation_audit import build_evaluation_environment
from evaluation_profile import (experiment_root, freeze_evaluation_model, resolve_evaluation_profile,
                                validate_evaluation_demands)
from simulation import run_episode
from simulation_setup import resolve_evaluation_model, load_experiment
from evaluation_metrics import (TEST_SEEDS, episode_uncertainty, paired_patient_comparison,
                                validate_test_seeds)

STRATEGIES = [
    ("shortest_distance", "Shortest road distance"),
    ("shortest_time", "Shortest estimated time"),
    ("bed", "Available beds"), ("rule", "Suitability"),
    ("heuristic", "Heuristic"), ("greedy_reward", "Highest estimated patient reward"),
    ("dqn", "DQN"),
]


def run_single(env, base_hospitals, scenario, seed, mode, agent=None, gui=False,
               ambulance_count=10, queue_policy="severity", aging_seconds=300,
               config=None, output_dir=None, on_outcome=None, quiet=False,
               on_decision=None, initial_occupancy_range=None):
    from experiment_config import SimulationConfig
    config = config or SimulationConfig(emergency_yield=True)
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
                quiet=quiet, on_decision=on_decision,
                initial_occupancy_range=initial_occupancy_range)
    finally:
        if agent is not None:
            agent.epsilon = epsilon
            agent.model.train(training)


def aggregate_results(results, bootstrap_samples=2000):
    versions = {r.get("reward_version", 3) for r in results}
    if len(versions) != 1:
        raise ValueError("Cannot aggregate different reward versions.")
    admission_modes = {r.get("hospital_admission_mode", "precheck") for r in results}
    if len(admission_modes) != 1:
        raise ValueError("Cannot aggregate different hospital admission modes.")
    congestion_modes = {r.get("congestion_reward_mode", "penalty") for r in results}
    if len(congestion_modes) != 1:
        raise ValueError("Cannot aggregate different congestion reward modes.")
    rates = {tuple(r.get("demand_rates_per_hour", ())) for r in results}
    if len(rates) != 1:
        raise ValueError("Cannot aggregate different hourly demand profiles.")
    driving = {tuple(sorted(r.get("emergency_yield", yield_settings(False)).items())) for r in results}
    if len(driving) != 1:
        raise ValueError("Cannot aggregate different emergency driving settings.")
    patients = sum(r["generated_patients"] for r in results)
    rewards = [r["avg_reward"] for r in results]
    summary = {
        "reward_version": next(iter(versions)),
        "hospital_admission_mode": next(iter(admission_modes)),
        "congestion_reward_mode": next(iter(congestion_modes)),
        "demand_rates_per_hour": list(next(iter(rates))),
        "emergency_yield": dict(next(iter(driving))),
        "rejection_count": sum(r.get("rejection_count", 0) for r in results),
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
    summary["uncertainty"] = episode_uncertainty(results, bootstrap_samples)
    if all("reward_components" in r for r in results):
        summary["avg_reward_components"] = {
            k: sum(r["reward_components"][k] for r in results) / patients if patients else 0.
            for k in results[0]["reward_components"]}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        help="Best model path; a *_last.pth path selects its paired *_best.pth.")
    parser.add_argument("--output-root", type=Path,
                        help="Training output root containing models/; used when --model is omitted.")
    parser.add_argument("--scenario", type=int, choices=range(10),
                        help="0 = all; default = saved web scenario or the CLI training default.")
    parser.add_argument("--scenario-profile", choices=("saved", "standard"), default="saved",
                        help="saved restores web demand/occupancy; standard explicitly selects CLI scenario demand.")
    parser.add_argument("--seeds", default=",".join(map(str, TEST_SEEDS)))
    parser.add_argument("--diagnostic-seeds", action="store_true",
                        help="Allow seen seeds and label the run diagnostic, not a final test.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--reference-strategy", default="shortest_time")
    parser.add_argument("--demand-levels", help="e.g. 0.1,0.3,1; default = training demand")
    parser.add_argument("--strategies", default=",".join(mode for mode, _ in STRATEGIES))
    parser.add_argument("--ambulances", type=int, choices=range(1, 11), help="default = training configuration")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--emergency-yield", action=argparse.BooleanOptionalAction, default=None,
                        help="Override emergency driving ON/OFF; default = saved training setting.")
    parser.add_argument("--repeat-background-traffic", action=argparse.BooleanOptionalAction, default=None,
                        help="Override hourly passenger traffic repetition; default = saved training setting.")
    parser.add_argument("--trace-decisions", action="store_true",
                        help="Save candidate hospitals, Q values, masks and the chosen action.")
    parser.add_argument("--reward-version", type=int, choices=(3, 4),
                        help="Override scoring for a common v3/v4 model comparison; model parameters stay fixed.")
    parser.add_argument("--admission-mode", choices=("precheck", "request", "arrival"),
                        help="Override admission handling; default = saved training configuration.")
    parser.add_argument("--congestion-reward", choices=("penalty", "balanced"),
                        help="Override hospital occupancy scoring; default = saved training configuration.")
    parser.add_argument("--quiet", action="store_true",
                        help="Hide SUMO output; keep one summary per strategy.")
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive.")
    try:
        source_model = resolve_evaluation_model(args.model, args.output_root)
        run_id = new_run_id("eval")
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
    if args.reward_version is not None:
        config = replace(config, reward_version=args.reward_version)
    if args.admission_mode is not None:
        config = replace(config, admission_mode=args.admission_mode)
    if args.congestion_reward is not None:
        config = replace(config, congestion_reward_mode=args.congestion_reward)
    print(f"[Evaluation] model={model} episode={agent.loaded_episode} selection=best "
          f"reward_version={config.reward_version} "
          f"emergency_yield={'ON' if config.emergency_yield else 'OFF'} "
          f"trained_emergency_yield={'ON' if trained_emergency_yield else 'OFF'}", flush=True)
    ambulance_count = args.ambulances or manifest["ambulance_count"]
    modes = args.strategies.split(",")
    if any(mode not in dict(STRATEGIES) for mode in modes):
        parser.error("Unknown strategy.")
    seeds = parse_numbers(args.seeds, int)
    try:
        validate_test_seeds(seeds, manifest, agent.loaded_episode, diagnostic=args.diagnostic_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        levels = (parse_numbers(args.demand_levels) if args.demand_levels
                  else manifest.get("validation_demands", [config.demand_scale]))
        scenarios, occupancy, profile_source = resolve_evaluation_profile(
            manifest, config, args.scenario, args.scenario_profile)
        validate_evaluation_demands(scenarios, config, levels)
    except ValueError as exc:
        parser.error(str(exc))
    environment = build_evaluation_environment(env, hospitals, config, scenarios, levels, ambulance_count,
                                              manifest["queue_policy"], manifest["aging_seconds"], occupancy)
    print(f"[Evaluation profile] source={profile_source} occupancy={occupancy} "
          f"repeat_background_traffic={config.repeat_background_traffic} "
          f"scenarios={[(s['id'], s['prob'], s['scale']) for s in scenarios]}", flush=True)
    record = partial(append_record, log_path=root / "experiment_history.csv")
    run_metadata = {"seeds": seeds, "demand_levels": levels, "modes": modes,
                    "experiment_signature": manifest["signature"],
                    "model_kind": "best", "model_episode": agent.loaded_episode,
                    "evaluation_split": "diagnostic" if args.diagnostic_seeds else "test",
                    "learning_stats": agent.training_statistics() if hasattr(agent, "training_statistics") else {},
                    "requested_model": str(args.model) if args.model is not None else None,
                    "reward_version": config.reward_version,
                    "hospital_admission_mode": config.admission_mode,
                    "congestion_reward_mode": config.congestion_reward_mode,
                    "demand_rates_per_hour": list(config.demand_rates_per_hour),
                    "emergency_yield": yield_settings(config.emergency_yield),
                    "trained_emergency_yield": yield_settings(trained_emergency_yield),
                    "trace_decisions": args.trace_decisions,
                    "profile_source": profile_source,
                    "evaluation_environment": environment, "model_snapshot": snapshot}
    (Path(snapshot["manifest_path"]).parent / "evaluation.json").write_text(
        json.dumps(dict(run_id=run_id, **run_metadata), ensure_ascii=False, indent=2), encoding="utf-8")
    record(run_id, "evaluation", "run_start", status="started", model_path=str(model),
           ambulance_limit=ambulance_count, metrics=run_metadata)
    grouped = defaultdict(list)
    patient_panels = defaultdict(list)
    for scenario in scenarios:
        for demand in levels:
            evaluation_config = replace(config, demand_scale=demand)
            for seed in seeds:
                for mode in modes:
                    print(f"[Evaluation] scenario={scenario['id']} demand={demand:g} "
                          f"seed={seed} strategy={mode} started", flush=True)
                    def outcome(row):
                        patient_panels[(scenario["id"], demand, mode)].append(
                            dict(row, scenario_id=scenario["id"], seed=seed, demand_scale=demand))
                        record(run_id, "evaluation", "mission", scenario_id=scenario["id"],
                               strategy=mode, status="censored" if row["censored"] else (
                                   "arrived" if row["arrived"] else "failed"),
                               model_path=str(model), metrics={**row, "seed": seed, "demand_scale": demand})
                    def decision(row):
                        record(run_id, "evaluation", "decision", scenario_id=scenario["id"],
                               strategy=mode, model_path=str(model),
                               metrics={**row, "seed": seed, "demand_scale": demand})
                    result = run_single(
                        env, hospitals, scenario, seed, mode, agent if mode == "dqn" else None,
                        gui=args.gui, ambulance_count=ambulance_count,
                        queue_policy=manifest["queue_policy"], aging_seconds=manifest["aging_seconds"],
                        config=evaluation_config,
                        output_dir=root / "sumo" / run_id / f"s{scenario['id']}_d{demand}_seed{seed}_{mode}",
                        on_outcome=outcome, quiet=args.quiet,
                        on_decision=decision if args.trace_decisions else None,
                        initial_occupancy_range=occupancy)
                    grouped[(scenario["id"], demand, mode)].append(result)
                    record(run_id, "evaluation", "strategy_result", scenario_id=scenario["id"],
                           strategy=mode, status="completed", model_path=str(model), metrics=result)
                    print(f"[Evaluation] {mode} completed | arrived={result['arrived']}/"
                          f"{result['generated_patients']} golden={result['golden_rate']:.1f}% "
                          f"reward/patient={result['avg_reward']:.3f}", flush=True)
    for (scenario_id, demand, mode), results in grouped.items():
        summary = aggregate_results(results, args.bootstrap_samples)
        print(f"scenario={scenario_id} demand={demand:g} {mode}: {summary}", flush=True)
        record(run_id, "evaluation", "strategy_average", scenario_id=scenario_id,
               strategy=mode, status="completed", model_path=str(model),
               metrics={**summary, "demand_scale": demand, "seeds": seeds})
    panels = {(s, d) for s, d, _ in patient_panels}
    for scenario_id, demand in sorted(panels):
        reference = patient_panels.get((scenario_id, demand, args.reference_strategy))
        if not reference:
            continue
        for mode in modes:
            if mode == args.reference_strategy:
                continue
            paired = paired_patient_comparison(reference, patient_panels[(scenario_id, demand, mode)],
                                               args.bootstrap_samples)
            record(run_id, "evaluation", "paired_comparison", scenario_id=scenario_id,
                   strategy=mode, status="completed", model_path=str(model),
                   metrics=dict(reference=args.reference_strategy, demand_scale=demand, **paired))
            interval = paired["ci95"]["reward_per_patient"] if paired["ci95"] else None
            print(f"[Paired] {mode} - {args.reference_strategy} | "
                  f"reward/patient={paired['delta']['reward_per_patient']:.3f} "
                  f"ci95={interval} seed_pairs={paired['seed_pairs']}", flush=True)
    record(run_id, "evaluation", "run_end", status="completed", model_path=str(model),
           metrics={"runs": sum(len(r) for r in grouped.values())})


def manifest_path_root(agent):
    # Manifests live at <output_root>/training/experiment_v<version>_*.json.
    return experiment_root(agent.metadata["manifest_path"])


if __name__ == "__main__":
    main()
