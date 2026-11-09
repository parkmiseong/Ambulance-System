"""Train reproducible sequential DQN episodes with a shared evaluation engine."""
import argparse
import copy
import contextlib
import os
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import time

import numpy as np

from baseline import state_dimension
from dqn_agent import DQNAmbulanceAgent
from experiment_config import (SimulationConfig, SCENARIOS, isolated_random_state,
                               parse_numbers, seed_everything, curriculum_multiplier)
from experiment_log import append_record, new_run_id
from simulation import run_episode
from simulation_setup import prepare_environment, make_manifest, manifest_signature
from training_graphs import append_episode_metrics, plot_training_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def suppress_stdout(enabled):
    """Suppress only optional progress chatter; errors still propagate."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            yield


def validate_panel(env, hospitals, agent, config, scenario_ids, seeds,
                   ambulance_count, queue_policy, aging_seconds, output_dir, quiet=False):
    results = []
    epsilon, training = agent.epsilon, agent.model.training
    try:
        with isolated_random_state():
            agent.epsilon = 0.
            agent.model.eval()
            for scenario_id in scenario_ids:
                scenario = SCENARIOS[scenario_id - 1]
                for seed in seeds:
                    panel_started = time.monotonic()
                    print(f"[Validation {len(results) + 1}/{len(scenario_ids) * len(seeds)}] "
                          f"scenario={scenario_id} seed={seed} demand x{config.demand_scale:g} started",
                          flush=True)
                    results.append(run_episode(
                        env, hospitals, scenario, seed, "dqn", config, agent=agent,
                        ambulance_count=ambulance_count, queue_policy=queue_policy,
                        aging_seconds=aging_seconds,
                        output_dir=Path(output_dir) / f"scenario{scenario_id}_seed{seed}",
                        quiet=quiet))
                    print(f"[Validation] seed={seed} completed in "
                          f"{time.monotonic() - panel_started:.1f}s", flush=True)
    finally:
        agent.epsilon = epsilon
        agent.model.train(training)
    generated = sum(r["generated_patients"] for r in results)
    return {
        "generated_patients": generated,
        "avg_reward": sum(r["reward"] for r in results) / generated if generated else None,
        "arrival_rate": 100. * sum(r["arrived"] for r in results) / generated if generated else 0.,
        "golden_rate": 100. * sum(r["golden_success"] for r in results) / generated if generated else 0.,
        "results": results,
    }


def train(num_episodes=100, gui=False, ambulance_count=10, eval_every=5, resume=False,
          queue_policy="severity", aging_seconds=300, virtual_count=0,
          virtual_strategy="far", placement_seed=42, config=None, curriculum=(),
          validation_seeds=(30001, 30002, 30003), validation_scenarios=(1,),
          output_root=None, quiet=False):
    config = config or SimulationConfig()
    if num_episodes < 1 or eval_every < 0 or not 1 <= ambulance_count <= 10:
        raise ValueError("Require episodes >= 1, eval_every >= 0, ambulances in 1..10.")
    if not validation_seeds or not validation_scenarios:
        raise ValueError("A validation panel must not be empty.")
    if any(i < 1 or i > len(SCENARIOS) for i in validation_scenarios):
        raise ValueError("Validation scenario IDs must be in 1..9.")
    for multiplier in curriculum:
        replace(config, demand_scale=config.demand_scale * multiplier)
    seed_everything(config.seed)
    root = Path(output_root or PROJECT_ROOT / "results").resolve()
    run_id = new_run_id("train")
    output = root / "sumo" / run_id
    record = partial(append_record, log_path=root / "experiment_history.csv")
    with suppress_stdout(quiet):
        env, hospitals, edges, access = prepare_environment(
            config, virtual_count, virtual_strategy, placement_seed, output / "setup", quiet=quiet)
    manifest = make_manifest(env, hospitals, edges, access, config, ambulance_count,
                             queue_policy, aging_seconds, curriculum)
    manifest.update(validation_seeds=list(validation_seeds),
                    validation_scenarios=list(validation_scenarios))
    manifest["signature"] = manifest_signature(manifest)
    tag = f"v3_{manifest['signature']}"
    model_path = root / "models" / f"dqn_{tag}_last.pth"
    best_path = root / "models" / f"dqn_{tag}_best.pth"
    checkpoint_path = root / "training" / f"checkpoint_{tag}.pth"
    manifest_path = root / "training" / f"experiment_{tag}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        if not manifest_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"No compatible v3 checkpoint: {checkpoint_path}")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume settings differ from the saved experiment.")
    else:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    agent = DQNAmbulanceAgent(state_dimension(len(hospitals)), len(hospitals))
    agent.metadata = {
        "manifest_path": str(manifest_path),
        "experiment_signature": manifest["signature"],
        "selection_metric": "validation_reward_per_generated_patient",
        "best_validation_reward": None,
    }
    start_episode = 1
    if resume:
        agent.load_model(str(checkpoint_path), resume=True)
        start_episode = int(agent.loaded_episode) + 1
    previous_episodes = start_episode - 1
    target_episode = previous_episodes + num_episodes
    print(f"[Training] previous={previous_episodes} episodes | "
          f"additional={num_episodes} | target_total={target_episode}", flush=True)
    best_score = agent.metadata.get("best_validation_reward")
    metrics_dir = root / "training" / tag / run_id
    csv_path = metrics_dir / "training_metrics.csv"
    started = time.monotonic()
    record(run_id, "training", "run_start", status="started",
           ambulance_limit=ambulance_count, model_path=str(model_path),
           metrics={"config": config.to_dict(), "manifest_path": str(manifest_path),
                    "requested_episodes": num_episodes, "resume": resume,
                    "previous_episodes": previous_episodes,
                    "target_total_episodes": target_episode,
                    "queue_policy": queue_policy, "curriculum": list(curriculum),
                    "eval_every": eval_every, "validation_seeds": list(validation_seeds),
                    "validation_scenarios": list(validation_scenarios),
                    "quiet": quiet})
    if not quiet:
        print(f"[Experiment] hospitals={len(hospitals)}, state_dim={agent.state_dim}")
        print(f"[Manifest] {manifest_path}")
        print(f"[Last model] {model_path}")
    try:
        for episode in range(start_episode, start_episode + num_episodes):
            scenario = SCENARIOS[(episode - 1) % len(SCENARIOS)]
            multiplier = curriculum_multiplier(episode, curriculum)
            episode_config = replace(config, demand_scale=config.demand_scale * multiplier)
            episode_started = time.monotonic()
            def on_outcome(row):
                record(run_id, "training", "mission", episode=episode,
                       scenario_id=scenario["id"], scenario_name=scenario["name"],
                       status="censored" if row["censored"] else ("arrived" if row["arrived"] else "failed"),
                       ambulance_limit=ambulance_count, model_path=str(model_path), metrics=row)
            result = run_episode(
                env, hospitals, scenario, config.seed + 10000 + episode, "dqn",
                episode_config, agent=agent, training=True, gui=gui,
                ambulance_count=ambulance_count, queue_policy=queue_policy,
                aging_seconds=aging_seconds, output_dir=output / f"episode{episode}",
                on_outcome=on_outcome, quiet=quiet)
            if episode % 10 == 0:
                agent.update_target_network()
            result.update(epsilon=agent.epsilon, replay_memory=len(agent.memory))
            record(run_id, "training", "episode", episode=episode,
                   scenario_id=scenario["id"], status="completed",
                   ambulance_limit=ambulance_count, model_path=str(model_path),
                   duration_seconds=time.monotonic() - episode_started, metrics=result)
            append_episode_metrics(csv_path, {
                "episode": episode, "scenario": scenario["id"],
                "previous_episodes": previous_episodes,
                "run_episode": episode - previous_episodes,
                "patients": result["generated_patients"], "missions": result["missions"],
                "arrived": result["arrived"], "failed": result["failed"],
                "reward": result["reward"], "avg_reward": result["avg_reward"],
                "avg_loss": result["avg_loss"], "epsilon": agent.epsilon,
                "episode_seconds": time.monotonic() - episode_started})
            print(f"Episode {episode}/{target_episode} "
                  f"(previous={previous_episodes}, "
                  f"this_run={episode - previous_episodes}/{num_episodes}) | "
                  f"demand x{episode_config.demand_scale:g} | "
                  f"generated={result['generated_patients']} arrived={result['arrived']} "
                  f"failed={result['failed']} censored={result['censored']} | "
                  f"golden={result['golden_rate']:.1f}% wait={result['avg_wait']:.1f}s | "
                  f"reward/patient={result['avg_reward']:.3f} loss={result['avg_loss']:.4f} "
                  f"epsilon={agent.epsilon:.4f}", flush=True)
            if eval_every and episode % eval_every == 0:
                # Keep completed training even if the following evaluation fails.
                with suppress_stdout(quiet):
                    agent.save_model(str(model_path), episode=episode)
                    agent.save_model(str(checkpoint_path), episode=episode, full_state=True)
                validation = validate_panel(
                    env, hospitals, agent, config, validation_scenarios, validation_seeds,
                    ambulance_count, queue_policy, aging_seconds, output / f"validation{episode}", quiet=quiet)
                record(run_id, "training", "fixed_evaluation", episode=episode,
                       status="completed", model_path=str(model_path), metrics=validation)
                score = validation["avg_reward"]
                if score is not None and (best_score is None or score > best_score):
                    best_score = score
                    agent.metadata["best_validation_reward"] = score
                    with suppress_stdout(quiet):
                        agent.save_model(str(best_path), episode=episode)
                print(f"[Validation] reward/patient={score}, arrival={validation['arrival_rate']:.1f}%")
            with suppress_stdout(quiet):
                agent.save_model(str(model_path), episode=episode)
                agent.save_model(str(checkpoint_path), episode=episode, full_state=True)
    except Exception as exc:
        record(run_id, "training", "run_end", status="failed", model_path=str(model_path),
               duration_seconds=time.monotonic() - started,
               metrics={"error": f"{type(exc).__name__}: {exc}"})
        raise
    plot_training_metrics(csv_path, metrics_dir / "training_dashboard.png")
    record(run_id, "training", "run_end", status="completed", model_path=str(model_path),
           duration_seconds=time.monotonic() - started,
           metrics={"completed_episodes": num_episodes, "best_validation_reward": best_score,
                    "previous_episodes": previous_episodes,
                    "total_completed_episodes": target_episode,
                    "best_model": str(best_path) if best_score is not None else None,
                    "last_model": str(model_path)})
    print(f"[Training completed] previous={previous_episodes} + "
          f"this_run={num_episodes} = total={target_episode} episodes", flush=True)
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Sequential ambulance DQN training")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Additional episodes in this run; with --resume, add to the saved total.")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--ambulances", type=int, default=10)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--quiet", action="store_true",
                        help="Hide SUMO/setup output; keep episode and validation summaries.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--virtual-count", type=int, choices=[0, 2, 5], default=0)
    parser.add_argument("--virtual-strategy", choices=["far", "random", "demand-proxy"], default="far")
    parser.add_argument("--placement-seed", type=int, default=42)
    parser.add_argument("--queue-policy", choices=["severity", "fifo", "hybrid"], default="severity")
    parser.add_argument("--aging-seconds", type=float, default=300.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--demand-scale", type=float, default=1.)
    parser.add_argument("--curriculum", default="", help="Demand multipliers, e.g. 0.1,0.3,1; each stage lasts 9 episodes")
    parser.add_argument("--treatment-seconds", type=float, default=900.)
    parser.add_argument("--bed-fraction", type=float, default=.08)
    parser.add_argument("--max-wait", type=float, default=1800.)
    parser.add_argument("--mission-timeout", type=float, default=1800.)
    parser.add_argument("--candidate-limit", type=int, default=1200)
    parser.add_argument("--validation-seeds", default="30001,30002,30003")
    parser.add_argument("--validation-scenarios", default="1")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    try:
        config = SimulationConfig(
            arrival_steps=args.steps, demand_scale=args.demand_scale,
            max_wait=args.max_wait, mission_timeout=args.mission_timeout,
            treatment_seconds=args.treatment_seconds, bed_fraction=args.bed_fraction,
            seed=args.seed, candidate_limit=args.candidate_limit)
        curriculum = parse_numbers(args.curriculum) if args.curriculum else ()
        train(args.episodes, args.gui, args.ambulances, args.eval_every, args.resume,
              args.queue_policy, args.aging_seconds, args.virtual_count,
              args.virtual_strategy, args.placement_seed, config, curriculum,
              parse_numbers(args.validation_seeds, int),
              parse_numbers(args.validation_scenarios, int), args.output_root, args.quiet)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
