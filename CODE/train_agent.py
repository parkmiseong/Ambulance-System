"""Train reproducible sequential DQN episodes with a shared evaluation engine."""
import argparse
import contextlib
import os
from dataclasses import replace
from functools import partial
import json
from itertools import product
from pathlib import Path
import time

from baseline import state_dimension
from dqn_agent import DQNAmbulanceAgent
from experiment_config import (SimulationConfig, LearningConfig, SCENARIOS, isolated_random_state,
                               parse_numbers, seed_everything, curriculum_multiplier,
                               network_architecture)
from experiment_config import scenario_for_config
from dashboard_demand import PATIENTS_PER_HOUR, DEFAULT_DURATION, demand_metadata
from emergency_yield import yield_settings
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
                   ambulance_count, queue_policy, aging_seconds, output_dir, quiet=False,
                   demand_levels=None):
    results = []
    levels = tuple(demand_levels) if demand_levels is not None else (config.demand_scale,)
    if not levels:
        raise ValueError("Validation demand levels must not be empty.")
    epsilon, training = agent.epsilon, agent.model.training
    try:
        with isolated_random_state():
            agent.epsilon = 0.
            agent.model.eval()
            for scenario_id, seed, demand in product(scenario_ids, seeds, levels):
                scenario = scenario_for_config(SCENARIOS[scenario_id - 1], config)
                panel_started = time.monotonic()
                print(f"[Validation {len(results) + 1}/{len(scenario_ids) * len(seeds) * len(levels)}] "
                      f"scenario={scenario_id} seed={seed} demand x{demand:g} started", flush=True)
                results.append(run_episode(
                    env, hospitals, scenario, seed, "dqn", replace(config, demand_scale=demand), agent=agent,
                    ambulance_count=ambulance_count, queue_policy=queue_policy,
                    aging_seconds=aging_seconds,
                    output_dir=Path(output_dir) / f"scenario{scenario_id}_seed{seed}_d{demand:g}",
                    quiet=quiet))
                print(f"[Validation] seed={seed} demand x{demand:g} completed in "
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
          output_root=None, quiet=False, trace_decisions=False, validation_demands=None,
          cycle_curriculum=False, training_scenarios=None, learning_config=None, max_updates=None):
    config = config or SimulationConfig(emergency_yield=True, repeat_background_traffic=True)
    learning_config = learning_config or LearningConfig()
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive cumulative update limit.")
    all_training_scenarios = tuple(range(1, len(SCENARIOS) + 1))
    training_scenarios = (all_training_scenarios if training_scenarios is None
                          else tuple(training_scenarios))
    if not training_scenarios or any(type(i) is not int or not 1 <= i <= len(SCENARIOS)
                                     for i in training_scenarios):
        raise ValueError("Training scenario IDs must be integers in 1..9; provide at least one.")
    if num_episodes < 1 or eval_every < 0 or not 1 <= ambulance_count <= 10:
        raise ValueError("Require episodes >= 1, eval_every >= 0, ambulances in 1..10.")
    if not validation_seeds or not validation_scenarios:
        raise ValueError("A validation panel must not be empty.")
    if any(i < 1 or i > len(SCENARIOS) for i in validation_scenarios):
        raise ValueError("Validation scenario IDs must be in 1..9.")
    for multiplier in curriculum:
        replace(config, demand_scale=config.demand_scale * multiplier)
    validation_demands = tuple(validation_demands) if validation_demands is not None else (config.demand_scale,)
    if not validation_demands:
        raise ValueError("Validation demand levels must not be empty.")
    for demand in validation_demands:
        replace(config, demand_scale=demand)
    seed_everything(config.seed)
    root = Path(output_root or PROJECT_ROOT / "results").resolve()
    runtime_engine = None
    if resume:
        from prepare_resume_copy import validate_recorded_engine
        runtime_engine = validate_recorded_engine(root)
    run_id = new_run_id("train")
    output = root / "sumo" / run_id
    record = partial(append_record, log_path=root / "experiment_history.csv")
    with suppress_stdout(quiet):
        env, hospitals, edges, access = prepare_environment(
            config, virtual_count, virtual_strategy, placement_seed, output / "setup", quiet=quiet)
    manifest = make_manifest(env, hospitals, edges, access, config, ambulance_count,
                             queue_policy, aging_seconds, curriculum, learning_config=learning_config)
    manifest.update(learning_config=learning_config.to_dict(), validation_seeds=list(validation_seeds),
                    validation_scenarios=list(validation_scenarios),
                    validation_demands=list(validation_demands),
                    cycle_curriculum=cycle_curriculum)
    if config.demand_rates_per_hour:
        manifest["demand_profile"] = demand_metadata()
    # Preserve existing all-scenario signatures and resume compatibility.
    if training_scenarios != all_training_scenarios:
        manifest["training_scenarios"] = list(training_scenarios)
    manifest["signature"] = manifest_signature(manifest)
    tag = f"v{manifest['version']}_{manifest['signature']}"
    model_path = root / "models" / f"dqn_{tag}_last.pth"
    best_path = root / "models" / f"dqn_{tag}_best.pth"
    checkpoint_path = root / "training" / f"checkpoint_{tag}.pth"
    manifest_path = root / "training" / f"experiment_{tag}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        if not manifest_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"No compatible checkpoint: {checkpoint_path}. Resume requires identical settings, including "
                "emergency driving. Use --no-emergency-yield for legacy CLI checkpoints trained with yielding OFF; "
                "use --no-repeat-background-traffic for legacy CLI checkpoints without periodic traffic. "
                "Changed driving rules require a new experiment.")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume settings differ from the saved experiment.")
    else:
        if any(path.exists() for path in (model_path, best_path, checkpoint_path, manifest_path)):
            raise FileExistsError("Experiment already exists. Use --resume or a different --output-root/--seed.")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    agent = DQNAmbulanceAgent(
        state_dimension(len(hospitals), config.observation_version), len(hospitals),
        architecture=manifest.get("agent_architecture", network_architecture(config)),
        reward_scale=learning_config.reward_scale, learning_config=learning_config)
    agent.metadata = {
        "manifest_path": str(manifest_path),
        "experiment_signature": manifest["signature"],
        "selection_metric": "validation_reward_per_generated_patient",
        "best_validation_reward": None,
    }
    start_episode = 1
    if resume:
        agent.load_model(str(checkpoint_path), resume=True)
        agent.metadata["manifest_path"] = str(manifest_path)
        start_episode = int(agent.loaded_episode) + 1
    previous_episodes = start_episode - 1
    target_episode = previous_episodes + num_episodes
    last_completed = previous_episodes
    if max_updates is not None and getattr(agent, "training_updates", 0) >= max_updates:
        raise ValueError("The checkpoint already reached max_updates; choose a larger limit.")
    agent.update_limit = max_updates
    first_training_seed = config.seed + 10001
    if any(first_training_seed <= s <= config.seed + 10000 + target_episode for s in validation_seeds):
        raise ValueError("Validation seeds overlap the requested training seed range.")
    print(f"[Training] previous={previous_episodes} episodes | "
          f"additional={num_episodes} | target_total={target_episode}", flush=True)
    if runtime_engine is not None:
        print(f"[Engine] SUMO {runtime_engine.get('version')} | {runtime_engine['path']}", flush=True)
    if training_scenarios != all_training_scenarios:
        print(f"[Training scenarios] {','.join(map(str, training_scenarios))}", flush=True)
    print(f"[Observation] version={config.observation_version} dim={agent.state_dim} "
          f"pending_reward={config.pending_reward_feature}", flush=True)
    print(f"[Learning] gamma={learning_config.gamma:g} target={learning_config.target_sync_mode} "
          f"interval={learning_config.target_sync_interval} epsilon={learning_config.epsilon_schedule}", flush=True)
    print(f"[Driving] emergency_yield={'ON' if config.emergency_yield else 'OFF'} "
          f"reaction_distance_m={yield_settings(config.emergency_yield)['reaction_distance_m']}", flush=True)
    print(f"[Traffic] repeat_background={'ON' if config.repeat_background_traffic else 'OFF'}", flush=True)
    if config.demand_rates_per_hour:
        print(f"[Demand] hourly rates={config.demand_rates_per_hour} x{config.demand_scale:g} | "
              f"generation={config.arrival_steps}s | values are expectations, not fixed counts", flush=True)
    best_score = agent.metadata.get("best_validation_reward")
    metrics_dir = root / "training" / tag / run_id
    csv_path = metrics_dir / "training_metrics.csv"
    started = time.monotonic()
    record(run_id, "training", "run_start", status="started",
           ambulance_limit=ambulance_count, model_path=str(model_path),
           metrics={"config": config.to_dict(), "learning_config": learning_config.to_dict(),
                    "runtime_engine": runtime_engine,
                    "emergency_yield": yield_settings(config.emergency_yield),
                    "max_updates": max_updates, "manifest_path": str(manifest_path),
                    "requested_episodes": num_episodes, "resume": resume,
                    "previous_episodes": previous_episodes,
                    "target_total_episodes": target_episode,
                    "queue_policy": queue_policy, "curriculum": list(curriculum),
                    "training_scenarios": list(training_scenarios),
                    "eval_every": eval_every, "validation_seeds": list(validation_seeds),
                    "validation_scenarios": list(validation_scenarios),
                    "validation_demands": list(validation_demands),
                    "cycle_curriculum": cycle_curriculum,
                    "quiet": quiet, "trace_decisions": trace_decisions})
    if not quiet:
        print(f"[Experiment] hospitals={len(hospitals)}, state_dim={agent.state_dim}")
        print(f"[Manifest] {manifest_path}")
        print(f"[Last model] {model_path}")
    try:
        for episode in range(start_episode, start_episode + num_episodes):
            scenario_id = training_scenarios[(episode - 1) % len(training_scenarios)]
            scenario = scenario_for_config(SCENARIOS[scenario_id - 1], config)
            multiplier = curriculum_multiplier(episode, curriculum, cycle=cycle_curriculum)
            episode_config = replace(config, demand_scale=config.demand_scale * multiplier)
            episode_started = time.monotonic()
            def on_outcome(row):
                record(run_id, "training", "mission", episode=episode,
                       scenario_id=scenario["id"], scenario_name=scenario["name"],
                       status="censored" if row["censored"] else ("arrived" if row["arrived"] else "failed"),
                       ambulance_limit=ambulance_count, model_path=str(model_path), metrics=row)
            def on_decision(row):
                record(run_id, "training", "decision", episode=episode,
                       scenario_id=scenario["id"], strategy="dqn", model_path=str(model_path),
                       metrics={**row, "seed": config.seed + 10000 + episode,
                                "demand_scale": episode_config.demand_scale})
            result = run_episode(
                env, hospitals, scenario, config.seed + 10000 + episode, "dqn",
                episode_config, agent=agent, training=True, gui=gui,
                ambulance_count=ambulance_count, queue_policy=queue_policy,
                aging_seconds=aging_seconds, output_dir=output / f"episode{episode}",
                on_outcome=on_outcome, quiet=quiet,
                on_decision=on_decision if trace_decisions else None)
            last_completed = episode
            if (learning_config.target_sync_mode == "episodes"
                    and episode % learning_config.target_sync_episodes == 0):
                agent.update_target_network()
            stats = agent.training_statistics() if hasattr(agent, "training_statistics") else {}
            result.update({k: v for k, v in stats.items() if k != "exploration_rate"})
            result.update(epsilon=agent.epsilon, replay_memory=len(agent.memory),
                          total_exploration_rate=stats.get("exploration_rate", 0.))
            agent.metadata.update(trained_episodes=episode,
                                  training_seed_range=[first_training_seed, config.seed + 10000 + episode])
            at_limit = max_updates is not None and getattr(agent, "training_updates", 0) >= max_updates
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
                "episode_seconds": time.monotonic() - episode_started,
                **{k: result.get(k, 0) for k in ("training_updates", "episode_updates", "target_updates",
                    "total_decisions", "episode_decisions", "episode_exploratory_decisions", "exploration_rate")}})
            print(f"Episode {episode}/{target_episode} "
                  f"(previous={previous_episodes}, "
                  f"this_run={episode - previous_episodes}/{num_episodes}) | "
                  f"scenario={scenario['id']} p={scenario['prob']:g} traffic={scenario['scale']:g} | "
                  f"demand x{episode_config.demand_scale:g} | "
                  f"generated={result['generated_patients']} arrived={result['arrived']} "
                  f"failed={result['failed']} censored={result['censored']} | "
                  f"golden={result['golden_rate']:.1f}% wait={result['avg_wait']:.1f}s | "
                  f"reward/patient={result['avg_reward']:.3f} loss={result['avg_loss']:.4f} "
                  f"epsilon={agent.epsilon:.4f} updates={result.get('training_updates', 0)} "
                  f"explore={100 * result.get('exploration_rate', 0.):.1f}%", flush=True)
            if eval_every and (episode % eval_every == 0 or episode == target_episode or at_limit):
                # Keep completed training even if the following evaluation fails.
                with suppress_stdout(quiet):
                    agent.save_model(str(model_path), episode=episode)
                    agent.save_model(str(checkpoint_path), episode=episode, full_state=True)
                validation = validate_panel(
                    env, hospitals, agent, config, validation_scenarios, validation_seeds,
                    ambulance_count, queue_policy, aging_seconds, output / f"validation{episode}",
                    quiet=quiet, demand_levels=validation_demands)
                record(run_id, "training", "fixed_evaluation", episode=episode,
                       status="completed", model_path=str(model_path), metrics=validation)
                score = validation["avg_reward"]
                learned = getattr(agent, "training_updates", 0) > 0
                if score is not None and learned and (best_score is None or score > best_score):
                    best_score = score
                    agent.metadata["best_validation_reward"] = score
                    with suppress_stdout(quiet):
                        agent.save_model(str(best_path), episode=episode)
                print(f"[Validation] reward/patient={score}, arrival={validation['arrival_rate']:.1f}%")
                if not learned:
                    print("[Validation] No optimizer updates yet; best model was not promoted.", flush=True)
            with suppress_stdout(quiet):
                agent.save_model(str(model_path), episode=episode)
                agent.save_model(str(checkpoint_path), episode=episode, full_state=True)
            if at_limit:
                break
    except Exception as exc:
        record(run_id, "training", "run_end", status="failed", model_path=str(model_path),
               duration_seconds=time.monotonic() - started,
               metrics={"error": f"{type(exc).__name__}: {exc}"})
        raise
    plot_training_metrics(csv_path, metrics_dir / "training_dashboard.png")
    record(run_id, "training", "run_end", status="completed", model_path=str(model_path),
           duration_seconds=time.monotonic() - started,
           metrics={"completed_episodes": last_completed - previous_episodes,
                    "training_updates": getattr(agent, "training_updates", 0),
                    "best_validation_reward": best_score,
                    "previous_episodes": previous_episodes,
                    "total_completed_episodes": last_completed,
                    "best_model": str(best_path) if best_score is not None else None,
                    "last_model": str(model_path)})
    print(f"[Training completed] previous={previous_episodes} + "
          f"this_run={last_completed - previous_episodes} = total={last_completed} episodes", flush=True)
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Sequential ambulance DQN training")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Additional episodes in this run; with --resume, add to the saved total.")
    parser.add_argument("--train-scenarios",
                        help="Comma-separated training scenario IDs, e.g. 6 or 2,6; default cycles through 1..9. "
                             "Curriculum stages still last 9 episodes.")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--ambulances", type=int, default=10)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--emergency-yield", action=argparse.BooleanOptionalAction, default=True,
                        help="Match web driving: blue-light priority and sublane yielding ON by default. "
                             "Use --no-emergency-yield to reproduce legacy CLI runs.")
    parser.add_argument("--repeat-background-traffic", action=argparse.BooleanOptionalAction, default=True,
                        help="Repeat original hourly passenger traffic through the drain period, matching web. "
                             "Use --no-repeat-background-traffic to reproduce legacy CLI runs.")
    parser.add_argument("--quiet", action="store_true",
                        help="Hide SUMO/setup output; keep episode and validation summaries.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trace-decisions", action="store_true",
                        help="Save candidate hospitals, Q values and action masks to the experiment log.")
    parser.add_argument("--pending-reward-feature", choices=("estimate", "zero"), default="estimate",
                        help="V5 pending-mission reward estimate, or an equal-shape zero-feature control.")
    parser.add_argument("--golden-margin-feature", action="store_true")
    parser.add_argument("--candidate-reward-feature", action="store_true")
    parser.add_argument("--gamma", type=float, default=.95)
    parser.add_argument("--learning-rate", type=float, default=.0005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-capacity", type=int, default=30000)
    parser.add_argument("--reward-scale", type=float, default=100.)
    parser.add_argument("--target-sync-mode", choices=("updates", "episodes", "soft"), default="updates")
    parser.add_argument("--target-sync-interval", type=int, default=200)
    parser.add_argument("--target-sync-episodes", type=int, default=10)
    parser.add_argument("--target-tau", type=float, default=.005)
    parser.add_argument("--epsilon-schedule", choices=("updates_exponential", "decisions_linear"),
                        default="updates_exponential")
    parser.add_argument("--epsilon-start", type=float, default=1.)
    parser.add_argument("--epsilon-min", type=float, default=.05)
    parser.add_argument("--epsilon-decay", type=float, default=.9995)
    parser.add_argument("--epsilon-decay-decisions", type=int, default=10000)
    parser.add_argument("--max-updates", type=int,
                        help="Cumulative gradient-update limit; finish the current episode before stopping.")
    parser.add_argument("--virtual-count", type=int, choices=[0, 2, 5], default=0)
    parser.add_argument("--virtual-strategy", choices=["far", "random", "demand-proxy"], default="far")
    parser.add_argument("--placement-seed", type=int, default=42)
    parser.add_argument("--queue-policy", choices=["severity", "fifo", "hybrid"], default="severity")
    parser.add_argument("--aging-seconds", type=float, default=300.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int,
                        help="Patient generation seconds; default 3600 for gangnam, 500 for legacy.")
    parser.add_argument("--demand-profile", choices=("gangnam", "legacy"), default="gangnam",
                        help="Default: calibrated Gangnam hourly demand. legacy reproduces old dense experiments.")
    parser.add_argument("--demand-scale", type=float, default=1.)
    parser.add_argument("--curriculum", default="", help="Demand multipliers, e.g. 0.1,0.3,1; each stage lasts 9 episodes")
    parser.add_argument("--cycle-curriculum", action="store_true",
                        help="Repeat curriculum demand levels instead of keeping only the final level.")
    parser.add_argument("--treatment-seconds", type=float, default=900.)
    parser.add_argument("--bed-fraction", type=float, default=.08)
    parser.add_argument("--max-wait", type=float, default=1800.)
    parser.add_argument("--mission-timeout", type=float, default=1800.)
    parser.add_argument("--candidate-limit", type=int, default=1200)
    parser.add_argument("--admission-mode", choices=("precheck", "request", "arrival"), default="precheck",
                        help="arrival checks admission after transport; request/precheck reproduce earlier experiments.")
    parser.add_argument("--congestion-reward", choices=("penalty", "balanced"), default="penalty",
                        help="balanced rewards spare capacity below 50%% occupancy; penalty preserves legacy experiments.")
    parser.add_argument("--validation-seeds", default="30001,30002,30003")
    parser.add_argument("--validation-scenarios",
                        help="Default: selected training scenarios, or 1 when training all scenarios.")
    parser.add_argument("--validation-demand-levels", help="Fixed absolute demand levels, e.g. 0.1,0.3")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    try:
        config = SimulationConfig(
            arrival_steps=args.steps if args.steps is not None else (DEFAULT_DURATION if args.demand_profile == "gangnam" else 500),
            demand_rates_per_hour=PATIENTS_PER_HOUR if args.demand_profile == "gangnam" else (),
            demand_scale=args.demand_scale,
            max_wait=args.max_wait, mission_timeout=args.mission_timeout,
            treatment_seconds=args.treatment_seconds, bed_fraction=args.bed_fraction,
            seed=args.seed, candidate_limit=args.candidate_limit,
            pending_reward_feature=args.pending_reward_feature,
            golden_margin_feature=args.golden_margin_feature,
            candidate_reward_feature=args.candidate_reward_feature,
            emergency_yield=args.emergency_yield,
            repeat_background_traffic=args.repeat_background_traffic)
        config = replace(config, admission_mode=args.admission_mode)
        config = replace(config, congestion_reward_mode=args.congestion_reward)
        learning = LearningConfig(gamma=args.gamma, learning_rate=args.learning_rate,
            batch_size=args.batch_size, replay_capacity=args.replay_capacity, reward_scale=args.reward_scale,
            target_sync_mode=args.target_sync_mode, target_sync_interval=args.target_sync_interval,
            target_sync_episodes=args.target_sync_episodes, target_tau=args.target_tau,
            epsilon_schedule=args.epsilon_schedule, epsilon_start=args.epsilon_start,
            epsilon_min=args.epsilon_min, epsilon_decay=args.epsilon_decay,
            epsilon_decay_decisions=args.epsilon_decay_decisions)
        selected = parse_numbers(args.train_scenarios, int) if args.train_scenarios is not None else None
        validation_ids = (parse_numbers(args.validation_scenarios, int) if args.validation_scenarios
                          else selected or [1])
        curriculum = parse_numbers(args.curriculum) if args.curriculum else ()
        train(args.episodes, args.gui, args.ambulances, args.eval_every, args.resume,
              args.queue_policy, args.aging_seconds, args.virtual_count,
              args.virtual_strategy, args.placement_seed, config, curriculum,
              parse_numbers(args.validation_seeds, int),
              validation_ids, args.output_root, args.quiet,
              trace_decisions=args.trace_decisions,
              validation_demands=parse_numbers(args.validation_demand_levels) if args.validation_demand_levels else None,
              cycle_curriculum=args.cycle_curriculum,
              training_scenarios=selected, learning_config=learning, max_updates=args.max_updates)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
