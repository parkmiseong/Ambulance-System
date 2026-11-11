"""Isolate terminal hospital-choice learning and audit chronological replay rewards.

This diagnostic uses the production observation, masks, rewards, transition
collector and DQN. Its one-patient environment has scripted travel times and
does not model SUMO traffic or establish real-world policy performance.
No existing checkpoint or training log is modified.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from baseline import build_state_vector, state_dimension
from dqn_agent import DQNAmbulanceAgent
from episode_transitions import EpisodeTransitions
from experiment_config import SimulationConfig, parse_numbers, seed_everything
from hospital_capacity import HospitalCapacity
from hospital_policy import valid_actions
from reward import calculate_reward, HOSPITAL_TYPES

ROOT = Path(__file__).resolve().parents[1]


def single_patient_case(rng):
    """Swap hospital quality randomly: index is not a shortcut to the answer."""
    ktas = rng.randint(1, 5)
    travel = [rng.uniform(30., 110.), rng.uniform(250., 600.)]
    rng.shuffle(travel)
    occupancy = rng.uniform(.2, .6)
    hospitals = [dict(id=f"H{i}", type=HOSPITAL_TYPES[0], occupancy=occupancy,
                      reachable=True, estimated_travel_time=seconds,
                      route_length=seconds * 10., sumo_x=0., sumo_y=0.)
                 for i, seconds in enumerate(travel)]
    config = SimulationConfig(observation_version=3)
    HospitalCapacity(hospitals, config)
    state = build_state_vector((rng.uniform(0, 10000), rng.uniform(0, 10000)),
                               ktas, hospitals, config=config)
    actions = valid_actions(hospitals, ktas, config.occupancy_limit)
    rewards = [calculate_reward(seconds, ktas, h, occupancy_at_selection=occupancy,
                                reward_version=4) for seconds, h in zip(travel, hospitals)]
    return state, actions, rewards, travel


def evaluate_single_patient(agent, cases):
    correct, regrets, scaled_errors = 0, [], []
    for state, actions, rewards, _ in cases:
        q = agent.q_values(state)
        selected = max(actions, key=lambda a: q[a])
        best = max(actions, key=lambda a: rewards[a])
        correct += selected == best
        regrets.append(rewards[best] - rewards[selected])
        scaled_errors.extend(abs(float(q[a]) - rewards[a] / agent.reward_scale) for a in actions)
    return dict(cases=len(cases), best_reward_choice_rate=correct / len(cases),
                mean_patient_reward_regret=float(np.mean(regrets)),
                mean_scaled_q_absolute_error=float(np.mean(scaled_errors)))


def single_patient_experiment(seeds, episodes=1200, evaluation_cases=256):
    if episodes < 32 or evaluation_cases < 1 or not seeds:
        raise ValueError("Require >=32 episodes, >=1 evaluation case and at least one seed.")
    # All runs use the same held-out cases; their generation consumes no training RNG.
    evaluation_rng = random.Random(91001)
    cases = [single_patient_case(evaluation_rng) for _ in range(evaluation_cases)]
    results = []
    for seed in seeds:
        seed_everything(seed)
        rng = random.Random(seed + 100000)
        agent = DQNAmbulanceAgent(state_dimension(2, observation_version=3), 2,
                                  architecture="hospital_shared_v4", reward_scale=100.)
        curve = [dict(episode=0, **evaluate_single_patient(agent, cases))]
        started = time.monotonic()
        for episode in range(1, episodes + 1):
            state, actions, rewards, _ = single_patient_case(rng)
            action = agent.select_action(state, actions)
            transitions = EpisodeTransitions(agent)
            transitions.begin(state, action, actions)
            transitions.add_reward(rewards[action])
            transitions.close()
            if episode % 10 == 0:
                agent.update_target_network()
            if episode in (100, 300, 600, episodes):
                metrics = dict(episode=episode, **evaluate_single_patient(agent, cases))
                curve.append(metrics)
                print(f"[Single patient] seed={seed} episode={episode} "
                      f"best-choice={metrics['best_reward_choice_rate']:.1%} "
                      f"regret={metrics['mean_patient_reward_regret']:.3f}", flush=True)
        results.append(dict(seed=seed, curve=curve, replay=len(agent.memory),
                            epsilon=agent.epsilon, seconds=time.monotonic() - started))
    return dict(environment="scripted one-patient terminal episodes; not SUMO",
                architecture="hospital_shared_v4", reward_version=4, reward_scale=100.,
                episodes_per_seed=episodes, evaluation_seed=91001,
                results=results,
                all_final_choice_rates_at_least_95_percent=all(
                    run["curve"][-1]["best_reward_choice_rate"] >= .95 for run in results))


def audit_episode(events):
    """Attribute outcomes to the current interval, preserving log callback order."""
    decisions, assigned, intervals, outcomes = [], {}, [], []
    unattributed = 0.
    for kind, data in events:
        if kind == "decision":
            assigned[data["patient_id"]] = len(decisions)
            decisions.append(data)
            intervals.append(0.)
        elif kind == "mission":
            reward = float(data["reward"])
            receiver = len(decisions) - 1
            if receiver < 0:
                unattributed += reward
            else:
                intervals[receiver] += reward
            outcomes.append(dict(patient_id=data["patient_id"], reward=reward,
                                 receiver_index=receiver))
    # Resolve after reading all events, but never across episodes.
    completed = []
    for outcome in outcomes:
        own = assigned.get(outcome["patient_id"])
        receiver = outcome["receiver_index"]
        completed.append(dict(**outcome, own_decision_index=own,
                              intervening_decisions=receiver - own if own is not None else None,
                              same_patient_interval=own is not None and receiver == own))
    return dict(decisions=len(decisions), interval_rewards=intervals,
                unattributed_reward=unattributed, outcomes=completed)


def audit_checkpoint(checkpoint_path, log_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    signature = checkpoint.get("metadata", {}).get("experiment_signature")
    if not signature:
        raise ValueError("Checkpoint has no experiment signature.")
    last_episode = int(checkpoint["episode"])
    groups, latest_completed = defaultdict(list), {}
    with Path(log_path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row.get("run_type") != "training" or signature not in row.get("model_path", "")
                    or not row.get("episode")):
                continue
            episode = int(row["episode"])
            if episode > last_episode:
                continue
            key = (row["run_id"], episode)
            if row["record_type"] in ("decision", "mission"):
                groups[key].append((row["record_type"], json.loads(row["metrics_json"])))
            elif row["record_type"] == "episode" and row.get("status") == "completed":
                latest_completed[episode] = key
    episodes = []
    for episode, key in sorted(latest_completed.items()):
        audit = audit_episode(groups[key])
        episodes.append(dict(episode=episode, run_id=key[0], **audit))
    memory = checkpoint.get("replay_memory", [])
    if not memory:
        raise ValueError("Audit requires a full checkpoint with replay memory.")
    replay_rewards = np.asarray([item[2] for item in memory], dtype=float)
    terminals = np.asarray([item[2] for item in memory if item[4]], dtype=float)
    scale = float(checkpoint.get("reward_scale", 1.))
    reconstructed = np.asarray([reward / scale for ep in episodes for reward in ep["interval_rewards"]])
    # deque may have dropped older entries; compare its retained suffix only.
    coverage = all(ep["decisions"] > 0 for ep in episodes) and len(reconstructed) >= len(memory)
    matches = bool(coverage and np.allclose(reconstructed[-len(memory):], replay_rewards,
                                           rtol=1e-6, atol=1e-6))
    outcomes = [outcome for ep in episodes for outcome in ep["outcomes"]]
    assigned = [outcome for outcome in outcomes if outcome["own_decision_index"] is not None]
    own = [outcome for outcome in assigned if outcome["same_patient_interval"]]
    distances = [outcome["intervening_decisions"] for outcome in assigned]
    reward_abs = sum(abs(outcome["reward"]) for outcome in outcomes)
    summary = dict(
        saved_episode=last_episode, replay_size=len(memory), traced_episodes=len(episodes),
        trace_coverage_sufficient_for_replay_comparison=bool(coverage),
        chronological_rewards_match_saved_replay=matches,
        reward_scale=scale, zero_reward_transitions=int((replay_rewards == 0).sum()),
        terminal_transitions=len(terminals),
        terminal_absolute_reward_share=float(abs(terminals).sum() / abs(replay_rewards).sum())
            if abs(replay_rewards).sum() else 0.,
        assigned_patient_outcomes=len(assigned),
        outcomes_in_same_patient_interval=len(own),
        same_patient_interval_rate=len(own) / len(assigned) if assigned else None,
        unassigned_patient_outcomes=sum(o["own_decision_index"] is None for o in outcomes),
        mean_intervening_decisions=float(np.mean(distances)) if distances else None,
        max_intervening_decisions=max(distances) if distances else None,
        same_patient_absolute_reward_share=sum(abs(o["reward"]) for o in own) / reward_abs
            if reward_abs else None,
        warning="Cross-patient interval rewards are valid system rewards, not proof of a replay bug.",
    )
    return dict(summary=summary, episodes=episodes)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--log", type=Path, default=ROOT / "results/experiment_history.csv")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--evaluation-cases", type=int, default=256)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    # Small MLP batches run faster without CPU thread-pool overhead; local to this process.
    torch.set_num_threads(1)
    checksum = digest(args.checkpoint)
    audit = audit_checkpoint(args.checkpoint, args.log)
    print("[Replay audit]", json.dumps(audit["summary"], ensure_ascii=False), flush=True)
    experiment = single_patient_experiment(parse_numbers(args.seeds, int), args.episodes,
                                           args.evaluation_cases)
    if digest(args.checkpoint) != checksum:
        raise RuntimeError("The source checkpoint changed while diagnostics ran; rerun after training stops.")
    output = args.output_dir or ROOT / "results/diagnostics" / (
        time.strftime("learning_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    report = dict(source_checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checksum,
                  single_patient=experiment, replay_audit=audit)
    (output / "diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Diagnostic report] {output / 'diagnosis.json'}", flush=True)


if __name__ == "__main__":
    main()
