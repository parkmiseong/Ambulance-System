"""Controlled delayed-outcome experiments; never updates production checkpoints.

Patient contexts and rewards are independent. Only the logical notification
schedule changes: each result before the next decision, or all results after
the final decision. This is NOT a physical SUMO travel-time simulation.
"""
import argparse
from itertools import product
import json
import math
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from diagnose_learning import ROOT, digest, single_patient_case
from dqn_agent import DQNAmbulanceAgent
from episode_transitions import EpisodeTransitions
from experiment_config import parse_numbers, seed_everything


PROFILES = {
    "single": (1, False, False, False),
    "sequential_2": (2, False, False, False),
    "overlap_2": (2, True, False, False),
    "sequential_3": (3, False, False, False),
    "overlap_3": (3, True, False, False),
    "reverse_3": (3, True, True, False),
    "overlap_3_history": (3, True, False, True),
}
DEFAULT_PROFILES = tuple(name for name in PROFILES if name != "overlap_3_history")


class ControlledAgent(DQNAmbulanceAgent):
    """Keep target synchronization equal across different episode lengths."""
    def __init__(self):
        super().__init__(33, 2, architecture="hospital_shared_v4", reward_scale=100.)
        self.updates = 0
        self.target_updates = 0

    def train_step(self, batch_size=32):
        loss = super().train_step(batch_size)
        if loss is not None:
            self.updates += 1
            if self.updates % 100 == 0:
                self.update_target_network()
                self.target_updates += 1
        return loss


def observation(case, index, patients, pending_reward=None, reward_scale=100.):
    state = case[0].copy()
    # Expose decision stage equally in sequential and delayed conditions.
    # Other fields stay identical: no queues, resource competition or traffic.
    state[4] = index / (patients + 3602.)
    state[5] = (patients - index) / patients
    if pending_reward is not None:
        # Diagnostic-only replacement of an otherwise zero feature. In this
        # deterministic task, prior selected cases' rewards are computable at
        # selection time. No future patient's reward is exposed.
        state[14] = pending_reward / reward_scale
    return state


def interval_rewards(rewards, delayed):
    return [0.] * (len(rewards) - 1) + [math.fsum(rewards)] if delayed else list(rewards)


def discounted_return(rewards, delayed, gamma=.95):
    return math.fsum(gamma ** index * reward
                     for index, reward in enumerate(interval_rewards(rewards, delayed)))


def oracle_return(cases, delayed, gamma=.95):
    # Enumerate up to 2^3 complete policies, rather than assuming a greedy oracle.
    return max(discounted_return([case[2][action] for case, action in zip(cases, actions)],
                                 delayed, gamma)
               for actions in product(*(case[1] for case in cases)))


def train_episode(agent, cases, delayed, reverse=False, forced_actions=None, history=False):
    transitions = EpisodeTransitions(agent)
    rewards, selected, finish_order = [], [], []
    for index, case in enumerate(cases):
        state = observation(case, index, len(cases),
                            math.fsum(rewards) if history else None,
                            getattr(agent, "reward_scale", 100.))
        action = (forced_actions[index] if forced_actions is not None
                  else agent.select_action(state, case[1]))
        transitions.begin(state, action, case[1])
        selected.append(action)
        rewards.append(case[2][action])
        if not delayed:
            transitions.add_reward(rewards[-1])
            finish_order.append(index)
    if delayed:
        finish_order = list(reversed(range(len(cases)))) if reverse else list(range(len(cases)))
        for index in finish_order:
            transitions.add_reward(rewards[index])
    transitions.close()
    return dict(selected=selected, patient_rewards=rewards, finish_order=finish_order,
                total_reward=transitions.total_reward, transitions=transitions.count)


def evaluate(agent, cases, delayed, history=False):
    if not history:
        states = np.stack([observation(case, index, len(episode))
                           for episode in cases for index, case in enumerate(episode)])
        with torch.no_grad():
            values = agent.model(torch.as_tensor(states, device=agent.device)).cpu().numpy()
    cursor, correct, patient_regrets, return_regrets = 0, 0, [], []
    for episode in cases:
        rewards = []
        for index, case in enumerate(episode):
            q = (agent.q_values(observation(case, index, len(episode),
                                            math.fsum(rewards), agent.reward_scale))
                 if history else values[cursor])
            action = max(case[1], key=lambda a: q[a])
            best = max(case[1], key=lambda a: case[2][a])
            correct += action == best
            patient_regrets.append(case[2][best] - case[2][action])
            rewards.append(case[2][action])
            cursor += 1
        return_regrets.append(oracle_return(episode, delayed, agent.gamma)
                              - discounted_return(rewards, delayed, agent.gamma))
    return dict(evaluation_episodes=len(cases), patient_choices=cursor,
                best_patient_reward_choice_rate=correct / cursor,
                mean_patient_reward_regret=float(np.mean(patient_regrets)),
                mean_discounted_episode_return_regret=float(np.mean(return_regrets)))


def experiment(seeds, decisions=3600, evaluation_episodes=256, profiles=DEFAULT_PROFILES):
    if decisions < 96 or decisions % 6 or evaluation_episodes < 1 or not seeds:
        raise ValueError("Decision budget must be >=96 and divisible by 6; require evaluation cases and seeds.")
    if not profiles or any(name not in PROFILES for name in profiles):
        raise ValueError("Unknown or empty profile selection.")
    evaluation_rng = random.Random(92001)
    # Same base contexts across conditions; each profile takes the first N patients.
    base_cases = [[single_patient_case(evaluation_rng) for _ in range(3)]
                  for _ in range(evaluation_episodes)]
    results = []
    for name in profiles:
        patients, delayed, reverse, history = PROFILES[name]
        held_out = [episode[:patients] for episode in base_cases]
        for seed in seeds:
            seed_everything(seed)
            rng = random.Random(seed + 100000)
            agent = ControlledAgent()
            curve = [dict(decisions=0, **evaluate(agent, held_out, delayed, history))]
            started = time.monotonic()
            for episode in range(1, decisions // patients + 1):
                base = [single_patient_case(rng) for _ in range(3)]
                train_episode(agent, base[:patients], delayed, reverse, history=history)
                used = episode * patients
                if used in (300, 900, 1800, decisions):
                    metrics = dict(decisions=used, **evaluate(agent, held_out, delayed, history))
                    curve.append(metrics)
                    print(f"[Overlap] {name} seed={seed} decisions={used} "
                          f"best-choice={metrics['best_patient_reward_choice_rate']:.1%} "
                          f"return-regret={metrics['mean_discounted_episode_return_regret']:.3f}",
                          flush=True)
            results.append(dict(profile=name, seed=seed, patients=patients,
                                delayed=delayed, reversed_completion=reverse,
                                pending_reward_history_feature=history,
                                training_episodes=decisions // patients,
                                replay=len(agent.memory), updates=agent.updates,
                                target_updates=agent.target_updates, epsilon=agent.epsilon,
                                seconds=time.monotonic() - started, curve=curve))
    # Fail rather than report an unfair budget comparison.
    assert len({r["updates"] for r in results}) == 1
    assert len({r["target_updates"] for r in results}) == 1
    assert len({r["epsilon"] for r in results}) == 1
    summary = []
    for profile in profiles:
        group = [r for r in results if r["profile"] == profile]
        summary.append(dict(profile=profile,
                            choice_rates=[r["curve"][-1]["best_patient_reward_choice_rate"] for r in group],
                            mean_choice_rate=float(np.mean([
                                r["curve"][-1]["best_patient_reward_choice_rate"] for r in group])),
                            mean_return_regret=float(np.mean([
                                r["curve"][-1]["mean_discounted_episode_return_regret"] for r in group]))))
    return dict(environment="independent scripted patient rewards with logical observation delay; not SUMO",
                decision_budget_per_run=decisions, evaluation_seed=92001,
                target_sync="every 100 actual gradient updates in every profile",
                assumptions=["two hospitals, no queues/capacity competition/traffic",
                             "same observation for corresponding sequential/delayed patient contexts",
                             "only result notification order changes within each matched patient-count pair",
                             "history control alone adds prior pending reward sum / 100 at feature 14",
                             "history values are exact in this scripted task, not guaranteed in SUMO",
                             "no production checkpoints or parameters updated"],
                summary=summary, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Read-only provenance and integrity check")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--decisions", type=int, default=3600)
    parser.add_argument("--evaluation-episodes", type=int, default=256)
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES),
                        help="Comma-separated conditions; include overlap_3_history for the observation control")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checksum = digest(args.checkpoint)
    report = experiment(parse_numbers(args.seeds, int), args.decisions, args.evaluation_episodes,
                        args.profiles.split(","))
    if digest(args.checkpoint) != checksum:
        raise RuntimeError("The source checkpoint changed while diagnostics ran.")
    report.update(source_checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checksum)
    output = args.output_dir or ROOT / "results/diagnostics" / (
        time.strftime("overlap_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    (output / "overlap.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Summary]", json.dumps(report["summary"]), flush=True)
    print(f"[Overlap report] {output / 'overlap.json'}", flush=True)


if __name__ == "__main__":
    main()
