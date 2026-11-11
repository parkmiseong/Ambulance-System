"""Versioned, reproducible research settings shared by training and evaluation."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import os
import random

import numpy as np
import torch

STATE_SCHEMA_VERSION = 5
EXPERIMENT_VERSION = 6
SCENARIOS = [
    {"id": i + 1, "prob": p, "scale": scale, "name": f"demand{p}-traffic{scale}"}
    for i, (p, scale) in enumerate((p, s) for p in (.2, .3, .4) for s in (.3, .5, .8))
]


@dataclass(frozen=True)
class SimulationConfig:
    arrival_steps: int = 500
    demand_scale: float = 1.0
    max_wait: float = 1800.
    mission_timeout: float = 1800.
    treatment_seconds: float = 900.
    bed_fraction: float = .08
    occupancy_limit: float = .85
    seed: int = 42
    candidate_limit: int = 1200
    reward_version: int = 4
    observation_version: int = STATE_SCHEMA_VERSION
    pending_reward_feature: str = "estimate"
    golden_margin_feature: bool = False
    candidate_reward_feature: bool = False

    def __post_init__(self):
        if self.observation_version not in (3, 4, 5):
            raise ValueError("observation_version must be 3, 4 or 5.")
        if type(self.golden_margin_feature) is not bool or type(self.candidate_reward_feature) is not bool:
            raise ValueError("Candidate feature switches must be boolean.")
        if self.observation_version < 5 and (self.golden_margin_feature or self.candidate_reward_feature):
            raise ValueError("Candidate features require observation_version 5.")
        if self.pending_reward_feature not in ("estimate", "zero"):
            raise ValueError("pending_reward_feature must be estimate or zero.")
        if self.reward_version not in (3, 4):
            raise ValueError("reward_version must be 3 or 4.")
        if (self.arrival_steps < 1 or self.candidate_limit < 1
                or int(self.arrival_steps) != self.arrival_steps
                or int(self.candidate_limit) != self.candidate_limit):
            raise ValueError("arrival_steps and candidate_limit must be positive.")
        for name in ("max_wait", "mission_timeout", "treatment_seconds", "bed_fraction", "occupancy_limit"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.demand_scale) or not 0 <= self.demand_scale <= 2.5:
            raise ValueError("demand_scale must be between 0 and 2.5.")
        if not 0 < self.bed_fraction <= self.occupancy_limit <= 1:
            raise ValueError("Require 0 < bed_fraction <= occupancy_limit <= 1.")
        if not 0 <= self.seed < 2**31 - 100000:
            raise ValueError("seed is out of range.")

    @property
    def drain_deadline(self):
        # A last arrival may wait max_wait, then use its full mission timeout.
        return self.arrival_steps + self.max_wait + self.mission_timeout + 2

    def to_dict(self):
        return asdict(self)



@dataclass(frozen=True)
class LearningConfig:
    gamma: float = .95
    learning_rate: float = .0005
    batch_size: int = 32
    replay_capacity: int = 30000
    reward_scale: float = 100.
    target_sync_mode: str = "updates"
    target_sync_interval: int = 200
    target_sync_episodes: int = 10
    target_tau: float = .005
    epsilon_start: float = 1.
    epsilon_min: float = .05
    epsilon_decay: float = .9995
    epsilon_schedule: str = "updates_exponential"
    epsilon_decay_decisions: int = 10000
    gradient_clip_norm: float = 5.

    def __post_init__(self):
        for name in ("learning_rate", "reward_scale", "gradient_clip_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("batch_size", "replay_capacity", "target_sync_interval",
                     "target_sync_episodes", "epsilon_decay_decisions"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must cover batch_size.")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in 0..1.")
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1.")
        if not math.isfinite(self.epsilon_decay) or not 0 < self.epsilon_decay <= 1:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if not math.isfinite(self.target_tau) or not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must be in (0, 1].")
        if self.target_sync_mode not in ("updates", "episodes", "soft"):
            raise ValueError("Unknown target synchronization mode.")
        if self.epsilon_schedule not in ("updates_exponential", "decisions_linear"):
            raise ValueError("Unknown epsilon schedule.")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def legacy(cls, reward_scale=1.):
        return cls(reward_scale=reward_scale, target_sync_mode="episodes")


def network_architecture(config):
    if config.observation_version == 5:
        return "hospital_shared_v6"
    if config.observation_version == 4:
        return "hospital_shared_v5"
    return "hospital_shared_v4" if config.reward_version == 4 else "flat_v3"


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


@contextmanager
def isolated_random_state():
    """Validation must not alter exploration, replay sampling, or resume RNG."""
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def generate_schedule(scenario, edges, seed, config):
    probability = float(scenario["prob"]) * config.demand_scale
    if not 0 <= probability <= 1:
        raise ValueError("Scaled arrival probability must be between zero and one.")
    if not edges:
        raise ValueError("No verified patient edges.")
    # Separate streams make demand levels nested and preserve each call's data.
    arrivals = random.Random(seed)
    attributes = random.Random(seed + 100000)
    schedule = {}
    for step in range(1, config.arrival_steps + 1):
        edge = attributes.choice(edges)
        ktas = attributes.randint(1, 5)
        if arrivals.random() < probability:
            schedule[step] = {
                "edge_id": edge, "ktas": ktas, "pre_ktas": ktas,
                "occurrence_time": float(step), "patient_id": f"p{step}",
            }
    return schedule


def curriculum_multiplier(episode, levels, cycle=False):
    """Advance every nine scenarios, then keep the final demand level."""
    if not levels:
        return 1.
    stage = (episode - 1) // len(SCENARIOS)
    return levels[stage % len(levels) if cycle else min(stage, len(levels) - 1)]


def parse_numbers(value, cast=float):
    values = [cast(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Provide at least one value.")
    return values
