"""Versioned, reproducible research settings shared by training and evaluation."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import os
import random

import numpy as np
import torch

STATE_SCHEMA_VERSION = 4
EXPERIMENT_VERSION = 5
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

    def __post_init__(self):
        if self.observation_version not in (3, 4):
            raise ValueError("observation_version must be 3 or 4.")
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


def network_architecture(config):
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
