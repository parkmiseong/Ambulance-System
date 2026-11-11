"""Masked Double DQN with versioned learning settings and resumable counters."""
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from experiment_config import LearningConfig

REPLAY_SCHEMA_VERSION = 3


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(),
                                nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim))

    def forward(self, x):
        return self.fc(x)


class HospitalQNetwork(nn.Module):
    """Shared hospital scorer; feature permutations permute output Q values."""
    def __init__(self, state_dim, action_dim, global_dim=15, hospital_dim=9):
        super().__init__()
        if state_dim != global_dim + hospital_dim * action_dim:
            raise ValueError("HospitalQNetwork state size differs from its observation layout.")
        self.action_dim, self.global_dim = action_dim, global_dim
        self.hospital_dim = hospital_dim
        self.scorer = nn.Sequential(nn.Linear(global_dim + 4 * hospital_dim, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        hospitals = x[..., self.global_dim:].reshape(*x.shape[:-1], self.action_dim, self.hospital_dim)
        context = torch.cat((hospitals.mean(dim=-2), hospitals.amin(dim=-2),
                             hospitals.amax(dim=-2)), dim=-1)
        common = torch.cat((x[..., :self.global_dim], context), dim=-1)
        common = common.unsqueeze(-2).expand(*hospitals.shape[:-1], common.shape[-1])
        return self.scorer(torch.cat((common, hospitals), dim=-1)).squeeze(-1)


class HospitalHistoryQNetwork(HospitalQNetwork):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim, global_dim=16)


class HospitalFeatureQNetwork(HospitalQNetwork):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim, global_dim=16, hospital_dim=11)


class DQNAmbulanceAgent:
    COUNTERS = ("total_decisions", "exploratory_decisions", "total_transitions",
                "training_updates", "target_updates")

    def __init__(self, state_dim, action_dim, *, architecture="flat_v3",
                 reward_scale=None, learning_config=None):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.metadata, self.loaded_episode = {}, 0
        self.last_action_exploratory = False
        self.learning_config_explicit = learning_config is not None
        if learning_config is None:
            learning_config = LearningConfig.legacy(1. if reward_scale is None else reward_scale)
        elif reward_scale is not None and float(reward_scale) != learning_config.reward_scale:
            raise ValueError("reward_scale and learning_config differ.")
        self._configure_learning(learning_config)
        self.memory = deque(maxlen=learning_config.replay_capacity)
        for name in self.COUNTERS:
            setattr(self, name, 0)
        self.update_limit = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_networks(architecture)

    def _configure_learning(self, config):
        self.learning_config = config
        self.gamma, self.learning_rate = config.gamma, config.learning_rate
        self.epsilon, self.epsilon_min = config.epsilon_start, config.epsilon_min
        self.epsilon_decay, self.reward_scale = config.epsilon_decay, config.reward_scale

    def _build_networks(self, architecture):
        network = {"flat_v3": DQNNetwork, "hospital_shared_v4": HospitalQNetwork,
                   "hospital_shared_v5": HospitalHistoryQNetwork,
                   "hospital_shared_v6": HospitalFeatureQNetwork}.get(architecture)
        if network is None:
            raise ValueError(f"Unknown network architecture: {architecture}")
        self.architecture = architecture
        self.model = network(self.state_dim, self.action_dim).to(self.device)
        self.target_model = network(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # Initialization is a copy, not a learning-driven synchronization.
        self.target_model.load_state_dict(self.model.state_dict())

    def q_values(self, state):
        tensor = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device)
        with torch.no_grad():
            return self.model(tensor.unsqueeze(0)).squeeze(0).cpu().numpy()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_updates += 1

    def _update_target_after_gradient(self):
        config = self.learning_config
        if config.target_sync_mode == "updates":
            if self.training_updates % config.target_sync_interval == 0:
                self.update_target_network()
        elif config.target_sync_mode == "soft":
            with torch.no_grad():
                for target, online in zip(self.target_model.parameters(), self.model.parameters()):
                    target.lerp_(online, config.target_tau)
            self.target_updates += 1

    def record_decision(self):
        """Called only by the training engine, once per actual hospital choice."""
        self.total_decisions += 1
        self.exploratory_decisions += int(self.last_action_exploratory)
        config = self.learning_config
        if config.epsilon_schedule == "decisions_linear":
            fraction = min(self.total_decisions / config.epsilon_decay_decisions, 1.)
            self.epsilon = config.epsilon_start + fraction * (config.epsilon_min - config.epsilon_start)

    def training_statistics(self):
        result = {name: getattr(self, name) for name in self.COUNTERS}
        result["exploration_rate"] = (self.exploratory_decisions / self.total_decisions
                                      if self.total_decisions else 0.)
        return result

    def select_action(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(self.action_dim))
        valid_actions = [int(a) for a in valid_actions if 0 <= int(a) < self.action_dim]
        if not valid_actions:
            raise ValueError("No valid hospital action.")
        if np.random.rand() <= self.epsilon:
            self.last_action_exploratory = True
            return random.choice(valid_actions)
        self.last_action_exploratory = False
        masked = np.full(self.action_dim, -np.inf, dtype=np.float32)
        values = self.q_values(state)
        masked[valid_actions] = values[valid_actions]
        return int(np.argmax(masked))

    def store_transition(self, state, action, reward, next_state, done, next_valid_actions=None):
        state = np.array(state, dtype=np.float32, copy=True)
        next_state = np.array(next_state, dtype=np.float32, copy=True)
        if state.shape != (self.state_dim,) or next_state.shape != (self.state_dim,):
            raise ValueError("Replay states must match state_dim.")
        if not 0 <= int(action) < self.action_dim:
            raise ValueError("Replay action is out of range.")
        next_mask = np.zeros(self.action_dim, dtype=np.bool_)
        if not done:
            if next_valid_actions is None:
                raise ValueError("Nonterminal replay requires next valid actions.")
            actions = [int(a) for a in next_valid_actions]
            if not actions or any(a < 0 or a >= self.action_dim for a in actions):
                raise ValueError("Nonterminal replay requires in-range next actions.")
            next_mask[actions] = True
        self.memory.append((state, int(action), float(reward) / self.reward_scale,
                            next_state, bool(done), next_mask))
        self.total_transitions += 1

    def train_step(self, batch_size=None):
        batch_size = self.learning_config.batch_size if batch_size is None else batch_size
        if len(self.memory) < batch_size:
            return None
        if self.update_limit is not None and self.training_updates >= self.update_limit:
            return None
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        next_masks = torch.as_tensor(np.asarray(next_masks), dtype=torch.bool, device=self.device)
        current_q = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q = torch.zeros_like(rewards)
            nonterminal = ~dones
            if nonterminal.any():
                masks = next_masks[nonterminal]
                if not masks.any(dim=1).all():
                    raise ValueError("Nonterminal replay has no valid next action.")
                following = next_states[nonterminal]
                online_q = self.model(following).masked_fill(~masks, -torch.inf)
                next_actions = online_q.argmax(dim=1, keepdim=True)
                next_q[nonterminal] = self.target_model(following).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q
        loss = nn.SmoothL1Loss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.learning_config.gradient_clip_norm)
        self.optimizer.step()
        self.training_updates += 1
        self._update_target_after_gradient()
        if self.learning_config.epsilon_schedule == "updates_exponential":
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save_model(self, filepath="dqn_ambulance_model_ktas_v2.pth", *, episode=None, full_state=False):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        payload = {
            "replay_schema_version": REPLAY_SCHEMA_VERSION, "architecture": self.architecture,
            "reward_scale": self.reward_scale, "metadata": self.metadata,
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(), "epsilon": self.epsilon,
            "episode": episode, "state_dim": self.state_dim, "action_dim": self.action_dim,
            "learning_stats": self.training_statistics(),
        }
        if self.learning_config_explicit:
            payload["learning_config"] = self.learning_config.to_dict()
        if full_state:
            payload.update(replay_memory=list(self.memory), python_random_state=random.getstate(),
                           numpy_random_state=np.random.get_state(), torch_random_state=torch.get_rng_state())
            if torch.cuda.is_available():
                payload["cuda_random_states"] = torch.cuda.get_rng_state_all()
        temporary = str(filepath) + ".tmp"
        torch.save(payload, temporary)
        os.replace(temporary, filepath)
        print(f"[DQN] Saved: {filepath}")

    def load_model(self, filepath="dqn_ambulance_model_ktas_v2.pth", *, resume=False):
        if not os.path.exists(filepath):
            return False
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        if (checkpoint.get("state_dim", self.state_dim) != self.state_dim
                or checkpoint.get("action_dim", self.action_dim) != self.action_dim):
            raise ValueError("Checkpoint observation/action dimensions differ.")
        if resume and checkpoint.get("replay_schema_version") != REPLAY_SCHEMA_VERSION:
            raise ValueError("Incompatible replay schema: start a fresh sequential training run.")
        expected_signature = self.metadata.get("experiment_signature")
        saved_metadata = checkpoint.get("metadata", {})
        if resume and expected_signature and saved_metadata.get("experiment_signature") != expected_signature:
            raise ValueError("Checkpoint experiment settings do not match.")
        architecture = checkpoint.get("architecture", "flat_v3")
        scale = float(checkpoint.get("reward_scale", 1.))
        if resume and (architecture != self.architecture or scale != self.reward_scale):
            raise ValueError("Checkpoint network/reward scaling differs from this training run.")
        saved_config = checkpoint.get("learning_config")
        settings = LearningConfig(**saved_config) if saved_config is not None else LearningConfig.legacy(scale)
        if settings.reward_scale != scale:
            raise ValueError("Checkpoint learning configuration and reward scale differ.")
        if resume and self.learning_config_explicit and (
                saved_config is None or settings != self.learning_config):
            raise ValueError("Checkpoint learning settings differ from this training run.")
        self._configure_learning(settings)
        self.learning_config_explicit = saved_config is not None
        self.memory = deque(maxlen=settings.replay_capacity)
        self._build_networks(architecture)
        self.metadata = dict(saved_metadata)
        self.loaded_episode = int(checkpoint.get("episode") or 0)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        stats = checkpoint.get("learning_stats", {})
        for name in self.COUNTERS:
            setattr(self, name, int(stats.get(name, 0)))
        if resume:
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = float(checkpoint.get("epsilon", settings.epsilon_start))
            self.memory.extend(checkpoint.get("replay_memory", []))
            if "python_random_state" in checkpoint:
                random.setstate(checkpoint["python_random_state"])
            if "numpy_random_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_random_state"])
            if "torch_random_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_random_state"].cpu())
            if torch.cuda.is_available() and "cuda_random_states" in checkpoint:
                torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_random_states"]])
        return True
