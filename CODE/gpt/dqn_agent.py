# -*- coding: utf-8 -*-
"""KTAS DQN agent. 유효 action mask + Double DQN + SmoothL1Loss."""

import os
import random
import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


class DQNAmbulanceAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=30000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.0005

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_model = DQNNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state, valid_actions=None):
        """유효 병원 action 안에서만 선택."""
        if valid_actions is None:
            valid_actions = list(range(self.action_dim))
        valid_actions = [int(a) for a in valid_actions if 0 <= int(a) < self.action_dim]

        if not valid_actions:
            raise ValueError("선택 가능한 action이 없습니다.")

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)

        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0).detach().cpu().numpy()

        masked = np.full(self.action_dim, -np.inf, dtype=np.float32)
        masked[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked))

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        ))

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(
            np.asarray(next_states), dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.model(states).gather(1, actions)

        # Double DQN: online model로 action 선택, target model로 평가
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_model(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

        return float(loss.item())

    def save_model(self, filepath="dqn_ambulance_model_ktas_v2.pth", *, episode=None, full_state=False):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode": episode,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }
        if full_state:
            payload["replay_memory"] = list(self.memory)
            payload["python_random_state"] = random.getstate()
            payload["numpy_random_state"] = np.random.get_state()
            payload["torch_random_state"] = torch.get_rng_state()
        torch.save(payload, filepath)
        print(f"[DQN 에이전트] 저장 완료: {filepath}")

    def load_model(self, filepath="dqn_ambulance_model_ktas_v2.pth", *, resume=False):
        if not os.path.exists(filepath):
            return False
        # 최신 PyTorch는 weights_only=True가 기본이며 전체 학습 체크포인트는
        # deque/NumPy RNG 객체를 포함하므로 신뢰하는 로컬 파일만 로드한다.
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        if checkpoint.get("state_dim", self.state_dim) != self.state_dim or checkpoint.get("action_dim", self.action_dim) != self.action_dim:
            raise ValueError("체크포인트의 상태/행동 차원이 현재 환경과 다릅니다")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        if resume:
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = float(checkpoint.get("epsilon", 1.0))
            if "replay_memory" in checkpoint:
                self.memory.clear()
                self.memory.extend(checkpoint["replay_memory"])
            if "python_random_state" in checkpoint:
                random.setstate(checkpoint["python_random_state"])
            if "numpy_random_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_random_state"])
            if "torch_random_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_random_state"])
        return True
