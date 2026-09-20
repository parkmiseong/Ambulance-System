#!/usr/bin/env python3

import os
import random
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
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_model = DQNNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def select_action(self, state, valid_actions=None):
        if valid_actions is None or len(valid_actions) == 0:
            valid_actions = list(range(self.action_dim))

        valid_actions = [int(a) for a in valid_actions if 0 <= int(a) < self.action_dim]

        # 탐험 시에도 유효한 행동 중에서만 무작위 선택
        if np.random.rand() <= valid_actions:
            return random.choice(valid_actions)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)

        # 유효하지 않은 행동의 Q값을 음의 무한대로 마스킹
        masked_q = torch.full_like(q_values, -1e9)
        masked_q[valid_actions] = q_values[valid_actions]

        return int(torch.argmax(masked_q).item())

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
        next_states = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.model(states).gather(1, actions)

        # Double DQN: 행동 선택은 online network, 값 평가는 target network.
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_model(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.SmoothL1Loss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    def save_model(self, filepath="dqn_ambulance_model_v2.pth"):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            filepath,
        )
        print(f"\n[DQN 에이전트] 가중치 저장 완료: {filepath}")

    def load_model(self, filepath="dqn_ambulance_model_v2.pth"):
        if not os.path.exists(filepath):
            print(f"\n[DQN 에이전트] 저장된 모델 파일이 없어 새로 시작합니다: {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint.get("target_model_state_dict", checkpoint["model_state_dict"]))

        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass

        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        print(f"\n[DQN 에이전트] 학습 모델 로드 완료: {filepath} (Epsilon: {self.epsilon:.4f})")
        return True
