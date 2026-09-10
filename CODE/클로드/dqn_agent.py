# dqn_agent.py  ── 개선 버전
#!/usr/bin/env python3
"""
[수정 사항]
  1. Vanilla DQN → Double DQN (overestimation bias 제거)
  2. target network: init 1회 → N스텝마다 주기적 업데이트
  3. epsilon: 곱셈(0.995^n) → 선형 감쇠 (decay_steps 기반, 더 예측 가능)
  4. MSE loss → Huber loss (이상치 robust)
  5. gradient clipping 추가 (norm ≤ 10)
  6. 관측 정규화: RunningMeanStd 온라인 통계 기반
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# ── 신경망 ──────────────────────────────────────────────────
class DQNNetwork(nn.Module):
    """
    Dueling DQN 아키텍처:
      공유 feature 추출 → Value 스트림 + Advantage 스트림 분리
      Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
    원본 단일 스트림보다 value function 학습이 안정적.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        feat = self.feature(x)
        val  = self.value_stream(feat)           # (B, 1)
        adv  = self.adv_stream(feat)             # (B, A)
        # Q = V + A - mean(A) : mean subtraction으로 identifiability 확보
        return val + adv - adv.mean(dim=1, keepdim=True)


# ── 관측 정규화 ──────────────────────────────────────────────
class RunningNorm:
    """
    Welford 온라인 알고리즘으로 실행 평균·분산 추적.
    학습 중 관측값이 자동 정규화되어 스케일 불일치 문제 해결.
    """
    def __init__(self, dim):
        self.n    = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2   = np.ones(dim,  dtype=np.float64)

    def update(self, x: np.ndarray):
        self.n += 1
        delta       = x - self.mean
        self.mean  += delta / self.n
        self.M2    += delta * (x - self.mean)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.M2 / max(self.n, 1)) + 1e-8
        return ((x - self.mean) / std).astype(np.float32)


# ── DQN 에이전트 ─────────────────────────────────────────────
class DQNAmbulanceAgent:
    def __init__(self, state_dim, action_dim,
                 lr=5e-4, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=3000,
                 batch_size=64, memory_size=20000, min_memory=256,
                 target_update_freq=200):   # ← [수정 ②] 주기적 업데이트
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.min_memory   = min_memory
        self.target_update_freq = target_update_freq

        # [수정 ③] 선형 epsilon 감쇠
        self.eps_start      = eps_start
        self.eps_end        = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.train_count    = 0

        self.memory = deque(maxlen=memory_size)
        self.norm   = RunningNorm(state_dim)   # [수정 ⑥]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # [수정 ①] Double DQN: online / target 두 네트워크
        self.online = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn   = nn.HuberLoss()        # [수정 ④]

        self.loss_history   = []
        self.reward_history = []

    # ── epsilon (선형 감쇠) ──
    @property
    def epsilon(self) -> float:
        frac = min(1.0, self.train_count / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    # ── 행동 선택 ──
    def select_action(self, state: np.ndarray) -> int:
        self.norm.update(state)
        norm_s = self.norm.normalize(state)
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            t = torch.FloatTensor(norm_s).unsqueeze(0).to(self.device)
            return self.online(t).argmax().item()

    def greedy_action(self, state: np.ndarray) -> int:
        """평가 전용 (epsilon=0)"""
        norm_s = self.norm.normalize(state)
        with torch.no_grad():
            t = torch.FloatTensor(norm_s).unsqueeze(0).to(self.device)
            return self.online(t).argmax().item()

    # ── 전이 저장 ──
    def store_transition(self, state, action, reward, next_state, done):
        s  = self.norm.normalize(np.array(state,      dtype=np.float64))
        ns = self.norm.normalize(np.array(next_state, dtype=np.float64))
        self.memory.append((s, int(action), float(reward), ns, float(done)))
        self.reward_history.append(float(reward))

    # ── 학습 1스텝 ──
    def train_step(self) -> float | None:
        if len(self.memory) < self.min_memory:
            return None

        batch  = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d = zip(*batch)

        S  = torch.FloatTensor(np.array(s)).to(self.device)
        A  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        R  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        NS = torch.FloatTensor(np.array(ns)).to(self.device)
        D  = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        # [수정 ①] Double DQN:
        #   online network → 최적 action 선택
        #   target network → 선택된 action의 Q값 평가
        with torch.no_grad():
            best_a   = self.online(NS).argmax(1, keepdim=True)   # online으로 선택
            q_next   = self.target(NS).gather(1, best_a)          # target으로 평가
            q_target = R + self.gamma * q_next * (1 - D)

        q_pred = self.online(S).gather(1, A)
        loss   = self.loss_fn(q_pred, q_target)   # [수정 ④] Huber loss

        self.optimizer.zero_grad()
        loss.backward()
        # [수정 ⑤] gradient clipping
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_count += 1
        self.loss_history.append(loss.item())

        # [수정 ②] 주기적 target 업데이트
        if self.train_count % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()
