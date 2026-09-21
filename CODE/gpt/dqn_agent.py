#!/usr/bin/env python3

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# DQN 신경망
# ============================================================

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


# ============================================================
# DQN Ambulance Agent
# ============================================================

class DQNAmbulanceAgent:

    def __init__(self, state_dim, action_dim):

        self.state_dim = state_dim
        self.action_dim = action_dim

        # ----------------------------------------------------
        # Replay Memory
        # ----------------------------------------------------

        self.memory = deque(
            maxlen=30000
        )

        # ----------------------------------------------------
        # DQN Hyperparameters
        # ----------------------------------------------------

        self.gamma = 0.95

        # 초기 탐험률
        self.epsilon = 1.0

        # 최소 탐험률
        self.epsilon_min = 0.05

        # 기존 0.995보다 조금 완만하게 감소
        self.epsilon_decay = 0.999

        # 학습률
        self.learning_rate = 0.0005

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"[DQN] 사용 장치: {self.device}"
        )

        # ----------------------------------------------------
        # Online Network
        # ----------------------------------------------------

        self.model = DQNNetwork(
            state_dim,
            action_dim
        ).to(self.device)

        # ----------------------------------------------------
        # Target Network
        # ----------------------------------------------------

        self.target_model = DQNNetwork(
            state_dim,
            action_dim
        ).to(self.device)

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        # 초기에는 두 네트워크를 동일하게 설정
        self.update_target_network()

    # ========================================================
    # Target Network 업데이트
    # ========================================================

    def update_target_network(self):
        """
        Online Network의 가중치를
        Target Network에 복사합니다.
        """

        self.target_model.load_state_dict(
            self.model.state_dict()
        )

    # ========================================================
    # Action 선택
    # ========================================================

    def select_action(
        self,
        state,
        valid_actions=None
    ):
        """
        ε-greedy 방식으로 행동을 선택합니다.

        Parameters
        ----------
        state : numpy array
            현재 상태

        valid_actions : list
            현재 상황에서 선택 가능한 병원 action 목록

        Returns
        -------
        int
            선택된 병원 action index
        """

        # ----------------------------------------------------
        # valid_actions가 없으면 모든 action 허용
        # ----------------------------------------------------

        if (
            valid_actions is None
            or len(valid_actions) == 0
        ):

            valid_actions = list(
                range(self.action_dim)
            )

        # ----------------------------------------------------
        # Action index 정리
        # ----------------------------------------------------

        valid_actions = [
            int(action)
            for action in valid_actions
            if 0 <= int(action) < self.action_dim
        ]

        # 모든 action이 제거된 경우
        # 안전하게 전체 action을 사용
        if len(valid_actions) == 0:

            valid_actions = list(
                range(self.action_dim)
            )

        # ====================================================
        # 1. Exploration
        # ====================================================
        #
        # 기존 코드의 문제:
        #
        # if np.random.rand() <= valid_actions:
        #
        # valid_actions는 list이므로 잘못된 비교입니다.
        #
        # 올바른 ε-greedy:
        #
        # random < epsilon
        #
        # ====================================================

        if np.random.rand() < self.epsilon:

            selected_action = random.choice(
                valid_actions
            )

            return int(selected_action)

        # ====================================================
        # 2. Exploitation
        # ====================================================

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.model(
                state_tensor
            ).squeeze(0)

        # ----------------------------------------------------
        # Invalid Action Mask
        # ----------------------------------------------------
        #
        # 선택 불가능한 병원의 Q-value를
        # 매우 작은 값으로 변경합니다.
        #
        # 따라서 argmax를 수행해도
        # valid_actions 중에서만 선택됩니다.
        # ----------------------------------------------------

        masked_q = torch.full_like(
            q_values,
            -1e9
        )

        valid_indices = torch.as_tensor(
            valid_actions,
            dtype=torch.long,
            device=self.device
        )

        masked_q[valid_indices] = (
            q_values[valid_indices]
        )

        selected_action = torch.argmax(
            masked_q
        ).item()

        return int(selected_action)

    # ========================================================
    # Replay Memory 저장
    # ========================================================

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        하나의 transition을 Replay Memory에 저장합니다.

        transition:

            state
              ↓
            action
              ↓
            reward
              ↓
            next_state
              ↓
            done
        """

        self.memory.append(
            (
                np.asarray(
                    state,
                    dtype=np.float32
                ),

                int(action),

                float(reward),

                np.asarray(
                    next_state,
                    dtype=np.float32
                ),

                bool(done),
            )
        )

    # ========================================================
    # Replay Memory 크기
    # ========================================================

    def get_memory_size(self):
        """
        현재 Replay Memory에 저장된
        transition 개수를 반환합니다.
        """

        return len(self.memory)

    # ========================================================
    # DQN 학습
    # ========================================================

    def train_step(
        self,
        batch_size=32
    ):
        """
        Replay Memory에서 batch를 추출하여
        DQN을 한 번 학습합니다.

        Double DQN 방식을 사용합니다.
        """

        # ----------------------------------------------------
        # 충분한 데이터가 없으면 학습하지 않음
        # ----------------------------------------------------

        if len(self.memory) < batch_size:

            return None

        # ----------------------------------------------------
        # Random Mini-batch
        # ----------------------------------------------------

        batch = random.sample(
            self.memory,
            batch_size
        )

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = zip(*batch)

        # ----------------------------------------------------
        # Tensor 변환
        # ----------------------------------------------------

        states = torch.as_tensor(
            np.asarray(states),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        next_states = torch.as_tensor(
            np.asarray(next_states),
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        # ====================================================
        # Current Q Value
        # ====================================================

        current_q = self.model(
            states
        ).gather(
            1,
            actions
        )

        # ====================================================
        # Double DQN
        # ====================================================
        #
        # Online Network:
        #   다음 행동 선택
        #
        # Target Network:
        #   선택된 행동의 Q-value 평가
        #
        # 이렇게 하면 일반 DQN의
        # Q-value 과대평가 문제를 줄일 수 있습니다.
        # ====================================================

        with torch.no_grad():

            # ------------------------------------------------
            # Online Network가 다음 행동 선택
            # ------------------------------------------------

            next_actions = self.model(
                next_states
            ).argmax(
                dim=1,
                keepdim=True
            )

            # ------------------------------------------------
            # Target Network가 Q-value 평가
            # ------------------------------------------------

            next_q = self.target_model(
                next_states
            ).gather(
                1,
                next_actions
            )

            # ------------------------------------------------
            # Bellman Target
            # ------------------------------------------------

            target_q = (
                rewards
                + self.gamma
                * next_q
                * (1.0 - dones)
            )

        # ====================================================
        # Loss
        # ====================================================

        loss = nn.SmoothL1Loss()(
            current_q,
            target_q
        )

        # ====================================================
        # Backpropagation
        # ====================================================

        self.optimizer.zero_grad()

        loss.backward()

        # Gradient Explosion 방지
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=5.0
        )

        self.optimizer.step()

        # ====================================================
        # Epsilon 감소
        # ====================================================
        #
        # train_step이 호출될 때마다
        # 탐험률을 조금씩 감소시킵니다.
        #
        # epsilon_decay = 0.999
        #
        # 기존 0.995보다 천천히 감소하기 때문에
        # 충분한 탐험을 유지할 수 있습니다.
        # ====================================================

        if self.epsilon > self.epsilon_min:

            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

        return float(
            loss.item()
        )

    # ========================================================
    # 모델 저장
    # ========================================================

    def save_model(
        self,
        filepath="dqn_ambulance_model_v2.pth"
    ):
        """
        학습된 DQN 모델을 저장합니다.
        """

        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),

                "target_model_state_dict":
                    self.target_model.state_dict(),

                "optimizer_state_dict":
                    self.optimizer.state_dict(),

                "epsilon":
                    self.epsilon,

                "state_dim":
                    self.state_dim,

                "action_dim":
                    self.action_dim,
            },
            filepath
        )

        print(
            f"\n[DQN 에이전트] "
            f"가중치 저장 완료: {filepath}"
        )

    # ========================================================
    # 모델 불러오기
    # ========================================================

    def load_model(
        self,
        filepath="dqn_ambulance_model_v2.pth"
    ):
        """
        저장된 DQN 모델을 불러옵니다.

        Returns
        -------
        bool
            로드 성공 여부
        """

        # ----------------------------------------------------
        # 파일 존재 여부 확인
        # ----------------------------------------------------

        if not os.path.exists(filepath):

            print(
                f"\n[DQN 에이전트] "
                f"저장된 모델 파일이 없어 "
                f"새로 시작합니다: {filepath}"
            )

            return False

        # ----------------------------------------------------
        # Checkpoint 로드
        # ----------------------------------------------------

        checkpoint = torch.load(
            filepath,
            map_location=self.device
        )

        # ----------------------------------------------------
        # Online Network
        # ----------------------------------------------------

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        # ----------------------------------------------------
        # Target Network
        # ----------------------------------------------------

        self.target_model.load_state_dict(
            checkpoint.get(
                "target_model_state_dict",
                checkpoint[
                    "model_state_dict"
                ]
            )
        )

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        if "optimizer_state_dict" in checkpoint:

            try:

                self.optimizer.load_state_dict(
                    checkpoint[
                        "optimizer_state_dict"
                    ]
                )

            except Exception:

                # Optimizer 구조가 달라도
                # 모델 자체는 로드할 수 있도록 처리
                pass

        # ----------------------------------------------------
        # Epsilon
        # ----------------------------------------------------

        self.epsilon = checkpoint.get(
            "epsilon",
            self.epsilon_min
        )

        print(
            f"\n[DQN 에이전트] "
            f"학습 모델 로드 완료: {filepath} "
            f"(Epsilon: {self.epsilon:.4f})"
        )

        return True