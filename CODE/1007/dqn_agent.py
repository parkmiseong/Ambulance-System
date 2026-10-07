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

    def __init__(
        self,
        state_dim,
        action_dim
    ):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(
                state_dim,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                action_dim
            ),
        )

    def forward(self, x):

        return self.fc(x)


# ============================================================
# DQN Ambulance Agent
# ============================================================

class DQNAmbulanceAgent:

    def __init__(
        self,
        state_dim,
        action_dim
    ):

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

        # 할인율
        self.gamma = 0.95

        # 초기 탐험률
        self.epsilon = 1.0

        # 최소 탐험률
        self.epsilon_min = 0.05

        # 탐험률 감소
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

        # 처음에는 두 네트워크를 동일하게 설정
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
    # Action Mask 생성
    # ========================================================

    def create_action_mask(
        self,
        valid_actions
    ):

        """
        선택 가능한 action만 True가 되는
        Boolean mask를 생성합니다.

        예:

        valid_actions = [1, 3, 5]

        →

        [False, True, False, True, False, True, ...]
        """

        mask = torch.zeros(
            self.action_dim,
            dtype=torch.bool,
            device=self.device
        )

        for action in valid_actions:

            action = int(action)

            if 0 <= action < self.action_dim:

                mask[action] = True

        return mask

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

        valid_actions가 주어지면
        선택 가능한 병원 중에서만 행동을 선택합니다.
        """

        # ----------------------------------------------------
        # valid_actions가 없으면 전체 action 사용
        # ----------------------------------------------------

        if (
            valid_actions is None
            or len(valid_actions) == 0
        ):

            valid_actions = list(
                range(self.action_dim)
            )

        # ----------------------------------------------------
        # 잘못된 action 제거
        # ----------------------------------------------------

        valid_actions = [

            int(action)

            for action in valid_actions

            if 0 <= int(action) < self.action_dim
        ]

        # ----------------------------------------------------
        # 모든 action이 제거된 경우
        # ----------------------------------------------------

        if len(valid_actions) == 0:

            valid_actions = list(
                range(self.action_dim)
            )

        # ====================================================
        # Exploration
        # ====================================================

        if np.random.rand() < self.epsilon:

            return int(
                random.choice(
                    valid_actions
                )
            )

        # ====================================================
        # Exploitation
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

        action_mask = self.create_action_mask(
            valid_actions
        )

        masked_q = q_values.clone()

        masked_q[
            ~action_mask
        ] = -1e9

        selected_action = torch.argmax(
            masked_q
        ).item()

        return int(
            selected_action
        )

    # ========================================================
    # Replay Memory 저장
    # ========================================================

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_valid_actions=None
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

        next_valid_actions는
        다음 상태에서 실제로 선택 가능한 action입니다.
        """

        if next_valid_actions is None:

            next_valid_actions = list(
                range(self.action_dim)
            )

        else:

            next_valid_actions = [

                int(action)

                for action in next_valid_actions

                if 0 <= int(action) < self.action_dim
            ]

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

                next_valid_actions,
            )
        )

    # ========================================================
    # Replay Memory 크기
    # ========================================================

    def get_memory_size(self):

        return len(
            self.memory
        )

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

        Double DQN + Action Mask를 사용합니다.
        """

        # ----------------------------------------------------
        # 학습 가능한 데이터가 부족한 경우
        # ----------------------------------------------------

        if len(self.memory) < batch_size:

            return None

        # ----------------------------------------------------
        # Mini Batch
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
            dones,
            next_valid_actions
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
        # Current Q
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

        with torch.no_grad():

            # ------------------------------------------------
            # Online Network Q-value
            # ------------------------------------------------

            next_online_q = self.model(
                next_states
            )

            # ------------------------------------------------
            # Invalid Action Mask
            # ------------------------------------------------

            masked_next_q = next_online_q.clone()

            for i in range(batch_size):

                valid_actions = (
                    next_valid_actions[i]
                )

                if (
                    valid_actions is None
                    or len(valid_actions) == 0
                ):

                    valid_actions = list(
                        range(self.action_dim)
                    )

                action_mask = self.create_action_mask(
                    valid_actions
                )

                masked_next_q[i][
                    ~action_mask
                ] = -1e9

            # ------------------------------------------------
            # Online Network가 다음 행동 선택
            # ------------------------------------------------

            next_actions = torch.argmax(
                masked_next_q,
                dim=1,
                keepdim=True
            )

            # ------------------------------------------------
            # Target Network가 평가
            # ------------------------------------------------

            next_target_q = self.target_model(
                next_states
            )

            next_q = next_target_q.gather(
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

        if self.epsilon > self.epsilon_min:

            self.epsilon = max(

                self.epsilon_min,

                self.epsilon
                * self.epsilon_decay
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
        학습된 DQN 모델과 학습 상태를 저장합니다.
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

                "gamma":
                    self.gamma,

                "learning_rate":
                    self.learning_rate,

                "epsilon_min":
                    self.epsilon_min,

                "epsilon_decay":
                    self.epsilon_decay,

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
        """

        # ----------------------------------------------------
        # 파일 존재 여부
        # ----------------------------------------------------

        if not os.path.exists(
            filepath
        ):

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
        # State Dimension 확인
        # ----------------------------------------------------

        saved_state_dim = checkpoint.get(
            "state_dim",
            self.state_dim
        )

        saved_action_dim = checkpoint.get(
            "action_dim",
            self.action_dim
        )

        if saved_state_dim != self.state_dim:

            raise ValueError(

                "저장된 모델의 state_dim과 "
                "현재 state_dim이 다릅니다. "

                f"(저장: {saved_state_dim}, "
                f"현재: {self.state_dim})"
            )

        if saved_action_dim != self.action_dim:

            raise ValueError(

                "저장된 모델의 action_dim과 "
                "현재 action_dim이 다릅니다. "

                f"(저장: {saved_action_dim}, "
                f"현재: {self.action_dim})"
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

        if (
            "optimizer_state_dict"
            in checkpoint
        ):

            try:

                self.optimizer.load_state_dict(

                    checkpoint[
                        "optimizer_state_dict"
                    ]
                )

            except Exception:

                print(
                    "[DQN] "
                    "Optimizer 상태는 "
                    "불러오지 않고 모델만 복원합니다."
                )

        # ----------------------------------------------------
        # Hyperparameter 복원
        # ----------------------------------------------------

        self.gamma = checkpoint.get(
            "gamma",
            self.gamma
        )

        self.epsilon_min = checkpoint.get(
            "epsilon_min",
            self.epsilon_min
        )

        self.epsilon_decay = checkpoint.get(
            "epsilon_decay",
            self.epsilon_decay
        )

        self.learning_rate = checkpoint.get(
            "learning_rate",
            self.learning_rate
        )

        # ----------------------------------------------------
        # Epsilon 복원
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