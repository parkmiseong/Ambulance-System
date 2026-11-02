"""Replay transitions between hospital decisions on the shared SUMO clock.

Rewards belong to the interval in which outcomes are observed, including
outcomes of older, still-running missions. Completion order never defines the
decision order. Gamma is applied per hospital decision, not per SUMO second.
"""
import numpy as np


class EpisodeTransitions:
    def __init__(self, agent, batch_size=32):
        self.agent = agent
        self.batch_size = batch_size
        self.pending = None
        self.reward = 0.0
        self.total_reward = 0.0
        self.unattributed_reward = 0.0
        self.count = 0
        self.losses = []

    def begin(self, state, action, valid_actions):
        """Commit the preceding interval, then begin the selected action."""
        actions = tuple(valid_actions)
        if action not in actions:
            raise ValueError("The selected action must be valid at this decision.")
        snapshot = np.array(state, dtype=np.float32, copy=True)
        self._emit(snapshot, False, actions)
        self.pending = (snapshot, int(action))
        self.reward = 0.0

    def add_reward(self, reward):
        reward = float(reward)
        self.total_reward += reward
        if self.pending is None:
            # No controllable action precedes these outcomes. Do not invent one.
            self.unattributed_reward += reward
        else:
            self.reward += reward

    def close(self):
        """Flush once, after all completed and censored outcomes are recorded."""
        if self.pending is not None:
            self._emit(np.zeros_like(self.pending[0]), True, ())
            self.pending = None
            self.reward = 0.0

    def _emit(self, next_state, done, next_valid_actions):
        if self.pending is None:
            return
        state, action = self.pending
        self.agent.store_transition(
            state, action, self.reward, next_state, done,
            next_valid_actions=next_valid_actions,
        )
        self.count += 1
        loss = self.agent.train_step(self.batch_size)
        if loss is not None:
            self.losses.append(loss)
