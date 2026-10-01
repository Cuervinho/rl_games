"""Common base class for the agents.

Contents:
  - BaseAgent : holds the shared hyperparameters (lr, gamma, the epsilon
                schedule, episode count), implements _decay_epsilon(),
                predict() and the shared training log line, and declares
                select_action/train/save/load/info as abstract.

Each algorithm implements its own train() loop.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import numpy as np


class BaseAgent(ABC):
    """Common base for the tabular and neural agents."""

    #: Human-readable name, used in save/info messages.
    label: str = "Agent"

    def __init__(
        self,
        env_id: str,
        *,
        lr: float,
        gamma: float,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.env_id = env_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.training_episodes = 0

    # ── ε-greedy schedule ─────────────────────────────────────────────

    def _decay_epsilon(self) -> None:
        """Apply one step of exponential ε decay, floored at epsilon_end."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ── policy ────────────────────────────────────────────────────────

    def _to_state(self, obs: np.ndarray) -> Any:
        """Convert a raw observation into whatever `select_action` expects.

        Defaults to passing the observation straight through; the tabular
        agent overrides this to discretise.
        """
        return obs

    def predict(
        self, obs: np.ndarray, *, deterministic: bool = True
    ) -> tuple[int, None]:
        """Choose an action for `obs`. Returns (action, None) like SB3."""
        action = self.select_action(self._to_state(obs), deterministic=deterministic)
        return action, None

    @abstractmethod
    def select_action(self, state: Any, *, deterministic: bool = False) -> int:
        """Pick an action for an already-converted state."""

    # ── training and persistence ───────────────────────────────────────
    #
    # These are @abstractmethod so a subclass that forgets one fails loudly at
    # construction ("Can't instantiate abstract class ... without ... 'save'")
    # rather than with an AttributeError at the end of a long training run.
    # @classmethod must stay the OUTER decorator on `load`; reversed, the
    # abstractness is silently lost.

    @abstractmethod
    def train(self, total_episodes: int, log_interval: int) -> list[float]:
        """Run the training loop, returning per-episode total rewards."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the agent to `path`."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self:
        """Restore an agent previously written by `save`."""

    @abstractmethod
    def info(self) -> str:
        """Multi-line human-readable summary."""

    # ── shared reporting ──────────────────────────────────────────────

    def _log_episode(
        self,
        episode: int,
        total_episodes: int,
        rewards_history: list[float],
        log_interval: int,
        extra: str,
    ) -> None:
        """Print the periodic training line shared by both agents."""
        avg = np.mean(rewards_history[-log_interval:])
        print(
            f"Episode {episode}/{total_episodes} | "
            f"Avg Reward: {avg:.2f} | "
            f"Epsilon: {self.epsilon:.4f} | "
            f"{extra}"
        )
