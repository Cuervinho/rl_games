"""Deep Q-Network (DQN) in PyTorch.

Contents:
  - QNetwork     : fully-connected network mapping state -> Q(s, a)
  - ReplayBuffer : fixed-size FIFO buffer of (s, a, r, s', done) transitions
  - DQNAgent     : epsilon-greedy policy, training loop, target-network sync,
                   and save/load

QNetwork and parts of DQNAgent are exercises; see CHEATSHEET.md.
"""
import random
from collections import deque
from pathlib import Path
from typing import Self

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl_games import envs
from rl_games.agents.base import BaseAgent


# ── Neural network ────────────────────────────────────────────────────


class QNetwork(nn.Module):
    """Maps a state to one Q-value per action.

    Two hidden layers is the standard baseline for LunarLander-sized
    problems: enough capacity to matter, small enough to train on CPU.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        # EXERCISE: build the network.
        #   - input is `state_dim` wide, output is `action_dim` wide, because
        #     the net returns one Q-value per action in a single forward pass
        #   - `hidden` units in between, with a nonlinearity after each
        #     hidden layer (without one you have a linear model)
        #   - no activation on the output: Q-values are unbounded
        raise NotImplementedError("QNetwork.__init__ -- see CHEATSHEET.md")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Q-values for a batch of states, shape (batch, action_dim)."""
        # EXERCISE: run `x` through the layers defined in __init__.
        raise NotImplementedError("QNetwork.forward -- see CHEATSHEET.md")


# ── Replay buffer ────────────────────────────────────────────────────


class ReplayBuffer:
    """Fixed-size FIFO buffer that stores transitions for experience replay."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque[tuple] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ── Agent ─────────────────────────────────────────────────────────────


class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent implemented from scratch.

    Hyperparameters are intentionally exposed as constructor args so you
    can experiment with them directly.
    """

    label = "DQN"

    def __init__(
        self,
        env_id: str,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 10,
        hidden: int = 128,
    ) -> None:
        super().__init__(
            env_id,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
        )
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        env = envs.make(env_id)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = int(env.action_space.n)  # type: ignore[attr-defined]
        env.close()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_capacity)

    # ── policy ────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, *, deterministic: bool = False) -> int:
        """Epsilon-greedy action for `state`.

        With probability self.epsilon pick uniformly at random (unless
        `deterministic`), otherwise pick argmax of the online network's
        Q-values.
        """
        # EXERCISE: implement epsilon-greedy over self.q_net.
        #   - wrap the forward pass in torch.no_grad(): this is inference, so
        #     there is no need to build a graph
        #   - the net expects a batch dimension, a single state does not have one
        raise NotImplementedError("DQNAgent.select_action -- see CHEATSHEET.md")

    # ── learning step ─────────────────────────────────────────────────

    def _learn(self) -> float:
        """Sample a mini-batch from the buffer and perform one gradient step.

        Returns the batch loss value, or 0.0 while the buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        # EXERCISE: one DQN gradient step.
        #   1. sample a batch and turn each column into a tensor on self.device
        #   2. current Q: self.q_net(states), keeping only the actions taken
        #      (torch.gather does this)
        #   3. target Q: r + gamma * max_a' Q_target(s', a') * (1 - done).
        #      Use the TARGET network here, under torch.no_grad(), and note
        #      the (1 - done) factor -- a terminal state has no future reward
        #   4. self.loss_fn(current, target), then zero_grad / backward / step
        #   5. return loss.item()
        raise NotImplementedError("DQNAgent._learn -- see CHEATSHEET.md")

    # ── training loop ─────────────────────────────────────────────────

    def train(self, total_episodes: int = 500, log_interval: int = 10) -> list[float]:
        env = envs.make(self.env_id)
        rewards_history: list[float] = []

        for episode in range(1, total_episodes + 1):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False

            # Environment loop
            while not done:
                # Select action
                action = self.select_action(obs)
                # Take action
                next_obs, reward, terminated, truncated, _ = env.step(action)
                # Update state
                done = terminated or truncated
                # Update buffer
                self.buffer.push(obs, action, float(reward), next_obs, done)
                # Update Q-network
                self._learn()
                # Update state and total reward
                obs = next_obs
                total_reward += reward

            self._decay_epsilon()
            self.training_episodes += 1
            rewards_history.append(total_reward)

            if episode % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())

            if episode % log_interval == 0:
                self._log_episode(
                    episode,
                    total_episodes,
                    rewards_history,
                    log_interval,
                    f"Buffer: {len(self.buffer)}",
                )

        env.close()
        return rewards_history

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_net_state": self.q_net.state_dict(),
            "target_net_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_episodes": self.training_episodes,
            "env_id": self.env_id,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
            "target_update_freq": self.target_update_freq,
        }
        torch.save(data, path)
        print(f"Saved {self.label} agent to {path}")

    @classmethod
    def load(cls, path: Path) -> Self:
        data = torch.load(path, weights_only=False)
        agent = cls(
            data["env_id"],
            lr=data["lr"],
            gamma=data["gamma"],
            epsilon_start=data["epsilon"],
            epsilon_end=data["epsilon_end"],
            epsilon_decay=data["epsilon_decay"],
            batch_size=data["batch_size"],
            target_update_freq=data["target_update_freq"],
        )
        agent.q_net.load_state_dict(data["q_net_state"])
        agent.target_net.load_state_dict(data["target_net_state"])
        agent.optimizer.load_state_dict(data["optimizer_state"])
        agent.training_episodes = data["training_episodes"]
        return agent

    def info(self) -> str:
        params = sum(p.numel() for p in self.q_net.parameters())
        return (
            f"{self.label} agent for {self.env_id}\n"
            f"  Episodes trained  : {self.training_episodes}\n"
            f"  Network params    : {params:,}\n"
            f"  Epsilon           : {self.epsilon:.4f}\n"
            f"  LR / Gamma        : {self.lr} / {self.gamma}\n"
            f"  Batch size        : {self.batch_size}\n"
            f"  Target update     : every {self.target_update_freq} episodes\n"
            f"  Device            : {self.device}"
        )
