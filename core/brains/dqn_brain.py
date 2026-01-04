"""
Deep Q-Network (DQN) brain with experience replay.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from collections import namedtuple, deque
from .base import BrainBase


# Experience tuple for replay memory
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class DQNNet(nn.Module):
    """
    Q-Network implemented as a Multi-Layer Perceptron.
    """

    def __init__(self, input_size, hidden_sizes, output_size):
        super(DQNNet, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ReplayMemory:
    """
    Experience replay buffer for DQN training.
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save an experience."""
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        """Sample a random batch."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNBrain(BrainBase):
    """
    Deep Q-Learning brain with experience replay and target network.
    """

    def __init__(self, input_size, output_size, hidden_layers, lr, mem_size,
                 batch_size, gamma, eps_start, eps_end, eps_decay, device='cpu'):
        """
        Initialize DQN brain.

        Args:
            input_size: State input size
            output_size: Number of actions
            hidden_layers: List of hidden layer sizes
            lr: Learning rate
            mem_size: Replay memory size
            batch_size: Training batch size
            gamma: Discount factor
            eps_start: Starting epsilon for exploration
            eps_end: Final epsilon
            eps_decay: Epsilon decay rate
            device: 'cuda' or 'cpu'
        """
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = device

        # Policy and target networks
        self.policy_net = DQNNet(input_size, hidden_layers, output_size).to(device)
        self.target_net = DQNNet(input_size, hidden_layers, output_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(mem_size)
        self.steps_done = 0

    def select_action(self, state, force_greedy=False):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state array
            force_greedy: If True, always select best action

        Returns:
            torch.Tensor: Action tensor [[action_idx]]
        """
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if force_greedy or random.random() > eps_threshold:
            # Exploit: choose best action
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            # Explore: random action
            return torch.tensor([[random.randrange(self.output_size)]], device=self.device, dtype=torch.long)

    def get_action(self, state, force_greedy=True):
        """
        Get action for given state (for BrainBase interface).

        Args:
            state: State array
            force_greedy: Always use greedy policy (no exploration)

        Returns:
            int: Action index
        """
        return self.select_action(state, force_greedy=force_greedy).item()

    def optimize_model(self):
        """
        Perform one step of optimization on a batch from memory.

        Returns:
            float: Loss value (0.0 if not enough samples)
        """
        if len(self.memory) < self.batch_size:
            return 0.0

        # Sample batch
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # Convert to tensors
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(self.device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(self.device)

        # Compute Q(s, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s') = max_a Q(s', a)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()

        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma * (1.0 - done_batch)) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        """
        Save DQN brain to file.

        Args:
            path: File path (should end with .pth)
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)

    def load(self, path):
        """
        Load DQN brain from file.

        Args:
            path: File path to load from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint.get('steps_done', 0)

    def clone(self):
        """
        Create a copy of the DQN brain.

        Returns:
            DQNBrain: Cloned brain
        """
        # Note: Cloning DQN is complex due to optimizer state
        # For now, just copy the networks
        clone = DQNBrain(
            self.input_size, self.output_size,
            [], self.optimizer.param_groups[0]['lr'],
            len(self.memory.memory), self.batch_size,
            self.gamma, self.eps_start, self.eps_end,
            self.eps_decay, self.device
        )
        clone.policy_net.load_state_dict(self.policy_net.state_dict())
        clone.target_net.load_state_dict(self.target_net.state_dict())
        return clone
