"""
Convolutional Neural Network brain using PyTorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BrainBase


class CNNBrain(nn.Module, BrainBase):
    """
    CNN-based brain for visual perception.
    Uses convolutional layers to process spatial vision data.
    """

    def __init__(self, map_size, num_channels=3, num_actions=5, device='cpu'):
        """
        Initialize CNN brain.

        Args:
            map_size: Size of the vision map (e.g., 7 for 7x7)
            num_channels: Number of input channels (food, predators, allies)
            num_actions: Number of possible actions
            device: 'cuda' or 'cpu'
        """
        super(CNNBrain, self).__init__()
        self.map_size = map_size
        self.num_channels = num_channels
        self.num_actions = num_actions
        self.device = device

        # Convolutional layers (no pooling - preserve spatial precision)
        # Padding=1 maintains size with kernel=3
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Flattened size after convolution
        self.flattened_size = 64 * map_size * map_size

        # Fully connected layers (+1 for energy state)
        self.fc1 = nn.Linear(self.flattened_size + 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, num_actions)

        self.to(device)

    def forward(self, vision_input, state_input):
        """
        Forward pass through the network.

        Args:
            vision_input: Vision tensor [batch, channels, height, width]
            state_input: State tensor [batch, 1] (energy)

        Returns:
            torch.Tensor: Action logits
        """
        # Process vision through conv layers
        x = F.relu(self.conv1(vision_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = torch.flatten(x, 1)

        # Concatenate with state (energy)
        x = torch.cat((x, state_input), dim=1)

        # Decision layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc_out(x)

        return output

    def get_action(self, vision_input, state_input):
        """
        Get action from vision and state inputs.

        Args:
            vision_input: Vision array [channels, height, width]
            state_input: State array [1] (energy)

        Returns:
            int: Action index
        """
        v_tensor = torch.tensor(vision_input, dtype=torch.float32).unsqueeze(0).to(self.device)
        s_tensor = torch.tensor(state_input, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits = self.forward(v_tensor, s_tensor)
            action_probs = F.softmax(action_logits, dim=1)
            return torch.argmax(action_probs).item()

    def save(self, path):
        """
        Save brain to .pth file.

        Args:
            path: File path (should end with .pth)
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load brain from .pth file.

        Args:
            path: File path to load from
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to(self.device)

    def clone(self):
        """
        Create a deep copy of the brain.

        Returns:
            CNNBrain: Cloned brain
        """
        clone = CNNBrain(self.map_size, self.num_channels, self.num_actions, self.device)
        # Explicitly move state_dict to target device to avoid mixed device issues
        state_dict = self.state_dict()
        # Map all tensors to clone's device
        clone.load_state_dict({k: v.to(clone.device) for k, v in state_dict.items()})
        return clone

    @staticmethod
    def crossover(parent1, parent2):
        """
        Perform genetic crossover between two CNN brains.

        Args:
            parent1: First parent brain
            parent2: Second parent brain

        Returns:
            CNNBrain: Child brain
        """
        # Create child on CPU to avoid device mismatch during crossover
        child = CNNBrain(
            parent1.map_size,
            parent1.num_channels,
            parent1.num_actions,
            device='cpu'
        )

        # Move all states to CPU for crossover operations
        p1_state = {k: v.cpu() for k, v in parent1.state_dict().items()}
        p2_state = {k: v.cpu() for k, v in parent2.state_dict().items()}

        # Perform crossover on CPU
        child_state = {}
        for key in p1_state:
            mask = torch.rand_like(p1_state[key]) > 0.5
            # Start with parent1's weights
            child_state[key] = p1_state[key].clone()
            # Apply parent2's weights where mask is False
            child_state[key][~mask] = p2_state[key][~mask]

        # Load crossovered state
        child.load_state_dict(child_state)

        # Move child to parent's device
        target_device = parent1.device
        child.to(target_device)
        # Ensure all parameters are on target device
        for param in child.parameters():
            param.data = param.data.to(target_device)
        for buffer in child.buffers():
            buffer.data = buffer.data.to(target_device)

        return child

    def mutate(self, mutation_rate, mutation_strength):
        """
        Apply random mutations to the brain.

        Args:
            mutation_rate: Probability of mutating each weight
            mutation_strength: Standard deviation of mutation noise
        """
        with torch.no_grad():
            for param in self.parameters():
                mask = (torch.rand_like(param) < mutation_rate).to(param.device)
                noise = (torch.randn_like(param) * mutation_strength).to(param.device)
                param.add_(noise * mask)
