"""
NumPy-based neural network brain for Genetic Algorithm.
"""
import numpy as np
from .base import BrainBase


class NumpyBrain(BrainBase):
    """
    Simple feedforward neural network implemented with NumPy.
    Used for Genetic Algorithm evolution.
    """

    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initialize the neural network.

        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output actions
        """
        self.layers_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []

        # Initialize weights using He initialization
        for i in range(len(self.layers_sizes) - 1):
            input_dim = self.layers_sizes[i]
            output_dim = self.layers_sizes[i + 1]
            w = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
            b = np.zeros(output_dim)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, inputs):
        """
        Forward pass through the network.

        Args:
            inputs: Input array

        Returns:
            np.array: Output activations
        """
        x = np.array(inputs, dtype=float)

        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = np.maximum(0, x)  # ReLU activation

        # Output layer with sigmoid
        output = np.dot(x, self.weights[-1]) + self.biases[-1]
        output_activated = 1 / (1 + np.exp(-output))  # Sigmoid

        return output_activated

    def get_action(self, inputs):
        """
        Get action by selecting max output.

        Args:
            inputs: State input array

        Returns:
            int: Action index
        """
        output_values = self.forward(inputs)
        return np.argmax(output_values)

    def save(self, path):
        """
        Save brain to .npz file.

        Args:
            path: File path (should end with .npz)
        """
        save_dict = {}
        for i, w in enumerate(self.weights):
            save_dict[f'W{i}'] = w
        for i, b in enumerate(self.biases):
            save_dict[f'b{i}'] = b
        np.savez(path, **save_dict)

    def load(self, path):
        """
        Load brain from .npz file.

        Args:
            path: File path to load from
        """
        data = np.load(path)
        self.weights = []
        self.biases = []

        # Count layers
        num_layers = 0
        while f'W{num_layers}' in data:
            num_layers += 1

        # Load weights and biases
        for i in range(num_layers):
            self.weights.append(data[f'W{i}'])
            self.biases.append(data[f'b{i}'])

        # Reconstruct layer sizes
        self.layers_sizes = [self.weights[0].shape[0]]
        for w in self.weights:
            self.layers_sizes.append(w.shape[1])

    def clone(self):
        """
        Create a deep copy of the brain.

        Returns:
            NumpyBrain: Cloned brain
        """
        clone = NumpyBrain(1, [1], 1)  # Dummy init
        clone.layers_sizes = self.layers_sizes.copy()
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = [b.copy() for b in self.biases]
        return clone

    @staticmethod
    def crossover(parent1, parent2):
        """
        Perform genetic crossover between two brains.

        Args:
            parent1: First parent brain
            parent2: Second parent brain

        Returns:
            NumpyBrain: Child brain
        """
        child = NumpyBrain(1, [1], 1)  # Dummy init
        child.layers_sizes = parent1.layers_sizes.copy()
        child.weights = []
        child.biases = []

        for i in range(len(parent1.weights)):
            # Crossover weights
            w_child = parent1.weights[i].copy()
            mask = np.random.rand(*parent1.weights[i].shape) > 0.5
            w_child[mask] = parent2.weights[i][mask]
            child.weights.append(w_child)

            # Crossover biases
            b_child = parent1.biases[i].copy()
            mask = np.random.rand(*parent1.biases[i].shape) > 0.5
            b_child[mask] = parent2.biases[i][mask]
            child.biases.append(b_child)

        return child

    def mutate(self, mutation_rate, mutation_strength):
        """
        Apply random mutations to the brain.

        Args:
            mutation_rate: Probability of mutating each weight
            mutation_strength: Standard deviation of mutation noise
        """
        for i in range(len(self.weights)):
            # Mutate weights
            mask_w = np.random.rand(*self.weights[i].shape) < mutation_rate
            noise_w = np.random.randn(*self.weights[i].shape) * mutation_strength
            self.weights[i] += mask_w * noise_w

            # Mutate biases
            mask_b = np.random.rand(*self.biases[i].shape) < mutation_rate
            noise_b = np.random.randn(*self.biases[i].shape) * mutation_strength
            self.biases[i] += mask_b * noise_b
