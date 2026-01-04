"""
Abstract base class for all trainers.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class TrainerBase(ABC):
    """
    Base class for all training algorithms.
    Defines the interface that all trainers must implement.
    """

    def __init__(self, config):
        """
        Initialize trainer with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.current_generation = 0
        self.current_episode = 0
        self.is_training = False
        self.metrics = {}

    @abstractmethod
    def train_step(self) -> Dict[str, Any]:
        """
        Execute one training step (generation for GA/CNN, episode for DQN).

        Returns:
            dict: Metrics from this step (fitness, reward, etc.)
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current training metrics.

        Returns:
            dict: Current metrics (best fitness, average, etc.)
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path):
        """
        Save training checkpoint.

        Args:
            path: Directory or file path to save to
        """
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        """
        Load training checkpoint.

        Args:
            path: Directory or file path to load from
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset trainer to initial state.
        """
        pass

    @abstractmethod
    def get_best_brain(self):
        """
        Get the best brain from training.

        Returns:
            BrainBase: Best performing brain
        """
        pass

    @abstractmethod
    def get_frame(self):
        """
        Get current simulation frame for rendering.

        Returns:
            np.array: RGB frame array or None
        """
        pass

    def is_finished(self) -> bool:
        """
        Check if training is complete.

        Returns:
            bool: True if training should stop
        """
        if hasattr(self, 'max_generations'):
            return self.current_generation >= self.max_generations
        if hasattr(self, 'max_episodes'):
            return self.current_episode >= self.max_episodes
        return False
