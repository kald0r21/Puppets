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

        # Early stopping
        self.best_metric_value = -float('inf')
        self.steps_without_improvement = 0
        self.early_stopping_triggered = False

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
        # Check early stopping
        if self.early_stopping_triggered:
            return True

        # Check max iterations
        if hasattr(self, 'max_generations'):
            return self.current_generation >= self.max_generations
        if hasattr(self, 'max_episodes'):
            return self.current_episode >= self.max_episodes
        return False

    def check_early_stopping(self, current_metric: float, method_config: dict) -> bool:
        """
        Check if early stopping criteria is met.

        Args:
            current_metric: Current metric value to track (higher is better)
            method_config: Method-specific config (ga, cnn, or dqn)

        Returns:
            bool: True if early stopping triggered
        """
        if not method_config.get('early_stopping_enabled', False):
            return False

        patience = method_config.get('early_stopping_patience', 20)

        # Check if we have improvement
        if current_metric > self.best_metric_value:
            self.best_metric_value = current_metric
            self.steps_without_improvement = 0
            return False
        else:
            self.steps_without_improvement += 1

        # Check if patience exceeded
        if self.steps_without_improvement >= patience:
            self.early_stopping_triggered = True
            return True

        return False
