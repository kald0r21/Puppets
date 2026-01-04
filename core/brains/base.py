"""
Abstract base class for all brain implementations.
"""
from abc import ABC, abstractmethod


class BrainBase(ABC):
    """
    Base class for all neural network brains.
    Defines the interface that all brains must implement.
    """

    @abstractmethod
    def get_action(self, *args, **kwargs):
        """
        Select an action based on input state.

        Returns:
            int: Action index (0-4 typically)
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Save brain weights to file.

        Args:
            path: File path to save to
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Load brain weights from file.

        Args:
            path: File path to load from
        """
        pass

    @abstractmethod
    def clone(self):
        """
        Create a deep copy of the brain.

        Returns:
            BrainBase: Cloned brain
        """
        pass
