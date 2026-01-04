"""
Entity classes for the simulation.
Defines agents, predators, food, and walls.
"""
import random
from abc import ABC, abstractmethod


class Entity(ABC):
    """Base class for all entities in the world."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Agent(Entity):
    """
    Agent entity that can learn and survive in the world.
    """

    def __init__(self, x, y, brain=None, config=None):
        super().__init__(x, y)
        self.brain = brain
        self.config = config or {}

        # Energy & survival
        self.energy = self.config.get('start_energy', 100)
        self.max_energy = self.config.get('start_energy', 100)
        self.is_alive = True

        # Stats
        self.fitness = 0.0
        self.food_eaten_count = 0
        self.predators_killed = 0

        # For reward shaping (GA/CNN)
        self.last_dist_to_food_sq = float('inf')

        # For CNN vision upgrades
        self.current_perception_radius = self.config.get('start_perception_radius', 1)

    def reset(self, x, y):
        """Reset agent to starting state."""
        self.x = x
        self.y = y
        self.energy = self.config.get('start_energy', 100)
        self.max_energy = self.config.get('start_energy', 100)
        self.is_alive = True
        self.fitness = 0.0
        self.food_eaten_count = 0
        self.predators_killed = 0
        self.last_dist_to_food_sq = float('inf')
        self.current_perception_radius = self.config.get('start_perception_radius', 1)

    def die(self):
        """Kill the agent."""
        self.is_alive = False
        self.energy = 0


class Predator(Entity):
    """
    Predator entity that hunts agents.
    """

    def __init__(self, x, y, predator_id=0, brain=None, config=None):
        super().__init__(x, y)
        self.id = predator_id
        self.brain = brain
        self.config = config or {}

        self.is_alive = True
        self.strength = self.config.get('predator_base_strength', 5)

        # For DQN
        self.last_state = None
        self.last_action = None

    def reset(self, x, y):
        """Reset predator to starting state."""
        self.x = x
        self.y = y
        self.is_alive = True
        self.strength = self.config.get('predator_base_strength', 5)
        self.last_state = None
        self.last_action = None

    def die(self):
        """Kill the predator."""
        self.is_alive = False


class Food:
    """Food item that agents can eat."""

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Wall:
    """Wall obstacle that blocks movement."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
