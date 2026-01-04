"""
Genetic Algorithm Trainer.
"""
import random
import numpy as np
from typing import Dict, Any
from .base import TrainerBase
from core.simulation.world import World
from core.simulation.entities import Agent
from core.brains.numpy_brain import NumpyBrain
from core.simulation.utils import get_toroidal_distance


class GATrainer(TrainerBase):
    """
    Trainer using Genetic Algorithm evolution.
    """

    def __init__(self, config):
        super().__init__(config)

        self.grid_width = config['simulation']['grid_width']
        self.grid_height = config['simulation']['grid_height']
        self.population_size = config['ga']['population_size']
        self.max_generations = config['ga']['num_generations']
        self.max_turns = config['ga']['max_turns_per_gen']
        self.mutation_rate = config['ga']['mutation_rate']
        self.mutation_strength = config['ga']['mutation_strength']
        self.elitism_count = config['ga']['elitism_count']
        self.hidden_layers = config['ga']['hidden_layers']

        # Initialize population
        self.population = []
        self._initialize_population()

        # Current world (for rendering)
        self.world = None

        # Best brain tracker
        self.best_brain = None
        self.best_fitness = -float('inf')

        # Metrics
        self.fitness_history = []
        self.avg_fitness_history = []

    def _initialize_population(self):
        """Create initial random population."""
        self.population = []
        input_size = 11  # Standard for GA
        output_size = 5

        for _ in range(self.population_size):
            brain = NumpyBrain(input_size, self.hidden_layers, output_size)
            agent = Agent(
                random.randint(0, self.grid_width - 1),
                random.randint(0, self.grid_height - 1),
                brain=brain,
                config=self.config['simulation']
            )
            self.population.append(agent)

    def get_agent_state(self, agent):
        """
        Get state for an agent (sensor input).

        Args:
            agent: Agent to sense for

        Returns:
            tuple: (inputs, min_dist_food_sq)
        """
        inputs = [agent.energy / 1000.0]
        radius = self.config['simulation'].get('smart_perception_radius', 10)
        norm_dist = radius * 2

        vec_food = (norm_dist, norm_dist)
        vec_pred = (norm_dist, norm_dist)
        vec_ally = (norm_dist, norm_dist)
        vec_wall = (norm_dist, norm_dist)

        min_dist_food_sq = float('inf')
        min_dist_pred_sq = float('inf')
        min_dist_ally_sq = float('inf')
        min_dist_wall_sq = float('inf')

        count_pred = 0
        count_ally = 0

        # Find nearest food
        for (fx, fy) in self.world.food_positions:
            dx, dy = get_toroidal_distance(agent.x, agent.y, fx, fy, self.world.width, self.world.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_food_sq and dist_sq <= radius * radius:
                min_dist_food_sq = dist_sq
                vec_food = (dx / radius, dy / radius)

        # Find nearest predator
        for p in self.world.get_alive_predators():
            dx, dy = get_toroidal_distance(agent.x, agent.y, p.x, p.y, self.world.width, self.world.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius * radius:
                count_pred += 1
                if dist_sq < min_dist_pred_sq:
                    min_dist_pred_sq = dist_sq
                    vec_pred = (dx / radius, dy / radius)

        # Find nearest ally
        for a in self.world.get_alive_agents():
            if a == agent:
                continue
            dx, dy = get_toroidal_distance(agent.x, agent.y, a.x, a.y, self.world.width, self.world.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius * radius:
                count_ally += 1
                if dist_sq < min_dist_ally_sq:
                    min_dist_ally_sq = dist_sq
                    vec_ally = (dx / radius, dy / radius)

        # Find nearest wall
        for (wx, wy) in self.world.wall_positions:
            dx, dy = get_toroidal_distance(agent.x, agent.y, wx, wy, self.world.width, self.world.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_wall_sq and dist_sq <= radius * radius:
                min_dist_wall_sq = dist_sq
                vec_wall = (dx / radius, dy / radius)

        inputs.extend(vec_food)
        inputs.extend(vec_pred)
        inputs.extend(vec_ally)
        inputs.append(count_pred / 5.0)
        inputs.append(count_ally / 5.0)
        inputs.extend(vec_wall)

        return inputs, min_dist_food_sq

    def update_agent(self, agent):
        """Update one agent for one turn."""
        if not agent.is_alive:
            return

        current_inputs, dist_now_sq = self.get_agent_state(agent)
        action = agent.brain.get_action(current_inputs)

        # Move
        move_successful = self.world.move_entity(agent, action)

        # Handle food
        self.world.handle_food_consumption(agent)

        # Energy costs
        if action == 4:  # Idle
            agent.energy -= self.config['simulation'].get('idle_cost', 3)
        else:  # Move
            agent.energy -= self.config['simulation'].get('move_cost', 1)
            if not move_successful:
                agent.energy -= self.config['simulation'].get('wall_hit_penalty', 3)

        # Death check
        if agent.energy <= 0:
            agent.die()
        else:
            # Fitness tracking with reward shaping
            fitness_bonus = 1.0
            reward_shaping_factor = self.config['simulation'].get('reward_shaping_factor', 0.1)

            if agent.last_dist_to_food_sq != float('inf'):
                _, dist_after_move_sq = self.get_agent_state(agent)
                if dist_after_move_sq != float('inf'):
                    dist_diff = agent.last_dist_to_food_sq - dist_after_move_sq
                    fitness_bonus += dist_diff * reward_shaping_factor

            agent.last_dist_to_food_sq = dist_now_sq
            agent.fitness += fitness_bonus

    def update_predators(self):
        """Update all predators (simple chase AI)."""
        vision = self.config['simulation'].get('predator_vision', 10)

        for predator in self.world.get_alive_predators():
            target_agent = None
            min_dist_sq = float('inf')

            # Find nearest agent
            for agent in self.world.get_alive_agents():
                dx, dy = get_toroidal_distance(predator.x, predator.y, agent.x, agent.y,
                                                self.world.width, self.world.height)
                dist_sq = dx * dx + dy * dy
                if dist_sq < min_dist_sq and dist_sq <= vision ** 2:
                    min_dist_sq = dist_sq
                    target_agent = agent

            # Move towards target
            if target_agent:
                dx, dy = get_toroidal_distance(predator.x, predator.y, target_agent.x, target_agent.y,
                                                self.world.width, self.world.height)
                if abs(dx) > abs(dy):
                    action = 3 if dx > 0 else 2  # Right or Left
                else:
                    action = 1 if dy > 0 else 0  # Down or Up
            else:
                action = random.randint(0, 4)  # Random

            self.world.move_entity(predator, action)

        # Handle collisions
        self.world.handle_collisions()

    def train_step(self) -> Dict[str, Any]:
        """
        Execute one generation of evolution.

        Returns:
            dict: Metrics from this generation
        """
        self.current_generation += 1

        # Create world with current population
        self.world = World(
            self.grid_width,
            self.grid_height,
            self.config['simulation'],
            agents=self.population,
            single_agent_mode=False
        )

        # Run simulation
        for turn in range(self.max_turns):
            for agent in self.world.get_alive_agents():
                self.update_agent(agent)

            self.update_predators()
            self.world.respawn_predators()

            if not self.world.get_alive_agents():
                break

        # Evaluate population
        evaluated_population = sorted(
            self.world.agents,
            key=lambda a: (a.food_eaten_count * 1000) + a.fitness,
            reverse=True
        )

        best_agent = evaluated_population[0]
        best_fitness = (best_agent.food_eaten_count * 1000) + best_agent.fitness
        avg_fitness = sum((a.food_eaten_count * 1000) + a.fitness for a in evaluated_population) / len(
            evaluated_population)

        # Track best
        if best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_brain = best_agent.brain.clone()

        self.fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

        # Evolve population
        new_population = []

        # Elitism: preserve best agents
        for i in range(self.elitism_count):
            elite_brain = evaluated_population[i].brain.clone()
            agent = Agent(
                random.randint(0, self.grid_width - 1),
                random.randint(0, self.grid_height - 1),
                brain=elite_brain,
                config=self.config['simulation']
            )
            new_population.append(agent)

        # Crossover and mutation
        while len(new_population) < self.population_size:
            parent1 = random.choice(evaluated_population[:50])
            parent2 = random.choice(evaluated_population[:50])
            child_brain = NumpyBrain.crossover(parent1.brain, parent2.brain)
            child_brain.mutate(self.mutation_rate, self.mutation_strength)
            agent = Agent(
                random.randint(0, self.grid_width - 1),
                random.randint(0, self.grid_height - 1),
                brain=child_brain,
                config=self.config['simulation']
            )
            new_population.append(agent)

        self.population = new_population

        return {
            'generation': self.current_generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'best_food_eaten': best_agent.food_eaten_count,
            'alive_agents': len(self.world.get_alive_agents())
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        return {
            'generation': self.current_generation,
            'best_fitness': self.fitness_history[-1] if self.fitness_history else 0,
            'avg_fitness': self.avg_fitness_history[-1] if self.avg_fitness_history else 0
        }

    def save_checkpoint(self, path):
        """Save best brain."""
        if self.best_brain:
            self.best_brain.save(f"{path}/best_brain_gen_{self.current_generation}.npz")

    def load_checkpoint(self, path):
        """Load checkpoint (not implemented for GA)."""
        pass

    def reset(self):
        """Reset training."""
        self.current_generation = 0
        self._initialize_population()
        self.fitness_history = []
        self.avg_fitness_history = []
        self.best_brain = None
        self.best_fitness = -float('inf')

    def get_best_brain(self):
        """Get best brain."""
        return self.best_brain

    def get_frame(self):
        """Get current frame (returns None, rendering handled externally)."""
        return None
