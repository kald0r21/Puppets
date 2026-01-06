"""
CNN Trainer with Genetic Algorithm.
"""
import random
import numpy as np
import torch
from typing import Dict, Any
from .base import TrainerBase
from core.simulation.world import World
from core.simulation.entities import Agent
from core.brains.cnn_brain import CNNBrain


class CNNTrainer(TrainerBase):
    """
    Trainer using CNN brains with genetic evolution.
    """

    def __init__(self, config):
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_width = config['simulation']['grid_width']
        self.grid_height = config['simulation']['grid_height']
        self.population_size = config['cnn']['population_size']
        self.max_generations = config['cnn']['num_generations']
        self.max_turns = config['cnn']['max_turns_per_gen']
        self.mutation_rate = config['cnn']['mutation_rate']
        self.mutation_strength = config['cnn']['mutation_strength']
        self.elitism_count = config['cnn']['elitism_count']

        self.max_perception_radius = config['cnn']['max_perception_radius']
        self.map_size = self.max_perception_radius * 2 + 1

        # Initialize population
        self.population = []
        self._initialize_population()

        # Current world
        self.world = None

        # Best brain tracker
        self.best_brain = None
        self.best_fitness = -float('inf')
        self.best_brain_path = None

        # Metrics
        self.fitness_history = []
        self.avg_fitness_history = []

    def _initialize_population(self):
        """Create initial random population with CNN brains."""
        self.population = []

        for _ in range(self.population_size):
            brain = CNNBrain(self.map_size, num_channels=3, num_actions=5, device=self.device)
            agent = Agent(
                random.randint(0, self.grid_width - 1),
                random.randint(0, self.grid_height - 1),
                brain=brain,
                config=self.config['simulation']
            )
            self.population.append(agent)

    def get_agent_vision(self, agent):
        """
        Get vision state for agent (3-channel vision map).

        Returns:
            tuple: (vision_map, state_data)
        """
        food_map = np.zeros((self.map_size, self.map_size))
        pred_map = np.zeros((self.map_size, self.map_size))
        ally_map = np.zeros((self.map_size, self.map_size))

        alive_agents = self.world.get_alive_agents()

        for dy in range(-self.max_perception_radius, self.max_perception_radius + 1):
            for dx in range(-self.max_perception_radius, self.max_perception_radius + 1):
                # Check if within current vision radius
                is_in_fog = abs(dx) > agent.current_perception_radius or abs(dy) > agent.current_perception_radius

                if not is_in_fog:
                    check_x = (agent.x + dx) % self.world.width
                    check_y = (agent.y + dy) % self.world.height
                    map_x = dx + self.max_perception_radius
                    map_y = dy + self.max_perception_radius

                    # Food
                    if (check_x, check_y) in self.world.food_positions:
                        food_map[map_y, map_x] = 1.0

                    # Predators
                    for p in self.world.get_alive_predators():
                        if p.x == check_x and p.y == check_y:
                            pred_map[map_y, map_x] = 1.0
                            break

                    # Allies
                    for a in alive_agents:
                        if a == agent:
                            continue
                        if a.x == check_x and a.y == check_y:
                            ally_map[map_y, map_x] = 1.0
                            break

        vision_map = np.stack([food_map, pred_map, ally_map], axis=0)
        state_data = np.array([agent.energy / agent.max_energy])

        return vision_map, state_data

    def update_agent(self, agent):
        """Update one agent for one turn."""
        if not agent.is_alive:
            return

        vision_map, state_data = self.get_agent_vision(agent)
        action = agent.brain.get_action(vision_map, state_data)

        # Move
        self.world.move_entity(agent, action)

        # Handle food
        self.world.handle_food_consumption(agent)

        # Energy costs
        if action == 4:  # Idle
            agent.energy -= self.config['simulation'].get('idle_cost', 3)
        else:  # Move
            agent.energy -= self.config['simulation'].get('move_cost', 2)

        # Death check
        if agent.energy <= 0:
            agent.die()
        else:
            agent.fitness += 1

    def update_predators(self):
        """Simple predator AI."""
        vision = self.config['simulation'].get('predator_vision', 10)

        for predator in self.world.get_alive_predators():
            target = None
            min_dist = float('inf')

            for agent in self.world.get_alive_agents():
                dx = abs(predator.x - agent.x)
                dy = abs(predator.y - agent.y)
                dist_x = min(dx, self.world.width - dx)
                dist_y = min(dy, self.world.height - dy)
                dist = dist_x + dist_y

                if dist < min_dist and dist <= vision:
                    min_dist = dist
                    target = agent

            if target:
                if predator.x != target.x:
                    predator.x += 1 if target.x > predator.x else -1
                if predator.y != target.y:
                    predator.y += 1 if target.y > predator.y else -1
                predator.x %= self.world.width
                predator.y %= self.world.height
            else:
                action = random.randint(0, 4)
                self.world.move_entity(predator, action)

        self.world.handle_collisions()

    def train_step(self) -> Dict[str, Any]:
        """Execute one generation."""
        self.current_generation += 1

        # Create world
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

        # Evaluate
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

        # Check early stopping
        self.check_early_stopping(best_fitness, self.config['cnn'])

        # Evolution
        new_population = []

        # Elitism
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
            child_brain = CNNBrain.crossover(parent1.brain, parent2.brain)
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
            'best_food_eaten': best_agent.food_eaten_count
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        return {
            'generation': self.current_generation,
            'best_fitness': self.fitness_history[-1] if self.fitness_history else 0,
            'avg_fitness': self.avg_fitness_history[-1] if self.avg_fitness_history else 0
        }

    def save_checkpoint(self, path):
        """Save best brain only."""
        import os
        if self.best_brain:
            # Remove old best brain if exists
            if self.best_brain_path and os.path.exists(self.best_brain_path):
                os.remove(self.best_brain_path)

            # Save new best brain with fitness in filename
            filename = f"cnn_best_fitness_{int(self.best_fitness)}.pth"
            self.best_brain_path = os.path.join(path, filename)
            self.best_brain.save(self.best_brain_path)

            # Also save metadata
            import json
            metadata = {
                'method': 'CNN',
                'generation': self.current_generation,
                'best_fitness': self.best_fitness,
                'map_size': self.map_size,
                'num_channels': 3,
                'num_actions': 5,
                'config': self.config['cnn']
            }
            metadata_path = os.path.join(path, filename.replace('.pth', '_metadata.json'))
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def load_checkpoint(self, path):
        """Load checkpoint."""
        pass

    def reset(self):
        """Reset training."""
        self.current_generation = 0
        self._initialize_population()
        self.fitness_history = []
        self.avg_fitness_history = []
        self.best_brain = None
        self.best_fitness = -float('inf')
        self.best_brain_path = None
        self.best_metric_value = -float('inf')
        self.steps_without_improvement = 0
        self.early_stopping_triggered = False

    def get_best_brain(self):
        """Get best brain."""
        return self.best_brain

    def get_frame(self):
        """Get current frame."""
        return None
