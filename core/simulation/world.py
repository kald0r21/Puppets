"""
World class - manages the simulation environment.
Unified for GA, CNN, and DQN methods.
"""
import random
from typing import List, Tuple, Set
from .entities import Agent, Predator, Food, Wall
from .utils import get_toroidal_distance


class World:
    """
    The world in which agents, predators, food, and walls exist.
    Manages all entities and their interactions.
    """

    def __init__(self, width, height, config, agents=None, single_agent_mode=False):
        """
        Initialize the world.

        Args:
            width: Grid width
            height: Grid height
            config: Configuration dictionary
            agents: List of agents (for GA/CNN) or single agent (for DQN)
            single_agent_mode: True for DQN (single agent), False for GA/CNN (population)
        """
        self.width = width
        self.height = height
        self.config = config
        self.single_agent_mode = single_agent_mode

        # Entities
        if single_agent_mode:
            self.agent = agents  # Single agent for DQN
            self.agents = []
        else:
            self.agents = agents or []  # List of agents for GA/CNN
            self.agent = None

        self.predators: List[Predator] = []
        self.food_positions: Set[Tuple[int, int]] = set()
        self.wall_positions: Set[Tuple[int, int]] = set()

        # Initialize world
        self._initialize_world()

    def _initialize_world(self):
        """Spawn initial food, walls, and predators."""
        self.spawn_walls(self.config.get('num_walls', 0))
        self.spawn_food(self.config.get('num_food', 0))
        self.spawn_predators(self.config.get('predator_count', 0))

    def spawn_food(self, amount):
        """Spawn food items in random positions."""
        for _ in range(amount):
            max_attempts = 100
            for _ in range(max_attempts):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)

                # Avoid walls for cleaner spawning
                if (x, y) not in self.wall_positions:
                    self.food_positions.add((x, y))
                    break

    def spawn_walls(self, amount):
        """Spawn wall obstacles in random positions."""
        for _ in range(amount):
            max_attempts = 100
            for _ in range(max_attempts):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)

                # Don't spawn on agent position (important for DQN single agent)
                if self.single_agent_mode and self.agent:
                    if x == self.agent.x and y == self.agent.y:
                        continue

                self.wall_positions.add((x, y))
                break

    def spawn_predators(self, amount):
        """Spawn predator entities."""
        start_id = len(self.predators)
        for i in range(amount):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            predator = Predator(x, y, predator_id=start_id + i, config=self.config)
            self.predators.append(predator)

    def move_entity(self, entity, action, allow_idle=True):
        """
        Move an entity based on action.

        Args:
            entity: Agent or Predator to move
            action: 0=up, 1=down, 2=left, 3=right, 4=idle
            allow_idle: Whether idle action is allowed

        Returns:
            bool: True if move was successful (not blocked by wall)
        """
        if action == 4 and allow_idle:
            return True  # Idle action

        next_x, next_y = entity.x, entity.y

        if action == 0:  # Up
            next_y = (entity.y - 1) % self.height
        elif action == 1:  # Down
            next_y = (entity.y + 1) % self.height
        elif action == 2:  # Left
            next_x = (entity.x - 1) % self.width
        elif action == 3:  # Right
            next_x = (entity.x + 1) % self.width

        # Check for wall collision
        if (next_x, next_y) in self.wall_positions:
            return False  # Move blocked
        else:
            entity.x = next_x
            entity.y = next_y
            return True  # Move successful

    def get_alive_agents(self) -> List[Agent]:
        """Get all living agents."""
        if self.single_agent_mode:
            return [self.agent] if self.agent and self.agent.is_alive else []
        else:
            return [agent for agent in self.agents if agent.is_alive]

    def get_alive_predators(self) -> List[Predator]:
        """Get all living predators."""
        return [p for p in self.predators if p.is_alive]

    def get_nearby_allies(self, agent_in_fight: Agent) -> int:
        """
        Count allies near an agent (for fight mechanics).

        Args:
            agent_in_fight: Agent being attacked

        Returns:
            int: Number of nearby allies
        """
        allies_count = 0
        ally_radius = self.config.get('predator_ally_radius', 3)

        for agent in self.get_alive_agents():
            if agent == agent_in_fight:
                continue

            dx, dy = get_toroidal_distance(
                agent_in_fight.x, agent_in_fight.y,
                agent.x, agent.y,
                self.width, self.height
            )
            dist_sq = dx * dx + dy * dy

            if dist_sq <= ally_radius ** 2:
                allies_count += 1

        return allies_count

    def handle_fight(self, predator: Predator, agent: Agent):
        """
        Handle combat between predator and agent.

        Args:
            predator: Attacking predator
            agent: Defending agent
        """
        # Calculate strengths
        allies_count = self.get_nearby_allies(agent)
        group_bonus = allies_count * self.config.get('predator_ally_bonus', 2)
        agent_strength = agent.food_eaten_count + group_bonus
        predator_strength = predator.strength

        total_strength = agent_strength + predator_strength
        if total_strength == 0:
            win_chance = 0.5
        else:
            win_chance = agent_strength / total_strength

        # Determine outcome
        if random.random() < win_chance:
            # Agent wins
            kill_license_level = self.config.get('kill_license_level', 5)
            kill_cost = self.config.get('kill_cost_pellets', 1)

            if agent.food_eaten_count >= kill_license_level:
                predator.die()
                agent.food_eaten_count -= kill_cost
                agent.predators_killed += 1
        else:
            # Predator wins
            agent.die()

    def handle_food_consumption(self, agent: Agent):
        """
        Handle agent eating food.

        Args:
            agent: Agent that might eat food
        """
        if (agent.x, agent.y) in self.food_positions:
            # Remove food
            self.food_positions.remove((agent.x, agent.y))

            # Reward agent
            eat_gain = self.config.get('eat_gain', 150)
            max_energy_gain = self.config.get('max_energy_gain_per_food', 10)

            agent.energy += eat_gain
            agent.food_eaten_count += 1
            agent.max_energy += max_energy_gain

            if agent.energy > agent.max_energy:
                agent.energy = agent.max_energy

            # Respawn food
            self.spawn_food(1)

            # Vision upgrade for CNN
            if 'vision_upgrades' in self.config:
                vision_upgrades = self.config['vision_upgrades']
                if agent.food_eaten_count in vision_upgrades:
                    agent.current_perception_radius = vision_upgrades[agent.food_eaten_count]

            return True

        return False

    def handle_collisions(self):
        """Check and handle all predator-agent collisions."""
        if self.single_agent_mode:
            if not self.agent or not self.agent.is_alive:
                return

            for predator in self.get_alive_predators():
                if predator.x == self.agent.x and predator.y == self.agent.y:
                    self.handle_fight(predator, self.agent)
                    if not self.agent.is_alive:
                        break
        else:
            for agent in self.get_alive_agents():
                for predator in self.get_alive_predators():
                    if predator.x == agent.x and predator.y == agent.y:
                        self.handle_fight(predator, agent)
                        if not predator.is_alive:
                            break

    def respawn_predators(self):
        """Respawn predators if enabled and needed."""
        if self.config.get('predator_respawn', False):
            alive_predators = self.get_alive_predators()
            predator_count = self.config.get('predator_count', 5)
            missing = predator_count - len(alive_predators)

            if missing > 0:
                self.spawn_predators(missing)

        # Clean up dead predators
        self.predators = self.get_alive_predators()

    def step(self):
        """
        Execute one simulation step (for GA/CNN with built-in agent logic).
        This is for backward compatibility.
        """
        # Note: This method is kept for GA/CNN compatibility
        # DQN uses external step logic
        pass

    def get_grid_state(self):
        """
        Get the current grid state as a dictionary.
        Useful for rendering and debugging.

        Returns:
            dict: Contains positions of all entities
        """
        return {
            'agents': [(a.x, a.y) for a in self.get_alive_agents()],
            'predators': [(p.x, p.y) for p in self.get_alive_predators()],
            'food': list(self.food_positions),
            'walls': list(self.wall_positions),
            'width': self.width,
            'height': self.height
        }
