"""
DQN Trainer with Coevolution.
"""
import random
import numpy as np
import torch
import math
from typing import Dict, Any
from .base import TrainerBase
from core.simulation.world import World
from core.simulation.entities import Agent, Predator
from core.brains.dqn_brain import DQNBrain
from core.simulation.utils import get_toroidal_distance


class DQNTrainer(TrainerBase):
    """
    Trainer using Deep Q-Learning with predator-prey coevolution.
    """

    def __init__(self, config):
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_width = config['simulation']['grid_width']
        self.grid_height = config['simulation']['grid_height']
        self.max_episodes = config['dqn']['num_episodes']
        self.max_turns = config['dqn']['max_turns_per_episode']

        # Create DQN brains
        self.agent_brain = DQNBrain(
            input_size=12, output_size=5,
            hidden_layers=config['dqn']['agent_hidden_layers'],
            lr=config['dqn']['agent_learning_rate'],
            mem_size=config['dqn']['agent_memory_size'],
            batch_size=config['dqn']['agent_batch_size'],
            gamma=config['dqn']['gamma'],
            eps_start=config['dqn']['agent_eps_start'],
            eps_end=config['dqn']['agent_eps_end'],
            eps_decay=config['dqn']['agent_eps_decay'],
            device=self.device
        )

        self.predator_brain = DQNBrain(
            input_size=9, output_size=5,
            hidden_layers=config['dqn']['predator_hidden_layers'],
            lr=config['dqn']['predator_learning_rate'],
            mem_size=config['dqn']['predator_memory_size'],
            batch_size=config['dqn']['predator_batch_size'],
            gamma=config['dqn']['gamma'],
            eps_start=config['dqn']['predator_eps_start'],
            eps_end=config['dqn']['predator_eps_end'],
            eps_decay=config['dqn']['predator_eps_decay'],
            device=self.device
        )

        # Current world
        self.world = None
        self.current_agent = None

        # Metrics tracking
        self.reward_history = []
        self.avg_reward_history = []

        # Best model tracking
        self.best_avg_reward = -float('inf')
        self.best_agent_brain_path = None
        self.best_predator_brain_path = None

    def get_agent_state(self, agent):
        """Get state representation for agent."""
        inputs = [agent.energy / 1000.0]
        radius = self.config['simulation'].get('smart_perception_radius', 10)
        norm_dist = radius * 2

        vec_food = (norm_dist, norm_dist)
        vec_pred = (norm_dist, norm_dist)
        vec_ally = (norm_dist, norm_dist)
        vec_wall = (norm_dist, norm_dist)

        min_dist_food_sq = float('inf')
        min_dist_pred_sq = float('inf')
        min_dist_wall_sq = float('inf')

        count_pred = 0
        count_ally = 0

        # Food
        for (fx, fy) in self.world.food_positions:
            dx, dy = get_toroidal_distance(agent.x, agent.y, fx, fy, self.world.width, self.world.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_food_sq and dist_sq <= radius * radius:
                min_dist_food_sq = dist_sq
                vec_food = (dx / radius, dy / radius)

        # Predators
        for p in self.world.get_alive_predators():
            dx, dy = get_toroidal_distance(agent.x, agent.y, p.x, p.y, self.world.width, self.world.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius * radius:
                count_pred += 1
                if dist_sq < min_dist_pred_sq:
                    min_dist_pred_sq = dist_sq
                    vec_pred = (dx / radius, dy / radius)

        # Walls
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
        inputs.append(agent.predators_killed / 10.0)

        return np.array(inputs)

    def get_predator_state(self, predator):
        """Get state representation for predator."""
        agent = self.current_agent
        radius = self.config['simulation'].get('predator_vision', 10)
        norm_dist = radius * 2

        vec_agent = (norm_dist, norm_dist)
        vec_pred = (norm_dist, norm_dist)
        vec_wall = (norm_dist, norm_dist)

        min_dist_wall_sq = float('inf')
        min_dist_pred_sq = float('inf')
        count_pred = 0

        # Agent vector
        if agent.is_alive:
            dx, dy = get_toroidal_distance(predator.x, predator.y, agent.x, agent.y,
                                            self.world.width, self.world.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius * radius:
                vec_agent = (dx / radius, dy / radius)

        # Other predators
        for p in self.world.get_alive_predators():
            if p.id == predator.id:
                continue
            dx, dy = get_toroidal_distance(predator.x, predator.y, p.x, p.y,
                                            self.world.width, self.world.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius * radius:
                count_pred += 1
                if dist_sq < min_dist_pred_sq:
                    min_dist_pred_sq = dist_sq
                    vec_pred = (dx / radius, dy / radius)

        # Walls
        for (wx, wy) in self.world.wall_positions:
            dx, dy = get_toroidal_distance(predator.x, predator.y, wx, wy,
                                            self.world.width, self.world.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_wall_sq and dist_sq <= radius * radius:
                min_dist_wall_sq = dist_sq
                vec_wall = (dx / radius, dy / radius)

        is_agent_vulnerable = 1.0 if agent.food_eaten_count < self.config['simulation'].get('kill_license_level',
                                                                                             5) else 0.0

        inputs = []
        inputs.extend(vec_agent)
        inputs.extend(vec_pred)
        inputs.extend(vec_wall)
        inputs.append(count_pred / 5.0)
        inputs.append(is_agent_vulnerable)
        inputs.append(agent.food_eaten_count / 10.0)

        return np.array(inputs)

    def train_step(self) -> Dict[str, Any]:
        """Execute one episode of training."""
        self.current_episode += 1

        # Create new agent and world
        self.current_agent = Agent(
            random.randint(0, self.grid_width - 1),
            random.randint(0, self.grid_height - 1),
            brain=None,
            config=self.config['simulation']
        )

        self.world = World(
            self.grid_width,
            self.grid_height,
            self.config['simulation'],
            agents=self.current_agent,
            single_agent_mode=True
        )

        agent_state = self.get_agent_state(self.current_agent)
        episode_reward = 0.0

        for turn in range(self.max_turns):
            # Agent action
            agent_action_tensor = self.agent_brain.select_action(agent_state)
            agent_action = agent_action_tensor.item()

            # Predator actions
            predator_actions = {}
            for p in self.world.get_alive_predators():
                p_state = self.get_predator_state(p)
                p_action_tensor = self.predator_brain.select_action(p_state)
                p.last_state = p_state
                p.last_action = p_action_tensor
                predator_actions[p.id] = p_action_tensor.item()

            # Calculate distance to food before move (for reward shaping)
            dist_before = float('inf')
            for (fx, fy) in self.world.food_positions:
                dx, dy = get_toroidal_distance(self.current_agent.x, self.current_agent.y, fx, fy,
                                                self.grid_width, self.grid_height)
                d = math.sqrt(dx * dx + dy * dy)
                if d < dist_before:
                    dist_before = d

            # Execute actions
            reward_agent = 0.0
            rewards_predator = {p.id: 0.0 for p in self.world.get_alive_predators()}

            # Move agent
            move_ok = self.world.move_entity(self.current_agent, agent_action)
            if agent_action == 4:  # Idle
                self.current_agent.energy -= self.config['simulation'].get('idle_cost', 3)
                reward_agent -= self.config['simulation'].get('idle_cost', 3)
            else:
                self.current_agent.energy -= self.config['simulation'].get('move_cost', 1)
                reward_agent -= self.config['simulation'].get('move_cost', 1)
                if not move_ok:
                    reward_agent -= self.config['simulation'].get('wall_hit_penalty', 3)

            # Move predators
            for p in self.world.get_alive_predators():
                p_act = predator_actions.get(p.id)
                if p_act is not None:
                    dist_to_agent_before = np.linalg.norm(p.last_state[0:2])
                    p_move_ok = self.world.move_entity(p, p_act)
                    p_next_state = self.get_predator_state(p)
                    dist_to_agent_after = np.linalg.norm(p_next_state[0:2])

                    if dist_to_agent_after < dist_to_agent_before:
                        rewards_predator[p.id] += 0.5
                    if p_act == 4:
                        rewards_predator[p.id] -= 0.5
                    else:
                        rewards_predator[p.id] -= 0.1
                    if not p_move_ok:
                        rewards_predator[p.id] -= 1.0

            # Reward shaping (guidance toward food)
            dist_after = float('inf')
            for (fx, fy) in self.world.food_positions:
                dx, dy = get_toroidal_distance(self.current_agent.x, self.current_agent.y, fx, fy,
                                                self.grid_width, self.grid_height)
                d = math.sqrt(dx * dx + dy * dy)
                if d < dist_after:
                    dist_after = d

            dist_diff = dist_before - dist_after
            if dist_diff > 0:
                reward_agent += dist_diff * 3.0
            elif dist_diff < 0:
                reward_agent += dist_diff * 3.0 * 1.5

            # Food consumption
            if self.world.handle_food_consumption(self.current_agent):
                reward_agent += 500

            # Handle fights
            agent_was_alive = self.current_agent.is_alive
            for p in self.world.get_alive_predators():
                if p.x == self.current_agent.x and p.y == self.current_agent.y:
                    predator_was_alive = p.is_alive
                    self.world.handle_fight(p, self.current_agent)

                    if not self.current_agent.is_alive:
                        rewards_predator[p.id] += 500
                        break
                    if not p.is_alive and predator_was_alive:
                        reward_agent += 300
                        rewards_predator[p.id] -= 300

            # Energy check
            if self.current_agent.energy <= 0:
                self.current_agent.die()

            done = not self.current_agent.is_alive
            if done and agent_was_alive:
                reward_agent -= 1000

            # Get next state
            agent_next_state = self.get_agent_state(self.current_agent)

            # Store experiences
            for p in self.world.get_alive_predators():
                if p.last_state is not None:
                    p_next_state = self.get_predator_state(p)
                    p_done = not p.is_alive or done
                    self.predator_brain.memory.push(p.last_state, p.last_action,
                                                    rewards_predator[p.id], p_next_state, p_done)

            # Respawn predators
            if self.config['simulation'].get('predator_respawn', False):
                alive_predators = self.world.get_alive_predators()
                missing = self.config['simulation'].get('predator_count', 5) - len(alive_predators)
                if missing > 0:
                    self.world.spawn_predators(missing)
                self.world.predators = self.world.get_alive_predators()

            # Store agent experience
            self.agent_brain.memory.push(agent_state, agent_action_tensor, reward_agent, agent_next_state, done)
            episode_reward += reward_agent
            agent_state = agent_next_state

            # Optimize
            self.agent_brain.optimize_model()
            self.predator_brain.optimize_model()

            if done:
                break

        # Update target networks
        if self.current_episode % 10 == 0:
            self.agent_brain.update_target_network()
            self.predator_brain.update_target_network()

        # Track metrics
        self.reward_history.append(episode_reward)
        if len(self.reward_history) >= 10:
            avg_reward = sum(self.reward_history[-10:]) / 10
            self.avg_reward_history.append(avg_reward)
        else:
            self.avg_reward_history.append(episode_reward)

        # Track best model
        current_avg_reward = self.avg_reward_history[-1]
        if current_avg_reward > self.best_avg_reward:
            self.best_avg_reward = current_avg_reward

        # Check early stopping
        self.check_early_stopping(current_avg_reward, self.config['dqn'])

        return {
            'episode': self.current_episode,
            'reward': episode_reward,
            'avg_reward': self.avg_reward_history[-1],
            'food_eaten': self.current_agent.food_eaten_count,
            'predators_killed': self.current_agent.predators_killed
        }

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        return {
            'episode': self.current_episode,
            'reward': self.reward_history[-1] if self.reward_history else 0,
            'avg_reward': self.avg_reward_history[-1] if self.avg_reward_history else 0
        }

    def save_checkpoint(self, path):
        """Save best agent and predator brains only."""
        import os
        import json

        # Remove old best brains if they exist
        if self.best_agent_brain_path and os.path.exists(self.best_agent_brain_path):
            os.remove(self.best_agent_brain_path)
        if self.best_predator_brain_path and os.path.exists(self.best_predator_brain_path):
            os.remove(self.best_predator_brain_path)

        # Save new best brains with reward in filename
        agent_filename = f"dqn_agent_best_reward_{int(self.best_avg_reward)}.pth"
        predator_filename = f"dqn_predator_best_reward_{int(self.best_avg_reward)}.pth"

        self.best_agent_brain_path = os.path.join(path, agent_filename)
        self.best_predator_brain_path = os.path.join(path, predator_filename)

        self.agent_brain.save(self.best_agent_brain_path)
        self.predator_brain.save(self.best_predator_brain_path)

        # Save metadata
        metadata = {
            'method': 'DQN',
            'episode': self.current_episode,
            'best_avg_reward': self.best_avg_reward,
            'agent_input_size': 12,
            'agent_output_size': 5,
            'agent_hidden_layers': self.config['dqn']['agent_hidden_layers'],
            'predator_input_size': 9,
            'predator_output_size': 5,
            'predator_hidden_layers': self.config['dqn']['predator_hidden_layers'],
            'config': self.config['dqn']
        }
        metadata_path = os.path.join(path, agent_filename.replace('.pth', '_metadata.json'))
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_checkpoint(self, path):
        """Load checkpoint."""
        self.agent_brain.load(f"{path}/agent_ep_{self.current_episode}.pth")
        self.predator_brain.load(f"{path}/predator_ep_{self.current_episode}.pth")

    def reset(self):
        """Reset training."""
        self.current_episode = 0
        self.reward_history = []
        self.avg_reward_history = []
        self.best_avg_reward = -float('inf')
        self.best_agent_brain_path = None
        self.best_predator_brain_path = None
        self.best_metric_value = -float('inf')
        self.steps_without_improvement = 0
        self.early_stopping_triggered = False

    def get_best_brain(self):
        """Get agent brain."""
        return self.agent_brain

    def get_frame(self):
        """Get current frame."""
        return None
