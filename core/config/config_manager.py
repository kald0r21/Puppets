"""
Configuration management system.
Handles loading, saving, and validation of configurations.
"""
import json
import os
from typing import Dict, Any
from pathlib import Path


class ConfigManager:
    """
    Manages configurations for all training methods.
    Handles JSON load/save and validation.
    """

    def __init__(self):
        self.config_dir = Path(__file__).parent / 'defaults'
        self.current_config = None

    def load_default(self, method: str) -> Dict[str, Any]:
        """
        Load default configuration for a method.

        Args:
            method: 'GA', 'CNN', or 'DQN'

        Returns:
            dict: Configuration dictionary
        """
        method = method.lower()
        config_file = self.config_dir / f"{method}_default.json"

        if not config_file.exists():
            raise FileNotFoundError(f"Default config for {method} not found: {config_file}")

        with open(config_file, 'r') as f:
            config = json.load(f)

        self.current_config = config
        return config

    def load_from_file(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.

        Args:
            path: Path to JSON file

        Returns:
            dict: Configuration dictionary
        """
        with open(path, 'r') as f:
            config = json.load(f)

        self.current_config = config
        return config

    def save_to_file(self, config: Dict[str, Any], path: str):
        """
        Save configuration to a JSON file.

        Args:
            config: Configuration dictionary
            path: Path to save to
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    def merge(self, base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration dictionaries (deep merge).

        Args:
            base: Base configuration
            overrides: Override values

        Returns:
            dict: Merged configuration
        """
        result = base.copy()

        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge(result[key], value)
            else:
                result[key] = value

        return result

    def validate(self, config: Dict[str, Any]) -> list:
        """
        Validate configuration.

        Args:
            config: Configuration to validate

        Returns:
            list: List of error messages (empty if valid)
        """
        errors = []

        # Check required top-level keys
        required_keys = ['simulation', 'method', 'visualization']
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required key: {key}")

        # Validate method
        if 'method' in config:
            method = config['method'].upper()
            if method not in ['GA', 'CNN', 'DQN']:
                errors.append(f"Invalid method: {method}. Must be GA, CNN, or DQN")

            # Check method-specific config
            method_key = method.lower()
            if method_key not in config:
                errors.append(f"Missing configuration for method: {method}")

        # Validate simulation params
        if 'simulation' in config:
            sim = config['simulation']
            required_sim = ['grid_width', 'grid_height', 'start_energy']
            for key in required_sim:
                if key not in sim:
                    errors.append(f"Missing simulation parameter: {key}")

            # Check positive values
            if 'grid_width' in sim and sim['grid_width'] <= 0:
                errors.append("grid_width must be positive")
            if 'grid_height' in sim and sim['grid_height'] <= 0:
                errors.append("grid_height must be positive")

        return errors

    def get_method(self, config: Dict[str, Any]) -> str:
        """
        Get method from config.

        Args:
            config: Configuration dictionary

        Returns:
            str: Method name (GA, CNN, or DQN)
        """
        return config.get('method', 'GA').upper()

    def create_default_config(self, method: str) -> Dict[str, Any]:
        """
        Create a default configuration programmatically.

        Args:
            method: 'GA', 'CNN', or 'DQN'

        Returns:
            dict: Default configuration
        """
        method = method.upper()

        base_config = {
            "version": "1.0",
            "method": method,
            "simulation": {
                "grid_width": 40,
                "grid_height": 40,
                "num_food": 45,
                "num_walls": 50,
                "start_energy": 100,
                "eat_gain": 150,
                "max_energy_gain_per_food": 10,
                "move_cost": 1,
                "idle_cost": 3,
                "wall_hit_penalty": 3,
                "smart_perception_radius": 10,
                "reward_shaping_factor": 0.1,
                "predator_count": 5,
                "predator_respawn": True,
                "predator_vision": 10,
                "predator_base_strength": 5,
                "predator_ally_bonus": 2,
                "predator_ally_radius": 3,
                "kill_license_level": 5,
                "kill_cost_pellets": 1
            },
            "visualization": {
                "fps": 60,
                "cell_size": 14,
                "colors": {
                    "background": [20, 20, 20],
                    "grid": [40, 40, 40],
                    "food": [0, 255, 0],
                    "agent": [0, 150, 255],
                    "predator": [255, 0, 0],
                    "wall": [100, 100, 100]
                }
            }
        }

        if method == "GA":
            base_config["ga"] = {
                "population_size": 100,
                "num_generations": 200,
                "max_turns_per_gen": 1000,
                "mutation_rate": 0.05,
                "mutation_strength": 0.5,
                "elitism_count": 10,
                "hidden_layers": [16, 16],
                "early_stopping_patience": 20,
                "early_stopping_enabled": True
            }
        elif method == "CNN":
            base_config["cnn"] = {
                "population_size": 100,
                "num_generations": 50,
                "max_turns_per_gen": 1000,
                "mutation_rate": 0.05,
                "mutation_strength": 0.5,
                "elitism_count": 10,
                "max_perception_radius": 3,
                "start_perception_radius": 1,
                "vision_upgrades": {
                    "2": 2,
                    "5": 3
                },
                "early_stopping_patience": 10,
                "early_stopping_enabled": True
            }
            base_config["simulation"]["start_perception_radius"] = 1
        elif method == "DQN":
            base_config["dqn"] = {
                "num_episodes": 2000,
                "max_turns_per_episode": 1000,
                "agent_hidden_layers": [64, 32],
                "agent_learning_rate": 0.001,
                "agent_memory_size": 50000,
                "agent_batch_size": 128,
                "agent_eps_start": 0.9,
                "agent_eps_end": 0.05,
                "agent_eps_decay": 30000,
                "predator_hidden_layers": [64, 32],
                "predator_learning_rate": 0.001,
                "predator_memory_size": 50000,
                "predator_batch_size": 128,
                "predator_eps_start": 0.9,
                "predator_eps_end": 0.05,
                "predator_eps_decay": 20000,
                "gamma": 0.99,
                "early_stopping_patience": 50,
                "early_stopping_enabled": True
            }

        return base_config
