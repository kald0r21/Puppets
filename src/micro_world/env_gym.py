# env_gym.py
# Gymnasium wrapper for enhanced MicroWorld with evolution and survival mechanics

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.scripts.micro_world_vision import (
    World, Agent, ACTIONS, OBS_R, MAX_F, CAMERA_SIZE,
    EVOLUTION_THRESHOLDS, MAX_LEVEL
)


class MicroWorldVisionEnv(gym.Env):
    """
    Gymnasium environment for MicroWorld

    Observation Space:
        - RGB image (HWC): (13, 13, 3) uint8 [0, 255]
        - Channels: R=pheromones, G=food, B=safe_zones+walls

    Action Space:
        - Discrete(9): Stay, Up, Down, Right, Left, and 4 diagonals

    Reward Structure:
        - Food gain: +0.4 per food eaten
        - Movement cost: -0.005
        - Energy decay: -0.001 per step
        - Night drain: -0.003 to -0.009 (less in safe zones)
        - Safe zone bonus: +0.02 when protected at night
        - Wall penalty: -0.1 for collisions
        - Stuck penalty: -0.05 for repetitive positions
        - Predator penalty: -0.05 to -0.25 based on proximity
        - Evolution bonus: +2.0 for leveling up
        - Survival bonus: +0.001 per step alive

    Episode Termination:
        - Agent runs out of energy (death)
        - Maximum steps reached (truncation)
    """

    metadata = {
        "render_modes": ["rgb_array", "ansi"],
        "render_fps": 10
    }

    def __init__(
            self,
            seed: int = 123,
            max_steps: int = 5000,
            render_mode: str | None = None,
            n_pellets: int = 500
    ):
        """
        Initialize environment

        Args:
            seed: Random seed for reproducibility
            max_steps: Maximum steps per episode
            render_mode: 'rgb_array' or 'ansi' for visualization
            n_pellets: Number of food pellets to spawn
        """
        super().__init__()

        self.n_pellets = n_pellets
        self.world = World(seed=seed, n_pellets=n_pellets)
        self.agent = Agent(self.world, agent_id=0)
        self.rng = np.random.RandomState(seed)
        self.max_steps = max_steps
        self._step_count = 0
        self.render_mode = render_mode

        # Observation space: RGB image
        cam_h = cam_w = CAMERA_SIZE[0]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(cam_h, cam_w, 3),  # HWC format
            dtype=np.uint8
        )

        # Action space: 9 discrete actions
        self.action_space = spaces.Discrete(len(ACTIONS))

    def _get_obs(self):
        """Get current observation (agent's egocentric view)"""
        img_chw = self.agent.get_obs_img().astype(np.float32)  # [3, H, W] in [0, 1]
        img_hwc = np.clip(img_chw, 0.0, 1.0).transpose(1, 2, 0)  # [H, W, 3]
        return (img_hwc * 255).astype(np.uint8)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict | None = None
    ):
        """
        Reset environment to initial state

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Create new world and agent
        world_seed = int(self.rng.randint(0, 100_000))
        self.world = World(seed=world_seed, n_pellets=self.n_pellets)
        self.agent = Agent(self.world, agent_id=0)
        self._step_count = 0

        obs = self._get_obs()
        info = {
            "energy": float(self.agent.energy),
            "score": float(self.agent.score),
            "pellets": self.world.count_pellets(),
            "level": self.agent.level,
            "pellets_eaten": self.agent.pellets_eaten,
            "evolution_progress": self.agent.get_evolution_progress(),
            "day_progress": self.world.day_progress,
            "is_day": self.world.is_day,
        }

        return obs, info

    def step(self, action: int):
        """
        Execute one environment step

        Args:
            action: Action to take (0-8)

        Returns:
            observation: New observation after action
            reward: Reward for this step
            terminated: Whether episode ended (agent died)
            truncated: Whether episode was truncated (max steps)
            info: Additional information
        """
        self._step_count += 1

        # World dynamics (diffusion, day/night cycle, predators)
        self.world.step_fields()

        # Track level before action
        prev_level = self.agent.level

        # Agent acts and gets reward
        reward, alive = self.agent.step(int(action))

        # Check if agent evolved
        evolved = self.agent.level > prev_level

        # Get new observation
        obs = self._get_obs()

        # Episode termination
        terminated = not alive
        truncated = self._step_count >= self.max_steps

        # Information dictionary
        info = {
            "energy": float(self.agent.energy),
            "score": float(self.agent.score),
            "alive": alive,
            "pellets": self.world.count_pellets(),
            "pellets_eaten": self.agent.pellets_eaten,
            "level": self.agent.level,
            "evolution_progress": self.agent.get_evolution_progress(),
            "evolved": evolved,
            "day_progress": self.world.day_progress,
            "is_day": self.world.is_day,
            "steps": self._step_count,
        }

        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        """
        Render the environment

        Returns:
            RGB array if render_mode='rgb_array', ANSI string if 'ansi'
        """
        if self.render_mode == "rgb_array":
            # Create RGB representation of world
            F = np.clip(self.world.F / MAX_F, 0.0, 1.0)  # Food (normalized)
            P = np.clip(self.world.P, 0.0, 1.0)  # Pheromones
            H = np.clip(self.world.H, 0.0, 1.0)  # Safe zones

            # Build RGB image
            img_r = (P * 255).astype(np.uint8)
            img_g = (F * 255).astype(np.uint8)
            img_b = (H * 255).astype(np.uint8)

            rgb = np.stack([img_r, img_g, img_b], axis=-1)

            # Mark walls as gray
            wall_indices = np.where(self.world.W > 0)
            rgb[wall_indices] = [100, 100, 100]

            # Mark agent position
            ay, ax = self.agent.y, self.agent.x
            if 0 <= ay < self.world.h and 0 <= ax < self.world.w:
                rgb[ay, ax] = self.agent.color

            # Mark predators as bright red
            for pred in self.world.predators:
                py, px = pred.y, pred.x
                if 0 <= py < self.world.h and 0 <= px < self.world.w:
                    rgb[py, px] = [255, 0, 0]

            return rgb

        elif self.render_mode == "ansi":
            # Simple ASCII rendering
            chars = " .,:;+=*#@"
            F = np.clip(self.world.F / MAX_F, 0, 1)
            img = (F * (len(chars) - 1)).astype(int)

            # Mark agent
            img[self.agent.y, self.agent.x] = len(chars) - 1

            # Build string
            rows = []
            for y in range(self.world.h):
                row = "".join(chars[img[y, x]] for x in range(self.world.w))
                rows.append(row)

            return "\n".join(rows)

        return None

    def close(self):
        """Clean up resources"""
        pass