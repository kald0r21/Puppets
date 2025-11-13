# src/scripts/env_gym.py
# Gymnasium wrapper with evolution system

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- POPRAWIONY IMPORT ---
from src.scripts.micro_world_vision import World, Agent, ACTIONS, OBS_R, MAX_F, CAMERA_SIZE
# --- KONIEC POPRAWKI ---


class MicroWorldVisionEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "ansi"], "render_fps": 10}

    def __init__(self, seed: int = 123, max_steps: int = 5000, render_mode: str | None = None,
                 n_pellets: int = 400):  # Zwiększone peletki
        super().__init__()
        self.n_pellets = n_pellets
        self.world = World(seed=seed, n_pellets=n_pellets)
        self.agent = Agent(self.world, agent_id=0)
        self.rng = np.random.RandomState(seed)
        self.max_steps = max_steps
        self._step_count = 0
        self.render_mode = render_mode

        cam_h = cam_w = CAMERA_SIZE[0]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(cam_h, cam_w, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(len(ACTIONS))

    def _get_obs(self):
        img_chw = self.agent.get_obs_img().astype(np.float32)
        img_hwc = np.clip(img_chw, 0.0, 1.0).transpose(1, 2, 0)
        return (img_hwc * 255).astype(np.uint8)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.world = World(seed=int(self.rng.randint(0, 100_000)), n_pellets=self.n_pellets)
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
        self._step_count += 1
        self.world.step_fields()  # Świat się aktualizuje (w tym cykl dnia)

        prev_level = self.agent.level
        reward, alive = self.agent.step(int(action))
        evolved = self.agent.level > prev_level

        obs = self._get_obs()
        terminated = not alive
        truncated = self._step_count >= self.max_steps
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
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "rgb_array":
            F = np.clip(self.world.F, 0.0, 1.0)  # Jedzenie (Zielony)
            P = np.clip(self.world.P, 0.0, 1.0)  # Feromony (Czerwony)
            H = np.clip(self.world.H, 0.0, 1.0)  # Schronienie (Niebieski)

            img_r = (P * 255).astype(np.uint8)
            img_g = (F * 255).astype(np.uint8)
            img_b = (H * 255).astype(np.uint8)  # Schronienia są niebieskie

            rgb = np.stack([img_r, img_g, img_b], axis=-1)

            # Ściany są szare
            wall_indices = np.where(self.world.W > 0)
            rgb[wall_indices] = [100, 100, 100]

            # Agent
            ay, ax = self.agent.y, self.agent.x
            if 0 <= ay < self.world.h and 0 <= ax < self.world.w:
                rgb[ay, ax] = self.agent.color

            return rgb
        elif self.render_mode == "ansi":
            # ... (Tryb ANSI jest teraz zbyt skomplikowany, pomijamy) ...
            return "ANSI rendering not supported with new world state."
        return None

    def close(self):
        pass