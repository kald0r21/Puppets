# src/scripts/micro_world_vision.py
# Wersja 3.2: Usunięcie szkodliwej "kary za głód"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, time, os, sys, signal

# ---------------------------- Config ----------------------------
H, W = 128, 128
OBS_R = 5
CAMERA_SIZE = (2 * OBS_R + 1, 2 * OBS_R + 1)  # 11x11

DIFF_ALPHA = 0.12
MAX_F = 1.0

MOVE_COST = 0.0005
E_DECAY = 0.0008
PELLET_VALUE = 1.0

# Food scent gradient
FOOD_SCENT_STRENGTH = 0.8
FOOD_SCENT_RADIUS = 8

# Evolution
MAX_LEVEL = 5
EVOLUTION_THRESHOLDS = [0, 8, 20, 40, 80]
LEVEL_COLORS = [
    (255, 255, 255), (100, 255, 100), (100, 100, 255),
    (255, 100, 255), (255, 215, 0),
]


def get_level_stats(level):
    return {
        1: (1.0, 5, 8.0, 4.0), 2: (1.2, 6, 10.0, 5.0), 3: (1.5, 7, 12.0, 6.0),
        4: (1.8, 8, 15.0, 8.0), 5: (2.0, 10, 20.0, 10.0),
    }[level]


ACTIONS = [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, -1), (1, 1)]
N_ACTIONS = len(ACTIONS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------- Assets ----------------------------
class Asset:
    def __init__(self, y, x, asset_type):
        self.y, self.x, self.type, self.active = y, x, asset_type, True

    def interact(self, agent): pass

    def update(self, world): pass


class Predator(Asset):
    def __init__(self, y, x):
        super().__init__(y, x, 'predator')
        self.energy, self.speed, self.damage = 100.0, 0.3, 1.0
        self.detection_range, self.step_count = 8, 0

    def interact(self, agent):
        if self.active and agent.shield == 0:
            agent.energy -= self.damage
            return -5.0
        return 0.0

    def update(self, world):
        self.step_count += 1
        if self.step_count % 3 != 0: return
        if random.random() < 0.3:
            dy, dx = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            new_y, new_x = self.y + dy, self.x + dx
            # ZMIANA: Sprawdzenie granic świata
            if (0 <= new_y < world.h and 0 <= new_x < world.w and
                    world.W[new_y, new_x] == 0):
                self.y, self.x = new_y, new_x


class ToxicZone(Asset):
    def __init__(self, y, x, radius=2):
        super().__init__(y, x, 'toxic')
        self.radius, self.damage_per_step = radius, 0.2

    def interact(self, agent):
        # ZMIANA: Usunięto logikę "wrap-around" dla dystansu
        dy = abs(agent.y - self.y)
        dx = abs(agent.x - self.x)
        dist = np.sqrt(dy * dy + dx * dx)
        if dist <= self.radius and agent.shield == 0:
            agent.energy -= self.damage_per_step
            return -1.0
        return 0.0


class PowerUp(Asset):
    def __init__(self, y, x, boost_type='speed'):
        super().__init__(y, x, 'powerup')
        self.boost_type, self.duration, self.consumed = boost_type, 150, False

    def interact(self, agent):
        if not self.consumed:
            self.consumed, self.active = True, False
            if self.boost_type == 'speed':
                agent.temp_speed_boost, agent.speed = self.duration, agent.speed * 1.5
            elif self.boost_type == 'vision':
                agent.temp_vision_boost, agent.vision_range = self.duration, agent.vision_range + 3
            elif self.boost_type == 'shield':
                agent.shield = self.duration
            return 10.0
        return 0.0


class Wall(Asset):
    def __init__(self, y, x): super().__init__(y, x, 'wall')


class Shelter(Asset):
    def __init__(self, y, x):
        super().__init__(y, x, 'shelter')


# ---------------------------- World ----------------------------
class World:
    def __init__(self, h=H, w=W, seed=42, n_pellets=400):
        self.h, self.w = h, w
        self.rng = np.random.RandomState(seed)
        self.n_pellets = n_pellets

        self.F = np.zeros((h, w), dtype=np.float32)  # Food
        self.P = np.zeros((h, w), dtype=np.float32)  # Pheromone
        self.T = np.zeros((h, w), dtype=np.float32)  # Toxic
        self.W = np.zeros((h, w), dtype=np.float32)  # Wall
        self.S = np.zeros((h, w), dtype=np.float32)  # Scent
        self.H = np.zeros((h, w), dtype=np.float32)  # Home (Shelter)

        self.assets = []
        self.spawn_pellets(n_pellets)
        self.spawn_assets()

        self.global_step = 0
        self.pellets_eaten = 0

        self.day_duration = 1000
        self.night_duration = 500
        self.total_cycle = self.day_duration + self.night_duration
        self.is_day = True
        self.day_progress = 0.0

        self.update_food_scent()

    def spawn_assets(self):
        # Predators
        n_predators = self.rng.randint(2, 4)
        for _ in range(n_predators):
            self.assets.append(Predator(self.rng.randint(0, self.h), self.rng.randint(0, self.w)))

        # Toxic zones
        n_toxic = self.rng.randint(3, 6)
        for _ in range(n_toxic):
            y, x, radius = self.rng.randint(0, self.h), self.rng.randint(0, self.w), self.rng.randint(2, 3)
            toxic = ToxicZone(y, x, radius)
            self.assets.append(toxic)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ty, tx = y + dy, x + dx
                    if 0 <= ty < self.h and 0 <= tx < self.w:
                        if dy * dy + dx * dx <= radius * radius:
                            self.T[ty, tx] = 1.0

        # Walls (Labirynt)
        n_walls = self.rng.randint(80, 120)
        for _ in range(n_walls):
            y, x = self.rng.randint(0, self.h), self.rng.randint(0, self.w)
            for dy, dx in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]:
                wy, wx = y + dy, x + dx
                if 0 <= wy < self.h and 0 <= wx < self.w and self.W[wy, wx] == 0:
                    self.assets.append(Wall(wy, wx))
                    self.W[wy, wx] = 1.0

        # Power-ups
        n_powerups = self.rng.randint(5, 8)
        for _ in range(n_powerups):
            boost_type = self.rng.choice(['speed', 'vision', 'shield'])
            self.assets.append(PowerUp(self.rng.randint(0, self.h), self.rng.randint(0, self.w), boost_type))

        # Schronienia 2x2
        n_shelters = 10
        for _ in range(n_shelters):
            y = self.rng.randint(0, self.h - 2)
            x = self.rng.randint(0, self.w - 2)
            if np.sum(self.W[y:y + 2, x:x + 2]) == 0:
                for dy in range(2):
                    for dx in range(2):
                        sy, sx = y + dy, x + dx
                        if self.H[sy, sx] == 0:
                            self.assets.append(Shelter(sy, sx))
                            self.H[sy, sx] = 1.0

    def update_food_scent(self):
        self.S.fill(0.0)
        food_positions = np.argwhere(self.F > 0)
        for fy, fx in food_positions:
            for dy in range(-FOOD_SCENT_RADIUS, FOOD_SCENT_RADIUS + 1):
                for dx in range(-FOOD_SCENT_RADIUS, FOOD_SCENT_RADIUS + 1):
                    dist2 = dy * dy + dx * dx
                    if dist2 <= FOOD_SCENT_RADIUS * FOOD_SCENT_RADIUS:
                        ty, tx = fy + dy, fx + dx
                        if 0 <= ty < self.h and 0 <= tx < self.w:
                            strength = FOOD_SCENT_STRENGTH * np.exp(-dist2 / (2 * (FOOD_SCENT_RADIUS / 2) ** 2))
                            self.S[ty, tx] = max(self.S[ty, tx], strength)

    def spawn_pellets(self, n):
        for _ in range(n):
            attempts = 0
            while attempts < 100:
                y, x = self.rng.randint(0, self.h), self.rng.randint(0, self.w)
                if self.F[y, x] == 0 and self.W[y, x] == 0 and self.T[y, x] == 0 and self.H[y, x] == 0:
                    self.F[y, x] = PELLET_VALUE
                    break
                attempts += 1

    def laplace(self, X):
        up = np.roll(X, -1, 0);
        down = np.roll(X, 1, 0)
        left = np.roll(X, -1, 1);
        right = np.roll(X, 1, 1)
        L = (up + down + left + right - 4 * X)
        L[0, :] += X[1, :] - X[0, :]
        L[-1, :] += X[-2, :] - X[-1, :]
        L[:, 0] += X[:, 1] - X[:, 0]
        L[:, -1] += X[:, -2] - X[:, -1]
        return L

    def step_fields(self):
        self.global_step += 1
        cycle_time = self.global_step % self.total_cycle
        self.is_day = cycle_time < self.day_duration
        if self.is_day:
            self.day_progress = cycle_time / self.day_duration
        else:
            self.day_progress = -1.0

        self.P += 0.10 * DIFF_ALPHA * self.laplace(self.P);
        self.P -= 0.02 * self.P
        np.clip(self.P, 0.0, 1.0, out=self.P)

        self.F *= 0.999;
        np.clip(self.F, 0.0, 1.0, out=self.F)

        if self.global_step % 5 == 0: self.update_food_scent()
        for asset in self.assets: asset.update(self)

        if self.global_step % 30 == 0:
            current_pellets = int(np.sum(self.F > 0))
            if current_pellets < self.n_pellets * 0.6:
                self.spawn_pellets(min(30, self.n_pellets - current_pellets))
                self.update_food_scent()

        if self.global_step % 150 == 0:
            active_powerups = sum(1 for a in self.assets if a.type == 'powerup' and a.active)
            if active_powerups < 5:
                boost_type = self.rng.choice(['speed', 'vision', 'shield'])
                self.assets.append(PowerUp(self.rng.randint(0, self.h), self.rng.randint(0, self.w), boost_type))

    def count_pellets(self):
        return int(np.sum(self.F > 0))

    def get_assets_at(self, y, x):
        return [a for a in self.assets if a.y == y and a.x == x and a.active]


# ---------------------------- Agent ----------------------------
class Agent:
    def __init__(self, world: World, agent_id=0, spawn_safe=True):
        self.world, self.agent_id = world, agent_id
        if spawn_safe:
            safe_spawn = False;
            attempts = 0
            while not safe_spawn and attempts < 100:
                self.y, self.x = world.rng.randint(0, world.h), world.rng.randint(0, world.w)
                safe = True
                for asset in world.assets:
                    dist = np.sqrt((self.y - asset.y) ** 2 + (self.x - asset.x) ** 2)
                    if (asset.type == 'predator' and dist < 15) or \
                            (asset.type == 'toxic' and dist < asset.radius + 2):
                        safe = False;
                        break
                if world.W[self.y, self.x] > 0 or world.T[self.y, self.x] > 0: safe = False
                if safe: safe_spawn = True
                attempts += 1
        else:
            self.y, self.x = world.rng.randint(0, world.h), world.rng.randint(0, world.w)

        self.level, self.pellets_eaten = 1, 0
        speed, vision, init_energy, eat_gain = get_level_stats(self.level)
        self.base_speed, self.base_vision = speed, vision
        self.speed, self.vision_range = speed, vision
        self.eat_gain, self.color = eat_gain, LEVEL_COLORS[self.level - 1]

        self.energy, self.max_energy = init_energy, init_energy * 2
        self.score, self.facing = 0.0, (-1, 0)
        self.steps_without_food = 0
        self.split_threshold = 15.0 + (self.level - 1) * 3.0
        self.temp_speed_boost, self.temp_vision_boost, self.shield = 0, 0, 0
        self.last_bump_penalty = 0.0

    def evolve(self):
        if self.level >= MAX_LEVEL: return False
        next_threshold = EVOLUTION_THRESHOLDS[self.level]
        if self.pellets_eaten >= next_threshold:
            self.level += 1
            speed, vision, _, eat_gain = get_level_stats(self.level)
            self.base_speed, self.base_vision = speed, vision
            self.speed, self.vision_range = speed, vision
            self.eat_gain, self.color = eat_gain, LEVEL_COLORS[self.level - 1]
            self.split_threshold = 15.0 + (self.level - 1) * 3.0
            self.energy += 10.0
            self.max_energy = get_level_stats(self.level)[2] * 2
            self.score += 10.0
            print(f"🎉 Agent {self.agent_id + 1} EVOLVED to level {self.level}!")
            return True
        return False

    def get_evolution_progress(self):
        if self.level >= MAX_LEVEL: return 1.0
        next_threshold = EVOLUTION_THRESHOLDS[self.level]
        prev_threshold = EVOLUTION_THRESHOLDS[self.level - 1]
        progress = (self.pellets_eaten - prev_threshold) / (next_threshold - prev_threshold)
        return max(0.0, min(1.0, progress))

    def get_obs_img(self):
        R = self.vision_range
        cam_h = cam_w = 2 * R + 1

        F_patch = np.zeros((cam_h, cam_w), dtype=np.float32)
        S_patch = np.zeros((cam_h, cam_w), dtype=np.float32)
        P_patch = np.zeros((cam_h, cam_w), dtype=np.float32)
        T_patch = np.zeros((cam_h, cam_w), dtype=np.float32)
        W_patch = np.zeros((cam_h, cam_w), dtype=np.float32)
        H_patch = np.zeros((cam_h, cam_w), dtype=np.float32)
        W_patch.fill(1.0)

        y_min_world = max(0, self.y - R);
        y_max_world = min(self.world.h, self.y + R + 1)
        x_min_world = max(0, self.x - R);
        x_max_world = min(self.world.w, self.x + R + 1)
        y_min_cam = y_min_world - (self.y - R);
        y_max_cam = y_max_world - (self.y - R)
        x_min_cam = x_min_world - (self.x - R);
        x_max_cam = x_max_world - (self.x - R)

        if y_max_cam > y_min_cam and x_max_cam > x_min_cam:
            F_patch[y_min_cam:y_max_cam, x_min_cam:x_max_cam] = self.world.F[y_min_world:y_max_world,
                                                                x_min_world:x_max_world]
            S_patch[y_min_cam:y_max_cam, x_min_cam:x_max_cam] = self.world.S[y_min_world:y_max_world,
                                                                x_min_world:x_max_world]
            P_patch[y_min_cam:y_max_cam, x_min_cam:x_max_cam] = self.world.P[y_min_world:y_max_world,
                                                                x_min_world:x_max_world]
            T_patch[y_min_cam:y_max_cam, x_min_cam:x_max_cam] = self.world.T[y_min_world:y_max_world,
                                                                x_min_world:x_max_world]
            W_patch[y_min_cam:y_max_cam, x_min_cam:x_max_cam] = self.world.W[y_min_world:y_max_world,
                                                                x_min_world:x_max_world]
            H_patch[y_min_cam:y_max_cam, x_min_cam:x_max_cam] = self.world.H[y_min_world:y_max_world,
                                                                x_min_world:x_max_world]

        Rch = np.clip(P_patch, 0.0, 1.0)
        Gch = np.clip(F_patch + S_patch * 0.5, 0.0, 1.0)
        Bch = np.clip(T_patch + W_patch + H_patch, 0.0, 1.0)

        for asset in self.world.assets:
            if asset.type == 'predator' and asset.active:
                rel_y, rel_x = asset.y - self.y, asset.x - self.x
                if abs(rel_y) <= R and abs(rel_x) <= R:
                    Rch[rel_y + R, rel_x + R] = 1.0

        img = np.stack([Rch, Gch, Bch], axis=0).astype(np.float32)

        if self.world.is_day:
            progress = self.world.day_progress
            img_w = img.shape[2]
            white_end = int(img_w * 0.5)
            gray_end = int(img_w * progress)
            if gray_end > white_end:
                img[:, 0, :white_end] = 1.0
                img[:, 0, white_end:gray_end] = 0.5
            else:
                img[:, 0, :gray_end] = 1.0
        else:
            img[:, 0, :] = 0.1

        dy, dx = self.facing
        k = 0
        if (dy, dx) == (0, 1):
            k = 1
        elif (dy, dx) == (1, 0):
            k = 2
        elif (dy, dx) == (0, -1):
            k = 3
        img = np.rot90(img, k=k, axes=(1, 2)).copy()

        if img.shape[1] != CAMERA_SIZE[0] or img.shape[2] != CAMERA_SIZE[1]:
            standard_img = np.zeros((3, CAMERA_SIZE[0], CAMERA_SIZE[1]), dtype=np.float32)
            h_src, w_src = img.shape[1], img.shape[2]
            h_dst, w_dst = CAMERA_SIZE[0], CAMERA_SIZE[1]
            if h_src >= h_dst and w_src >= w_dst:
                start_y, start_x = (h_src - h_dst) // 2, (w_src - w_dst) // 2
                standard_img = img[:, start_y:start_y + h_dst, start_x:start_x + w_dst]
            else:
                start_y, start_x = (h_dst - h_src) // 2, (w_dst - w_src) // 2
                standard_img[:, start_y:start_y + h_src, start_x:start_x + w_src] = img
            return standard_img

        return img

    def act(self, action_id: int):
        self.last_bump_penalty = 0.0
        dy, dx = ACTIONS[action_id]
        if (dy, dx) != (0, 0): self.facing = (np.sign(dy), np.sign(dx))
        new_y, new_x = self.y + dy, self.x + dx

        if not (0 <= new_y < self.world.h and 0 <= new_x < self.world.w):
            self.last_bump_penalty = -0.1
            return
        if self.world.W[new_y, new_x] > 0:
            self.last_bump_penalty = -0.1
            return

        if self.speed > 1.0 and random.random() < (self.speed - 1.0):
            new_y_2, new_x_2 = new_y + dy, new_x + dx
            if (0 <= new_y_2 < self.world.h and 0 <= new_x_2 < self.world.w and
                    self.world.W[new_y_2, new_x_2] == 0):
                self.y, self.x = new_y_2, new_x_2
            else:
                self.y, self.x = new_y, new_x
        else:
            self.y, self.x = new_y, new_x

    def emit_pheromone(self, amount=0.15):
        amount *= (1.0 + 0.1 * (self.level - 1))
        self.world.P[self.y, self.x] = np.clip(self.world.P[self.y, self.x] + amount, 0.0, 1.0)

    def eat(self):
        if self.world.F[self.y, self.x] > 0:
            self.world.F[self.y, self.x] = 0.0
            self.world.pellets_eaten += 1
            self.pellets_eaten += 1
            evolved = self.evolve()
            gain = self.eat_gain
            self.energy = min(self.energy + gain, self.max_energy)
            self.steps_without_food = 0
            if evolved: return gain + 30.0
            return gain
        else:
            self.steps_without_food += 1
            return 0.0

    def interact_with_assets(self):
        total_reward = 0.0
        if self.temp_speed_boost > 0:
            self.temp_speed_boost -= 1;
            if self.temp_speed_boost == 0: self.speed = self.base_speed
        if self.temp_vision_boost > 0:
            self.temp_vision_boost -= 1
            if self.temp_vision_boost == 0: self.vision_range = self.base_vision
        if self.shield > 0: self.shield -= 1

        for asset in self.world.get_assets_at(self.y, self.x):
            if asset.type == 'shelter': continue
            if self.shield == 0 or asset.type == 'powerup':
                total_reward += asset.interact(self)

        for asset in self.world.assets:
            if asset.type == 'toxic' and asset.active and self.shield == 0:
                total_reward += asset.interact(self)

        return total_reward

    def step(self, action_id: int):
        self.act(action_id)
        bump_penalty = self.last_bump_penalty
        gain = self.eat()
        asset_reward = self.interact_with_assets()

        if gain > 0:
            self.emit_pheromone(0.2)
        else:
            self.emit_pheromone(0.03)

        if action_id == 0:
            stay_penalty = 0.03  # Utrzymujemy 0.03
        else:
            stay_penalty = 0.0

        move_penalty = MOVE_COST if action_id != 0 else 0
        self.energy -= (move_penalty + E_DECAY + stay_penalty)

        reward = gain + asset_reward - move_penalty - E_DECAY - stay_penalty + bump_penalty

        # --- KLUCZOWA ZMIANA: USUNIĘCIE KARY ZA GŁÓD ---
        # Ta kara "zagłuszała" inne sygnały i powodowała paraliż

        # if self.steps_without_food > 50:
        #     reward -= 0.01 * (self.steps_without_food - 50)

        # --- KONIEC ZMIANY ---

        if self.energy >= self.split_threshold:
            reward += (15.0 + (self.level - 1) * 5.0)
            self.score += 1.0
            self.energy *= 0.6
            self.steps_without_food = 0

        if hasattr(self, 'last_action') and action_id == self.last_action:
            self.repeat_count += 1
            if self.repeat_count > 10: reward -= 0.1
        else:
            self.repeat_count = 0
        self.last_action = action_id

        if not hasattr(self, 'visited'): self.visited = set()
        pos = (self.y, self.x)
        if pos not in self.visited:
            reward += 0.001
            self.visited.add(pos)

        in_shelter = self.world.H[self.y, self.x] > 0

        if self.world.is_day:
            if in_shelter:
                reward -= 0.1
        else:  # Jest Noc
            if not in_shelter:
                night_penalty = 5.0
                self.energy -= night_penalty
                reward -= night_penalty

        alive = self.energy > 0.0
        return reward, alive


# --- Reszta pliku bez zmian ---
class VisionPolicy(nn.Module):
    def __init__(self, in_ch=3, n_actions=N_ACTIONS, cam_hw=CAMERA_SIZE):
        super().__init__()
        h, w = cam_hw
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Dropout2d(0.1)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, h, w)
            enc = self.encoder(dummy)
            self.flat_dim = enc.view(1, -1).shape[1]

        self.policy_head = nn.Sequential(
            nn.Linear(self.flat_dim, 256), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_actions)
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.flat_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1)
        logits = self.policy_head(z)
        value = self.value_head(z)
        return logits, value


def get_rng_state():
    return {
        "py_random": random.getstate(),
        "np_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    random.setstate(state["py_random"])
    np.random.set_state(state["np_random"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


if __name__ == "__main__":
    print("✓ Balanced life simulation initialized")
    print("✓ DAY/NIGHT CYCLE | CLOSED WORLD | 2x2 SHELTERS")