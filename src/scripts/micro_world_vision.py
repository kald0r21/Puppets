# micro_world_vision.py
# Enhanced visual RL micro-world with improved survival mechanics, memory system, and predators
# Features: Wall avoidance, night survival, memory buffer, fine-tuning support

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random, time, os, sys, signal
from collections import deque

# ---------------------------- Config ----------------------------
H, W = 128, 128  # grid size
OBS_R = 6  # camera radius -> (2R+1) x (2R+1) pixels (zwiększone)
CAMERA_SIZE = (2 * OBS_R + 1, 2 * OBS_R + 1)

DIFF_ALPHA = 0.08  # diffusion coefficient
DECAY = 0.005  # resource decay
MAX_F = 2.0  # clamp for field

# Energy and survival
MOVE_COST = 0.005  # Reduced cost for better survival
EAT_GAIN = 0.4  # Increased gain
E_DECAY = 0.001  # Slower decay
SPLIT_THRESH = 3.5
INIT_ENERGY = 2.0  # More initial energy
EAT_BITE = 1.0

# Night mechanics
NIGHT_ENERGY_DRAIN = 0.003  # Slow drain instead of instant death
SAFE_ZONE_PROTECTION = 0.98  # 98% reduction in night drain when in safe zone

# Evolution thresholds
EVOLUTION_THRESHOLDS = [0, 10, 30, 60, 100]  # pellets needed for each level
MAX_LEVEL = len(EVOLUTION_THRESHOLDS) - 1
LEVEL_COLORS = [
    (200, 200, 200),  # Level 0: White
    (100, 200, 255),  # Level 1: Light Blue
    (100, 255, 100),  # Level 2: Green
    (255, 255, 100),  # Level 3: Yellow
    (255, 150, 50),  # Level 4: Orange
]

# 9-directional move (8 directions + stay)
ACTIONS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
N_ACTIONS = len(ACTIONS)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------- World ----------------------------
class World:
    def __init__(self, h=H, w=W, seed=42, n_pellets=500):
        self.h, self.w = h, w
        self.rng = np.random.RandomState(seed)
        self.F = np.zeros((h, w), dtype=np.float32)  # Food
        self.P = np.zeros((h, w), dtype=np.float32)  # Pheromones
        self.H = np.zeros((h, w), dtype=np.float32)  # Safe zones (Hideouts)
        self.W = np.zeros((h, w), dtype=np.float32)  # Walls

        # Day/night cycle
        self.day_length = 500
        self.night_length = 200
        self.day_progress = 0
        self.is_day = True

        # Predators
        self.predators = []
        self.max_predators = 3
        self._spawn_predators()

        # Generate world structures
        self._generate_walls()
        self._generate_safe_zones(n_zones=8)
        self._generate_food_pellets(n_pellets)

    def _generate_walls(self):
        """Generate wall structures - corridors and rooms"""
        # Border walls
        self.W[0, :] = 1.0
        self.W[-1, :] = 1.0
        self.W[:, 0] = 1.0
        self.W[:, -1] = 1.0

        # Some internal walls - vertical corridors
        for _ in range(5):
            x = self.rng.randint(10, self.w - 10)
            y_start = self.rng.randint(5, self.h // 2)
            length = self.rng.randint(20, 50)
            for y in range(y_start, min(y_start + length, self.h - 1)):
                self.W[y, x] = 1.0

        # Horizontal corridors
        for _ in range(5):
            y = self.rng.randint(10, self.h - 10)
            x_start = self.rng.randint(5, self.w // 2)
            length = self.rng.randint(20, 50)
            for x in range(x_start, min(x_start + length, self.w - 1)):
                self.W[y, x] = 1.0

    def _generate_safe_zones(self, n_zones=8):
        """Generate larger, more accessible safe zones"""
        for _ in range(n_zones):
            while True:
                cy = self.rng.randint(15, self.h - 15)
                cx = self.rng.randint(15, self.w - 15)

                # Check if area is free from walls
                if np.sum(self.W[cy - 10:cy + 10, cx - 10:cx + 10]) < 5:
                    break

            # Create larger circular safe zone
            yy, xx = np.ogrid[:self.h, :self.w]
            dy = yy - cy
            dx = xx - cx
            dist2 = dy * dy + dx * dx
            radius = 8  # Larger radius
            mask = (dist2 <= radius * radius).astype(np.float32)

            # Smooth edges
            mask = np.where(dist2 <= (radius - 2) ** 2, 1.0,
                            np.where(dist2 <= radius ** 2, 0.5, 0.0))

            self.H = np.maximum(self.H, mask)

    def _generate_food_pellets(self, n_pellets):
        """Generate food pellets throughout the world"""
        for _ in range(n_pellets):
            attempts = 0
            while attempts < 100:
                y = self.rng.randint(1, self.h - 1)
                x = self.rng.randint(1, self.w - 1)

                # Don't place on walls or safe zones
                if self.W[y, x] == 0 and self.H[y, x] < 0.5:
                    self.F[y, x] = MAX_F
                    break
                attempts += 1

    def _spawn_predators(self):
        """Spawn predator entities"""
        for i in range(self.max_predators):
            pred = Predator(self, pred_id=i)
            self.predators.append(pred)

    def count_pellets(self):
        """Count remaining food pellets"""
        return int(np.sum(self.F > MAX_F * 0.5))

    def laplace(self, X):
        """Diffusion operator with periodic boundaries"""
        up = np.roll(X, -1, axis=0)
        down = np.roll(X, 1, axis=0)
        left = np.roll(X, -1, axis=1)
        right = np.roll(X, 1, axis=1)
        return (up + down + left + right - 4 * X)

    def step_fields(self):
        """Update world state: diffusion, decay, day/night cycle"""
        # Diffusion with walls blocking
        laplace_F = self.laplace(self.F)
        laplace_F *= (1.0 - self.W)  # Walls block diffusion
        self.F += DIFF_ALPHA * laplace_F
        self.F -= DECAY * self.F

        # Pheromone diffusion (faster spread)
        laplace_P = self.laplace(self.P)
        laplace_P *= (1.0 - self.W)
        self.P += 0.2 * DIFF_ALPHA * laplace_P
        self.P -= 0.015 * self.P  # Faster decay

        # Regenerate food slowly
        REGEN_RATE = 0.002
        self.F += REGEN_RATE * (1.0 - self.W)

        # Clamp values
        np.clip(self.F, 0.0, MAX_F, out=self.F)
        np.clip(self.P, 0.0, 1.0, out=self.P)

        # Day/night cycle
        self.day_progress += 1
        if self.is_day:
            if self.day_progress >= self.day_length:
                self.is_day = False
                self.day_progress = 0
        else:
            if self.day_progress >= self.night_length:
                self.is_day = True
                self.day_progress = 0

        # Move predators
        for pred in self.predators:
            if self.is_day:
                pred.move_random()


# ---------------------------- Predator ----------------------------
class Predator:
    def __init__(self, world: 'World', pred_id: int = 0):
        self.world = world
        self.pred_id = pred_id
        # Start in random position away from safe zones
        while True:
            self.y = world.rng.randint(10, world.h - 10)
            self.x = world.rng.randint(10, world.w - 10)
            if world.H[self.y, self.x] < 0.1 and world.W[self.y, self.x] == 0:
                break
        self.color = (255, 0, 0)  # Red

    def move_random(self):
        """Random walk avoiding walls"""
        moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        random.shuffle(moves)

        for dy, dx in moves:
            ny = np.clip(self.y + dy, 0, self.world.h - 1)
            nx = np.clip(self.x + dx, 0, self.world.w - 1)

            if self.world.W[ny, nx] == 0:
                self.y, self.x = ny, nx
                break

    def is_near(self, y, x, radius=3):
        """Check if predator is near given position"""
        dy = abs(self.y - y)
        dx = abs(self.x - x)
        return (dy <= radius and dx <= radius)


# ---------------------------- Agent with Memory ----------------------------
class Agent:
    def __init__(self, world: World, agent_id: int = 0):
        self.world = world
        self.agent_id = agent_id

        # Start in safe position
        while True:
            self.y = world.rng.randint(5, world.h - 5)
            self.x = world.rng.randint(5, world.w - 5)
            if world.W[self.y, self.x] == 0:
                break

        self.energy = INIT_ENERGY
        self.score = 0.0
        self.facing = (-1, 0)  # Start facing up

        # Evolution system
        self.pellets_eaten = 0
        self.level = 0
        self.color = LEVEL_COLORS[0]

        # Memory buffer for recent observations (for learning continuity)
        self.memory_buffer = deque(maxlen=16)  # Store last 16 states
        self.action_history = deque(maxlen=8)  # Last 8 actions

        # Wall collision memory
        self.stuck_counter = 0
        self.last_positions = deque(maxlen=5)
        self.last_positions.append((self.y, self.x))

    def get_evolution_progress(self):
        """Get progress toward next level (0-1)"""
        if self.level >= MAX_LEVEL:
            return 1.0
        current_threshold = EVOLUTION_THRESHOLDS[self.level]
        next_threshold = EVOLUTION_THRESHOLDS[self.level + 1]
        progress = (self.pellets_eaten - current_threshold) / (next_threshold - current_threshold)
        return np.clip(progress, 0.0, 1.0)

    def check_evolution(self):
        """Check if agent should evolve to next level"""
        if self.level < MAX_LEVEL:
            if self.pellets_eaten >= EVOLUTION_THRESHOLDS[self.level + 1]:
                self.level += 1
                self.color = LEVEL_COLORS[self.level]
                return True
        return False

    def is_stuck(self):
        """Detect if agent is stuck (same position repeatedly)"""
        if len(self.last_positions) < 5:
            return False
        unique_positions = len(set(self.last_positions))
        return unique_positions <= 2

    def get_obs_img(self):
        """Return egocentric RGB image [C,H,W] with enhanced channels"""
        R = OBS_R
        ys = np.clip(np.arange(self.y - R, self.y + R + 1), 0, self.world.h - 1)
        xs = np.clip(np.arange(self.x - R, self.x + R + 1), 0, self.world.w - 1)

        # Extract patches
        F_patch = self.world.F[np.ix_(ys, xs)]
        P_patch = self.world.P[np.ix_(ys, xs)]
        H_patch = self.world.H[np.ix_(ys, xs)]
        W_patch = self.world.W[np.ix_(ys, xs)]

        # Create RGB channels
        # R channel: Pheromones (trails to food)
        Rch = np.clip(P_patch, 0.0, 1.0)

        # G channel: Food (normalized)
        Gch = np.clip(F_patch / MAX_F, 0.0, 1.0)

        # B channel: Multi-purpose
        # - Safe zones (blue)
        # - Walls (dark blue)
        # - Predators (if visible, very bright)
        Bch = np.clip(H_patch, 0.0, 1.0) * 0.7  # Safe zones
        Bch += W_patch * 0.3  # Walls darker

        # Add predator warnings
        for pred in self.world.predators:
            py_rel = pred.y - (self.y - R)
            px_rel = pred.x - (self.x - R)
            if 0 <= py_rel < len(ys) and 0 <= px_rel < len(xs):
                # Predator shows as bright spot
                Rch[py_rel, px_rel] = 1.0
                Bch[py_rel, px_rel] = 0.0
                Gch[py_rel, px_rel] = 0.0

        # Ego-centric marker (center position)
        yy, xx = np.meshgrid(np.arange(CAMERA_SIZE[0]), np.arange(CAMERA_SIZE[1]), indexing="ij")
        gauss = np.exp(-((yy - R) ** 2 + (xx - R) ** 2) / (2 * (R / 3) ** 2))

        # Combine with slight ego glow in green
        Gch = np.maximum(Gch, gauss.astype(np.float32) * 0.3)

        img = np.stack([Rch, Gch, Bch], axis=0).astype(np.float32)

        # Rotate based on facing direction for true egocentric view
        dy, dx = self.facing
        k = 0
        if (dy, dx) == (0, 1):
            k = 1  # right
        elif (dy, dx) == (1, 0):
            k = 2  # down
        elif (dy, dx) == (0, -1):
            k = 3  # left

        img = np.rot90(img, k=k, axes=(1, 2)).copy()
        return img

    def act(self, action_id: int):
        """Move agent with wall avoidance"""
        dy, dx = ACTIONS[action_id]

        # Update facing direction
        if (dy, dx) != (0, 0):
            self.facing = (np.sign(dy) if dy != 0 else 0,
                           np.sign(dx) if dx != 0 else 0)

        # Try to move
        ny = np.clip(self.y + dy, 0, self.world.h - 1)
        nx = np.clip(self.x + dx, 0, self.world.w - 1)

        # Check for walls
        if self.world.W[ny, nx] > 0:
            # Blocked! Stay in place and mark as stuck
            self.stuck_counter += 1
            return False  # Movement failed
        else:
            self.y, self.x = ny, nx
            self.stuck_counter = 0
            self.last_positions.append((self.y, self.x))
            return True  # Movement succeeded

    def emit_pheromone(self, amount=0.08):
        """Leave pheromone trail"""
        self.world.P[self.y, self.x] = np.clip(
            self.world.P[self.y, self.x] + amount, 0.0, 1.0
        )

    def eat(self):
        """Consume food at current position"""
        f = float(self.world.F[self.y, self.x])
        if f <= 0.5:  # Only count significant food
            return 0.0

        take = min(EAT_BITE, f)
        self.world.F[self.y, self.x] = f - take
        gain = EAT_GAIN * take
        self.energy += gain

        # Track pellets for evolution
        if f >= MAX_F * 0.5:  # Was a full pellet
            self.pellets_eaten += 1

        return gain

    def check_predator_threat(self):
        """Check if any predator is nearby and calculate danger"""
        if not self.world.is_day:
            return 0.0  # Predators don't move at night

        danger = 0.0
        for pred in self.world.predators:
            if pred.is_near(self.y, self.x, radius=5):
                dist = abs(pred.y - self.y) + abs(pred.x - self.x)
                danger += (5 - dist) * 0.05  # Closer = more danger
        return danger

    def step(self, action_id: int):
        """Execute action and return reward, alive status"""
        # Store action in history
        self.action_history.append(action_id)

        # Attempt movement
        moved = self.act(action_id)

        # Penalties for bad behavior
        wall_penalty = -0.1 if not moved else 0.0
        stuck_penalty = -0.05 if self.is_stuck() else 0.0

        # Eat food
        food_gain = self.eat()

        # Leave pheromone trail (stronger if found food)
        pheromone_strength = 0.10 if food_gain > 0 else 0.03
        self.emit_pheromone(pheromone_strength)

        # Energy costs
        move_cost = MOVE_COST if moved else 0.0
        base_drain = E_DECAY

        # Night survival mechanics
        night_drain = 0.0
        safe_bonus = 0.0
        if not self.world.is_day:
            # Check if in safe zone
            safety = self.world.H[self.y, self.x]
            if safety > 0.5:
                # In safe zone - minimal drain
                night_drain = NIGHT_ENERGY_DRAIN * (1.0 - SAFE_ZONE_PROTECTION)
                safe_bonus = 0.02  # Small bonus for being smart
            else:
                # Outside safe zone - significant drain
                night_drain = NIGHT_ENERGY_DRAIN * 3.0

        # Predator threat
        predator_danger = self.check_predator_threat()
        predator_penalty = -predator_danger

        # Update energy
        self.energy -= (move_cost + base_drain + night_drain)

        # Calculate reward
        reward = (food_gain - move_cost - base_drain - night_drain +
                  safe_bonus + wall_penalty + stuck_penalty + predator_penalty)

        # Bonus for evolution
        evolved = self.check_evolution()
        if evolved:
            reward += 2.0  # Big reward for leveling up!
            self.score += 10.0

        # Survival bonus (small reward for staying alive)
        reward += 0.001

        # Check if alive
        alive = self.energy > 0.0

        # Store observation in memory buffer
        obs = self.get_obs_img()
        self.memory_buffer.append({
            'obs': obs,
            'action': action_id,
            'reward': reward,
            'energy': self.energy,
            'position': (self.y, self.x)
        })

        return reward, alive


# ---------------------------- Enhanced CNN Policy with Memory ----------------------------
class VisionPolicy(nn.Module):
    def __init__(self, in_ch=3, n_actions=N_ACTIONS, cam_hw=CAMERA_SIZE):
        super().__init__()
        h, w = cam_hw

        # Enhanced CNN with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # LSTM for temporal memory
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, h, w)
            x = self.conv1(dummy)
            x = self.conv2(x)
            x = self.conv3(x)
            self.flat_dim = x.view(1, -1).shape[1]

        self.lstm = nn.LSTM(self.flat_dim, 256, batch_first=True)
        self.lstm_hidden = None

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_actions)
        )

        # Value head (for critic)
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, hidden=None):
        """Forward pass with LSTM memory"""
        # CNN features
        z = self.conv1(x)
        z = self.conv2(z)
        z = self.conv3(z)
        z = z.view(z.size(0), -1)

        # LSTM for sequence modeling
        z = z.unsqueeze(1)  # Add sequence dimension
        if hidden is None:
            lstm_out, hidden = self.lstm(z)
        else:
            lstm_out, hidden = self.lstm(z, hidden)
        lstm_out = lstm_out.squeeze(1)

        # Policy and value outputs
        logits = self.policy_head(lstm_out)
        value = self.value_head(lstm_out)

        return logits, value, hidden

    def reset_hidden(self, batch_size=1):
        """Reset LSTM hidden state"""
        self.lstm_hidden = None


# ---------------------------- Checkpoint utils ----------------------------
def save_checkpoint(path, step, world, agent, policy, optimizer, cfg: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "step": step,
        "world": {
            "h": world.h, "w": world.w,
            "F": world.F, "P": world.P, "H": world.H, "W": world.W,
            "day_progress": world.day_progress, "is_day": world.is_day
        },
        "agent": {
            "y": agent.y, "x": agent.x,
            "energy": float(agent.energy),
            "score": float(agent.score),
            "facing": agent.facing,
            "level": agent.level,
            "pellets_eaten": agent.pellets_eaten
        },
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, policy, optimizer):
    p = torch.load(path, map_location="cpu")
    w = World(h=p["world"]["h"], w=p["world"]["w"])
    w.F = p["world"]["F"].copy()
    w.P = p["world"]["P"].copy()
    w.H = p["world"]["H"].copy()
    w.W = p["world"]["W"].copy()
    w.day_progress = p["world"].get("day_progress", 0)
    w.is_day = p["world"].get("is_day", True)

    a = Agent(w)
    a.y, a.x = p["agent"]["y"], p["agent"]["x"]
    a.energy = p["agent"]["energy"]
    a.score = p["agent"]["score"]
    a.facing = tuple(p["agent"]["facing"])
    a.level = p["agent"].get("level", 0)
    a.pellets_eaten = p["agent"].get("pellets_eaten", 0)
    a.color = LEVEL_COLORS[a.level]

    policy.load_state_dict(p["model"])
    optimizer.load_state_dict(p["optimizer"])

    step = p["step"]
    cfg = p.get("cfg", {})
    return step, w, a, cfg


# ---------------------------- Training with fine-tuning support ----------------------------
def train(steps=10000, render=False, seed=123,
          ckpt_path="checkpoints/vision_last.pt",
          fine_tune_from=None):
    """
    Train with optional fine-tuning from existing checkpoint

    Args:
        fine_tune_from: Path to checkpoint to fine-tune from (or None for fresh training)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    world = World(seed=seed)
    agent = Agent(world)
    policy = VisionPolicy().to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    t = 0

    # Fine-tuning mode
    if fine_tune_from and os.path.exists(fine_tune_from):
        print(f"🔄 Fine-tuning from: {fine_tune_from}")
        t, world, agent, cfg = load_checkpoint(fine_tune_from, policy, optimizer)
        # Reduce learning rate for fine-tuning
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4
        print(f"✓ Loaded checkpoint from step {t}, reduced LR to 1e-4")
    # Resume normal training
    elif os.path.exists(ckpt_path):
        try:
            t, world, agent, _ = load_checkpoint(ckpt_path, policy, optimizer)
            print(f"✓ Resumed training from step {t}")
        except Exception as e:
            print(f"Resume failed: {e}")

    gamma = 0.98
    trajectory = []

    def select_action(obs_tensor, hidden):
        logits, value, new_hidden = policy(obs_tensor, hidden)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        return a.item(), dist.log_prob(a), value, new_hidden

    # SIGINT save
    def handle_sigint(sig, frame):
        print("\n💾 SIGINT -> saving checkpoint...")
        save_checkpoint(ckpt_path, t, world, agent, policy, optimizer,
                        {"lr": optimizer.param_groups[0]['lr'], "gamma": gamma})
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    policy.train()
    hidden_state = None

    print("🚀 Training started!")
    print(f"   Steps: {steps}, Device: {DEVICE}")

    while t < steps:
        t += 1
        world.step_fields()

        obs = agent.get_obs_img()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)

        a, logp, value, hidden_state = select_action(obs_tensor, hidden_state)
        r, alive = agent.step(a)

        trajectory.append({
            'logp': logp,
            'value': value,
            'reward': r
        })

        # Update every N steps or on death
        if (t % 256 == 0) or (not alive):
            if len(trajectory) > 0:
                # Compute returns and advantages
                returns = []
                R = 0.0
                for step_data in reversed(trajectory):
                    R = step_data['reward'] + gamma * R
                    returns.append(R)
                returns = list(reversed(returns))
                returns_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

                # Normalize returns
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

                # Calculate advantages
                values = torch.stack([s['value'].squeeze() for s in trajectory])
                advantages = returns_t - values.detach()

                # Policy loss
                logps = torch.stack([s['logp'] for s in trajectory])
                policy_loss = -(logps * advantages).mean()

                # Value loss
                value_loss = ((values.squeeze() - returns_t) ** 2).mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()

                trajectory.clear()

        # Respawn on death
        if not alive:
            agent = Agent(world)
            hidden_state = None
            policy.reset_hidden()

        # Periodic checkpoint
        if t % 2000 == 0:
            save_checkpoint(ckpt_path, t, world, agent, policy, optimizer,
                            {"lr": optimizer.param_groups[0]['lr'], "gamma": gamma})
            pellets_left = world.count_pellets()
            print(f"[{t:6d}] 💾 Checkpoint | E={agent.energy:.2f} | "
                  f"Score={agent.score:.0f} | Lvl={agent.level} | "
                  f"Pellets={pellets_left} | Day={world.is_day}")

    print(f"\n✅ Training complete! Final: E={agent.energy:.2f}, Score={agent.score:.0f}")
    return policy, world, agent


if __name__ == "__main__":
    render_flag = "--render" in sys.argv
    fine_tune_flag = "--finetune" in sys.argv

    if fine_tune_flag:
        # Fine-tune from last checkpoint
        train(steps=5000, render=render_flag,
              fine_tune_from="checkpoints/vision_last.pt")
    else:
        # Normal training
        train(steps=10000, render=render_flag)