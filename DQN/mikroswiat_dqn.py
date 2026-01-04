import numpy as np
import random
import pygame
import math
import os
import csv
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

try:
    import config_dqn as cfg
except ImportError:
    print("BRAK PLIKU KONFIGURACJI! Upewnij się, że masz plik 'config_dqn.py'.")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")


# --- LOGOWANIE ---
def inicjuj_log_wynikow(sciezka_pliku):
    try:
        with open(sciezka_pliku, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epizod', 'wynik_agentow', 'pop_agentow', 'wynik_predatorow', 'pop_predatorow'])
    except IOError:
        pass


def dopisz_log_wynikow(sciezka_pliku, epizod, rew_a, pop_a, rew_p, pop_p):
    try:
        with open(sciezka_pliku, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epizod, rew_a, pop_a, rew_p, pop_p])
    except IOError:
        pass


# --- MÓZG (DQN) ---
class DQNNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DQNNet, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, input_size, output_size, hidden_layers, lr, mem_size, batch_size, gamma, eps_start, eps_end,
                 eps_decay):
        self.output_size = output_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.policy_net = DQNNet(input_size, hidden_layers, output_size).to(device)
        self.target_net = DQNNet(input_size, hidden_layers, output_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(mem_size)
        self.steps_done = 0

    def select_action(self, state):
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps:
            with torch.no_grad():
                return self.policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)).max(1)[1].view(
                    1, 1)
        else:
            return torch.tensor([[random.randrange(self.output_size)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size: return
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(device)

        q_eval = self.policy_net(state_batch).gather(1, action_batch)
        q_next = self.target_net(next_state_batch).max(1)[0].detach()
        q_target = (q_next * self.gamma * (1.0 - done_batch)) + reward_batch

        loss = F.smooth_l1_loss(q_eval, q_target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# --- HELPER: TOROIDAL DISTANCE ---
def get_toroidal_distance(x1, y1, x2, y2, w, h):
    dx = x2 - x1
    dy = y2 - y1
    if dx > w / 2:
        dx -= w
    elif dx < -w / 2:
        dx += w
    if dy > h / 2:
        dy -= h
    elif dy < -h / 2:
        dy += h
    return dx, dy


# --- ENTITY CLASSES ---
class AgentEntity:
    def __init__(self, w, h, start_energy=None, x=None, y=None):
        self.x = random.randint(0, w - 1) if x is None else x
        self.y = random.randint(0, h - 1) if y is None else y
        self.energy = cfg.START_ENERGY if start_energy is None else start_energy
        self.is_alive = True
        self.age = 0
        self.food_eaten_count = 0
        self.predators_killed = 0
        self.last_action_idx = 4
        self.is_camouflaged = False


class PredatorEntity:
    def __init__(self, w, h, id, start_energy=None, x=None, y=None):
        self.id = id
        self.x = random.randint(0, w - 1) if x is None else x
        self.y = random.randint(0, h - 1) if y is None else y

        # --- PREDATOR BIOLOGY ---
        self.energy = cfg.PREDATOR_START_ENERGY if start_energy is None else start_energy
        self.age = 0
        self.strength = cfg.PREDATOR_BASE_STRENGTH
        self.is_alive = True

        self.last_state = None
        self.last_action = None
        self.last_action_idx = 4
        self.is_sniffing = False


# --- WORLD CLASS ---
class World:
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.agents = []
        self.agents.append(AgentEntity(w, h))

        self.walls = set()
        self.food = set()
        self.mud = set()
        self.scent_grid = np.zeros((w, h))

        self.spawn_walls(cfg.NUM_WALLS)
        self.spawn_mud(cfg.NUM_MUD)
        self.spawn_food(cfg.NUM_FOOD)

        self.predators = []
        # Spawn initial predators
        self.spawn_predators(cfg.PREDATOR_COUNT_START)

    def spawn_food(self, n):
        for _ in range(n):
            while True:
                x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
                if (x, y) not in self.walls and (x, y) not in self.food:
                    self.food.add((x, y));
                    break

    def spawn_walls(self, n):
        for _ in range(n):
            while True:
                x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
                if len(self.agents) > 0:
                    if (x, y) == (self.agents[0].x, self.agents[0].y): continue
                self.walls.add((x, y));
                break

    def spawn_mud(self, n):
        for _ in range(n):
            while True:
                x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
                if (x, y) not in self.walls:
                    self.mud.add((x, y));
                    break

    def spawn_predators(self, n):
        start = 10000 + len(self.predators)  # Unikalne ID
        for i in range(n): self.predators.append(PredatorEntity(self.width, self.height, start + i))

    def update_scents(self):
        self.scent_grid *= cfg.SCENT_DECAY
        for a in self.get_alive_agents():
            amount = cfg.SCENT_ADDED * 0.1 if a.is_camouflaged else cfg.SCENT_ADDED
            self.scent_grid[a.x, a.y] = min(1.0, self.scent_grid[a.x, a.y] + amount)

    def move_entity(self, ent, action):
        if isinstance(ent, AgentEntity): ent.is_camouflaged = False
        if isinstance(ent, PredatorEntity): ent.is_sniffing = False
        ent.last_action_idx = action
        cost = cfg.IDLE_COST
        steps = 1

        if action == 4: return True, cost

        if action == 6:
            if isinstance(ent, AgentEntity):
                ent.is_camouflaged = True
                cost = cfg.CAMO_COST
            else:
                ent.is_sniffing = True
                cost = cfg.SNIFF_COST
            return True, cost

        if action == 5:
            steps = 2
            cost = cfg.DASH_COST
        else:
            cost = cfg.MOVE_COST

        success = True
        for _ in range(steps):
            nx, ny = ent.x, ent.y
            if action == 0 or action == 5:
                ny = (ny - 1) % self.height
            elif action == 1:
                ny = (ny + 1) % self.height
            elif action == 2:
                nx = (nx - 1) % self.width
            elif action == 3:
                nx = (nx + 1) % self.width

            if (nx, ny) in self.walls:
                success = False
                break

            ent.x, ent.y = nx, ny
            if (ent.x, ent.y) in self.mud:
                cost += (cfg.MUD_MOVE_COST - cfg.MOVE_COST)

        return success, cost

    def handle_fight(self, p, a):
        win_chance = 0.5
        if a.food_eaten_count > 0:
            win_chance = a.food_eaten_count / (a.food_eaten_count + p.strength)

        if random.random() < win_chance:
            if a.food_eaten_count >= cfg.KILL_LICENSE_LEVEL:
                p.is_alive = False
                a.food_eaten_count -= cfg.KILL_COST_PELLETS
                a.predators_killed += 1
        else:
            a.is_alive = False
            # PREDATOR GAINS ENERGY FROM KILL
            p.energy += cfg.PREDATOR_EAT_GAIN

    # --- UNIWERSALNE ROZMNAŻANIE (AGENCI I DRAPIEŻNICY) ---
    def handle_reproduction(self):
        # 1. Agenci
        alive_agents = self.get_alive_agents()
        if len(alive_agents) < cfg.MAX_POPULATION:
            new_babies = []
            for a in alive_agents:
                if len(alive_agents) + len(new_babies) >= cfg.MAX_POPULATION: break
                if a.energy >= cfg.REPRODUCTION_THRESHOLD and a.age > 50:
                    a.energy -= cfg.REPRODUCTION_COST
                    off_x, off_y = random.randint(-1, 1), random.randint(-1, 1)
                    child = AgentEntity(self.width, self.height, start_energy=cfg.REPRODUCTION_COST,
                                        x=(a.x + off_x) % self.width, y=(a.y + off_y) % self.height)
                    new_babies.append(child)
            self.agents.extend(new_babies)

        # 2. Drapieżnicy
        alive_preds = self.get_alive_predators()
        if len(alive_preds) < cfg.PREDATOR_MAX_POPULATION:
            new_monsters = []
            for p in alive_preds:
                if len(alive_preds) + len(new_monsters) >= cfg.PREDATOR_MAX_POPULATION: break
                if p.energy >= cfg.PREDATOR_REPRODUCTION_THRESHOLD and p.age > 50:
                    p.energy -= cfg.PREDATOR_REPRODUCTION_COST
                    off_x, off_y = random.randint(-1, 1), random.randint(-1, 1)
                    # Nowe ID
                    new_id = 10000 + len(self.predators) + len(new_monsters) + random.randint(0, 9999)
                    child = PredatorEntity(self.width, self.height, id=new_id,
                                           start_energy=cfg.PREDATOR_REPRODUCTION_COST, x=(p.x + off_x) % self.width,
                                           y=(p.y + off_y) % self.height)
                    new_monsters.append(child)
            self.predators.extend(new_monsters)

    def get_alive_agents(self):
        return [a for a in self.agents if a.is_alive]

    def get_alive_predators(self):
        return [p for p in self.predators if p.is_alive]


# --- LIDAR ---
def get_lidar(ent, world, target_type='wall'):
    dirs = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
    readings = []
    MAX_RANGE = 5.0

    targets = set()
    if target_type == 'wall':
        targets = world.walls
    elif target_type == 'food':
        targets = world.food
    elif target_type == 'predator':
        targets = {(p.x, p.y) for p in world.get_alive_predators() if getattr(ent, 'id', -99) != p.id}
    elif target_type == 'agent':
        if isinstance(ent, PredatorEntity):
            for a in world.get_alive_agents():
                visible = True
                if a.is_camouflaged and not ent.is_sniffing: visible = False
                if visible: targets.add((a.x, a.y))
        else:
            for a in world.get_alive_agents():
                if a != ent: targets.add((a.x, a.y))

    for dx, dy in dirs:
        dist = MAX_RANGE
        for r in range(1, int(MAX_RANGE) + 1):
            cx = (ent.x + dx * r) % world.width
            cy = (ent.y + dy * r) % world.height
            if (cx, cy) in targets:
                dist = float(r)
                break
        readings.append(dist / MAX_RANGE)
    return readings


# --- STATE FUNCTIONS ---
def get_agent_state(agent, world):
    state = [agent.energy / 200.0]

    min_dist = float('inf')
    vec_food = [0, 0]
    for fx, fy in world.food:
        dx, dy = get_toroidal_distance(agent.x, agent.y, fx, fy, world.width, world.height)
        d = dx * dx + dy * dy
        if d < min_dist:
            min_dist = d
            vec_food = [dx / 20.0, dy / 20.0]
    state.extend(vec_food)
    state.extend(get_lidar(agent, world, 'wall'))
    state.extend(get_lidar(agent, world, 'predator'))
    is_mud = 1.0 if (agent.x, agent.y) in world.mud else 0.0
    state.append(is_mud)
    state.append(agent.last_action_idx / 6.0)
    state.append(1.0 if agent.is_camouflaged else 0.0)
    return np.array(state, dtype=np.float32)


def get_predator_state(predator, world):
    # Predator also needs energy awareness!
    state = [predator.energy / 300.0]

    vec_agent = [0, 0]
    min_dist = float('inf')
    alive_agents = world.get_alive_agents()

    for a in alive_agents:
        can_see = True
        if a.is_camouflaged and not predator.is_sniffing: can_see = False

        if can_see:
            dx, dy = get_toroidal_distance(predator.x, predator.y, a.x, a.y, world.width, world.height)
            d = dx * dx + dy * dy
            if d <= cfg.PREDATOR_VISION ** 2 and d < min_dist:
                min_dist = d
                vec_agent = [dx / 20.0, dy / 20.0]

    state.extend(vec_agent)
    state.extend(get_lidar(predator, world, 'wall'))
    scent_val = world.scent_grid[predator.x, predator.y]
    state.append(scent_val)
    is_mud = 1.0 if (predator.x, predator.y) in world.mud else 0.0
    state.append(is_mud)
    state.append(predator.last_action_idx / 6.0)
    state.append(1.0 if predator.is_sniffing else 0.0)
    return np.array(state, dtype=np.float32)


# --- DRAWING ---
def draw_world(screen, world):
    screen.fill(cfg.COLOR_BG)
    for mx, my in world.mud:
        r = pygame.Rect(mx * cfg.CELL_SIZE, my * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(screen, cfg.COLOR_MUD, r)
    for wx, wy in world.walls:
        r = pygame.Rect(wx * cfg.CELL_SIZE, wy * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(screen, cfg.COLOR_WALL, r)
    for fx, fy in world.food:
        r = pygame.Rect(fx * cfg.CELL_SIZE, fy * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(screen, cfg.COLOR_FOOD, r)

    for p in world.get_alive_predators():
        col = (255, 100, 100) if p.is_sniffing else cfg.COLOR_PREDATOR
        r = pygame.Rect(p.x * cfg.CELL_SIZE, p.y * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(screen, col, r)

    for a in world.get_alive_agents():
        col = cfg.COLOR_AGENT_CAMO if a.is_camouflaged else cfg.COLOR_AGENT
        if a.age > cfg.MAX_AGE * 0.8: col = (150, 150, 255)
        r = pygame.Rect(a.x * cfg.CELL_SIZE, a.y * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(screen, col, r)

    pygame.display.flip()


# --- MAIN LOOP ---
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
    pygame.display.set_caption(f"ECOSYSTEM SIM ({device})")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    if not os.path.exists(cfg.MODEL_SAVE_DIR): os.makedirs(cfg.MODEL_SAVE_DIR)
    inicjuj_log_wynikow(cfg.LOG_FILE)

    agent_brain = DQNAgent(cfg.AGENT_INPUT_SIZE, cfg.AGENT_OUTPUT_SIZE, cfg.AGENT_HIDDEN_LAYERS,
                           cfg.AGENT_LEARNING_RATE, cfg.AGENT_MEMORY_SIZE, cfg.AGENT_BATCH_SIZE, cfg.GAMMA,
                           cfg.AGENT_EPS_START, cfg.AGENT_EPS_END, cfg.AGENT_EPS_DECAY)

    # Predator Input +1 because we added Energy to inputs!
    predator_brain = DQNAgent(cfg.PREDATOR_INPUT_SIZE + 1, cfg.PREDATOR_OUTPUT_SIZE, cfg.PREDATOR_HIDDEN_LAYERS,
                              cfg.PREDATOR_LEARNING_RATE, cfg.PREDATOR_MEMORY_SIZE, cfg.PREDATOR_BATCH_SIZE, cfg.GAMMA,
                              cfg.PREDATOR_EPS_START, cfg.PREDATOR_EPS_END, cfg.PREDATOR_EPS_DECAY)

    total_agent_rewards = []
    best_avg = -float('inf')
    patience = 0

    for episode in range(cfg.NUM_EPISODES):
        world = World(cfg.GRID_WIDTH, cfg.GRID_HEIGHT)

        ep_reward_pop_a = 0
        ep_reward_pop_p = 0

        for turn in range(cfg.MAX_TURNS_PER_EPISODE):
            for e in pygame.event.get():
                if e.type == pygame.QUIT: return

            # === AGENT PHASE ===
            alive_agents = world.get_alive_agents()
            if not alive_agents and not world.get_alive_predators(): break  # All dead

            for agent in alive_agents:
                state = get_agent_state(agent, world)
                action_tensor = agent_brain.select_action(state)
                action = action_tensor.item()

                # Guidance setup
                min_food_dist = float('inf')
                for fx, fy in world.food:
                    dx, dy = get_toroidal_distance(agent.x, agent.y, fx, fy, cfg.GRID_WIDTH, cfg.GRID_HEIGHT)
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < min_food_dist: min_food_dist = d

                success, cost = world.move_entity(agent, action)
                reward = -cost
                agent.energy -= cost
                agent.age += 1
                if not success: reward -= cfg.WALL_HIT_PENALTY

                # Guidance calc
                min_food_dist_after = float('inf')
                for fx, fy in world.food:
                    dx, dy = get_toroidal_distance(agent.x, agent.y, fx, fy, cfg.GRID_WIDTH, cfg.GRID_HEIGHT)
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < min_food_dist_after: min_food_dist_after = d
                diff = min_food_dist - min_food_dist_after
                if diff > 0:
                    reward += diff * 3.0
                else:
                    reward += diff * 4.5

                if (agent.x, agent.y) in world.food:
                    agent.energy += cfg.EAT_GAIN
                    agent.food_eaten_count += 1
                    world.food.remove((agent.x, agent.y))
                    world.spawn_food(1)
                    reward += 300

                if agent.age > cfg.MAX_AGE:
                    agent.is_alive = False
                    reward -= 100
                if agent.energy <= 0:
                    agent.is_alive = False
                    reward -= 500

                ep_reward_pop_a += reward
                next_state = get_agent_state(agent, world)
                done = not agent.is_alive
                agent_brain.memory.push(state, action_tensor, reward, next_state, done)

            agent_brain.optimize_model()

            # === PREDATOR PHASE ===
            alive_preds = world.get_alive_predators()
            pred_actions = {}
            for p in alive_preds:
                s_p = get_predator_state(p, world)
                act_p_tens = predator_brain.select_action(s_p)
                p.last_state = s_p
                p.last_action = act_p_tens
                pred_actions[p.id] = act_p_tens.item()

            for p in alive_preds:
                act = pred_actions.get(p.id)
                succ, c = world.move_entity(p, act)
                rew_p = -c  # Cost of living/moving
                p.energy -= c
                p.age += 1

                if not succ: rew_p -= 1.0

                # Interactions
                for a in world.get_alive_agents():
                    if a.x == p.x and a.y == p.y:
                        world.handle_fight(p, a)
                        if not a.is_alive:
                            rew_p += 500

                # Starvation / Old Age
                if p.age > cfg.PREDATOR_MAX_AGE:
                    p.is_alive = False
                    rew_p -= 100
                if p.energy <= 0:
                    p.is_alive = False
                    rew_p -= 500

                ep_reward_pop_p += rew_p
                ns_p = get_predator_state(p, world)
                pd = not p.is_alive
                predator_brain.memory.push(p.last_state, p.last_action, rew_p, ns_p, pd)

            predator_brain.optimize_model()

            # === WORLD UPDATES ===
            world.handle_reproduction()
            world.update_scents()

            # EMERGENCY RESPAWN (Safety net for extinction)
            if cfg.PREDATOR_RESPAWN:
                if len(world.get_alive_predators()) < 2:
                    world.spawn_predators(2)

            # Draw
            draw_world(screen, world)
            pop_a = len(world.get_alive_agents())
            pop_p = len(world.get_alive_predators())
            txt = font.render(f"Ep:{episode + 1} | Ag:{pop_a} Pr:{pop_p}", True, (255, 255, 255))
            screen.blit(txt, (5, 5))
            pygame.display.flip()

        # End Episode
        total_agent_rewards.append(ep_reward_pop_a)
        avg_10 = sum(total_agent_rewards[-10:]) / 10 if len(total_agent_rewards) >= 10 else 0
        print(f"Ep {episode + 1}: Agents={ep_reward_pop_a:.0f} | Preds={ep_reward_pop_p:.0f}")
        dopisz_log_wynikow(cfg.LOG_FILE, episode + 1, ep_reward_pop_a, len(world.get_alive_agents()), ep_reward_pop_p,
                           len(world.get_alive_predators()))

        if episode % 10 == 0:
            agent_brain.target_net.load_state_dict(agent_brain.policy_net.state_dict())
            predator_brain.target_net.load_state_dict(predator_brain.policy_net.state_dict())

        # Early Stopping
        if cfg.EARLY_STOPPING_ENABLED and episode >= cfg.EARLY_STOPPING_MIN_EPISODES:
            curr_avg = sum(total_agent_rewards[-cfg.EARLY_STOPPING_WINDOW:]) / cfg.EARLY_STOPPING_WINDOW
            if curr_avg > best_avg:
                best_avg = curr_avg
                patience = 0
                torch.save(agent_brain.policy_net.state_dict(), os.path.join(cfg.MODEL_SAVE_DIR, "best_ecosystem.pth"))
                print(f"NEW RECORD: {best_avg:.2f}")
            else:
                patience += 1
                if patience >= cfg.EARLY_STOPPING_PATIENCE:
                    print("EARLY STOPPING.");
                    break

        if episode % 50 == 0:
            torch.save(agent_brain.policy_net.state_dict(), os.path.join(cfg.MODEL_SAVE_DIR, f"agent_ep_{episode}.pth"))
            torch.save(predator_brain.policy_net.state_dict(),
                       os.path.join(cfg.MODEL_SAVE_DIR, f"predator_ep_{episode}.pth"))

    pygame.quit()


if __name__ == "__main__":
    run_simulation()