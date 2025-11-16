import numpy as np
import random
import pygame
import math
import os
import config_dqn as cfg  # <-- Importujemy config v36
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque

# --- Sprawdzenie GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie do obliczeń DQN: {device}")


# --- FUNKCJE LOGOWANIA ---
def inicjuj_log_wynikow(sciezka_pliku):
    try:
        with open(sciezka_pliku, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epizod', 'wynik', 'sredni_wynik_10_ost', 'zjedzone', 'zabite'])
    except IOError as e:
        print(f"Błąd przy tworzeniu pliku logu: {e}")


def dopisz_log_wynikow(sciezka_pliku, epizod, wynik, sredni_wynik, zjedzone, zabite):
    try:
        with open(sciezka_pliku, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epizod, wynik, sredni_wynik, zjedzone, zabite])
    except IOError as e:
        print(f"Błąd przy zapisie do pliku logu: {e}")


# --- 1. MÓZG (Sieć Q) ---
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


# --- 2. PAMIĘĆ (Replay Memory) ---
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


# --- 3. GŁÓWNY AGENT/TRENER ---
class DQNAgent:
    def __init__(self, input_size, output_size, hidden_layers, lr, mem_size, batch_size, gamma, eps_start, eps_end,
                 eps_decay):
        self.input_size = input_size
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
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if random.random() > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.output_size)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(device)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma * (1.0 - done_batch)) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# --- 4. ŚWIAT I AKTORZY ---

def get_toroidal_distance(x1, y1, x2, y2, w, h):
    dx = x2 - x1
    dy = y2 - y1
    if dx > w / 2:
        dx = dx - w
    elif dx < -w / 2:
        dx = dx + w
    if dy > h / 2:
        dy = dy - h
    elif dy < -h / 2:
        dy = dy + h
    return dx, dy


class AgentEntity:
    def __init__(self, world_width, world_height):
        self.x = random.randint(0, world_width - 1)
        self.y = random.randint(0, world_height - 1)
        self.energy = cfg.START_ENERGY
        self.is_alive = True
        self.food_eaten_count = 0
        self.max_energy = cfg.START_ENERGY
        self.predators_killed = 0


class PredatorEntity:
    def __init__(self, world_width, world_height, id):
        self.id = id
        self.x = random.randint(0, world_width - 1)
        self.y = random.randint(0, world_height - 1)
        self.is_alive = True
        self.strength = cfg.PREDATOR_BASE_STRENGTH
        self.last_state = None
        self.last_action = None


class World:
    def __init__(self, width, height, agent_entity):
        self.width = width
        self.height = height
        self.agent_entity = agent_entity
        self.wall_positions = set()
        self.food_positions = set()
        self.spawn_walls(cfg.NUM_WALLS)
        self.spawn_food(cfg.NUM_FOOD)
        self.predators = []
        self.spawn_predators(cfg.PREDATOR_COUNT)

    def spawn_food(self, amount):
        for _ in range(amount):
            while True:
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                if (x, y) not in self.wall_positions and (x, y) not in self.food_positions:
                    self.food_positions.add((x, y))
                    break

    def spawn_walls(self, amount):
        for _ in range(amount):
            while True:
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                if x != self.agent_entity.x or y != self.agent_entity.y:
                    self.wall_positions.add((x, y))
                    break

    def spawn_predators(self, amount):
        start_id = len(self.predators)
        for i in range(amount):
            self.predators.append(PredatorEntity(self.width, self.height, id=start_id + i))

    def get_alive_predators(self):
        return [p for p in self.predators if p.is_alive]

    def get_nearby_allies(self, agent_in_fight):
        return 0

    def handle_fight(self, predator, agent):
        agent_strength = agent.food_eaten_count
        predator_strength = predator.strength
        total_strength = agent_strength + predator_strength
        if total_strength == 0:
            win_chance = 0.5
        else:
            win_chance = agent_strength / total_strength

        if random.random() < win_chance:
            if agent.food_eaten_count >= cfg.KILL_LICENSE_LEVEL:
                predator.is_alive = False
                agent.food_eaten_count -= cfg.KILL_COST_PELLETS
                agent.predators_killed += 1
        else:
            agent.is_alive = False
            agent.energy = 0

    def move_entity(self, entity, action):
        next_x, next_y = entity.x, entity.y
        if action == 0:
            next_y = (entity.y - 1) % self.height
        elif action == 1:
            next_y = (entity.y + 1) % self.height
        elif action == 2:
            next_x = (entity.x - 1) % self.width
        elif action == 3:
            next_x = (entity.x + 1) % self.width
        elif action == 4:
            pass

        if (next_x, next_y) in self.wall_positions:
            return False
        else:
            entity.x, entity.y = next_x, next_y
            return True


# --- 5. FUNKCJE SENSE (STANU) ---
def get_agent_state(agent, world):
    inputs = [agent.energy / 1000.0]
    radius = cfg.SMART_PERCEPTION_RADIUS
    norm_dist = radius * 2
    vec_food = (norm_dist, norm_dist);
    vec_pred = (norm_dist, norm_dist)
    vec_ally = (norm_dist, norm_dist);
    vec_wall = (norm_dist, norm_dist)
    min_dist_food_sq = float('inf');
    min_dist_pred_sq = float('inf')
    min_dist_ally_sq = float('inf');
    min_dist_wall_sq = float('inf')
    count_pred = 0;
    count_ally = 0

    for (fx, fy) in world.food_positions:
        dx, dy = get_toroidal_distance(agent.x, agent.y, fx, fy, world.width, world.height)
        dist_sq = dx * dx + dy * dy
        if dist_sq < min_dist_food_sq and dist_sq <= radius * radius:
            min_dist_food_sq = dist_sq;
            vec_food = (dx / radius, dy / radius)
    for p in world.get_alive_predators():
        dx, dy = get_toroidal_distance(agent.x, agent.y, p.x, p.y, world.width, world.height)
        dist_sq = dx * dx + dy * dy
        if dist_sq <= radius * radius:
            count_pred += 1
            if dist_sq < min_dist_pred_sq:
                min_dist_pred_sq = dist_sq;
                vec_pred = (dx / radius, dy / radius)

    for (wx, wy) in world.wall_positions:
        dx, dy = get_toroidal_distance(agent.x, agent.y, wx, wy, world.width, world.height)
        dist_sq = dx * dx + dy * dy
        if dist_sq < min_dist_wall_sq and dist_sq <= radius * radius:
            min_dist_wall_sq = dist_sq;
            vec_wall = (dx / radius, dy / radius)

    inputs.extend(vec_food);
    inputs.extend(vec_pred);
    inputs.extend(vec_ally)
    inputs.append(count_pred / 5.0);
    inputs.append(count_ally / 5.0);
    inputs.extend(vec_wall)
    inputs.append(agent.predators_killed / 10.0)
    return np.array(inputs)


def get_predator_state(predator, world):
    agent = world.agent_entity
    radius = cfg.PREDATOR_VISION
    norm_dist = radius * 2
    vec_agent = (norm_dist, norm_dist);
    vec_pred = (norm_dist, norm_dist)
    vec_wall = (norm_dist, norm_dist)
    min_dist_agent_sq = float('inf');
    min_dist_pred_sq = float('inf')
    min_dist_wall_sq = float('inf')
    count_pred = 0

    if agent.is_alive:
        dx, dy = get_toroidal_distance(predator.x, predator.y, agent.x, agent.y, world.width, world.height)
        dist_sq = dx * dx + dy * dy
        if dist_sq <= radius * radius:
            min_dist_agent_sq = dist_sq
            vec_agent = (dx / radius, dy / radius)

    for p in world.get_alive_predators():
        if p.id == predator.id: continue
        dx, dy = get_toroidal_distance(predator.x, predator.y, p.x, p.y, world.width, world.height)
        dist_sq = dx * dx + dy * dy
        if dist_sq <= radius * radius:
            count_pred += 1
            if dist_sq < min_dist_pred_sq:
                min_dist_pred_sq = dist_sq
                vec_pred = (dx / radius, dy / radius)

    for (wx, wy) in world.wall_positions:
        dx, dy = get_toroidal_distance(predator.x, predator.y, wx, wy, world.width, world.height)
        dist_sq = dx * dx + dy * dy
        if dist_sq < min_dist_wall_sq and dist_sq <= radius * radius:
            min_dist_wall_sq = dist_sq
            vec_wall = (dx / radius, dy / radius)

    is_agent_vulnerable = 1.0 if agent.food_eaten_count < cfg.KILL_LICENSE_LEVEL else 0.0

    inputs = []
    inputs.extend(vec_agent);
    inputs.extend(vec_pred);
    inputs.extend(vec_wall)
    inputs.append(count_pred / 5.0);
    inputs.append(is_agent_vulnerable)
    inputs.append(agent.food_eaten_count / 10.0)

    return np.array(inputs)


# --- 6. FUNKCJA RYSUJĄCA (Pygame) ---
def draw_world(screen, world, agent):
    screen.fill(cfg.COLOR_BG)
    for x in range(0, cfg.WINDOW_WIDTH, cfg.CELL_SIZE):
        pygame.draw.line(screen, cfg.COLOR_GRID, (x, 0), (x, cfg.WINDOW_HEIGHT))
    for y in range(0, cfg.WINDOW_HEIGHT, cfg.CELL_SIZE):
        pygame.draw.line(screen, cfg.COLOR_GRID, (0, y), (cfg.WINDOW_WIDTH, y))
    for (x, y) in world.wall_positions:
        rect = pygame.Rect(x * cfg.CELL_SIZE, y * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(screen, cfg.COLOR_WALL, rect)
    for (x, y) in world.food_positions:
        rect = pygame.Rect(x * cfg.CELL_SIZE, y * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(screen, cfg.COLOR_FOOD, rect)
    for p in world.get_alive_predators():
        rect = pygame.Rect(p.x * cfg.CELL_SIZE, p.y * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(screen, cfg.COLOR_PREDATOR, rect)
    if agent.is_alive:
        base_blue = 150
        level_bonus = agent.food_eaten_count * 10
        blue_val = min(255, base_blue + level_bonus)
        agent_color = (0, blue_val, 255)
        rect = pygame.Rect(agent.x * cfg.CELL_SIZE, agent.y * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(screen, agent_color, rect)
    pygame.display.flip()


# --- 7. PĘTLA GŁÓWNA (Trener Koewolucji) ---
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
    pygame.display.set_caption(f"Mikroświat v36 - Silne Nagrody ({device})")
    clock = pygame.time.Clock()
    try:
        font = pygame.font.SysFont(None, 30)
    except Exception:
        font = pygame.font.Font(None, 30)

    if not os.path.exists(cfg.MODEL_SAVE_DIR):
        os.makedirs(cfg.MODEL_SAVE_DIR)
        print(f"Stworzono folder: {cfg.MODEL_SAVE_DIR}")

    sciezka_logu = cfg.LOG_FILE
    inicjuj_log_wynikow(sciezka_logu)
    print(f"Log wyników będzie zapisywany w: {sciezka_logu}")

    agent_brain = DQNAgent(
        input_size=12, output_size=5, hidden_layers=cfg.AGENT_HIDDEN_LAYERS,
        lr=cfg.AGENT_LEARNING_RATE, mem_size=cfg.AGENT_MEMORY_SIZE,
        batch_size=cfg.AGENT_BATCH_SIZE, gamma=cfg.GAMMA,
        eps_start=cfg.AGENT_EPS_START, eps_end=cfg.AGENT_EPS_END, eps_decay=cfg.AGENT_EPS_DECAY
    )
    predator_brain = DQNAgent(
        input_size=9, output_size=5, hidden_layers=cfg.PREDATOR_HIDDEN_LAYERS,
        lr=cfg.PREDATOR_LEARNING_RATE, mem_size=cfg.PREDATOR_MEMORY_SIZE,
        batch_size=cfg.PREDATOR_BATCH_SIZE, gamma=cfg.GAMMA,
        eps_start=cfg.PREDATOR_EPS_START, eps_end=cfg.PREDATOR_EPS_END, eps_decay=cfg.PREDATOR_EPS_DECAY
    )

    print("Start symulacji...")
    total_agent_rewards = []
    total_predator_rewards = []

    for episode in range(cfg.NUM_EPISODES):
        agent_entity = AgentEntity(cfg.GRID_WIDTH, cfg.GRID_HEIGHT)
        world = World(cfg.GRID_WIDTH, cfg.GRID_HEIGHT, agent_entity)

        agent_state = get_agent_state(agent_entity, world)

        episode_agent_reward = 0.0
        episode_predator_reward = 0.0

        for turn in range(cfg.MAX_TURNS_PER_EPISODE):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit();
                    print("Symulacja przerwana.");
                    return

            agent_action_tensor = agent_brain.select_action(agent_state)
            agent_action = agent_action_tensor.item()

            predator_actions = {}
            for p in world.get_alive_predators():
                p_state = get_predator_state(p, world)
                p_action_tensor = predator_brain.select_action(p_state)
                p.last_state = p_state
                p.last_action = p_action_tensor
                predator_actions[p.id] = p_action_tensor.item()

            # --- AKTUALIZACJA ŚWIATA ---
            reward_agent = 0.0
            rewards_predator = {p.id: 0.0 for p in world.get_alive_predators()}

            dist_to_food_before = np.linalg.norm(agent_state[1:3])

            move_ok = world.move_entity(agent_entity, agent_action)
            if agent_action == 4:
                agent_entity.energy -= cfg.IDLE_COST
                reward_agent -= cfg.IDLE_COST
            else:
                agent_entity.energy -= cfg.MOVE_COST
                reward_agent -= cfg.MOVE_COST
                if not move_ok:
                    reward_agent -= cfg.WALL_HIT_PENALTY

            for p in world.get_alive_predators():
                p_action = predator_actions.get(p.id)
                if p_action is not None:
                    dist_to_agent_before = np.linalg.norm(p.last_state[0:2])
                    move_ok = world.move_entity(p, p_action)
                    p_next_state_temp = get_predator_state(p, world)
                    dist_to_agent_after = np.linalg.norm(p_next_state_temp[0:2])
                    if dist_to_agent_after < dist_to_agent_before:
                        rewards_predator[p.id] += 0.5
                    if p_action == 4:
                        rewards_predator[p.id] -= 0.5
                    else:
                        rewards_predator[p.id] -= 0.1
                    if not move_ok: rewards_predator[p.id] -= 1.0

            if (agent_entity.x, agent_entity.y) in world.food_positions:
                agent_entity.energy += cfg.EAT_GAIN;
                agent_entity.food_eaten_count += 1
                agent_entity.max_energy += cfg.MAX_ENERGY_GAIN_PER_FOOD
                world.food_positions.remove((agent_entity.x, agent_entity.y))
                world.spawn_food(1)
                reward_agent += 500  # <-- ZMIANA v36: Silna nagroda

            agent_was_alive = agent_entity.is_alive

            for p in world.get_alive_predators():
                if p.x == agent_entity.x and p.y == agent_entity.y:
                    predator_was_alive = p.is_alive
                    world.handle_fight(p, agent_entity)

                    if not agent_entity.is_alive:
                        rewards_predator[p.id] += 500  # <-- ZMIANA v36: Silna nagroda
                        break
                    if not p.is_alive and predator_was_alive:
                        reward_agent += 300  # <-- ZMIANA v36: Silna nagroda
                        rewards_predator[p.id] -= 300  # <-- ZMIANA v36: Silna kara

            if agent_entity.energy <= 0:
                agent_entity.is_alive = False

            done = not agent_entity.is_alive
            if done and agent_was_alive:
                reward_agent -= 1000  # <-- ZMIANA v36: Silna kara

            agent_next_state = get_agent_state(agent_entity, world)
            dist_to_food_after = np.linalg.norm(agent_next_state[1:3])
            if dist_to_food_after < dist_to_food_before:
                reward_agent += 1.0

            for p in world.get_alive_predators():
                if p.last_state is not None:
                    p_next_state = get_predator_state(p, world)
                    p_done = not p.is_alive or done
                    predator_brain.memory.push(p.last_state, p.last_action, rewards_predator[p.id], p_next_state,
                                               p_done)
                    episode_predator_reward += rewards_predator[p.id]

            if cfg.PREDATOR_RESPAWN:
                alive_predators = world.get_alive_predators()
                missing = cfg.PREDATOR_COUNT - len(alive_predators)
                if missing > 0:
                    world.spawn_predators(missing)
                world.predators = world.get_alive_predators()

            agent_brain.memory.push(agent_state, agent_action_tensor, reward_agent, agent_next_state, done)
            episode_agent_reward += reward_agent

            state = agent_next_state

            agent_brain.optimize_model()
            predator_brain.optimize_model()

            draw_world(screen, world, agent_entity)
            gen_text = font.render(f"Epizod: {episode + 1}/{cfg.NUM_EPISODES}", True, (255, 255, 255))
            reward_text = font.render(f"Wynik Agenta: {episode_agent_reward:.0f}", True, (255, 255, 255))
            kills_text = font.render(f"Agent zabił: {agent_entity.predators_killed}", True, (255, 255, 255))
            screen.blit(gen_text, (5, 5))
            screen.blit(reward_text, (5, 35))
            screen.blit(kills_text, (5, 65))
            pygame.display.flip()
            clock.tick(cfg.FPS)

            if done:
                break

        # Koniec epizodu
        total_agent_rewards.append(episode_agent_reward)
        if len(world.predators) > 0:
            total_predator_rewards.append(episode_predator_reward / len(world.predators))
        else:
            total_predator_rewards.append(episode_predator_reward)

        avg_agent_reward = 0.0
        avg_pred_reward = 0.0

        if episode >= 10:
            avg_agent_reward = sum(total_agent_rewards[-10:]) / 10
            avg_pred_reward = sum(total_predator_rewards[-10:]) / 10
            print(
                f"Epizod {episode + 1}: Zakończony. Agent: {episode_agent_reward:.0f} (Śr: {avg_agent_reward:.2f}) | Drapieżnicy: {episode_predator_reward:.0f} (Śr: {avg_pred_reward:.2f})")
        else:
            print(
                f"Epizod {episode + 1}: Zakończony. Agent: {episode_agent_reward:.0f} | Drapieżnicy: {episode_predator_reward:.0f}")

        dopisz_log_wynikow(sciezka_logu,
                           episode + 1,
                           episode_agent_reward, avg_agent_reward,
                           agent_entity.food_eaten_count, agent_entity.predators_killed)

        if episode % 10 == 0:
            agent_brain.target_net.load_state_dict(agent_brain.policy_net.state_dict())
            predator_brain.target_net.load_state_dict(predator_brain.policy_net.state_dict())

        if episode % 100 == 0 and episode > 0:
            torch.save(agent_brain.policy_net.state_dict(), os.path.join(cfg.MODEL_SAVE_DIR, f"agent_ep_{episode}.pth"))
            torch.save(predator_brain.policy_net.state_dict(),
                       os.path.join(cfg.MODEL_SAVE_DIR, f"predator_ep_{episode}.pth"))
            print("Zapisano modele.")

    pygame.quit()
    print("\nSymulacja zakończona.")


if __name__ == "__main__":
    run_simulation()