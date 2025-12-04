import numpy as np
import random
import pygame
import math
import os
from CNN import config_cnn as cfg

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DEFINICJA URZĄDZENIA (GPU/CPU) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie do obliczeń sieci neuronowej: {device}")


# ----------------------------------------------------

# --- 2. MÓZG (Sieć Neuronowa) ---
#
# --- WERSJA V26: "Precyzyjny Mózg CNN" (Bez Max Pooling) ---
#
class NeuralNetwork(nn.Module):
    """
    Wersja Precyzyjna (No-Pooling):
    Zachowuje wymiary przestrzenne (7x7) przez warstwy konwolucyjne,
    co pozwala agentowi dokładnie wiedzieć, w której kratce jest jedzenie/wróg.
    """

    def __init__(self, map_size, num_channels, num_actions):
        super(NeuralNetwork, self).__init__()
        self.map_size = map_size  # np. 7 (promień 3 => 3+1+3)

        # 1. Konwolucja (Oczy): Wyciąganie cech bez pomniejszania mapy
        # Padding=1 utrzymuje rozmiar przy kernelu 3x3
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Obliczamy rozmiar po spłaszczeniu (Flatten)
        # Ponieważ nie ma Poolingu, rozmiar to: kanały * szer * wys
        self.flattened_size = 64 * self.map_size * self.map_size

        # 2. Część Decyzyjna (Mózg): Łączy wizję ze stanem (energią)
        # +1 to wejście na informację o energii
        self.fc1 = nn.Linear(self.flattened_size + 1, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, num_actions)

    def forward(self, vision_input, state_input):
        # A. Przetwarzanie obrazu
        x = F.relu(self.conv1(vision_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Spłaszczenie (z 3D na 1D)
        x = torch.flatten(x, 1)

        # B. Dołączenie stanu (energii)
        # state_input musi mieć wymiar (Batch, 1)
        x = torch.cat((x, state_input), dim=1)

        # C. Decyzja
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)

    def get_action(self, vision_input, state_input):
        v_tensor = torch.tensor(vision_input, dtype=torch.float32).unsqueeze(0).to(device)
        s_tensor = torch.tensor(state_input, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            action_logits = self.forward(v_tensor, s_tensor)
            action_probs = F.softmax(action_logits, dim=1)
            return torch.argmax(action_probs).item()


# --- OPERATORY GENETYCZNE (dla PyTorch) ---

def crossover_pytorch(parent1_brain, parent2_brain):
    """(Poprawiona wersja v24.1)"""
    child_brain = NeuralNetwork(
        map_size=cfg.MAX_PERCEPTION_RADIUS * 2 + 1,
        num_channels=3,
        num_actions=5
    )
    child_state = child_brain.state_dict()
    p1_state = {k: v.cpu() for k, v in parent1_brain.state_dict().items()}
    p2_state = {k: v.cpu() for k, v in parent2_brain.state_dict().items()}

    for key in p1_state:
        mask = torch.rand_like(p1_state[key]) > 0.5
        child_state[key].copy_(p1_state[key])
        child_state[key][~mask] = p2_state[key][~mask]

    child_brain.load_state_dict(child_state)
    child_brain.to(device)
    return child_brain


def mutate_pytorch(brain, mutation_rate, mutation_strength):
    # (Bez zmian)
    with torch.no_grad():
        for param in brain.parameters():
            mask = (torch.rand_like(param) < mutation_rate).to(param.device)
            noise = (torch.randn_like(param) * mutation_strength).to(param.device)
            param.add_(noise * mask)
    return brain


# --- 3. AKTORZY SYMULACJI ---

class Agent:
    def __init__(self, world_width, world_height, brain=None):
        self.x = random.randint(0, world_width - 1)
        self.y = random.randint(0, world_height - 1)
        self.energy = cfg.START_ENERGY
        self.is_alive = True
        self.fitness = 0
        self.food_eaten_count = 0
        self.max_energy = cfg.START_ENERGY
        self.current_perception_radius = cfg.START_PERCEPTION_RADIUS

        # --- ZMIANA: Przekazujemy `map_size` do konstruktora ---
        self.map_size = cfg.MAX_PERCEPTION_RADIUS * 2 + 1

        if brain:
            self.brain = brain
        else:
            self.brain = NeuralNetwork(
                map_size=self.map_size,
                num_channels=3,
                num_actions=5
            )
            self.brain.to(device)

    def sense(self, world):
        # (Bez zmian - ta funkcja już działa poprawnie)
        food_map = np.zeros((self.map_size, self.map_size))
        pred_map = np.zeros((self.map_size, self.map_size))
        ally_map = np.zeros((self.map_size, self.map_size))
        alive_agents_list = world.get_alive_agents()
        for dy in range(-cfg.MAX_PERCEPTION_RADIUS, cfg.MAX_PERCEPTION_RADIUS + 1):
            for dx in range(-cfg.MAX_PERCEPTION_RADIUS, cfg.MAX_PERCEPTION_RADIUS + 1):
                is_in_fog_of_war = abs(dx) > self.current_perception_radius or \
                                   abs(dy) > self.current_perception_radius
                if not is_in_fog_of_war:
                    check_x = (self.x + dx) % world.width
                    check_y = (self.y + dy) % world.height
                    map_x = dx + cfg.MAX_PERCEPTION_RADIUS
                    map_y = dy + cfg.MAX_PERCEPTION_RADIUS
                    if (check_x, check_y) in world.food_positions:
                        food_map[map_y, map_x] = 1.0
                    for p in world.get_alive_predators():
                        if p.x == check_x and p.y == check_y:
                            pred_map[map_y, map_x] = 1.0;
                            break
                    for a in alive_agents_list:
                        if a == self: continue
                        if a.x == check_x and a.y == check_y:
                            ally_map[map_y, map_x] = 1.0;
                            break
        vision_map = np.stack([food_map, pred_map, ally_map], axis=0)
        state_data = np.array([self.energy / self.max_energy])
        return vision_map, state_data

    def update(self, world):
        # (Bez zmian)
        if not self.is_alive:
            return
        vision_map, state_data = self.sense(world)
        action = self.brain.get_action(vision_map, state_data)
        self.move(action, world)
        if (self.x, self.y) in world.food_positions:
            self.energy += cfg.EAT_GAIN
            self.food_eaten_count += 1
            self.max_energy += cfg.MAX_ENERGY_GAIN_PER_FOOD
            if self.energy > self.max_energy:
                self.energy = self.max_energy
            world.food_positions.remove((self.x, self.y))
            world.spawn_food(1)
            if self.food_eaten_count in cfg.VISION_UPGRADES:
                self.current_perception_radius = cfg.VISION_UPGRADES[self.food_eaten_count]
        if action == 4:
            self.energy -= cfg.IDLE_COST
        else:
            self.energy -= cfg.MOVE_COST
        if self.energy <= 0:
            self.is_alive = False
        else:
            self.fitness += 1

    def move(self, action_index, world):
        # (Bez zmian)
        if action_index == 0:
            self.y = (self.y - 1) % world.height
        elif action_index == 1:
            self.y = (self.y + 1) % world.height
        elif action_index == 2:
            self.x = (self.x - 1) % world.width
        elif action_index == 3:
            self.x = (self.x + 1) % world.width
        elif action_index == 4:
            pass


class Predator:
    # (Bez zmian)
    def __init__(self, world_width, world_height):
        self.x = random.randint(0, world_width - 1)
        self.y = random.randint(0, world_height - 1)
        self.is_alive = True
        self.strength = cfg.PREDATOR_BASE_STRENGTH

    def update(self, world):
        if not self.is_alive: return
        target_agent = None
        min_dist = float('inf')
        for agent in world.get_alive_agents():
            dx = abs(self.x - agent.x)
            dy = abs(self.y - agent.y)
            dist_x = min(dx, world.width - dx)
            dist_y = min(dy, world.height - dy)
            dist = dist_x + dist_y
            if dist < min_dist and dist <= cfg.PREDATOR_VISION:
                min_dist = dist
                target_agent = agent
        if target_agent:
            if self.x != target_agent.x:
                self.x += 1 if target_agent.x > self.x else -1
            if self.y != target_agent.y:
                self.y += 1 if target_agent.y > self.y else -1
            self.x %= world.width
            self.y %= world.height
        else:
            action = random.randint(0, 4)
            if action == 0:
                self.y = (self.y - 1) % world.height
            elif action == 1:
                self.y = (self.y + 1) % world.height
            elif action == 2:
                self.x = (self.x - 1) % world.width
            elif action == 3:
                self.x = (self.x + 1) % world.width
        for agent in world.get_alive_agents():
            if agent.x == self.x and agent.y == self.y:
                world.handle_fight(self, agent)
                if not self.is_alive:
                    break


# --- 4. OTOCZENIE (Świat) ---

class World:
    # (Bez zmian)
    def __init__(self, width, height, initial_population):
        self.width = width
        self.height = height
        self.agents = initial_population
        self.predators = []
        self.spawn_predators(cfg.PREDATOR_COUNT)
        self.food_positions = set()
        self.spawn_food(cfg.NUM_FOOD)

    def spawn_food(self, amount):
        for _ in range(amount):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.food_positions.add((x, y))

    def spawn_predators(self, amount):
        for _ in range(amount):
            self.predators.append(Predator(self.width, self.height))

    def get_nearby_allies(self, agent_in_fight):
        allies_count = 0
        for agent in self.get_alive_agents():
            if agent == agent_in_fight: continue
            dist = math.sqrt((agent.x - agent_in_fight.x) ** 2 + (agent.y - agent_in_fight.y) ** 2)
            if dist <= cfg.PREDATOR_ALLY_RADIUS:
                allies_count += 1
        return allies_count

    def handle_fight(self, predator, agent):
        allies_count = self.get_nearby_allies(agent)
        group_bonus = allies_count * cfg.PREDATOR_ALLY_BONUS
        agent_strength = agent.food_eaten_count + group_bonus
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
            else:
                pass
        else:
            agent.is_alive = False
            agent.energy = 0

    def step(self):
        for agent in self.get_alive_agents():
            agent.update(self)
        for predator in self.get_alive_predators():
            predator.update(self)
        self.predators = self.get_alive_predators()
        if cfg.PREDATOR_RESPAWN:
            missing_predators = cfg.PREDATOR_COUNT - len(self.predators)
            if missing_predators > 0:
                self.spawn_predators(missing_predators)

    def get_alive_agents(self):
        return [agent for agent in self.agents if agent.is_alive]

    def get_alive_predators(self):
        return [p for p in self.predators if p.is_alive]


# --- 5. FUNKCJA RYSUJĄCA (Pygame) ---

def draw_world(screen, world):
    # (Bez zmian)
    screen.fill(cfg.COLOR_BG)
    for x in range(0, cfg.WINDOW_WIDTH, cfg.CELL_SIZE):
        pygame.draw.line(screen, cfg.COLOR_GRID, (x, 0), (x, cfg.WINDOW_HEIGHT))
    for y in range(0, cfg.WINDOW_HEIGHT, cfg.CELL_SIZE):
        pygame.draw.line(screen, cfg.COLOR_GRID, (0, y), (cfg.WINDOW_WIDTH, y))
    for (x, y) in world.food_positions:
        rect = pygame.Rect(x * cfg.CELL_SIZE, y * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(screen, cfg.COLOR_FOOD, rect)
    for p in world.get_alive_predators():
        rect = pygame.Rect(p.x * cfg.CELL_SIZE, p.y * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(screen, cfg.COLOR_PREDATOR, rect)
    for a in world.get_alive_agents():
        base_blue = 150
        level_bonus = a.food_eaten_count * 10
        blue_val = min(255, base_blue + level_bonus)
        agent_color = (0, blue_val, 255)
        rect = pygame.Rect(a.x * cfg.CELL_SIZE, a.y * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(screen, agent_color, rect)
        vision_radius_px = a.current_perception_radius * cfg.CELL_SIZE
        vision_surface = pygame.Surface((vision_radius_px * 2 + cfg.CELL_SIZE, vision_radius_px * 2 + cfg.CELL_SIZE),
                                        pygame.SRCALPHA)
        vision_surface.fill((0, 100, 255, 30))
        top_left_x = (a.x - a.current_perception_radius) * cfg.CELL_SIZE
        top_left_y = (a.y - a.current_perception_radius) * cfg.CELL_SIZE
        screen.blit(vision_surface, (top_left_x, top_left_y))
    pygame.display.flip()


# --- 6. PĘTLA GŁÓWNA (Algorytm Genetyczny) ---

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
    # --- ZMIANA (v26): Nowy tytuł ---
    pygame.display.set_caption(f"Mikroświat v26 - Precyzyjny CNN (Bez Pool) na {device}")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont(None, 30)
    except Exception:
        font = pygame.font.Font(None, 30)

    if not os.path.exists(cfg.MODEL_SAVE_DIR):
        os.makedirs(cfg.MODEL_SAVE_DIR)
        print(f"Stworzono folder: {cfg.MODEL_SAVE_DIR}")

    print("Start symulacji...")

    current_population = [Agent(cfg.GRID_WIDTH, cfg.GRID_HEIGHT) for _ in range(cfg.POPULATION_SIZE)]

    for gen in range(cfg.NUM_GENERATIONS):
        print(f"\n--- Generacja {gen + 1} / {cfg.NUM_GENERATIONS} ---")

        world = World(cfg.GRID_WIDTH, cfg.GRID_HEIGHT, current_population)

        for turn in range(cfg.MAX_TURNS_PER_GEN):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    print("Symulacja przerwana przez użytkownika.")
                    return
            world.step()
            draw_world(screen, world)

            gen_text = font.render(f"Generacja: {gen + 1}", True, (255, 255, 255))
            turn_text = font.render(f"Tura: {turn}", True, (255, 255, 255))
            alive_text = font.render(f"Żywi Agenci: {len(world.get_alive_agents())}", True, (255, 255, 255))
            screen.blit(gen_text, (5, 5))
            screen.blit(turn_text, (5, 35))
            screen.blit(alive_text, (5, 65))

            pygame.display.flip()
            clock.tick(cfg.FPS)

            if not world.get_alive_agents():
                pygame.time.wait(200)
                break

        evaluated_population = sorted(world.agents,
                                      key=lambda agent: (agent.food_eaten_count * 1000) + agent.fitness,
                                      reverse=True)

        best_agent = evaluated_population[0]
        best_fitness = (best_agent.food_eaten_count * 1000) + best_agent.fitness

        print(
            f"Najlepszy agent: {best_fitness} pkt (Zjadł: {best_agent.food_eaten_count}, Przeżył: {best_agent.fitness} tur)")

        total_fitness = sum((a.food_eaten_count * 1000) + a.fitness for a in evaluated_population)
        avg_fitness = total_fitness / len(evaluated_population)
        print(f"Średni wynik: {avg_fitness:.2f} pkt.")

        best_agent_brain = best_agent.brain
        save_path = os.path.join(cfg.MODEL_SAVE_DIR, f"gen_{gen + 1}_fitness_{best_fitness}_best_model.pth")
        torch.save(best_agent_brain.state_dict(), save_path)

        new_population = []

        for i in range(cfg.ELITISM_COUNT):
            elite_brain = evaluated_population[i].brain
            new_brain = NeuralNetwork(
                map_size=cfg.MAX_PERCEPTION_RADIUS * 2 + 1,
                num_channels=3,
                num_actions=5
            )
            # --- ZMIANA: Musimy przekazać `map_size` do konstruktora ---
            new_brain.map_size = elite_brain.map_size
            new_brain.load_state_dict(elite_brain.state_dict())
            new_brain.to(device)
            new_population.append(Agent(cfg.GRID_WIDTH, cfg.GRID_HEIGHT, brain=new_brain))

        while len(new_population) < cfg.POPULATION_SIZE:
            parent1 = random.choice(evaluated_population[:50])
            parent2 = random.choice(evaluated_population[:50])

            child_brain = crossover_pytorch(parent1.brain, parent2.brain)
            child_brain = mutate_pytorch(child_brain, cfg.MUTATION_RATE, cfg.MUTATION_STRENGTH)

            new_population.append(Agent(cfg.GRID_WIDTH, cfg.GRID_HEIGHT, brain=child_brain))

        current_population = new_population

    pygame.quit()
    print("\nSymulacja zakończona.")


if __name__ == "__main__":
    run_simulation()