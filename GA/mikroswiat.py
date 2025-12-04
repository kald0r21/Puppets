import numpy as np
import random
import pygame
import os
from GA import config as cfg
import csv  # <-- NOWY IMPORT DLA LOGOWANIA


# --- NOWE FUNKCJE LOGOWANIA (v32+) ---
def inicjuj_log_wynikow(sciezka_pliku):
    """Tworzy nowy plik CSV i zapisuje nagłówek."""
    try:
        with open(sciezka_pliku, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['generacja', 'najlepszy_fitness', 'sredni_fitness'])
    except IOError as e:
        print(f"Błąd przy tworzeniu pliku logu: {e}")


def dopisz_log_wynikow(sciezka_pliku, generacja, najlepszy_fitness, sredni_fitness):
    """Dopisuje nowy wiersz wyników do pliku CSV."""
    try:
        with open(sciezka_pliku, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([generacja, najlepszy_fitness, sredni_fitness])
    except IOError as e:
        print(f"Błąd przy zapisie do pliku logu: {e}")


# ------------------------------------

# --- 2. MÓZG (Sieć Neuronowa) ---
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes_list, output_size):
        self.layers_sizes = [input_size] + hidden_sizes_list + [output_size]
        self.weights = []
        self.biases = []
        for i in range(len(self.layers_sizes) - 1):
            input_dim = self.layers_sizes[i]
            output_dim = self.layers_sizes[i + 1]
            w = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
            b = np.zeros(output_dim)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, inputs):
        x = np.array(inputs, dtype=float)
        for i in range(len(self.weights) - 1):
            x = np.dot(x, self.weights[i]) + self.biases[i]
            x = np.maximum(0, x)
        output = np.dot(x, self.weights[-1]) + self.biases[-1]
        output_activated = 1 / (1 + np.exp(-output))
        return output_activated

    def get_action(self, inputs):
        output_values = self.forward(inputs)
        return np.argmax(output_values)

    @staticmethod
    def _crossover_matrix(p1, p2):
        child = p1.copy()
        mask = np.random.rand(*p1.shape) > 0.5
        child[mask] = p2[mask]
        return child

    @staticmethod
    def crossover(parent1_brain, parent2_brain):
        child_brain = NeuralNetwork(1, [1], 1)
        child_brain.weights = []
        child_brain.biases = []
        child_brain.layers_sizes = parent1_brain.layers_sizes.copy()
        for i in range(len(parent1_brain.weights)):
            w_child = NeuralNetwork._crossover_matrix(parent1_brain.weights[i], parent2_brain.weights[i])
            b_child = NeuralNetwork._crossover_matrix(parent1_brain.biases[i], parent2_brain.biases[i])
            child_brain.weights.append(w_child)
            child_brain.biases.append(b_child)
        return child_brain

    def mutate(self, mutation_rate):
        for i in range(len(self.weights)):
            mask_w = np.random.rand(*self.weights[i].shape) < mutation_rate
            noise_w = np.random.randn(*self.weights[i].shape) * cfg.MUTATION_STRENGTH
            self.weights[i] += mask_w * noise_w
            mask_b = np.random.rand(*self.biases[i].shape) < mutation_rate
            noise_b = np.random.randn(*self.biases[i].shape) * cfg.MUTATION_STRENGTH
            self.biases[i] += mask_b * noise_b


# --- FUNKCJE POMOCNICZE (Dystans Toroidalny) ---
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


# --- 3. AKTORZY SYMULACJI ---

class Agent:
    def __init__(self, world_width, world_height, brain=None):
        self.x = random.randint(0, world_width - 1)
        self.y = random.randint(0, world_height - 1)
        self.energy = cfg.START_ENERGY
        self.is_alive = True
        self.fitness = 0.0
        self.food_eaten_count = 0
        self.max_energy = cfg.START_ENERGY

        self.input_size = 11  # energia, 2x jedzenie, 2x wróg, 2x sojusznik, 2x liczniki, 2x ściana
        output_size = 5

        self.last_dist_to_food_sq = float('inf')

        if brain:
            self.brain = brain
        else:
            self.brain = NeuralNetwork(
                self.input_size,
                cfg.HIDDEN_LAYER_SIZES,
                output_size
            )

    def sense(self, world):
        inputs = [self.energy / 1000.0]
        radius = cfg.SMART_PERCEPTION_RADIUS
        norm_dist = radius * 2

        vec_food = (norm_dist, norm_dist)
        vec_pred = (norm_dist, norm_dist)
        vec_ally = (norm_dist, norm_dist)
        vec_wall = (norm_dist, norm_dist)

        min_dist_food_sq = float('inf')
        min_dist_pred_sq = float('inf')
        min_dist_ally_sq = float('inf')
        min_dist_wall_sq = float('inf')

        count_pred = 0
        count_ally = 0

        for (fx, fy) in world.food_positions:
            dx, dy = get_toroidal_distance(self.x, self.y, fx, fy, world.width, world.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_food_sq and dist_sq <= radius * radius:
                min_dist_food_sq = dist_sq
                vec_food = (dx / radius, dy / radius)

        for p in world.get_alive_predators():
            dx, dy = get_toroidal_distance(self.x, self.y, p.x, p.y, world.width, world.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius * radius:
                count_pred += 1
                if dist_sq < min_dist_pred_sq:
                    min_dist_pred_sq = dist_sq
                    vec_pred = (dx / radius, dy / radius)

        for a in world.get_alive_agents():
            if a == self: continue
            dx, dy = get_toroidal_distance(self.x, self.y, a.x, a.y, world.width, world.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius * radius:
                count_ally += 1
                if dist_sq < min_dist_ally_sq:
                    min_dist_ally_sq = dist_sq
                    vec_ally = (dx / radius, dy / radius)

        for (wx, wy) in world.wall_positions:
            dx, dy = get_toroidal_distance(self.x, self.y, wx, wy, world.width, world.height)
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

        return inputs, min_dist_food_sq

    def update(self, world):
        if not self.is_alive:
            return

        current_inputs, dist_now_sq = self.sense(world)
        action = self.brain.get_action(current_inputs)

        move_successful = self.move(action, world)

        if (self.x, self.y) in world.food_positions:
            self.energy += cfg.EAT_GAIN
            self.food_eaten_count += 1
            self.max_energy += cfg.MAX_ENERGY_GAIN_PER_FOOD
            if self.energy > self.max_energy:
                self.energy = self.max_energy
            world.food_positions.remove((self.x, self.y))
            world.spawn_food(1)

        if action == 4:
            self.energy -= cfg.IDLE_COST
        else:
            self.energy -= cfg.MOVE_COST
            if not move_successful:
                self.energy -= cfg.WALL_HIT_PENALTY

        if self.energy <= 0:
            self.is_alive = False
        else:
            fitness_bonus = 1.0
            if self.last_dist_to_food_sq != float('inf'):
                _, dist_after_move_sq = self.sense(world)
                if dist_after_move_sq != float('inf'):
                    dist_diff = self.last_dist_to_food_sq - dist_after_move_sq
                    fitness_bonus += dist_diff * cfg.REWARD_SHAPING_FACTOR
            self.last_dist_to_food_sq = dist_now_sq
            self.fitness += fitness_bonus

    def move(self, action_index, world):
        if action_index == 4:
            return True

        next_x, next_y = self.x, self.y
        if action_index == 0:
            next_y = (self.y - 1) % world.height
        elif action_index == 1:
            next_y = (self.y + 1) % world.height
        elif action_index == 2:
            next_x = (self.x - 1) % world.width
        elif action_index == 3:
            next_x = (self.x + 1) % world.width

        if (next_x, next_y) not in world.wall_positions:
            self.x = next_x
            self.y = next_y
            return True
        else:
            return False


class Predator:
    def __init__(self, world_width, world_height):
        self.x = random.randint(0, world_width - 1)
        self.y = random.randint(0, world_height - 1)
        self.is_alive = True
        self.strength = cfg.PREDATOR_BASE_STRENGTH

    def update(self, world):
        if not self.is_alive: return
        target_agent = None
        min_dist_sq = float('inf')
        for agent in world.get_alive_agents():
            dx, dy = get_toroidal_distance(self.x, self.y, agent.x, agent.y, world.width, world.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq and dist_sq <= cfg.PREDATOR_VISION ** 2:
                min_dist_sq = dist_sq
                target_agent = agent

        next_x, next_y = self.x, self.y
        if target_agent:
            dx, dy = get_toroidal_distance(self.x, self.y, target_agent.x, target_agent.y, world.width, world.height)
            if abs(dx) > abs(dy):
                next_x = (self.x + np.sign(dx)) % world.width
            else:
                next_y = (self.y + np.sign(dy)) % world.height
        else:
            action = random.randint(0, 4)
            if action == 0:
                next_y = (self.y - 1) % world.height
            elif action == 1:
                next_y = (self.y + 1) % world.height
            elif action == 2:
                next_x = (self.x - 1) % world.width
            elif action == 3:
                next_x = (self.x + 1) % world.width

        if (next_x, next_y) not in world.wall_positions:
            self.x = next_x
            self.y = next_y

        for agent in world.get_alive_agents():
            if agent.x == self.x and agent.y == self.y:
                world.handle_fight(self, agent)
                if not self.is_alive:
                    break


# --- 4. OTOCZENIE (Świat) ---

class World:
    def __init__(self, width, height, initial_population):
        self.width = width
        self.height = height
        self.agents = initial_population
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

    def spawn_predators(self, amount):
        for _ in range(amount):
            self.predators.append(Predator(self.width, self.height))

    def spawn_walls(self, amount):
        for _ in range(amount):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.wall_positions.add((x, y))

    def get_nearby_allies(self, agent_in_fight):
        allies_count = 0
        for agent in self.get_alive_agents():
            if agent == agent_in_fight: continue
            dx, dy = get_toroidal_distance(agent_in_fight.x, agent_in_fight.y, agent.x, agent.y, self.width,
                                           self.height)
            dist_sq = dx * dx + dy * dy
            if dist_sq <= cfg.PREDATOR_ALLY_RADIUS ** 2:
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
    for a in world.get_alive_agents():
        base_blue = 150
        level_bonus = a.food_eaten_count * 10
        blue_val = min(255, base_blue + level_bonus)
        agent_color = (0, blue_val, 255)
        rect = pygame.Rect(a.x * cfg.CELL_SIZE, a.y * cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE)
        pygame.draw.rect(screen, agent_color, rect)
    pygame.display.flip()


# --- 6. PĘTLA GŁÓWNA (Algorytm Genetyczny) ---

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
    pygame.display.set_caption(f"Mikroświat ({cfg.MODEL_SAVE_DIR})")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont(None, 30)
    except Exception:
        font = pygame.font.Font(None, 30)

    if not os.path.exists(cfg.MODEL_SAVE_DIR):
        os.makedirs(cfg.MODEL_SAVE_DIR)
        print(f"Stworzono folder: {cfg.MODEL_SAVE_DIR}")

    print("Start symulacji...")

    # --- DODANE LOGOWANIE v32+ ---
    sciezka_logu = f"log_wynikow_{cfg.MODEL_SAVE_DIR}.csv"
    inicjuj_log_wynikow(sciezka_logu)
    print(f"Log wyników będzie zapisywany w: {sciezka_logu}")
    # -----------------------------

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
        best_fitness = int((best_agent.food_eaten_count * 1000) + best_agent.fitness)

        print(
            f"Najlepszy agent: {best_fitness} pkt (Zjadł: {best_agent.food_eaten_count}, Przeżył: {best_agent.fitness:.0f} tur)")

        total_fitness = sum((a.food_eaten_count * 1000) + a.fitness for a in evaluated_population)
        avg_fitness = total_fitness / len(evaluated_population)
        print(f"Średni wynik: {avg_fitness:.2f} pkt.")

        # --- DODANE LOGOWANIE v32+ ---
        dopisz_log_wynikow(sciezka_logu, gen + 1, best_fitness, avg_fitness)
        # -----------------------------

        best_agent_brain = best_agent.brain
        save_path = os.path.join(cfg.MODEL_SAVE_DIR, f"gen_{gen + 1}_fitness_{best_fitness}_best_model.npz")

        save_dict = {}
        for i, w in enumerate(best_agent_brain.weights):
            save_dict[f'W{i}'] = w
        for i, b in enumerate(best_agent_brain.biases):
            save_dict[f'b{i}'] = b
        np.savez(save_path, **save_dict)

        new_population = []
        for i in range(cfg.ELITISM_COUNT):
            new_population.append(Agent(cfg.GRID_WIDTH, cfg.GRID_HEIGHT, brain=evaluated_population[i].brain))

        while len(new_population) < cfg.POPULATION_SIZE:
            parent1 = random.choice(evaluated_population[:50])
            parent2 = random.choice(evaluated_population[:50])
            child_brain = NeuralNetwork.crossover(parent1.brain, parent2.brain)
            child_brain.mutate(cfg.MUTATION_RATE)
            new_population.append(Agent(cfg.GRID_WIDTH, cfg.GRID_HEIGHT, brain=child_brain))

        current_population = new_population

    pygame.quit()
    print("\nSymulacja zakończona.")


if __name__ == "__main__":
    run_simulation()