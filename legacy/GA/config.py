# config_smart.py

# --- 1. USTAWIENIA ŚWIATA ---
GRID_WIDTH = 40
GRID_HEIGHT = 40
NUM_FOOD = 45
NUM_WALLS = 50  # Bloki ścian

# --- 2. USTAWIENIA AGENTA ---
START_ENERGY = 100
EAT_GAIN = 150
MAX_ENERGY_GAIN_PER_FOOD = 10
MOVE_COST = 1           # Bazowy koszt ruchu
IDLE_COST = 3           # Stanie jest BARDZO kosztowne
WALL_HIT_PENALTY = 3  # Dodatkowa kara za uderzenie w ścianę

# --- Ustawienia "Mózgu" v32 (Kara za ścianę) ---
HIDDEN_LAYER_SIZES = [16, 16] # 2 warstwy ukryte
SMART_PERCEPTION_RADIUS = 10  # Zasięg "zmysłów" agenta
REWARD_SHAPING_FACTOR = 0.1 # Bonus za zbliżenie się do jedzenia

# --- 3. USTAWIENIA PREDATORA ---
PREDATOR_COUNT = 5
PREDATOR_RESPAWN = True     # Drapieżnicy się odradzają
PREDATOR_VISION = 10
PREDATOR_BASE_STRENGTH = 5
PREDATOR_ALLY_BONUS = 2
PREDATOR_ALLY_RADIUS = 3

# --- Mechanika Kosztu Zabójstwa ---
KILL_LICENSE_LEVEL = 5
KILL_COST_PELLETS = 1

# --- 4. USTAWIENIA EWOLUCJI (GENETIC ALGORITHM) ---
POPULATION_SIZE = 100
NUM_GENERATIONS = 200 # Dłuższy trening
MAX_TURNS_PER_GEN = 1000
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.5
ELITISM_COUNT = 10

# --- 5. USTAWIENIA ZAPISU ---
MODEL_SAVE_DIR = "../best_models_smart_v32_wall_penalty"

# --- 6. USTAWIENIA WIZUALIZACJI ---
CELL_SIZE = 14
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 240 # Zwiększ, by przyspieszyć trening (np. 60)

# --- 7. KOLORY (RGB) ---
COLOR_BG = (20, 20, 20)
COLOR_FOOD = (0, 255, 0)
COLOR_AGENT = (0, 150, 255)
COLOR_PREDATOR = (255, 0, 0)
COLOR_GRID = (40, 40, 40)
COLOR_WALL = (100, 100, 100)