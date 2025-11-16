import pygame
import torch
import os
import re
import numpy as np
import config_dqn as cfg  # Upewnij się, że to config od koewolucji (v35+)
from mikroswiat_dqn import DQNNet, AgentEntity, World, draw_world, get_agent_state, get_predator_state, device


def find_available_models(folder):
    """Skanuje folder i zwraca listę dostępnych numerów epizodów."""
    if not os.path.exists(folder):
        print(f"Błąd: Folder '{folder}' nie istnieje.")
        return []

    available_episodes = set()
    for f in os.listdir(folder):
        match = re.search(r"agent_ep_(\d+).pth", f)
        if match:
            # Sprawdź, czy istnieje też pasujący model drapieżnika
            ep_num = int(match.group(1))
            pred_file = os.path.join(folder, f"predator_ep_{ep_num}.pth")
            if os.path.exists(pred_file):
                available_episodes.add(ep_num)

    return sorted(list(available_episodes))


def select_model(available_episodes):
    """Pyta użytkownika, który model wczytać."""
    print("\n--- Dostępne Zapisane Modele (Epizody) ---")
    if not available_episodes:
        print("Brak modeli do wczytania.")
        return None

    print(available_episodes)

    while True:
        try:
            choice = input(f"Wpisz numer epizodu, który chcesz obejrzeć (np. {available_episodes[-1]}): ")
            episode_num = int(choice)
            if episode_num in available_episodes:
                return episode_num
            else:
                print(f"Błąd: Nie znaleziono epizodu {episode_num}. Spróbuj ponownie.")
        except ValueError:
            print("Błąd: Wpisz poprawną liczbę.")


def run_observation():
    pygame.init()
    screen = pygame.display.set_mode((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT + 50))
    pygame.display.set_caption("Obserwatorium Koewolucji")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # 1. Znajdź i pozwól użytkownikowi wybrać model
    available_episodes = find_available_models(cfg.MODEL_SAVE_DIR)
    if not available_episodes:
        print(f"Nie znaleziono zapisanych modeli w folderze: {cfg.MODEL_SAVE_DIR}")
        print("Uruchom najpierw trening (mikroswiat_coevo.py).")
        return

    episode_num = select_model(available_episodes)
    if episode_num is None:
        return

    agent_path = os.path.join(cfg.MODEL_SAVE_DIR, f"agent_ep_{episode_num}.pth")
    pred_path = os.path.join(cfg.MODEL_SAVE_DIR, f"predator_ep_{episode_num}.pth")

    print(f"Wczytywanie modeli z epizodu: {episode_num}...")

    # Inicjalizacja sieci
    agent_brain = DQNNet(input_size=12, hidden_sizes=cfg.AGENT_HIDDEN_LAYERS, output_size=5).to(device)
    predator_brain = DQNNet(input_size=9, hidden_sizes=cfg.PREDATOR_HIDDEN_LAYERS, output_size=5).to(device)

    # Ładowanie wag
    try:
        agent_brain.load_state_dict(torch.load(agent_path, map_location=device))
        predator_brain.load_state_dict(torch.load(pred_path, map_location=device))
        agent_brain.eval()
        predator_brain.eval()
    except Exception as e:
        print(f"Błąd ładowania wag: {e}")
        return

    # 2. Pętla symulacji
    running = True
    while running:
        agent_entity = AgentEntity(cfg.GRID_WIDTH, cfg.GRID_HEIGHT)
        world = World(cfg.GRID_WIDTH, cfg.GRID_HEIGHT, agent_entity)

        episode_over = False
        step = 0

        while not episode_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    episode_over = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        episode_over = True
                    if event.key == pygame.K_r:  # Klawisz R do restartu
                        episode_over = True

                        # --- LOGIKA AGENTA (BEZ LOSOWOŚCI) ---
            state_agent = get_agent_state(agent_entity, world)
            with torch.no_grad():
                state_tensor = torch.tensor(state_agent, dtype=torch.float32).unsqueeze(0).to(device)
                action_agent = agent_brain(state_tensor).max(1)[1].item()

            # --- LOGIKA DRAPIEŻNIKÓW (BEZ LOSOWOŚCI) ---
            predator_actions = {}
            for p in world.get_alive_predators():
                state_pred = get_predator_state(p, world)
                with torch.no_grad():
                    state_tensor = torch.tensor(state_pred, dtype=torch.float32).unsqueeze(0).to(device)
                    action_pred = predator_brain(state_tensor).max(1)[1].item()
                    predator_actions[p.id] = action_pred

            # --- AKTUALIZACJA ŚWIATA ---
            world.move_entity(agent_entity, action_agent)
            if action_agent == 4:
                agent_entity.energy -= cfg.IDLE_COST
            else:
                agent_entity.energy -= cfg.MOVE_COST

            for p in world.get_alive_predators():
                p_act = predator_actions.get(p.id)
                if p_act is not None:
                    world.move_entity(p, p_act)

            if (agent_entity.x, agent_entity.y) in world.food_positions:
                agent_entity.energy += cfg.EAT_GAIN
                agent_entity.food_eaten_count += 1
                agent_entity.max_energy += cfg.MAX_ENERGY_GAIN_PER_FOOD
                world.food_positions.remove((agent_entity.x, agent_entity.y))
                world.spawn_food(1)

            for p in world.get_alive_predators():
                if p.x == agent_entity.x and p.y == agent_entity.y:
                    world.handle_fight(p, agent_entity)
                    if not agent_entity.is_alive: break

            if agent_entity.energy <= 0:
                agent_entity.is_alive = False

            if cfg.PREDATOR_RESPAWN:
                alive_count = len(world.get_alive_predators())
                if alive_count < cfg.PREDATOR_COUNT:
                    world.spawn_predators(cfg.PREDATOR_COUNT - alive_count)
                    world.predators = world.get_alive_predators()

            # --- RYSOWANIE ---
            screen.fill((30, 30, 30))
            surf = pygame.Surface((cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT))
            draw_world(surf, world, agent_entity)
            screen.blit(surf, (0, 0))

            status_text = f"Oglądasz Epizod: {episode_num} | Życie: {agent_entity.energy:.0f} | Zjadł: {agent_entity.food_eaten_count} | Zabił: {agent_entity.predators_killed}"

            text_color = (255, 255, 255)
            if agent_entity.food_eaten_count >= cfg.KILL_LICENSE_LEVEL:
                text_color = (255, 50, 50)
                status_text += " [TRYB ZABÓJCY]"

            help_text = "Naciśnij [R], aby zrestartować epizod."

            img_status = font.render(status_text, True, text_color)
            img_help = font.render(help_text, True, (150, 150, 150))
            screen.blit(img_status, (10, cfg.WINDOW_HEIGHT + 10))
            screen.blit(img_help, (10, cfg.WINDOW_HEIGHT + 30))

            pygame.display.flip()

            clock.tick(10)  # Zwolnione tempo
            step += 1

            if not agent_entity.is_alive:
                print(
                    f"Agent zginął. Wynik: Zjadł {agent_entity.food_eaten_count}, Zabił {agent_entity.predators_killed}")
                pygame.time.wait(1000)
                episode_over = True

    pygame.quit()


if __name__ == "__main__":
    run_observation()