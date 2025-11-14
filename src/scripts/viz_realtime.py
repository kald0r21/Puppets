# viz_realtime.py
# Enhanced realtime visualization with stat bars, predator display, and improved observation window

import argparse, os, numpy as np

try:
    import pygame
except Exception as e:
    raise SystemExit("pygame required: pip install pygame") from e

from src.scripts.micro_world_vision import World, Agent, ACTIONS, LEVEL_COLORS, MAX_LEVEL
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO, MaskablePPO


def build_obs_hwc(agent: Agent):
    img_chw = agent.get_obs_img().astype(np.float32)
    img_hwc = np.clip(img_chw, 0.0, 1.0).transpose(1, 2, 0)
    return (img_hwc * 255).astype(np.uint8)


def get_action_mask(agent: Agent, world: World):
    """Return boolean mask of valid actions"""
    y, x = agent.y, agent.x
    mask = np.ones(len(ACTIONS), dtype=bool)
    for i, (dy, dx) in enumerate(ACTIONS):
        ny, nx = y + dy, x + dx
        if not (0 <= ny < world.h and 0 <= nx < world.w):
            mask[i] = False
        elif world.W[ny, nx] > 0:
            mask[i] = False
    return mask


KEY_TO_ACTION = {
    pygame.K_KP5: 0, pygame.K_s: 0,
    pygame.K_UP: 1, pygame.K_w: 1,
    pygame.K_DOWN: 2, pygame.K_x: 2,
    pygame.K_RIGHT: 3, pygame.K_d: 3,
    pygame.K_LEFT: 4, pygame.K_a: 4,
    pygame.K_KP9: 5, pygame.K_e: 5,
    pygame.K_KP7: 6, pygame.K_q: 6,
    pygame.K_KP1: 7, pygame.K_z: 7,
    pygame.K_KP3: 8, pygame.K_c: 8
}


class Renderer:
    def __init__(self, cell_px=6):
        self.cell_px = cell_px
        self.screen = None
        self.font = None
        self.font_small = None
        self.font_large = None
        self.clock = None

    def init(self, w, h, title="MicroWorld Enhanced Viewer"):
        pygame.init()
        self.clock = pygame.time.Clock()

        # Layout: game area + right panel + bottom panel
        self.game_width = w * self.cell_px
        self.game_height = h * self.cell_px
        self.panel_width = 400
        self.bottom_height = 100

        screen_w = self.game_width + self.panel_width
        screen_h = self.game_height + self.bottom_height

        self.screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption(title)

        self.font = pygame.font.SysFont("monospace", 14, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 11)
        self.font_large = pygame.font.SysFont("monospace", 16, bold=True)

    def draw_stat_bar(self, surface, x, y, width, height, value, max_value,
                      color, bg_color=(40, 40, 40), label="", show_text=True):
        """Draw a stat bar with label"""
        # Background
        pygame.draw.rect(surface, bg_color, (x, y, width, height))

        # Filled portion
        fill_width = int((value / max_value) * width)
        if fill_width > 0:
            pygame.draw.rect(surface, color, (x, y, fill_width, height))

        # Border
        pygame.draw.rect(surface, (200, 200, 200), (x, y, width, height), 2)

        # Label
        if label and show_text:
            text = self.font_small.render(label, True, (255, 255, 255))
            surface.blit(text, (x, y - 18))

        # Value text
        if show_text:
            value_text = f"{value:.2f}/{max_value:.2f}"
            text = self.font_small.render(value_text, True, (255, 255, 255))
            text_rect = text.get_rect(center=(x + width // 2, y + height // 2))
            surface.blit(text, text_rect)

    def draw(self, world: World, agent: Agent, obs_hwc=None,
             info_text="", fps=0, model_type="Manual"):
        surf = self.screen
        cp = self.cell_px
        h, w = world.h, world.w

        # Background based on day/night
        if world.is_day:
            bg_color = (20, 25, 35)
            panel_bg = (30, 35, 45)
        else:
            bg_color = (5, 5, 15)
            panel_bg = (15, 15, 25)

        surf.fill(bg_color)

        # === GAME AREA ===
        game_surface = pygame.Surface((self.game_width, self.game_height))
        game_surface.fill(bg_color)

        # Render world
        F = np.clip(world.F / 2.0, 0.0, 1.0)  # Food (normalized)
        P = np.clip(world.P, 0.0, 1.0) ** 0.7  # Pheromones
        H = np.clip(world.H, 0.0, 1.0)  # Safe zones

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 1] = (F * 255).astype(np.uint8)  # Green = food
        rgb[..., 0] = (P * 255).astype(np.uint8)  # Red = pheromones
        rgb[..., 2] = (H * 180).astype(np.uint8)  # Blue = safe zones

        # Scale up
        frame = np.repeat(np.repeat(rgb, cp, axis=0), cp, axis=1)
        pygame.surfarray.blit_array(game_surface, frame.swapaxes(0, 1))

        # Draw walls (dark gray)
        wall_indices = np.argwhere(world.W > 0)
        for y_idx, x_idx in wall_indices:
            pygame.draw.rect(game_surface, (60, 60, 65),
                             (x_idx * cp, y_idx * cp, cp, cp))

        # Draw predators (red circles with glow)
        for pred in world.predators:
            py, px = pred.y, pred.x
            center = (px * cp + cp // 2, py * cp + cp // 2)
            # Glow effect
            for radius in [12, 9, 6]:
                alpha = 100 - radius * 5
                color = (255, alpha, alpha)
                pygame.draw.circle(game_surface, color, center, radius)
            # Core
            pygame.draw.circle(game_surface, (255, 50, 50), center, 4)
            pygame.draw.circle(game_surface, (150, 0, 0), center, 4, 2)

        # Draw agent with level indicator
        ay, ax = agent.y, agent.x
        center = (ax * cp + cp // 2, ay * cp + cp // 2)

        # Outer ring (level color)
        ring_radius = cp // 2 + agent.level * 2
        pygame.draw.circle(game_surface, agent.color, center, ring_radius, 3)

        # Inner circle (white)
        pygame.draw.circle(game_surface, (255, 255, 255), center, max(4, cp // 2))

        # Facing direction indicator
        dy, dx = agent.facing
        tip = (center[0] + int(dx * cp * 0.7), center[1] + int(dy * cp * 0.7))
        pygame.draw.line(game_surface, (255, 255, 0), center, tip, 3)

        surf.blit(game_surface, (0, 0))

        # === RIGHT PANEL ===
        panel_x = self.game_width + 10
        panel_y = 10

        # Draw panel background
        panel_rect = pygame.Rect(self.game_width, 0, self.panel_width, self.game_height)
        pygame.draw.rect(surf, panel_bg, panel_rect)
        pygame.draw.line(surf, (100, 100, 100),
                         (self.game_width, 0), (self.game_width, self.game_height), 2)

        # Title
        title = self.font_large.render("AGENT STATUS", True, (255, 255, 100))
        surf.blit(title, (panel_x + 10, panel_y))
        panel_y += 40

        # Model type
        model_text = self.font.render(f"Mode: {model_type}", True, (150, 200, 255))
        surf.blit(model_text, (panel_x + 10, panel_y))
        panel_y += 30

        # === STAT BARS ===
        bar_width = 300
        bar_height = 25
        bar_x = panel_x + 20

        # Energy bar
        energy_color = (100, 255, 100) if agent.energy > 1.0 else (255, 200, 50)
        if agent.energy < 0.5:
            energy_color = (255, 100, 100)
        self.draw_stat_bar(surf, bar_x, panel_y, bar_width, bar_height,
                           agent.energy, 3.0, energy_color, label="ENERGY")
        panel_y += 50

        # Score bar (visual representation)
        self.draw_stat_bar(surf, bar_x, panel_y, bar_width, bar_height,
                           min(agent.score, 100), 100, (255, 255, 100), label="SCORE")
        panel_y += 50

        # Evolution progress bar
        progress = agent.get_evolution_progress()
        progress_color = LEVEL_COLORS[min(agent.level + 1, MAX_LEVEL)]
        self.draw_stat_bar(surf, bar_x, panel_y, bar_width, bar_height,
                           progress, 1.0, progress_color, label=f"LEVEL {agent.level} → {agent.level + 1}")
        panel_y += 50

        # Day/night cycle bar
        if world.is_day:
            cycle_progress = world.day_progress / world.day_length
            cycle_color = (255, 255, 150)
            cycle_label = "DAY"
        else:
            cycle_progress = world.day_progress / world.night_length
            cycle_color = (100, 100, 200)
            cycle_label = "NIGHT"
        self.draw_stat_bar(surf, bar_x, panel_y, bar_width, bar_height,
                           cycle_progress, 1.0, cycle_color, label=cycle_label)
        panel_y += 60

        # === TEXT STATS ===
        stats_texts = [
            f"Pellets Eaten: {agent.pellets_eaten}",
            f"Level: {agent.level}/{MAX_LEVEL}",
            f"Position: ({agent.y}, {agent.x})",
            f"Facing: {agent.facing}",
            f"",
            f"World Stats:",
            f"Pellets Left: {world.count_pellets()}",
            f"Predators: {len(world.predators)}",
            f"FPS: {fps:.1f}",
        ]

        for i, text in enumerate(stats_texts):
            color = (200, 200, 200) if text else (100, 100, 100)
            rendered = self.font_small.render(text, True, color)
            surf.blit(rendered, (panel_x + 20, panel_y + i * 20))

        panel_y += len(stats_texts) * 20 + 30

        # === OBSERVATION WINDOW ===
        if obs_hwc is not None:
            # Title
            obs_title = self.font.render("AGENT VISION", True, (255, 255, 255))
            surf.blit(obs_title, (panel_x + 10, panel_y))
            panel_y += 25

            # Scale and display observation
            obs_scale = 10  # Larger scale for better visibility
            obs_h, obs_w = obs_hwc.shape[:2]
            obs_surface = pygame.Surface((obs_w * obs_scale, obs_h * obs_scale))

            # Scale up observation
            obs_scaled = np.repeat(np.repeat(obs_hwc, obs_scale, axis=0),
                                   obs_scale, axis=1)
            pygame.surfarray.blit_array(obs_surface, obs_scaled.swapaxes(0, 1))

            # Draw border around observation
            obs_rect_x = panel_x + (self.panel_width - obs_w * obs_scale) // 2
            pygame.draw.rect(surf, (255, 255, 255),
                             (obs_rect_x - 2, panel_y - 2,
                              obs_w * obs_scale + 4, obs_h * obs_scale + 4), 2)

            surf.blit(obs_surface, (obs_rect_x, panel_y))
            panel_y += obs_h * obs_scale + 20

            # Legend for observation channels
            legend_x = panel_x + 10
            legend_items = [
                ("Red:", "Pheromone trails", (255, 100, 100)),
                ("Green:", "Food sources", (100, 255, 100)),
                ("Blue:", "Safe zones", (100, 150, 255)),
                ("Bright Red:", "PREDATOR!", (255, 50, 50)),
            ]

            for i, (label, desc, color) in enumerate(legend_items):
                # Color box
                pygame.draw.rect(surf, color, (legend_x, panel_y + i * 20, 15, 15))
                pygame.draw.rect(surf, (200, 200, 200),
                                 (legend_x, panel_y + i * 20, 15, 15), 1)
                # Text
                text = self.font_small.render(f"{label} {desc}", True, (220, 220, 220))
                surf.blit(text, (legend_x + 20, panel_y + i * 20))

        # === BOTTOM PANEL (Controls) ===
        bottom_y = self.game_height + 10
        bottom_text = [
            "Controls: Arrows/WASD=Move | P=Pause | R=Reset | +/-=Speed | ESC=Quit",
            info_text
        ]

        for i, text in enumerate(bottom_text):
            if text:
                rendered = self.font_small.render(text, True, (200, 200, 200))
                surf.blit(rendered, (10, bottom_y + i * 20))

        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="Enhanced MicroWorld Visualizer")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model (.zip)")
    parser.add_argument("--cell", type=int, default=6,
                        help="Pixels per cell")
    parser.add_argument("--steps", type=int, default=2,
                        help="Simulation steps per frame")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed")
    parser.add_argument("--pellets", type=int, default=500,
                        help="Number of food pellets")
    args = parser.parse_args()

    # Initialize world and agent
    world = World(seed=args.seed, n_pellets=args.pellets)
    agent = Agent(world, agent_id=0)

    rend = Renderer(cell_px=args.cell)
    rend.init(world.w, world.h)

    # Load model if specified
    model = None
    model_type = "Manual Control"
    is_recurrent = False

    if args.model is not None:
        model_path = args.model
        if not os.path.isabs(model_path):
            if not model_path.startswith("checkpoints"):
                model_path = os.path.join("checkpoints", args.model)

        # Try loading different model types
        try:
            from custom_policy import SmallCNN
            policy_kwargs = {
                "features_extractor_class": SmallCNN,
                "features_extractor_kwargs": {"features_dim": 256},
                "net_arch": {"pi": [256, 128], "vf": [256, 128]}
            }

            # Try MaskablePPO first
            try:
                model = MaskablePPO.load(model_path, policy_kwargs=policy_kwargs)
                model_type = "MaskablePPO"
                print(f"✓ Loaded as MaskablePPO")
            except:
                try:
                    model = PPO.load(model_path, policy_kwargs=policy_kwargs)
                    model_type = "PPO"
                    print(f"✓ Loaded as PPO")
                except:
                    # Try RecurrentPPO
                    policy_kwargs["enable_critic_lstm"] = True
                    policy_kwargs["lstm_hidden_size"] = 256
                    model = RecurrentPPO.load(model_path, policy_kwargs=policy_kwargs)
                    model_type = "RecurrentPPO"
                    is_recurrent = True
                    print(f"✓ Loaded as RecurrentPPO")
        except Exception as e:
            print(f"⚠ Failed to load model: {e}")
            print("Continuing with manual control")

    # Simulation state
    running = True
    paused = False
    sim_steps_per_frame = max(1, int(args.steps))

    lstm_state = None
    episode_start = True

    print("\n🎮 Enhanced MicroWorld Viewer")
    print(f"   Mode: {model_type}")
    print(f"   World: {world.w}x{world.h}, {args.pellets} pellets")
    print(f"   Predators: {len(world.predators)}")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                    print(f"{'⏸ Paused' if paused else '▶ Resumed'}")
                elif event.key == pygame.K_r:
                    world = World(seed=args.seed, n_pellets=args.pellets)
                    agent = Agent(world, agent_id=0)
                    lstm_state = None
                    episode_start = True
                    print("🔄 Reset world")
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    sim_steps_per_frame = min(64, sim_steps_per_frame + 1)
                    print(f"⏩ Speed: {sim_steps_per_frame} steps/frame")
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    sim_steps_per_frame = max(1, sim_steps_per_frame - 1)
                    print(f"⏪ Speed: {sim_steps_per_frame} steps/frame")

        if not paused:
            for _ in range(sim_steps_per_frame):
                world.step_fields()

                if model is None:
                    # Manual control
                    keys = pygame.key.get_pressed()
                    action = 0
                    for k, a in KEY_TO_ACTION.items():
                        if keys[k]:
                            action = a
                            break
                    agent.step(action)
                else:
                    # AI control
                    obs = build_obs_hwc(agent)

                    if model_type == "MaskablePPO":
                        action_mask = get_action_mask(agent, world)
                        action, _ = model.predict(obs, action_masks=action_mask,
                                                  deterministic=True)
                    elif model_type == "RecurrentPPO":
                        action, lstm_state = model.predict(
                            obs, state=lstm_state,
                            episode_start=np.array([episode_start]),
                            deterministic=True
                        )
                        episode_start = False
                    else:
                        action, _ = model.predict(obs, deterministic=True)

                    agent.step(int(action))

                # Check for death and respawn
                if agent.energy <= 0:
                    print(f"💀 Agent died! Score: {agent.score:.0f}, Level: {agent.level}")
                    agent = Agent(world, agent_id=0)
                    lstm_state = None
                    episode_start = True

        # Draw everything
        obs_display = build_obs_hwc(agent) if model or True else None
        info = f"{'⏸ PAUSED' if paused else ''} | Speed: {sim_steps_per_frame}x"

        rend.draw(world, agent, obs_hwc=obs_display, info_text=info,
                  fps=rend.clock.get_fps(), model_type=model_type)

        rend.clock.tick(60)

    pygame.quit()
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()