# src/scripts/viz_realtime.py
# Aktualizacja o renderowanie schronień i cyklu dnia

import argparse
import os
import numpy as np

try:
    import pygame
except Exception as e:
    raise SystemExit("This script requires pygame. Install with: pip install pygame") from e

import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

# --- POPRAWKA IMPORTÓW ---
from src.scripts.custom_policy import SmallCNN  # Poprawiono z '.custom_policy'
from src.scripts.micro_world_vision import (
    World, Agent, ACTIONS, PELLET_VALUE, EVOLUTION_THRESHOLDS, MAX_LEVEL, MAX_F,
    get_level_stats, LEVEL_COLORS
)

# --- KONIEC POPRAWKI ---

SB3_AVAILABLE = True
try:
    from stable_baselines3 import PPO
except Exception:
    SB3_AVAILABLE = False


def build_obs_hwc(agent: Agent):
    img_chw = agent.get_obs_img().astype(np.float32)
    img_hwc = np.clip(img_chw, 0.0, 1.0).transpose(1, 2, 0)
    return (img_hwc * 255).astype(np.uint8)


KEY_TO_ACTION = {
    pygame.K_KP5: 0, pygame.K_s: 0, pygame.K_UP: 1, pygame.K_w: 1,
    pygame.K_DOWN: 2, pygame.K_x: 2, pygame.K_RIGHT: 3, pygame.K_d: 3,
    pygame.K_LEFT: 4, pygame.K_a: 4, pygame.K_KP9: 5, pygame.K_e: 5,
    pygame.K_KP7: 6, pygame.K_q: 6, pygame.K_KP1: 7, pygame.K_z: 7,
    pygame.K_KP3: 8, pygame.K_c: 8
}

ACTION_NAMES = ["STAY", "UP", "DOWN", "RIGHT", "LEFT", "UP-RIGHT", "UP-LEFT", "DOWN-LEFT", "DOWN-RIGHT"]


class Renderer:
    def __init__(self, cell_px=6):
        self.cell_px = cell_px
        self.screen, self.font, self.font_small, self.font_large = None, None, None, None
        self.clock = pygame.time.Clock()

    def init(self, w, h, title="MicroWorld Viewer"):
        pygame.init()
        screen_w = w * self.cell_px + 350
        screen_h = h * self.cell_px + 150
        self.screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("monospace", 12, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 10)
        self.font_large = pygame.font.SysFont("monospace", 16, bold=True)
        print(f"Screen initialized: {screen_w}x{screen_h}")

    def draw_evolution_bar(self, surf, x, y, agent: Agent):
        bar_w, bar_h = 200, 20
        pygame.draw.rect(surf, (50, 50, 50), (x, y, bar_w, bar_h))
        progress = agent.get_evolution_progress()
        fill_w = int(bar_w * progress)
        pygame.draw.rect(surf, agent.color, (x, y, fill_w, bar_h))
        pygame.draw.rect(surf, (200, 200, 200), (x, y, bar_w, bar_h), 2)
        text = f"Lv{agent.level} -> Lv{agent.level + 1}: {agent.pellets_eaten}/{EVOLUTION_THRESHOLDS[agent.level]}" \
            if agent.level < MAX_LEVEL else f"Lv{agent.level} MAX LEVEL!"
        text_surf = self.font_small.render(text, True, (255, 255, 255))
        surf.blit(text_surf, (x + 5, y + 3))

    def draw(self, world: World, agents: list, obs_list=None, info_text="", fps=0):
        surf = self.screen
        cp = self.cell_px
        h, w = world.h, world.w

        day_color = (10, 10, 20)
        night_color = (5, 5, 10)
        bg_color = day_color if world.is_day else night_color
        surf.fill(bg_color)

        F = world.F
        P = np.clip(world.P, 0.0, 1.0) ** 0.7
        H = world.H

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[..., 1] = (F * 255).astype(np.uint8)
        rgb[..., 0] = (P * 255).astype(np.uint8)
        rgb[..., 2] = (H * 200).astype(np.uint8)

        frame = np.repeat(np.repeat(rgb, cp, axis=0), cp, axis=1)

        game_surface = pygame.Surface((w * cp, h * cp))
        pygame.surfarray.blit_array(game_surface, frame.swapaxes(0, 1))

        wall_indices = np.argwhere(world.W > 0)
        for y, x in wall_indices:
            pygame.draw.rect(game_surface, (80, 80, 80), (x * cp, y * cp, cp, cp))

        if not world.is_day:
            night_overlay = pygame.Surface((w * cp, h * cp), pygame.SRCALPHA)
            night_overlay.fill((0, 0, 50, 100))
            game_surface.blit(night_overlay, (0, 0))

        surf.blit(game_surface, (0, 0))

        for agent in agents:
            ay, ax = agent.y, agent.x
            center = (ax * cp + cp // 2, ay * cp + cp // 2)
            radius = max(3, cp // 2 + agent.level)
            border_width = 2 + agent.level // 2
            pygame.draw.circle(surf, agent.color, center, radius, border_width)
            dy, dx = agent.facing
            tip = (center[0] + int(dx * cp * 0.6), center[1] + int(dy * cp * 0.6))
            pygame.draw.line(surf, (255, 255, 0), center, tip, 2)
            level_text = self.font_small.render(str(agent.level), True, agent.color)
            surf.blit(level_text, (center[0] - 3, center[1] - 20))
            if agent.level >= 3:
                pygame.draw.circle(surf, agent.color, center, agent.vision_range * cp, 1)

        if obs_list:
            cam_x = w * cp + 10
            cam_y = 10
            cam_scale = 8
            for i, obs_hwc in enumerate(obs_list):
                if obs_hwc is None: continue
                cam_surf = pygame.Surface((obs_hwc.shape[1] * cam_scale, obs_hwc.shape[0] * cam_scale))
                obs_scaled = np.repeat(np.repeat(obs_hwc, cam_scale, axis=0), cam_scale, axis=1)
                pygame.surfarray.blit_array(cam_surf, obs_scaled.swapaxes(0, 1))
                y_offset = i * (obs_hwc.shape[0] * cam_scale + 80)
                surf.blit(cam_surf, (cam_x, cam_y + y_offset))
                label = self.font.render(f"Agent {i + 1} (Lv{agents[i].level})", True, agents[i].color)
                surf.blit(label, (cam_x, cam_y + y_offset - 14))

        hud_y = h * cp + 5
        time_bar_w = 400
        time_bar_h = 10
        time_bar_x = (w * cp + 350 - time_bar_w) // 2
        cycle_time = world.global_step % world.total_cycle
        time_progress = cycle_time / world.total_cycle

        pygame.draw.rect(surf, (50, 50, 50), (time_bar_x, hud_y, time_bar_w, time_bar_h))

        if world.is_day:
            day_progress_w = int(time_bar_w * (world.day_progress * (world.day_duration / world.total_cycle)))
            pygame.draw.rect(surf, (255, 255, 0), (time_bar_x, hud_y, day_progress_w, time_bar_h))
            time_str = f"DZIEŃ ({world.day_progress * 100:.0f}%)"
        else:
            night_start_w = int(time_bar_w * (world.day_duration / world.total_cycle))
            night_progress = (cycle_time - world.day_duration) / world.night_duration
            night_progress_w = int(time_bar_w * (night_progress * (world.night_duration / world.total_cycle)))
            pygame.draw.rect(surf, (50, 50, 200),
                             (time_bar_x + night_start_w, hud_y, night_progress_w, time_bar_h))
            time_str = f"NOC ({night_progress * 100:.0f}%)"

        pygame.draw.rect(surf, (200, 200, 200), (time_bar_x, hud_y, time_bar_w, time_bar_h), 1)
        time_text = self.font.render(time_str, True, (255, 255, 255))
        surf.blit(time_text, (time_bar_x + time_bar_w + 10, hud_y - 2))

        pellets_count = world.count_pellets()
        lines = [
            f"Pellets: {pellets_count}  Total eaten: {world.pellets_eaten}  {info_text}",
            "",
            f"FPS: {fps:.0f}",
        ]

        for i, line in enumerate(lines):
            if line:
                text_surf = self.font.render(line, True, (255, 255, 255))
                surf.blit(text_surf, (8, hud_y + i * 18))

        for i, agent in enumerate(agents):
            bar_y = hud_y + 40 + i * 30
            info = f"A{i + 1}: E={agent.energy:.1f} score={agent.score:.0f} eaten={agent.pellets_eaten}"
            info_surf = self.font_small.render(info, True, agent.color)
            surf.blit(info_surf, (8, bar_y - 12))
            self.draw_evolution_bar(surf, 8, bar_y, agent)
            stats_x = 220
            stats = [f"Speed: {agent.speed:.1f}x", f"Vision: {agent.vision_range}", f"Gain: {agent.eat_gain:.1f}"]
            for j, stat in enumerate(stats):
                stat_surf = self.font_small.render(stat, True, (200, 200, 200))
                surf.blit(stat_surf, (stats_x + j * 80, bar_y + 3))

        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--cell", type=int, default=6)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--pellets", type=int, default=400)
    parser.add_argument("--two-agents", action="store_true")
    args = parser.parse_args()

    print(f"Initializing with seed={args.seed}, pellets={args.pellets}")
    print(f"Evolution system: 5 levels, thresholds: {EVOLUTION_THRESHOLDS[1:]}")
    print("DAY/NIGHT CYCLE ENABLED!")

    world = World(seed=args.seed, n_pellets=args.pellets)
    agent1 = Agent(world, agent_id=0)
    agents = [agent1]

    if args.two_agents:
        agent2 = Agent(world, agent_id=1)
        agents.append(agent2);
        print("Two-agent mode enabled")

    rend = Renderer(cell_px=args.cell)
    rend.init(world.w, world.h, title="MicroWorld Evolution (Day/Night)")

    model = None
    if args.model is not None:
        if not SB3_AVAILABLE: raise SystemExit("Stable-Baselines3 not installed.")

        model_path = args.model
        if not os.path.isabs(model_path):
            SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
            PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

            if not model_path.startswith("checkpoints"):
                model_path = os.path.join(PROJECT_ROOT, "checkpoints", args.model)
            else:
                model_path = os.path.join(PROJECT_ROOT, args.model)

        if not os.path.exists(model_path): raise SystemExit(f"Model not found: {model_path}")

        policy_kwargs = {
            "features_extractor_class": SmallCNN,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": dict(pi=[256, 128], vf=[256, 128]),
        }
        model = PPO.load(model_path, policy_kwargs=policy_kwargs)
        print(f"Loaded model: {model_path}")

    running, paused = True, False
    sim_steps_per_frame = max(1, int(args.steps))

    print("\n=== Controls ===");
    print("P: pause/resume, R: reset, +/-: speed, ESC/Q: quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_r:
                    world = World(seed=args.seed, n_pellets=args.pellets)
                    agent1 = Agent(world, agent_id=0);
                    agents = [agent1]
                    if args.two_agents: agents.append(Agent(world, agent_id=1))
                    print("World reset!")
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    sim_steps_per_frame = min(64, sim_steps_per_frame + 1)
                elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                    sim_steps_per_frame = max(1, sim_steps_per_frame - 1)

        if not paused:
            for _ in range(sim_steps_per_frame):
                world.step_fields()
                for i, agent in enumerate(agents):
                    prev_level = agent.level
                    if model is None and i == 0:
                        keys = pygame.key.get_pressed();
                        action = 0
                        for k, a in KEY_TO_ACTION.items():
                            if keys[k]: action = a; break
                        agent.step(action)
                    else:
                        obs = build_obs_hwc(agent)
                        action, _ = model.predict(obs, deterministic=True)
                        agent.step(int(action))

                    if agent.level > prev_level: print(f"🎉 Agent {agent.agent_id + 1} EVOLVED to Level {agent.level}!")

                    if agent.energy <= 0:
                        print(f"💀 Agent {agent.agent_id + 1} died (Lv{agent.level}, {agent.pellets_eaten} pellets)")
                        agents[i] = Agent(world, agent_id=agent.agent_id)

        mode_text = "AI" if model else "Manual"
        status = "PAUSED" if paused else mode_text
        fps = rend.clock.get_fps()
        obs_list = [build_obs_hwc(agent) for agent in agents] if model or args.two_agents else None

        rend.draw(world, agents, obs_list=obs_list,
                  info_text=f"{status} | Speed: {sim_steps_per_frame}x",
                  fps=fps)
        rend.clock.tick(60)

    pygame.quit()
    print(f"\n=== Final Stats ===")
    for agent in agents: print(
        f"Agent {agent.agent_id + 1}: Lv{agent.level}, E={agent.energy:.2f}, Score={agent.score:.0f}, Pellets={agent.pellets_eaten}")


if __name__ == "__main__":
    main()