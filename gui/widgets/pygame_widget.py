"""
Pygame widget for embedding Pygame in PyQt5 - WORKING VERSION.
"""
import pygame
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter


class PygameWidget(QWidget):
    """
    Widget that embeds a Pygame surface for rendering the simulation.
    Uses QImage conversion for proper display in PyQt5.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.trainer = None
        self.is_rendering = False
        self.render_timer = None

        # Pygame surface
        self.surface = None
        self.cell_size = 14
        self.qimage = None

        # Initialize UI
        self.setMinimumSize(560, 560)  # 40x40 grid at 14px cells

        # Initialize Pygame (headless)
        pygame.init()
        self.font = pygame.font.SysFont(None, 24)

    def start_rendering(self, trainer):
        """
        Start rendering the simulation.

        Args:
            trainer: Trainer instance to render
        """
        self.trainer = trainer
        self.is_rendering = True

        # Create timer for rendering updates
        self.render_timer = QTimer()
        self.render_timer.timeout.connect(self.render_frame)
        self.render_timer.start(33)  # ~30 FPS for display

    def stop_rendering(self):
        """Stop rendering."""
        self.is_rendering = False
        if self.render_timer:
            self.render_timer.stop()

    def render_frame(self):
        """Render one frame of the simulation."""
        if not self.is_rendering or not self.trainer:
            return

        try:
            # Get world from trainer
            world = self.trainer.world
            if not world:
                return

            # Create or resize surface if needed
            grid_width = world.width
            grid_height = world.height
            width = grid_width * self.cell_size
            height = grid_height * self.cell_size

            if not self.surface or self.surface.get_size() != (width, height):
                self.surface = pygame.Surface((width, height))

            # Render world
            self.draw_world(self.surface, world)

            # Add text overlay
            self.draw_overlay(self.surface, world)

            # Convert Pygame surface to QImage
            self.pygame_to_qimage()

            # Trigger repaint
            self.update()

        except Exception as e:
            print(f"Render error: {e}")
            import traceback
            traceback.print_exc()

    def draw_world(self, surface, world):
        """
        Draw the world onto the Pygame surface.

        Args:
            surface: Pygame surface
            world: World instance
        """
        # Get colors from config (use defaults if not available)
        config = world.config if hasattr(world, 'config') else {}
        viz_config = config.get('visualization', {})
        colors = viz_config.get('colors', {})

        bg_color = tuple(colors.get('background', [20, 20, 20]))
        grid_color = tuple(colors.get('grid', [40, 40, 40]))
        food_color = tuple(colors.get('food', [0, 255, 0]))
        agent_color = tuple(colors.get('agent', [0, 150, 255]))
        predator_color = tuple(colors.get('predator', [255, 0, 0]))
        wall_color = tuple(colors.get('wall', [100, 100, 100]))

        # Fill background
        surface.fill(bg_color)

        # Draw grid
        for x in range(0, world.width * self.cell_size, self.cell_size):
            pygame.draw.line(surface, grid_color, (x, 0), (x, world.height * self.cell_size))
        for y in range(0, world.height * self.cell_size, self.cell_size):
            pygame.draw.line(surface, grid_color, (0, y), (world.width * self.cell_size, y))

        # Draw walls
        for (x, y) in world.wall_positions:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(surface, wall_color, rect)

        # Draw food
        for (x, y) in world.food_positions:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(surface, food_color, rect)

        # Draw predators
        for p in world.get_alive_predators():
            rect = pygame.Rect(p.x * self.cell_size, p.y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(surface, predator_color, rect)

        # Draw agents
        for a in world.get_alive_agents():
            # Color based on food eaten
            blue_val = min(255, 150 + a.food_eaten_count * 10)
            color = (0, blue_val, 255)
            rect = pygame.Rect(a.x * self.cell_size, a.y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(surface, color, rect)

            # Draw vision radius for CNN (if applicable)
            if hasattr(a, 'current_perception_radius') and a.current_perception_radius > 0:
                vision_radius_px = a.current_perception_radius * self.cell_size
                center_x = a.x * self.cell_size + self.cell_size // 2
                center_y = a.y * self.cell_size + self.cell_size // 2
                # Draw semi-transparent circle
                pygame.draw.circle(surface, (0, 100, 255, 50), (center_x, center_y),
                                 vision_radius_px, 1)

    def draw_overlay(self, surface, world):
        """
        Draw text overlay with stats.

        Args:
            surface: Pygame surface
            world: World instance
        """
        try:
            # Get trainer metrics
            metrics = self.trainer.get_metrics() if self.trainer else {}

            # Generation/Episode
            if 'generation' in metrics:
                text = f"Gen: {metrics['generation']}"
            elif 'episode' in metrics:
                text = f"Ep: {metrics['episode']}"
            else:
                text = "Training..."

            # Alive agents
            alive = len(world.get_alive_agents())
            alive_text = f"Alive: {alive}"

            # Render text
            text_surface = self.font.render(text, True, (255, 255, 255))
            alive_surface = self.font.render(alive_text, True, (255, 255, 255))

            # Draw with background
            surface.blit(text_surface, (5, 5))
            surface.blit(alive_surface, (5, 30))

        except Exception as e:
            pass  # Ignore overlay errors

    def pygame_to_qimage(self):
        """Convert Pygame surface to QImage."""
        if not self.surface:
            return

        # Get surface dimensions
        width = self.surface.get_width()
        height = self.surface.get_height()

        # Convert to string buffer
        buffer = pygame.image.tostring(self.surface, 'RGB')

        # Create QImage from buffer
        self.qimage = QImage(buffer, width, height, width * 3, QImage.Format_RGB888)

    def paintEvent(self, event):
        """Qt paint event - displays the QImage."""
        super().paintEvent(event)

        if self.qimage:
            painter = QPainter(self)

            # Scale to fit widget while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(self.qimage).scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Center the image
            x = (self.width() - scaled_pixmap.width()) // 2
            y = (self.height() - scaled_pixmap.height()) // 2

            painter.drawPixmap(x, y, scaled_pixmap)
            painter.end()
