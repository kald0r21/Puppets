"""
Pygame widget for embedding Pygame in PyQt5.
"""
import pygame
import threading
import time
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer, pyqtSignal


class PygameWidget(QWidget):
    """
    Widget that embeds a Pygame surface for rendering the simulation.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.trainer = None
        self.is_rendering = False
        self.render_timer = None

        # Pygame surface
        self.surface = None
        self.cell_size = 14

        # Initialize UI
        self.setMinimumSize(560, 560)  # 40x40 grid at 14px cells

        # Initialize Pygame (headless)
        pygame.init()

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

            # Convert Pygame surface to QPixmap and display
            # (Simplified: in practice, you'd convert to QImage then QPixmap)
            self.update()

        except Exception as e:
            print(f"Render error: {e}")

    def draw_world(self, surface, world):
        """
        Draw the world onto the Pygame surface.

        Args:
            surface: Pygame surface
            world: World instance
        """
        # Get colors from config (use defaults if not available)
        bg_color = (20, 20, 20)
        grid_color = (40, 40, 40)
        food_color = (0, 255, 0)
        agent_color = (0, 150, 255)
        predator_color = (255, 0, 0)
        wall_color = (100, 100, 100)

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

    def paintEvent(self, event):
        """Qt paint event - displays the Pygame surface."""
        super().paintEvent(event)

        # In a full implementation, we'd convert the Pygame surface to QPixmap here
        # For now, this is a placeholder
        # The actual rendering happens via the render_frame() method
