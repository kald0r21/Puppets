"""
Statistics panel widget.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QProgressBar
from PyQt5.QtCore import Qt


class StatsPanel(QWidget):
    """
    Panel displaying current training statistics.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)

        # Main stats group
        stats_group = QGroupBox("Training Statistics")
        stats_layout = QVBoxLayout()

        # Generation/Episode counter
        self.generation_label = QLabel("Generation: 0 / 0")
        self.generation_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        stats_layout.addWidget(self.generation_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        stats_layout.addWidget(self.progress_bar)

        # Best fitness/reward
        self.best_label = QLabel("Best: 0")
        stats_layout.addWidget(self.best_label)

        # Average fitness/reward
        self.avg_label = QLabel("Average: 0.00")
        stats_layout.addWidget(self.avg_label)

        # Food eaten (if applicable)
        self.food_label = QLabel("Food Eaten: 0")
        stats_layout.addWidget(self.food_label)

        # Alive agents (for GA/CNN)
        self.alive_label = QLabel("Alive: 0")
        stats_layout.addWidget(self.alive_label)

        # Predators killed (for DQN)
        self.kills_label = QLabel("Kills: 0")
        stats_layout.addWidget(self.kills_label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Add stretch
        layout.addStretch()

    def update_metrics(self, metrics):
        """
        Update displayed metrics.

        Args:
            metrics: Dictionary of metrics from trainer
        """
        # Generation/Episode
        if 'generation' in metrics:
            gen = metrics['generation']
            max_gen = metrics.get('max_generations', '?')
            self.generation_label.setText(f"Generation: {gen} / {max_gen}")
            if max_gen != '?':
                progress = int((gen / max_gen) * 100)
                self.progress_bar.setValue(progress)
        elif 'episode' in metrics:
            ep = metrics['episode']
            max_ep = metrics.get('max_episodes', '?')
            self.generation_label.setText(f"Episode: {ep} / {max_ep}")
            if max_ep != '?':
                progress = int((ep / max_ep) * 100)
                self.progress_bar.setValue(progress)

        # Best fitness/reward
        if 'best_fitness' in metrics:
            self.best_label.setText(f"Best Fitness: {metrics['best_fitness']:.0f}")
        elif 'reward' in metrics:
            self.best_label.setText(f"Reward: {metrics['reward']:.0f}")

        # Average
        if 'avg_fitness' in metrics:
            self.avg_label.setText(f"Avg Fitness: {metrics['avg_fitness']:.2f}")
        elif 'avg_reward' in metrics:
            self.avg_label.setText(f"Avg Reward: {metrics['avg_reward']:.2f}")

        # Food eaten
        if 'best_food_eaten' in metrics:
            self.food_label.setText(f"Food Eaten: {metrics['best_food_eaten']}")
        elif 'food_eaten' in metrics:
            self.food_label.setText(f"Food Eaten: {metrics['food_eaten']}")

        # Alive agents
        if 'alive_agents' in metrics:
            self.alive_label.setText(f"Alive: {metrics['alive_agents']}")
            self.alive_label.setVisible(True)
        else:
            self.alive_label.setVisible(False)

        # Kills
        if 'predators_killed' in metrics:
            self.kills_label.setText(f"Kills: {metrics['predators_killed']}")
            self.kills_label.setVisible(True)
        else:
            self.kills_label.setVisible(False)
