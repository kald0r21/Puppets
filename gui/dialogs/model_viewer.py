"""
Model Viewer Dialog - Test and observe trained models.
"""
import os
import glob
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QPushButton, QComboBox, QGroupBox, QFormLayout,
                              QSpinBox, QCheckBox)
from PyQt5.QtCore import QTimer, pyqtSignal, Qt

from core.simulation.world import World
from core.simulation.entities import Agent
from core.brains.numpy_brain import NumpyBrain
from core.brains.cnn_brain import CNNBrain
from core.brains.dqn_brain import DQNBrain


class ModelViewer(QDialog):
    """
    Dialog for loading and testing trained models.
    Run the agent without training to observe behavior.
    """

    def __init__(self, config, parent=None):
        super().__init__(parent)

        self.config = config
        self.method = config.get('method', 'GA')
        self.world = None
        self.agent = None
        self.brain = None
        self.is_running = False
        self.timer = None
        self.turn = 0
        self.max_turns = 2000

        self.init_ui()
        self.load_available_models()

    def init_ui(self):
        """Initialize UI."""
        self.setWindowTitle("Model Viewer - Test Trained Models")
        self.resize(400, 500)

        layout = QVBoxLayout(self)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QFormLayout()

        self.model_combo = QComboBox()
        model_layout.addRow("Saved Model:", self.model_combo)

        self.load_button = QPushButton("Load Model")
        self.load_button.clicked.connect(self.load_model)
        model_layout.addRow("", self.load_button)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Test settings
        settings_group = QGroupBox("Test Settings")
        settings_layout = QFormLayout()

        self.max_turns_spin = QSpinBox()
        self.max_turns_spin.setRange(100, 10000)
        self.max_turns_spin.setValue(2000)
        self.max_turns_spin.valueChanged.connect(self.on_max_turns_changed)
        settings_layout.addRow("Max Turns:", self.max_turns_spin)

        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(1, 1000)
        self.speed_spin.setValue(60)
        settings_layout.addRow("Speed (FPS):", self.speed_spin)

        self.deterministic_check = QCheckBox()
        self.deterministic_check.setChecked(True)
        settings_layout.addRow("Deterministic:", self.deterministic_check)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Stats display
        stats_group = QGroupBox("Performance Stats")
        stats_layout = QFormLayout()

        self.turn_label = QLabel("0 / 2000")
        stats_layout.addRow("Turn:", self.turn_label)

        self.energy_label = QLabel("100")
        stats_layout.addRow("Energy:", self.energy_label)

        self.food_label = QLabel("0")
        stats_layout.addRow("Food Eaten:", self.food_label)

        self.kills_label = QLabel("0")
        stats_layout.addRow("Kills:", self.kills_label)

        self.status_label = QLabel("Alive")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        stats_layout.addRow("Status:", self.status_label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Control buttons
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("▶ Start Test")
        self.start_button.clicked.connect(self.start_test)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.clicked.connect(self.stop_test)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        self.reset_button = QPushButton("🔄 Reset")
        self.reset_button.clicked.connect(self.reset_test)
        button_layout.addWidget(self.reset_button)

        layout.addLayout(button_layout)

        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

    def load_available_models(self):
        """Load list of available saved models."""
        self.model_combo.clear()

        # Determine model directory based on method
        model_dirs = {
            'GA': 'saved_models/ga',
            'CNN': 'saved_models/cnn',
            'DQN': 'saved_models/dqn'
        }

        model_dir = model_dirs.get(self.method, 'saved_models/ga')

        if not os.path.exists(model_dir):
            self.model_combo.addItem("No models found")
            return

        # Find model files
        if self.method == 'GA':
            pattern = os.path.join(model_dir, '*.npz')
        else:  # CNN or DQN
            pattern = os.path.join(model_dir, '*.pth')

        model_files = glob.glob(pattern)

        if not model_files:
            self.model_combo.addItem("No models found")
            return

        # Sort by modification time (newest first)
        model_files.sort(key=os.path.getmtime, reverse=True)

        for model_file in model_files:
            self.model_combo.addItem(os.path.basename(model_file), model_file)

    def load_model(self):
        """Load the selected model."""
        if self.model_combo.currentData() is None:
            return

        model_path = self.model_combo.currentData()

        try:
            import json
            import torch

            # Try to load metadata file
            metadata_path = model_path.replace('.pth', '_metadata.json').replace('.npz', '_metadata.json')
            metadata = None

            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

            # Create appropriate brain based on method
            if self.method == 'GA':
                if metadata:
                    input_size = metadata.get('input_size', 11)
                    hidden_layers = metadata.get('hidden_layers', [16, 16])
                    output_size = metadata.get('output_size', 5)
                else:
                    # Fallback to defaults
                    input_size = 11
                    hidden_layers = [16, 16]
                    output_size = 5

                self.brain = NumpyBrain(input_size, hidden_layers, output_size)
                self.brain.load(model_path)

            elif self.method == 'CNN':
                if metadata:
                    map_size = metadata.get('map_size', 7)
                    num_channels = metadata.get('num_channels', 3)
                    num_actions = metadata.get('num_actions', 5)
                else:
                    # Fallback to defaults
                    map_size = 7
                    num_channels = 3
                    num_actions = 5

                self.brain = CNNBrain(map_size, num_channels, num_actions, device='cpu')
                self.brain.load(model_path)

            elif self.method == 'DQN':
                if metadata:
                    input_size = metadata.get('agent_input_size', 12)
                    output_size = metadata.get('agent_output_size', 5)
                    hidden_layers = metadata.get('agent_hidden_layers', [64, 32])
                else:
                    # Fallback to defaults
                    input_size = 12
                    output_size = 5
                    hidden_layers = [64, 32]

                self.brain = DQNBrain(
                    input_size, output_size, hidden_layers,
                    0.001, 50000, 128, 0.99,
                    0.9, 0.05, 30000, device='cpu'
                )
                self.brain.load(model_path)

            status_msg = "Model Loaded"
            if metadata:
                if 'best_fitness' in metadata:
                    status_msg += f" (Fitness: {int(metadata['best_fitness'])})"
                elif 'best_avg_reward' in metadata:
                    status_msg += f" (Reward: {int(metadata['best_avg_reward'])})"

            self.status_label.setText(status_msg)
            self.status_label.setStyleSheet("color: blue; font-weight: bold;")

        except Exception as e:
            self.status_label.setText(f"Load Failed: {e}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            import traceback
            traceback.print_exc()

    def start_test(self):
        """Start testing the model."""
        if not self.brain:
            self.status_label.setText("No model loaded!")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            return

        # Create world and agent
        self.reset_test()

        # Start simulation timer
        self.is_running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        fps = self.speed_spin.value()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_step)
        self.timer.start(1000 // fps)

    def stop_test(self):
        """Stop testing."""
        self.is_running = False
        if self.timer:
            self.timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def reset_test(self):
        """Reset the test environment."""
        import random

        # Create agent with loaded brain
        self.agent = Agent(
            random.randint(0, self.config['simulation']['grid_width'] - 1),
            random.randint(0, self.config['simulation']['grid_height'] - 1),
            brain=self.brain,
            config=self.config['simulation']
        )

        # Create world
        self.world = World(
            self.config['simulation']['grid_width'],
            self.config['simulation']['grid_height'],
            self.config['simulation'],
            agents=self.agent,
            single_agent_mode=True
        )

        self.turn = 0
        self.update_stats()

    def update_step(self):
        """Execute one simulation step."""
        if not self.is_running or not self.world or not self.agent.is_alive:
            self.stop_test()
            if self.agent and not self.agent.is_alive:
                self.status_label.setText("Dead")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
            return

        if self.turn >= self.max_turns:
            self.stop_test()
            self.status_label.setText("Max Turns Reached")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
            return

        # Get state and action from brain
        # (Simplified - actual implementation depends on method)
        # For GA/DQN: use get_agent_state
        # For CNN: use get_agent_vision

        # Move agent based on brain decision
        # Update world
        # Handle food, predators, etc.

        self.turn += 1
        self.update_stats()

    def update_stats(self):
        """Update statistics display."""
        if not self.agent:
            return

        self.turn_label.setText(f"{self.turn} / {self.max_turns}")
        self.energy_label.setText(str(int(self.agent.energy)))
        self.food_label.setText(str(self.agent.food_eaten_count))
        self.kills_label.setText(str(self.agent.predators_killed))

        if self.agent.is_alive:
            self.status_label.setText("Alive")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")

    def on_max_turns_changed(self, value):
        """Handle max turns change."""
        self.max_turns = value
