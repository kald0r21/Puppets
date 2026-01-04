"""
Advanced parameter editor widget with all config options.
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QScrollArea, QLabel,
                              QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
                              QFormLayout, QTabWidget)
from PyQt5.QtCore import pyqtSignal, Qt

from core.localization import tr


class ParamEditor(QWidget):
    """
    Comprehensive widget for editing all configuration parameters.
    Organized by tabs for better navigation.
    """

    param_changed = pyqtSignal(str, str, object)  # section, key, value

    def __init__(self, parent=None):
        super().__init__(parent)

        self.config = None
        self.widgets = {}

        self.init_ui()

    def init_ui(self):
        """Initialize UI with tabs."""
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

    def set_config(self, config):
        """
        Set configuration and generate parameter widgets.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.widgets.clear()

        # Clear existing tabs
        while self.tabs.count() > 0:
            self.tabs.removeTab(0)

        # Create tabs based on config sections
        if 'simulation' in config:
            self.create_simulation_tab(config['simulation'])

        method = config.get('method', 'GA').lower()
        if method in config:
            if method == 'ga':
                self.create_ga_tab(config['ga'])
            elif method == 'cnn':
                self.create_cnn_tab(config['cnn'])
            elif method == 'dqn':
                self.create_dqn_tab(config['dqn'])

        if 'visualization' in config:
            self.create_visualization_tab(config['visualization'])

    def create_simulation_tab(self, sim_config):
        """Create simulation parameters tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        main_layout = QVBoxLayout(content)

        # World settings
        world_group = QGroupBox(tr('params.world_settings'))
        world_form = QFormLayout()

        self.add_spinbox(world_form, 'simulation', 'grid_width', tr('params.grid_width') + ':',
                        sim_config.get('grid_width', 40), 10, 200)
        self.add_spinbox(world_form, 'simulation', 'grid_height', tr('params.grid_height') + ':',
                        sim_config.get('grid_height', 40), 10, 200)
        self.add_spinbox(world_form, 'simulation', 'num_food', tr('params.food_count') + ':',
                        sim_config.get('num_food', 45), 0, 500)
        self.add_spinbox(world_form, 'simulation', 'num_walls', tr('params.wall_count') + ':',
                        sim_config.get('num_walls', 50), 0, 500)

        world_group.setLayout(world_form)
        main_layout.addWidget(world_group)

        # Agent energy settings
        energy_group = QGroupBox(tr('params.agent_energy'))
        energy_form = QFormLayout()

        self.add_spinbox(energy_form, 'simulation', 'start_energy', tr('params.start_energy') + ':',
                        sim_config.get('start_energy', 100), 10, 1000)
        self.add_spinbox(energy_form, 'simulation', 'eat_gain', tr('params.food_energy_gain') + ':',
                        sim_config.get('eat_gain', 150), 10, 1000)
        self.add_spinbox(energy_form, 'simulation', 'max_energy_gain_per_food', tr('params.max_energy_per_food') + ':',
                        sim_config.get('max_energy_gain_per_food', 10), 0, 100)
        self.add_spinbox(energy_form, 'simulation', 'move_cost', tr('params.move_cost') + ':',
                        sim_config.get('move_cost', 1), 0, 50)
        self.add_spinbox(energy_form, 'simulation', 'idle_cost', tr('params.idle_cost') + ':',
                        sim_config.get('idle_cost', 3), 0, 50)
        self.add_spinbox(energy_form, 'simulation', 'wall_hit_penalty', tr('params.wall_hit_penalty') + ':',
                        sim_config.get('wall_hit_penalty', 3), 0, 50)

        energy_group.setLayout(energy_form)
        main_layout.addWidget(energy_group)

        # Perception
        perception_group = QGroupBox(tr('params.perception'))
        perception_form = QFormLayout()

        self.add_spinbox(perception_form, 'simulation', 'smart_perception_radius', tr('params.perception_radius') + ':',
                        sim_config.get('smart_perception_radius', 10), 1, 50)
        self.add_doublespinbox(perception_form, 'simulation', 'reward_shaping_factor', tr('params.reward_shaping') + ':',
                              sim_config.get('reward_shaping_factor', 0.1), 0.0, 10.0, 2)

        perception_group.setLayout(perception_form)
        main_layout.addWidget(perception_group)

        # Predator settings
        predator_group = QGroupBox(tr('params.predator_settings'))
        predator_form = QFormLayout()

        self.add_spinbox(predator_form, 'simulation', 'predator_count', tr('params.predator_count') + ':',
                        sim_config.get('predator_count', 5), 0, 50)
        self.add_checkbox(predator_form, 'simulation', 'predator_respawn', tr('params.predator_respawn') + ':',
                         sim_config.get('predator_respawn', True))
        self.add_spinbox(predator_form, 'simulation', 'predator_vision', tr('params.predator_vision') + ':',
                        sim_config.get('predator_vision', 10), 1, 50)
        self.add_spinbox(predator_form, 'simulation', 'predator_base_strength', tr('params.predator_strength') + ':',
                        sim_config.get('predator_base_strength', 5), 1, 100)
        self.add_spinbox(predator_form, 'simulation', 'predator_ally_bonus', "Ally Bonus:",
                        sim_config.get('predator_ally_bonus', 2), 0, 50)
        self.add_spinbox(predator_form, 'simulation', 'predator_ally_radius', "Ally Radius:",
                        sim_config.get('predator_ally_radius', 3), 1, 20)

        predator_group.setLayout(predator_form)
        main_layout.addWidget(predator_group)

        # Kill mechanics
        kill_group = QGroupBox(tr('params.kill_mechanics'))
        kill_form = QFormLayout()

        self.add_spinbox(kill_form, 'simulation', 'kill_license_level', tr('params.kill_license_level') + ':',
                        sim_config.get('kill_license_level', 5), 0, 50)
        self.add_spinbox(kill_form, 'simulation', 'kill_cost_pellets', tr('params.kill_cost_pellets') + ':',
                        sim_config.get('kill_cost_pellets', 1), 0, 20)

        kill_group.setLayout(kill_form)
        main_layout.addWidget(kill_group)

        main_layout.addStretch()
        scroll.setWidget(content)
        self.tabs.addTab(scroll, tr('tabs.simulation'))

    def create_ga_tab(self, ga_config):
        """Create GA parameters tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)

        group = QGroupBox("Genetic Algorithm Settings")
        form = QFormLayout()

        self.add_spinbox(form, 'ga', 'population_size', "Population Size:",
                        ga_config.get('population_size', 100), 10, 1000)
        self.add_spinbox(form, 'ga', 'num_generations', "Generations:",
                        ga_config.get('num_generations', 200), 10, 10000)
        self.add_spinbox(form, 'ga', 'max_turns_per_gen', "Max Turns/Gen:",
                        ga_config.get('max_turns_per_gen', 1000), 100, 10000)
        self.add_doublespinbox(form, 'ga', 'mutation_rate', "Mutation Rate:",
                              ga_config.get('mutation_rate', 0.05), 0.0, 1.0, 3)
        self.add_doublespinbox(form, 'ga', 'mutation_strength', "Mutation Strength:",
                              ga_config.get('mutation_strength', 0.5), 0.0, 5.0, 2)
        self.add_spinbox(form, 'ga', 'elitism_count', "Elitism Count:",
                        ga_config.get('elitism_count', 10), 0, 100)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()

        scroll.setWidget(content)
        self.tabs.addTab(scroll, tr("tabs.ga_settings"))

    def create_cnn_tab(self, cnn_config):
        """Create CNN parameters tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)

        group = QGroupBox("CNN Settings")
        form = QFormLayout()

        self.add_spinbox(form, 'cnn', 'population_size', "Population Size:",
                        cnn_config.get('population_size', 100), 10, 1000)
        self.add_spinbox(form, 'cnn', 'num_generations', "Generations:",
                        cnn_config.get('num_generations', 50), 10, 1000)
        self.add_spinbox(form, 'cnn', 'max_turns_per_gen', "Max Turns/Gen:",
                        cnn_config.get('max_turns_per_gen', 1000), 100, 10000)
        self.add_doublespinbox(form, 'cnn', 'mutation_rate', "Mutation Rate:",
                              cnn_config.get('mutation_rate', 0.05), 0.0, 1.0, 3)
        self.add_doublespinbox(form, 'cnn', 'mutation_strength', "Mutation Strength:",
                              cnn_config.get('mutation_strength', 0.5), 0.0, 5.0, 2)
        self.add_spinbox(form, 'cnn', 'elitism_count', "Elitism Count:",
                        cnn_config.get('elitism_count', 10), 0, 100)
        self.add_spinbox(form, 'cnn', 'max_perception_radius', "Max Perception Radius:",
                        cnn_config.get('max_perception_radius', 3), 1, 20)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()

        scroll.setWidget(content)
        self.tabs.addTab(scroll, tr("tabs.cnn_settings"))

    def create_dqn_tab(self, dqn_config):
        """Create DQN parameters tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        main_layout = QVBoxLayout(content)

        # Training settings
        training_group = QGroupBox("Training Settings")
        training_form = QFormLayout()

        self.add_spinbox(training_form, 'dqn', 'num_episodes', "Episodes:",
                        dqn_config.get('num_episodes', 2000), 100, 100000)
        self.add_spinbox(training_form, 'dqn', 'max_turns_per_episode', "Max Turns/Episode:",
                        dqn_config.get('max_turns_per_episode', 1000), 100, 10000)
        self.add_doublespinbox(training_form, 'dqn', 'gamma', "Gamma (Discount):",
                              dqn_config.get('gamma', 0.99), 0.0, 1.0, 3)

        training_group.setLayout(training_form)
        main_layout.addWidget(training_group)

        # Agent network
        agent_group = QGroupBox("Agent Network")
        agent_form = QFormLayout()

        self.add_doublespinbox(agent_form, 'dqn', 'agent_learning_rate', "Learning Rate:",
                              dqn_config.get('agent_learning_rate', 0.001), 0.00001, 0.1, 5)
        self.add_spinbox(agent_form, 'dqn', 'agent_memory_size', "Memory Size:",
                        dqn_config.get('agent_memory_size', 50000), 1000, 1000000)
        self.add_spinbox(agent_form, 'dqn', 'agent_batch_size', "Batch Size:",
                        dqn_config.get('agent_batch_size', 128), 8, 1024)
        self.add_doublespinbox(agent_form, 'dqn', 'agent_eps_start', "Epsilon Start:",
                              dqn_config.get('agent_eps_start', 0.9), 0.0, 1.0, 2)
        self.add_doublespinbox(agent_form, 'dqn', 'agent_eps_end', "Epsilon End:",
                              dqn_config.get('agent_eps_end', 0.05), 0.0, 1.0, 2)
        self.add_spinbox(agent_form, 'dqn', 'agent_eps_decay', "Epsilon Decay Steps:",
                        dqn_config.get('agent_eps_decay', 30000), 1000, 1000000)

        agent_group.setLayout(agent_form)
        main_layout.addWidget(agent_group)

        # Predator network
        predator_group = QGroupBox("Predator Network")
        predator_form = QFormLayout()

        self.add_doublespinbox(predator_form, 'dqn', 'predator_learning_rate', "Learning Rate:",
                              dqn_config.get('predator_learning_rate', 0.001), 0.00001, 0.1, 5)
        self.add_spinbox(predator_form, 'dqn', 'predator_memory_size', "Memory Size:",
                        dqn_config.get('predator_memory_size', 50000), 1000, 1000000)
        self.add_spinbox(predator_form, 'dqn', 'predator_batch_size', "Batch Size:",
                        dqn_config.get('predator_batch_size', 128), 8, 1024)
        self.add_doublespinbox(predator_form, 'dqn', 'predator_eps_start', "Epsilon Start:",
                              dqn_config.get('predator_eps_start', 0.9), 0.0, 1.0, 2)
        self.add_doublespinbox(predator_form, 'dqn', 'predator_eps_end', "Epsilon End:",
                              dqn_config.get('predator_eps_end', 0.05), 0.0, 1.0, 2)
        self.add_spinbox(predator_form, 'dqn', 'predator_eps_decay', "Epsilon Decay Steps:",
                        dqn_config.get('predator_eps_decay', 20000), 1000, 1000000)

        predator_group.setLayout(predator_form)
        main_layout.addWidget(predator_group)

        main_layout.addStretch()
        scroll.setWidget(content)
        self.tabs.addTab(scroll, tr("tabs.dqn_settings"))

    def create_visualization_tab(self, viz_config):
        """Create visualization parameters tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)

        group = QGroupBox("Visualization Settings")
        form = QFormLayout()

        self.add_spinbox(form, 'visualization', 'fps', "Target FPS:",
                        viz_config.get('fps', 60), 1, 1000)
        self.add_spinbox(form, 'visualization', 'cell_size', "Cell Size (px):",
                        viz_config.get('cell_size', 14), 4, 50)

        group.setLayout(form)
        layout.addWidget(group)
        layout.addStretch()

        scroll.setWidget(content)
        self.tabs.addTab(scroll, tr("tabs.visualization"))

    def add_spinbox(self, form, section, key, label, value, min_val, max_val):
        """Add integer spinbox."""
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(value)
        spinbox.valueChanged.connect(lambda v: self.on_param_changed(section, key, v))
        form.addRow(label, spinbox)
        self.widgets[f"{section}.{key}"] = spinbox

    def add_doublespinbox(self, form, section, key, label, value, min_val, max_val, decimals):
        """Add float spinbox."""
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setDecimals(decimals)
        spinbox.setSingleStep(10 ** -decimals)
        spinbox.setValue(value)
        spinbox.valueChanged.connect(lambda v: self.on_param_changed(section, key, v))
        form.addRow(label, spinbox)
        self.widgets[f"{section}.{key}"] = spinbox

    def add_checkbox(self, form, section, key, label, value):
        """Add checkbox."""
        checkbox = QCheckBox()
        checkbox.setChecked(value)
        checkbox.stateChanged.connect(lambda state: self.on_param_changed(section, key, state == Qt.Checked))
        form.addRow(label, checkbox)
        self.widgets[f"{section}.{key}"] = checkbox

    def on_param_changed(self, section, key, value):
        """Handle parameter change."""
        if self.config and section in self.config:
            self.config[section][key] = value
            self.param_changed.emit(section, key, value)
