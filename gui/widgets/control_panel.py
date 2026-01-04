"""
Control panel widget for simulation controls.
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QComboBox, QPushButton, QSlider, QGroupBox,
                              QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal


class ControlPanel(QWidget):
    """
    Left panel containing simulation controls.
    """

    # Signals
    method_changed = pyqtSignal(str)
    start_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    speed_changed = pyqtSignal(int)
    config_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_config = None
        self.is_paused = False

        self.init_ui()

    def init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)

        # Method selection
        method_group = QGroupBox("Method")
        method_layout = QVBoxLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems(['GA', 'CNN', 'DQN'])
        self.method_combo.currentTextChanged.connect(self.on_method_changed)
        method_layout.addWidget(self.method_combo)

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Control buttons
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()

        self.start_button = QPushButton("▶ Start")
        self.start_button.clicked.connect(self.start_clicked.emit)
        controls_layout.addWidget(self.start_button)

        self.pause_button = QPushButton("⏸ Pause")
        self.pause_button.clicked.connect(self.pause_clicked.emit)
        self.pause_button.setEnabled(False)
        controls_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.clicked.connect(self.stop_clicked.emit)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Speed control
        speed_group = QGroupBox("Speed")
        speed_layout = QVBoxLayout()

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(1000)
        self.speed_slider.setValue(60)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        speed_layout.addWidget(self.speed_slider)

        self.speed_label = QLabel("60 FPS")
        self.speed_label.setAlignment(Qt.AlignCenter)
        speed_layout.addWidget(self.speed_label)

        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)

        # Parameters (will be populated dynamically)
        self.params_group = QGroupBox("Parameters")
        self.params_layout = QVBoxLayout()
        self.params_group.setLayout(self.params_layout)
        layout.addWidget(self.params_group)

        # Add stretch to push everything to top
        layout.addStretch()

    def set_config(self, config):
        """
        Set current configuration and update UI.

        Args:
            config: Configuration dictionary
        """
        self.current_config = config

        # Update method combo (without triggering signal)
        self.method_combo.blockSignals(True)
        method = config.get('method', 'GA')
        index = self.method_combo.findText(method)
        if index >= 0:
            self.method_combo.setCurrentIndex(index)
        self.method_combo.blockSignals(False)

        # Update parameters
        self.update_parameters()

    def update_parameters(self):
        """Update parameter widgets based on current config."""
        # Clear existing widgets and layouts
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                # Clear child layout
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
                item.layout().deleteLater()

        if not self.current_config:
            return

        method = self.current_config['method'].lower()

        # Add key parameters
        if method in self.current_config:
            method_config = self.current_config[method]

            # Population/Episodes
            if 'population_size' in method_config:
                self.add_param_spinbox("Population", 'population_size', method_config, 10, 1000)
            elif 'num_episodes' in method_config:
                self.add_param_spinbox("Episodes", 'num_episodes', method_config, 100, 10000)

            # Generations
            if 'num_generations' in method_config:
                self.add_param_spinbox("Generations", 'num_generations', method_config, 10, 1000)

            # Learning rate (DQN)
            if 'agent_learning_rate' in method_config:
                self.add_param_doublespinbox("Learning Rate", 'agent_learning_rate', method_config, 0.0001, 0.1)

    def add_param_spinbox(self, label, key, config, min_val, max_val):
        """Add integer parameter spinbox."""
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{label}:"))

        spinbox = QSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setValue(config.get(key, min_val))
        spinbox.valueChanged.connect(lambda v: self.on_param_changed(key, v))
        row.addWidget(spinbox)

        self.params_layout.addLayout(row)

    def add_param_doublespinbox(self, label, key, config, min_val, max_val):
        """Add float parameter spinbox."""
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{label}:"))

        spinbox = QDoubleSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setDecimals(4)
        spinbox.setValue(config.get(key, min_val))
        spinbox.valueChanged.connect(lambda v: self.on_param_changed(key, v))
        row.addWidget(spinbox)

        self.params_layout.addLayout(row)

    def on_method_changed(self, method):
        """Handle method selection change."""
        self.method_changed.emit(method)

    def on_speed_changed(self, value):
        """Handle speed slider change."""
        self.speed_label.setText(f"{value} FPS")
        self.speed_changed.emit(value)

    def on_param_changed(self, key, value):
        """Handle parameter change."""
        if self.current_config:
            method = self.current_config['method'].lower()
            if method in self.current_config:
                self.current_config[method][key] = value
                self.config_changed.emit(self.current_config)

    def set_training_active(self, active):
        """
        Update UI for training state.

        Args:
            active: True if training is running
        """
        self.start_button.setEnabled(not active)
        self.pause_button.setEnabled(active)
        self.stop_button.setEnabled(active)
        self.method_combo.setEnabled(not active)

    def set_paused(self, paused):
        """
        Update UI for pause state.

        Args:
            paused: True if training is paused
        """
        self.is_paused = paused
        if paused:
            self.pause_button.setText("▶ Resume")
        else:
            self.pause_button.setText("⏸ Pause")

    def update_language(self):
        """Update widget text for new language."""
        # Update group box titles
        # Note: This is a simplified implementation
        # In a full implementation, we would use translations for all labels
        pass
